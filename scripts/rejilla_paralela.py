"""
Ejecuta EN PARALELO las 38 combinaciones de las dos rejillas del notebook 04
(17 de profundidad + 21 de porcentaje) y deja los checkpoints exactamente
donde el notebook los espera, de forma que al reejecutarlo se los encuentre
hechos y solo genere tablas y graficas.

POR QUE: las 38 combinaciones son independientes entre si (pesos
reinicializados en cada una, mismo X_val/X_test, sin estado compartido),
pero `run_depth_grid`/`run_pct_grid` las recorren en un solo proceso. En una
maquina de 10 nucleos eso deja 9 sin usar: 2 h de reloj en vez de ~10 min.

POR QUE SHARDS: los checkpoints de train_utils se reescriben ENTEROS en cada
guardado (`_save_rows_checkpoint` vuelca la lista completa). Si 8 procesos
escribieran el mismo fichero se pisarian unos a otros. Cada worker escribe
su propio shard en datos/interim/paralelo/ y `--merge` los consolida.

Uso:
    # 8 workers en paralelo (ver scripts/lanzar_rejilla_paralela.sh)
    python scripts/rejilla_paralela.py --worker 0 --n-workers 8
    ...
    python scripts/rejilla_paralela.py --merge

Los parametros por defecto son los del "escenario A": mismo experimento que
la version secuencial, solo mas rapido.
  - patience=20 en vez de 100: la prueba con patience=400 mostro que no
    aparecen minimos tardios (la mejor epoca mediana es la 69, y con
    regularizacion baja a ~53), asi que esperar 100 epocas sin mejora es
    pagar por una certeza ya medida.
  - batch_size=256 en vez de 64: 1.38x por epoca. No se sube a 512 porque
    con 7074 filas serian solo 14 actualizaciones de gradiente por epoca
    (frente a 111 con 64) y cambia demasiado la dinamica de optimizacion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Limitar hilos ANTES de importar TensorFlow: con 8 procesos en 10 nucleos,
# dejar que cada uno abra 10 hilos provoca sobresuscripcion y va mas lento
# que en secuencial.
_THREADS = os.environ.get("REJILLA_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _THREADS)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, modelos, train_utils as tu  # noqa: E402

GENERADORES = ["noise", "gaussian", "rbig", "gan"]
SHARD_DIR = config.INTERIM_DIR / "paralelo"

# Purga / embargo: mismo valor y mismo calculo que en notebooks_src/04. El
# entrenamiento termina WINDOW_X_DAYS antes de VAL_START_DATE para que
# ninguna fila de entrenamiento comparta dias de entrada con la validacion.
EMBARGO_DAYS = config.WINDOW_X_DAYS
TRAIN_END = str((pd.Timestamp(config.VAL_START_DATE) - pd.Timedelta(days=EMBARGO_DAYS)).date())

# Checkpoints canonicos que lee notebooks/04 (mismos nombres que el notebook)
def _canon(rejilla: str, horizonte: int) -> dict:
    """Rutas de los checkpoints canonicos. El horizonte 1 conserva los
    nombres de siempre (es el caso base del notebook 04 y del README); los
    demas llevan sufijo `_h7` / `_h30`."""
    suf = "" if horizonte == config.WINDOW_Y_DAYS else f"_h{horizonte}"
    if rejilla == "depth":
        return {"rows": config.INTERIM_DIR / f"04_checkpoint_rejilla_profundidad{suf}.csv",
                "hist": config.INTERIM_DIR / f"04_checkpoint_rejilla_profundidad{suf}_histories.json",
                "tick": config.INTERIM_DIR / f"04_checkpoint_rejilla_profundidad{suf}_por_banco.csv",
                "value_col": "synth_years"}
    return {"rows": config.INTERIM_DIR / f"04_checkpoint_rejilla_porcentaje{suf}.csv",
            "hist": config.INTERIM_DIR / f"04_checkpoint_rejilla_porcentaje{suf}_histories.json",
            "tick": None,   # el notebook 04 no pide desglose por banco aqui
            "value_col": "pct_objetivo"}


# ---------------------------------------------------------------------------
# Datos y arquitectura
# ---------------------------------------------------------------------------
def cargar_datasets(nombres: set[str]) -> dict:
    """Carga solo los datasets que este worker necesita. X se pasa a float32:
    Keras entrena en float32 de todas formas, y asi cada worker ocupa la
    mitad de RAM (importante con 8 procesos a la vez)."""
    out = {}
    for name in GENERADORES:
        path = config.INTERIM_DIR / f"dataset_{name}.npz"
        if name not in nombres or not path.exists():
            continue
        npz = np.load(path, allow_pickle=True)
        # Misma correccion que el notebook 04: `is_synthetic` del nb03 marca
        # si el ULTIMO dia de la ventana es sintetico, no si ALGUNO lo es.
        # Sin esto, 59 ventanas que arrastran dias sinteticos se cuentan como
        # reales y `slice_by_pct` construye proporciones que no son las que
        # dice (ver train_utils.ventana_contiene_sintetico).
        out[name] = {
            "X": npz["X"].astype("float32"),
            "idx": pd.DatetimeIndex(npz["idx"]),
            "is_synth": tu.ventana_contiene_sintetico(npz["is_synthetic"], config.WINDOW_X_DAYS),
            "Y": {h: npz[f"Y_h{h}"].astype("float32") for h in config.HORIZONTES_DIAS},
        }
    return out


vista_horizonte = tu.vista_horizonte   # helper compartido con el notebook 04


def arquitectura_ganadora() -> str:
    """La misma arquitectura que elige el notebook 04: entre las redes que
    empatan dentro de UN ERROR ESTANDAR de la mejor MAE de validacion, la de
    menos parametros. Se delega en `tu.elegir_por_una_ee` para que script y
    notebook no puedan divergir."""
    tabla = pd.read_csv(config.TABLES_DIR / "04_comparacion_arquitecturas.csv", index_col=0)
    return tu.elegir_por_una_ee(tabla)


def constructor(nombre: str, n_channels: int, loss: str, dropout: float, l2: float):
    """Mismos argumentos de regularizacion que el diccionario `REG` del
    notebook 04: la rejilla tiene que entrenar exactamente el modelo que
    gano la comparacion de arquitecturas, no una version sin regularizar."""
    W, O = config.WINDOW_X_DAYS, config.N_PREDICTOR_TICKERS
    reg = dict(loss=loss, dropout=dropout, l2=l2)
    fabricas = {
        "dense": lambda: modelos.build_predictor_dense(W, n_channels, O, hidden_units=(128, 64), **reg),
        "cnn_1bloque": lambda: modelos.build_predictor_cnn(W, n_channels, O, conv_filters=(64,), **reg),
        "cnn_3bloques": lambda: modelos.build_predictor_cnn(W, n_channels, O, conv_filters=(64, 128, 128), **reg),
        "rnn_1capa": lambda: modelos.build_predictor_rnn(W, n_channels, O, lstm_units=(64,), **reg),
        "rnn_2capas": lambda: modelos.build_predictor_rnn(W, n_channels, O, lstm_units=(64, 128), **reg),
    }
    if nombre not in fabricas:
        raise ValueError(f"Arquitectura '{nombre}' no es una red keras entrenable; revisar manualmente.")
    return fabricas[nombre]


# ---------------------------------------------------------------------------
# Reparto de trabajo
# ---------------------------------------------------------------------------
def todas_las_combinaciones() -> list[dict]:
    """Las 38 combinaciones, en el mismo orden y con la misma semantica que
    run_depth_grid / run_pct_grid: en el punto sin sinteticos (synth_years=0
    o pct=0) los 4 generadores comparten dataset, asi que se entrena UNA
    sola vez con la etiqueta 'solo_reales'."""
    combos = []
    # La rejilla de PROFUNDIDAD se recorre para los tres horizontes: es la
    # que responde a "cuantos anios de sintetico" y la que interesa comparar
    # entre 1, 7 y 30 dias.
    for h in config.HORIZONTES_DIAS:
        for sy in config.SYNTH_DEPTH_YEARS_GRID:
            gens = ["solo_reales"] if sy <= 0 else GENERADORES
            for g in gens:
                combos.append({"rejilla": "depth", "generator": g,
                               "valor": float(sy), "horizonte": h})
    # La de PORCENTAJE se queda en 1 dia: es el eje que pide literalmente el
    # enunciado y multiplicarlo por 3 horizontes triplicaria el coste sin
    # anadir un eje nuevo.
    for p in config.PCT_SYNTH_GRID:
        gens = ["solo_reales"] if p <= 0 else GENERADORES
        for g in gens:
            combos.append({"rejilla": "pct", "generator": g,
                           "valor": float(p), "horizonte": config.WINDOW_Y_DAYS})
    return combos


def coste_estimado(combo: dict) -> float:
    """Proxy del coste: el tiempo va con el numero de filas de entrenamiento.
    Solo sirve para repartir carga, no hace falta que sea exacto."""
    anios_reales = 4.6
    if combo["rejilla"] == "depth":
        return min(combo["valor"], 24.4) + anios_reales
    p = combo["valor"]
    return anios_reales / max(1.0 - p, 1.0 / 30.0)  # n_total = n_real / (1-p)


def reparto(n_workers: int) -> list[list[dict]]:
    """LPT (longest processing time first): se ordenan las combinaciones de
    mas cara a mas barata y cada una va al worker menos cargado. Es
    determinista, asi que los 8 procesos calculan el mismo reparto sin
    hablar entre ellos."""
    cubos = [[] for _ in range(n_workers)]
    carga = [0.0] * n_workers
    for combo in sorted(todas_las_combinaciones(), key=coste_estimado, reverse=True):
        k = int(np.argmin(carga))
        cubos[k].append(combo)
        carga[k] += coste_estimado(combo)
    return cubos


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def entrenar(worker: int, n_workers: int, patience: int, batch_size: int,
             epochs: int, loss: str, dropout: float, l2: float) -> None:
    mis_combos = reparto(n_workers)[worker]
    if not mis_combos:
        print(f"[w{worker}] sin trabajo asignado")
        return

    necesarios = {c["generator"] for c in mis_combos if c["generator"] != "solo_reales"}
    necesarios.add(GENERADORES[0])  # referencia para las filas 'solo_reales'
    datasets = cargar_datasets(necesarios)
    ref = GENERADORES[0]

    # val/test dependen del horizonte (cada uno pierde un numero distinto de
    # ventanas al final), asi que se calculan una vez por horizonte.
    vt = {}
    for h in config.HORIZONTES_DIAS:
        Xh, Yh, ih, _ = vista_horizonte(datasets[ref], h)
        # Mismo helper que el notebook 04: excluye de validacion las
        # ventanas cuyo objetivo (media de los h dias siguientes) se mete
        # en el test.
        (Xv, Yv), (Xt, Yt) = tu.split_val_test(Xh, Yh, ih, horizonte=h)
        vt[h] = (Xv, Yv, Xt, Yt)

    n_canales = datasets[ref]["X"].shape[-1]
    arch = arquitectura_ganadora()
    build = constructor(arch, n_canales, loss, dropout, l2)
    print(f"[w{worker}] {len(mis_combos)} combinaciones | arquitectura={arch} "
          f"| patience={patience} batch={batch_size} dropout={dropout} l2={l2} "
          f"| train hasta {TRAIN_END} (embargo {EMBARGO_DAYS}d)", flush=True)

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    claves = {(c["rejilla"], c["horizonte"]) for c in mis_combos}
    filas = {k: [] for k in claves}
    historias = {k: {} for k in claves}
    por_banco = {k: {} for k in claves if k[0] == "depth"}

    for i, combo in enumerate(mis_combos, 1):
        rejilla, gen, valor = combo["rejilla"], combo["generator"], combo["valor"]
        horiz = combo["horizonte"]
        origen = ref if gen == "solo_reales" else gen
        X_full, Y_full, idx_full, is_synth = vista_horizonte(datasets[origen], horiz)
        X_val, Y_val, X_test, Y_test = vt[horiz]

        if rejilla == "depth":
            X_tr, Y_tr, _, pct_synth = tu.slice_by_depth(
                X_full, Y_full, idx_full, valor,
                TRAIN_END, config.REAL_INTRADAY_START_DATE, is_synth)
            fila_extra = {"synth_years": valor, "horizonte": horiz}
        else:
            X_tr, Y_tr, _, pct_synth = tu.slice_by_pct(
                X_full, Y_full, idx_full, valor, TRAIN_END, is_synth)
            fila_extra = {"pct_objetivo": valor, "horizonte": horiz}

        t0 = time.time()
        # Misma semilla antes de cada modelo, igual que en run_depth_grid:
        # es lo que hace que repartir las combinaciones entre 8 procesos de
        # oro modo cada vez de exactamente el mismo resultado que en secuencial.
        tu.set_seed()
        model = build()
        hist = model.fit(
            X_tr, Y_tr, epochs=epochs, batch_size=batch_size,
            validation_data=(X_val, Y_val), verbose=0,
            callbacks=tu._make_early_stopping(patience),
        )
        metrics = tu.evaluate_predictor(model, X_test, Y_test)
        # SIN el horizonte en la clave: `run_depth_grid` busca (generador,
        # valor) y el horizonte ya lo distingue el nombre del fichero. Si se
        # mete aqui, el notebook no reconoce los checkpoints y reentrena todo.
        clave = tu._grid_key(gen, valor)
        k = (rejilla, horiz)
        filas[k].append({"generator": gen, **fila_extra,
                         "n_train": len(X_tr), "pct_synth": pct_synth, **metrics})
        historias[k][clave] = tu._history_to_lists(hist.history)
        if rejilla == "depth":
            por_banco[k][clave] = tu.evaluate_predictor_per_ticker(
                model, X_test, Y_test, config.PREDICTOR_TICKERS)

        print(f"[w{worker}] {i}/{len(mis_combos)} {rejilla} h={horiz} {gen} {valor} "
              f"n={len(X_tr)} ep={len(hist.history['loss'])} "
              f"mae={metrics['mae']:.6f} ({time.time()-t0:.0f}s)", flush=True)

        del model
        from tensorflow import keras
        keras.backend.clear_session()
        _guardar_shard(worker, filas, historias, por_banco)

    print(f"[w{worker}] TERMINADO", flush=True)


def _guardar_shard(worker: int, filas: dict, historias: dict, por_banco: dict) -> None:
    """Se guarda tras cada combinacion: si un worker muere, lo ya hecho no
    se pierde y al relanzarlo el merge lo recoge igual. Un shard por
    (rejilla, horizonte) y worker."""
    for (rejilla, h), fs in filas.items():
        if fs:
            pd.DataFrame(fs).to_csv(SHARD_DIR / f"{rejilla}_h{h}_rows_w{worker}.csv", index=False)
    for (rejilla, h), hs in historias.items():
        if hs:
            (SHARD_DIR / f"{rejilla}_h{h}_hist_w{worker}.json").write_text(
                json.dumps({f"{g}|{float(v):.12g}": x for (g, v), x in hs.items()}), encoding="utf-8")
    for (rejilla, h), pb in por_banco.items():
        if not pb:
            continue
        marcos = []
        for (gen, valor), df in pb.items():
            out = df.reset_index().copy()
            out.insert(0, "synth_years", valor)
            out.insert(0, "generator", gen.split("|")[0])
            marcos.append(out)
        pd.concat(marcos, ignore_index=True).to_csv(
            SHARD_DIR / f"{rejilla}_h{h}_tick_w{worker}.csv", index=False)


# ---------------------------------------------------------------------------
# Consolidacion
# ---------------------------------------------------------------------------
def merge() -> None:
    """Consolida los shards de todos los workers en los checkpoints
    canonicos, uno por (rejilla, horizonte)."""
    if not SHARD_DIR.exists():
        print("No hay shards que consolidar."); return

    combos = todas_las_combinaciones()
    claves = sorted({(c["rejilla"], c["horizonte"]) for c in combos})
    total_ok = 0
    for rejilla, h in claves:
        meta = _canon(rejilla, h)
        vcol = meta["value_col"]

        shards = sorted(SHARD_DIR.glob(f"{rejilla}_h{h}_rows_w*.csv"))
        if shards:
            df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
            df = df.drop_duplicates(subset=["generator", vcol], keep="last")
            df = df.sort_values([vcol, "generator"]).reset_index(drop=True)
            df.to_csv(meta["rows"], index=False)
            total_ok += len(df)
            print(f"[merge] {rejilla} h={h}: {len(df)} filas -> {meta['rows'].name}")

        hs = sorted(SHARD_DIR.glob(f"{rejilla}_h{h}_hist_w*.json"))
        if hs:
            fusion = {}
            for f in hs:
                for k, v in json.loads(f.read_text(encoding="utf-8")).items():
                    # normaliza el formato antiguo "gen|hN|valor" -> "gen|valor"
                    partes = k.split("|")
                    if len(partes) == 3 and partes[1].startswith("h"):
                        k = f"{partes[0]}|{partes[2]}"
                    fusion[k] = v
            meta["hist"].write_text(json.dumps(fusion, indent=2), encoding="utf-8")
            print(f"[merge] {rejilla} h={h}: {len(fusion)} historiales")

        if meta["tick"] is not None:
            ts = sorted(SHARD_DIR.glob(f"{rejilla}_h{h}_tick_w*.csv"))
            if ts:
                df = pd.concat([pd.read_csv(f) for f in ts], ignore_index=True)
                df = df.drop_duplicates(subset=["generator", vcol, "ticker"], keep="last")
                df.to_csv(meta["tick"], index=False)

    print(f"\n[merge] {total_ok}/{len(combos)} combinaciones consolidadas.")
    if total_ok < len(combos):
        print("[merge] AVISO: faltan combinaciones; relanza los workers que fallaron.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--loss", default="mae")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--plan", action="store_true", help="muestra el reparto y sale")
    args = ap.parse_args()

    if args.merge:
        merge()
    elif args.plan:
        for k, combos in enumerate(reparto(args.n_workers)):
            total = sum(coste_estimado(c) for c in combos)
            print(f"w{k}: {len(combos):2d} combinaciones, coste relativo {total:6.1f}")
    elif args.worker is not None:
        entrenar(args.worker, args.n_workers, args.patience,
                 args.batch_size, args.epochs, args.loss, args.dropout, args.l2)
    else:
        ap.error("hace falta --worker K, --merge o --plan")


if __name__ == "__main__":
    main()
