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
CANON = {
    "depth": {
        "rows": config.INTERIM_DIR / "04_checkpoint_rejilla_profundidad.csv",
        "hist": config.INTERIM_DIR / "04_checkpoint_rejilla_profundidad_histories.json",
        "tick": config.INTERIM_DIR / "04_checkpoint_rejilla_profundidad_por_banco.csv",
        "value_col": "synth_years",
    },
    "pct": {
        "rows": config.INTERIM_DIR / "04_checkpoint_rejilla_porcentaje.csv",
        "hist": config.INTERIM_DIR / "04_checkpoint_rejilla_porcentaje_histories.json",
        "tick": None,  # el notebook 04 no pide desglose por banco en esta rejilla
        "value_col": "pct_objetivo",
    },
}


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
        out[name] = (
            npz["X"].astype("float32"), npz["Y"].astype("float32"),
            pd.DatetimeIndex(npz["idx"]), npz["is_synthetic"],
        )
    return out


PARAMS_POR_ARQUITECTURA = {
    "dense": 394_009, "cnn_1bloque": 197_889, "cnn_3bloques": 150_273,
    "rnn_1capa": 38_465, "rnn_2capas": 143_681,
}


def arquitectura_ganadora() -> str:
    """Misma regla que el notebook 04 (`ARQUITECTURA_RED`): entre las redes
    que empatan dentro de UN ERROR ESTANDAR de la mejor MAE de validacion,
    la que tiene menos parametros.

    No basta con `idxmin()`: las tres mejores estan separadas por 0.000003
    y el orden cambia entre ejecuciones, de modo que la rejilla podia acabar
    entrenando rnn_2capas (4x mas lenta) en vez de rnn_1capa por puro azar
    de la inicializacion."""
    tabla = pd.read_csv(config.TABLES_DIR / "04_comparacion_arquitecturas.csv", index_col=0)
    redes = tabla.drop(index=["constante", "baseline", "linear"], errors="ignore")
    se = redes["mae"].std() / max(len(redes) ** 0.5, 1)
    empatadas = redes[redes["mae"] <= redes["mae"].min() + se]
    return min(empatadas.index, key=lambda n: PARAMS_POR_ARQUITECTURA.get(n, 10**9))


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
    for sy in config.SYNTH_DEPTH_YEARS_GRID:
        gens = ["solo_reales"] if sy <= 0 else GENERADORES
        for g in gens:
            combos.append({"rejilla": "depth", "generator": g, "valor": float(sy)})
    for p in config.PCT_SYNTH_GRID:
        gens = ["solo_reales"] if p <= 0 else GENERADORES
        for g in gens:
            combos.append({"rejilla": "pct", "generator": g, "valor": float(p)})
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

    X_ref, Y_ref, idx_ref, _ = datasets[ref]
    val = (idx_ref >= pd.Timestamp(config.VAL_START_DATE)) & (idx_ref < pd.Timestamp(config.REAL_TEST_HOLDOUT_START_DATE))
    tst = idx_ref >= pd.Timestamp(config.REAL_TEST_HOLDOUT_START_DATE)
    X_val, Y_val, X_test, Y_test = X_ref[val], Y_ref[val], X_ref[tst], Y_ref[tst]

    arch = arquitectura_ganadora()
    build = constructor(arch, X_ref.shape[-1], loss, dropout, l2)
    print(f"[w{worker}] {len(mis_combos)} combinaciones | arquitectura={arch} "
          f"| patience={patience} batch={batch_size} dropout={dropout} l2={l2} "
          f"| train hasta {TRAIN_END} (embargo {EMBARGO_DAYS}d)", flush=True)

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    filas = {"depth": [], "pct": []}
    historias = {"depth": {}, "pct": {}}
    por_banco = {"depth": {}}

    for i, combo in enumerate(mis_combos, 1):
        rejilla, gen, valor = combo["rejilla"], combo["generator"], combo["valor"]
        origen = ref if gen == "solo_reales" else gen
        X_full, Y_full, idx_full, is_synth = datasets[origen]

        if rejilla == "depth":
            X_tr, Y_tr, _, pct_synth = tu.slice_by_depth(
                X_full, Y_full, idx_full, valor,
                TRAIN_END, config.REAL_INTRADAY_START_DATE, is_synth)
            fila_extra = {"synth_years": valor}
        else:
            X_tr, Y_tr, _, pct_synth = tu.slice_by_pct(
                X_full, Y_full, idx_full, valor, TRAIN_END, is_synth)
            fila_extra = {"pct_objetivo": valor}

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
        clave = tu._grid_key(gen, valor)
        filas[rejilla].append({"generator": gen, **fila_extra,
                               "n_train": len(X_tr), "pct_synth": pct_synth, **metrics})
        historias[rejilla][clave] = tu._history_to_lists(hist.history)
        if rejilla == "depth":
            por_banco["depth"][clave] = tu.evaluate_predictor_per_ticker(
                model, X_test, Y_test, config.PREDICTOR_TICKERS)

        print(f"[w{worker}] {i}/{len(mis_combos)} {rejilla} {gen} {valor} "
              f"n={len(X_tr)} ep={len(hist.history['loss'])} "
              f"mae={metrics['mae']:.6f} ({time.time()-t0:.0f}s)", flush=True)

        del model
        from tensorflow import keras
        keras.backend.clear_session()
        _guardar_shard(worker, filas, historias, por_banco)

    print(f"[w{worker}] TERMINADO", flush=True)


def _guardar_shard(worker: int, filas: dict, historias: dict, por_banco: dict) -> None:
    """Se guarda tras cada combinacion: si un worker muere, lo ya hecho no
    se pierde y al relanzarlo el merge lo recoge igual."""
    for rejilla in ("depth", "pct"):
        if filas[rejilla]:
            pd.DataFrame(filas[rejilla]).to_csv(SHARD_DIR / f"{rejilla}_rows_w{worker}.csv", index=False)
        if historias[rejilla]:
            (SHARD_DIR / f"{rejilla}_hist_w{worker}.json").write_text(
                json.dumps({f"{g}|{float(v):.12g}": h for (g, v), h in historias[rejilla].items()}),
                encoding="utf-8")
    if por_banco["depth"]:
        marcos = []
        for (gen, valor), df in por_banco["depth"].items():
            out = df.reset_index().copy()
            out.insert(0, "synth_years", valor)
            out.insert(0, "generator", gen)
            marcos.append(out)
        pd.concat(marcos, ignore_index=True).to_csv(
            SHARD_DIR / f"depth_tick_w{worker}.csv", index=False)


# ---------------------------------------------------------------------------
# Consolidacion
# ---------------------------------------------------------------------------
def merge() -> None:
    if not SHARD_DIR.exists():
        print("No hay shards que consolidar."); return

    for rejilla, meta in CANON.items():
        value_col = meta["value_col"]

        shards = sorted(SHARD_DIR.glob(f"{rejilla}_rows_w*.csv"))
        if shards:
            df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
            df = df.drop_duplicates(subset=["generator", value_col], keep="last")
            df = df.sort_values([value_col, "generator"]).reset_index(drop=True)
            df.to_csv(meta["rows"], index=False)
            print(f"[merge] {rejilla}: {len(df)} filas -> {meta['rows'].name}")

        hist_shards = sorted(SHARD_DIR.glob(f"{rejilla}_hist_w*.json"))
        if hist_shards:
            fusion = {}
            for f in hist_shards:
                fusion.update(json.loads(f.read_text(encoding="utf-8")))
            meta["hist"].write_text(json.dumps(fusion, indent=2), encoding="utf-8")
            print(f"[merge] {rejilla}: {len(fusion)} historiales -> {meta['hist'].name}")

        if meta["tick"] is not None:
            tick_shards = sorted(SHARD_DIR.glob(f"{rejilla}_tick_w*.csv"))
            if tick_shards:
                df = pd.concat([pd.read_csv(f) for f in tick_shards], ignore_index=True)
                df = df.drop_duplicates(subset=["generator", value_col, "ticker"], keep="last")
                df.to_csv(meta["tick"], index=False)
                print(f"[merge] {rejilla}: desglose por banco -> {meta['tick'].name}")

    esperadas = len(todas_las_combinaciones())
    hechas = sum(
        len(pd.read_csv(CANON[r]["rows"])) for r in CANON if CANON[r]["rows"].exists()
    )
    print(f"\n[merge] {hechas}/{esperadas} combinaciones consolidadas.")
    if hechas < esperadas:
        print("[merge] AVISO: faltan combinaciones; relanza los workers que fallaron "
              "(el notebook 04 entrenaria las que falten al ejecutarse).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--loss", default="mae")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--l2", type=float, default=1e-4)
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
