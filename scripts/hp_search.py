"""
Busqueda aleatoria de hiperparametros para el predictor del dia siguiente.

Objetivo: encontrar, para cada familia de arquitectura (densa / CNN / RNN),
la configuracion que MEJOR GENERALIZA — no la que menos loss de
entrenamiento tiene. Con ~250 muestras reales de entrenamiento y una
entrada de 60x50 valores, las redes de clase (150k-400k parametros)
sobreajustan en pocas epocas: la loss de validacion empieza a SUBIR
mientras la de entrenamiento sigue bajando. Esta busqueda explora
capacidad + regularizacion (dropout, L2, learning rate, pooling global)
para encontrar configuraciones cuya curva de validacion baje y se quede
plana, que es lo que pide el enunciado ("curvas de loss donde se vea que
el modelo ha convergido").

REGLA CRITICA: la seleccion se hace SIEMPRE sobre el conjunto de
VALIDACION. El test (`REAL_TEST_HOLDOUT_START_DATE` en adelante) no se
toca aqui bajo ningun concepto — se reserva intacto para la comparativa
final del notebook 04.

Se hacen dos busquedas:
  A) sobre la ventana REAL de entrenamiento (synth_years=0, ~250 muestras)
     -> es la que decide la arquitectura del paso "elegir arquitectura"
        del notebook 04.
  B) sobre el dataset a profundidad maxima (synth_years=28, ~7000 muestras)
     -> comprueba si la mejor arquitectura cambia cuando hay mucho mas
        dato, que es el regimen del modelo final de la comparativa.

Uso:
    python scripts/hp_search.py [--minutes 600] [--stage A|B|both]

Los resultados se guardan de forma incremental en
reports/tables/hpsearch_<stage>.csv, asi que se puede inspeccionar
mientras corre y no se pierde nada si se interrumpe.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

# Limitar los hilos de TensorFlow ANTES de importarlo: la busqueda lanza
# varios procesos worker en paralelo (un core por worker rinde mucho mas
# que un solo proceso peleandose por los 16 cores con modelos diminutos).
_THREADS = os.environ.get("HPSEARCH_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _THREADS)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, modelos, train_utils as tu  # noqa: E402

MAX_EPOCHS = 400
SEARCH_PATIENCE = 40  # patience moderado durante la busqueda (rapido);
                      # la config ganadora se re-verifica con patience alto


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_full(generator: str = "noise"):
    """Arrays completos + indice, para el modo walk-forward (los cortes se
    hacen despues, fold a fold)."""
    npz = np.load(config.INTERIM_DIR / f"dataset_{generator}.npz", allow_pickle=True)
    return npz["X"], npz["Y"], pd.DatetimeIndex(npz["idx"]), npz["is_synthetic"]


def load_split(synth_years: int, generator: str = "noise"):
    npz = np.load(config.INTERIM_DIR / f"dataset_{generator}.npz", allow_pickle=True)
    idx = pd.DatetimeIndex(npz["idx"])
    X, Y, is_synth = npz["X"], npz["Y"], npz["is_synthetic"]

    val_mask = (idx >= pd.Timestamp(config.VAL_START_DATE)) & (
        idx < pd.Timestamp(config.REAL_TEST_HOLDOUT_START_DATE)
    )
    X_val, Y_val = X[val_mask], Y[val_mask]

    X_tr, Y_tr, _, _ = tu.slice_by_depth(
        X, Y, idx, synth_years=synth_years,
        train_end=config.VAL_START_DATE,
        synth_anchor=config.REAL_INTRADAY_START_DATE,
        is_synthetic=is_synth,
    )
    return X_tr, Y_tr, X_val, Y_val


# ---------------------------------------------------------------------------
# Espacio de busqueda
# ---------------------------------------------------------------------------
DENSE_HIDDEN = [(8,), (16,), (32,), (64,), (128,), (16, 8), (32, 16), (64, 32), (128, 64)]
CNN_FILTERS = [(8,), (16,), (32,), (64,), (8, 16), (16, 32), (32, 64), (16, 32, 64), (64, 128, 128)]
RNN_UNITS = [(8,), (16,), (32,), (64,), (16, 8), (32, 16), (64, 128)]
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
L2S = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LRS = [1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5]
BATCHES = [16, 32, 64, 128]
DENSE_UNITS = [8, 16, 32, 64, 100]
KERNELS = [3, 5, 7]


def sample_config(rng: np.random.Generator, family: str) -> dict:
    cfg = {
        "family": family,
        "dropout": float(rng.choice(DROPOUTS)),
        "l2": float(rng.choice(L2S)),
        "learning_rate": float(rng.choice(LRS)),
        "batch_size": int(rng.choice(BATCHES)),
    }
    if family == "dense":
        cfg["hidden_units"] = DENSE_HIDDEN[rng.integers(len(DENSE_HIDDEN))]
    elif family == "cnn":
        cfg["conv_filters"] = CNN_FILTERS[rng.integers(len(CNN_FILTERS))]
        cfg["dense_units"] = int(rng.choice(DENSE_UNITS))
        cfg["global_pool"] = bool(rng.integers(2))
        cfg["kernel_size"] = int(rng.choice(KERNELS))
    elif family == "rnn":
        cfg["lstm_units"] = RNN_UNITS[rng.integers(len(RNN_UNITS))]
        cfg["dense_units"] = int(rng.choice(DENSE_UNITS))
        cfg["recurrent_dropout"] = float(rng.choice([0.0, 0.1, 0.2, 0.3]))
    return cfg


def build(cfg: dict, window_x: int, n_channels: int, output_dim: int, loss: str):
    common = dict(dropout=cfg["dropout"], l2=cfg["l2"], learning_rate=cfg["learning_rate"])
    if cfg["family"] == "dense":
        return modelos.build_predictor_dense(
            window_x, n_channels, output_dim,
            hidden_units=cfg["hidden_units"], loss=loss, **common
        )
    if cfg["family"] == "cnn":
        return modelos.build_predictor_cnn(
            window_x, n_channels, output_dim,
            conv_filters=cfg["conv_filters"], dense_units=cfg["dense_units"],
            kernel_size=cfg["kernel_size"], global_pool=cfg["global_pool"],
            loss=loss, **common
        )
    if cfg["family"] == "rnn":
        return modelos.build_predictor_rnn(
            window_x, n_channels, output_dim,
            lstm_units=cfg["lstm_units"], dense_units=cfg["dense_units"],
            recurrent_dropout=cfg["recurrent_dropout"], loss=loss, **common
        )
    raise ValueError(cfg["family"])


# ---------------------------------------------------------------------------
# Evaluacion de una configuracion
# ---------------------------------------------------------------------------
def evaluate_config(cfg, X_tr, Y_tr, X_val, Y_val, loss, seeds=(0, 1, 2)):
    """Entrena `cfg` con varias semillas y devuelve metricas promediadas.
    Con tan pocas muestras la varianza entre semillas es alta, asi que una
    sola ejecucion no es fiable para comparar configuraciones."""
    import tensorflow as tf
    from tensorflow import keras

    out = []
    for seed in seeds:
        keras.utils.set_random_seed(int(seed))
        model = build(cfg, X_tr.shape[1], X_tr.shape[2], Y_tr.shape[1], loss)
        hist = model.fit(
            X_tr, Y_tr, epochs=MAX_EPOCHS, batch_size=cfg["batch_size"],
            validation_data=(X_val, Y_val), verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=SEARCH_PATIENCE,
                    restore_best_weights=True,
                )
            ],
        )
        h = hist.history
        val = np.array(h["val_loss"])
        tr = np.array(h["loss"])
        best_ep = int(val.argmin())
        out.append({
            "best_val": float(val.min()),
            "final_val": float(val[-1]),
            "train_at_best": float(tr[best_ep]),
            "best_epoch": best_ep,
            "epochs_run": len(val),
            # cuanto sube la validacion despues del minimo (1.0 = no sube):
            "overfit_ratio": float(val[-1] / val.min()),
            # cuanto se separa validacion de entrenamiento en el optimo:
            "gap_ratio": float(val.min() / max(tr[best_ep], 1e-12)),
            "n_params": int(model.count_params()),
        })
        del model
        keras.backend.clear_session()
        gc.collect()

    df = pd.DataFrame(out)
    res = {f"{c}_mean": float(df[c].mean()) for c in df.columns}
    res["best_val_std"] = float(df["best_val"].std())
    return res


def evaluate_config_wf(cfg, X, Y, idx, is_syn, synth_years, folds, loss, seeds=(0,),
                       embargo_days: int = config.WINDOW_X_DAYS):
    """Igual que `evaluate_config` pero con validacion WALK-FORWARD: la
    configuracion se entrena y valida en varios cortes temporales
    sucesivos (entrenar con el pasado, validar con el futuro inmediato) y
    se agregan los resultados.

    La metrica que de verdad importa aqui no es solo la media entre
    cortes, sino tambien su DISPERSION (`best_val_std_folds`): una
    configuracion que gana en los 4 periodos es creible; una que gana solo
    en uno probablemente se este ajustando al ruido de ese periodo."""
    from tensorflow import keras

    out = []
    for fold_i, (vs, ve) in enumerate(folds):
        X_tr, Y_tr, X_v, Y_v, _ = tu.split_fold(
            X, Y, idx, vs, ve, synth_years=synth_years,
            synth_anchor=config.REAL_INTRADAY_START_DATE, is_synthetic=is_syn,
            embargo_days=embargo_days,
        )
        if len(X_tr) < 30 or len(X_v) < 10:
            continue
        for seed in seeds:
            keras.utils.set_random_seed(int(seed))
            model = build(cfg, X_tr.shape[1], X_tr.shape[2], Y_tr.shape[1], loss)
            hist = model.fit(
                X_tr, Y_tr, epochs=MAX_EPOCHS, batch_size=cfg["batch_size"],
                validation_data=(X_v, Y_v), verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=SEARCH_PATIENCE,
                    restore_best_weights=True)],
            )
            val = np.array(hist.history["val_loss"])
            tr = np.array(hist.history["loss"])
            best_ep = int(val.argmin())
            out.append({
                "fold": fold_i,
                "best_val": float(val.min()),
                "final_val": float(val[-1]),
                "overfit_ratio": float(val[-1] / val.min()),
                "gap_ratio": float(val.min() / max(tr[best_ep], 1e-12)),
                "epochs_run": len(val),
                "n_params": int(model.count_params()),
            })
            del model
            keras.backend.clear_session()
            gc.collect()

    if not out:
        raise ValueError("ningun fold utilizable")
    df = pd.DataFrame(out)
    per_fold = df.groupby("fold")["best_val"].mean()
    res = {f"{c}_mean": float(df[c].mean()) for c in df.columns if c != "fold"}
    # dispersion ENTRE cortes temporales = fiabilidad de la configuracion
    res["best_val_std_folds"] = float(per_fold.std())
    res["best_val_worst_fold"] = float(per_fold.max())
    res["n_folds_ok"] = int(per_fold.size)
    return res


def cfg_to_row(cfg: dict) -> dict:
    row = dict(cfg)
    for k in ("hidden_units", "conv_filters", "lstm_units"):
        if k in row:
            row[k] = "x".join(str(v) for v in row[k])
    return row


def save_atomic(rows: list, path: Path):
    """Guarda el CSV de forma atomica: escribe en un temporal y lo renombra.
    Asi, si el proceso muere justo mientras guarda, el CSV bueno anterior
    sigue intacto en vez de quedarse a medias."""
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
def run_stage(stage: str, synth_years: int, minutes: float, seeds, loss: str,
              worker: int = 0, families=("dense", "cnn", "rnn"),
              walk_forward: bool = False, n_folds: int = 4, val_months: int = 3,
              embargo_days: int = config.WINDOW_X_DAYS, generator: str = "noise"):
    # El nombre del fichero codifica el protocolo, para poder guardar y
    # comparar en el informe la variante CON purga y la variante SIN purga.
    suffix = f"wf_emb{embargo_days}" if walk_forward else "single"
    out_path = config.TABLES_DIR / f"hpsearch_{stage}_{suffix}_w{worker}.csv"

    if walk_forward:
        X, Y, idx, is_syn = load_full(generator)
        folds = tu.walk_forward_folds(n_folds=n_folds, val_months=val_months)
        n_tr = sum(len(tu.split_fold(X, Y, idx, vs, ve, synth_years,
                                     config.REAL_INTRADAY_START_DATE, is_syn,
                                     embargo_days)[0]) for vs, ve in folds)
        print(f"[{stage}/wf/w{worker}] {len(folds)} cortes walk-forward | "
              f"train acumulado {n_tr} | embargo {embargo_days}d | "
              f"generador '{generator}' | presupuesto {minutes:.0f} min | "
              f"familias {list(families)}", flush=True)
    else:
        X_tr, Y_tr, X_val, Y_val = load_split(synth_years, generator)
        print(f"[{stage}/w{worker}] train {X_tr.shape} | val {X_val.shape} | "
              f"presupuesto {minutes:.0f} min | familias {list(families)}", flush=True)

    # Semilla distinta por worker: cada proceso explora una zona distinta
    # del espacio en vez de repetir las mismas configuraciones.
    rng = np.random.default_rng(12345 + 1000 * worker)
    rows = []
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
        print(f"[{stage}/w{worker}] reanudando: {len(rows)} configs ya evaluadas", flush=True)

    t_end = time.time() + minutes * 60
    i = 0
    while time.time() < t_end:
        family = families[i % len(families)]
        cfg = sample_config(rng, family)
        i += 1
        try:
            t0 = time.time()
            if walk_forward:
                metrics = evaluate_config_wf(
                    cfg, X, Y, idx, is_syn, synth_years, folds, loss, seeds,
                    embargo_days=embargo_days,
                )
                extra = (f"std_folds={metrics['best_val_std_folds']:.6f} "
                         f"peor={metrics['best_val_worst_fold']:.6f}")
            else:
                metrics = evaluate_config(cfg, X_tr, Y_tr, X_val, Y_val, loss, seeds)
                extra = f"overfit={metrics['overfit_ratio_mean']:.2f}"
            row = {**cfg_to_row(cfg), **metrics, "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            save_atomic(rows, out_path)
            print(
                f"[{stage}/w{worker}][{len(rows):4d}] {family:5s} "
                f"val={metrics['best_val_mean']:.6f} {extra} "
                f"params={int(metrics['n_params_mean']):>7d} "
                f"({row['seconds']:.0f}s)",
                flush=True,
            )
        except Exception as e:  # una config mala no debe matar la busqueda
            print(f"[{stage}/w{worker}] config fallida ({family}): "
                  f"{type(e).__name__}: {e}", flush=True)

    print(f"[{stage}/w{worker}] TERMINADO: {len(rows)} configs -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=600)
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--families", default="dense,cnn,rnn")
    ap.add_argument("--walk-forward", action="store_true",
                    help="validacion walk-forward con purga (protocolo riguroso)")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--val-months", type=int, default=3)
    ap.add_argument("--embargo-days", type=int, default=config.WINDOW_X_DAYS)
    ap.add_argument("--generator", default="noise",
                    help="que dataset sintetico usar en la etapa B")
    ap.add_argument("--seeds", type=int, default=0,
                    help="0 = automatico (3 en corte unico etapa A, 1 en walk-forward)")
    args = ap.parse_args()

    families = tuple(args.families.split(","))
    synth_years = 0 if args.stage == "A" else config.SYNTH_DEPTH_YEARS_GRID[-1]
    stage_name = "A_real" if args.stage == "A" else "B_full"

    if args.seeds:
        seeds = tuple(range(args.seeds))
    elif args.walk_forward:
        # En walk-forward los 4 cortes ya promedian la varianza (y ademas
        # miden estabilidad temporal, que es mas informativo que repetir
        # la misma particion con otra semilla), asi que 1 semilla basta.
        seeds = (0,)
    else:
        seeds = (0, 1, 2) if args.stage == "A" else (0, 1)

    run_stage(stage_name, synth_years, args.minutes, seeds=seeds, loss="mae",
              worker=args.worker, families=families,
              walk_forward=args.walk_forward, n_folds=args.n_folds,
              val_months=args.val_months, embargo_days=args.embargo_days,
              generator=args.generator)


if __name__ == "__main__":
    main()
