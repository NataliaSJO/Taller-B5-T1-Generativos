"""Comparacion de arquitecturas del notebook 04, un proceso por candidata.

Es el ultimo tramo en serie que quedaba: 8 candidatas entrenadas una detras
de otra son ~10 min, y son INDEPENDIENTES entre si (cada una parte de
`set_seed()` y del mismo train/val). Repartidas en procesos bajan a lo que
tarde la mas lenta.

La equivalencia con el secuencial no es aproximada: cada worker llama a
`train_utils.run_architecture_comparison` con un diccionario de UNA entrada,
asi que ejecuta exactamente el mismo codigo, con la misma semilla puesta
justo antes de construir el modelo. Consolidar es concatenar.

    ./scripts/comparar_arquitecturas_paralelo.sh [python]

Escribe los dos ficheros que el notebook 04 reutiliza como checkpoint:
  reports/tables/04_comparacion_arquitecturas.csv
  datos/interim/04_checkpoint_arquitecturas_histories.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Igual que en rejilla_paralela: 1 hilo por proceso. Con N procesos peleando
# por los mismos cores, el paralelismo interno de TF resta en vez de sumar.
_THREADS = os.environ.get("REJILLA_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _THREADS)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", _THREADS)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config, modelos, train_utils as tu  # noqa: E402
import rejilla_paralela as rp  # noqa: E402

SHARD_DIR = config.INTERIM_DIR / "paralelo" / "arquitecturas"
CSV_FINAL = config.TABLES_DIR / "04_comparacion_arquitecturas.csv"
HIST_FINAL = config.INTERIM_DIR / "04_checkpoint_arquitecturas_histories.json"

# Mismo orden que el diccionario `architectures` del notebook 04: es el orden
# en el que se lee la tabla en el informe.
ORDEN = ["constante", "baseline", "linear", "dense",
         "cnn_1bloque", "cnn_3bloques", "rnn_1capa", "rnn_2capas"]


def datos():
    """Train (solo reales) / val / test, identicos a los del notebook 04."""
    ds = rp.cargar_datasets({rp.GENERADORES[0]})[rp.GENERADORES[0]]
    X, Y, idx, is_synth = rp.vista_horizonte(ds, config.WINDOW_Y_DAYS)
    X_tr, Y_tr, _, _ = tu.slice_by_depth(
        X, Y, idx, 0, rp.TRAIN_END, config.REAL_INTRADAY_START_DATE, is_synth)
    (X_val, Y_val), (X_test, Y_test) = tu.split_val_test(
        X, Y, idx, horizonte=config.WINDOW_Y_DAYS)
    return X_tr, Y_tr, X_val, Y_val, X_test, Y_test, X.shape[-1]


def fabrica(nombre: str, n_channels: int, loss: str, dropout: float, l2: float):
    """Las tres referencias no pasan por `rp.constructor` (no son redes)."""
    O = config.N_PREDICTOR_TICKERS
    if nombre == "constante":
        return lambda: modelos.build_predictor_constant()
    if nombre == "baseline":
        return lambda: modelos.build_predictor_baseline(output_dim=O)
    if nombre == "linear":
        return lambda: modelos.build_predictor_linear()
    return rp.constructor(nombre, n_channels, loss, dropout, l2)


def entrenar(nombre, epochs, batch_size, patience, loss, dropout, l2):
    X_tr, Y_tr, X_val, Y_val, X_test, Y_test, n_ch = datos()
    tabla, hist = tu.run_architecture_comparison(
        {nombre: fabrica(nombre, n_ch, loss, dropout, l2)},
        X_tr, Y_tr, X_val, Y_val, X_test, Y_test,
        epochs=epochs, batch_size=batch_size, verbose=0,
        early_stopping_patience=patience,
    )
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SHARD_DIR / f"{nombre}.csv")
    (SHARD_DIR / f"{nombre}.json").write_text(
        json.dumps({k: {m: [float(x) for x in v] for m, v in h.items()}
                    for k, h in hist.items()}, indent=2), encoding="utf-8")
    print(f"[{nombre}] listo  val_mae={tabla['val_mae'].iloc[0]:.6f}")


def merge():
    trozos, historias = [], {}
    for nombre in ORDEN:
        csv, js = SHARD_DIR / f"{nombre}.csv", SHARD_DIR / f"{nombre}.json"
        if not csv.exists():
            print(f"[merge] AVISO: falta {nombre}")
            continue
        trozos.append(pd.read_csv(csv, index_col=0))
        historias.update(json.loads(js.read_text(encoding="utf-8")))
    if not trozos:
        raise SystemExit("[merge] no hay ningun resultado que consolidar")
    tabla = pd.concat(trozos)
    CSV_FINAL.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_FINAL)
    HIST_FINAL.write_text(json.dumps(historias, indent=2), encoding="utf-8")
    print(f"[merge] {len(tabla)}/{len(ORDEN)} arquitecturas -> {CSV_FINAL.name}")
    print(f"[merge] ganadora por la regla de 1 e.e.: {tu.elegir_por_una_ee(tabla)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquitectura")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    a = ap.parse_args()
    if a.merge:
        merge()
    elif a.arquitectura:
        entrenar(a.arquitectura, a.epochs, a.batch_size, a.patience,
                 a.loss, a.dropout, a.l2)
    else:
        print("\n".join(ORDEN))


if __name__ == "__main__":
    main()
