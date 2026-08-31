"""
Experimento: ¿predice la fidelidad distribucional de un generador su
utilidad real aguas abajo?

La busqueda de hiperparametros (scripts/hp_search_generators.py) ordena los
generadores por lo bien que su distribucion sintetica se parece a la real
(MMD, Wasserstein, Frobenius). Pero lo que de verdad importa en este
proyecto no es que el sintetico "se parezca", sino que AYUDE a predecir. No
son lo mismo, y pueden discrepar: el generador de Ruido obtiene la mejor
fidelidad posible (MMD ~ 0) simplemente copiando muestras reales y
perturbandolas, sin aportar informacion nueva ninguna.

Este experimento lo contrasta directamente: se cogen varios GAN que abarcan
todo el rango de calidad distribucional (desde el mejor hasta el peor de la
busqueda) y, para cada uno, se recorre el pipeline COMPLETO —
entrenar GAN -> muestrear -> backfill de 28 anios -> entrenar el predictor —
midiendo el rendimiento en el MISMO test real. Al final se correlaciona
"calidad distribucional" con "utilidad aguas abajo".

Resultado informativo gane quien gane:
  - Si correlacionan, se valida usar la fidelidad como criterio de busqueda.
  - Si no correlacionan, el hallazgo es mas interesante todavia: elegir
    generadores por fidelidad distribucional seria enganoso, y habria que
    elegirlos por rendimiento aguas abajo.

Uso:
    python scripts/experimento_espectro_gan.py [--n-buenas 3] [--epochs 200]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import backfill as bf, config, features as feat, generators as gen  # noqa: E402
from src import modelos, train_utils as tu  # noqa: E402


def cargar_gans_del_espectro(n_buenas: int = 3) -> pd.DataFrame:
    """Selecciona GANs que cubran todo el rango de calidad: las `n_buenas`
    mejores, una intermedia (mediana) y dos malas (percentil 90 y la peor).
    Asi el eje X del experimento abarca ordenes de magnitud, no matices."""
    import glob

    d = pd.concat(
        [pd.read_csv(f) for f in glob.glob(str(config.TABLES_DIR / "hpsearch_gen_w*.csv"))],
        ignore_index=True,
    )
    g = d[d.family == "gan"].dropna(subset=["wasserstein_mean_mean"]).copy()
    # puntuacion compuesta por rangos (las tres metricas estan en escalas distintas)
    g["score"] = (
        g.mmd_mean.rank() + g.wasserstein_mean_mean.rank() + g.frobenius_corr_mean.rank()
    ) / 3.0
    g = g.sort_values("score").reset_index(drop=True)

    idx, etiquetas = [], []
    for k in range(min(n_buenas, len(g))):
        idx.append(k)
        etiquetas.append(f"buena_{k+1}")
    for pos, lbl in [(len(g) // 2, "intermedia"), (int(len(g) * 0.9), "mala"), (len(g) - 1, "muy_mala")]:
        if pos not in idx and pos < len(g):
            idx.append(pos)
            etiquetas.append(lbl)
    sel = g.iloc[idx].copy()
    sel["etiqueta"] = etiquetas
    return sel


def pipeline_completo(cfg: pd.Series, returns_pred, real_feats, pool_train,
                      X_val, Y_val, X_test, Y_test, idx_ref, epochs_pred: int):
    """GAN -> muestreo -> backfill 28 anios -> predictor -> metricas test."""
    g = gen.GANGenerator(
        latent_dim=int(cfg.latent_dim), epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        gen_hidden=tuple(int(x) for x in str(cfg.gen_hidden).split("x")),
        disc_hidden=tuple(int(x) for x in str(cfg.disc_hidden).split("x")),
        learning_rate=float(cfg.learning_rate),
        d_steps_per_g=int(cfg.d_steps_per_g), random_state=42,
    )
    g.fit(pool_train)
    synth = feat.clip_nonnegative_pool_columns(g.sample(50_000))

    full = bf.build_full_history_features(
        returns_pred, real_feats, synth,
        real_start=config.REAL_INTRADAY_START_DATE, k_neighbors=80, random_state=42,
    )
    rv = bf.rv_panel_from_full_history(full)
    combinado = pd.concat(
        [returns_pred[config.PREDICTOR_TICKERS],
         rv[config.PREDICTOR_TICKERS].add_suffix("_rv")], axis=1
    ).dropna()
    X, Y_wide, idx = feat.build_xy_windows(combinado, config.WINDOW_X_DAYS, config.WINDOW_Y_DAYS)
    Y = Y_wide[:, : config.N_PREDICTOR_TICKERS]

    X_tr, Y_tr, _, pct = tu.slice_by_depth(
        X, Y, idx, synth_years=config.SYNTH_DEPTH_YEARS_GRID[-1],
        train_end=config.VAL_START_DATE, synth_anchor=config.REAL_INTRADAY_START_DATE,
        is_synthetic=np.asarray(idx < pd.Timestamp(config.REAL_INTRADAY_START_DATE)),
    )
    modelo = modelos.build_predictor_cnn(
        config.WINDOW_X_DAYS, X.shape[-1], config.N_PREDICTOR_TICKERS,
        conv_filters=(64, 128, 128), loss="mae",
    )
    modelo.fit(X_tr, Y_tr, epochs=epochs_pred, batch_size=64,
               validation_data=(X_val, Y_val), verbose=0,
               callbacks=tu._make_early_stopping(40))
    m = tu.evaluate_predictor(modelo, X_test, Y_test)
    del modelo, g
    from tensorflow import keras
    keras.backend.clear_session()
    gc.collect()
    return m, pct, len(X_tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-buenas", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    sel = cargar_gans_del_espectro(args.n_buenas)
    print(f"{len(sel)} GAN seleccionados cubriendo el espectro de calidad:")
    for _, r in sel.iterrows():
        print(f"  {r.etiqueta:12s} MMD={r.mmd_mean:.4f} W1={r.wasserstein_mean_mean:.3f} "
              f"Frob={r.frobenius_corr_mean:.3f}")

    # datos (mismo split que los notebooks)
    returns_pred = pd.read_parquet(config.INTERIM_DIR / "returns_predictor.parquet")
    returns_pred = returns_pred[returns_pred.index >= pd.Timestamp(config.TOTAL_HISTORY_START_DATE)]
    intraday = pd.read_parquet(config.INTERIM_DIR / "intraday_features_real.parquet")
    real_feats = {
        tk: gg.set_index("date")[feat.INTRADAY_FEATURE_COLS]
        for tk, gg in intraday[intraday.ticker.isin(config.PREDICTOR_TICKERS)].groupby("ticker")
    }
    pool_full = np.load(config.INTERIM_DIR / "conditional_pool.npy")
    meta = pd.read_parquet(config.INTERIM_DIR / "conditional_pool_meta.parquet")
    pool = pool_full[~(meta["date"] >= pd.Timestamp(config.VAL_START_DATE)).values]
    rng0 = np.random.default_rng(42)
    sh = rng0.permutation(len(pool))
    pool_train = pool[sh[max(int(0.1 * len(pool)), 500):]]

    npz = np.load(config.INTERIM_DIR / "dataset_noise.npz", allow_pickle=True)
    idx_ref = pd.DatetimeIndex(npz["idx"])
    Xr, Yr = npz["X"], npz["Y"]
    vmask = (idx_ref >= pd.Timestamp(config.VAL_START_DATE)) & (
        idx_ref < pd.Timestamp(config.REAL_TEST_HOLDOUT_START_DATE))
    tmask = idx_ref >= pd.Timestamp(config.REAL_TEST_HOLDOUT_START_DATE)
    X_val, Y_val, X_test, Y_test = Xr[vmask], Yr[vmask], Xr[tmask], Yr[tmask]

    out_path = config.TABLES_DIR / "experimento_espectro_gan.csv"
    filas = []
    for _, cfg in sel.iterrows():
        t0 = time.time()
        try:
            m, pct, n_tr = pipeline_completo(
                cfg, returns_pred, real_feats, pool_train,
                X_val, Y_val, X_test, Y_test, idx_ref, args.epochs)
            filas.append({
                "etiqueta": cfg.etiqueta, "mmd": cfg.mmd_mean,
                "w1": cfg.wasserstein_mean_mean, "frobenius": cfg.frobenius_corr_mean,
                "test_mae": m["mae"], "test_mse": m["mse"],
                "precision_direccional": m["directional_accuracy"],
                "n_train": n_tr, "segundos": round(time.time() - t0, 1),
            })
            pd.DataFrame(filas).to_csv(out_path, index=False)
            print(f"  [{cfg.etiqueta:12s}] MMD={cfg.mmd_mean:.4f} -> test MAE={m['mae']:.6f} "
                  f"dir={m['directional_accuracy']:.1%} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{cfg.etiqueta}] FALLO: {type(e).__name__}: {e}", flush=True)

    if filas:
        df = pd.DataFrame(filas)
        print(f"\nGuardado -> {out_path}")
        if len(df) >= 3:
            print("\nCORRELACION calidad distribucional vs utilidad aguas abajo:")
            for met in ["mmd", "w1", "frobenius"]:
                c = df[met].corr(df["test_mae"], method="spearman")
                print(f"  Spearman({met}, test_MAE) = {c:+.3f}"
                      "   (positiva = mejor fidelidad -> mejor prediccion)")


if __name__ == "__main__":
    main()
