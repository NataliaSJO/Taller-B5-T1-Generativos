"""
v2 — Comparativa aguas abajo de cuatro variantes del canal de volatilidad.

La pregunta: recuperar la persistencia de la volatilidad (clustering),
¿mejora de verdad la prediccion, o el limite del problema esta en otro
sitio? Se comparan cuatro versiones del MISMO predictor, entrenadas sobre
la misma profundidad historica y evaluadas en el MISMO test real:

  1. sin_volatilidad  -> solo el canal de retorno (25 canales).
     Es el suelo: dice cuanto aporta la volatilidad, sea cual sea.
  2. v1_sin_memoria   -> backfill de v1 (dias independientes).
  3. v2_con_memoria   -> backfill secuencial de v2 (clustering recuperado).
  4. ohlc_real        -> volatilidad de Garman-Klass calculada del OHLC
     diario REAL de los 30 anios. No es sintetica: es el techo realista,
     lo mejor que se puede hacer sin datos de 5 minutos.

La cuarta variante es la referencia clave. Si la volatilidad real de OHLC
bate con holgura a las sinteticas, el mensaje es que el esfuerzo estaba
mal dirigido: habia informacion real disponible sin sintetizar nada.

No modifica nada de v1.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v2_persistencia_temporal"))

from src import backfill as bf, config, features as feat, generators as gen  # noqa: E402
from src import modelos, train_utils as tu  # noqa: E402
import backfill_persistente as v2  # noqa: E402
import volatilidad_ohlc as vol_ohlc  # noqa: E402

SYNTH_YEARS = config.SYNTH_DEPTH_YEARS_GRID[-1]


def _panel_a_dataset(returns_pred: pd.DataFrame, panel_rv: pd.DataFrame | None):
    """Monta (X, Y, idx) a partir de los retornos y un panel de volatilidad.
    Si `panel_rv` es None, el dataset lleva solo el canal de retorno."""
    if panel_rv is None:
        combinado = returns_pred[config.PREDICTOR_TICKERS].dropna()
    else:
        combinado = pd.concat(
            [returns_pred[config.PREDICTOR_TICKERS],
             panel_rv[config.PREDICTOR_TICKERS].add_suffix("_rv")], axis=1
        ).dropna()
    X, Y_wide, idx = feat.build_xy_windows(combinado, config.WINDOW_X_DAYS, config.WINDOW_Y_DAYS)
    return X, Y_wide[:, : config.N_PREDICTOR_TICKERS], idx


def _entrenar_y_evaluar(X, Y, idx, epochs: int, semillas=(0, 1)):
    """Entrena el predictor a profundidad maxima y evalua en el test real.
    Varias semillas porque las diferencias esperadas son pequenas y una
    sola ejecucion no las distinguiria del ruido de inicializacion."""
    from tensorflow import keras

    val = (idx >= pd.Timestamp(config.VAL_START_DATE)) & (idx < pd.Timestamp(config.REAL_TEST_HOLDOUT_START_DATE))
    test = idx >= pd.Timestamp(config.REAL_TEST_HOLDOUT_START_DATE)
    X_val, Y_val, X_test, Y_test = X[val], Y[val], X[test], Y[test]

    X_tr, Y_tr, _, _ = tu.slice_by_depth(
        X, Y, idx, synth_years=SYNTH_YEARS, train_end=config.VAL_START_DATE,
        synth_anchor=config.REAL_INTRADAY_START_DATE,
        is_synthetic=np.asarray(idx < pd.Timestamp(config.REAL_INTRADAY_START_DATE)),
    )

    res = []
    for s in semillas:
        keras.utils.set_random_seed(int(s))
        m = modelos.build_predictor_cnn(
            config.WINDOW_X_DAYS, X.shape[-1], config.N_PREDICTOR_TICKERS,
            conv_filters=(64, 128, 128), loss="mae",
        )
        m.fit(X_tr, Y_tr, epochs=epochs, batch_size=64,
              validation_data=(X_val, Y_val), verbose=0,
              callbacks=tu._make_early_stopping(40))
        res.append(tu.evaluate_predictor(m, X_test, Y_test))
        del m
        keras.backend.clear_session()
        gc.collect()
    d = pd.DataFrame(res)
    return {"test_mae": d.mae.mean(), "test_mse": d.mse.mean(),
            "precision_direccional": d.directional_accuracy.mean(),
            "mae_std_semillas": d.mae.std(), "n_train": len(X_tr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--semillas", type=int, default=2)
    args = ap.parse_args()
    semillas = tuple(range(args.semillas))

    returns_pred = pd.read_parquet(config.INTERIM_DIR / "returns_predictor.parquet")
    returns_pred = returns_pred[returns_pred.index >= pd.Timestamp(config.TOTAL_HISTORY_START_DATE)]
    intr = pd.read_parquet(config.INTERIM_DIR / "intraday_features_real.parquet")
    real_feats = {
        tk: g.set_index("date")[feat.INTRADAY_FEATURE_COLS].sort_index()
        for tk, g in intr.groupby("ticker")
    }
    real_feats_pred = {tk: v for tk, v in real_feats.items() if tk in config.PREDICTOR_TICKERS}

    filas = []

    # ---- 1. sin volatilidad -------------------------------------------
    X, Y, idx = _panel_a_dataset(returns_pred, None)
    filas.append({"variante": "1_sin_volatilidad", "persistencia_lag1": np.nan,
                  **_entrenar_y_evaluar(X, Y, idx, args.epochs, semillas)})
    print(f"  {filas[-1]['variante']:20s} MAE={filas[-1]['test_mae']:.6f} "
          f"dir={filas[-1]['precision_direccional']:.1%}", flush=True)

    # ---- 2. v1: backfill sin memoria ----------------------------------
    npz = np.load(config.INTERIM_DIR / "dataset_rbig.npz", allow_pickle=True)
    idx1 = pd.DatetimeIndex(npz["idx"])
    rv_v1 = pd.Series(npz["X"][:, -1, config.N_PREDICTOR_TICKERS], index=idx1)
    # OJO: medir la persistencia SOLO en el tramo sintetico. Si se mide sobre
    # toda la serie, el tramo real final (persistencia ~0.64) arrastra la
    # cifra hacia arriba (0.113 en vez de 0.075) y deja de ser comparable con
    # la fila de v2, que si se mide solo en el tramo sintetico.
    rv_v1_sint = rv_v1[rv_v1.index < pd.Timestamp(config.REAL_INTRADAY_START_DATE)]
    filas.append({"variante": "2_v1_sin_memoria",
                  "persistencia_lag1": v2.medir_persistencia(rv_v1_sint).get("pearson", np.nan),
                  **_entrenar_y_evaluar(npz["X"], npz["Y"], idx1, args.epochs, semillas)})
    print(f"  {filas[-1]['variante']:20s} MAE={filas[-1]['test_mae']:.6f} "
          f"dir={filas[-1]['precision_direccional']:.1%}", flush=True)

    # ---- 3. v2: backfill con memoria ----------------------------------
    from src import data_norgate as dn
    rg = dn.load_daily_prices(config.GENERATOR_TICKERS, start=config.REAL_INTRADAY_START_DATE)
    rets_gen = dn.compute_log_returns(rg, dropna=None)
    pool, meta = v2.construir_pool_con_retardo(rets_gen, real_feats)
    pool = pool[meta.date.values < np.datetime64(config.VAL_START_DATE)]
    g = gen.RBIGGenerator(n_iters=100, grid_size=1000, rotation="pca", random_state=42)
    synth = g.fit(pool).sample(60_000)
    synth[:, [1, 2, 5]] = np.clip(synth[:, [1, 2, 5]], 0, None)

    hist_v2 = v2.construir_historico_v2(returns_pred[config.PREDICTOR_TICKERS],
                                        real_feats_pred, synth, random_state=42)
    rv_panel_v2 = pd.DataFrame({tk: d["realized_vol"] for tk, d in hist_v2.items()}).sort_index()
    X, Y, idx = _panel_a_dataset(returns_pred, rv_panel_v2)
    pers = v2.medir_persistencia(rv_panel_v2["JPM"][rv_panel_v2.index < pd.Timestamp(config.REAL_INTRADAY_START_DATE)])
    filas.append({"variante": "3_v2_con_memoria", "persistencia_lag1": pers.get("pearson", np.nan),
                  **_entrenar_y_evaluar(X, Y, idx, args.epochs, semillas)})
    print(f"  {filas[-1]['variante']:20s} MAE={filas[-1]['test_mae']:.6f} "
          f"dir={filas[-1]['precision_direccional']:.1%}", flush=True)

    # ---- 4. volatilidad REAL de Garman-Klass (30 anios) ----------------
    rangos = vol_ohlc.panel_volatilidad_ohlc(config.PREDICTOR_TICKERS)
    rv_gk = pd.DataFrame({tk: d["garman_klass"] for tk, d in rangos.items()}).sort_index()
    X, Y, idx = _panel_a_dataset(returns_pred, rv_gk)
    pers = v2.medir_persistencia(rv_gk["JPM"])
    filas.append({"variante": "4_ohlc_real_gk", "persistencia_lag1": pers.get("pearson", np.nan),
                  **_entrenar_y_evaluar(X, Y, idx, args.epochs, semillas)})
    print(f"  {filas[-1]['variante']:20s} MAE={filas[-1]['test_mae']:.6f} "
          f"dir={filas[-1]['precision_direccional']:.1%}", flush=True)

    df = pd.DataFrame(filas)
    out = config.TABLES_DIR / "v2_comparativa_variantes.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nGuardado -> {out}")


if __name__ == "__main__":
    main()
