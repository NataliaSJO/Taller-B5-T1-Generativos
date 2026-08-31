"""
v2 — CONTROL POSITIVO: cambiar el target de retorno a volatilidad.

Por que hace falta este experimento
-----------------------------------
El proyecto concluye que los datos sinteticos no mejoran la prediccion.
Esa conclusion tiene un agujero: no distingue entre

    (a) "los sinteticos no aportan informacion util", y
    (b) "el predictor / el pipeline no funciona".

Sin descartar (b), (a) no es defendible. Este script lo descarta cambiando
UNA sola cosa — el target — y dejando todo lo demas igual:

    entrada  X : IDENTICA (la de los dataset_<gen>.npz del notebook 03,
                 25 canales de retorno + 25 de volatilidad, esta ultima
                 sintetica en el tramo historico)
    target   Y : volatilidad del dia siguiente, en vez del retorno

La volatilidad del dia siguiente SI es predecible — es exactamente lo que
significa el clustering que mide `backfill_persistente.medir_persistencia`
(persistencia 0.639 en la serie real). Asi que:

  - Si con este target el modelo aprende de verdad (bate al ingenuo), el
    pipeline esta validado y el resultado nulo en retornos es una
    propiedad del MERCADO, no un fallo del codigo. Esa es la afirmacion
    fuerte que el proyecto necesita.
  - Y ademas es el escenario donde los sinteticos deberian ayudar de
    verdad: el canal sintetico es la historia del propio objetivo.

De donde sale el target
-----------------------
De **Garman-Klass sobre el OHLC diario REAL** (ver `volatilidad_ohlc.py`),
no de la volatilidad realizada de 5 minutos. Es deliberado y es lo que
evita la circularidad: la realizada solo existe en los ultimos 2 anios, asi
que usarla obligaria a poner volatilidad SINTETICA como target en el tramo
historico — y entonces estariamos midiendo si el sintetico predice al
sintetico. Garman-Klass es real en los 30 anios completos y correlaciona
0.812 con la realizada verdadera en la ventana donde se pueden comparar.
Target real en todas partes; lo unico sintetico sigue siendo la ENTRADA.

El baseline que importa
-----------------------
`persistencia_naive`: predecir vol(t+1) = vol(t). En prediccion de
volatilidad es el baseline estandar y es MUY dificil de batir, porque la
volatilidad es fuertemente autocorrelada. Un modelo que no lo bata no vale
nada, por muy buena que sea su correlacion en bruto. Se reporta por eso el
"skill" = 1 - MAE_modelo / MAE_ingenuo.

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

from src import config, modelos, train_utils as tu  # noqa: E402
import volatilidad_ohlc as vol_ohlc  # noqa: E402

GENERADORES = ["noise", "gaussian", "rbig", "gan"]


def construir_target_volatilidad(idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Target = volatilidad Garman-Klass del dia SIGUIENTE, para los 25
    bancos del predictor, alineada con el `idx` de las ventanas.

    `idx` es la fecha del ULTIMO dia de cada ventana X (el "hoy" desde el
    que se predice, ver `features.build_xy_windows`), asi que el target es
    el valor en el siguiente dia de mercado: un `shift(-1)` sobre el
    calendario diario.

    Devuelve (Y, mascara_valida): las filas sin target (ultimo dia, o
    huecos) se marcan para descartarlas en todos los modelos por igual.
    """
    rangos = vol_ohlc.panel_volatilidad_ohlc(config.PREDICTOR_TICKERS)
    gk = pd.DataFrame({tk: d["garman_klass"] for tk, d in rangos.items()}).sort_index()
    gk = gk[config.PREDICTOR_TICKERS]

    # el valor de manana, indexado por la fecha de hoy
    gk_manana = gk.shift(-1)
    Y = gk_manana.reindex(idx).values.astype("float32")
    valida = np.isfinite(Y).all(axis=1)
    return Y, valida


def metricas_volatilidad(y_true: np.ndarray, y_pred: np.ndarray, mae_naive: float) -> dict:
    """MAE, R2 y correlacion agrupadas, mas el `skill` contra el ingenuo.
    El skill es la cifra que decide: positivo = bate a "manana como hoy"."""
    err = y_true - y_pred
    sse = float(np.sum(err**2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    mae = float(np.mean(np.abs(err)))
    return {
        "mae": mae,
        "r2": 1.0 - sse / sst if sst > 0 else np.nan,
        "corr": float(np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]),
        "skill_vs_naive": 1.0 - mae / mae_naive if mae_naive > 0 else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--paciencia", type=int, default=100)
    args = ap.parse_args()
    semillas = tuple(range(args.semillas))

    from tensorflow import keras

    # ---- datos: X identica a la de v1, target nuevo --------------------
    base = np.load(config.INTERIM_DIR / "dataset_rbig.npz", allow_pickle=True)
    idx = pd.DatetimeIndex(base["idx"])
    Y_vol, valida = construir_target_volatilidad(idx)
    print(f"ventanas: {len(idx)} | con target valido: {valida.sum()}", flush=True)

    idx_v = idx[valida]
    Y_v = Y_vol[valida]
    val = (idx_v >= pd.Timestamp(config.VAL_START_DATE)) & (
        idx_v < pd.Timestamp(config.REAL_TEST_HOLDOUT_START_DATE)
    )
    test = idx_v >= pd.Timestamp(config.REAL_TEST_HOLDOUT_START_DATE)

    # ---- baseline ingenuo: vol(t+1) = vol(t) ---------------------------
    rangos = vol_ohlc.panel_volatilidad_ohlc(config.PREDICTOR_TICKERS)
    gk = pd.DataFrame({tk: d["garman_klass"] for tk, d in rangos.items()}).sort_index()
    gk_hoy = gk[config.PREDICTOR_TICKERS].reindex(idx_v).values.astype("float32")
    mae_naive = float(np.mean(np.abs(Y_v[test] - gk_hoy[test])))
    naive = metricas_volatilidad(Y_v[test], gk_hoy[test], mae_naive)
    print(f"  BASELINE ingenuo (vol manana = vol hoy): MAE={naive['mae']:.6f} "
          f"corr={naive['corr']:.3f}", flush=True)

    filas = [{"modelo": "0_persistencia_naive", "pct_objetivo": np.nan,
              "n_train": 0, "pct_synth": np.nan, **naive}]

    def entrenar(X_tr, Y_tr, X_val, Y_val, X_te, Y_te, n_canales):
        res = []
        for s in semillas:
            keras.utils.set_random_seed(int(s))
            m = modelos.build_predictor_cnn(
                config.WINDOW_X_DAYS, n_canales, config.N_PREDICTOR_TICKERS,
                conv_filters=(8, 16), global_pool=True, loss="mae",
                learning_rate=1e-3,
            )
            m.fit(X_tr, Y_tr, epochs=args.epochs, batch_size=64,
                  validation_data=(X_val, Y_val), verbose=0,
                  callbacks=tu._make_early_stopping(args.paciencia))
            res.append(metricas_volatilidad(Y_te, m.predict(X_te, verbose=0), mae_naive))
            del m
            keras.backend.clear_session()
            gc.collect()
        d = pd.DataFrame(res)
        return {c: float(d[c].mean()) for c in d.columns} | {"mae_std_semillas": float(d.mae.std())}

    # ---- rejilla: % de sinteticos x generador --------------------------
    for gen in GENERADORES:
        npz = np.load(config.INTERIM_DIR / f"dataset_{gen}.npz", allow_pickle=True)
        X = npz["X"][valida]
        is_syn = npz["is_synthetic"][valida]
        X_val_, Y_val_ = X[val], Y_v[val]
        X_te, Y_te = X[test], Y_v[test]

        for pct in config.PCT_SYNTH_GRID:
            nombre = "1_solo_reales" if pct <= 0 else gen
            if pct <= 0 and any(f["modelo"] == "1_solo_reales" for f in filas):
                continue  # identico para los 4 generadores: se entrena una vez
            X_tr, Y_tr, _, pct_real = tu.slice_by_pct(
                X, Y_v, idx_v, pct, config.VAL_START_DATE, is_syn
            )
            m = entrenar(X_tr, Y_tr, X_val_, Y_val_, X_te, Y_te, X.shape[-1])
            filas.append({"modelo": nombre, "pct_objetivo": pct,
                          "n_train": len(X_tr), "pct_synth": pct_real, **m})
            print(f"  {nombre:16s} pct={pct_real:5.2f} n={len(X_tr):5d} "
                  f"MAE={m['mae']:.6f} R2={m['r2']:+.3f} "
                  f"skill={m['skill_vs_naive']:+.3f}", flush=True)

    df = pd.DataFrame(filas)
    out = config.TABLES_DIR / "v2_target_volatilidad.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nGuardado -> {out}")


if __name__ == "__main__":
    main()
