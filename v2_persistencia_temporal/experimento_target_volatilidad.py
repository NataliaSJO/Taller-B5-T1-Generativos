"""
v2 — CONTROL POSITIVO: target de volatilidad, con rigor de la literatura.

Por que hace falta este experimento
-----------------------------------
El proyecto concluye que los datos sinteticos no mejoran la prediccion.
Esa conclusion tiene un agujero: no distingue entre

    (a) "los sinteticos no aportan informacion util", y
    (b) "el predictor / el pipeline no funciona".

Sin descartar (b), (a) no es defendible. Este script lo descarta cambiando
el target — de retorno del dia siguiente a VOLATILIDAD del dia siguiente —
y dejando el resto del pipeline igual. La volatilidad SI es predecible (es
lo que significa el clustering: persistencia 0.639 en la serie real), asi
que si aqui el modelo aprende, el resultado nulo en retornos es una
propiedad del mercado y no un fallo del codigo.

RIGOR FINANCIERO
----------------
1. Se modela LOG-volatilidad, no volatilidad en nivel. La volatilidad
   realizada es aproximadamente lognormal (Andersen, Bollerslev, Diebold &
   Labys, 2003); en logaritmos el target es casi gaussiano y el error
   cuadratico es un criterio sensato. En nivel, con perdida MAE, el optimo
   es la MEDIANA condicional y la red degenera a predecir casi una
   constante: bate al ingenuo en MAE con correlacion ~0.07. Ese fallo se
   observo en la primera version de este script y es la razon del cambio.

2. El benchmark es HAR-RV (Corsi, 2009), no el paseo aleatorio. HAR
   regresa la volatilidad de manana sobre las medias diaria, semanal (5d)
   y mensual (22d) — captura la memoria larga con tres coeficientes y es
   el estandar de referencia en la literatura, notoriamente dificil de
   batir. Batir a un ingenuo ruidoso no significa nada; empatar con HAR si.

3. Se reportan tres baselines, no uno:
     - ingenuo    : vol(t+1) = vol(t)      (paseo aleatorio)
     - constante  : la mediana del train   (detecta el colapso a constante)
     - HAR-RV     : el benchmark serio
   La constante esta a proposito: si un modelo no la bate, no ha aprendido
   nada, por bueno que parezca su error.

4. Metricas propias del problema:
     - QLIKE (Patton, 2011): perdida ROBUSTA para volatilidad. Las demas
       (incluido R2) estan sesgadas porque la volatilidad no se observa,
       se estima con ruido; QLIKE y MSE son las unicas de uso comun que
       preservan el orden correcto de los modelos bajo esa condicion.
     - Regresion de Mincer-Zarnowitz: se regresa lo realizado sobre lo
       predicho. Un pronostico bien calibrado da constante 0 y pendiente
       1. Detecta sesgo sistematico que el error medio esconde.
     - Correlacion: detecta el colapso a constante (corr ~ 0).

5. El target sale de Garman-Klass sobre OHLC diario REAL (ver
   volatilidad_ohlc.py), no de la realizada de 5 min. Es deliberado: la
   realizada solo existe en los ultimos 2 anios, asi que usarla obligaria
   a poner volatilidad SINTETICA como target en el tramo historico y
   estariamos midiendo si el sintetico predice al sintetico. Garman-Klass
   es real en los 30 anios y correlaciona 0.812 con la realizada
   verdadera. Target real en todas partes; lo unico sintetico es la
   ENTRADA.

RIGOR INFORMATICO / DE IA
-------------------------
1. El modelo ve la MISMA informacion que sus baselines. En la primera
   version no la veia: el canal de volatilidad de la entrada era la
   realizada de 5 min (sintetica en el historico) mientras el target era
   Garman-Klass, asi que se le pedia autorregresion sobre una serie
   ausente de su entrada, y se comparaba contra un ingenuo que si la
   tenia. Ahora el log-GK real entra como canal. Sin esto la comparacion
   esta amanada en contra del modelo.

2. Estandarizacion con estadisticos calculados SOLO sobre el tramo de
   entrenamiento de cada configuracion, nunca sobre el conjunto completo:
   usar la media/desviacion globales filtra informacion del test.

3. Separacion temporal estricta train < val < test, sin barajar. El test
   (REAL_TEST_HOLDOUT_START_DATE en adelante) no lo ve ningun generador ni
   ninguna fase de entrenamiento.

4. Varias semillas por configuracion, con la desviacion reportada. Las
   diferencias esperadas son pequenas y una sola ejecucion no las
   distingue del ruido de inicializacion — de hecho en este proyecto ese
   ruido resulto ser del mismo orden que los efectos medidos.

5. Significancia con el test de Diebold-Mariano (1995) con errores
   estandar HAC de Newey-West, sobre el diferencial de perdida promediado
   POR DIA. Promediar por dia antes del contraste es lo correcto aqui: los
   25 bancos del mismo dia estan fuertemente correlacionados (un factor
   sectorial comun), asi que tratar las 3.050 predicciones como
   independientes exagera la significancia unas 5 veces.

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

# Suelo para la volatilidad antes de tomar logaritmos. Garman-Klass recorta
# la varianza a 0 cuando el termino de apertura-cierre domina al rango, y
# log(0) = -inf. El suelo equivale a una volatilidad diaria del 0.01%, muy
# por debajo de cualquier dia real de un banco.
SUELO_VOL = 1e-4


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def panel_log_volatilidad() -> pd.DataFrame:
    """log-volatilidad Garman-Klass diaria REAL de los 25 bancos, 30 anios."""
    rangos = vol_ohlc.panel_volatilidad_ohlc(config.PREDICTOR_TICKERS)
    gk = pd.DataFrame({tk: d["garman_klass"] for tk, d in rangos.items()}).sort_index()
    gk = gk[config.PREDICTOR_TICKERS]
    return np.log(gk.clip(lower=SUELO_VOL))


def features_har(log_vol: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Componentes HAR (Corsi, 2009) calculadas hasta el dia t incluido, es
    decir CAUSALES: media diaria, semanal (5 sesiones) y mensual (22)."""
    return {
        "d": log_vol,
        "s": log_vol.rolling(5, min_periods=5).mean(),
        "m": log_vol.rolling(22, min_periods=22).mean(),
    }


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def qlike(log_real: np.ndarray, log_pred: np.ndarray) -> float:
    """QLIKE de Patton (2011) sobre VARIANZAS, la perdida robusta al ruido
    del proxy de volatilidad:  QLIKE = sigma^2/h - log(sigma^2/h) - 1 >= 0,
    con minimo en h = sigma^2. Se entra en logaritmos y se exponencia."""
    r = np.exp(2.0 * log_real)   # varianza realizada
    h = np.exp(2.0 * log_pred)   # varianza pronosticada
    ratio = np.clip(r / np.maximum(h, 1e-12), 1e-8, 1e8)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mincer_zarnowitz(log_real: np.ndarray, log_pred: np.ndarray) -> tuple[float, float]:
    """Regresion realizado = a + b * predicho. Calibrado perfecto -> a=0, b=1.
    b < 1 indica pronostico demasiado volatil; b ~ 0, pronostico sin
    contenido (el caso del colapso a constante)."""
    x, y = log_pred.ravel(), log_real.ravel()
    if np.std(x) < 1e-12:
        return float("nan"), 0.0
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def metricas(log_real: np.ndarray, log_pred: np.ndarray) -> dict:
    """Bloque completo de metricas en espacio logaritmico + QLIKE en nivel."""
    err = log_real - log_pred
    sst = float(np.sum((log_real - log_real.mean()) ** 2))
    a, b = mincer_zarnowitz(log_real, log_pred)
    return {
        "mae_log": float(np.mean(np.abs(err))),
        "rmse_log": float(np.sqrt(np.mean(err**2))),
        "r2_log": 1.0 - float(np.sum(err**2)) / sst if sst > 0 else np.nan,
        "corr": float(np.corrcoef(log_real.ravel(), log_pred.ravel())[0, 1])
        if np.std(log_pred) > 1e-12 else 0.0,
        "qlike": qlike(log_real, log_pred),
        "mz_const": a,
        "mz_pendiente": b,
    }


def perdida_diaria(log_real: np.ndarray, log_pred: np.ndarray) -> np.ndarray:
    """Perdida cuadratica promediada POR DIA (una cifra por fecha), que es
    la serie sobre la que se hace Diebold-Mariano. Ver docstring del modulo:
    los 25 bancos del mismo dia no son observaciones independientes."""
    return np.mean((log_real - log_pred) ** 2, axis=1)


def diebold_mariano(p_modelo: np.ndarray, p_bench: np.ndarray, lags: int | None = None):
    """Test de Diebold-Mariano (1995) con varianza HAC de Newey-West.

    H0: ambos pronosticos tienen la misma precision esperada.
    Estadistico negativo => el modelo pierde MENOS que el benchmark.
    Devuelve (estadistico, p-valor bilateral)."""
    from scipy import stats

    d = np.asarray(p_modelo) - np.asarray(p_bench)
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    d_bar = d.mean()
    dc = d - d_bar
    if lags is None:  # regla habitual de Newey-West
        lags = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    gamma0 = float(np.mean(dc**2))
    var = gamma0
    for k in range(1, lags + 1):
        gk = float(np.mean(dc[k:] * dc[:-k]))
        var += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    var = max(var, 1e-20)
    dm = d_bar / np.sqrt(var / T)
    return float(dm), float(2.0 * (1.0 - stats.norm.cdf(abs(dm))))


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--paciencia", type=int, default=100)
    args = ap.parse_args()
    semillas = tuple(range(args.semillas))

    from tensorflow import keras

    # ---- panel real y ventanas ----------------------------------------
    log_vol = panel_log_volatilidad()
    har = features_har(log_vol)

    base = np.load(config.INTERIM_DIR / "dataset_rbig.npz", allow_pickle=True)
    idx = pd.DatetimeIndex(base["idx"])

    # target: log-vol del dia SIGUIENTE. `idx` es la fecha del ultimo dia de
    # la ventana X (el "hoy"), asi que el target es un shift(-1).
    Y = log_vol.shift(-1).reindex(idx).values.astype("float32")
    hoy = log_vol.reindex(idx).values.astype("float32")
    har_d = har["d"].reindex(idx).values.astype("float32")
    har_s = har["s"].reindex(idx).values.astype("float32")
    har_m = har["m"].reindex(idx).values.astype("float32")

    valida = (
        np.isfinite(Y).all(axis=1) & np.isfinite(hoy).all(axis=1)
        & np.isfinite(har_s).all(axis=1) & np.isfinite(har_m).all(axis=1)
    )
    idx_v = idx[valida]
    Y_v, hoy_v = Y[valida], hoy[valida]
    Hd, Hs, Hm = har_d[valida], har_s[valida], har_m[valida]
    print(f"ventanas: {len(idx)} | validas: {valida.sum()}", flush=True)

    tr_bench = idx_v < pd.Timestamp(config.VAL_START_DATE)
    test = idx_v >= pd.Timestamp(config.REAL_TEST_HOLDOUT_START_DATE)
    val = (idx_v >= pd.Timestamp(config.VAL_START_DATE)) & ~test
    print(f"train(bench)={tr_bench.sum()} val={val.sum()} test={test.sum()} dias", flush=True)

    filas, perdidas = [], {}

    def registrar(nombre, pred_test, extra=None):
        m = metricas(Y_v[test], pred_test)
        perdidas[nombre] = perdida_diaria(Y_v[test], pred_test)
        filas.append({"modelo": nombre, **(extra or {}), **m})
        print(f"  {nombre:22s} QLIKE={m['qlike']:.4f} rmse={m['rmse_log']:.4f} "
              f"corr={m['corr']:+.3f} MZ_b={m['mz_pendiente']:+.3f}", flush=True)

    # ---- baseline 1: ingenuo (paseo aleatorio) -------------------------
    registrar("b1_ingenuo", hoy_v[test])

    # ---- baseline 2: constante (mediana del train) ---------------------
    cte = np.median(Y_v[tr_bench], axis=0)
    registrar("b2_constante", np.tile(cte, (test.sum(), 1)))

    # ---- baseline 3: HAR-RV (Corsi 2009), OLS sobre el train -----------
    # pooled sobre los 25 bancos: un solo juego de coeficientes, como es
    # habitual cuando los activos son homogeneos (todos bancos US).
    def apilar(mask):
        return (np.column_stack([Hd[mask].ravel(), Hs[mask].ravel(), Hm[mask].ravel()]),
                Y_v[mask].ravel())

    A_tr, y_tr = apilar(tr_bench)
    A_tr = np.column_stack([np.ones(len(A_tr)), A_tr])
    coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
    A_te = np.column_stack([np.ones(test.sum() * Y_v.shape[1]),
                            Hd[test].ravel(), Hs[test].ravel(), Hm[test].ravel()])
    registrar("b3_har", (A_te @ coef).reshape(Y_v[test].shape))
    print(f"     HAR coef: c={coef[0]:+.3f} d={coef[1]:+.3f} "
          f"s={coef[2]:+.3f} m={coef[3]:+.3f}", flush=True)

    # ---- la red: rejilla % sinteticos x generador ----------------------
    # canal extra con el log-GK REAL, para que el modelo vea lo mismo que
    # sus baselines (ver punto 1 del rigor informatico).
    def entrenar(X_tr, Yt, X_val, Yv, X_te, n_canales):
        preds = []
        for s in semillas:
            keras.utils.set_random_seed(int(s))
            m = modelos.build_predictor_cnn(
                config.WINDOW_X_DAYS, n_canales, config.N_PREDICTOR_TICKERS,
                conv_filters=(32, 64), global_pool=True, loss="mse",
                learning_rate=1e-3, dropout=0.1,
            )
            m.fit(X_tr, Yt, epochs=args.epochs, batch_size=64,
                  validation_data=(X_val, Yv), verbose=0,
                  callbacks=tu._make_early_stopping(args.paciencia))
            preds.append(m.predict(X_te, verbose=0))
            del m
            keras.backend.clear_session()
            gc.collect()
        return preds

    for gen in GENERADORES:
        npz = np.load(config.INTERIM_DIR / f"dataset_{gen}.npz", allow_pickle=True)
        Xg = npz["X"][valida]
        is_syn = npz["is_synthetic"][valida]

        # canal log-GK real, replicado a lo largo de la ventana temporal
        gk_win = np.empty((len(idx_v), config.WINDOW_X_DAYS, Y_v.shape[1]), dtype="float32")
        pos = {d: i for i, d in enumerate(log_vol.index)}
        lv = log_vol.values.astype("float32")
        for i, f in enumerate(idx_v):
            j = pos[f]
            gk_win[i] = lv[j - config.WINDOW_X_DAYS + 1 : j + 1]
        X = np.concatenate([Xg, gk_win], axis=-1)

        for pct in config.PCT_SYNTH_GRID:
            nombre = "1_solo_reales" if pct <= 0 else gen
            if pct <= 0 and any(f["modelo"] == "1_solo_reales" for f in filas):
                continue
            X_tr, Y_tr, _, pct_real = tu.slice_by_pct(
                X, Y_v, idx_v, pct, config.VAL_START_DATE, is_syn
            )
            # estandarizacion con estadisticos SOLO del train de esta config
            mu = X_tr.reshape(-1, X.shape[-1]).mean(axis=0)
            sd = X_tr.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
            muy, sdy = Y_tr.mean(axis=0), Y_tr.std(axis=0) + 1e-8
            z = lambda A: ((A - mu) / sd).astype("float32")

            preds = entrenar(z(X_tr), (Y_tr - muy) / sdy, z(X[val]),
                             (Y_v[val] - muy) / sdy, z(X[test]), X.shape[-1])
            preds = [p * sdy + muy for p in preds]
            por_semilla = [metricas(Y_v[test], p)["qlike"] for p in preds]
            registrar(f"{nombre}_{int(pct*100):03d}", np.mean(preds, axis=0),
                      {"generador": nombre, "pct_objetivo": pct, "n_train": len(X_tr),
                       "pct_synth": pct_real, "qlike_std_semillas": float(np.std(por_semilla))})

    # ---- significancia: todo contra HAR --------------------------------
    df = pd.DataFrame(filas)
    dm_stat, dm_p = [], []
    for n in df.modelo:
        if n == "b3_har":
            dm_stat.append(np.nan); dm_p.append(np.nan); continue
        s, p = diebold_mariano(perdidas[n], perdidas["b3_har"])
        dm_stat.append(s); dm_p.append(p)
    df["dm_vs_har"], df["dm_pvalor"] = dm_stat, dm_p

    out = config.TABLES_DIR / "v2_target_volatilidad.csv"
    df.to_csv(out, index=False)
    cols = ["modelo", "qlike", "rmse_log", "corr", "mz_pendiente", "dm_vs_har", "dm_pvalor"]
    print("\n" + df[cols].to_string(index=False))
    print(f"\nGuardado -> {out}")
    print("\nLectura: DM negativo y p<0.05 => bate a HAR de forma significativa.")


if __name__ == "__main__":
    main()
