"""
Ingenieria de features:
  - ventanas X/Y de retornos diarios (mismo esquema que los notebooks de clase)
  - features intradia derivadas de las barras de 5 min (volatilidad realizada,
    gap overnight, retorno de apertura/cierre, rango) y su perfil a lo largo
    del dia (para el EDA de "estudiar la distribucion a lo largo del dia")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Ventanas X / Y sobre retornos diarios (esquema identico a los notebooks de
# clase: X = ventana pasada de `window_x` dias, Y = media de los `window_y`
# dias siguientes)
# ---------------------------------------------------------------------------
def build_xy_windows(
    returns: pd.DataFrame,
    window_x: int = config.WINDOW_X_DAYS,
    window_y: int = config.WINDOW_Y_DAYS,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Replica el bucle de los notebooks de clase:

        for i in range(window_x, len(returns) - window_y):
            X = returns[i-window_x : i]
            Y = mean(returns[i : i+window_y])

    Devuelve X (n, window_x, n_tickers), Y (n, n_tickers) y el indice de
    fechas (la fecha del ULTIMO dia de cada ventana X, es decir el "hoy"
    desde el que se predice), util para hacer splits train/test por fecha
    sin fuga temporal.
    """
    values = returns.values
    dates = returns.index
    n = len(returns)

    X_list, Y_list, idx_list = [], [], []
    for i in range(window_x, n - window_y):
        X_list.append(values[i - window_x : i])
        Y_list.append(values[i : i + window_y].mean(axis=0))
        idx_list.append(dates[i - 1])

    X = np.array(X_list)
    Y = np.array(Y_list)
    idx = pd.DatetimeIndex(idx_list)
    return X, Y, idx


def split_by_date(idx: pd.DatetimeIndex, split_date: str) -> tuple[np.ndarray, np.ndarray]:
    """Mascaras booleanas (train, test) segun una fecha de corte, para evitar
    la fuga temporal del train_test_split aleatorio de los notebooks de clase
    en las evaluaciones finales (aunque para replicar el ejercicio de clase
    tal cual tambien se usa random split en el notebook 02)."""
    train_mask = idx < pd.Timestamp(split_date)
    test_mask = ~train_mask
    return train_mask, test_mask


# ---------------------------------------------------------------------------
# Features intradia a partir de barras de 5 minutos
# ---------------------------------------------------------------------------
def _session_date(index: pd.DatetimeIndex) -> pd.Index:
    """Fecha de sesion (naive, sin hora) a partir de un indice datetime UTC."""
    return pd.Index(index.tz_convert(None).normalize())


def daily_intraday_features(bars: pd.DataFrame) -> pd.DataFrame:
    """A partir de barras de 5 min (index datetime UTC, columnas
    open/high/low/close/volume) de UN ticker, calcula por dia de sesion:

      - realized_vol : sqrt(sum(log-retornos 5min ^ 2))  (volatilidad
        realizada intradia, el estimador de referencia en microestructura)
      - open_30m_ret : log-retorno de los primeros 30 min (6 barras)
      - close_30m_ret: log-retorno de los ultimos 30 min (6 barras)
      - hl_range     : (max(high) - min(low)) / open del dia
      - n_bars       : numero de barras de 5 min ese dia (control de calidad)
    """
    if bars.empty:
        return pd.DataFrame(
            columns=["realized_vol", "open_30m_ret", "close_30m_ret", "hl_range", "n_bars"]
        )

    df = bars.copy()
    df["session_date"] = _session_date(df.index)
    df["log_close"] = np.log(df["close"])

    rows = []
    for date, g in df.groupby("session_date"):
        g = g.sort_index()
        r5 = g["log_close"].diff().dropna()
        rv = float(np.sqrt(np.sum(r5.values ** 2))) if len(r5) else np.nan

        n_open = min(6, len(g))
        open_ret = float(np.log(g["close"].iloc[n_open - 1] / g["open"].iloc[0])) if n_open else np.nan

        n_close = min(6, len(g))
        close_ret = (
            float(np.log(g["close"].iloc[-1] / g["open"].iloc[-n_close]))
            if n_close
            else np.nan
        )

        hl_range = float((g["high"].max() - g["low"].min()) / g["open"].iloc[0])

        rows.append(
            {
                "date": date,
                "realized_vol": rv,
                "open_30m_ret": open_ret,
                "close_30m_ret": close_ret,
                "hl_range": hl_range,
                "n_bars": len(g),
            }
        )

    out = pd.DataFrame(rows).set_index("date").sort_index()
    return out


def intraday_time_of_day_profile(bars: pd.DataFrame) -> pd.DataFrame:
    """Perfil de retornos/volumen a lo largo del dia (agrupando por la hora:minuto
    de cada barra de 5 min, sin importar la fecha). Es la pieza central del
    EDA pedido: "estudiar la distribucion a lo largo del dia". Devuelve, por
    cada uno de los ~78 slots horarios de la sesion:
      - mean_ret, std_ret : media y desviacion del log-retorno de 5 min
      - mean_abs_ret      : |retorno| medio (proxy de volatilidad por slot)
      - mean_volume
    """
    if bars.empty:
        return pd.DataFrame()

    df = bars.copy()
    df["session_date"] = _session_date(df.index)
    # Hora LOCAL del mercado (America/New_York): en UTC el mismo minuto de
    # apertura/cierre cae en dos relojes distintos segun horario de
    # verano/invierno, lo que duplicaria las franjas horarias (79 -> ~90).
    df["time_of_day"] = df.index.tz_convert("America/New_York").strftime("%H:%M")
    df["log_close"] = np.log(df["close"])
    df["ret_5m"] = df.groupby("session_date")["log_close"].diff()

    prof = (
        df.dropna(subset=["ret_5m"])
        .groupby("time_of_day")
        .agg(
            mean_ret=("ret_5m", "mean"),
            std_ret=("ret_5m", "std"),
            mean_abs_ret=("ret_5m", lambda x: x.abs().mean()),
            mean_volume=("volume", "mean"),
        )
        .sort_index()
    )
    return prof


# ---------------------------------------------------------------------------
# Pool conjunto (retorno diario real, features intradia reales) para
# entrenar los 4 generadores condicionales (notebook 02)
# ---------------------------------------------------------------------------
INTRADAY_FEATURE_COLS = ["realized_vol", "open_30m_ret", "close_30m_ret", "hl_range"]

# Columnas del pool conjunto [log_return, *INTRADAY_FEATURE_COLS] que son
# fisicamente no-negativas (realized_vol = sqrt(suma de cuadrados), hl_range
# = (high-low)/open). Un generador NO restringido a soportes positivos (la
# Gaussiana multivariante, sobre todo) puede muestrear valores negativos
# ahi, que no tienen sentido: hay que recortarlos a 0 tras generar.
NONNEGATIVE_POOL_COLS = ["realized_vol", "hl_range"]


def clip_nonnegative_pool_columns(
    pool: np.ndarray, feature_cols: list[str] = INTRADAY_FEATURE_COLS
) -> np.ndarray:
    """Recorta a >= 0, in-place sobre una copia, las columnas del pool
    conjunto [log_return, *feature_cols] que son fisicamente no-negativas
    (ver NONNEGATIVE_POOL_COLS). Se aplica a la salida de CUALQUIER
    generador (los 4 por igual) antes de usarla en el backfill."""
    out = pool.copy()
    all_cols = ["log_return", *feature_cols]
    for j, col in enumerate(all_cols):
        if col in NONNEGATIVE_POOL_COLS:
            out[:, j] = np.clip(out[:, j], 0, None)
    return out


def build_conditional_pool(
    returns_daily: pd.DataFrame,
    intraday_feats_by_ticker: dict[str, pd.DataFrame],
    feature_cols: list[str] = INTRADAY_FEATURE_COLS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Junta, para TODOS los tickers y dias donde hay a la vez retorno
    diario real (Norgate) y features intradia reales (EODHD), un pool de
    muestras conjuntas [retorno, realized_vol, open_30m_ret, close_30m_ret,
    hl_range]. Es el dataset de entrenamiento (no condicionado por ticker,
    solo por el valor del retorno) de los 4 generadores: cuantas mas
    muestras, mejor generalizan (de ahi usar el universo AMPLIO de bancos,
    no solo los 25 del predictor).

    Devuelve (pool: array (N, 1+len(feature_cols)), meta: DataFrame con
    columnas ticker/date alineada fila a fila con `pool`, para trazabilidad).
    """
    rows = []
    meta = []
    for tk, feats in intraday_feats_by_ticker.items():
        if tk not in returns_daily.columns or feats.empty:
            continue
        common_idx = returns_daily.index.intersection(feats.index)
        if len(common_idx) == 0:
            continue
        r = returns_daily.loc[common_idx, tk]
        f = feats.loc[common_idx, feature_cols]
        block = np.concatenate([r.values.reshape(-1, 1), f.values], axis=1)
        valid = ~np.isnan(block).any(axis=1)
        rows.append(block[valid])
        meta.append(pd.DataFrame({"ticker": tk, "date": common_idx[valid]}))

    pool = np.concatenate(rows, axis=0) if rows else np.empty((0, 1 + len(feature_cols)))
    meta_df = pd.concat(meta, ignore_index=True) if meta else pd.DataFrame(columns=["ticker", "date"])

    pool, meta_df = _drop_implausible_rows(pool, meta_df, feature_cols)
    return pool, meta_df


# Cortes de cordura por columna (ver config.py): valores por encima de esto
# para una accion cotizando en continuo casi siempre son un fallo de datos
# (ticker en proceso de exclusion/quiebra con cruces erraticos), no
# volatilidad real de mercado.
_SANITY_CAPS = {
    "log_return": ("abs", config.POOL_MAX_ABS_LOG_RETURN),
    "realized_vol": ("abs", config.POOL_MAX_REALIZED_VOL),
    "open_30m_ret": ("abs", config.POOL_MAX_ABS_INTRADAY_RET),
    "close_30m_ret": ("abs", config.POOL_MAX_ABS_INTRADAY_RET),
    "hl_range": ("abs", config.POOL_MAX_HL_RANGE),
}


def _drop_implausible_rows(
    pool: np.ndarray, meta_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(pool) == 0:
        return pool, meta_df
    all_cols = ["log_return", *feature_cols]
    keep = np.ones(len(pool), dtype=bool)
    for j, col in enumerate(all_cols):
        if col not in _SANITY_CAPS:
            continue
        _, cap = _SANITY_CAPS[col]
        keep &= np.abs(pool[:, j]) <= cap
    n_dropped = (~keep).sum()
    if n_dropped:
        import warnings

        warnings.warn(
            f"build_conditional_pool: descartadas {n_dropped}/{len(pool)} filas "
            f"({n_dropped / len(pool):.2%}) por valores fuera de rango plausible "
            f"(tickers probablemente en proceso de quiebra/exclusion con cruces "
            f"erraticos, no volatilidad real de mercado). Ver config.POOL_MAX_*.",
            stacklevel=3,
        )
    return pool[keep], meta_df.loc[keep].reset_index(drop=True)
