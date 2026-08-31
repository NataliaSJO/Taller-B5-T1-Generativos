"""
v2 — Backfill que PRESERVA LA PERSISTENCIA de la volatilidad.

Diagnostico que motiva esta version
-----------------------------------
En v1, `src/backfill.py` muestrea la volatilidad sintetica de cada dia de
forma INDEPENDIENTE, condicionando solo al retorno de ese mismo dia. El
resultado, medido sobre la serie sintetica de JPM:

    persistencia (lag 1)      REAL (5 min)    SINTETICA v1
    Pearson                      +0.631          +0.080
    Spearman (rangos)            +0.636          +0.051
    Pearson sobre log(vol)       +0.656          +0.070
    Informacion mutua            +0.304          +0.000

O sea: la serie sintetica de v1 NO tiene clustering de volatilidad, el
hecho estilizado mas robusto de las series financieras. Da igual lo bien
que el generador modele la distribucion conjunta de UN dia: al muestrear
cada dia por separado, el backfill destruye la dinamica temporal.

Eso explica el resultado negativo de v1 (§6.3 del README): la calidad
distribucional del generador no predecia la utilidad aguas abajo, porque
lo que aportaria valor predictivo — la dinamica — se perdia en el backfill,
fuera cual fuera el generador.

Que cambia en v2
----------------
El muestreo pasa a ser SECUENCIAL y condicionado tambien al pasado:

    v1:  RV_t  ~  p( . | retorno_t )                 (dias independientes)
    v2:  RV_t  ~  p( . | retorno_t , RV_{t-1} )      (cadena con memoria)

Para eso el pool de entrenamiento incluye la volatilidad RETARDADA como
una variable mas, de modo que el generador aprende la distribucion
conjunta (retorno_t, RV_{t-1}, RV_t, ...) y el emparejamiento condicional
puede usar dos variables en vez de una.

Esto NO toca nada de v1: importa de `src/` sin modificarlo, y escribe sus
resultados con prefijo `v2_`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, features as feat  # noqa: E402

# columna 0 = retorno del dia, columna 1 = volatilidad del dia ANTERIOR,
# resto = features intradia del dia actual
COLS_V2 = ["log_return", "rv_lag1", *feat.INTRADAY_FEATURE_COLS]


def construir_pool_con_retardo(
    returns_daily: pd.DataFrame,
    intraday_feats_by_ticker: dict[str, pd.DataFrame],
    feature_cols: list[str] = feat.INTRADAY_FEATURE_COLS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Como `features.build_conditional_pool` pero anadiendo la volatilidad
    realizada del dia ANTERIOR como variable de condicionamiento.

    Cada fila es: [retorno_t, RV_{t-1}, RV_t, open30_t, close30_t, hl_t].
    Solo se conservan dias en los que el dia previo es realmente el dia de
    mercado anterior (no se cruza un hueco), para que `rv_lag1` signifique
    de verdad "ayer"."""
    filas, meta = [], []
    for tk, f in intraday_feats_by_ticker.items():
        if tk not in returns_daily.columns or f.empty:
            continue
        f = f.sort_index()
        comun = returns_daily.index.intersection(f.index)
        if len(comun) < 30:
            continue
        r = returns_daily.loc[comun, tk]
        ff = f.loc[comun, feature_cols]
        rv_lag = ff["realized_vol"].shift(1)

        bloque = np.column_stack([r.values, rv_lag.values, ff.values])
        ok = ~np.isnan(bloque).any(axis=1)
        filas.append(bloque[ok])
        meta.append(pd.DataFrame({"ticker": tk, "date": comun[ok]}))

    pool = np.concatenate(filas) if filas else np.empty((0, len(COLS_V2)))
    meta_df = pd.concat(meta, ignore_index=True) if meta else pd.DataFrame(columns=["ticker", "date"])
    return pool, meta_df


def _muestreo_condicional_2d(pool: np.ndarray, retorno: float, rv_previa: float,
                             rng: np.random.Generator, k_vecinos: int = 60,
                             peso_rv: float = 1.0) -> np.ndarray:
    """Elige una fila del pool cercana en (retorno, RV previa) y devuelve
    sus features del dia actual.

    Las dos variables de condicionamiento viven en escalas distintas, asi
    que se estandarizan antes de medir distancia; `peso_rv` permite dar mas
    o menos importancia a la persistencia frente al retorno."""
    ret_pool, rvlag_pool = pool[:, 0], pool[:, 1]
    s_ret = ret_pool.std() + 1e-12
    s_rv = rvlag_pool.std() + 1e-12
    d2 = ((ret_pool - retorno) / s_ret) ** 2 + peso_rv * ((rvlag_pool - rv_previa) / s_rv) ** 2

    k = min(k_vecinos, len(pool))
    vecinos = np.argpartition(d2, k - 1)[:k]
    d = np.sqrt(d2[vecinos])
    bw = np.median(d) + 1e-9
    w = np.exp(-0.5 * (d / bw) ** 2)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        w = np.ones_like(w)
        total = w.sum()
    elegido = rng.choice(vecinos, p=w / total)
    return pool[elegido, 2:]


def backfill_persistente(real_returns: pd.Series, pool_sintetico: np.ndarray,
                         rv_inicial: float | None = None, k_vecinos: int = 60,
                         peso_rv: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """Genera la serie de features intradia dia a dia, EN ORDEN, arrastrando
    la volatilidad generada del dia anterior como condicionante del
    siguiente. Ese arrastre es lo que crea el clustering."""
    rng = np.random.default_rng(random_state)
    fechas = real_returns.index
    n_feat = pool_sintetico.shape[1] - 2
    salida = np.empty((len(fechas), n_feat))

    # arranque: si no se da, se usa la volatilidad mediana del pool
    rv_prev = float(np.median(pool_sintetico[:, 1])) if rv_inicial is None else float(rv_inicial)

    for i, ret in enumerate(real_returns.values):
        fila = _muestreo_condicional_2d(pool_sintetico, float(ret), rv_prev,
                                        rng, k_vecinos, peso_rv)
        salida[i] = fila
        rv_prev = float(fila[0])  # la RV generada hoy condiciona la de manana

    return pd.DataFrame(salida, index=fechas, columns=feat.INTRADAY_FEATURE_COLS)


def construir_historico_v2(returns_by_ticker: pd.DataFrame,
                           real_intraday_feats: dict[str, pd.DataFrame],
                           pool_sintetico: np.ndarray,
                           real_start: str = config.REAL_INTRADAY_START_DATE,
                           k_vecinos: int = 60, peso_rv: float = 1.0,
                           random_state: int = 42) -> dict[str, pd.DataFrame]:
    """Equivalente a `backfill.build_full_history_features` pero usando el
    muestreo secuencial con memoria."""
    out = {}
    corte = pd.Timestamp(real_start)
    for i, tk in enumerate(returns_by_ticker.columns):
        r = returns_by_ticker[tk].dropna()
        r_old, r_new = r[r.index < corte], r[r.index >= corte]

        sint = backfill_persistente(r_old, pool_sintetico, k_vecinos=k_vecinos,
                                    peso_rv=peso_rv, random_state=random_state + i)
        sint["is_synthetic"] = True

        reales = real_intraday_feats.get(tk, pd.DataFrame(columns=feat.INTRADAY_FEATURE_COLS))
        reales = reales.reindex(r_new.index)[feat.INTRADAY_FEATURE_COLS]
        reales["is_synthetic"] = False

        out[tk] = pd.concat([sint, reales]).sort_index()
    return out


def medir_persistencia(serie: pd.Series) -> dict:
    """Persistencia con 4 medidas, incluida una no lineal (informacion
    mutua), para no depender de que la dependencia sea lineal."""
    from scipy.stats import spearmanr
    from sklearn.feature_selection import mutual_info_regression

    s = serie.dropna()
    if len(s) < 50:
        return {}
    x, y = s.values[:-1], s.values[1:]
    return {
        "pearson": float(pd.Series(x).corr(pd.Series(y))),
        "spearman": float(spearmanr(x, y).statistic),
        "pearson_log": float(pd.Series(np.log(x + 1e-12)).corr(pd.Series(np.log(y + 1e-12)))),
        "info_mutua": float(mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0]),
    }
