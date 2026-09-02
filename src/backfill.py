"""
Backfill condicional: dado el retorno diario REAL conocido de un dia
historico (Norgate, hasta 30 anios), genera un vector de features intradia
SINTETICO consistente con ese retorno, a partir de un pool de muestras
conjuntas [retorno, features] generado por uno de los 4 generadores
(src/generators.py) entrenados sobre la ventana real (2020-11 en adelante).

*** "Backfill" aqui NO es pandas `.bfill()` *** (que propaga hacia atras
el SIGUIENTE valor conocido, usando datos del futuro para rellenar el
pasado — eso si seria sospechoso). Aqui, para cada dia sin 5 min reales,
se usa el retorno REAL Y CONTEMPORANEO de ESE MISMO dia (nunca del
futuro, nunca el target) para consultar que feature intradia es plausible
segun la relacion aprendida en la ventana real reciente. Ver README,
seccion "Logica financiera", para el argumento completo de por que no hay
fuga de informacion y por que un `.ffill()`/`.bfill()` literal seria peor
(dejaria una volatilidad constante ~24 anios, ciega a todas las crisis).

Los generadores de src/generators.py son todos INCONDICIONALES: aprenden la
distribucion conjunta (retorno, features) y solo saben muestrear pares
nuevos de esa conjunta, no "features | retorno=x" directamente. Para
condicionar por el retorno YA CONOCIDO de cada dia historico se usa
"conditional matching": de un pool grande de muestras conjuntas sinteticas,
para cada retorno real se buscan las muestras cuyo retorno sintetico esta
mas cerca (kernel gaussiano sobre la distancia en retorno) y se toma UNA de
ellas al azar con esas probabilidades (muestreo ponderado, no la media: asi
el backfill sigue siendo una muestra generativa con la variabilidad real de
los datos, no una curva suavizada).

Este mismo mecanismo se aplica igual a los 4 generadores (incluido Ruido),
para que la comparacion entre ellos en el notebook 04 refleje solo la
calidad de cada generador, no un truco de condicionamiento distinto por
metodo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def conditional_match_sample(
    pool: np.ndarray,
    query_returns: np.ndarray,
    k_neighbors: int = 50,
    bandwidth: float | None = None,
    random_state: int = config.RANDOM_SEED,
) -> np.ndarray:
    """Para cada valor de `query_returns` (M,), busca en `pool` (N, 1+F)
    (columna 0 = retorno sintetico, columnas 1: = features) los
    `k_neighbors` mas cercanos en retorno y devuelve UNA muestra de sus
    features, elegida al azar con pesos de un kernel gaussiano sobre la
    distancia en retorno (bandwidth = percentil 25 de las distancias al
    vecino mas lejano si no se especifica). Devuelve (M, F)."""
    rng = np.random.default_rng(random_state)
    n_pool = len(pool)
    k = min(k_neighbors, n_pool)

    order = np.argsort(pool[:, 0])
    sorted_returns = pool[order, 0]
    sorted_features = pool[order, 1:]

    center_idx = np.searchsorted(sorted_returns, query_returns)
    center_idx = np.clip(center_idx, 0, n_pool - 1)

    half = k // 2
    lo = np.clip(center_idx - half, 0, max(n_pool - k, 0))

    out = np.empty((len(query_returns), sorted_features.shape[1]))
    for i, (q, start) in enumerate(zip(query_returns, lo)):
        idx_window = np.arange(start, start + k)
        dist = np.abs(sorted_returns[idx_window] - q)
        bw = bandwidth if bandwidth is not None else (np.median(dist) + 1e-8)
        weights = np.exp(-0.5 * (dist / bw) ** 2)
        weights_sum = weights.sum()
        if weights_sum <= 0 or not np.isfinite(weights_sum):
            weights = np.ones_like(weights)
            weights_sum = weights.sum()
        probs = weights / weights_sum
        chosen = rng.choice(idx_window, p=probs)
        out[i] = sorted_features[chosen]

    return out


def backfill_ticker_features(
    real_returns: pd.Series,
    synth_pool: np.ndarray,
    k_neighbors: int = 50,
    random_state: int = config.RANDOM_SEED,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Backfill de features intradia sinteticas para TODAS las fechas de
    `real_returns` (retorno diario real de un ticker, ej. los ~24 anios sin
    5 min reales), usando `synth_pool` (salida de un generador.sample(N))."""
    from .features import INTRADAY_FEATURE_COLS

    cols = feature_cols or INTRADAY_FEATURE_COLS
    values = conditional_match_sample(
        synth_pool, real_returns.values, k_neighbors=k_neighbors, random_state=random_state
    )
    return pd.DataFrame(values, index=real_returns.index, columns=cols)


def build_full_history_features(
    returns_by_ticker: pd.DataFrame,
    real_intraday_feats: dict[str, pd.DataFrame],
    synth_pool: np.ndarray,
    real_start: str = config.REAL_INTRADAY_START_DATE,
    k_neighbors: int = 50,
    random_state: int = config.RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Construye, para cada ticker de `returns_by_ticker`, el panel completo
    de 30 anios de features intradia: REALES a partir de `real_start`
    (2020-11-02: primera barra de 5 min que sirve EODHD)
    (usa `real_intraday_feats[ticker]`, calculado de barras de 5 min
    reales) y SINTETICAS (via `synth_pool`, la salida de un generador) antes
    de esa fecha. Devuelve {ticker: DataFrame con columna extra 'is_synthetic'}."""
    from .features import INTRADAY_FEATURE_COLS

    out = {}
    real_start_ts = pd.Timestamp(real_start)
    for i, tk in enumerate(returns_by_ticker.columns):
        r = returns_by_ticker[tk].dropna()
        r_old = r[r.index < real_start_ts]
        r_new = r[r.index >= real_start_ts]

        # Semilla distinta por ticker (derivada de random_state) para que el
        # muestreo ponderado no repita la misma secuencia de numeros
        # aleatorios en los 25 bancos.
        synth_feats = backfill_ticker_features(
            r_old, synth_pool, k_neighbors=k_neighbors, random_state=random_state + i
        )
        synth_feats["is_synthetic"] = True

        real_feats = real_intraday_feats.get(tk, pd.DataFrame(columns=INTRADAY_FEATURE_COLS))
        real_feats = real_feats.reindex(r_new.index)[INTRADAY_FEATURE_COLS]
        real_feats["is_synthetic"] = False

        full = pd.concat([synth_feats, real_feats]).sort_index()
        out[tk] = full
    return out


def rv_panel_from_full_history(
    full_history_by_ticker: dict[str, pd.DataFrame], feature_col: str = "realized_vol"
) -> pd.DataFrame:
    """Convierte la salida de `build_full_history_features` (un DataFrame
    largo por ticker) en un panel ancho (date x ticker) de una sola feature
    (por defecto `realized_vol`), listo para pasar a
    `features.build_xy_windows` junto con los retornos diarios."""
    return pd.DataFrame({tk: df[feature_col] for tk, df in full_history_by_ticker.items()}).sort_index()
