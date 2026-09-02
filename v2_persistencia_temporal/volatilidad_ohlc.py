"""
v2 — Volatilidad REAL de 30 anios a partir del OHLC diario.

El proyecto v1 asume que la volatilidad intradia solo se puede conocer con
barras de 5 minutos, y por eso sintetiza los ~24 anios que no las tienen.
Pero el dump de Norgate trae maximo y minimo DIARIOS reales de los 30 anios
completos, y con eso se puede estimar la volatilidad con estimadores
clasicos de rango:

  - Parkinson (1980), usa high-low:
        sigma^2 = (1 / (4 ln2)) * ln(H/L)^2
    Mucho mas eficiente que la volatilidad close-to-close (usa el recorrido
    del dia, no solo el cierre), pero ignora los saltos overnight.

  - Garman-Klass (1980), usa open-high-low-close:
        sigma^2 = 0.5*ln(H/L)^2 - (2 ln2 - 1)*ln(C/O)^2
    Aun mas eficiente que Parkinson al incorporar apertura y cierre.

Esto sirve para dos cosas, ambas importantes:

  1. VALIDAR el backfill sintetico. En v1 no habia forma de comprobar si la
     volatilidad sintetica de 1998 se parecia a la real, porque no habia
     volatilidad real de 1998. Con Parkinson/Garman-Klass si la hay, asi
     que por primera vez se puede medir el error del backfill en el propio
     periodo historico en vez de solo confiar en el empalme.

  2. Cuestionar la premisa del proyecto. Si un estimador de rango recupera
     buena parte de la informacion de la volatilidad realizada intradia,
     entonces sintetizarla aporta menos de lo que parecia. Es una
     comprobacion honesta que v1 nunca hizo.

Este modulo NO modifica nada de v1: solo lee de `src/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, data_norgate as dn  # noqa: E402

_LN2 = np.log(2.0)


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Volatilidad diaria de Parkinson a partir del rango high-low."""
    hl = np.log(high / low)
    return np.sqrt(hl**2 / (4.0 * _LN2))


def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series,
                 close: pd.Series) -> pd.Series:
    """Volatilidad diaria de Garman-Klass (usa OHLC completo)."""
    hl = np.log(high / low)
    co = np.log(close / open_)
    var = 0.5 * hl**2 - (2.0 * _LN2 - 1.0) * co**2
    return np.sqrt(np.clip(var, 0.0, None))


def panel_volatilidad_ohlc(tickers: list[str] | None = None,
                           start: str = config.TOTAL_HISTORY_START_DATE,
                           end: str = config.DAILY_END_DATE) -> dict[str, pd.DataFrame]:
    """Devuelve {ticker: DataFrame con columnas parkinson y garman_klass}
    para todo el historico disponible (30 anios), calculado con datos
    REALES de Norgate."""
    tickers = tickers or config.PREDICTOR_TICKERS
    ohlcv = dn.load_daily_ohlcv(tickers, start=start, end=end)
    out = {}
    for tk, g in ohlcv.groupby("symbol"):
        g = g.set_index("date").sort_index()
        valido = (g[["open", "high", "low", "close"]] > 0).all(axis=1)
        g = g[valido]
        out[tk] = pd.DataFrame(
            {
                "parkinson": parkinson(g["high"], g["low"]),
                "garman_klass": garman_klass(g["open"], g["high"], g["low"], g["close"]),
            },
            index=g.index,
        )
    return out


def comparar_con_realizada(tickers: list[str] | None = None) -> pd.DataFrame:
    """En la ventana donde SI hay barras de 5 min, compara los estimadores
    de rango con la volatilidad realizada intradia. Responde a: ¿cuanta
    informacion de la volatilidad intradia se recupera solo con el OHLC
    diario, sin datos de alta frecuencia?"""
    from scipy.stats import spearmanr

    tickers = tickers or config.PREDICTOR_TICKERS
    rangos = panel_volatilidad_ohlc(tickers)
    intr = pd.read_parquet(config.INTERIM_DIR / "intraday_features_real.parquet")

    filas = []
    for tk in tickers:
        sub = intr[intr.ticker == tk]
        if sub.empty or tk not in rangos:
            continue
        rv = sub.set_index("date")["realized_vol"].sort_index()
        r = rangos[tk]
        idx = rv.index.intersection(r.index)
        if len(idx) < 50:
            continue
        filas.append({
            "ticker": tk,
            "n": len(idx),
            "pearson_parkinson": float(np.corrcoef(rv.loc[idx], r.loc[idx, "parkinson"])[0, 1]),
            "spearman_parkinson": float(spearmanr(rv.loc[idx], r.loc[idx, "parkinson"]).statistic),
            "pearson_gk": float(np.corrcoef(rv.loc[idx], r.loc[idx, "garman_klass"])[0, 1]),
            "spearman_gk": float(spearmanr(rv.loc[idx], r.loc[idx, "garman_klass"]).statistic),
            "ratio_nivel_parkinson": float(r.loc[idx, "parkinson"].mean() / rv.loc[idx].mean()),
        })
    return pd.DataFrame(filas).set_index("ticker")


if __name__ == "__main__":
    print("Volatilidad de rango (OHLC diario real) vs volatilidad realizada (5 min)")
    print("Ventana: donde existen ambas (2020-11 en adelante, ~5.5 anios)\n")
    df = comparar_con_realizada()
    print(df.round(3).to_string())
    print("\nMEDIAS:")
    print(df.mean(numeric_only=True).round(3).to_string())
    out = config.TABLES_DIR / "v2_volatilidad_ohlc_vs_realizada.csv"
    df.to_csv(out)
    print(f"\nGuardado -> {out}")
