"""
Carga de precios diarios reales (hasta 36 anios) desde el dump DuckDB de
Norgate para el universo de bancos definido en src/config.py.

Esta es la fuente de la serie "larga" (backbone real de hasta 30 anios) que
se usa como target para el backfill sintetico y como base de las ventanas
X/Y de la tarea supervisada.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from . import config


def load_daily_prices(
    tickers: list[str] | None = None,
    start: str = config.DAILY_START_DATE,
    end: str = config.DAILY_END_DATE,
    duckdb_path=config.NORGATE_DUCKDB,
) -> pd.DataFrame:
    """Devuelve un DataFrame (date x ticker) de precios de cierre ajustados
    a total return (dividendos reinvertidos), tal y como se usa en los
    notebooks de clase (`precios_close`)."""
    tickers = tickers or config.PREDICTOR_TICKERS
    con = duckdb.connect(str(duckdb_path), read_only=True)
    placeholders = ",".join(["?"] * len(tickers))
    query = f"""
        select date, symbol, close_totalreturn
        from export.bank_prices_daily
        where symbol in ({placeholders})
          and date between ? and ?
        order by date
    """
    df = con.execute(query, tickers + [start, end]).df()
    con.close()

    prices = df.pivot(index="date", columns="symbol", values="close_totalreturn")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    # Reordenamos columnas para que coincidan siempre con config.PREDICTOR_TICKERS
    prices = prices.reindex(columns=tickers)
    return prices


def load_daily_ohlcv(
    tickers: list[str] | None = None,
    start: str = config.DAILY_START_DATE,
    end: str = config.DAILY_END_DATE,
    duckdb_path=config.NORGATE_DUCKDB,
) -> pd.DataFrame:
    """Devuelve OHLCV diario (unadjusted + total return close) en formato
    largo (tidy), util para comparar con los precios reconstruidos a partir
    de las barras intradia."""
    tickers = tickers or config.PREDICTOR_TICKERS
    con = duckdb.connect(str(duckdb_path), read_only=True)
    placeholders = ",".join(["?"] * len(tickers))
    query = f"""
        select symbol, date,
               open_unadjusted as open, high_unadjusted as high,
               low_unadjusted as low, close_unadjusted as close,
               close_totalreturn, volume
        from export.bank_prices_daily
        where symbol in ({placeholders})
          and date between ? and ?
        order by symbol, date
    """
    df = con.execute(query, tickers + [start, end]).df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_log_returns(prices: pd.DataFrame, dropna: str = "any") -> pd.DataFrame:
    """log-retornos diarios, igual que `np.log(precios_close).diff()` en
    los notebooks de clase.

    `dropna="any"` (por defecto, universo REDUCIDO/predictor): quita
    cualquier fila con un NaN en cualquier ticker, para tener una matriz
    densa y alineada — razonable con 25 tickers todos con historia
    completa.

    `dropna=None` (universo AMPLIO/generador, ~150 tickers, historia
    parcial a proposito): NO se descarta nada por fila; cada ticker
    conserva sus propios NaNs (dias sin cotizar). Los consumidores (ej.
    `features.build_conditional_pool`) hacen la interseccion valida
    ticker a ticker, en vez de exigir que los 150 coticen el mismo dia.
    """
    returns = np.log(prices).diff()
    if dropna == "any":
        returns = returns.dropna(how="any")
    return returns


def coverage_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Pequena tabla de cobertura por ticker (primer/ultimo dato, % de NaNs)
    util para el EDA y para el README."""
    rows = []
    for col in prices.columns:
        s = prices[col]
        valid = s.dropna()
        rows.append(
            {
                "ticker": col,
                "first_date": valid.index.min() if len(valid) else pd.NaT,
                "last_date": valid.index.max() if len(valid) else pd.NaT,
                "n_obs": len(valid),
                "pct_nan": 1 - len(valid) / len(s),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
