"""
Cliente y descargador de barras intradia de 5 minutos de EOD Historical Data
(EODHD) para el universo de bancos.

La API de EODHD solo tiene barras de 5 min desde ~finales de 2020 para estos
tickers (~5 anios), frente a los ~36 anios de historico diario de Norgate.
Ese hueco es precisamente el que se rellena con datos sinteticos en el
notebook 03 (backfill condicional).

Todo lo descargado se cachea en datos/raw/eodhd_5m/<TICKER>.parquet para no
repetir llamadas al API entre ejecuciones. La API key NUNCA se imprime ni se
guarda en el cache (solo se usa en la URL de la request).
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request

import pandas as pd

from . import config

BASE_URL = "https://eodhd.com/api/intraday/{symbol}.US"


def _fetch_chunk(symbol: str, api_key: str, dt_from: dt.datetime, dt_to: dt.datetime) -> list[dict]:
    params = (
        f"interval=5m&api_token={api_key}&fmt=json"
        f"&from={int(dt_from.timestamp())}&to={int(dt_to.timestamp())}"
    )
    url = f"{BASE_URL.format(symbol=symbol)}?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def download_ticker_5m(
    symbol: str,
    api_key: str | None = None,
    start: str = config.INTRADAY_DOWNLOAD_START_DATE,
    end: str = config.INTRADAY_DOWNLOAD_END_DATE,
    chunk_days: int = config.EODHD_MAX_SPAN_DAYS,
    sleep_s: float = 0.15,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Descarga todas las barras de 5 min disponibles para `symbol` entre
    `start` y `end`, troceando en ventanas de `chunk_days` (limite del API).
    Devuelve un DataFrame indexado por datetime UTC con columnas
    open/high/low/close/volume."""
    api_key = api_key or config.load_eodhd_api_key()
    d_start = dt.datetime.strptime(start, "%Y-%m-%d")
    d_end = dt.datetime.strptime(end, "%Y-%m-%d")

    frames = []
    cursor = d_start
    while cursor < d_end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days), d_end)
        rows = None
        for attempt in range(max_retries):
            try:
                rows = _fetch_chunk(symbol, api_key, cursor, chunk_end)
                break
            except urllib.error.HTTPError as e:
                if e.code == 422:
                    rows = []
                    break
                time.sleep(1.0 * (attempt + 1))
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        if rows:
            frames.append(pd.DataFrame(rows))
        time.sleep(sleep_s)
        cursor = chunk_end

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop_duplicates(subset="datetime").sort_values("datetime")
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    return df


def get_ticker_5m_cached(
    symbol: str,
    api_key: str | None = None,
    force_refresh: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Igual que `download_ticker_5m` pero cachea el resultado en
    datos/raw/eodhd_5m/<symbol>.parquet."""
    cache_dir = config.RAW_DIR / "eodhd_5m"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    df = download_ticker_5m(symbol, api_key=api_key, **kwargs)
    df.to_parquet(cache_path)
    return df


def download_universe_5m(
    tickers: list[str] | None = None,
    api_key: str | None = None,
    force_refresh: bool = False,
    verbose: bool = True,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Descarga (o lee de cache) las barras de 5 min de todo el universo.
    Devuelve un dict {ticker: DataFrame}."""
    tickers = tickers or config.GENERATOR_TICKERS
    api_key = api_key or config.load_eodhd_api_key()
    out = {}
    for tk in tickers:
        df = get_ticker_5m_cached(tk, api_key=api_key, force_refresh=force_refresh, **kwargs)
        if verbose:
            span = f"{df.index.min()} -> {df.index.max()}" if len(df) else "SIN DATOS"
            print(f"{tk:6s} {len(df):7d} barras  [{span}]")
        out[tk] = df
    return out
