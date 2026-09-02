# %% [markdown]
# # 00 · Descarga y consolidación de datos
#
# Construye los dos datasets base del proyecto:
#
# - **Retornos diarios reales** (hasta ~36 años, Norgate) para el universo
#   reducido (25 bancos, backbone del predictor final) y el universo amplio
#   (150 bancos, usado solo para entrenar los generadores con más datos).
# - **Barras de 5 minutos reales** (EODHD, desde 2020-11 — todo lo que sirve
#   el API, ~5,5 años) y sus features
#   intradía derivadas (volatilidad realizada, retorno de apertura/cierre,
#   rango), para el universo amplio.
#
# Todo se cachea en `datos/raw/` y `datos/interim/` (gitignored, pesan
# demasiado y contienen la ruta al `APkey`); este notebook solo hay que
# relanzarlo si cambia el universo, las fechas, o se borra la caché.
#
# **Por qué dos universos distintos** (`src/config.py`): el predictor final
# necesita ~30 años de retorno diario REAL por banco (solo 25 bancos los
# tienen completos en el dump de Norgate). Los generadores, en cambio, solo
# necesitan datos de la ventana real (2020-11 en adelante) — así que para
# darles más muestras
# con las que aprender bien la distribución conjunta (retorno, features
# intradía) se usan hasta 150 bancos, aunque no coticen desde 1990.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

# Recarga automatica de src/ al editarlo: sin esto, si se edita un modulo
# de src/ con el kernel ya arrancado, Jupyter sigue usando la version que
# importo la primera vez (y aparecen errores tipo "unexpected keyword
# argument" con codigo que en disco si es correcto).
try:
    ip = get_ipython()
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
except NameError:
    pass  # ejecutandose fuera de IPython/Jupyter

import numpy as np
import pandas as pd

from src import config, data_eodhd as de, data_norgate as dn, features as feat

# %% [markdown]
# ## 1. Retornos diarios reales (Norgate)
#
# Universo reducido (25 bancos, ~36 años completos).

# %%
prices_predictor = dn.load_daily_prices(config.PREDICTOR_TICKERS)
returns_predictor = dn.compute_log_returns(prices_predictor)
print("precios:", prices_predictor.shape, " retornos:", returns_predictor.shape)

cobertura = dn.coverage_report(prices_predictor)
cobertura

# %%
assert cobertura["pct_nan"].max() < 0.05, "Algún ticker del universo predictor tiene demasiados huecos"
cobertura.to_csv(config.TABLES_DIR / "00_cobertura_predictor.csv")

# %% [markdown]
# Universo amplio (150 bancos), retorno diario real solo desde el inicio de
# la ventana real (2020-11 en adelante, es lo único que necesitan los
# generadores).

# %%
prices_generator = dn.load_daily_prices(
    config.GENERATOR_TICKERS, start=config.REAL_INTRADAY_START_DATE
)
# dropna=None: NO exigimos que los 150 bancos coticen el mismo dia (varios
# no tienen historia completa a proposito, ver seccion introductoria). Cada
# ticker conserva sus propios NaN; build_conditional_pool los filtra 1 a 1.
returns_generator = dn.compute_log_returns(prices_generator, dropna=None)
print("precios:", prices_generator.shape, " retornos:", returns_generator.shape)
print(f"NaNs: {returns_generator.isna().mean().mean():.1%} de media por ticker (normal: altas/bajas parciales)")

# %% [markdown]
# ## 2. Barras de 5 minutos reales (EODHD)
#
# Se descargan (o se leen de caché) para el universo amplio. La API key se
# lee de `datos/APkey` (gitignored) y nunca se imprime.

# %%
bars_by_ticker = de.download_universe_5m(tickers=config.GENERATOR_TICKERS)

# %%
n_ok = sum(1 for df in bars_by_ticker.values() if len(df) > 0)
print(f"{n_ok}/{len(bars_by_ticker)} tickers con barras de 5 min descargadas")

cobertura_intradia = pd.DataFrame(
    {
        "ticker": list(bars_by_ticker.keys()),
        "n_bars": [len(df) for df in bars_by_ticker.values()],
        "first": [df.index.min() if len(df) else pd.NaT for df in bars_by_ticker.values()],
        "last": [df.index.max() if len(df) else pd.NaT for df in bars_by_ticker.values()],
    }
).set_index("ticker")
cobertura_intradia.to_csv(config.TABLES_DIR / "00_cobertura_intradia.csv")
cobertura_intradia.sort_values("n_bars").head(10)

# %% [markdown]
# ## 3. Features intradía diarias (volatilidad realizada, etc.)
#
# Primero se descarta, DÍA A DÍA, cualquier sesión con menos de
# `MIN_BARS_PER_SESSION` barras (feed caído, apertura tardía — no cierres
# anticipados legítimos por festivo, esos sí se quedan). Después, un ticker
# entra en el pool de entrenamiento de los generadores solo si le quedan al
# menos `MIN_SESSIONS_FOR_GENERATOR_POOL` sesiones válidas (filtra bancos
# intervenidos/fusionados a mitad de la ventana real).

# %%
intraday_feats_all = {
    tk: feat.daily_intraday_features(bars) for tk, bars in bars_by_ticker.items()
}
intraday_feats_all = {
    tk: f[f["n_bars"] >= config.MIN_BARS_PER_SESSION] for tk, f in intraday_feats_all.items()
}
intraday_feats = {
    tk: f for tk, f in intraday_feats_all.items()
    if len(f) >= config.MIN_SESSIONS_FOR_GENERATOR_POOL
}
print(
    f"{len(intraday_feats)}/{len(intraday_feats_all)} tickers con "
    f">= {config.MIN_SESSIONS_FOR_GENERATOR_POOL} sesiones intradía válidas"
)

# %% [markdown]
# ## 4. Pool condicional (retorno diario real, features intradía reales)
#
# Dataset de entrenamiento de los 4 generadores del notebook 02: cada fila
# es un día real de un banco cualquiera del universo amplio, con su retorno
# diario y sus 4 features intradía reales. Cuantas más muestras, mejor
# generalizan los generadores — de ahí usar el universo amplio.

# %%
pool, pool_meta = feat.build_conditional_pool(returns_generator, intraday_feats)
print("pool:", pool.shape, "  columnas:", ["log_return", *feat.INTRADAY_FEATURE_COLS])
pool_meta["ticker"].value_counts().describe()

# %%
assert len(pool) > 5000, "Pool de entrenamiento de los generadores demasiado pequeño"

# %% [markdown]
# ## 5. Guardar datasets intermedios (`datos/interim/`, gitignored)

# %%
np.save(config.INTERIM_DIR / "conditional_pool.npy", pool)
pool_meta.to_parquet(config.INTERIM_DIR / "conditional_pool_meta.parquet")
returns_predictor.to_parquet(config.INTERIM_DIR / "returns_predictor.parquet")
returns_generator.to_parquet(config.INTERIM_DIR / "returns_generator.parquet")

intraday_long = (
    pd.concat([f.assign(ticker=tk) for tk, f in intraday_feats.items()])
    .rename_axis("date")
    .reset_index()
)
intraday_long.to_parquet(config.INTERIM_DIR / "intraday_features_real.parquet")

print("Guardado en", config.INTERIM_DIR)
