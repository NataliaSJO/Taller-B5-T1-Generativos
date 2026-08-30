# %% [markdown]
# # 03 · Backfill condicional y construcción del dataset final
#
# Con los 4 pools sintéticos del notebook 02 (muestras conjuntas
# `[retorno, features intradía]` sin condicionar), este notebook rellena
# los **28 años sin barras de 5 min reales** de los 25 bancos del predictor:
# para cada día histórico se conoce el retorno diario REAL (Norgate) y se
# le empareja, por "conditional matching" (vecino más cercano ponderado por
# un kernel gaussiano sobre la distancia en retorno — ver
# `src/backfill.py`), una muestra de features intradía del pool sintético
# correspondiente.
#
# El resultado son 4 datasets completos de 30 años (2 reales + 28
# sintéticos), uno por generador, listos para la rejilla de entrenamiento
# del notebook 04.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import backfill as bf, config, data_norgate as dn, features as feat, plotting as pl

# %% [markdown]
# ## 1. Cargar retornos reales (25 bancos, 30 años) y features intradía
#    reales (últimos ~2 años)

# %%
returns_predictor = pd.read_parquet(config.INTERIM_DIR / "returns_predictor.parquet")
returns_predictor = returns_predictor[returns_predictor.index >= pd.Timestamp(config.TOTAL_HISTORY_START_DATE)]
print("retornos reales:", returns_predictor.shape, returns_predictor.index.min(), "->", returns_predictor.index.max())

intraday_long = pd.read_parquet(config.INTERIM_DIR / "intraday_features_real.parquet")
real_intraday_feats = {
    tk: g.set_index("date")[feat.INTRADAY_FEATURE_COLS]
    for tk, g in intraday_long[intraday_long["ticker"].isin(config.PREDICTOR_TICKERS)].groupby("ticker")
}
print(f"features intradia reales disponibles para {len(real_intraday_feats)}/{config.N_PREDICTOR_TICKERS} bancos del predictor")

# %% [markdown]
# ## 2. Cargar los 4 pools sintéticos del notebook 02

# %%
synthetic_pools = {}
for name in ["noise", "gaussian", "rbig", "gan"]:
    path = config.INTERIM_DIR / f"synthetic_pool_{name}.npy"
    if path.exists():
        synthetic_pools[name] = np.load(path)
print("generadores disponibles:", list(synthetic_pools.keys()))
assert len(synthetic_pools) >= 3, "Faltan pools sinteticos del notebook 02 (ejecutalo primero)"

# %% [markdown]
# ## 3. Backfill condicional: un panel de volatilidad realizada de 30 años
#    por generador

# %%
full_history_by_generator = {}
rv_panel_by_generator = {}
for name, pool in synthetic_pools.items():
    full_hist = bf.build_full_history_features(
        returns_predictor, real_intraday_feats, pool,
        real_start=config.REAL_INTRADAY_START_DATE, k_neighbors=80, random_state=42,
    )
    full_history_by_generator[name] = full_hist
    rv_panel_by_generator[name] = bf.rv_panel_from_full_history(full_hist)
    n_synth_days = int(full_hist[config.PREDICTOR_TICKERS[0]]["is_synthetic"].sum())
    print(f"{name}: panel {rv_panel_by_generator[name].shape}, "
          f"{n_synth_days} dias sinteticos por ticker (de {len(full_hist[config.PREDICTOR_TICKERS[0]])})")

# %% [markdown]
# ## 4. Validación visual: la serie de volatilidad, 30 años, real + sintética
#
# Un banco de ejemplo (JPM): tramo sintético (28 años, gris) + tramo real
# (2 años, color) para 3 generadores. Lo importante no es que la parte
# sintética "acierte" día a día (es imposible sin datos reales), sino que
# el NIVEL y la VARIABILIDAD sean coherentes con el tramo real — sin saltos
# artificiales en el empalme.

# %%
demo_ticker = "JPM"
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for ax, name in zip(axes, ["noise", "gaussian", "rbig"]):
    panel = full_history_by_generator[name][demo_ticker]
    is_synth = panel["is_synthetic"]
    ax.plot(panel.index[is_synth], panel.loc[is_synth, "realized_vol"],
            color="#b7b6ae", linewidth=0.8, label="sintético (28 años)")
    ax.plot(panel.index[~is_synth], panel.loc[~is_synth, "realized_vol"],
            color=pl.color_for(name), linewidth=0.8, label="real (2 años)")
    ax.axvline(pd.Timestamp(config.REAL_INTRADAY_START_DATE), color="#52514e", linewidth=0.8, linestyle=":")
    ax.set_title(f"{demo_ticker} — volatilidad realizada diaria — generador: {name}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    pl.style_axes(ax)
axes[-1].set_xlabel("fecha")
fig.tight_layout()
pl.savefig(fig, "03_backfill_serie_temporal_JPM")
fig

# %% [markdown]
# ## 5. Comprobación de continuidad en el empalme (2 años reales / 28
#    sintéticos)
#
# Para cada generador y cada banco, ratio entre el nivel medio de
# volatilidad sintética (últimos 2 años de la parte sintética, justo antes
# del empalme) y el nivel medio real (primeros 2 años de la parte real,
# justo después). Un ratio cercano a 1 indica que no hay salto artificial.

# %%
rows = []
cutoff = pd.Timestamp(config.REAL_INTRADAY_START_DATE)
for name, full_hist in full_history_by_generator.items():
    for tk, panel in full_hist.items():
        synth_tail = panel.loc[(panel.index < cutoff) & (panel.index >= cutoff - pd.Timedelta(days=730)), "realized_vol"]
        real_head = panel.loc[(panel.index >= cutoff) & (panel.index < cutoff + pd.Timedelta(days=730)), "realized_vol"]
        if len(synth_tail) and len(real_head):
            rows.append({"generador": name, "ticker": tk, "ratio_empalme": synth_tail.mean() / real_head.mean()})

empalme = pd.DataFrame(rows).groupby("generador")["ratio_empalme"].agg(["mean", "std"])
empalme.to_csv(config.TABLES_DIR / "03_continuidad_empalme.csv")
empalme

# %% [markdown]
# ## 6. Dataset final por generador: ventanas X/Y de 30 años
#
# `X`: ventana de `WINDOW_X_DAYS` días con, por banco, [retorno diario,
# volatilidad realizada (real o sintética según la fecha)]. `Y`: retorno del
# **día siguiente** por banco (`WINDOW_Y_DAYS=1`).

# %%
datasets_by_generator = {}
for name, rv_panel in rv_panel_by_generator.items():
    combined = pd.concat(
        [returns_predictor[config.PREDICTOR_TICKERS],
         rv_panel[config.PREDICTOR_TICKERS].add_suffix("_rv")],
        axis=1,
    ).dropna()
    X, Y_wide, idx = feat.build_xy_windows(combined, config.WINDOW_X_DAYS, config.WINDOW_Y_DAYS)
    Y = Y_wide[:, : config.N_PREDICTOR_TICKERS]  # solo retornos, no la parte "_rv" de Y
    is_synthetic = np.asarray(idx < cutoff)
    datasets_by_generator[name] = (X, Y, idx, is_synthetic)
    print(f"{name}: X {X.shape}  Y {Y.shape}  rango {idx.min().date()} -> {idx.max().date()}  "
          f"({is_synthetic.mean():.1%} de las filas con volatilidad sintetica en algun punto)")

# %% [markdown]
# ## 7. Guardar (`datos/interim/`, gitignored)

# %%
for name, (X, Y, idx, is_synthetic) in datasets_by_generator.items():
    np.savez(
        config.INTERIM_DIR / f"dataset_{name}.npz",
        X=X, Y=Y, idx=idx.values.astype("datetime64[ns]"), is_synthetic=is_synthetic,
    )
print("Guardados:", [f"dataset_{n}.npz" for n in datasets_by_generator])
