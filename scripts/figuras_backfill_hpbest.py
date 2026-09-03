"""
Rehace el backfill condicional del notebook 03 con los generadores
entrenados con los hiperparametros GANADORES de la busqueda, y guarda la
serie de 30 anios de JPM, la continuidad del empalme y la persistencia
(autocorrelacion a un dia) del tramo sintetico.

Mismo mecanismo de conditional matching, mismos k=80 vecinos y misma
semilla que el notebook: lo unico que cambia es el pool sintetico que
entra. Salidas con sufijo `_hpbest`; no se toca ninguna figura existente.

Uso:
    python scripts/figuras_backfill_hpbest.py --datos RUTA_A/datos
    python scripts/figuras_backfill_hpbest.py --datos ... --sin-gan
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import backfill as bf, config, features as feat, plotting as pl  # noqa: E402
from figuras_generadores_hpbest import (  # noqa: E402
    CFG_HPBEST, ETIQUETA, ORDEN, construir_pool, entrenar, partir_pool,
)

K_VECINOS = 80


def autocorr_lag1(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.autocorr(lag=1)) if len(x) > 10 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=config.DATA_DIR)
    ap.add_argument("--sin-gan", action="store_true")
    args = ap.parse_args()

    interim = args.datos / "interim"
    nombres = [n for n in ORDEN if not (args.sin_gan and n == "gan")]

    # --- 1. Pools sinteticos con la configuracion ganadora ------------------
    pools: dict[str, np.ndarray] = {}
    pendientes = []
    for nombre in nombres:
        cache = interim / f"synthetic_pool_{nombre}_hpbest.npy"
        if cache.exists():
            pools[nombre] = np.load(cache)
            print(f"[pool] {ETIQUETA[nombre]}: reutilizado de {cache.name}", flush=True)
        else:
            pendientes.append(nombre)

    if pendientes:
        pool_full, meta = construir_pool(args.datos)
        pool_train, _ = partir_pool(pool_full, meta)
        for nombre in pendientes:
            _, synth, seg = entrenar(nombre, CFG_HPBEST[nombre], pool_train)
            pools[nombre] = synth
            np.save(interim / f"synthetic_pool_{nombre}_hpbest.npy", synth)
            print(f"[pool] {ETIQUETA[nombre]}: entrenado ({seg:.0f}s) y guardado", flush=True)

    # --- 2. Datos reales del predictor -------------------------------------
    retornos = pd.read_parquet(interim / "returns_predictor.parquet")
    retornos = retornos[retornos.index >= pd.Timestamp(config.TOTAL_HISTORY_START_DATE)]
    largo = pd.read_parquet(interim / "intraday_features_real.parquet")
    reales = {
        tk: g.set_index("date")[feat.INTRADAY_FEATURE_COLS]
        for tk, g in largo[largo["ticker"].isin(config.PREDICTOR_TICKERS)].groupby("ticker")
    }
    print(f"[datos] retornos {retornos.shape} | intradia real para {len(reales)} bancos", flush=True)

    # --- 3. Backfill condicional -------------------------------------------
    historias = {}
    for nombre in nombres:
        t0 = time.time()
        historias[nombre] = bf.build_full_history_features(
            retornos, reales, pools[nombre],
            real_start=config.REAL_INTRADAY_START_DATE,
            k_neighbors=K_VECINOS, random_state=config.RANDOM_SEED,
        )
        n_synth = int(historias[nombre][config.PREDICTOR_TICKERS[0]]["is_synthetic"].sum())
        print(f"[backfill] {ETIQUETA[nombre]}: {n_synth} dias sinteticos por ticker "
              f"({time.time() - t0:.0f}s)", flush=True)

    # --- 4. Figura: JPM, 30 anios ------------------------------------------
    demo = "JPM"
    fig, axes = plt.subplots(len(nombres), 1, figsize=(11, 8 / 3 * len(nombres)),
                             sharex=True, squeeze=False)
    for ax, nombre in zip(axes[:, 0], nombres):
        panel = historias[nombre][demo]
        es_synth = panel["is_synthetic"]
        ax.plot(panel.index[es_synth], panel.loc[es_synth, "realized_vol"],
                color="#b7b6ae", linewidth=0.8, label="sintetico (~24 anios)")
        ax.plot(panel.index[~es_synth], panel.loc[~es_synth, "realized_vol"],
                color=pl.color_for(nombre), linewidth=0.8, label="real (~5,5 anios)")
        ax.axvline(pd.Timestamp(config.REAL_INTRADAY_START_DATE),
                   color="#52514e", linewidth=0.8, linestyle=":")
        ax.set_title(f"{demo} - volatilidad realizada diaria - {ETIQUETA[nombre]} "
                     f"(hiperparametros optimizados)", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        pl.style_axes(ax)
    axes[-1, 0].set_xlabel("fecha")
    fig.tight_layout()
    pl.savefig(fig, "03_backfill_serie_temporal_JPM_hpbest")
    plt.close(fig)
    print("      -> 03_backfill_serie_temporal_JPM_hpbest.png", flush=True)

    # --- 5. Empalme y persistencia -----------------------------------------
    corte = pd.Timestamp(config.REAL_INTRADAY_START_DATE)
    filas = []
    for nombre in nombres:
        for tk, panel in historias[nombre].items():
            cola = panel.loc[(panel.index < corte) & (panel.index >= corte - pd.Timedelta(days=730)),
                             "realized_vol"]
            cabeza = panel.loc[(panel.index >= corte) & (panel.index < corte + pd.Timedelta(days=730)),
                               "realized_vol"]
            if not len(cola) or not len(cabeza):
                continue
            filas.append({
                "generador": nombre,
                "ticker": tk,
                "ratio_empalme": cola.mean() / cabeza.mean(),
                "autocorr_sintetico": autocorr_lag1(panel.loc[panel["is_synthetic"], "realized_vol"]),
                "autocorr_real": autocorr_lag1(panel.loc[~panel["is_synthetic"], "realized_vol"]),
            })

    detalle = pd.DataFrame(filas)
    resumen = detalle.groupby("generador").agg(
        ratio_empalme_media=("ratio_empalme", "mean"),
        ratio_empalme_std=("ratio_empalme", "std"),
        autocorr_sintetico=("autocorr_sintetico", "mean"),
        autocorr_real=("autocorr_real", "mean"),
    )
    resumen.to_csv(config.TABLES_DIR / "03_continuidad_empalme_hpbest.csv")
    print("\n" + resumen.round(4).to_string(), flush=True)
    print(f"\nTabla -> {config.TABLES_DIR / '03_continuidad_empalme_hpbest.csv'}", flush=True)


if __name__ == "__main__":
    main()
