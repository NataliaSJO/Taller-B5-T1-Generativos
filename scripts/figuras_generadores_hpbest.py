"""
Regenera los diagnosticos del notebook 02 con los hiperparametros GANADORES
de la busqueda (`scripts/analizar_hpsearch.py --que generadores`), y los
compara con los hiperparametros DE CLASE que usa el notebook.

Por que existe este script y no una celda mas del notebook 02: el notebook
documenta la ejecucion con la configuracion de partida (la del material de
clase), y sus figuras son las que cita el README. Aqui se reentrena lo mismo
con la configuracion optimizada y se guarda todo con sufijo `_hpbest`, sin
tocar ninguna figura existente.

Protocolo IDENTICO al del notebook 02 para que la comparacion sea limpia:
mismo pool condicional, misma exclusion de val+test, mismo 10% de holdout
real con semilla 42, mismas 50.000 muestras sinteticas por generador y el
mismo recorte de columnas no negativas.

Uso:
    python scripts/figuras_generadores_hpbest.py --datos RUTA_A/datos
    python scripts/figuras_generadores_hpbest.py --datos ... --sin-gan

`--datos` hace falta porque `datos/` esta gitignored: apunta a la copia del
repositorio donde si estan los ficheros intermedios y el dump de Norgate.
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

from src import config, data_norgate as dn, features as feat, generators as gen, plotting as pl  # noqa: E402
import hp_search_generators as hps  # noqa: E402  (mmd_rbf / evaluate_synth)

N_SYNTH = 50_000
SEMILLA = 42

# --- Configuraciones de clase (las que ejecuta el notebook 02) -------------
CFG_CLASE = {
    "noise": dict(sigma=0.15, relative=True, random_state=SEMILLA),
    "gaussian": dict(shrinkage=True, random_state=SEMILLA),
    "rbig": dict(n_iters=20, grid_size=400, random_state=SEMILLA),
    "gan": dict(
        latent_dim=32, epochs=1000, batch_size=64,
        gen_hidden=(64, 128, 64), disc_hidden=(64, 32),
        learning_rate=1e-4, d_steps_per_g=2, random_state=SEMILLA,
    ),
}

# --- Configuraciones ganadoras de la busqueda (regla de 1 error estandar) --
CFG_HPBEST = {
    "noise": dict(sigma=0.0185, relative=True, noise_dist="student_t", df=4.0, random_state=SEMILLA),
    "gaussian": dict(
        shrinkage=False, shrinkage_alpha=None, shrinkage_target="diagonal",
        marginal="rank_gauss", random_state=SEMILLA,
    ),
    "rbig": dict(n_iters=100, grid_size=800, rotation="pca", random_state=SEMILLA),
    "gan": dict(
        latent_dim=48, epochs=2000, batch_size=128,
        gen_hidden=(16, 32, 16), disc_hidden=(64, 32),
        learning_rate=3e-4, d_steps_per_g=5, random_state=SEMILLA,
    ),
}

ORDEN = ["noise", "gaussian", "rbig", "gan"]
ETIQUETA = {"noise": "Ruido", "gaussian": "Gaussiana", "rbig": "RBIG", "gan": "GAN"}

# Colores del "antes/despues": la direccion del cambio, no la identidad del
# generador. Se refuerza con la forma del punto (hueco = clase, relleno =
# optimizada) para no depender solo del color.
C_CLASE = "#8d9aa0"
C_MEJORA = "#0f6b63"
C_EMPEORA = "#a4482f"


# ---------------------------------------------------------------------------
# 1. Pool condicional (reconstruido igual que en el notebook 00)
# ---------------------------------------------------------------------------
def construir_pool(datos_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    duckdb_path = (
        datos_dir / "extracted" / "norgate_bancos_us_export_20260602_1043"
        / "bancos_us_norgate.duckdb"
    )
    if not duckdb_path.exists():
        raise SystemExit(f"No encuentro el dump de Norgate en {duckdb_path}")

    print("[1/5] Retornos diarios reales del universo generador (Norgate)...", flush=True)
    precios = dn.load_daily_prices(
        config.GENERATOR_TICKERS,
        start=config.REAL_INTRADAY_START_DATE,
        duckdb_path=duckdb_path,
    )
    retornos = dn.compute_log_returns(precios, dropna=None)

    print("[2/5] Features intradia reales (EODHD, ya cacheadas)...", flush=True)
    largo = pd.read_parquet(datos_dir / "interim" / "intraday_features_real.parquet")
    feats = {
        tk: g.set_index("date")[feat.INTRADAY_FEATURE_COLS]
        for tk, g in largo.groupby("ticker")
    }

    pool, meta = feat.build_conditional_pool(retornos, feats)
    print(f"      pool reconstruido: {pool.shape[0]:,} filas x {pool.shape[1]} columnas", flush=True)
    return pool, meta


def partir_pool(pool_full: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mismo troceo que el notebook 02: fuera val+test, y un 10% aleatorio
    como referencia real contra la que comparar los sinteticos."""
    holdout = meta["date"] >= pd.Timestamp(config.VAL_START_DATE)
    pool = pool_full[~holdout.values]
    rng = np.random.default_rng(SEMILLA)
    barajado = rng.permutation(len(pool))
    n_val = max(int(0.1 * len(pool)), 500)
    val_idx, train_idx = barajado[:n_val], barajado[n_val:]
    print(
        f"      tras quitar val+test: {len(pool):,}  ->  train {len(train_idx):,} | "
        f"referencia real {len(val_idx):,}",
        flush=True,
    )
    return pool[train_idx], pool[val_idx]


# ---------------------------------------------------------------------------
# 2. Entrenar y medir
# ---------------------------------------------------------------------------
def entrenar(nombre: str, cfg: dict, pool_train: np.ndarray):
    clase = gen.GENERATOR_REGISTRY[nombre]
    t0 = time.time()
    g = clase(**cfg)
    g.fit(pool_train)
    synth = feat.clip_nonnegative_pool_columns(g.sample(N_SYNTH))
    return g, synth, time.time() - t0


def medir(synth: np.ndarray, pool_val: np.ndarray) -> dict:
    return hps.evaluate_synth(pool_val, synth, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# 3. Figuras
# ---------------------------------------------------------------------------
def fig_histogramas(synth_por_gen: dict, pool_val: np.ndarray) -> None:
    cols = ["log_return", *feat.INTRADAY_FEATURE_COLS]
    nombres = [n for n in ORDEN if n in synth_por_gen]
    fig, axes = plt.subplots(
        len(nombres), len(cols), figsize=(3.1 * len(cols), 2.6 * len(nombres)), squeeze=False
    )
    for i, nombre in enumerate(nombres):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            pl.plot_real_vs_synthetic_hist(
                pool_val[:, j], synth_por_gen[nombre][:, j],
                xlabel=col if i == len(nombres) - 1 else "",
                title=ETIQUETA[nombre] if j == 0 else "", ax=ax, bins=40,
            )
            if j > 0:
                ax.set_ylabel("")
    fig.suptitle(
        "Real vs. sintetico con los hiperparametros ganadores de la busqueda",
        y=1.005, fontsize=12,
    )
    fig.tight_layout()
    pl.savefig(fig, "02_real_vs_sintetico_por_generador_hpbest")
    plt.close(fig)
    print("      -> 02_real_vs_sintetico_por_generador_hpbest.png", flush=True)


def fig_convergencia_rbig(g_clase, g_best) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for g, etiqueta, estilo in (
        (g_clase, "clase (20 iter., rotacion aleatoria)", "--"),
        (g_best, "optimizada (100 iter., rotacion PCA)", "-"),
    ):
        if g is None:
            continue
        h = g.excess_kurtosis_history_
        ax.plot(range(1, len(h) + 1), h, estilo, linewidth=1.8,
                color=pl.color_for("rbig"), alpha=1.0 if estilo == "-" else 0.55,
                label=f"{etiqueta} -> {h[-1]:.3f}")
    ax.set_xlabel("iteracion RBIG")
    ax.set_ylabel("exceso de curtosis medio |k-3|")
    ax.set_title("RBIG: convergencia hacia una Normal conjunta")
    ax.legend(frameon=False, fontsize=9)
    pl.style_axes(ax)
    fig.tight_layout()
    pl.savefig(fig, "02_rbig_convergencia_hpbest")
    plt.close(fig)
    print("      -> 02_rbig_convergencia_hpbest.png", flush=True)


def fig_convergencia_gan(g_clase, g_best) -> None:
    presentes = [(g, t) for g, t in ((g_clase, "clase"), (g_best, "optimizada")) if g is not None]
    fig, axes = plt.subplots(1, len(presentes), figsize=(5.6 * len(presentes), 3.9), squeeze=False)
    for ax, (g, titulo) in zip(axes[0], presentes):
        ax.plot(g.history_["d_loss"], color=pl.color_for("gan"), linewidth=1.2, label="discriminador")
        ax.plot(g.history_["g_loss"], color=pl.color_for("gan"), linewidth=1.2,
                linestyle="--", alpha=0.7, label="generador")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (binary cross-entropy)")
        ax.set_title(f"GAN - configuracion {titulo}")
        ax.legend(frameon=False, fontsize=9)
        pl.style_axes(ax)
    fig.tight_layout()
    pl.savefig(fig, "02_gan_convergencia_hpbest")
    plt.close(fig)
    print("      -> 02_gan_convergencia_hpbest.png", flush=True)


def fig_comparacion(tabla: pd.DataFrame) -> None:
    """Un panel por metrica (unidades distintas -> escalas distintas, nunca
    dos ejes en el mismo panel). Cada fila es un generador y la linea va de
    la configuracion de clase a la optimizada; menor es mejor en las tres."""
    metricas = [
        ("mmd", "MMD (kernel RBF)"),
        ("wasserstein_mean", "Wasserstein-1 medio"),
        ("frobenius_corr", "Frobenius de correlacion"),
    ]
    nombres = [n for n in ORDEN if n in tabla.index.get_level_values(0)]
    fig, axes = plt.subplots(1, len(metricas), figsize=(4.3 * len(metricas), 3.6), squeeze=False)

    for ax, (metrica, titulo) in zip(axes[0], metricas):
        x0s, x1s = [], []
        for i, nombre in enumerate(nombres):
            x0 = float(tabla.loc[(nombre, "clase"), metrica])
            x1 = float(tabla.loc[(nombre, "hpbest"), metrica])
            x0s.append(x0)
            x1s.append(x1)
            color = C_MEJORA if x1 <= x0 else C_EMPEORA
            ax.plot([x0, x1], [i, i], color=color, linewidth=2.2, solid_capstyle="round", zorder=1)
            ax.scatter([x0], [i], s=44, facecolors="none", edgecolors=C_CLASE, linewidths=1.6, zorder=2)
            ax.scatter([x1], [i], s=50, color=color, zorder=3)

        lo, hi = min(x0s + x1s), max(x0s + x1s)
        margen = max((hi - lo) * 0.30, 1e-6)
        ax.set_xlim(lo - margen * 0.45, hi + margen)
        for i, nombre in enumerate(nombres):
            x1 = float(tabla.loc[(nombre, "hpbest"), metrica])
            x0 = float(tabla.loc[(nombre, "clase"), metrica])
            ax.annotate(
                f"{x1:.3f}", (max(x0, x1), i), textcoords="offset points",
                xytext=(9, -3.5), fontsize=8.5, color="#3d4a4f",
            )
        ax.set_yticks(range(len(nombres)))
        ax.set_yticklabels([ETIQUETA[n] for n in nombres], fontsize=10)
        ax.invert_yaxis()
        ax.set_title(titulo, fontsize=10.5)
        pl.style_axes(ax)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=C_CLASE, markersize=7, label="hiperparametros de clase"),
        Line2D([], [], marker="o", linestyle="none", color=C_MEJORA, markersize=7,
               label="optimizados: mejora"),
        Line2D([], [], marker="o", linestyle="none", color=C_EMPEORA, markersize=7,
               label="optimizados: empeora"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Efecto de la busqueda de hiperparametros (menor = mejor en las tres metricas)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    pl.savefig(fig, "02_fidelidad_clase_vs_hpbest")
    plt.close(fig)
    print("      -> 02_fidelidad_clase_vs_hpbest.png", flush=True)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, default=config.DATA_DIR,
                    help="carpeta datos/ (gitignored) con extracted/ e interim/")
    ap.add_argument("--sin-gan", action="store_true", help="salta los dos entrenamientos de GAN")
    args = ap.parse_args()

    pool_full, meta = construir_pool(args.datos)
    pool_train, pool_val = partir_pool(pool_full, meta)

    filas: list[dict] = []
    synth_hpbest: dict[str, np.ndarray] = {}
    modelos: dict[tuple[str, str], object] = {}

    rapidos = ["noise", "gaussian", "rbig"]
    print("[3/5] Generadores rapidos (ambas configuraciones)...", flush=True)
    for nombre in rapidos:
        for etiqueta, cfgs in (("clase", CFG_CLASE), ("hpbest", CFG_HPBEST)):
            g, synth, seg = entrenar(nombre, cfgs[nombre], pool_train)
            m = medir(synth, pool_val)
            filas.append({"generador": nombre, "config": etiqueta, **m, "segundos": round(seg, 1)})
            modelos[(nombre, etiqueta)] = g
            if etiqueta == "hpbest":
                synth_hpbest[nombre] = synth
                np.save(args.datos / "interim" / f"synthetic_pool_{nombre}_hpbest.npy", synth)
            print(f"      {ETIQUETA[nombre]:<10} {etiqueta:<7} "
                  f"MMD={m['mmd']:.6f}  W1={m['wasserstein_mean']:.4f}  "
                  f"Frob={m['frobenius_corr']:.3f}  ({seg:.0f}s)", flush=True)

    def volcar():
        tabla = pd.DataFrame(filas).set_index(["generador", "config"]).sort_index()
        tabla.to_csv(config.TABLES_DIR / "02_generadores_clase_vs_hpbest.csv")
        return tabla

    print("[4/5] Figuras con lo que ya hay...", flush=True)
    tabla = volcar()
    fig_histogramas(synth_hpbest, pool_val)
    fig_convergencia_rbig(modelos.get(("rbig", "clase")), modelos.get(("rbig", "hpbest")))
    fig_comparacion(tabla)

    if not args.sin_gan:
        print("[5/5] GAN (lento: 2000 epochs x 5 pasos de discriminador)...", flush=True)
        for etiqueta, cfgs in (("hpbest", CFG_HPBEST), ("clase", CFG_CLASE)):
            g, synth, seg = entrenar("gan", cfgs["gan"], pool_train)
            m = medir(synth, pool_val)
            filas.append({"generador": "gan", "config": etiqueta, **m, "segundos": round(seg, 1)})
            modelos[("gan", etiqueta)] = g
            if etiqueta == "hpbest":
                synth_hpbest["gan"] = synth
                np.save(args.datos / "interim" / "synthetic_pool_gan_hpbest.npy", synth)
            print(f"      GAN        {etiqueta:<7} MMD={m['mmd']:.6f}  "
                  f"W1={m['wasserstein_mean']:.4f}  Frob={m['frobenius_corr']:.3f}  ({seg:.0f}s)",
                  flush=True)

        tabla = volcar()
        fig_histogramas(synth_hpbest, pool_val)
        fig_convergencia_gan(modelos.get(("gan", "clase")), modelos.get(("gan", "hpbest")))
        fig_comparacion(tabla)

    print("\n" + tabla.round(6).to_string(), flush=True)
    print(f"\nTabla -> {config.TABLES_DIR / '02_generadores_clase_vs_hpbest.csv'}", flush=True)


if __name__ == "__main__":
    main()
