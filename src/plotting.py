"""
Estilo y funciones de grafico compartidas por todos los notebooks, para que
todas las figuras del proyecto (curvas de loss, comparativas de generadores,
distribuciones real vs. sintetico, resultados finales) sean visualmente
consistentes.

Paleta categorica fija (nunca se reasigna por orden/ranking, siempre por
identidad de serie) tomada de la skill de dataviz del equipo, validada para
que los pares adyacentes sean distinguibles en daltonismo:
  blue #2a78d6, orange #eb6834, aqua #1baf7a, yellow #eda100, magenta #e87ba4
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import config

# Colores fijos por identidad de serie (nunca por ranking/posicion)
PALETTE = {
    "solo_reales": "#2a78d6",  # blue  - referencia "sin sinteticos"
    "noise": "#eb6834",        # orange
    "gaussian": "#1baf7a",     # aqua
    "rbig": "#eda100",         # yellow
    "gan": "#e87ba4",          # magenta
    "real": "#2a78d6",         # blue  - datos reales (en comparativas real/sintetico)
    "synthetic": "#eb6834",    # orange - datos sinteticos (generico)
}

MUTED_GRID = "#d8d7d2"
TEXT_SECONDARY = "#52514e"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED_GRID)
    ax.spines["bottom"].set_color(MUTED_GRID)
    ax.grid(True, axis="y", color=MUTED_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY)


def color_for(name: str, fallback_idx: int = 0) -> str:
    if name in PALETTE:
        return PALETTE[name]
    extra = ["#4a3aa7", "#008300", "#e34948"]
    return extra[fallback_idx % len(extra)]


def plot_loss_history(history: dict, title: str = "", ax=None, color=None):
    """Curva de convergencia (loss train/val) de UN entrenamiento keras
    (`model.fit(...).history`). Es la grafica que pide el enunciado para
    "ver que el modelo ha convergido"."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    c = color or PALETTE["solo_reales"]
    ax.plot(history["loss"], label="train", color=c, linewidth=1.8)
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val", color=c, linewidth=1.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False)
    style_axes(ax)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_loss_grid(histories: dict, ncols: int = 3, figsize_per=(4.2, 3.2)):
    """Rejilla de curvas de loss para varios entrenamientos a la vez.
    `histories`: {etiqueta: history.history}."""
    n = len(histories)
    ncols = min(ncols, n) or 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (label, hist) in zip(axes, histories.items()):
        gen_name = label[0] if isinstance(label, tuple) else label
        plot_loss_history(hist, title=str(label), ax=ax, color=color_for(gen_name))
    for ax in axes[len(histories):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_intraday_profile(profile, feature: str = "mean_abs_ret", ax=None, title=None):
    """Perfil de una variable intradia (media |retorno 5min|, volumen medio,
    etc.) a lo largo de las ~78 franjas horarias de la sesion. Es el grafico
    central del EDA "distribucion a lo largo del dia"."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(profile))
    ax.plot(x, profile[feature].values, color=PALETTE["real"], linewidth=1.8)
    step = max(len(profile) // 12, 1)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(profile.index[::step], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("hora (ET, horario de mercado)")
    ax.set_ylabel(feature)
    ax.set_title(title or f"Perfil intradia: {feature}", fontsize=11)
    style_axes(ax)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_real_vs_synthetic_hist(real_values, synth_values, xlabel="", title="", ax=None, bins=40):
    """Histograma superpuesto real vs. sintetico para UNA variable (ej.
    volatilidad realizada), para validar visualmente que el generador
    reproduce la distribucion real."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    rng = (
        min(np.nanmin(real_values), np.nanmin(synth_values)),
        max(np.nanmax(real_values), np.nanmax(synth_values)),
    )
    ax.hist(real_values, bins=bins, range=rng, density=True, alpha=0.55,
            color=PALETTE["real"], label="real")
    ax.hist(synth_values, bins=bins, range=rng, density=True, alpha=0.55,
            color=PALETTE["synthetic"], label="sintetico")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("densidad")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False)
    style_axes(ax)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_depth_grid_results(results, metric: str = "mae", ax=None):
    """A partir de la tabla que devuelve train_utils.run_depth_grid, dibuja
    `metric` (test MAE/MSE) frente a `synth_years` (anios de backfill
    sintetico anadidos), una linea por generador (color fijo por
    identidad, ver PALETTE)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for gen_name, g in results.groupby("generator"):
        g = g.sort_values("synth_years")
        ax.plot(
            g["synth_years"], g[metric],
            marker="o", markersize=5, linewidth=1.8,
            color=color_for(gen_name), label=gen_name,
        )
    ax.set_xlabel("anios de backfill sintetico anadidos")
    ax.set_ylabel(f"test {metric.upper()}")
    ax.set_title(f"Predictor del dia siguiente: test {metric.upper()} vs. anios de sinteticos anadidos", fontsize=11)
    ax.legend(frameon=False, title="generador")
    style_axes(ax)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def savefig(fig, name: str, subdir: str | None = None):
    """Guarda una figura en reports/figures/ (o un subdirectorio), en PNG a
    150dpi, para que el enunciado ("el codigo debe generar todas las
    graficas") quede satisfecho con ficheros versionables en el repo."""
    out_dir = config.FIGURES_DIR / subdir if subdir else config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
