"""Figura: como varia el MAE al anadir anios de backfill sintetico, por
horizonte y por generador, siempre contra el modelo nulo de SU horizonte.

    ./scripts/grafica_mae_vs_anios.py            (tras la rejilla del nb 04)

Dos decisiones de diseno que son las que hacen la figura honesta:

1. Un panel por horizonte, cada uno con SU escala y SU constante. Los MAE
   no son comparables entre horizontes: el objetivo a 30 dias es ~5x mas
   pequeno que el de 1 dia solo por promediar, asi que superponerlos en un
   mismo eje sugiere una mejora que no existe.

2. La banda de +-1 error estandar del predictor constante, por bootstrap
   sobre los dias del test. Sin ella el eje Y ampliado convierte
   variaciones en la quinta cifra decimal en "tendencias" visibles, que es
   justo el error de lectura que esta figura tiene que evitar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config, modelos, plotting as pl, train_utils as tu  # noqa: E402
from rejilla_paralela import TRAIN_END, GENERADORES, cargar_datasets  # noqa: E402

N_BOOTSTRAP = 4000


def referencia_constante(datos: dict, h: int) -> tuple[float, float]:
    """MAE en test del modelo nulo y su error estandar (bootstrap por dias).

    La unidad de remuestreo es el DIA, no la prediccion individual: los 25
    bancos de una misma fecha comparten factor sectorial y tratarlos como
    independientes estrecharia la banda unas 5 veces sin justificacion.
    """
    X, Y, idx, is_synth = tu.vista_horizonte(datos, h)
    X_tr, Y_tr, _, _ = tu.slice_by_depth(
        X, Y, idx, 0, TRAIN_END, config.REAL_INTRADAY_START_DATE, is_synth)
    _, (X_test, Y_test) = tu.split_val_test(X, Y, idx, horizonte=h)

    modelo = modelos.build_predictor_constant().fit(X_tr, Y_tr)
    err = np.abs(Y_test - modelo.predict(X_test)).mean(axis=1)

    rng = np.random.default_rng(config.RANDOM_SEED)
    boot = np.array([err[rng.integers(0, len(err), len(err))].mean()
                     for _ in range(N_BOOTSTRAP)])
    return float(err.mean()), float(boot.std())


def resultados(h: int) -> pd.DataFrame:
    nombre = ("04_checkpoint_rejilla_profundidad.csv" if h == config.WINDOW_Y_DAYS
              else f"04_checkpoint_rejilla_profundidad_h{h}.csv")
    ruta = config.INTERIM_DIR / nombre
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


def main() -> None:
    datos = cargar_datasets({GENERADORES[0]})[GENERADORES[0]]

    fig, axes = plt.subplots(1, len(config.HORIZONTES_DIAS),
                             figsize=(5 * len(config.HORIZONTES_DIAS), 4.6))
    resumen = []

    for ax, h in zip(np.atleast_1d(axes), config.HORIZONTES_DIAS):
        cte, ee = referencia_constante(datos, h)
        tabla = resultados(h)

        ax.axhspan(cte - ee, cte + ee, color=pl.MUTED_GRID, alpha=0.5,
                   label="±1 e.e. de la constante")
        ax.axhline(cte, color="#52514e", lw=1.7, label=f"constante ({cte:.5f})")

        for gen in GENERADORES:
            sub = (tabla[tabla.generator == gen].sort_values("synth_years")
                   if len(tabla) else pd.DataFrame())
            if len(sub):
                ax.plot(sub.synth_years, sub.mae, marker="o", ms=5, lw=1.8,
                        color=pl.color_for(gen), label=gen)

        solo = tabla[tabla.generator == "solo_reales"] if len(tabla) else pd.DataFrame()
        if len(solo):
            ax.scatter([0], [solo.mae.iloc[0]], s=110, facecolors="none",
                       edgecolors=pl.PALETTE["solo_reales"], linewidths=2.2,
                       zorder=5, label="solo reales")
            con_sintetico = tabla[tabla.generator != "solo_reales"]
            resumen.append({
                "horizonte": h, "constante": cte, "e.e.": ee,
                "solo_reales": float(solo.mae.iloc[0]),
                "mejor": float(con_sintetico.mae.min()) if len(con_sintetico) else np.nan,
            })

        ax.set_title(f"horizonte {h} día(s)   ·   {len(tabla)}/21 combinaciones",
                     fontsize=10.5)
        ax.set_xlabel("años de backfill sintético añadidos")
        ax.set_ylabel("test MAE")
        ax.legend(frameon=False, fontsize=8)
        pl.style_axes(ax)

    fig.suptitle("¿Mejora el predictor al añadir años de datos sintéticos?", fontsize=13)
    fig.tight_layout()
    pl.savefig(fig, "04_mae_vs_anios_por_horizonte")
    print("figura -> reports/figures/04_mae_vs_anios_por_horizonte.png")

    if resumen:
        r = pd.DataFrame(resumen)
        r["mejor_vs_cte_%"] = (r["mejor"] - r["constante"]) / r["constante"] * 100
        r["en_e.e."] = (r["mejor"] - r["constante"]).abs() / r["e.e."]
        r.round(6).to_csv(config.TABLES_DIR / "04_mae_vs_anios_resumen.csv", index=False)
        print("\n" + r.round(6).to_string(index=False))
        print("\n'en_e.e.' = cuantos errores estandar separan al mejor modelo de la")
        print("constante. Por debajo de 1, la diferencia es indistinguible del ruido.")


if __name__ == "__main__":
    main()
