# %% [markdown]
# # 05 · Análisis de resultados
#
# Lee los resultados guardados por los notebooks 02 y 04 (no reentrena
# nada) y construye las tablas/gráficas finales para el README y la
# presentación: ¿mejora el predictor al añadir historia sintética?, ¿qué
# generador funciona mejor?, ¿coincide con qué generador preserva mejor la
# distribución conjunta real (notebook 02)?

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, plotting as pl

# %% [markdown]
# ## 1. Cargar resultados

# %%
results = pd.read_csv(config.TABLES_DIR / "04_resultados_rejilla_profundidad.csv")
corr_quality = pd.read_csv(config.TABLES_DIR / "02_calidad_correlacion_generadores.csv", index_col=0)
empalme = pd.read_csv(config.TABLES_DIR / "03_continuidad_empalme.csv", index_col=0)
results.head()

# %% [markdown]
# ## 2. Tabla final: MAE por generador y años de backfill sintético añadidos
#
# `synth_years=0` es la referencia "solo reales": en ese punto los 4
# generadores entrenan exactamente con los mismos datos (todavía no se ha
# añadido ningún día sintético), así que comparten valor. Se construye la
# tabla con esa fila replicada para las 4 columnas — en vez de dejar una
# columna `solo_reales` suelta que solo tendría dato en la fila 0 y dejaría
# el resto de la tabla lleno de huecos.

# %%
GENERADORES = [g for g in results["generator"].unique() if g != "solo_reales"]
baseline = results[results["generator"] == "solo_reales"].iloc[0]
baseline_mae = baseline["mae"]


def pivot_con_baseline(metric: str) -> pd.DataFrame:
    """Pivot (synth_years x generador) de `metric`, con la fila
    synth_years=0 rellenada con el valor de referencia 'solo reales' para
    los 4 generadores (en ese punto son el mismo modelo)."""
    piv = (
        results[results["generator"] != "solo_reales"]
        .pivot_table(index="synth_years", columns="generator", values=metric)
    )
    piv.loc[0] = baseline[metric]
    return piv.sort_index()[GENERADORES]


pivot_mae = pivot_con_baseline("mae")
pivot_dir = pivot_con_baseline("directional_accuracy")
mejora_pct = (baseline_mae - pivot_mae) / baseline_mae * 100

pivot_mae.to_csv(config.TABLES_DIR / "05_tabla_mae_final.csv")
pivot_dir.to_csv(config.TABLES_DIR / "05_tabla_precision_direccional.csv")
mejora_pct.to_csv(config.TABLES_DIR / "05_tabla_mejora_pct.csv")

print(
    "Referencia 'solo reales' (synth_years=0, ~4,6 años reales sin sintéticos): "
    f"MAE {baseline_mae:.5f} | precisión direccional {baseline['directional_accuracy']:.1%}"
)
pivot_mae.round(5)

# %% [markdown]
# Mejora relativa del MAE frente a la referencia (positivo = mejor que
# entrenar solo con datos reales):

# %%
mejora_pct.round(2)

# %% [markdown]
# Precisión direccional (fracción de días con el signo del retorno
# acertado; 0.5 = azar):

# %%
pivot_dir.round(4)

# %% [markdown]
# ## 3. ¿Mejora el predictor al añadir historia sintética?

# %%
fig = pl.plot_depth_grid_results(results, metric="mae")
pl.savefig(fig, "05_resultado_final_mae")
fig

# %% [markdown]
# ## 4. Mejor generador a máxima profundidad sintética vs. calidad de
#    reconstrucción de la distribución conjunta (notebook 02)
#
# Si el generador que MEJOR reproduce la distribución conjunta real
# (`dist_frobenius_corr` más bajo) es también el que da mejor MAE final, es
# una señal fuerte de que la calidad del generador —y no solo la cantidad
# de datos sintéticos— importa.

# %%
synth_max = results["synth_years"].max()
final_depth = (
    results[results["synth_years"] == synth_max]
    .set_index("generator")[["mae", "mse", "pct_synth"]]
    .join(corr_quality, how="left")
    .sort_values("mae")
)
final_depth.to_csv(config.TABLES_DIR / "05_tabla_generador_final.csv")
final_depth

# %%
fig, ax = plt.subplots(figsize=(6, 4))
for name, row in final_depth.iterrows():
    ax.scatter(row["dist_frobenius_corr"], row["mae"], s=70, color=pl.color_for(name), label=name)
    ax.annotate(name, (row["dist_frobenius_corr"], row["mae"]), textcoords="offset points",
                xytext=(6, 4), fontsize=9)
ax.set_xlabel("distancia Frobenius correlación real vs. sintética (notebook 02, menor = mejor)")
ax.set_ylabel(f"test MAE con +{synth_max} años de sintéticos")
ax.set_title("Calidad del generador vs. rendimiento final del predictor")
pl.style_axes(ax)
fig.tight_layout()
pl.savefig(fig, "05_calidad_generador_vs_mae")
fig

# %% [markdown]
# ## 5. Precisión direccional (¿acierta el signo del retorno?) y desglose
#    por banco
#
# El MAE mide error de MAGNITUD; para un "predictor de precios" también
# importa si acierta la DIRECCIÓN (sube/baja) — 0.5 = tan bueno como una
# moneda. Y el MAE por banco (notebook 04, misma idea que el bloque final
# de `Taller_con_Datos_SP500_promedio.ipynb`) evita que la comparación
# quede dominada por los bancos de mayor volatilidad (ver notebook 01).

# %%
fig = pl.plot_depth_grid_results(results, metric="directional_accuracy")
pl.savefig(fig, "05_precision_direccional")
fig

# %%
mae_por_banco = pd.read_csv(config.TABLES_DIR / "04_mae_por_banco.csv", index_col=0)
mae_por_banco.round(5)

# %% [markdown]
# ## 6. Continuidad del empalme real/sintético por generador (notebook 03)

# %%
empalme.round(3)

# %% [markdown]
# ## 7. Resumen para el README / presentación
#
# Esta celda imprime, en texto, el resumen que debe copiarse (o citarse) en
# el README y en las diapositivas de resultados.

# %%
mejor_generador = final_depth["mae"].idxmin()
mejora_mejor = mejora_pct.loc[synth_max, mejor_generador]
print(
    f"- MAE solo con la ventana real (sin sintéticos): {baseline_mae:.5f}\n"
    f"- Mejor generador con +{synth_max} años de sintéticos: {mejor_generador} "
    f"(MAE {final_depth.loc[mejor_generador, 'mae']:.5f}, "
    f"{mejora_mejor:+.1f}% vs. solo reales)\n"
    f"- % de días con volatilidad sintética con +{synth_max} años añadidos: "
    f"{final_depth.loc[mejor_generador, 'pct_synth']:.1%}\n"
)
