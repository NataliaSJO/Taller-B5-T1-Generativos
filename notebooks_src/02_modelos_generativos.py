# %% [markdown]
# # 02 · Los 4 modelos generativos
#
# Entrena los 4 generadores del taller (Ruido, Gaussiana, RBIG, GAN — ver
# `src/generators.py` y la justificación de por qué estos 3 tipos + 1 simple
# en `Material_clase/` en el README) sobre el **pool condicional**
# construido en el notebook 00: muestras reales
# `[retorno_diario, realized_vol, open_30m_ret, close_30m_ret, hl_range]`
# de hasta 150 bancos en la ventana real (2020-11 → 2025-06).
#
# Los 4 generadores son **incondicionales**: aprenden la distribución
# conjunta `(retorno, features)` y solo saben muestrear pares nuevos de esa
# conjunta. El paso de condicionar por el retorno diario YA CONOCIDO de cada
# día histórico (para el backfill de los ~24 años sin 5 min reales) se hace
# aparte, en el notebook 03, con el mismo mecanismo de "conditional
# matching" para los 4 — así la comparación del notebook 04 mide solo la
# calidad de cada generador, no un truco de condicionamiento distinto.
#
# **Importante (fuga de datos):** el pool de entrenamiento de los
# generadores excluye tanto el tramo de validación como el de test final
# del predictor (`VAL_START_DATE` en adelante) — los generadores nunca ven
# esas fechas, ni siquiera indirectamente a través de sus estadísticos.

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
import matplotlib.pyplot as plt

from src import config, features as feat, generators as gen, plotting as pl

RANDOM_STATE = 42

# %% [markdown]
# ## 1. Cargar el pool condicional y quitar val/test

# %%
pool_full = np.load(config.INTERIM_DIR / "conditional_pool.npy")
pool_meta = pd.read_parquet(config.INTERIM_DIR / "conditional_pool_meta.parquet")
assert len(pool_full) == len(pool_meta)

holdout_mask = pool_meta["date"] >= pd.Timestamp(config.VAL_START_DATE)
pool = pool_full[~holdout_mask.values]
print(f"pool total: {len(pool_full):,}  ->  tras quitar val+test: {len(pool):,}")
print("columnas: [log_return, " + ", ".join(feat.INTRADAY_FEATURE_COLS) + "]")

# %%
rng = np.random.default_rng(RANDOM_STATE)
shuffled = rng.permutation(len(pool))
n_val = max(int(0.1 * len(pool)), 500)
val_idx, train_idx = shuffled[:n_val], shuffled[n_val:]
pool_train, pool_val = pool[train_idx], pool[val_idx]
print(f"train: {len(pool_train):,}   val (referencia real para comparar): {len(pool_val):,}")

# %% [markdown]
# ## 2. Ruido (modelo simple)
#
# El "ejemplo muy tonto" de `Taller_GANs.ipynb`: recicla muestras reales y
# les suma ruido gaussiano proporcional a la escala de cada columna. No
# optimiza nada -> no hay curva de convergencia que mostrar aquí (es la
# referencia mínima con la que comparar los otros 3).

# %%
gen_noise = gen.NoiseGenerator(sigma=0.15, relative=True, random_state=RANDOM_STATE)
gen_noise.fit(pool_train)
synth_noise = feat.clip_nonnegative_pool_columns(gen_noise.sample(50_000))
print("synth_noise:", synth_noise.shape)

# %% [markdown]
# ## 3. Gaussiana multivariante
#
# Igual que `Taller_Gaussian_solution.ipynb`: ajusta `N(mu, Sigma)` sobre el
# vector conjunto y muestrea de ella. Cierre analítico -> tampoco hay curva
# de "loss" iterativa; el diagnóstico de calidad es comparar directamente la
# distribución sintética con la real (más abajo).

# %%
gen_gauss = gen.GaussianGenerator(shrinkage=True, random_state=RANDOM_STATE)
gen_gauss.fit(pool_train)
synth_gauss = feat.clip_nonnegative_pool_columns(gen_gauss.sample(50_000))
print("synth_gauss:", synth_gauss.shape, " shrinkage aplicado:", round(gen_gauss.shrinkage_, 3))

# %% [markdown]
# ## 4. RBIG (Rotation-Based Iterative Gaussianization)
#
# El método con el que la propia diapositiva de teoría del taller (Prof.
# Laparra) compara los GAN. A diferencia de la Gaussiana, SÍ itera —
# alternando gaussianización marginal + rotación aleatoria — así que sí
# tiene un diagnóstico de convergencia real: el exceso de curtosis medio
# (|kurtosis-3|) de los datos transformados, que debe tender a 0 a medida
# que se acercan a una Normal conjunta.

# %%
gen_rbig = gen.RBIGGenerator(n_iters=20, grid_size=400, random_state=RANDOM_STATE)
gen_rbig.fit(pool_train)
synth_rbig = feat.clip_nonnegative_pool_columns(gen_rbig.sample(50_000))
print("synth_rbig:", synth_rbig.shape)

# %%
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(
    range(1, len(gen_rbig.excess_kurtosis_history_) + 1),
    gen_rbig.excess_kurtosis_history_,
    marker="o", markersize=4, linewidth=1.8, color=pl.color_for("rbig"),
)
ax.set_xlabel("iteración RBIG")
ax.set_ylabel("exceso de curtosis medio |k−3|")
ax.set_title("RBIG: convergencia hacia una Normal conjunta")
pl.style_axes(ax)
fig.tight_layout()
pl.savefig(fig, "02_rbig_convergencia")
fig

# %% [markdown]
# ## 5. GAN (requiere TensorFlow)
#
# GAN densa (generador/discriminador feed-forward) en el mismo espíritu
# que `Taller_GANs.ipynb`, entrenada sobre el mismo pool. `learning_rate`
# bajo y varios pasos de discriminador por paso de generador (ver
# `GANGenerator`, `src/generators.py`) son los hiperparámetros que
# mejor controlan el colapso de modo característico de un GAN vainilla
# con pérdida BCE en un problema de baja dimensión como este (d=5); más
# epochs no ayuda una vez alcanzado ese punto — el criterio para elegir
# `epochs=1000` fue justo ese, no simplemente "más es mejor". Si
# TensorFlow no está disponible, esta celda se salta con un aviso.

# %%
try:
    gen_gan = gen.GANGenerator(
        latent_dim=32, epochs=1000, batch_size=64,
        gen_hidden=(64, 128, 64), disc_hidden=(64, 32),
        learning_rate=1e-4, d_steps_per_g=2, random_state=RANDOM_STATE,
    )
    gen_gan.fit(pool_train)
    synth_gan = feat.clip_nonnegative_pool_columns(gen_gan.sample(50_000))
    print("synth_gan:", synth_gan.shape)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(gen_gan.history_["d_loss"], label="discriminador", color=pl.color_for("gan"), linewidth=1.5)
    ax.plot(gen_gan.history_["g_loss"], label="generador", color=pl.color_for("gan"),
            linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (binary cross-entropy)")
    ax.set_title("GAN: convergencia del entrenamiento adversarial")
    ax.legend(frameon=False)
    pl.style_axes(ax)
    fig.tight_layout()
    pl.savefig(fig, "02_gan_convergencia")
    GAN_AVAILABLE = True
except ImportError as e:
    print(f"[AVISO] GAN no disponible en este entorno: {e}")
    synth_gan = None
    GAN_AVAILABLE = False

# %% [markdown]
# ## 6. Diagnóstico: ¿cada generador reproduce la distribución real?
#
# Histograma real (holdout) vs. sintético para cada una de las 4 features,
# una fila por generador.

# %%
synthetic_pools = {"noise": synth_noise, "gaussian": synth_gauss, "rbig": synth_rbig}
if GAN_AVAILABLE:
    synthetic_pools["gan"] = synth_gan

cols = ["log_return", *feat.INTRADAY_FEATURE_COLS]
fig, axes = plt.subplots(
    len(synthetic_pools), len(cols), figsize=(3.1 * len(cols), 2.6 * len(synthetic_pools))
)
for i, (name, synth) in enumerate(synthetic_pools.items()):
    for j, col in enumerate(cols):
        ax = axes[i, j]
        pl.plot_real_vs_synthetic_hist(
            pool_val[:, j], synth[:, j], xlabel=col if i == len(synthetic_pools) - 1 else "",
            title=name if j == 0 else "", ax=ax, bins=40,
        )
        if j > 0:
            ax.set_ylabel("")
fig.tight_layout()
pl.savefig(fig, "02_real_vs_sintetico_por_generador")
fig

# %% [markdown]
# ## 7. Diagnóstico: matriz de correlación real vs. sintética
#
# ¿Cada generador reproduce también la DEPENDENCIA entre retorno y features
# (no solo las marginales de arriba)? Distancia de Frobenius entre la
# matriz de correlación real (holdout) y la de cada sintético — cuanto más
# baja, mejor conserva el generador la estructura conjunta.

# %%
corr_real = np.corrcoef(pool_val.T)
rows = []
for name, synth in synthetic_pools.items():
    corr_synth = np.corrcoef(synth.T)
    dist = float(np.linalg.norm(corr_real - corr_synth))
    rows.append({"generador": name, "dist_frobenius_corr": dist})
corr_quality = pd.DataFrame(rows).set_index("generador").sort_values("dist_frobenius_corr")
corr_quality.to_csv(config.TABLES_DIR / "02_calidad_correlacion_generadores.csv")
corr_quality

# %% [markdown]
# ## 8. Guardar los pools sintéticos (`datos/interim/`, gitignored)
#
# El notebook 03 los usa para el backfill condicional de los ~24 años sin
# 5 minutos reales.

# %%
for name, synth in synthetic_pools.items():
    np.save(config.INTERIM_DIR / f"synthetic_pool_{name}.npy", synth)
print("Guardados:", list(synthetic_pools.keys()))

# %% [markdown]
# ## 9. Anexo · hiperparametros ganadores de la busqueda
#
# Todo lo anterior usa los hiperparametros **de partida**, los del material
# de clase. La busqueda de `scripts/hp_search_generators.py` encontro
# configuraciones mejores, y `scripts/analizar_hpsearch.py` elige una por
# familia con la regla de un error estandar:
#
# | Generador | Configuracion elegida |
# |---|---|
# | Ruido | `sigma=0.0185`, relativo, ruido **t-Student** (4 g.l.) |
# | Gaussiana | sin shrinkage, marginal **`rank_gauss`** (copula) |
# | RBIG | **`n_iters=100`, `grid_size=800`, rotacion PCA** |
# | GAN | `latent=48`, 2000 epochs, `batch=128`, `lr=3e-4`, **`d_steps_per_g=5`** |
#
# Este anexo repite los tres diagnosticos de arriba con esas
# configuraciones, sobre **el mismo pool, la misma exclusion de val+test y
# la misma particion** que el cuerpo del notebook, para que la comparacion
# solo mida el efecto de los hiperparametros:
#
# ```bash
# python scripts/figuras_generadores_hpbest.py --datos <ruta>/datos
# ```
#
# | Generador | MMD clase → óptima | W1 clase → óptima | Frobenius clase → óptima |
# |---|---|---|---|
# | Ruido | 0.000000 → 0.000000 | 0.0226 → 0.0190 | 0.127 → 0.115 |
# | Gaussiana | 0.038105 → 0.004253 | 0.2120 → 0.0197 | 0.097 → 0.292 |
# | RBIG | 0.000009 → 0.000549 | 0.0372 → 0.0230 | 0.196 → 0.117 |
# | GAN | 0.524680 → 0.009631 | 0.9431 → 0.1400 | 1.886 → 0.354 |
#
# Lectura: la **Gaussiana** es el cambio grande — `rank_gauss` conserva las
# marginales reales (colas pesadas incluidas) y modela solo la dependencia,
# que es lo unico que una Normal si puede capturar; a cambio empeora la
# matriz de correlacion. El **GAN** deja de colapsar al subir
# `d_steps_per_g`. Ruido y RBIG se mueven poco.
#
# ![Real vs sintetico con hiperparametros optimizados](../reports/figures/02_real_vs_sintetico_por_generador_hpbest.png)
#
# ![Efecto de la busqueda sobre las tres metricas](../reports/figures/02_fidelidad_clase_vs_hpbest.png)
#
# ![Convergencia de RBIG, clase vs optimizada](../reports/figures/02_rbig_convergencia_hpbest.png)
#
# ![Convergencia del GAN, clase vs optimizada](../reports/figures/02_gan_convergencia_hpbest.png)
#
# **Cuidado con leer la columna de Frobenius como un ranking.** Repitiendo
# la MISMA configuracion sobre el MISMO pool y cambiando solo que 10% de
# filas cae en el holdout (12 particiones), la distancia de Frobenius del
# Ruido se mueve entre **0.112 y 0.388** (media 0.223, desv. 0.065) y la de
# la Gaussiana entre **0.105 y 0.339** (media 0.200, desv. 0.062). Las
# diferencias entre Ruido, Gaussiana y RBIG caben dentro de esa dispersion:
# lo unico que queda fuera de ella es el colapso del GAN sin optimizar. La
# causa de fondo es que la particion es **por fila** y en un mismo dia hay
# decenas de bancos correlacionados, asi que el holdout no es independiente
# del entrenamiento; lo correcto seria partir **por dia**.
#
# Nada de esto cambia los notebooks 03-05: el pipeline final sigue usando
# los pools de la configuracion de partida, porque la seleccion del
# generador se hace por rendimiento aguas abajo y ahi las cuatro familias
# empatan (ver README, seccion 6.3).
