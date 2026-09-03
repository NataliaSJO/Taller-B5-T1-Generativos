"""
Anade a los notebooks 02 y 03 un anexo en markdown con los resultados de
reejecutarlos con los hiperparametros GANADORES de la busqueda
(`scripts/figuras_generadores_hpbest.py` y `scripts/figuras_backfill_hpbest.py`).

El cuerpo de los dos notebooks NO se toca: sigue documentando la ejecucion
con los hiperparametros de partida (los del material de clase), que son los
que alimentan los notebooks 03-05 y las cifras del README. El anexo se
limita a ensenar que cambia al optimizar los generadores.

Las tablas del anexo se construyen leyendo los CSV de `reports/tables/`, no
se transcriben a mano. El script es idempotente: si el anexo ya esta, lo
sustituye en vez de duplicarlo.

Uso:
    python scripts/anexo_hpbest_a_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config  # noqa: E402
from py_to_ipynb import split_cells, to_notebook  # noqa: E402

MARCA = "Anexo · hiperparametros ganadores de la busqueda"
ETIQUETA = {"noise": "Ruido", "gaussian": "Gaussiana", "rbig": "RBIG", "gan": "GAN"}
ORDEN = ["noise", "gaussian", "rbig", "gan"]


def _fmt(x: float, dec: int = 4) -> str:
    return "—" if pd.isna(x) else f"{x:.{dec}f}"


def tabla_fidelidad() -> str:
    t = pd.read_csv(config.TABLES_DIR / "02_generadores_clase_vs_hpbest.csv")
    t = t.set_index(["generador", "config"])
    filas = [
        "| Generador | MMD clase → óptima | W1 clase → óptima | Frobenius clase → óptima |",
        "|---|---|---|---|",
    ]
    for g in ORDEN:
        if (g, "clase") not in t.index or (g, "hpbest") not in t.index:
            continue
        c, h = t.loc[(g, "clase")], t.loc[(g, "hpbest")]
        filas.append(
            f"| {ETIQUETA[g]} | {_fmt(c['mmd'], 6)} → {_fmt(h['mmd'], 6)} "
            f"| {_fmt(c['wasserstein_mean'])} → {_fmt(h['wasserstein_mean'])} "
            f"| {_fmt(c['frobenius_corr'], 3)} → {_fmt(h['frobenius_corr'], 3)} |"
        )
    return "\n".join(filas)


def tabla_empalme() -> str:
    t = pd.read_csv(config.TABLES_DIR / "03_continuidad_empalme_hpbest.csv").set_index("generador")
    filas = [
        "| Generador | ratio de empalme | desv. entre bancos | autocorr. sintética | autocorr. real |",
        "|---|---|---|---|---|",
    ]
    for g in ORDEN:
        if g not in t.index:
            continue
        r = t.loc[g]
        filas.append(
            f"| {ETIQUETA[g]} | {_fmt(r['ratio_empalme_media'], 3)} "
            f"| {_fmt(r['ratio_empalme_std'], 3)} "
            f"| {_fmt(r['autocorr_sintetico'], 3)} | {_fmt(r['autocorr_real'], 3)} |"
        )
    return "\n".join(filas)


ANEXO_02 = f"""# %% [markdown]
# ## 9. {MARCA}
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
{{tabla_fidelidad}}
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
"""

ANEXO_03 = f"""# %% [markdown]
# ## 8. {MARCA}
#
# El mismo backfill condicional —mismos `k=80` vecinos, mismo kernel, misma
# semilla— alimentado con los pools de los generadores optimizados del
# anexo del notebook 02:
#
# ```bash
# python scripts/figuras_backfill_hpbest.py --datos <ruta>/datos
# ```
#
# ![Backfill de JPM con generadores optimizados](../reports/figures/03_backfill_serie_temporal_JPM_hpbest.png)
#
{{tabla_empalme}}
#
# Tres lecturas:
#
# 1. **La Gaussiana deja de ser una banda plana.** Con la configuracion de
#    clase su volatilidad sintetica no tenia colas, asi que los ~24 anios
#    salian sin crisis; con `rank_gauss` conserva las marginales reales y
#    vuelven a aparecer picos. RBIG ensena ademas un racimo claro en
#    2008-09.
# 2. **El empalme no mejora.** El nivel sintetico sigue un 15-20% por
#    encima del real inmediatamente posterior. Ese sesgo no lo arregla el
#    generador porque no viene solo de el: la ventana de medida sintetica
#    (nov-2018 → oct-2020) incluye el COVID y la real de despues es la
#    recuperacion, y ademas el pool son 150 bancos —muchos pequenos y
#    volatiles— frente a los 25 grandes supervivientes del predictor.
# 3. **La persistencia sigue rota.** La autocorrelacion a un dia del tramo
#    sintetico se queda muy por debajo de la del tramo real, con cualquier
#    generador. Es la consecuencia directa de muestrear cada dia de forma
#    independiente: no es un defecto del generador, sino del mecanismo de
#    condicionamiento (ver `v2_persistencia_temporal/`, que lo corrige
#    condicionando tambien a la volatilidad del dia anterior).
"""


def anexar(src_py: Path, nb_ipynb: Path, texto: str) -> None:
    fuente = src_py.read_text(encoding="utf-8")
    marcador = f"## {MARCA}"

    # idempotencia: si ya hay anexo, se corta por la marca y se reescribe
    if marcador in fuente:
        fuente = fuente[: fuente.index("# %% [markdown]\n# " + marcador)].rstrip("\n")
    fuente = fuente.rstrip("\n") + "\n\n" + texto
    src_py.write_text(fuente, encoding="utf-8")

    nb = json.loads(nb_ipynb.read_text(encoding="utf-8"))
    nb["cells"] = [
        c for c in nb["cells"]
        if not (c["cell_type"] == "markdown" and any(marcador in ln for ln in c["source"]))
    ]
    nb["cells"].extend(to_notebook(split_cells(texto))["cells"])
    nb_ipynb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"anexo -> {src_py.name} y {nb_ipynb.name}", flush=True)


def main() -> None:
    anexar(
        ROOT / "notebooks_src" / "02_modelos_generativos.py",
        ROOT / "notebooks" / "02_modelos_generativos.ipynb",
        ANEXO_02.format(tabla_fidelidad=_comentar(tabla_fidelidad())),
    )
    anexar(
        ROOT / "notebooks_src" / "03_backfill_condicional.py",
        ROOT / "notebooks" / "03_backfill_condicional.ipynb",
        ANEXO_03.format(tabla_empalme=_comentar(tabla_empalme())),
    )


def _comentar(bloque: str) -> str:
    return "\n".join(f"# {ln}" for ln in bloque.split("\n"))


if __name__ == "__main__":
    main()
