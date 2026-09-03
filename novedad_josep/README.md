# novedad_josep · Los generadores con los hiperparámetros optimizados

Material añadido sobre los notebooks **02 (modelos generativos)** y **03
(backfill condicional)**. Va en esta carpeta aparte para no tocar la
estructura del repositorio: **los notebooks de `notebooks/` no se han
modificado** y siguen siendo los canónicos, los que alimentan los notebooks
04 y 05 y las cifras del README principal.

## Qué contiene

```
figuras_generadores_hpbest.py     rehace los diagnósticos del notebook 02 con
                                  los hiperparámetros ganadores de la búsqueda
                                  y los compara con los de clase
figuras_backfill_hpbest.py        rehace el backfill del notebook 03 con esos
                                  generadores optimizados
anexo_hpbest_a_notebooks.py       inyecta el anexo en markdown en los notebooks
figuras/                          las 5 figuras nuevas
tablas/                           las 2 tablas nuevas
notebooks_con_anexo/              copia de los notebooks 02 y 03 con el anexo
                                  añadido al final (el cuerpo es idéntico)
```

## Cómo se generó

Mismo pool (141.065 filas, reconstruido desde el dump de Norgate y el
intradía ya cacheado), misma exclusión de validación y test, misma partición
y misma semilla que el notebook 02. Lo único que cambia entre las dos
columnas de cada tabla son los hiperparámetros:

```bash
python novedad_josep/figuras_generadores_hpbest.py --datos <ruta>/datos
python novedad_josep/figuras_backfill_hpbest.py    --datos <ruta>/datos
python novedad_josep/anexo_hpbest_a_notebooks.py
```

Hace falta `--datos` porque `datos/` está en el `.gitignore`: hay que
apuntar a la copia local que tenga `extracted/` (dump de Norgate) e
`interim/` (features intradía cacheadas de EODHD). Los dos primeros scripts
escriben en `reports/figures/` y `reports/tables/`; las copias que hay aquí
se movieron a mano a `figuras/` y `tablas/`.

## Configuraciones ganadoras

Salen de `scripts/analizar_hpsearch.py --que generadores`, con la regla de un
error estándar y desempate por Wasserstein:

| Generador | Configuración |
|---|---|
| Ruido | `sigma=0.0185`, relativo, ruido t-Student (4 g.l.) |
| Gaussiana | sin shrinkage, marginal `rank_gauss` (cópula) |
| RBIG | `n_iters=100`, `grid_size=800`, rotación PCA |
| GAN | `latent=48`, 2000 epochs, `batch=128`, `lr=3e-4`, `d_steps_per_g=5` |

## Resultados

| Generador | MMD clase → óptima | W1 clase → óptima | Frobenius clase → óptima | empalme |
|---|---|---|---|---|
| Ruido | 0.000000 → 0.000000 | 0.023 → 0.019 | 0.127 → 0.115 | 1.196 |
| Gaussiana | 0.038 → 0.004 | 0.212 → 0.020 | 0.097 → 0.292 | 1.145 |
| RBIG | 0.000009 → 0.00055 | 0.037 → 0.023 | 0.196 → 0.117 | 1.186 |
| GAN | 0.525 → 0.010 | 0.943 → 0.140 | 1.886 → 0.354 | 1.301 → **1.025** |

1. **El colapso de modo del GAN era cuestión de hiperparámetros.** El
   parámetro decisivo es `d_steps_per_g`. Su ratio de empalme pasa además de
   1.301 a 1.025 — prácticamente sin escalón de nivel, el mejor de los
   cuatro.
2. **`rank_gauss` arregla las marginales de la Gaussiana** (MMD ×9, W1 ×11)
   y su serie de 30 años deja de ser una banda plana sin crisis; a cambio
   empeora la matriz de correlación. La cópula compra marginales a cambio de
   dependencia.
3. **RBIG con rotación PCA** mejora W1 y Frobenius, y el racimo de
   volatilidad de 2008-09 se ve claramente en la serie de JPM.
4. **Lo que no cambia: la persistencia.** La autocorrelación a un día del
   tramo sintético se queda en 0.08–0.12 frente a 0.588 del tramo real con
   los cuatro generadores. El clustering lo destruye el *conditional
   matching* —un sorteo independiente por día—, no el generador, así que
   optimizar el generador no lo recupera. Ver `v2_persistencia_temporal/`.

## Aviso metodológico: el Frobenius no soporta un ranking

Repitiendo la **misma** configuración sobre el **mismo** pool y cambiando
solo qué 10 % de filas cae en el holdout, con 12 particiones:

| Generador (config. de clase) | Frobenius mín. | máx. | media | desv. |
|---|---|---|---|---|
| Ruido | 0.112 | 0.388 | 0.223 | 0.065 |
| Gaussiana | 0.105 | 0.339 | 0.200 | 0.062 |

Las diferencias entre Ruido, Gaussiana y RBIG caben enteras dentro de esa
dispersión: lo único que queda fuera de la banda es el colapso del GAN sin
optimizar. La causa es que la partición del pool es **por fila**, y en un
mismo día hay decenas de bancos correlacionados, así que el holdout no es
independiente del entrenamiento. Lo correcto sería partir **por día**.

Nada de esto cambia los notebooks 04 y 05: la selección del generador se
hace por rendimiento aguas abajo, y ahí las cuatro familias empatan (README
principal, sección 6.3).
