# Taller B5-T1 · Generación de datos financieros sintéticos

**Predecir el retorno del día siguiente de acciones bancarias de EEUU, usando
volatilidad intradía real (desde 2020-11, barras de 5 min de EOD Historical
Data) allí donde existe, y volatilidad intradía sintética — generada por 4
modelos generativos distintos — para reconstruir los ~24 años anteriores de
los que solo tenemos precio de cierre diario (Norgate).**

**Resultado (§6-7, con los 6 notebooks ejecutados end-to-end sobre datos
reales): no ayudan — y el porqué importa más que el resultado.** Con las
redes correctamente regularizadas, las **38 configuraciones** de las dos
rejillas (4 generadores × 5 profundidades, y × 6 porcentajes) caben en un
rango de MAE de **0.000022**: el **3% de un error estándar**. Todas
aterrizan en el mismo punto, y ese punto es el **predictor constante**
—predecir la media por banco, sin mirar la ventana de entrada— que da
0.011763 en test. Entrenar con 24 años de historia sintética da lo mismo
que entrenar con 4,6 años reales, que da lo mismo que entrenar **sin un
solo dato real** (`pct=1.0` → 0.011748).

Este informe documenta además **cómo se llegó primero a la conclusión
contraria**. Sin regularización las redes memorizaban el entrenamiento
(hasta 344 parámetros por muestra de entrenamiento), y las diferencias
entre configuraciones —que parecían decir "el Ruido gana", "los sintéticos
mejoran un 0,3%"— eran diferencias en *cuánto sobreajustaba cada una*, no
en cuánta señal extraía. Al impedir la memorización, esas diferencias
desaparecieron: la amplitud de la rejilla cayó de 0.000396 a 0.000022. Ver
§6.4.

## 1. El problema financiero

Un predictor de retorno diario que use *features* de microestructura
intradía (volatilidad realizada, retorno de apertura/cierre, rango) en
lugar de solo el precio de cierre suele generalizar mejor — pero esas
*features* solo se pueden calcular con datos de alta frecuencia, y los
proveedores de datos intradía (EOD Historical Data incluido) solo cubren
los últimos años, mientras que el precio de cierre diario de un banco
cotizado puede tener 30+ años de historia real.

Este proyecto responde a: **¿compensa rellenar esa historia "perdida" con
datos sintéticos de microestructura, generados a partir de lo poco que sí
tenemos real, para entrenar una red mejor?** Y si compensa, **¿con qué tipo
de generador compensa más?**

La respuesta se construye con datos reales de principio a fin:
- **Retorno diario real** de 25 bancos de EEUU, hasta 36 años (1990-2026),
  del dump de Norgate (`datos/norgate_bancos_us_export_20260602_1043.zip`).
- **Barras de 5 minutos reales** de hasta 150 bancos de EEUU, ~5,5 años
  (2020-11 a 2026-08 — todo lo que sirve el endpoint `/intraday`, que no
  tiene profundidad anterior a finales de 2020), descargadas de la API de
  [EOD Historical Data](https://eodhd.com/)
  con la key del aula (`datos/APkey`, **no está en el repo** — ver
  [§9 Entorno](#9-entorno)).

## 2. Los 4 modelos generativos (y por qué estos)

El enunciado pide 3 tipos de modelo generativo distintos "de los vistos en
clase" + 1 modelo simple. `Material_clase/` trae notebook completo de 2
técnicas (GAN, Gaussiana), y la propia teoría del taller
(`2026_Taller_Generativos.pdf`, diapositivas "GANs vs RBIG") compara esas
GAN contra **RBIG** — el método del propio profesor del taller (Valero
Laparra) — como tercera alternativa, con resultados cuantitativos incluidos
en la diapositiva. El modelo simple obligatorio ("que coja datos originales
y les añada ruido") es literalmente una celda de `Taller_GANs.ipynb`
("Ejemplo muy tonto (datos con ruido)").

| Generador | Origen | Idea |
|---|---|---|
| **Ruido** | `Taller_GANs.ipynb`, celda "ejemplo muy tonto" | Reutiliza muestras reales + ruido gaussiano proporcional a la escala de cada variable. No optimiza nada: es el suelo de referencia. |
| **Gaussiana multivariante** | `Taller_Gaussian_solution.ipynb` | Ajusta `N(μ, Σ)` sobre el vector conjunto y muestrea de ella. Se usa *shrinkage* de Ledoit-Wolf sobre Σ (mismo modelo, estimador más robusto que el `np.cov` de clase). |
| **RBIG** (Rotation-Based Iterative Gaussianization) | Teoría del taller, comparado con GAN en las diapositivas | Alterna gaussianización marginal (vía función de distribución empírica) + rotación ortogonal aleatoria, hasta que los datos son ≈ N(0,I). Implementado desde cero en `src/generators.py` (no hay paquete `rbig` en PyPI). |
| **GAN** | `Taller_GANs.ipynb` | Generador/discriminador densos, entrenamiento adversarial por lotes con *ratio* adaptativo D/G — arquitectura idéntica en espíritu a la de clase. |

**Diferencia con el recipe de clase**: en clase, el GAN/Gaussiana generan
directamente ventanas `(X, Y)` de retornos y se comparan añadiendo distintas
cantidades de muestras sintéticas *completas* a un `train_test_split`
aleatorio. Aquí, en cambio, los 4 generadores aprenden la distribución
conjunta real `[retorno_diario, volatilidad_realizada, retorno_apertura_30m,
retorno_cierre_30m, rango_intradía]` sobre la ventana real (2020-11 →
2025-06, ~4,6 años; val y test quedan fuera) (todos
son generadores **incondicionales** — no cambia el mecanismo de
entrenamiento entre ellos), y **el retorno diario ya conocido de cada día
histórico (real, de Norgate) se usa para condicionar por *conditional
matching*** qué muestra sintética de features intradía le corresponde (ver
§4). Es una extensión del material de clase a un problema nuevo, no una
copia del ejercicio: replicar el recipe de clase tal cual sobre estos datos
no aportaría nada que el ejercicio original no mostrara ya.

## 3. Los dos universos de bancos

| | `PREDICTOR_TICKERS` | `GENERATOR_TICKERS` |
|---|---|---|
| Nº bancos | 25 | hasta 150 |
| Para qué | Backbone de 30 años del predictor final | Pool de entrenamiento de los 4 generadores |
| Requisito | Retorno diario real completo desde 1990 en Norgate | Solo necesitan datos en la ventana real (2020-11 en adelante) |
| Selección | Bancos EEUU (`domicile == "United States Of America"`), activos, `Diversified Banks`/`Regional Banks`, ordenados por `shares_outstanding`, con `first_quoted_date == 1990-01-02` **y** cobertura real verificada en `export.bank_prices_daily` (2 candidatos con "primera fecha" 1990 pero solo 352 filas reales en el dump — descartados, ver `src/config.py`) | Mismo filtro de universo pero sin exigir historia larga — más bancos (incluso poco líquidos o intervenidos) dan un pool de entrenamiento más rico para los generadores. Un banco entra si tiene ≥60 sesiones intradía válidas en la ventana real; unas pocas filas (<0.1%) con valores imposibles — tickers en proceso de quiebra/exclusión con cruces de precio erráticos, no volatilidad de mercado real — se descartan por cordura (`config.POOL_MAX_*`). |

`PREDICTOR_TICKERS ⊂ GENERATOR_TICKERS` siempre.

## 4. Pipeline (`notebooks/00` → `05`)

```
00_descarga_datos          Norgate (duckdb) + EODHD (API, cacheado) -> pool real
        |                  conjunto [retorno, features intradía] + cobertura
        v
01_eda_intradia             "estudiar la distribución a lo largo del día":
        |                    forma de U de la volatilidad/volumen intradía,
        |                    y por qué la vol. realizada NO es redundante
        |                    con el retorno diario (correlación ~0.45)
        v
02_modelos_generativos      Entrena Ruido/Gaussiana/RBIG/GAN sobre el pool
        |                    real (excluyendo val+test); diagnóstico:
        |                    real vs. sintético por variable + distancia
        |                    de correlación
        v
03_backfill_condicional     Por cada generador, rellena ~24 años de
        |                    volatilidad SINTÉTICA condicionada al retorno
        |                    diario REAL (conditional matching); construye
        |                    4 datasets de 30 años (ventanas X/Y)
        v
04_entrenamiento_predictor  (a) elige arquitectura con SOLO la ventana
        |                    real disponible, midiendo en VALIDACIÓN;
        |                    (b) fija esa arquitectura y compara la rejilla
        |                    profundidad histórica × generador en TEST
        v
05_analisis_resultados      Tablas/gráficas finales: ¿mejora con más
                             sintéticos?, ¿qué generador gana?, ¿se
                             corresponde con qué generador reconstruye
                             mejor la distribución real (notebook 02)?
```

Los notebooks importan todo de `src/` (nunca redefinen lógica): las
arquitecturas de red están en `src/modelos.py` — **no se tocan sus
hiperparámetros**, se pasan como argumentos desde el notebook al llamar a
`build_predictor_cnn(...)`, `GANGenerator(epochs=..., ...)`, etc.

### Ventanas / fechas (`src/config.py`)

```
1996-05-29 ─────────────── 2020-11-02 ── 2025-06-01 ── 2025-12-01 ── 2026-05-29
│ ~24 años, retorno real + vol. SINTÉTICA │ ~5,5 años reales (5 min de EODHD)  │
└───────────── entrenamiento (segun synth_years) ─────┘   VAL      │   TEST    │
                                           ~4,6 años
```

La frontera 2020-11-02 **no es una elección de diseño**: es la primera
sesión que sirve el endpoint `/intraday` de EODHD. Todo el intradía real
descargado se usa como real; solo se sintetiza lo que de verdad no existe.

El entrenamiento **siempre termina en `VAL_START_DATE`**: val (~6 meses) y
test (~6 meses) se comen el último año de la ventana real, así que la
ventana "solo reales" (`synth_years=0`) para entrenar es ~4,6 años —
`synth_years` cuenta cuántos años de backfill sintético se añaden ANTES de
`REAL_INTRADAY_START_DATE`, no cuántos años totales de entrenamiento hay
(ver `src/train_utils.py::slice_by_depth`).

`X` = ventana de 60 días de `[retorno diario, volatilidad realizada]` por
banco (50 canales = 2 × 25 bancos); `Y` = retorno del **día siguiente**
por banco. Los generadores del notebook 02 **nunca ven** datos desde
`VAL_START_DATE` en adelante (ni para entrenarse ni indirectamente vía
estadísticos) — ni validación ni test se contaminan.

El notebook 04 mantiene separados los dos usos de la muestra real final:
la arquitectura se selecciona con el tramo de **validación**
(`[VAL_START_DATE, REAL_TEST_HOLDOUT_START_DATE)`), y el tramo de **test**
(`[REAL_TEST_HOLDOUT_START_DATE, DAILY_END_DATE]`) se usa después, una vez
fijada la arquitectura, para comparar los datasets reales/sintéticos. Por
eso el notebook 04 genera `04_comparacion_arquitecturas.csv` con métricas de
validación (`split=validation`), mientras que
`04_resultados_rejilla_profundidad.csv` y las tablas del notebook 05 contienen
métricas de test.

### Sobre la métrica: MAE como *loss*, no solo como número final

Los notebooks de clase compilan con `loss='mse'` y reportan MAE después de
entrenar, como métrica secundaria. La diapositiva de teoría del taller
(`2026_Taller_Generativos.pdf`, "REAL PROBLEM", pág. 11-12 — el caso
LST/IASI que motiva todo el ejercicio) dice, en cambio:

> **Learning: minimize MAE**, `min_θ D(ŷ, y)`

y sus resultados (pág. 27-28, "Errors: Only real data = 4.81K...") se
reportan en la unidad NATURAL del target (Kelvin) — la ventaja práctica de
entrenar con MAE en vez de MSE, que queda en unidades al cuadrado y no es
directamente interpretable. La propia teoría del taller recomienda MAE
como función de aprendizaje para el problema real que lo motiva.

Para nuestro problema esto no es un detalle menor: un retorno diario es
heavy-tailed (notebook 02: la Gaussiana no reproduce el pico leptocúrtico
de la distribución real). Entrenar con MSE dejaría que los pocos días de
retorno extremo dominen el gradiente; MAE trata cada día por igual. Por eso
el notebook 04 entrena con `loss='mae'` (parámetro `LOSS_FUNCTION` en el
notebook, `loss` en `build_predictor_*` — nunca hardcodeado en
`src/modelos.py`, ver la nota "no modificar hiperparámetros aquí" del
propio fichero).

Además, una MAE **pooleada** sobre los 25 bancos a la vez queda dominada
por los de mayor volatilidad (GBCI es ~1.6x más volátil que JPM, notebook
01) — la comparación final de `Taller_con_Datos_SP500_promedio.ipynb` es
precisamente un desglose **por banco** ("per-ticker MAE"), que es lo que
reproduce el notebook 04 (`04_mae_por_banco.csv`/`.png`). Se añade también
**precisión direccional** (`% de días con el signo del retorno acertado`,
0.5 = azar) como métrica adicional específica de finanzas: el MAE mide
error de magnitud, pero para un "predictor de precios" también importa si
acierta la dirección.

### Sobre la convergencia: `EarlyStopping` con paciencia alta

El notebook 04 entrena cada modelo con `EarlyStopping` (monitoriza
`val_loss`, se queda con los mejores pesos vistos) en vez de un número fijo
de epochs — pero con `patience=100`: hace falta que `val_loss` lleve 100
epochs SEGUIDAS sin mejorar antes de parar. Eso asegura que la curva de
loss llega con un tramo largo y plano, la evidencia visual de convergencia
que pide el enunciado, y no solo "dejó de mejorar hace poco". Los techos de
epochs (`EPOCHS_ARQUITECTURA`, `EPOCHS_REJILLA`) se dejan holgados para que
sea `EarlyStopping`, no el techo, quien decida cuándo parar.

## 5. Lógica financiera: por qué el diseño aguanta

Antes de los resultados, la pregunta que importa: **¿hay fuga de
información (look-ahead bias) en algún punto del pipeline?** Repaso
explícito, causal, día a día.

**1. La ventana `X`/`Y` no tiene fuga.** Para una fecha "hoy" = día `t-1`:
`X` es la ventana de 60 días `[t-61, ..., t-1]` (retorno + volatilidad
realizada, ambos ya CERRADOS y conocidos al final de `t-1`); `Y` es el
retorno de `t` (el día siguiente, aún no observado). El modelo nunca ve
nada de `t` para construir `X` (`features.build_xy_windows`).

**2. El backfill sintético tampoco tiene fuga hacia el target — y NO es un
`.bfill()` de pandas.** "Backfill" aquí es "rellenar historia pasada", no
el método de pandas que propaga hacia atrás el *siguiente* valor conocido
(que sí sería sospechoso: usaría, p.ej., un dato de 2025 para describir
1998). Lo que hace `src/backfill.py::conditional_match_sample` para un día
histórico `t-1` sin barras de 5 min reales es: tomar el **retorno REAL ya
conocido de ESE MISMO día `t-1`** (Norgate, contemporáneo, no del futuro) y
usarlo para consultar, en el pool de pares `(retorno, volatilidad)`
aprendido en la ventana real de 2024-2025, qué volatilidad es plausible
para un retorno de esa magnitud. Ningún dato de 2024-2025 se copia
literalmente a 1998; solo se usa la RELACIÓN aprendida ahí, aplicada al
retorno propio de 1998. Ni el target (el retorno de `t`) ni ningún dato
posterior a `t-1` interviene en absoluto.

*¿Por qué no, entonces, un `.ffill()`/`.bfill()` literal (propagar el
último o el próximo valor real conocido)?* Porque dejaría una volatilidad
**constante durante ~24 años**, ciega a la puntocom, 2008 o el COVID. `03_backfill_serie_temporal_JPM.png` muestra que
el *conditional matching* reproduce picos de volatilidad justo en esos años
de crisis — porque usa el retorno real de cada día, que sí las capta. Un
`.ffill()` destruiría esa señal.

**3. Lo que el backfill SÍ implica: la feature sintética es menos
informativa que la real, no solo "aproximada".** En los días reales, la
volatilidad realizada se mide de forma independiente del retorno diario
(vienen de fuentes distintas: barras intradía vs. cierre-a-cierre) y solo
están correlacionadas (~0.45, notebook 01) — el residuo es información
genuina. En los días sintéticos, la volatilidad se **deriva** del propio
retorno de ese día (vía *conditional matching*), así que, por construcción,
lleva menos información marginal que no esté ya en el canal de retorno. Es
una limitación real del método: la ventaja esperable de la profundidad
histórica sintética probablemente venga más de darle al modelo **más
ejemplos de la relación retorno-pasado → retorno-futuro** (más contexto de
mercado, más regímenes, más crisis vistas) que de aportar señal nueva vía
la volatilidad intradía en sí en esos años. Es "más sintéticos = más
contexto histórico con una feature de volatilidad plausible pero derivada",
no "más sintéticos = más información intradía real".

**4. Separación temporal estricta, sin excepciones.** Los generadores
(notebook 02) solo ven el pool real hasta `VAL_START_DATE`; ni validación
ni test entran en su entrenamiento, ni siquiera indirectamente vía
estadísticos agregados. El entrenamiento del predictor (notebook 04) para
CUALQUIER profundidad (`synth_years`) termina siempre en `VAL_START_DATE`;
val y test son exactamente los mismos ~6+~6 meses reales para las 4
versiones del generador, así que la comparación entre generadores es
"misma arquitectura, mismos datos de evaluación, distinto backfill" — la
única variable que cambia.

**5. Sesgo de supervivencia — limitación reconocida, no oculta.** El
universo de 25 bancos (`PREDICTOR_TICKERS`) son bancos **activos hoy**
(`delisted=False`); los bancos que quebraron (SVB, Signature Bank, First
Republic, marzo 2023) no están. El predictor se evalúa solo sobre bancos
que sobrevivieron 30+ años — un sesgo de supervivencia estándar y conocido
en ML financiero, que probablemente hace que el problema sea algo más fácil
(o al menos distinto) que predecir sobre el universo completo
punto-en-el-tiempo.

## 6. Resultados

Los 6 notebooks (`00` → `05`) están **ejecutados de principio a fin, en
orden, contra datos 100% reales** (universo completo de 150 bancos para
los generadores, 25 para el predictor). Todas las cifras de esta sección
están citadas literalmente de `reports/tables/` y `reports/figures/`.

### 6.1 Los generadores con los hiperparámetros de clase (notebooks 00-03)

> Esta subsección refleja la ejecución con los hiperparámetros **de
> partida** (los del material de clase). La búsqueda de §6.2 los mejora
> sustancialmente, así que las cifras de aquí sirven como **punto de
> comparación**, no como resultado final.

- **`01_perfil_intradia_volatilidad.png`**: forma de "U" clásica en JPM y
  GBCI (alta al abrir 9:30 ET, mínima a mediodía, repunta al cerrar).
- **`01_relacion_retorno_vs_realized_vol.png`**: correlación
  |retorno diario| vs. volatilidad realizada ≈ 0.44-0.46 — relacionadas
  pero lejos de ser la misma variable (justifica el backfill).
- **`02_real_vs_sintetico_por_generador.png`**: Ruido y RBIG reproducen
  las 5 marginales casi exactamente; la Gaussiana captura la forma general
  pero no el pico leptocúrtico de los retornos/volatilidad reales (las
  colas pesadas de los datos financieros son precisamente lo que una
  Normal no puede representar — es la limitación que motiva RBIG).
- **`02_rbig_convergencia.png`**: el exceso de curtosis medio de RBIG baja
  de ~0.53 a ~0.03 en 20 iteraciones — convergencia real hacia una Normal
  conjunta.
- **`02_calidad_correlacion_generadores.csv`**: distancia de Frobenius
  entre la matriz de correlación real y la sintética (menor = mejor) —
  **Gaussiana 0.18 < RBIG 0.24 < Ruido 0.25 < GAN 1.64**. El GAN es aquí el
  que peor reproduce la correlación conjunta — pero **NO por una limitación
  inherente**: con los hiperparámetros que encuentra la búsqueda (§6.2) baja
  a **0.36**, a la altura de los demás. La primera versión de este informe
  atribuía ese 1.27 a "una limitación conocida de los GAN vainilla en baja
  dimensión"; el experimento lo desmiente y la explicación correcta es
  simplemente que la configuración de clase no está ajustada para este
  problema.
- **`03_backfill_serie_temporal_JPM.png`**: la volatilidad sintética de
  JPM (~24 años) muestra picos claros en 2001-02, 2008-09 y 2020 —
  **coherentes con crisis reales** (dot-com, financiera, COVID) porque el *conditional matching* usa el retorno diario
  REAL de esos días, no una serie inventada.
- **`03_continuidad_empalme.csv`**: el nivel medio de volatilidad
  sintética justo antes de 2020-11 es **1.17-1.30×** el nivel real justo
  después (Gaussiana 1.17, RBIG 1.17, Ruido 1.20, GAN 1.30). Conviene ser
  literal: un ratio de 1.0 sería "sin salto"; lo que hay es un sesgo
  sistemático de nivel del 17-30%, más marcado en el GAN.

### 6.2 Búsqueda de hiperparámetros de los generadores

> **Sobre el tamaño de la búsqueda.** Una versión anterior de este informe
> titulaba esta sección "7.426 evaluaciones". Ese número contaba
> *evaluaciones*, no cobertura: al agrupar por configuración distinta eran
> 2.971 de Ruido, **3** de Gaussiana, **11** de RBIG y 707 de GAN — la
> Gaussiana se evaluó 2.928 veces sobre un espacio de tres configuraciones.
> La búsqueda actual, con el pool nuevo (141.065 filas en vez de 53.282),
> son **512 evaluaciones**: Ruido 168, Gaussiana 158 (espacio completo),
> RBIG 146 (espacio completo), GAN 40. Las conclusiones 2 y 3 de abajo
> descansan sobre espacios barridos por completo; la 1, sobre 40
> configuraciones de GAN en vez de 707, así que es la que menos apoyo
> tiene ahora.

Los generadores no se dejaron con los hiperparámetros de clase: se hizo una
búsqueda aleatoria paralelizada sobre **arquitectura e hiperparámetros**,
midiendo la fidelidad de la distribución conjunta sintética frente a datos
reales no vistos con tres métricas complementarias — **MMD** (kernel RBF,
captura marginales *y* dependencia), **Wasserstein-1** medio sobre las
marginales, y distancia de **Frobenius** entre matrices de correlación.
Código en `scripts/hp_search_generators.py`; selección con la regla de
"un error estándar" en `scripts/analizar_hpsearch.py`.

| Generador | Configuración elegida | MMD | W1 | Frobenius |
|---|---|---|---|---|
| Ruido | σ=0.019, relativo, ruido **t-Student** (4 gl) | 0.000000 | 0.0211 | 0.332 |
| Gaussiana | sin shrinkage, marginal **`rank_gauss`** | 0.004142 | 0.0195 | 0.287 |
| RBIG | **n_iters=100, grid=800, rotación PCA** | 0.000000 | 0.0225 | 0.144 |
| GAN | latent=48, 2000 ép., bs=128, lr=3e-4, **d_steps=5** | 0.001516 | 0.0776 | 0.300 |

Tres resultados que corrigen afirmaciones de la versión anterior de este
informe. Los tres se han **replicado** con la búsqueda nueva sobre el pool
ampliado, con magnitudes distintas a las de entonces:

1. **El colapso de modo del GAN era cuestión de hiperparámetros, no una
   limitación inherente.** Con la configuración de clase la distancia de
   Frobenius es 1.64; la mejor configuración de la búsqueda baja a **0.21**.
   El parámetro decisivo es cuántos pasos de discriminador se dan por cada
   paso de generador, y el efecto es monótono hasta saturar en 5:

   | `d_steps_per_g` | 2 | 3 | 4 | **5** | 6 |
   |---|---|---|---|---|---|
   | W1 medio | 1.024 | 0.397 | 0.248 | **0.163** | 0.341 |
   | Frobenius medio | 2.136 | 1.456 | 0.746 | **0.539** | 1.017 |

2. **La Gaussiana mejora 8.8× en MMD ajustándose a la cópula.** El marginal
   `rank_gauss` (llevar cada columna a N(0,1) por su distribución empírica,
   ajustar ahí la Normal y deshacer la transformación) baja el MMD medio de
   0.0771 a 0.0088 y el W1 medio de 0.304 a 0.019 — un factor 16 en este
   último. La lectura financiera es limpia: una Normal **no** puede
   representar las colas pesadas de los retornos, pero **sí** su estructura
   de dependencia; separando ambas cosas, el modelo gaussiano deja de ser el
   peor con diferencia.

   Matiz que conviene no ocultar: `rank_gauss` mejora las marginales y el
   MMD, pero **empeora** la distancia de Frobenius media (0.66 → 0.83). La
   transformación de cópula ayuda a la forma de cada variable, no a la
   matriz de correlación.

3. **RBIG tiene un óptimo, no mejora monótonamente.** El W1 medio toca
   fondo hacia 50-60 iteraciones (0.026) y vuelve a subir a 100 (0.036),
   porque cada iteración añade error de interpolación de la rejilla de
   cuantiles. (La búsqueda nueva no muestreó por encima de 100 iteraciones,
   así que la degradación más allá de ese punto queda sin reverificar.)

### 6.3 El experimento clave: ¿predice la fidelidad la utilidad?

Toda la búsqueda anterior optimiza una cosa (*¿se parece el sintético al
real?*) que **no es** la pregunta del proyecto (*¿ayuda el sintético a
predecir?*). Son criterios distintos y podían discrepar, así que se
contrastó directamente: se cogieron 6 GAN que cubren **tres órdenes de
magnitud** de calidad distribucional y, para cada uno, se recorrió el
pipeline completo — entrenar GAN → muestrear → backfill de ~24 años →
entrenar el predictor → medir en el **mismo test real**
(`scripts/experimento_espectro_gan.py`).

| GAN | MMD | test MAE | precisión direccional |
|---|---|---|---|
| buena_1 | 0.00111 | **0.011774** | 51.5% |
| buena_2 | 0.00152 | 0.011761 | 51.9% |
| buena_3 | 0.00185 | **0.011780** ← el peor | 51.2% |
| intermedia | 0.03558 | 0.011769 | 52.5% |
| **mala** | 0.30045 | **0.011753** ← el mejor | **52.9%** |
| muy_mala | 0.89109 | 0.011769 | 52.0% |

**El mejor resultado aguas abajo lo da el GAN etiquetado como "malo"**, con
un MMD 270× peor que el mejor. Pero la lectura correcta **no** es "peor
generador, mejor predictor" — eso sería sobreinterpretar 6 puntos. Es esta:
mientras el MMD varía un factor **800×**, todos los MAE caben en un
**0.23%** de dispersión, y los seis rodean al predictor constante
(0.011763). **La calidad distribucional del generador es irrelevante para
el rendimiento final.**

> ⚠️ Esta tabla es la única de §6 que sigue entrenando su predictor **sin
> regularización** (`scripts/experimento_espectro_gan.py` construye una
> `cnn_3bloques` sin dropout ni L2, en vez de la arquitectura y el `REG` del
> notebook 04). No invalida la conclusión —la dispersión ya es
> despreciable— pero para ser estrictamente comparable con §6.4 habría que
> alinear ese script con el resto del pipeline.

*¿Por qué?* Encaja con la limitación estructural documentada en §5.3: la
volatilidad sintética se **deriva** del retorno real de cada día vía
*conditional matching*. Por bien que el generador imite la distribución
conjunta, la feature resultante aporta poca información marginal que no
esté ya en el canal de retorno. La ganancia frente a "solo reales" viene
de **tener más contexto histórico** con el que entrenar, no de la calidad
del sintético.

**Consecuencia metodológica**: el generador del pipeline final se elige
por rendimiento aguas abajo, no por MMD. Y la misma cautela aplica al σ
del Ruido, cuyas métricas de fidelidad lo empujan hacia σ→0, que es
memorización pura (copiar datos reales) y no generación.

### 6.4 El predictor del día siguiente: ¿ayudan los sintéticos?

**No. Nada de lo que se prueba aquí bate a predecir una constante.**

#### El modelo nulo, que faltaba

La comparación de arquitecturas incluye ahora `constante`: predecir la media
por banco del entrenamiento, ignorando por completo la ventana `X`. Es la
referencia que importa. El otro suelo, `baseline`, repite el retorno del día
anterior; como los retornos diarios son casi incorrelados en el tiempo, eso
es *activamente* peor que no predecir nada, y compararse solo contra él hace
que cualquier red parezca buena.

`04_comparacion_arquitecturas.csv`, MAE en **validación**:

| Modelo | MAE val | vs. constante | Precisión direccional |
|---|---|---|---|
| **rnn_1capa** | **0.011496** | −0.000014 | 53.1% |
| rnn_2capas | 0.011497 | −0.000012 | 53.0% |
| cnn_3bloques | 0.011498 | −0.000012 | 52.8% |
| `constante` | 0.011509 | — | 52.8% |
| cnn_1bloque | 0.011871 | +0.000362 ← peor que la constante | 51.7% |
| dense | 0.012273 | +0.000764 ← peor que la constante | 50.1% |
| `baseline` (repetir ayer) | 0.017142 | +0.005633 | 46.3% |
| `linear` | 0.020827 | +0.009317 | 50.9% |

La mejor red le gana a la constante por **0.000014**, el 2% de un error
estándar. Dos de las cinco redes son peores que ella.

#### La regularización cambió el orden — y desenmascaró a la densa

Antes de aplicar `dropout=0.3` y `l2=1e-4`, la `dense` marcaba 0.011579 y
parecía competitiva. Con regularización marca 0.012273, la peor de las
redes. Aquella cifra venía de un modelo con **394.009 parámetros
entrenándose sobre 1.145 ventanas** —344 parámetros por muestra— que
memorizaba el entrenamiento y del que `restore_best_weights` rescataba un
punto afortunado de las primeras épocas. Al impedirle memorizar se ve su
capacidad real de generalizar, que es peor que no predecir nada.

La regularización también arregló las curvas de convergencia. En los 37
entrenamientos de las rejillas, el `val_loss` sube ahora una **mediana del
+0.20%** tras su mínimo (máximo +0.85%), frente al **+7.4% de mediana y
+17.6% de máximo** sin regularizar. Ahora las curvas de
`04_loss_curvas_*.png` muestran de verdad lo que el enunciado pide ver.

#### Las dos rejillas

`04_resultados_rejilla_profundidad.csv`, test MAE a máxima profundidad
(+24 años sintéticos, 84% de filas sintéticas):

| Generador | MAE test | vs. constante | Δ vs. solo reales |
|---|---|---|---|
| RBIG | 0.011753 | −0.08% | +0.05% |
| Ruido | 0.011753 | −0.08% | +0.05% |
| GAN | 0.011755 | −0.07% | +0.04% |
| Gaussiana | 0.011766 | +0.02% | −0.06% |
| solo reales (`synth_years=0`) | 0.011759 | −0.03% | — |
| **`constante`** | **0.011763** | — | — |

Toda la rejilla de profundidad cae entre 0.011744 y 0.011766 —**amplitud
0.000022**— y la de porcentaje entre 0.011744 y 0.011768. Un error estándar,
por bootstrap de días sobre el propio test, es **0.000698**: la amplitud
completa de las 38 configuraciones es el **3% de un error estándar**.

El caso límite es el más elocuente. `pct=1.0` entrena **sin una sola fila
real** y da 0.011748-0.011757 — indistinguible de entrenar con todo real.
Si el modelo aprendiera algo de los datos, quitarle todos los reales
tendría que notarse. No se nota.

#### Por qué el informe anterior decía lo contrario

La versión previa de esta tabla daba "el Ruido gana con +0.33% de mejora" y
"RBIG empeora un 1.4%". Aquellos números salían de modelos sin regularizar,
y lo que medían era **cuánto sobreajustaba cada configuración**, no cuánta
señal extraía: con 1.145-7.074 muestras y decenas de miles de parámetros,
qué punto rescataba `restore_best_weights` dependía del azar de la
inicialización. Al regularizar, la amplitud de la rejilla cayó de 0.000396
a 0.000022 y el orden entre generadores se volvió intercambiable.

Esa es la lección metodológica del proyecto, y es más valiosa que la
respuesta a la pregunta original: **sin un modelo nulo en la tabla y sin
control del sobreajuste, un pipeline de este tipo produce rankings de
generadores que parecen significativos y no lo son.**

### 6.5 Cómo reproducir estos números

Todo lo anterior sale de ejecutar, en orden, `00` → `05` con
`jupyter nbconvert --to notebook --execute --inplace` (o abriendo cada
notebook y "Run All") sobre un kernel con `requirements.txt` instalado —
ver §9. Los notebooks ya están guardados con sus salidas; no hace falta
volver a ejecutarlos para leer los resultados, solo para reproducirlos o
cambiar hiperparámetros.

Las búsquedas de hiperparámetros y el experimento del espectro son
scripts aparte, no notebooks, porque son procesos de horas que se
paralelizan sobre todos los núcleos:

```bash
python scripts/hp_search_generators.py --minutes 400 --worker 0   # generadores
python scripts/hp_search.py --stage A --walk-forward --embargo-days 60  # predictor
python scripts/analizar_hpsearch.py                                # elegir ganadores
python scripts/experimento_espectro_gan.py                         # fidelidad vs utilidad
```

## 7. Conclusiones

**1. Los datos sintéticos no ayudan, y el experimento lo demuestra con
claridad.** Las 38 configuraciones caben en el 3% de un error estándar y
todas coinciden con el predictor constante. El caso `pct=1.0` —entrenar sin
una sola fila real— da el mismo resultado que entrenar con todo real: si el
modelo estuviera aprendiendo algo de los datos, eso tendría que notarse.

**2. La conclusión contraria del informe anterior venía de sobreajuste, no
de señal.** Con redes de hasta 394.009 parámetros sobre 1.145 ventanas, las
diferencias entre generadores medían cuánto memorizaba cada configuración.
Al añadir `dropout=0.3` y `l2=1e-4`, la amplitud de la rejilla cayó de
0.000396 a 0.000022 y el ranking se volvió intercambiable. **Ésta es la
lección metodológica principal del trabajo**: sin modelo nulo en la tabla y
sin control del sobreajuste, este tipo de pipeline produce rankings que
parecen significativos y no lo son.

**3. La explicación es estructural, no accidental.** La volatilidad
sintética se deriva, por construcción, del retorno real conocido de cada
día: su correlación con `|retorno|` es 0.452 en el tramo sintético frente a
0.455 en el real — se reproduce la relación con fidelidad, y por eso mismo
no aporta información nueva más allá del canal de retorno que el modelo ya
tiene. Peor aún, el *conditional matching* muestrea cada día de forma
independiente, así que **destruye la agrupación de volatilidad**: la
autocorrelación a un día de la volatilidad realizada es **0.088 en el tramo
sintético frente a 0.587 en el real**. Durante 24 de los 30 años, el segundo
canal de entrada tiene una estructura temporal que ningún mercado produce.

**4. La fidelidad distribucional no predice la utilidad.** Variando el MMD
del GAN un factor 800×, el MAE final varía un 0.23% (§6.3). El caso extremo
es el generador de Ruido, que obtiene MMD ≈ 0 simplemente **copiando**
muestras reales y perturbándolas: fidelidad perfecta por memorización, no
por generación. Elegir el generador por fidelidad habría sido elegirlo por
el criterio equivocado.

**5. Sobre el tamaño de las búsquedas, con precisión.** Una versión anterior
titulaba §6.2 "7.426 evaluaciones". Ese número contaba evaluaciones, no
cobertura: la Gaussiana se evaluó 2.928 veces sobre un espacio de **tres**
configuraciones, y RBIG 468 veces sobre **once**. Conviene reportar
configuraciones distintas, no evaluaciones.

**6. Lo que este trabajo no puede responder.** Con un test de 122 días y 25
bancos que correlacionan 0.715 entre sí, el tamaño muestral efectivo está
mucho más cerca de 122 observaciones que de 3.050. Un error estándar del MAE
es 0.000698 — el 6% del propio MAE. Cualquier efecto menor que eso es
indetectable con estos datos, y el efecto que buscábamos resultó ser al
menos 30 veces más pequeño.

## 8. Estructura del repositorio

```
├── README.md
├── requirements.txt
├── Material_clase/            material docente (GAN, Gaussiana, teoría RBIG)
├── datos/
│   ├── APkey                  API key de EODHD (gitignored, NUNCA subir)
│   ├── *.zip                  dump Norgate (gitignored, pesa 243 MB)
│   ├── raw/eodhd_5m/*.parquet cache de barras de 5 min (gitignored)
│   └── interim/                datasets intermedios (gitignored)
├── notebooks_src/*.py         fuente "Jupytext" (celdas `# %%`) de cada notebook
├── notebooks/*.ipynb          notebooks ejecutados, con resultados (fuente en notebooks_src/)
├── src/
│   ├── config.py               universos, fechas, ventanas, rutas
│   ├── data_norgate.py         precios/retornos diarios (DuckDB)
│   ├── data_eodhd.py           descarga + cache de barras de 5 min
│   ├── features.py             ventanas X/Y, features intradía, pool condicional
│   ├── generators.py           Ruido / Gaussiana / RBIG / GAN
│   ├── backfill.py             conditional matching + construcción del dataset final
│   ├── modelos.py               arquitecturas de red (predictor + GAN)
│   ├── train_utils.py          mezcla real/sintético, rejilla, métricas
│   └── plotting.py             estilo y gráficas compartidas
├── reports/
│   ├── figures/*.png            todas las gráficas generadas (versionadas)
│   └── tables/*.csv             todas las tablas generadas (versionadas)
└── scripts/
    ├── py_to_ipynb.py           conversor .py (celdas `# %%`) -> .ipynb
    ├── hp_search_generators.py  busqueda de hiperparametros de los 4 generadores
    ├── hp_search.py             busqueda del predictor (walk-forward + purga)
    ├── analizar_hpsearch.py     elige ganadores con la regla de 1 error estandar
    └── experimento_espectro_gan.py   ¿predice la fidelidad la utilidad? (§6.3)
```

## 9. Entorno

```bash
pip install -r requirements.txt
```

En Windows, `pip install tensorflow` (o `torch`) puede fallar con
`OSError: [Errno 2] ... file name too long` — el límite de 260 caracteres
de ruta de Windows chocando con las rutas internas del paquete (más
probable cuanto más larga sea la ruta del proyecto). Tres salidas:

1. Un entorno **conda** con una ruta corta suele bastar sin tocar el
   registro, porque `envs/<nombre>/...` es mucho más corto que la ruta del
   repo bajo `Desktop\...`:
   ```bash
   conda create -n taller_gen python=3.11 -y
   conda activate taller_gen
   pip install -r requirements.txt
   ```
2. Activar *long paths* de Windows
   (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`,
   requiere admin).
3. WSL o Google Colab (TensorFlow viene preinstalado; sube `datos/APkey` y
   el zip de Norgate, o móntalos desde Drive).

La API key de EODHD vive solo en `datos/APkey` (gitignored) y se lee con
`src.config.load_eodhd_api_key()` — nunca se imprime ni se commitea.

### Cómo reproducir desde cero

1. `datos/APkey` con la key de EODHD; descomprimir el zip de Norgate en
   `datos/extracted/` (o dejar que `src/data_norgate.py` lo lea vía la ruta
   de `config.py`).
2. `notebooks/00_descarga_datos.ipynb` — descarga (cachea) EODHD y
   construye el pool real. Tarda ~15-20 min la primera vez (150 tickers ×
   ~5 años de barras de 5 min); las siguientes ejecuciones usan la caché.
3. `01` → `02` → `03` → `04` → `05` en orden. `02` y `04` requieren
   TensorFlow; `04` es el más lento (varios modelos entrenados hasta
   convergencia con `EarlyStopping` de paciencia alta — ver §4).

## 10. Limitaciones y trabajo futuro

- El *conditional matching* (vecino ponderado por kernel gaussiano sobre
  el retorno) es una aproximación a la muestra condicional `features |
  retorno`, no una condicional exacta — es deliberadamente el MISMO
  mecanismo para los 4 generadores, para que la comparación del notebook
  04 mida solo la calidad de cada generador.
- El backfill asume que la relación `(retorno diario, features intradía)`
  aprendida en 2024-2026 es representativa de 1996-2024; es la hipótesis
  de trabajo central del proyecto, no un hecho verificado independientemente.
- La rejilla de arquitecturas (notebook 04) usa los hiperparámetros por
  defecto de `src/modelos.py`; con más tiempo de cómputo valdría la pena
  una búsqueda más fina (learning rate, tamaño de ventana, nº de capas).
- El generador de la GAN termina en `activation='tanh'`, con rango útil
  real en `[-1, 1]` — pero nuestras columnas valen típicamente 0.01-0.03 en
  magnitud, muy por debajo de eso. `GANGenerator.fit()` reescala cada
  columna por su percentil 99.5 de `|valor|` antes de entrenar (así el
  generador aprovecha el rango completo de `tanh`) y deshace el reescalado
  en `.sample()` — práctica estándar en GANs, no cambia el modelo, solo la
  escala en la que opera.
- **El GAN entrena con pasos manuales de `tf.GradientTape`** (ver
  `src/generators.py::GANGenerator`) en vez del truco clásico de Keras 1/2
  de congelar el discriminador y compilar un modelo combinado
  (`Taller_GANs.ipynb`): ese truco depende de que Keras fije la lista de
  variables entrenables al compilar y no la actualice después, algo que
  Keras 3 ya no garantiza. Con `GradientTape` se piden gradientes solo de
  las variables del discriminador (paso D) o solo de las del generador
  (paso G), sin depender de esa gestión interna de `.trainable`.
- **El colapso de modo del GAN se resolvió con hiperparámetros.** Con la
  configuración de clase el generador colapsa (produce casi el mismo punto
  sin importar el ruido de entrada) y la distancia de Frobenius es 1.27.
  La búsqueda (§6.2) lo lleva a **0.36**, al nivel de RBIG. El parámetro
  decisivo es `d_steps_per_g` — cuántos pasos de discriminador se dan por
  cada paso de generador — con mejora monótona (W1 medio 1.20 → 0.82 →
  0.66 → 0.37 → **0.27** al pasar de 1 a 5 pasos) y saturación en 5. Más
  epochs no ayuda una vez ahí. Conviene señalar el error de método que
  esto destapó: en una primera pasada se recortó `d_steps=5` del espacio
  de búsqueda "por coste", y resultó ser justo el parámetro más
  influyente; el óptimo quedaba fuera del espacio explorado. Cuando el
  mejor resultado cae en el **borde** de la rejilla de búsqueda, casi
  siempre significa que la rejilla es demasiado estrecha, no que se haya
  encontrado el óptimo.
- **Pero nada de eso mejoró el predictor** (§6.3): la calidad
  distribucional del generador resultó no predecir el rendimiento aguas
  abajo. Es la limitación más importante de todo el trabajo, y es
  estructural: la volatilidad sintética se deriva del retorno real de cada
  día, así que aporta poca información marginal por muy bien modelada que
  esté.
