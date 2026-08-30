# Taller B5-T1 · Generación de datos financieros sintéticos

**Predecir el retorno del día siguiente de acciones bancarias de EEUU, usando
volatilidad intradía real (últimos 2 años, barras de 5 min de EOD Historical
Data) allí donde existe, y volatilidad intradía sintética — generada por 4
modelos generativos distintos — para reconstruir los 28 años anteriores de
los que solo tenemos precio de cierre diario (Norgate).**

**Resultado (§6, con los 6 notebooks ejecutados end-to-end sobre datos
reales): sí ayuda.** 3 de los 4 generadores igualan o mejoran al modelo
entrenado solo con la ventana real; el mejor (Ruido, +28 años de
backfill) sube la precisión direccional del predictor de 51.4% a 53.0% y
baja el MAE un 0.33%. El cuarto (RBIG) empeora de forma clara y
explicable — la comparación entre generadores es en sí uno de los
hallazgos del proyecto, no solo un detalle técnico.

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
- **Barras de 5 minutos reales** de hasta 150 bancos de EEUU, ~2 años
  (2024-2026), descargadas de la API de [EOD Historical Data](https://eodhd.com/)
  con la key del aula (`datos/APkey`, **no está en el repo** — ver
  [§8 Entorno](#8-entorno)).

## 2. Los 4 modelos generativos (y por qué estos)

El enunciado pide 3 tipos de modelo generativo distintos "de los vistos en
clase" + 1 modelo simple. `Material_clase/` solo trae notebook completo de
2 técnicas (GAN, Gaussiana), pero la propia teoría del taller
(`2026_Taller_Generativos.pdf`, diapositivas "GANs vs RBIG") compara esas
GAN precisamente contra **RBIG** — el método del propio profesor del
taller (Valero Laparra) — como tercera alternativa, con resultados
cuantitativos incluidos en la diapositiva. Y el modelo simple obligatorio
("que coja datos originales y les añada ruido") es literalmente una celda
de `Taller_GANs.ipynb` ("Ejemplo muy tonto (datos con ruido)").

| Generador | Origen | Idea |
|---|---|---|
| **Ruido** | `Taller_GANs.ipynb`, celda "ejemplo muy tonto" | Reutiliza muestras reales + ruido gaussiano proporcional a la escala de cada variable. No optimiza nada: es el suelo de referencia. |
| **Gaussiana multivariante** | `Taller_Gaussian_solution.ipynb` | Ajusta `N(μ, Σ)` sobre el vector conjunto y muestrea de ella. Se usa *shrinkage* de Ledoit-Wolf sobre Σ (mismo modelo, estimador más robusto que el `np.cov` de clase). |
| **RBIG** (Rotation-Based Iterative Gaussianization) | Teoría del taller, comparado con GAN en las diapositivas | Alterna gaussianización marginal (vía función de distribución empírica) + rotación ortogonal aleatoria, hasta que los datos son ≈ N(0,I). Implementado desde cero en `src/generators.py` (no hay paquete `rbig` en PyPI). |
| **GAN** | `Taller_GANs.ipynb` | Generador/discriminador densos, entrenamiento adversarial por lotes con *ratio* adaptativo D/G — arquitectura idéntica en espíritu a la de clase. |

**Diferencia importante con el recipe de clase** (y con lo que pedía la
primera versión de este proyecto): en clase, el GAN/Gaussiana generan
directamente ventanas `(X, Y)` de retornos y se comparan añadiendo distintas
cantidades de muestras sintéticas *completas* a un `train_test_split`
aleatorio. **Aquí no** — replicar eso tal cual sobre nuestros datos no
añade nada que el ejercicio de clase no mostrara ya. En su lugar, los 4
generadores aprenden la distribución conjunta real
`[retorno_diario, volatilidad_realizada, retorno_apertura_30m,
retorno_cierre_30m, rango_intradía]` sobre la ventana real de 2 años (todos
son generadores **incondicionales** — no cambia el mecanismo de
entrenamiento entre ellos), y **el retorno diario ya conocido de cada día
histórico (real, de Norgate) se usa para condicionar por *conditional
matching*** qué muestra sintética de features intradía le corresponde (ver
§4). Es una extensión genuina del material de clase a un problema nuevo,
no una copia.

## 3. Los dos universos de bancos

| | `PREDICTOR_TICKERS` | `GENERATOR_TICKERS` |
|---|---|---|
| Nº bancos | 25 | hasta 150 |
| Para qué | Backbone de 30 años del predictor final | Pool de entrenamiento de los 4 generadores |
| Requisito | Retorno diario real completo desde 1990 en Norgate | Solo necesitan datos en la ventana real de 2 años |
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
03_backfill_condicional     Por cada generador, rellena 28 años de
        |                    volatilidad SINTÉTICA condicionada al retorno
        |                    diario REAL (conditional matching); construye
        |                    4 datasets de 30 años (ventanas X/Y)
        v
04_entrenamiento_predictor  (a) elige arquitectura con SOLO los 2 años
        |                    reales; (b) entrena esa arquitectura en la
        |                    rejilla profundidad_histórica × generador,
        |                    evaluando siempre en el mismo test real
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
1996-05-29 ─────────────── 2024-05-29 ── 2025-06-01 ── 2025-12-01 ── 2026-05-29
│  28 años, retorno real + vol. SINTÉTICA │  2 años reales                     │
└───────────── entrenamiento (segun synth_years) ─────┘   VAL      │   TEST    │
                                            ~1 año
```

El entrenamiento **siempre termina en `VAL_START_DATE`**: val (~6 meses) y
test (~6 meses) se comen la mitad de los 2 años reales, así que la ventana
"solo reales" (`synth_years=0`) para entrenar es en realidad solo ~1 año —
`synth_years` cuenta cuántos años de backfill sintético se añaden ANTES de
`REAL_INTRADAY_START_DATE`, no cuántos años totales de entrenamiento hay
(ver `src/train_utils.py::slice_by_depth`).

`X` = ventana de 60 días de `[retorno diario, volatilidad realizada]` por
banco (50 canales = 2 × 25 bancos); `Y` = retorno del **día siguiente**
por banco. Los generadores del notebook 02 **nunca ven** datos desde
`VAL_START_DATE` en adelante (ni para entrenarse ni indirectamente vía
estadísticos) — ni validación ni test se contaminan.

### Sobre la métrica: MAE como *loss*, no solo como número final

Los notebooks de clase compilan siempre con `loss='mse'` y reportan MAE
después de entrenar, como metrica secundaria. Pero la propia diapositiva de
teoría del taller (`2026_Taller_Generativos.pdf`, "REAL PROBLEM", pág.
11-12 — el caso LST/IASI que motiva todo el ejercicio) dice explícitamente:

> **Learning: minimize MAE**, `min_θ D(ŷ, y)`

y sus resultados (pág. 27-28, "Errors: Only real data = 4.81K...") se
reportan en la unidad NATURAL del target (Kelvin) — la ventaja práctica de
entrenar con MAE en vez de MSE, que queda en unidades al cuadrado y no es
directamente interpretable. Es decir: **la propia teoría del taller
recomienda MAE como función de aprendizaje para el problema real; el
notebook de alumno solo la usa como métrica de reporte.**

Para nuestro problema esto no es un detalle menor: un retorno diario es
heavy-tailed — lo demostramos nosotros mismos en el notebook 02 (la
Gaussiana no reproduce el pico leptocúrtico de la distribución real).
Entrenar con MSE deja que los pocos días de retorno extremo dominen el
gradiente; MAE trata cada día por igual, más robusto y más fiel al
principio de la teoría del taller que a la implementación literal del
notebook de juguete. Por eso el notebook 04 entrena con `loss='mae'`
(parámetro `LOSS_FUNCTION`, configurable — nunca hardcodeado en
`src/modelos.py`, ver la nota "no modificar hiperparámetros aquí" del
propio fichero).

También se corrige otro detalle: una MAE **pooleada** sobre los 25 bancos a
la vez queda dominada por los de mayor volatilidad (GBCI es ~1.6x más
volátil que JPM, notebook 01) — la comparación final de
`Taller_con_Datos_SP500_promedio.ipynb` es precisamente un desglose **por
banco** ("per-ticker MAE"), que es lo que reproduce el notebook 04
(`04_mae_por_banco.csv`/`.png`). Y se añade **precisión direccional**
(`% de días con el signo del retorno acertado`, 0.5 = azar) como métrica
adicional específica de finanzas: el MAE mide error de magnitud, pero para
un "predictor de precios" también importa si acierta la dirección.

## 5. Lógica financiera: por qué el diseño aguanta

Antes de los resultados, la pregunta que importa: **¿hay fuga de
información (look-ahead bias) en algún punto del pipeline?** Repaso
explícito, causal, día a día.

**1. La ventana `X`/`Y` no tiene fuga.** Para una fecha "hoy" = día `t-1`:
`X` es la ventana de 60 días `[t-61, ..., t-1]` (retorno + volatilidad
realizada, ambos ya CERRADOS y conocidos al final de `t-1`); `Y` es el
retorno de `t` (el día siguiente, aún no observado). El modelo nunca ve
nada de `t` para construir `X`. Esto es simplemente correcto por
construcción (`features.build_xy_windows`), pero merece decirse explícito
porque es la base de todo lo demás.

**2. El backfill sintético tampoco tiene fuga hacia el target — y NO es un
`.bfill()` de pandas.** "Backfill" aquí es "rellenar historia pasada", no
el método de pandas que propaga hacia atrás el *siguiente* valor
conocido (que sí sería sospechoso: usaría, p.ej., un dato de 2025 para
describir 1998). Lo que hace `src/backfill.py::conditional_match_sample`
para un día histórico `t-1` sin barras de 5 min reales es: tomar el
**retorno REAL ya conocido de ESE MISMO día `t-1`** (Norgate, contemporáneo,
no del futuro) y usarlo para consultar, en el pool de pares
`(retorno, volatilidad)` aprendido en la ventana real de 2024-2025, qué
volatilidad es plausible para un retorno de esa magnitud. Ningún dato de
2024-2025 se copia literalmente a 1998; solo se usa la RELACIÓN aprendida
ahí, aplicada al retorno propio de 1998. Ni el target (el retorno de `t`)
ni ningún dato posterior a `t-1` interviene en absoluto.

*¿Por qué no, entonces, un `.ffill()`/`.bfill()` literal (propagar el
último o el próximo valor real conocido)?* Porque haría algo mucho peor
que cualquier fuga: dejaría una volatilidad **constante durante 28 años**,
ciega a la puntocom, 2008, el COVID o la crisis bancaria de 2023. Ya hay
prueba de que esto importa: `03_backfill_serie_temporal_JPM.png` muestra
que el *conditional matching* reproduce picos de volatilidad justo en esos
años de crisis — porque usa el retorno real de cada día, que sí las
capta. Un `.ffill()` destruiría esa señal.

**3. Ojo con lo que el backfill SÍ implica: la feature sintética es menos
informativa que la real, no solo "aproximada".** En los días reales, la
volatilidad realizada se mide de forma independiente del retorno diario
(vienen de fuentes distintas: barras intradía vs. cierre-a-cierre) y solo
están correlacionadas (~0.45, notebook 01) — el residuo es información
genuina. En los días sintéticos, la volatilidad se **deriva** del propio
retorno de ese día (vía *conditional matching*) — así que, por
construcción, lleva menos información marginal que no esté ya en el canal
de retorno. Es una limitación real del método, no solo un matiz: significa
que la ventaja esperable de la profundidad histórica sintética probablemente
venga más de darle al modelo **más ejemplos de la relación retorno-pasado →
retorno-futuro** (más contexto de mercado, más regímenes, más crisis vistas)
que de aportar señal nueva vía la volatilidad intradía en sí en esos años.
Es una historia honesta y sigue siendo interesante — pero no es "más
sintéticos = más información intradía real", es "más sintéticos = más
contexto histórico con una feature de volatilidad plausible pero derivada".

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
punto-en-el-tiempo. Se declara explícitamente en vez de ignorarlo.

## 6. Resultados

Los 6 notebooks (`00` → `05`) están **ejecutados de principio a fin, en
orden, contra datos 100% reales** (universo completo de 150 bancos para
los generadores, 25 para el predictor). Todas las cifras de esta sección
están citadas literalmente de `reports/tables/` y `reports/figures/` — no
son ilustrativas.

### 6.1 Los generadores (notebooks 00-03)

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
  **Ruido 0.39 < Gaussiana 0.40 < RBIG 0.48 < GAN 1.27**. El GAN mejoró
  mucho tras el ajuste de hiperparámetros (ver §6.2) pero sigue siendo el
  que peor reproduce la correlación conjunta — una limitación conocida de
  los GAN vainilla en baja dimensión, no un fallo de la implementación
  (ver §9).
- **`03_backfill_serie_temporal_JPM.png`**: la volatilidad sintética de
  JPM (28 años) muestra picos claros en 2001-02, 2008-09, 2020 y
  2023 — **coherentes con crisis reales** (dot-com, financiera, COVID,
  banca regional) porque el *conditional matching* usa el retorno diario
  REAL de esos días, no una serie inventada.
- **`03_continuidad_empalme.csv`**: el nivel medio de volatilidad
  sintética justo antes de 2024 es ~1.24-1.28× el nivel real justo
  después — sin salto artificial.

### 6.2 El GAN sobre Keras 3: de roto a competitivo

Este proyecto se desarrolló casi entero sin poder instalar TensorFlow
localmente (ver §8); al conseguirlo, el GAN de `Taller_GANs.ipynb`
resultó no funcionar en absoluto sobre Keras 3 (el truco clásico de
congelar el discriminador no aplica ya — ver §9), y una vez arreglado
(reescrito con `tf.GradientTape`, ver `src/generators.py::GANGenerator`)
colapsaba de modo severamente. Un barrido de hiperparámetros dirigido
llevó la distancia de Frobenius real-vs-sintético de **~3.0 (colapso
severo) a ~1.27**, cambiando 3 cosas sin salir de la familia "GAN
vainilla con pérdida BCE" de clase:

| Cambio | Frobenius |
|---|---|
| Config. original (`Taller_GANs.ipynb`, lr=1e-3, 3000 epochs) | ~3.0 |
| + `learning_rate=1e-4` | ~1.3 |
| + 2 pasos de discriminador por paso de generador | ~1.25 |
| + reescalar cada columna a `[-1,1]` antes de `tanh` (ver §9) | **~1.05 en el barrido, 1.27 en la ejecución final** |

### 6.3 El predictor del día siguiente: ¿ayudan los sintéticos?

**Arquitectura ganadora** (`04_comparacion_arquitecturas.csv`, entrenada
SOLO con la ventana real de ~1 año, con `EarlyStopping`): **`cnn_3bloques`**
— exactamente `cnn_model_2` de `Taller_GANs.ipynb`, la arquitectura con la
que la propia clase compara sus generadores. Gana en MAE (0.011789) **y**
en precisión direccional (0.515, la mejor de las 7 arquitecturas
comparadas) — sin la tensión entre ambas métricas que se veía en
ejecuciones preliminares con menos epochs.

**Rejilla final** (`04_resultados_rejilla_profundidad.csv`,
`05_tabla_generador_final.csv`), test MAE / precisión direccional con
`+28` años de historia sintética añadida vs. solo la ventana real:

| Generador | MAE (+28 años) | Δ MAE vs. solo reales | Precisión direccional |
|---|---|---|---|
| solo reales (`synth_years=0`) | 0.011789 | — | 51.4% |
| **Ruido** | **0.011750** | **+0.33%** | **53.0%** |
| GAN | 0.011762 | +0.23% | 52.0% |
| Gaussiana | 0.011785 | +0.04% | 50.0% |
| RBIG | 0.011959 | **−1.4%** (empeora) | 45.5% |

**Lectura honesta**: 3 de los 4 generadores igualan o mejoran ligeramente
al modelo entrenado solo con datos reales — con el Ruido (el modelo
"simple" obligatorio del enunciado) ganando por MAE y precisión
direccional a `+28` años, y el GAN igualándolo en el punto intermedio
(`synth_years=14`, 52.9% de precisión direccional, la mejor de toda la
rejilla). La mejora en MAE es modesta (~0.3%, esperable: predecir el
signo/magnitud del retorno diario de un banco líquido es un problema
cercano a la eficiencia de mercado, no hay milagros), pero la mejora en
**precisión direccional es consistente y más fácil de interpretar**: pasar
de 51.4% (solo reales) a ~53% con el generador adecuado es una ventaja
real, aunque pequeña, sobre lanzar una moneda.

**RBIG es la excepción, y es una excepción explicable, no ruido.** Es el
único generador cuyo rendimiento se **degrada por debajo del baseline**
según se añade profundidad, y es también el que muestra la relación menos
favorable entre calidad de reconstrucción de la distribución conjunta
(notebook 02) y rendimiento final (`05_calidad_generador_vs_mae.png`): a
pesar de tener mejor distancia de Frobenius que el GAN, da peor MAE final
que los otros 3. Es exactamente el tipo de hallazgo que pide el paso 5 del
enunciado ("comparar entre los distintos tipos de modelos generativos
usados") — la calidad del generador importa, y no toda métrica de
fidelidad distribucional predice igual de bien la utilidad río abajo.

### 6.4 Cómo reproducir estos números

Todo lo anterior sale de ejecutar, en orden, `00` → `05` con
`jupyter nbconvert --to notebook --execute --inplace` (o abriendo cada
notebook y "Run All") sobre un kernel con `requirements.txt` instalado —
ver §8. Los notebooks ya están guardados con sus salidas; no hace falta
volver a ejecutarlos para leer los resultados, solo para reproducirlos o
cambiar hiperparámetros.

## 7. Estructura del repositorio

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
├── notebooks/*.ipynb          notebooks generados de notebooks_src/ (scripts/py_to_ipynb.py)
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
└── scripts/py_to_ipynb.py       conversor .py (celdas `# %%`) -> .ipynb
```

## 8. Entorno

```bash
pip install -r requirements.txt
```

**Recomendado: Google Colab** (TensorFlow viene preinstalado; sube
`datos/APkey` y el zip de Norgate, o móntalos desde Drive). En local en
Windows, `pip install tensorflow` (o `torch`) puede fallar con
`OSError: [Errno 2] ... file name too long` — es el límite de 260
caracteres de ruta de Windows chocando con las rutas internas del paquete
(mucho más probable cuanto más larga sea la ruta del proyecto), no un
problema de este proyecto. Tres salidas:
1. Activar *long paths* de Windows
   (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`,
   requiere admin).
2. Un entorno **conda** con una ruta corta suele bastar sin tocar el
   registro, porque `envs/<nombre>/...` es mucho más corto que la ruta del
   repo bajo `Desktop\...`:
   ```bash
   conda create -n taller_gen python=3.11 -y
   conda activate taller_gen
   pip install -r requirements.txt
   ```
3. WSL o Colab.

La API key de EODHD vive solo en `datos/APkey` (gitignored) y se lee con
`src.config.load_eodhd_api_key()` — nunca se imprime ni se commitea.

### Cómo reproducir desde cero

1. `datos/APkey` con la key de EODHD; descomprimir el zip de Norgate en
   `datos/extracted/` (o dejar que `src/data_norgate.py` lo lea vía la ruta
   de `config.py`).
2. `notebooks/00_descarga_datos.ipynb` — descarga (cachea) EODHD y
   construye el pool real. Tarda ~15-20 min la primera vez (150 tickers ×
   ~5 años de barras de 5 min); las siguientes ejecuciones usan la caché.
3. `01` → `02` → `03` en orden (no necesitan TensorFlow).
4. `04` → `05` (requieren TensorFlow — Colab).

## 9. Limitaciones y trabajo futuro

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
- El generador de la GAN termina en `activation='tanh'` (igual que
  `Taller_GANs.ipynb`), con rango útil real en `[-1, 1]` — pero nuestras
  columnas valen típicamente 0.01-0.03 en magnitud, muy por debajo de eso:
  sin corregir esto, el generador solo usaría una rebanada minúscula del
  rango de `tanh` cerca de 0, perdiendo resolución. `GANGenerator.fit()`
  reescala cada columna por su percentil 99.5 de `|valor|` ANTES de
  entrenar (así el generador aprovecha el rango completo de `tanh`) y
  deshace el reescalado en `.sample()` — práctica estándar en GANs, no
  cambia el modelo, solo la escala en la que opera. Confirmado
  empíricamente en el barrido de hiperparámetros (§6.2): mejora la
  distancia de Frobenius de ~1.25 a ~1.05 manteniendo el resto de
  hiperparámetros iguales.
- **El GAN sobre Keras 3 no funcionaba en absoluto** hasta que se probó con
  TensorFlow real (este proyecto se desarrolló casi entero sin poder
  instalar TensorFlow localmente — ver §8): el truco clásico de
  `Taller_GANs.ipynb` (congelar el discriminador y compilar un modelo
  combinado) depende de que Keras fije la lista de variables entrenables
  al compilar y no la actualice después; en Keras 3 se comprobó
  empíricamente que ya no es así, así que el discriminador dejaba de
  aprender del todo. Se reescribió con pasos manuales de `tf.GradientTape`
  (ver `src/generators.py::GANGenerator`, ~20x más rápido que la
  alternativa de recompilar en cada paso).
- **El GAN, ya funcionando, colapsa de modo** (mode collapse: el generador
  aprende a producir casi el mismo punto sin importar el ruido de entrada)
  en este problema de baja dimensión (d=5) — un fallo bien documentado de
  los GAN vainilla con pérdida BCE, no un error de esta implementación. Un
  barrido de hiperparámetros (learning rate, nº de pasos de discriminador
  por paso de generador, tamaño de red, nº de epochs) redujo bastante el
  problema sin cambiar de familia de modelo (seguir siendo el GAN vainilla
  de clase): con la configuración de `Taller_GANs.ipynb` (Adam por
  defecto, lr=1e-3) la distancia de Frobenius real-vs-sintético del
  notebook 02 fue **~3.0** (colapso severo); bajando a **lr=1e-4** bajó a
  **~1.3**; añadiendo además **2 pasos de discriminador por cada paso de
  generador** bajó a **~1.25** — todavía peor que RBIG/Gaussiana/Ruido
  (~0.4-0.5) pero muy lejos del colapso total. Más epochs (probado hasta
  3000) EMPEORA el colapso, no lo mejora — otro indicio de que es
  colapso de modo y no falta de entrenamiento.
