# Taller B5-T1 · Generación de datos financieros sintéticos

**Predecir el retorno del día siguiente de acciones bancarias de EEUU, usando
volatilidad intradía real (últimos 2 años, barras de 5 min de EOD Historical
Data) allí donde existe, y volatilidad intradía sintética — generada por 4
modelos generativos distintos — para reconstruir los 28 años anteriores de
los que solo tenemos precio de cierre diario (Norgate).**

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
  [§7 Entorno](#7-entorno)).

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

## 5. Resultados

### 5.1 Lo que sí se ha ejecutado y verificado en este entorno

Notebooks 00-03 no necesitan TensorFlow y se han ejecutado íntegros contra
datos reales — universo completo de 150 bancos (149 pasan el filtro de
cobertura mínima; el pool condicional final tiene 53.282 muestras reales
`[retorno, features intradía]`, de las que 25.183 quedan tras excluir
val+test); las figuras y tablas de `reports/` son reales, no ilustrativas:

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
  entre la matriz de correlación real y la sintética — Ruido (0.39) ≈
  Gaussiana (0.40) < RBIG (0.48): RBIG gana en marginales, pero el ruido y
  la Gaussiana reproducen mejor la correlación LINEAL retorno↔volatilidad.
- **`03_backfill_serie_temporal_JPM.png`**: la volatilidad sintética de
  JPM (28 años) muestra picos claros en 2001-02, 2008-09, 2020 y
  2023 — **coherentes con crisis reales** (dot-com, financiera, COVID,
  banca regional) porque el *conditional matching* usa el retorno diario
  REAL de esos días, no una serie inventada.
- **`03_continuidad_empalme.csv`**: el nivel medio de volatilidad
  sintética justo antes de 2024 es ~1.24-1.28× el nivel real justo
  después — sin salto artificial (la diferencia es coherente con que
  2022-24 incluye la crisis bancaria regional de 2023, más volátil que
  2024-26).

### 5.2 Lo que requiere TensorFlow (notebooks 04-05)

El notebook 04 (arquitectura + rejilla profundidad×generador) y el 05
(tablas/gráficas finales) **no se han podido ejecutar en esta máquina**:
`pip install tensorflow` falla aquí por el límite de "long paths" de
Windows (ver §7). El código está completo, usa exactamente las mismas
utilidades ya verificadas (`src/train_utils.py`, `src/modelos.py`) y lee
directamente de los `.npz` que deja el notebook 03 — **ejecutar
`04_entrenamiento_predictor.ipynb` y `05_analisis_resultados.ipynb` en
Colab (o en local tras resolver el problema de PATH) rellena
`reports/tables/04_*.csv`, `05_*.csv` y sus gráficas correspondientes**,
que es lo que hay que citar en la presentación.

## 6. Estructura del repositorio

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

## 7. Entorno

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

## 8. Limitaciones y trabajo futuro

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
  `Taller_GANs.ipynb`), que satura fuera de `[-1, 1]`. Los datos del pool
  ya están en esa escala de forma natural (log-retornos/volatilidad
  intradía, recortados a valores plausibles — ver `config.POOL_MAX_*`), así
  que no hace falta normalizar antes de entrenar, pero el extremo de la
  cola queda ligeramente comprimido; una capa de salida lineal sería más
  apropiada si se reentrena con otra escala de datos.
- **El GAN sobre Keras 3 no funcionaba en absoluto** hasta que se probó con
  TensorFlow real (este proyecto se desarrolló casi entero sin poder
  instalar TensorFlow localmente — ver §7): el truco clásico de
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
