# Taller B5-T1 · Generación de datos financieros sintéticos

**Predecir el retorno del día siguiente de acciones bancarias de EEUU, usando
volatilidad intradía real (desde 2020-11, barras de 5 min de EOD Historical
Data) allí donde existe, y volatilidad intradía sintética — generada por 4
modelos generativos distintos — para reconstruir los ~24 años anteriores de
los que solo tenemos precio de cierre diario (Norgate).**

**Resultado (§6-7, con los 6 notebooks ejecutados end-to-end sobre datos
reales): un negativo, pero controlado — que es lo que le da valor.**

1. **Da igual qué generador se use.** Tras **7.428 evaluaciones**
   optimizando la fidelidad distribucional, un experimento controlado
   (§6.3) muestra que **esa fidelidad no predice la utilidad**: variando la
   calidad del generador un factor 2.266×, el rendimiento final varía un
   0.64%. El GAN mejor optimizado no bate a añadir ruido gaussiano.
2. **El ranking entre generadores es ruido de inicialización** (§6.6).
   Repitiendo la rejilla entera dos veces, la correlación de rangos entre
   ejecuciones fue **−0.10**, con saltos de hasta 6.36 puntos en la misma
   configuración. Aquí no hay un "generador ganador" que declarar.
3. **La causa raíz se encontró, se arregló y no cambió nada** (§6.8). El
   backfill destruía el clustering de volatilidad (información mutua 0.000
   vs 0.304 real); v2 lo recupera casi entero (0.075 → 0.593) y el MAE se
   mueve 5·10⁻⁷, dentro del ruido de semilla.
4. **El cuello de botella es el predictor, no el dato sintético** (§6.9).
   En un target donde la señal es demostrable (volatilidad: HAR-RV logra R²
   fuera de muestra de +0.21), **ninguna de las 42 redes probadas se
   acerca** — todas pierden contra un modelo lineal de tres coeficientes
   de 2009, con Diebold-Mariano p<0.05.

Por eso este trabajo **no afirma "los datos sintéticos no ayudan"**: afirma
que no ayudan a *este* predictor, que no aprende ni una señal que sí existe.
Separar ambas cosas exige antes un predictor que demuestre que aprende — y
ese control, que casi nunca se hace, es la aportación metodológica principal
del trabajo.

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
        |                    real disponible, midiendo en VALIDACIÓN, y
        |                    contra un predictor CONSTANTE como modelo nulo;
        |                    (b) fija esa arquitectura y compara dos rejillas
        |                    en TEST: profundidad (años) y porcentaje
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

### Sobre la convergencia: regularización primero, `EarlyStopping` después

El enunciado pide "para cada entrenamiento, incluir las curvas de loss donde
se vea que el modelo ha convergido". Conseguirlo requirió dos cosas, y la
importante no es la obvia.

**La que de verdad importa: regularizar.** Sin `dropout` ni `L2`, las redes
memorizaban —hasta 394.009 parámetros sobre 1.104 ventanas, 344 por
muestra— y el `val_loss` **subía** tras su mínimo: mediana +7,4 %, máximo
+17,6 %. Eso no es una curva convergida por mucho que se alargue el
entrenamiento. Con `dropout=0.3` y `L2=1e-4` (aplicados por igual a la
selección de arquitectura y a las rejillas, vía el diccionario `REG`), la
subida baja a una mediana del **+0,20 %** (máximo +0,85 %) en los 38
entrenamientos: curva plana de verdad.

**La secundaria: `EarlyStopping`** con `patience=20` sobre `val_loss` y
`restore_best_weights=True`, de modo que el modelo evaluado es el de la
mejor época, no el de la última. La paciencia se bajó de 100 a 20 tras
comprobar empíricamente que no aparecen mínimos tardíos: reentrenando dos
configuraciones con `patience=400`, el mejor punto apenas se movió (época 74
→ 81 y 92 → 110) y el MAE final cambió menos de un error estándar. Esperar
100 épocas sin mejora era pagar por una certeza ya medida.

**Cómo leer las curvas.** En todas ellas la validación queda por debajo del
train. No es un error ni sobreajuste invertido: el propio predictor
constante ya da train 0.0151 y val 0.0115, porque el tramo de validación
(seis meses de 2025) es más tranquilo que los ~29 años de entrenamiento. Esa
distancia la fija el reparto temporal de los datos, no el modelo. Lo único
que informa es la **forma** de la curva de validación tras su mínimo.

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

### 6.2 Búsqueda de hiperparámetros de los generadores (7.428 evaluaciones)

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

| GAN | MMD | Frobenius | test MAE | precisión direccional |
|---|---|---|---|---|
| buena_1 | 0.00111 | 0.238 | 0.011777 | 49.9 % |
| buena_2 | 0.00152 | 0.300 | **0.011776** | 50.5 % |
| buena_3 | 0.00185 | 0.212 | **0.011776** | 50.7 % |
| intermedia | 0.03558 | 1.221 | 0.011778 | 50.3 % |
| mala | 0.30045 | 2.465 | 0.011779 | 50.0 % |
| muy_mala | 0.89109 | 2.886 | 0.011780 | 50.1 % |

**Variando el MMD un factor 800×, el test MAE se mueve 0.000004** — el 0,6 %
de un error estándar. Los seis quedan además ligeramente por encima del
predictor constante (0.011762) y con precisión direccional clavada en el
50 %.

Un matiz que corrige a la versión anterior de este informe: ahora que el
experimento usa **el mismo predictor que la rejilla** (misma arquitectura,
misma regularización, misma purga), las correlaciones de Spearman entre
fidelidad y MAE salen **positivas y altas** (MMD +0.83, W1 +0.94, Frobenius
+0.83), no negativas. Es decir, el orden *sí* es el esperado: mejor
generador, mejor MAE.

Pero eso no rehabilita la fidelidad como criterio, porque **el efecto
completo es 175 veces menor que el error de medida**. Es exactamente el tipo
de correlación que parece impresionante hasta que se mira la escala del eje.
Antes salía negativa simplemente porque el ruido de entrenamiento superaba a
la señal y el signo caía al azar.

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

**Protocolo de selección** (`04_comparacion_arquitecturas.csv`, entrenando
SOLO con la ventana real disponible): la arquitectura se elige por el error
en **validación** (`val_mae`), nunca en test — el test se reserva íntegro
para la comparación de generadores. Y no por el mínimo, sino por la **regla
del un error estándar** (`tu.elegir_por_una_ee`): entre las que caen dentro
de 1 e.e. de la mejor se toma la más simple.

Aplicar esa regla aquí no es un tecnicismo, porque los números dicen algo
incómodo:

| | val_mae | e.e. | nº parámetros | test mae |
|---|---|---|---|---|
| rnn_2capas | 0.011485 | 0.000756 | 143.681 | 0.011956 |
| **rnn_1capa** ← elegida | 0.011518 | 0.000764 | **38.465** | 0.011867 |
| cnn_3bloques | 0.011563 | 0.000754 | 150.273 | 0.011811 |
| cnn_1bloque | 0.011594 | 0.000744 | 197.889 | 0.011995 |
| dense | 0.011813 | 0.000747 | 394.009 | 0.012074 |

**El error estándar (0.00075) es el doble del rango completo entre las
cinco redes** (0.011485 → 0.011813 = 0.00033). Las cinco caen dentro de
1 e.e. de la mejor: con 126 días de validación **no es que se elija bien la
arquitectura, es que no se puede elegir**. Quedarse con el mínimo sería
elegir la que mejor encaja con el ruido de ese corte — y además la más
cara. La regla de 1 e.e. selecciona `rnn_1capa`, 4× más pequeña que la del
mínimo.

El e.e. se calcula con el **día** como unidad de observación, no la
predicción individual: los 25 bancos de una misma fecha comparten un factor
sectorial, así que dividir por √(25·n_días) exageraría la precisión unas 5
veces.

**Rejilla por años de backfill** (`04_resultados_rejilla_profundidad.csv`),
test MAE / precisión direccional con `+28` años de historia sintética:

| Generador | test MAE | Precisión direccional |
|---|---|---|
| referencia `synth_years=0` (n=211) | 0.011906 | 50.4% |
| RBIG | **0.011745** | **52.9%** |
| GAN | 0.011842 | 51.1% |
| Ruido | 0.011856 | 51.9% |
| Gaussiana | 0.011867 | 49.5% |

**Esta tabla no admite la lectura "gana RBIG"** — ver §6.6. En la ejecución
anterior el mismo puesto lo ocupaba otro generador. Se incluye porque el
enunciado la pide, no porque ordene nada.

Hay además un matiz de la propia referencia que conviene explicitar: con
`synth_years=0` quedan 211 ventanas, de las cuales un **28% contiene algún
día sintético** en su entrada. No es un cero limpio, porque una ventana de
60 días que cruza la frontera real/sintética arrastra días sintéticos
aunque su último día sea real. El cero limpio lo da la rejilla por
porcentaje (§6.5).

### 6.5 El eje que pide el enunciado: porcentaje de datos sintéticos

El paso 3 pide "datasets que tengan distinto **porcentaje** de datos
sintéticos y reales" y el paso 5 ver "cómo meter más o menos datos
sintéticos modifica el comportamiento". La rejilla por años es la rejilla
natural del problema financiero, pero en porcentaje se amontona entre el
92% y el 98%: no deja ver la forma de la curva. `PCT_SYNTH_GRID` barre el
eje completo (`train_utils.slice_by_pct`), manteniendo todas las ventanas
reales y añadiendo las sintéticas más recientes hasta la proporción pedida,
en bloque temporal continuo.

Media de los 4 generadores en cada nivel (`05_tabla_porcentaje_sintetico.csv`):

| % sintético | n_train | acierto | ±e.e. | vs. referencia |
|---|---|---|---|---|
| **0% (real puro)** | 152 | **50.1%** | — | — |
| 25% | 203 | 51.92% | 0.38 | +1.82 |
| 50% | 304 | 51.72% | 0.29 | +1.62 |
| 75% | 608 | 51.41% | 0.14 | +1.31 |
| **90%** | 1.520 | **53.54%** | 0.91 | **+3.44** |
| 100% (sin ancla real) | 6.885 | 52.27% | 0.19 | +2.17 |

Aquí el 0% **sí** es un cero limpio: solo ventanas íntegramente reales. Y da
**50.1%**, una moneda al aire — que es la referencia honesta contra la que
medir.

Todos los niveles con sintéticos la superan, con **pico en el 90%** y caída
al pasar al 100%, donde ya no queda ningún dato real de ancla. Ese mismo
patrón —máximo antes del 100%, caída al quitar el ancla real— aparece por
separado en el experimento de volatilidad de v2 (§6.9), y es lo que le da
crédito: dos experimentos independientes, no un p-valor.

Tendencia sobre los 20 puntos con sintéticos: **+1.26 puntos de acierto por
cada 100% de sintético**, Spearman ρ=+0.393 (p=0.087). Sugerente, no
concluyente — y el 75% rompe la monotonía.

### 6.6 Cuánto de todo esto es señal: el suelo de ruido

**El ranking entre generadores no se reproduce.** Se ejecutó la rejilla
completa dos veces —mismo código, mismos datos, misma arquitectura— y se
comparó:

| | |
|---|---|
| Correlación de rangos (Spearman) entre las dos ejecuciones | **−0.10** |
| Cambio absoluto medio en acierto direccional | 2.36 puntos |
| Cambio máximo en una misma configuración | **6.36 puntos** |

Casos concretos: `noise` a 7 años pasó de 49.6% a 55.5% (de los peores a
el mejor); `noise` a 28 años, de 52.5% a 46.1%. El rango entre generadores
dentro de una misma profundidad es de 1.8 a 4.5 puntos — **menor que el
ruido entre ejecuciones**.

Con una semilla por configuración, comparar generadores mide la
inicialización, no el generador. Por eso §6.5 promedia entre los 4 en vez
de proclamar un ganador, y por eso la tabla de §6.4 se presenta con la
advertencia explícita.

**Corolario metodológico**: el arreglo correcto es promediar varias
semillas por configuración. No se hizo por coste de cómputo, y se declara
como limitación (§10) en vez de disimularse.

### 6.7 Separación temporal: qué está limpio y qué no

Auditoría explícita del pipeline, porque es donde se cometen los errores
que invalidan un trabajo entero:

| | |
|---|---|
| Generadores entrenados excluyendo val+test | ✅ (`nb02`, `holdout_mask`) |
| Backfill causal (usa el retorno real **contemporáneo**, nunca futuro) | ✅ |
| Ventanas de **test** que solapan con train | ✅ **0 de 122** |
| Separación efectiva train→test | 127 días de mercado (ventana X = 60) |
| Purga train→validación | ✅ 61 días naturales (`embargo_days`) |
| Arquitectura elegida en validación, no en test | ✅ |
| Retornos con `close_totalreturn` (dividendos reinvertidos) | ✅ |
| Garman-Klass sobre OHLC **sin ajustar** | ✅ correcto: H/L y C/O son ratios intradía, el factor de ajuste se cancela |
| Sesgo de supervivencia | ⚠️ reconocido, ver §5 |

Dos de estas casillas **estaban mal y se corrigieron** durante el
desarrollo, y merece la pena decirlo porque son errores típicos:

1. **La arquitectura se elegía por el MAE de test.** `run_architecture_comparison`
   solo evaluaba en test, y el notebook hacía `idxmin()` sobre esa columna
   — mientras la documentación afirmaba tres veces que se elegía por
   validación. Como la ganadora se propaga a las dos rejillas, el sesgo
   habría contaminado todos los resultados.
2. **No había purga entre train y validación.** 59 de las 126 ventanas de
   validación (47%) solapaban con el tramo de entrenamiento, lo que hacía
   optimistas el early stopping y la selección. La purga estaba
   implementada (`split_fold`) pero solo se usaba en la búsqueda
   walk-forward, no en las rejillas que producen los resultados
   reportados. Cuesta un 40% de las muestras de entrenamiento en la
   configuración más pequeña (252 → 152 ventanas); se paga.

### 6.8 v2: se arregla la causa raíz… y sigue sin cambiar nada

Carpeta `v2_persistencia_temporal/` (extensión; no modifica v1).

Investigando *por qué* la fidelidad no predecía la utilidad apareció una
causa concreta: `src/backfill.py` muestrea la volatilidad de cada día de
forma **independiente**, condicionando solo al retorno de ese mismo día.
Medido sobre el tramo sintético:

| persistencia (lag 1) | REAL | sintética v1 |
|---|---|---|
| Pearson | +0.631 | +0.075 |
| Información mutua (no lineal) | +0.304 | **+0.000** |

La serie sintética **no tiene clustering de volatilidad** — el hecho
estilizado más robusto de las series financieras. Información mutua
exactamente cero. Por bien que un generador modele la distribución
conjunta *de un día*, el backfill destruía la dinámica temporal al
muestrear cada día por separado: los generadores competían en algo que el
backfill luego borraba.

`backfill_persistente.py` lo arregla muestreando en secuencia y
condicionando también a `RV_{t-1}`. **La persistencia pasa de 0.075 a
0.593** (real: 0.639) y la información mutua de 0.000 a 0.244.

Y aun así, aguas abajo (`v2_comparativa_variantes.csv`):

| variante | persistencia | test MAE | ± ruido de semilla |
|---|---|---|---|
| sin canal de volatilidad | — | 0.0118186 | 5.5e-06 |
| v1, sin memoria | 0.075 | 0.0118230 | 5.4e-07 |
| v2, con memoria | 0.593 | 0.0118225 | 7.0e-06 |
| OHLC real (Garman-Klass) | 0.731 | 0.0118185 | 2.9e-06 |

El MAE se mueve 5·10⁻⁷ — **del mismo orden que el ruido de semilla**. Y la
variante **sin ningún canal de volatilidad** empata con la volatilidad
**real** de 30 años. Ni el sintético malo, ni el bueno, ni el dato real
mejoran sobre no tener el canal.

### 6.9 El control que faltaba: ¿es el dato, o es el modelo?

Todo lo anterior tiene un agujero: no distingue entre *"los sintéticos no
aportan"* y *"el predictor no funciona"*. Sin descartar lo segundo, lo
primero no es defendible.

`experimento_target_volatilidad.py` lo cierra cambiando **una sola cosa**,
el target: de retorno del día siguiente a **volatilidad** del día
siguiente, que sí es predecible (es lo que significa el clustering). El
target sale de Garman-Klass sobre OHLC diario **real** de los 30 años — no
de la realizada de 5 min, que obligaría a poner sintético como target y
estaríamos midiendo si el sintético predice al sintético.

Con el rigor que el problema exige: **log-volatilidad** (aproximadamente
lognormal, Andersen et al. 2003), **HAR-RV** (Corsi 2009) como benchmark en
vez del paseo aleatorio, **QLIKE** (Patton 2011) como pérdida robusta al
ruido del proxy, regresión de **Mincer-Zarnowitz** para calibración, y
**Diebold-Mariano** con errores HAC de Newey-West sobre el diferencial de
pérdida promediado por día.

| | QLIKE | corr | pendiente MZ |
|---|---|---|---|
| **HAR-RV** (3 coeficientes, 2009) | **0.3448** | **0.465** | **0.863** |
| mejor red (de 42 configuraciones) | 0.3714 | 0.045 | 0.183 |
| baseline constante | 0.4410 | 0.065 | 0.280 |
| baseline ingenuo | 0.5223 | 0.417 | 0.418 |

**Ninguna de las 42 redes bate a HAR**: todas dan Diebold-Mariano positivo
con p<0.05. Y el target **sí** es predecible — HAR saca R² fuera de muestra
de **+0.21** con calibración 0.863.

**Ablación de arquitectura.** La búsqueda de hiperparámetros había
seleccionado `global_pool=True`, que sustituye el `Flatten` por un
`GlobalAveragePooling1D`: ese pooling **promedia sobre el eje temporal**, de
modo que la red no puede distinguir si un valor viene de ayer o de hace 60
días. Para predecir la volatilidad de mañana —cuyo predictor dominante es
la de ayer— es incapacitante. Corriendo las dos variantes, quitar el
pooling casi triplica la correlación (0.041 → 0.142 en el nivel del 75%),
así que el orden temporal importaba… pero 0.142 sigue siendo la cuarta
parte del 0.465 de HAR. **El pooling era un problema, no *el* problema.**

Que la búsqueda lo eligiera es coherente: optimizaba sobre el target de
retornos, donde no hay señal, y ahí una arquitectura incapaz de aprender no
pierde nada frente a las demás.

### 6.10 Cómo reproducir estos números

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
python scripts/hp_search_generators.py --minutes 60 --worker 0    # generadores
python scripts/analizar_hpsearch.py                               # elegir ganadores
python scripts/experimento_espectro_gan.py                        # fidelidad vs utilidad
```

**Atajo para el notebook 04.** Sus 38 entrenamientos son independientes entre
sí, así que se pueden repartir entre procesos. `scripts/rejilla_paralela.py`
lo hace y escribe en los mismos checkpoints que el notebook lee, de modo que
al ejecutarlo después se los encuentra hechos y solo genera tablas y
gráficas:

```bash
./scripts/lanzar_rejilla_paralela.sh 10       # 10 workers
```

En una máquina de 10 núcleos: **9 min 36 s** frente a las 2 h 02 min de la
versión secuencial. El resultado es idéntico porque `train_utils.set_seed()`
re-fija `config.RANDOM_SEED` antes de construir **cada** modelo, así que no
depende de en qué orden ni en qué proceso se entrene cada combinación.

## 7. Conclusiones

**1. Qué generador se use no importa — y es el hallazgo más sólido.**
Variando la fidelidad distribucional del generador un factor **2.266×**, el
rendimiento final varía un **0.64%** (§6.3). El GAN con mejor fidelidad de
los 725 evaluados da el peor resultado aguas abajo. Lo que aporta valor es
tener más contexto histórico con el que entrenar, no la sofisticación del
modelo generativo: un GAN cuidadosamente optimizado no bate a añadir ruido
gaussiano a datos reales.

**2. El ranking entre generadores no es señal: es ruido de
inicialización.** Es la conclusión que más nos costó aceptar, porque
invalida cualquier "ganador". Repitiendo la rejilla completa dos veces —
mismo código, mismos datos, misma arquitectura— la correlación de rangos
entre ambas ejecuciones fue **−0.10**, con cambios de hasta **6.36 puntos**
de precisión direccional en la misma configuración (§6.6). Con una semilla
por configuración, comparar generadores mide la semilla, no el generador.
Cualquier lectura de tipo "el generador X es el mejor" en este trabajo
sería un artefacto.

**3. Optimizar la métrica equivocada es un riesgo real, no teórico.** Se
gastaron **7.428** evaluaciones optimizando fidelidad distribucional, y esa
métrica resultó no predecir la utilidad. El caso extremo es el generador de
Ruido: obtiene MMD ≈ 0 (fidelidad perfecta) simplemente **copiando**
muestras reales — memorización, no generación. Sin el experimento de §6.3
se habría elegido el generador por el criterio incorrecto sin saberlo.

**4. La causa raíz se encontró y se arregló, y no sirvió de nada.** El
backfill de v1 destruía el clustering de volatilidad (información mutua
0.000 frente a 0.304 real). El muestreo secuencial de v2 lo recupera casi
entero (0.075 → 0.593, real 0.639) y el MAE se mueve **5·10⁻⁷**, dentro del
ruido de semilla. Más aún: la variante **sin canal de volatilidad** empata
con la volatilidad **real** de 30 años (§6.8). Recuperar el realismo era
necesario para que el sintético fuera creíble, pero no suficiente para que
fuera útil.

**5. El cuello de botella es el predictor, no el dato sintético — y esto
acota lo que el resto del trabajo puede afirmar.** Cambiando a un target
donde la señal es demostrable (volatilidad: HAR-RV alcanza R² fuera de
muestra de **+0.21**), **ninguna de las 42 configuraciones de red se le
acerca**; todas pierden contra HAR con Diebold-Mariano p<0.05 (§6.9). Un
modelo lineal de tres coeficientes de 2009 bate a todo el deep learning que
probamos.

La consecuencia es incómoda pero es la honesta: **no podemos concluir "los
datos sintéticos no ayudan"**. Lo que hemos demostrado es que no ayudan a
*este* predictor, que es incapaz de extraer siquiera una señal que sí
existe. Distinguir ambas cosas requeriría un predictor que primero
demuestre que aprende.

**6. Lo que sí se mueve al variar el porcentaje.** Con la referencia
limpia —152 ventanas íntegramente reales, **50.1%**, una moneda al aire—
todos los niveles con sintéticos la superan, con pico en el **90%
(53.5%, +3.4 puntos)** y caída al llegar al 100%, donde ya no queda ningún
dato real de ancla (§6.5). El mismo patrón, máximo antes del 100% y caída
al quitar el ancla, aparece por separado en el control de volatilidad
(§6.9). Que dos experimentos independientes coincidan es lo que sostiene
esta lectura; el p-valor (0.087) por sí solo no lo haría.

**7. Auditoría de fugas: dos errores encontrados y corregidos.** La
arquitectura se estaba eligiendo por el MAE de **test** (y la ganadora se
propaga a todas las rejillas), y no había **purga** entre entrenamiento y
validación — 47% de las ventanas de validación solapaban con el train
(§6.7). Ambos están corregidos y verificados: 0 de 122 ventanas de test
solapan con entrenamiento, y hay 61 días naturales de separación
train→validación. Se documentan en vez de borrarse porque son los errores
que más a menudo invalidan un trabajo de este tipo sin que nadie lo note.

**8. La lección metodológica.** Antes de preguntarse *"¿mejora mi modelo
con datos sintéticos?"* hay que verificar tres cosas que este trabajo
aprendió por las malas: que el modelo **puede aprender la tarea** (un
control positivo con un benchmark que lo demuestre); que las diferencias
comparadas **superan el ruido de la propia ejecución** (repetir con varias
semillas); y que **el protocolo hace lo que la documentación dice** — aquí
afirmaba tres veces que seleccionaba en validación mientras el código
miraba el test. Sin lo primero, un resultado nulo no distingue el dato del
modelo. Sin lo segundo, se declara un ganador que la siguiente ejecución
desmiente. Sin lo tercero, nada de lo anterior importa.

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
├── v2_persistencia_temporal/  anexo: backfill con memoria y control con
│                               volatilidad REAL de Garman-Klass (ver §10)
├── reports/
│   ├── figures/*.png            todas las gráficas generadas (versionadas)
│   └── tables/*.csv             todas las tablas generadas (versionadas)
└── scripts/
    ├── py_to_ipynb.py           conversor .py (celdas `# %%`) -> .ipynb
    ├── hp_search_generators.py  busqueda de hiperparametros de los 4 generadores
    ├── hp_search.py             busqueda del predictor (walk-forward + purga)
    ├── analizar_hpsearch.py     elige ganadores con la regla de 1 error estandar
    ├── rejilla_paralela.py      las 38 combinaciones del nb 04 en N procesos
    ├── lanzar_rejilla_paralela.sh   lanza los workers y consolida checkpoints
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
- La comparación de arquitecturas usa `dropout=0.3` y `L2=1e-4` fijos, no
  buscados: se eligieron para que las curvas converjan, no por optimizar el
  MAE. Con más tiempo valdría la pena una búsqueda fina de learning rate,
  tamaño de ventana y número de capas, aunque dado que las tres mejores
  redes empatan con el modelo nulo, es improbable que cambiara la conclusión.
- **El tamaño del test limita lo que se puede afirmar.** 122 días y 25
  bancos que correlacionan 0,715 entre sí: las "3.050 observaciones" son
  efectivamente ~122. Un error estándar del MAE es 0.000689, el 6 % del
  propio MAE, así que cualquier efecto menor que eso es indetectable —
  y el buscado resultó ser 20 veces más pequeño.
- **El horizonte elegido es el más difícil posible.** El retorno a un día de
  un banco líquido tiene 92 veces más ruido que señal; a 30 días la
  proporción baja a 15×. El enunciado no exigía ningún horizonte concreto
  ("los datos son a elección de los estudiantes"), así que repetir el
  estudio sobre una media a varias semanas es la continuación natural.
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
