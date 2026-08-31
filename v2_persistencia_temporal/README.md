# v2 · Persistencia temporal en el backfill

Extensión del proyecto principal. **No modifica nada de v1**: importa de
`src/` sin tocarlo y escribe sus salidas con prefijo `v2_`, para que la
comparación antes/después sea limpia y ambas versiones sigan siendo
ejecutables.

## El problema que resuelve

El resultado central de v1 (§6.3 del README principal) fue **negativo**:
tras 7.426 evaluaciones optimizando la fidelidad distribucional de los
generadores, esa fidelidad **no predecía** la utilidad aguas abajo. La
calidad del generador variaba un factor 2.266× y el rendimiento final solo
un 0,64%.

Investigando por qué, apareció la causa. `src/backfill.py` muestrea la
volatilidad de cada día de forma **independiente**, condicionando solo al
retorno de ese mismo día. Medido sobre la serie sintética de JPM:

| persistencia (lag 1) | REAL (5 min) | sintética v1 |
|---|---|---|
| Pearson | +0.631 | +0.080 |
| Spearman (rangos) | +0.636 | +0.051 |
| Pearson sobre log(vol) | +0.656 | +0.070 |
| Información mutua (no lineal) | +0.304 | **+0.000** |

La serie sintética de v1 **no tiene clustering de volatilidad** — el hecho
estilizado más robusto de las series financieras. Información mutua
exactamente cero: ni dependencia lineal ni no lineal con el día anterior.

Y ahí está la explicación del resultado negativo: por bien que un generador
modele la distribución conjunta **de un día**, el backfill destruye la
**dinámica temporal** al muestrear cada día por separado. Los generadores
competían en algo que el backfill luego borraba.

## El cambio

El muestreo pasa a ser secuencial y con memoria:

```
v1:  RV_t  ~  p( · | retorno_t )                (días independientes)
v2:  RV_t  ~  p( · | retorno_t , RV_{t-1} )     (cadena con memoria)
```

Para que el generador pueda aprender esa condicional, el pool de
entrenamiento incorpora la volatilidad **retardada** como variable más:
cada fila es `[retorno_t, RV_{t-1}, RV_t, open30_t, close30_t, hl_t]`. El
emparejamiento condicional pasa entonces a usar dos variables, y la
volatilidad generada hoy condiciona la de mañana — ese arrastre es lo que
crea el clustering.

### Resultado

| persistencia (lag 1) | REAL | v1 | **v2** |
|---|---|---|---|
| Pearson | +0.631 | +0.080 | **+0.616** |
| Spearman | +0.636 | +0.051 | **+0.576** |
| Pearson sobre log | +0.656 | +0.070 | **+0.618** |
| Información mutua | +0.304 | +0.000 | **+0.244** |

De cero a prácticamente el nivel real. La información mutua recupera ~80%
de la dependencia no lineal del día anterior.

> Las dos tablas de arriba están medidas sobre **JPM**. La tabla de
> resultados aguas abajo (más abajo) mide sobre el canal de volatilidad del
> propio dataset, y da cifras ligeramente distintas (0.075 y 0.593) porque
> es otra serie, no otra conclusión. Importa medir siempre en el **mismo
> tramo**: si la persistencia de v1 se mide sobre la serie completa en vez
> de solo sobre el tramo sintético, el tramo real final (≈0.64) la arrastra
> a 0.113 y deja de ser comparable con v2.

## Segunda aportación: volatilidad REAL de 30 años desde el OHLC

`volatilidad_ohlc.py` calcula los estimadores de rango clásicos —
**Parkinson** (1980, usa high-low) y **Garman-Klass** (1980, usa OHLC
completo) — sobre el histórico diario de Norgate, que tiene máximo y mínimo
reales de los 30 años sin un solo hueco.

Contrastados contra la volatilidad realizada verdadera en la ventana donde
sí hay barras de 5 minutos (media sobre 20 bancos, 1.399 días):

| estimador | Pearson | Spearman | ratio de nivel |
|---|---|---|---|
| Parkinson | 0.775 | 0.704 | 1.018 |
| **Garman-Klass** | **0.812** | 0.733 | — |

Esto importa por dos motivos, y **cuestiona la premisa de v1**:

1. **Da una verdad de campo que v1 no tenía.** En v1 no había forma de
   comprobar si la volatilidad sintética de 1998 se parecía a la real,
   porque no existía volatilidad real de 1998. Ahora sí: se puede medir el
   error del backfill en el propio periodo histórico, en vez de confiar
   solo en la continuidad del empalme.
2. **Recupera el 81% de la correlación con la volatilidad intradía usando
   solo datos reales ya disponibles**, frente al 0.45 que da el retorno
   diario por sí solo. Es decir: buena parte de lo que v1 sintetizaba se
   podía haber obtenido, real, del propio OHLC. Conviene decirlo con
   claridad en vez de presentar el backfill como la única vía posible.

## Ficheros

```
volatilidad_ohlc.py                Parkinson y Garman-Klass sobre 30 años
                                   reales, y su validación contra la
                                   volatilidad realizada
backfill_persistente.py            pool con retardo + muestreo secuencial con
                                   memoria + medir_persistencia() con 4
                                   métricas (una no lineal)
experimento_v2.py                  comparativa aguas abajo de las 4 variantes
experimento_target_volatilidad.py  control positivo: mismo pipeline, target
                                   de volatilidad en vez de retorno
```

## Cómo ejecutarlo

```bash
python v2_persistencia_temporal/volatilidad_ohlc.py                # validación OHLC
python v2_persistencia_temporal/experimento_v2.py                  # comparativa 4 variantes
python v2_persistencia_temporal/experimento_target_volatilidad.py  # control positivo
```

## Resultado aguas abajo: la hipótesis queda REFUTADA

`experimento_v2.py` contrasta las cuatro variantes sobre el mismo test real
(`reports/tables/v2_comparativa_variantes.csv`):

| variante | persistencia (lag 1) | test MAE | ± ruido de semilla | dir. |
|---|---|---|---|---|
| 1 · sin canal de volatilidad | — | 0.0118186 | 5.5e-06 | 48.3% |
| 2 · v1, sin memoria | 0.075 | 0.0118230 | 5.4e-07 | 49.0% |
| 3 · v2, con memoria | 0.593 | 0.0118225 | 7.0e-06 | 48.5% |
| 4 · OHLC real (Garman-Klass) | 0.731 | 0.0118185 | 2.9e-06 | 48.6% |

La persistencia sube ×8 (0.075 → 0.593, casi el nivel real de 0.639) y el
MAE se mueve 5·10⁻⁷. **La dispersión entre variantes (4.5·10⁻⁶) es del
mismo orden que el ruido de inicialización dentro de cada variante**: no
son diferencias, es ruido.

El remate está en las filas 1 y 4. La variante **sin ningún canal de
volatilidad** empata en el mejor MAE con la volatilidad **real** de 30 años
—la que correlaciona 0.812 con la realizada verdadera—. Ni el sintético
malo, ni el sintético bueno, ni el dato real mejoran sobre no tener el
canal.

### Qué significa

Recuperar el clustering era necesario para que el sintético fuera
*realista*, pero no es suficiente para que sea *útil*. El límite no está en
la calidad del dato sintético: está en que la volatilidad, sintética o
real, no lleva información sobre el **signo ni la magnitud del retorno del
día siguiente**. Eso refuerza y explica el resultado negativo de v1 en vez
de contradecirlo — y es consistente con lo que se espera de un mercado
eficiente: la volatilidad de ayer predice la volatilidad de mañana (por eso
existe el clustering), no el retorno de mañana.

## Control positivo: ¿funciona el pipeline?

La conclusión anterior tiene un agujero que hay que cerrar: no distingue
entre "los sintéticos no aportan" y "el predictor no funciona".
`experimento_target_volatilidad.py` lo cierra cambiando **una sola cosa**,
el target — de retorno del día siguiente a **volatilidad** del día
siguiente— y dejando la entrada idéntica.

El target se toma de Garman-Klass sobre OHLC diario **real**, no de la
volatilidad realizada de 5 minutos: la realizada solo existe en los últimos
2 años, así que usarla obligaría a poner volatilidad sintética como target
en el tramo histórico, y estaríamos midiendo si el sintético predice al
sintético. Con Garman-Klass el target es real en los 30 años y lo único
sintético sigue siendo la entrada.

### Rigor financiero

- **Se modela log-volatilidad, no volatilidad en nivel.** La volatilidad
  realizada es aproximadamente lognormal (Andersen, Bollerslev, Diebold &
  Labys, 2003). No es un detalle: la primera versión de este experimento
  usaba nivel con pérdida MAE, cuyo óptimo es la **mediana condicional**,
  y la red degeneró a emitir casi una constante — batía al ingenuo en MAE
  con correlación 0.067. El baseline constante da correlación 0.065: la
  coincidencia confirma el diagnóstico de forma medible.
- **El benchmark es HAR-RV** (Corsi, 2009), no el paseo aleatorio. Regresa
  la volatilidad de mañana sobre las medias diaria, semanal y mensual;
  captura la memoria larga con tres coeficientes y es notoriamente difícil
  de batir. Batir a un ingenuo ruidoso no significa nada.
- **Tres baselines**, no uno: ingenuo, constante (detector de colapso) y
  HAR.
- **QLIKE** (Patton, 2011) como pérdida principal: es robusta a que la
  volatilidad no se observe sino que se estime con ruido, condición bajo
  la cual otras métricas ordenan mal los modelos.
- **Regresión de Mincer-Zarnowitz**: pendiente 1 = bien calibrado.

Ajustados sobre el test real (122 días):

| | RMSE log | R² | corr | QLIKE | pendiente MZ |
|---|---|---|---|---|---|
| ingenuo | 0.4506 | −0.164 | 0.417 | 0.5223 | 0.418 |
| constante | 0.4227 | −0.024 | 0.065 | 0.4410 | 0.280 |
| **HAR-RV** | **0.3712** | **+0.210** | **0.465** | **0.3448** | **0.863** |

Coeficientes HAR: +0.199 diario, +0.343 semanal, +0.393 mensual, suma
0.935 — positivos, crecientes con el horizonte y con reversión a la media,
el patrón que reporta la literatura. **HAR alcanza R² fuera de muestra de
+0.21**, así que en este target sí hay señal demostrable, cosa que con el
target de retornos nunca se pudo enseñar.

### Rigor de IA

- **El modelo ve la misma información que sus baselines.** En la primera
  versión no la veía: el canal de volatilidad de la entrada era la
  realizada de 5 min mientras el target era Garman-Klass, así que se le
  pedía autorregresión sobre una serie ausente de su entrada y se le
  comparaba contra un ingenuo que sí la tenía. La comparación estaba
  amañada en su contra. Ahora el log-GK real entra como canal.
- **Estandarización con estadísticos solo del tramo de entrenamiento** de
  cada configuración; los globales filtrarían información del test.
- **Varias semillas** con desviación reportada — en este proyecto el ruido
  de inicialización resultó ser del mismo orden que los efectos medidos.
- **Diebold-Mariano** (1995) con errores estándar HAC de Newey-West sobre
  el diferencial de pérdida promediado **por día**. Los 25 bancos del mismo
  día comparten un factor sectorial: tratar las 3.050 predicciones como
  independientes exagera la significancia unas 5 veces.

Resultados en `reports/tables/v2_target_volatilidad.csv`.
