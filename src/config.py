"""
Configuracion estructural del proyecto: universos de bancos, rutas, fechas y
constantes de las que depende todo el pipeline (descarga, features,
generadores, notebooks).

Los HIPERPARAMETROS de entrenamiento (epochs, batch_size, learning_rate,
tamanos de capas, etc.) NO viven aqui: se fijan en cada notebook al llamar a
las funciones de src/modelos.py y src/generators.py, para poder iterar sin
tocar el codigo fuente.

------------------------------------------------------------------------
PROBLEMA: predecir el retorno diario de un banco con una red que usa, como
feature extra, la volatilidad realizada intradia (calculada de barras de
5 min). EODHD solo tiene barras de 5 min desde 2020-11 (~5.5 anios); Norgate
tiene retorno diario REAL de hasta 36 anios. Para poder entrenar la red con
mucha mas historia que esos 5.5 anios, se rellenan (backfill) los ~24 anios
anteriores con volatilidad intradia SINTETICA, generada de forma condicional
al retorno diario real conocido de cada dia (que si tenemos, via Norgate),
usando 4 generadores: Ruido, Gaussiana, RBIG y GAN (ver src/generators.py).

Se usan DOS universos de bancos distintos, con proposito distinto:
  - GENERATOR_TICKERS: universo AMPLIO (hasta 150 bancos), para tener
    muchas muestras reales (retorno diario, features intradia) con las que
    entrenar bien los 4 generadores. Solo hace falta que tengan datos en la
    ventana real (2020-11 en adelante); no importa que no coticen desde 1990.
  - PREDICTOR_TICKERS: universo REDUCIDO (25 bancos) con historia diaria
    REAL completa desde 1990 (~36 anios) en el dump de Norgate. Es el
    universo sobre el que se construye la tarea supervisada final (ventanas
    X/Y de 30 anios, con volatilidad real 5.5 anios + sintetica 24.5 anios).
------------------------------------------------------------------------
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "datos"
RAW_DIR = DATA_DIR / "raw"                 # cache crudo de EODHD (gitignored)
INTERIM_DIR = DATA_DIR / "interim"         # datasets intermedios (gitignored)
PROCESSED_DIR = DATA_DIR / "processed"     # datasets finales pequenos (versionables)
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
TABLES_DIR = ROOT_DIR / "reports" / "tables"

NORGATE_DUCKDB = (
    DATA_DIR / "extracted" / "norgate_bancos_us_export_20260602_1043"
    / "bancos_us_norgate.duckdb"
)
APKEY_PATH = DATA_DIR / "APkey"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_eodhd_api_key() -> str:
    """Lee la API key de EODHD desde datos/APkey. Nunca hardcodear la key ni
    imprimirla: el fichero esta en .gitignore."""
    return APKEY_PATH.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Universo REDUCIDO (predictor): 25 bancos con ~36 anios de historia diaria
# REAL completa en el dump de Norgate y cobertura EODHD confirmada. Mezcla de
# "money center" (BAC, WFC, JPM, C) y bancos regionales de distinto tamano.
# ---------------------------------------------------------------------------
PREDICTOR_TICKERS = [
    "BAC", "WFC", "JPM", "C", "HBAN", "USB", "TFC", "KEY", "RF", "FITB",
    "VLY", "FHN", "PNC", "WAFD", "FNB", "FULT", "ASB", "WBS", "MTB", "ZION",
    "BOH", "UBSI", "CVBF", "CBSH", "GBCI",
]
# NB: ONB (Old National Bancorp) y SFNC (Simmons First) se descartaron pese a
# aparecer en banks_universe.csv con first_quoted_date=1990-01-02: la tabla
# export.bank_prices_daily solo tiene 352 filas reales para esos dos assetids
# (desde 2025-01-02), un gap de datos del propio dump de Norgate. Sustituidos
# por WAFD y BOH, que si tienen los ~36 anios completos y cobertura EODHD.

N_PREDICTOR_TICKERS = len(PREDICTOR_TICKERS)

# ---------------------------------------------------------------------------
# Universo AMPLIO (generadores): hasta 150 bancos activos de EEUU (Diversified
# Banks, Regional Banks, Thrifts & Mortgage Finance) segun banks_universe.csv,
# ordenados por shares_outstanding. No se exige historia larga: solo se usan
# en la ventana real (2020-11 en adelante) para entrenar los generadores.
# Se cachea y filtra por cobertura real en notebooks/00_descarga_datos.ipynb; los que no
# tengan suficientes datos (ej. bancos intervenidos/fusionados) se descartan
# automaticamente ahi, no aqui.
GENERATOR_TICKERS = [
    "BAC", "WFC", "JPM", "C", "HBAN", "USB", "TFC", "KEY", "RF", "FITB",
    "VLY", "FHN", "CFG", "FLG", "PNC", "ONB", "FNB", "COLB", "TFSL", "EBC",
    "MCHB", "HOMB", "FRCB", "FULT", "ASB", "WBS", "BANC", "MTB", "ZION",
    "NWBI", "SFNC", "FFIN", "AUB", "UBSI", "EWBC", "CVBF", "CBSH", "CFFN",
    "PFS", "GBCI", "HOPE", "FHB", "UCB", "PNBK", "OZK", "WAL", "CLBK", "FCF",
    "FIBK", "SSB", "FFBC", "SBCF", "WSBC", "RNST", "PB", "BRBS", "BUSE",
    "BBT", "HWC", "TOWN", "PNFP", "WAFD", "UMBF", "BKU", "ABCB", "CATY",
    "WTFC", "KRNY", "CFR", "BOKF", "SBNY", "IBOC", "HTH", "TRMK", "FRME",
    "OCFC", "AX", "WSFS", "SFBS", "FBK", "OSBC", "CBU", "NBTB", "HBNC",
    "STEL", "CNOB", "INDB", "LOB", "BY", "NBBK", "TCBI", "TBBK", "DCOM",
    "NFBK", "FBNC", "BOH", "AMTB", "STBA", "NBHC", "FMNB", "GABC", "EFSC",
    "PEBO", "NPB", "CUBI", "BANR", "HFWA", "FFIC", "SHBI", "BANF", "TCBK",
    "BCAL", "SICP", "HBT", "OBK", "EGBN", "SBSI", "HAFC", "AMAL", "BFST",
    "CCNE", "SYBT", "MCBS", "UVSP", "FSUN", "BWB", "CPF", "LKFN", "PBFS",
    "WABC", "FRBA", "FRST", "SRCE", "FMBH", "PDLB", "TFIN", "MPB", "CASH",
    "NRIM", "CARE", "MSBI", "FSBC", "CBAN", "RVSB", "CIVB", "IBCP", "WNEB",
    "FISI", "FBLA", "RBCAA",
]
N_GENERATOR_TICKERS = len(GENERATOR_TICKERS)

# Una sesion de 5 min individual necesita al menos MIN_BARS_PER_SESSION
# barras para contar como "valida" (una sesion normal tiene BARS_PER_SESSION
# = 78; los cierres anticipados por festivo, ej. el dia despues de Accion de
# Gracias, cierran a las 13:00 ET ~= 42 barras — 40 deja pasar esos cierres
# anticipados legitimos y descarta sesiones realmente rotas: feed caido,
# halt, apertura tardia).
MIN_BARS_PER_SESSION = 40

# Numero minimo de sesiones intradia validas (>= MIN_BARS_PER_SESSION barras)
# que debe tener un ticker en la ventana real para entrar en el pool de
# entrenamiento de los generadores (filtra bancos intervenidos/fusionados a
# mitad de la ventana, con muy pocos dias de cotizacion real).
MIN_SESSIONS_FOR_GENERATOR_POOL = 60

# Cortes de cordura sobre el pool condicional (retorno diario real, features
# intradia reales) de bancos AMPLIADOS: al usar 150 bancos (algunos poco
# liquidos o intervenidos, ver arriba), aparecen unas pocas filas con
# valores imposibles para una accion cotizando en continuo (retornos
# diarios >50%, rangos intradia >100%): tipicamente 1-2 tickers en proceso
# de quiebra/exclusion (ej. bancos regionales intervenidos en 2023-2025)
# con cruces de precio erraticos, no volatilidad real de mercado. Se
# descartan esas filas (son <0.1% del pool) antes de entrenar los
# generadores para que no distorsionen covarianza/colas.
POOL_MAX_ABS_LOG_RETURN = 0.5
POOL_MAX_REALIZED_VOL = 0.3
POOL_MAX_ABS_INTRADAY_RET = 0.5
POOL_MAX_HL_RANGE = 1.0

# ---------------------------------------------------------------------------
# Fechas: 30 anios totales = 5.5 reales (EODHD 5 min, desde 2020-11) + 24.5 de
# backfill sintetico (retorno diario REAL de Norgate + volatilidad intradia
# SINTETICA).
#
# NB: la ventana real NO es una eleccion de diseno, es todo lo que da EODHD:
# el endpoint /intraday solo sirve barras de 5 min desde finales de 2020, y
# REAL_INTRADAY_START_DATE se pone en la primera sesion realmente descargada
# (2020-11-02, ver reports/tables/00_cobertura_intradia.csv). Todo lo que hay
# descargado se usa; solo se sintetiza lo que de verdad no existe.
# ---------------------------------------------------------------------------
DAILY_START_DATE = "1990-01-02"              # primera fecha disponible en Norgate
DAILY_END_DATE = "2026-05-29"                # ultima fecha disponible en Norgate
TOTAL_HISTORY_START_DATE = "1996-05-29"      # 30 anios antes de DAILY_END_DATE

REAL_INTRADAY_YEARS = 5.5
REAL_INTRADAY_START_DATE = "2020-11-02"      # primera barra de 5 min real en EODHD
SYNTH_BACKFILL_START_DATE = TOTAL_HISTORY_START_DATE
SYNTH_BACKFILL_END_DATE = "2020-10-30"       # dia habil previo a REAL_INTRADAY_START_DATE

# Descarga EODHD: se pide desde el primer dia que sirve el API, que es
# justo el inicio de la ventana real; el recorte fino a
# REAL_INTRADAY_START_DATE se hace en notebooks/00 y en features.py.
INTRADAY_DOWNLOAD_START_DATE = "2020-11-01"
INTRADAY_DOWNLOAD_END_DATE = "2026-08-30"

# Dentro de la ventana real (2020-11 -> 2026-05), los DOS ULTIMOS tramos se
# dejan
# fuera del entrenamiento (ni de los generadores ni del predictor):
#   - VAL:  [VAL_START_DATE, REAL_TEST_HOLDOUT_START_DATE)   ~6 meses
#   - TEST: [REAL_TEST_HOLDOUT_START_DATE, DAILY_END_DATE]   ~6 meses
# Los generadores del notebook 02 solo ven datos anteriores a VAL_START_DATE
# (evita fuga: ni siquiera ven las estadisticas del tramo de validacion).
VAL_START_DATE = "2025-06-01"
REAL_TEST_HOLDOUT_START_DATE = "2025-12-01"

# ---------------------------------------------------------------------------
# Ventanas de la tarea supervisada (mismo esquema que los notebooks de clase:
# X = ventana pasada de `WINDOW_X_DAYS` retornos diarios [+ vol. realizada]).
# WINDOW_Y_DAYS=1 -> se predice el retorno del DIA SIGUIENTE (no una media a
# N dias como en los notebooks de clase): es "el predictor del dia siguiente"
# que pide el enunciado del proyecto.
# ---------------------------------------------------------------------------
WINDOW_X_DAYS = 60
WINDOW_Y_DAYS = 1

# Horizontes de prediccion que se comparan en el notebook 04. El enunciado
# no exige ninguno ("los datos son a eleccion de los estudiantes"), y la
# dificultad cambia mucho entre ellos: el ruido del target baja con la raiz
# del horizonte mientras la deriva se mantiene, asi que la relacion
# ruido/senal pasa de 92x a 1 dia, a 38x a 7 dias y a 15x a 30 dias.
#
# Los tres comparten EXACTAMENTE las mismas ventanas de entrada X; lo unico
# que cambia es el target, de modo que la comparacion entre horizontes aisla
# el efecto del horizonte y nada mas.
HORIZONTES_DIAS = [1, 7, 30]

# Rejilla de "profundidad": anios de backfill SINTETICO incluidos ANTES de
# REAL_INTRADAY_START_DATE, usada en el notebook 04 para comparar "con vs.
# sin sinteticos". El entrenamiento SIEMPRE termina en VAL_START_DATE (fijo,
# no depende de la profundidad) — ver train_utils.slice_by_depth.
#
# OJO: la ventana real disponible para ENTRENAR no son los 5.5 anios enteros
# de REAL_INTRADAY_YEARS, porque VAL (~6 meses) + TEST (~6 meses) se comen
# el final; lo que queda es (VAL_START_DATE - REAL_INTRADAY_START_DATE)
# ~= 4.6 anios (2020-11 -> 2025-06). synth_years=0 -> solo esos ~4.6 anios
# reales, sin ningun dia sintetico; synth_years=24 -> + todo el backfill
# sintetico disponible (1996-05 -> 2020-10, ~24.4 anios).
SYNTH_DEPTH_YEARS_GRID = [0, 1, 2, 5, 10, 24]

# Rejilla de PORCENTAJE de filas sinteticas, la que responde literalmente al
# paso 3 del enunciado ("datasets que tengan distinto porcentaje de datos
# sinteticos y reales") y al paso 5 ("como meter mas o menos datos sinteticos
# modifica el comportamiento del modelo") — ver train_utils.slice_by_pct.
#
# Hacen falta las DOS rejillas y no son redundantes:
#   - SYNTH_DEPTH_YEARS_GRID es la rejilla natural del PROBLEMA (cuantos anios
#     de historia bancaria se recuperan), pero en porcentaje cae en 0% y luego
#     ~57/72/80/84%: sigue dejando vacio todo el tramo 0-57%.
#   - PCT_SYNTH_GRID barre ese eje de forma uniforme, que es donde se ve de
#     verdad si "mas sintetico" ayuda, satura o empieza a estorbar.
# El 1.0 (entrenar SOLO con sinteticos, sin ninguna fila real) es el caso
# limite util: mide cuanta senal real hace falta como ancla.
PCT_SYNTH_GRID = [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
# Semilla unica de todo el proyecto. Se usa para inicializar los pesos de
# cada red, para el muestreo de los 4 generadores y para el backfill
# condicional.
#
# POR QUE HACE FALTA: las arquitecturas candidatas del notebook 04 empatan
# dentro del 0.5% de un error estandar (las tres mejores separadas por
# 0.000003 en MAE de validacion). Sin semilla fija, el orden entre ellas
# cambia de una ejecucion a otra por el azar de la inicializacion de pesos
# — se observo que la "ganadora" pasaba de rnn_1capa a rnn_2capas al
# reejecutar sin cambiar nada relevante. Como esa eleccion condiciona todo
# el resto del pipeline (las dos rejillas y el experimento del espectro),
# tiene que ser reproducible.
#
# OJO: fijar la semilla hace el resultado REPRODUCIBLE, no ROBUSTO. Que dos
# arquitecturas empaten sigue siendo cierto; lo unico que se elimina es que
# el empate se resuelva de forma distinta cada vez. Para decidir entre
# candidatas empatadas hay que promediar varias semillas o aplicar la regla
# de un error estandar (ver scripts/analizar_hpsearch.py).
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Barras de 5 min
# ---------------------------------------------------------------------------
BARS_PER_SESSION = 78   # sesion NYSE/Nasdaq de 6.5h / 5 min

# Maximo de dias por request al endpoint /intraday de EODHD (limite del API
# ~ 600-650 dias por llamada; se deja margen).
EODHD_MAX_SPAN_DAYS = 550
