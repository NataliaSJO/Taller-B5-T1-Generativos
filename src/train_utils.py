"""
Utilidades de entrenamiento/evaluacion compartidas por los notebooks 02-04:
  - mezclar datos reales + sinteticos en distintas proporciones
  - evaluar cualquier modelo (keras, sklearn o el BaselinePredictor) con las
    mismas metricas (MSE, MAE)
  - recorrer la rejilla (arquitectura) x (generador) x (n_reales, n_sinteticos)
    y devolver una tabla de resultados + el historial de loss de cada run,
    para las curvas de convergencia y las tablas que pide el enunciado.

Los hiperparametros de entrenamiento (epochs, batch_size, la propia rejilla)
se pasan siempre como argumentos desde el notebook.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraccion de dias en los que el SIGNO del retorno predicho coincide
    con el real (sube/baja) — metrica especifica de finanzas, mas relevante
    que el error de magnitud si lo que importa es la direccion de mercado.
    0.5 = tan bueno como lanzar una moneda."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def evaluate_predictor(model, X_test: np.ndarray, Y_test: np.ndarray) -> dict:
    """Evalua cualquier modelo con interfaz `.predict(X)` (BaselinePredictor,
    sklearn LinearRegression o un modelo keras) con las mismas metricas,
    POOLEADAS sobre los 25 bancos a la vez (MAE/MSE conjuntos). Para el
    desglose por banco (el que usa la comparacion final de clase, ver
    Taller_con_Datos_SP500_promedio.ipynb) usar `evaluate_predictor_per_ticker`."""
    Y_pred = model.predict(X_test)
    Y_pred = np.asarray(Y_pred).reshape(Y_test.shape)
    return {
        "mse": mse(Y_test, Y_pred),
        "mae": mae(Y_test, Y_pred),
        "directional_accuracy": directional_accuracy(Y_test, Y_pred),
    }


def evaluate_predictor_per_ticker(
    model, X_test: np.ndarray, Y_test: np.ndarray, ticker_names: list[str]
) -> pd.DataFrame:
    """MAE/MSE/precision direccional POR BANCO, igual que el bloque final de
    Taller_con_Datos_SP500_promedio.ipynb ("Generate a grouped bar chart to
    visualize the per-ticker MAE"). Una MAE pooleada sobre los 25 bancos a
    la vez esconde que cada uno tiene una escala de volatilidad distinta
    (ver notebook 01: GBCI ~1.6x mas volatil que JPM) y queda dominada por
    los bancos con mayor volatilidad; el desglose por ticker es la
    comparacion correcta."""
    Y_pred = np.asarray(model.predict(X_test)).reshape(Y_test.shape)
    rows = []
    for j, tk in enumerate(ticker_names):
        y_true, y_pred = Y_test[:, j], Y_pred[:, j]
        rows.append(
            {
                "ticker": tk,
                "mae": mae(y_true, y_pred),
                "mse": mse(y_true, y_pred),
                "directional_accuracy": directional_accuracy(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def ventana_contiene_sintetico(
    is_synthetic_ultimo_dia: np.ndarray,
    window_x: int = config.WINDOW_X_DAYS,
) -> np.ndarray:
    """Convierte "el ULTIMO dia de la ventana es sintetico" en "ALGUN dia
    de la ventana es sintetico", que es lo que de verdad hace falta.

    El notebook 03 guarda `is_synthetic = idx < cutoff`, donde `idx` es la
    fecha del ULTIMO dia de cada ventana X. Eso marca como reales las
    ventanas que cruzan la frontera real/sintetica aunque arrastren hasta
    `window_x - 1` dias sinteticos en su entrada: en este dataset son 59
    ventanas mal etiquetadas.

    El sesgo iba en la direccion prudente (subestimaba el contenido
    sintetico, no lo inflaba), pero hacia que `pct_synth` no significara lo
    que decia su documentacion — y `slice_by_pct` construye las
    proporciones del enunciado justamente con esa mascara.

    Como `idx[j]` es el ultimo dia de la ventana j, el PRIMER dia de esa
    ventana es `idx[j - window_x + 1]`; basta con desplazar la mascara.
    Las primeras `window_x - 1` ventanas empiezan antes de `idx[0]`, asi
    que heredan su valor.
    """
    m = np.asarray(is_synthetic_ultimo_dia, dtype=bool)
    n, w = len(m), int(window_x)
    if n == 0 or w <= 1:
        return m.copy()
    relleno = np.full(min(w - 1, n), m[0], dtype=bool)
    return np.concatenate([relleno, m[: max(n - w + 1, 0)]])[:n]


def slice_by_depth(
    X: np.ndarray,
    Y: np.ndarray,
    idx: pd.DatetimeIndex,
    synth_years: float,
    train_end: str,
    synth_anchor: str,
    is_synthetic: np.ndarray | None = None,
    embargo_days: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, float]:
    """Recorta (X, Y) a las ventanas cuya fecha cae en
    [synth_anchor - synth_years, train_end).

    Los dos extremos son conceptos DISTINTOS y no hay que confundirlos:
      - `train_end` es SIEMPRE el mismo (normalmente `VAL_START_DATE`): el
        final del tramo de entrenamiento no depende de cuanta historia
        sintetica se anada.
      - `synth_anchor` (normalmente `REAL_INTRADAY_START_DATE`) es el punto
        a partir del cual se cuenta hacia atras cuanta historia SINTETICA
        se incluye. `synth_years=0` dejaria solo la ventana real disponible
        para entrenar (`synth_anchor` -> `train_end`, sin ningun dia
        sintetico); `synth_years=28` anade ademas los 28 anios de backfill
        sintetico anteriores a `synth_anchor`.

    (Ojo: la ventana real para ENTRENAR no son los ~2 anios completos de
    REAL_INTRADAY_YEARS, porque VAL+TEST se comen el resto — ver
    config.SYNTH_DEPTH_YEARS_GRID.)

    Si se pasa `is_synthetic` (mascara booleana alineada con `idx`, True si
    la fila usa volatilidad sintetica en algun dia de su ventana X — ver
    `ventana_contiene_sintetico`), tambien devuelve `pct_synth`, la fraccion
    REAL de filas sinteticas del recorte (no una aproximacion por
    calendario).

    *** PURGA (`embargo_days`) ***
    El entrenamiento no llega hasta `train_end`, sino hasta
    `train_end - embargo_days` dias naturales. Sin esto, las ultimas
    ventanas de entrenamiento comparten casi toda su X con las primeras
    ventanas de validacion (cada muestra arrastra `WINDOW_X_DAYS` dias de
    historia), lo que no es look-ahead pero si crea dependencia
    estadistica entre train y validacion y hace la validacion OPTIMISTA
    — y con ella el early stopping y la eleccion de arquitectura.
    Medido en este dataset: sin purga, 59 de las 126 ventanas de
    validacion (47%) solapaban con el tramo de entrenamiento.

    Por defecto es 0 para no cambiar el comportamiento de `split_fold`,
    que ya resta el embargo por su cuenta antes de llamar aqui (si se
    aplicara en los dos sitios se restaria dos veces). Las rejillas
    (`run_depth_grid`, `run_pct_grid`) lo pasan explicitamente.
    """
    end = pd.Timestamp(train_end) - pd.Timedelta(days=embargo_days)
    start = pd.Timestamp(synth_anchor) - pd.Timedelta(days=synth_years * 365.25)
    mask = (idx >= start) & (idx < end)
    pct_synth = float(is_synthetic[mask].mean()) if is_synthetic is not None and mask.any() else 0.0
    return X[mask], Y[mask], idx[mask], pct_synth


def slice_by_pct(
    X: np.ndarray,
    Y: np.ndarray,
    idx: pd.DatetimeIndex,
    pct_synth_target: float,
    train_end: str,
    is_synthetic: np.ndarray,
    embargo_days: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, float]:
    """Recorta (X, Y) a un dataset con la PROPORCION pedida de filas
    sinteticas, manteniendo TODAS las reales disponibles.

    Por que existe esta funcion ademas de `slice_by_depth`: el enunciado
    (paso 3) pide "datasets que tengan distinto PORCENTAJE de datos
    sinteticos y reales", y el paso 5 pide ver "como meter mas o menos
    datos sinteticos modifica el comportamiento del modelo". Recortando por
    ANIOS de profundidad, la rejilla natural del problema
    (`SYNTH_DEPTH_YEARS_GRID` = 0, 7, 14, 21, 28) cae en 0% y luego
    87%-96%: cuatro puntos amontonados en el extremo alto y todo el tramo
    intermedio sin muestrear. Parametrizando por porcentaje se cubre el eje
    completo (0, 25, 50, 75, 90, 96%), que es justo el eje del que habla el
    enunciado.

    Las filas sinteticas se toman siempre las MAS RECIENTES (las contiguas
    a la ventana real), no una muestra aleatoria del historico: asi el
    tramo de entrenamiento sigue siendo un bloque temporal continuo y la
    unica variable que cambia entre datasets es cuanta historia sintetica
    se anade, no de que epoca procede.

    Con `n_r` filas reales, para una fraccion objetivo `p` hacen falta
    `n_s = n_r * p / (1 - p)` sinteticas; `p = 1.0` significa entrenar
    solo con sinteticas (todas las disponibles).

    `embargo_days`: purga entre el final del entrenamiento y el inicio de
    la validacion — ver `slice_by_depth`, mismo motivo y mismo valor.
    """
    end = pd.Timestamp(train_end) - pd.Timedelta(days=embargo_days)
    en_rango = idx < end
    is_synthetic = np.asarray(is_synthetic, dtype=bool)

    reales = np.flatnonzero(en_rango & ~is_synthetic)
    sinteticas = np.flatnonzero(en_rango & is_synthetic)

    p = float(pct_synth_target)
    if p >= 1.0:
        elegidas = sinteticas
    elif p <= 0.0:
        elegidas = reales
    else:
        n_s = int(round(len(reales) * p / (1.0 - p)))
        # las mas recientes: `sinteticas` ya viene ordenada por fecha
        elegidas = np.concatenate([sinteticas[max(len(sinteticas) - n_s, 0):], reales])

    elegidas = np.sort(elegidas)
    pct = float(is_synthetic[elegidas].mean()) if len(elegidas) else 0.0
    return X[elegidas], Y[elegidas], idx[elegidas], pct


def walk_forward_folds(
    n_folds: int = 4,
    val_months: int = 3,
    final_end: str = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Cortes temporales para validacion walk-forward con ventana de
    entrenamiento EXPANSIVA.

    Devuelve `n_folds` pares (val_start, val_end) consecutivos de
    `val_months` meses cada uno, terminando en `final_end` (por defecto
    `REAL_TEST_HOLDOUT_START_DATE`, para que el TEST no se toque nunca).
    El entrenamiento de cada fold va desde el inicio que corresponda segun
    la profundidad sintetica hasta `val_start` de ese fold — es decir, se
    entrena siempre con el pasado y se valida con el futuro inmediato, que
    es como se usaria el modelo en produccion.

    Por que esto y no un unico corte train/val: con una sola ventana de
    validacion (~126 ventanas) las diferencias pequenas entre
    configuraciones son indistinguibles del ruido, y al comparar cientos
    de configuraciones se acaba eligiendo la que mejor se ajusta al ruido
    de ESE corte concreto. Con varios cortes se obtiene una media y una
    DISPERSION entre periodos: una configuracion que gana en los 4 cortes
    es creible; una que gana en uno solo, no.

    Ejemplo con los valores por defecto (4 cortes de 3 meses):
        fold 1: train [inicio, 2024-12) | val [2024-12, 2025-03)
        fold 2: train [inicio, 2025-03) | val [2025-03, 2025-06)
        fold 3: train [inicio, 2025-06) | val [2025-06, 2025-09)
        fold 4: train [inicio, 2025-09) | val [2025-09, 2025-12)
    """
    end = pd.Timestamp(final_end or config.REAL_TEST_HOLDOUT_START_DATE)
    folds = []
    for k in range(n_folds):
        val_end = end - pd.DateOffset(months=val_months * k)
        val_start = val_end - pd.DateOffset(months=val_months)
        folds.append((val_start, val_end))
    return list(reversed(folds))


def split_fold(
    X: np.ndarray,
    Y: np.ndarray,
    idx: pd.DatetimeIndex,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    synth_years: float,
    synth_anchor: str,
    is_synthetic: np.ndarray | None = None,
    embargo_days: int = config.WINDOW_X_DAYS,
):
    """Datos de un fold walk-forward: entrena hasta `val_start` (con la
    profundidad sintetica pedida) y valida en [val_start, val_end).

    *** PURGA / EMBARGO (`embargo_days`) ***
    El entrenamiento no llega hasta `val_start`, sino hasta
    `val_start - embargo_days` dias naturales. Motivo: cada muestra usa
    una ventana de `WINDOW_X_DAYS` dias de historia, asi que una muestra
    de entrenamiento fechada pocos dias antes de `val_start` comparte casi
    toda su ventana de entrada con las primeras muestras de validacion.
    Eso no es look-ahead (esa muestra no usa nada posterior a su propia
    fecha), pero crea dependencia estadistica entre train y validacion y
    hace que la validacion salga OPTIMISTA. Purgar ese solape es la
    practica estandar en validacion cruzada de series financieras
    (purging/embargo, Lopez de Prado). El coste es perder ~`embargo_days`
    de entrenamiento en cada fold; el beneficio es que la metrica de
    validacion es honesta.

    `embargo_days=0` desactiva la purga (no recomendado; solo para
    comparar con el protocolo ingenuo).
    """
    train_end = val_start - pd.Timedelta(days=embargo_days)
    X_tr, Y_tr, _, pct_synth = slice_by_depth(
        X, Y, idx, synth_years=synth_years, train_end=str(train_end),
        synth_anchor=synth_anchor, is_synthetic=is_synthetic,
    )
    val_mask = (idx >= val_start) & (idx < val_end)
    return X_tr, Y_tr, X[val_mask], Y[val_mask], pct_synth


def _make_early_stopping(patience: int | None):
    """Callback de EarlyStopping (monitoriza val_loss, restaura los mejores
    pesos). `patience` alto a proposito: hace falta ver el val_loss
    estable durante muchas epocas seguidas (no solo dejar de mejorar un
    par de epocas) para poder afirmar que el modelo ha convergido — que
    es justo lo que tienen que mostrar las curvas de loss. `patience=None`
    desactiva el early stopping (epochs fijas)."""
    if patience is None:
        return []
    from tensorflow import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]


def run_architecture_comparison(
    architectures: dict[str, callable],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0,
    early_stopping_patience: int | None = 100,
) -> tuple[pd.DataFrame, dict]:
    """Entrena cada arquitectura de `architectures` (dict nombre -> funcion
    factoria SIN argumentos que devuelve un modelo listo para `.fit`) sobre
    el mismo train/val/test y devuelve una tabla comparativa + historiales de
    loss (para elegir la arquitectura del paso 4 del enunciado).

    *** COLUMNAS DE VALIDACION Y DE TEST, Y CUAL USAR ***
    La tabla trae las metricas por DUPLICADO:
      - `val_mae`, `val_mse`, `val_directional_accuracy` -> medidas en
        VALIDACION. Son las unicas que se pueden usar para ELEGIR la
        arquitectura.
      - `mae`, `mse`, `directional_accuracy` -> medidas en TEST. Solo para
        REPORTAR, una vez la eleccion ya esta hecha.

    Elegir por la columna de test seria seleccionar sobre el conjunto de
    evaluacion: con 7 candidatas se acabaria escogiendo la que mejor encaja
    con el ruido concreto de ese test, y las metricas de la ganadora
    quedarian sesgadas a la baja. Ademas la arquitectura elegida se propaga
    a las rejillas posteriores, asi que el sesgo contaminaria TODOS los
    numeros del trabajo, no solo esta tabla.

    (Usar `X_val` para el early stopping, como se hace abajo, es distinto y
    si es correcto: no elige entre modelos, solo decide cuando parar uno.)

    `early_stopping_patience`: nº de epochs sin mejora en val_loss antes de
    parar (y quedarse con los mejores pesos vistos, no los ultimos) — mas
    rapido y menos sobreajuste que forzar siempre `epochs` fijas. `None` lo
    desactiva (fiel al notebook de clase, que entrena epochs fijas)."""
    rows = []
    histories = {}
    for name, factory in architectures.items():
        model = factory()
        is_keras = hasattr(model, "compile")  # dense/cnn/rnn de src.modelos
        t0 = time.time()

        if is_keras:
            hist = model.fit(
                X_train, Y_train,
                epochs=epochs, batch_size=batch_size,
                validation_data=(X_val, Y_val), verbose=verbose,
                callbacks=_make_early_stopping(early_stopping_patience),
            )
            histories[name] = hist.history
            X_val_eval, X_test_eval = X_val, X_test
        else:
            # sklearn (LinearRegression) o BaselinePredictor: sin historial
            # de epochs; LinearRegression necesita la ventana X aplanada.
            needs_flat = type(model).__name__ == "LinearRegression"
            X_fit = X_train.reshape(X_train.shape[0], -1) if needs_flat else X_train
            model.fit(X_fit, Y_train)
            X_val_eval = X_val.reshape(X_val.shape[0], -1) if needs_flat else X_val
            X_test_eval = X_test.reshape(X_test.shape[0], -1) if needs_flat else X_test

        val_metrics = evaluate_predictor(model, X_val_eval, Y_val)
        test_metrics = evaluate_predictor(model, X_test_eval, Y_test)

        # Error estandar del val_mae, con el DIA como unidad de observacion
        # (no la prediccion individual): los 25 bancos de una misma fecha
        # comparten un factor sectorial y no son independientes, asi que
        # dividir por sqrt(25*n_dias) exageraria la precision unas 5 veces.
        # Este e.e. es lo que permite aplicar la regla de 1 e.e. al elegir
        # arquitectura, en vez de quedarse con el minimo a secas.
        Y_pred_val = np.asarray(model.predict(X_val_eval)).reshape(Y_val.shape)
        err_por_dia = np.abs(Y_val - Y_pred_val).mean(axis=1)
        val_mae_se = float(err_por_dia.std(ddof=1) / np.sqrt(len(err_por_dia)))

        rows.append({
            "model": name, "fit_seconds": time.time() - t0,
            "n_params": int(model.count_params()) if is_keras else np.nan,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_mae_se": val_mae_se,
            **test_metrics,
        })

    return pd.DataFrame(rows).set_index("model"), histories


def elegir_por_una_ee(
    tabla: pd.DataFrame,
    col_metrica: str = "val_mae",
    col_ee: str = "val_mae_se",
    col_complejidad: str = "n_params",
) -> str:
    """Regla del UN ERROR ESTANDAR: entre las arquitecturas cuyo error de
    validacion cae dentro de 1 e.e. del mejor, se elige la MAS SIMPLE (menos
    parametros), no la del minimo.

    Por que no el minimo: en este trabajo la diferencia entre las mejores
    arquitecturas es del orden de 1e-5 de MAE, mientras que el propio ruido
    de inicializacion mueve el resultado varias milesimas (README §6.6).
    Quedarse con el minimo es elegir la que mejor encaja con el ruido de ESE
    corte concreto, y ademas suele ser la mas cara: entre `rnn_2capas` y
    `cnn_3bloques` hay 3.3x de coste de computo por 8 millonesimas de MAE.

    Es la misma regla que ya usa `scripts/analizar_hpsearch.py` para los
    generadores, aplicada aqui por coherencia (Hastie, Tibshirani &
    Friedman, "The Elements of Statistical Learning", §7.10)."""
    mejor = tabla[col_metrica].idxmin()
    umbral = tabla.loc[mejor, col_metrica] + tabla.loc[mejor, col_ee]
    candidatas = tabla[tabla[col_metrica] <= umbral]
    if col_complejidad in candidatas and candidatas[col_complejidad].notna().any():
        return str(candidatas[col_complejidad].idxmin())
    return str(mejor)


def run_depth_grid(
    build_model_fn: callable,
    datasets_by_generator: dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]],
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    synth_years_grid: list[float],
    train_end: str,
    synth_anchor: str,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 0,
    ticker_names: list[str] | None = None,
    early_stopping_patience: int | None = 100,
    embargo_days: int = config.WINDOW_X_DAYS,
) -> tuple[pd.DataFrame, dict, dict]:
    """Para cada generador de `datasets_by_generator` (nombre -> (X, Y, idx,
    is_synthetic) con el historico COMPLETO de 30 anios construido con el
    backfill de ESE generador, ver notebook 03) y cada profundidad de
    `synth_years_grid` (anios de backfill sintetico anadidos ANTES de
    `synth_anchor`, ver `slice_by_depth`), recorta el dataset, entrena
    `build_model_fn()` (arquitectura fija, pesos reinicializados en cada
    run) y evalua en el mismo (X_test, Y_test) real.

    `synth_years=0` es identico para los 4 generadores (no hay ningun dia
    sintetico todavia: solo la ventana real disponible para entrenar) y
    sirve de referencia "sin sinteticos"; por eso solo se entrena una vez
    con la etiqueta "solo_reales" en vez de una vez por generador.

    Si se pasa `ticker_names` (25 nombres, en el mismo orden que las
    columnas de Y), tambien se calcula el desglose MAE/MSE/precision
    direccional POR BANCO (`evaluate_predictor_per_ticker`), igual que la
    comparacion final de `Taller_con_Datos_SP500_promedio.ipynb`.

    `early_stopping_patience`: ver `run_architecture_comparison` — con
    datasets que van de ~250 a ~7300 filas segun la profundidad, epochs
    fijas es especialmente ineficiente (los recortes pequenos convergen o
    sobreajustan mucho antes que los grandes); parar cuando val_loss deja
    de mejorar evita ambos problemas a la vez.

    Devuelve (tabla_resultados, historiales, resultados_por_ticker) donde
    `historiales` mapea (generador, synth_years) -> history.history (curvas
    de loss) y `resultados_por_ticker` mapea (generador, synth_years) ->
    DataFrame (25 filas) o {} si no se paso `ticker_names`."""
    rows = []
    histories = {}
    per_ticker = {}
    ref_generator = next(iter(datasets_by_generator))

    for synth_years in synth_years_grid:
        generators_this_depth = (
            {"solo_reales": datasets_by_generator[ref_generator]}
            if synth_years <= 0
            else datasets_by_generator
        )
        for gen_name, (X_full, Y_full, idx_full, is_synth_full) in generators_this_depth.items():
            X_tr, Y_tr, _, pct_synth = slice_by_depth(
                X_full, Y_full, idx_full, synth_years, train_end, synth_anchor,
                is_synth_full, embargo_days=embargo_days,
            )
            model = build_model_fn()
            hist = model.fit(
                X_tr, Y_tr, epochs=epochs, batch_size=batch_size,
                validation_data=(X_val, Y_val), verbose=verbose,
                callbacks=_make_early_stopping(early_stopping_patience),
            )
            metrics = evaluate_predictor(model, X_test, Y_test)
            rows.append(
                {
                    "generator": gen_name, "synth_years": synth_years,
                    "n_train": len(X_tr), "pct_synth": pct_synth, **metrics,
                }
            )
            histories[(gen_name, synth_years)] = hist.history
            if ticker_names is not None:
                per_ticker[(gen_name, synth_years)] = evaluate_predictor_per_ticker(
                    model, X_test, Y_test, ticker_names
                )

    return pd.DataFrame(rows), histories, per_ticker


def run_pct_grid(
    build_model_fn: callable,
    datasets_by_generator: dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]],
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    pct_grid: list[float],
    train_end: str,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 0,
    ticker_names: list[str] | None = None,
    early_stopping_patience: int | None = 100,
    embargo_days: int = config.WINDOW_X_DAYS,
) -> tuple[pd.DataFrame, dict, dict]:
    """Igual que `run_depth_grid` pero recortando por PORCENTAJE de filas
    sinteticas (`slice_by_pct`) en vez de por anios de profundidad.

    Es la rejilla que responde literalmente al paso 3 del enunciado
    ("datasets que tengan distinto porcentaje de datos sinteticos y
    reales") y al paso 5 ("como meter mas o menos datos sinteticos
    modifica el comportamiento del modelo"): un dataset por cada punto de
    `pct_grid` y por cada generador, todos evaluados en el MISMO test real.

    `pct=0` es identico para los 4 generadores (no entra ninguna fila
    sintetica), asi que se entrena una sola vez con la etiqueta
    "solo_reales", igual que en `run_depth_grid`.

    Devuelve (tabla, historiales, por_ticker) con la misma forma que
    `run_depth_grid`, indexando por (generador, pct_objetivo)."""
    rows = []
    histories = {}
    per_ticker = {}
    ref_generator = next(iter(datasets_by_generator))

    for pct in pct_grid:
        generators_this_pct = (
            {"solo_reales": datasets_by_generator[ref_generator]}
            if pct <= 0
            else datasets_by_generator
        )
        for gen_name, (X_full, Y_full, idx_full, is_synth_full) in generators_this_pct.items():
            X_tr, Y_tr, _, pct_real = slice_by_pct(
                X_full, Y_full, idx_full, pct, train_end, is_synth_full,
                embargo_days=embargo_days,
            )
            model = build_model_fn()
            hist = model.fit(
                X_tr, Y_tr, epochs=epochs, batch_size=batch_size,
                validation_data=(X_val, Y_val), verbose=verbose,
                callbacks=_make_early_stopping(early_stopping_patience),
            )
            metrics = evaluate_predictor(model, X_test, Y_test)
            rows.append(
                {
                    "generator": gen_name, "pct_objetivo": pct,
                    "n_train": len(X_tr), "pct_synth": pct_real, **metrics,
                }
            )
            histories[(gen_name, pct)] = hist.history
            if ticker_names is not None:
                per_ticker[(gen_name, pct)] = evaluate_predictor_per_ticker(
                    model, X_test, Y_test, ticker_names
                )

    return pd.DataFrame(rows), histories, per_ticker
