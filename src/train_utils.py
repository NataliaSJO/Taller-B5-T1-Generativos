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


def slice_by_depth(
    X: np.ndarray,
    Y: np.ndarray,
    idx: pd.DatetimeIndex,
    synth_years: float,
    train_end: str,
    synth_anchor: str,
    is_synthetic: np.ndarray | None = None,
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
    la fila usa volatilidad sintetica en algun dia de su ventana X), tambien
    devuelve `pct_synth`, la fraccion REAL de filas sinteticas del recorte
    (no una aproximacion por calendario).
    """
    end = pd.Timestamp(train_end)
    start = pd.Timestamp(synth_anchor) - pd.Timedelta(days=synth_years * 365.25)
    mask = (idx >= start) & (idx < end)
    pct_synth = float(is_synthetic[mask].mean()) if is_synthetic is not None and mask.any() else 0.0
    return X[mask], Y[mask], idx[mask], pct_synth


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
            X_test_eval = X_test
        else:
            # sklearn (LinearRegression) o BaselinePredictor: sin historial
            # de epochs; LinearRegression necesita la ventana X aplanada.
            needs_flat = type(model).__name__ == "LinearRegression"
            X_fit = X_train.reshape(X_train.shape[0], -1) if needs_flat else X_train
            model.fit(X_fit, Y_train)
            X_test_eval = X_test.reshape(X_test.shape[0], -1) if needs_flat else X_test

        metrics = evaluate_predictor(model, X_test_eval, Y_test)
        rows.append({"model": name, "fit_seconds": time.time() - t0, **metrics})

    return pd.DataFrame(rows).set_index("model"), histories


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
                X_full, Y_full, idx_full, synth_years, train_end, synth_anchor, is_synth_full
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
