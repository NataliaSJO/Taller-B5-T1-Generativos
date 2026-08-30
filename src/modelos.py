"""
Arquitecturas de red (predictor de retornos y GAN), tal cual se vieron en
clase (Material_clase/*.ipynb), parametrizadas.

*** NO MODIFICAR HIPERPARAMETROS AQUI ***
Todas las funciones reciben epochs/batch_size/unidades/filtros como
argumentos. Los notebooks son los que deciden esos valores al llamar a estas
funciones (build_xxx(...)) y al `.fit(...)`; este fichero solo define la
FORMA de cada red (numero y tipo de capas), no sus hiperparametros de
entrenamiento.

Requiere tensorflow (tf.keras). Si no esta instalado (p.ej. en local por el
limite de "long paths" de Windows, ver README), importar este modulo lanzara
un ImportError claro; en Google Colab funciona sin nada que instalar.
"""

from __future__ import annotations

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "src.modelos requiere tensorflow (tf.keras). Instalalo con "
        "`pip install tensorflow` o ejecuta los notebooks en Google Colab "
        "(donde ya viene preinstalado). Ver README.md, seccion 'Entorno'."
    ) from e

from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Predictor de retornos: X (window_x, n_tickers) -> Y (n_tickers,)
# ---------------------------------------------------------------------------
class BaselinePredictor:
    """Modelo "naive": predice que el retorno futuro sera igual al retorno
    del ultimo dia de la ventana X. No tiene parametros que ajustar (no
    hace falta .fit real); sirve de suelo de referencia, igual que en
    Taller_con_Datos_SP500_promedio.ipynb.

    `output_dim` es necesario cuando X trae mas canales que los que hay que
    predecir (ej. [retorno, volatilidad realizada] por banco -> Y solo son
    retornos): se asume que los primeros `output_dim` canales de X son la
    misma variable que Y, en el mismo orden."""

    def __init__(self, output_dim: int | None = None):
        self.output_dim = output_dim

    def fit(self, X, Y=None):
        return self

    def predict(self, X):
        last_step = X[:, -1, :]
        if self.output_dim is not None:
            last_step = last_step[:, : self.output_dim]
        return last_step


def build_predictor_baseline(output_dim: int | None = None) -> BaselinePredictor:
    return BaselinePredictor(output_dim=output_dim)


def build_predictor_linear() -> LinearRegression:
    """Regresion lineal sobre la ventana X aplanada -> Y. El notebook debe
    aplanar X antes de llamar a .fit/.predict (X.reshape(n, -1))."""
    return LinearRegression()


def build_predictor_dense(
    window_x: int, n_tickers: int, output_dim: int, hidden_units=(128, 64), loss: str = "mse"
):
    """Red densa simple (Flatten + Dense...), como el "Modelo simple con
    capas densas" de clase.

    `loss`: los notebooks de clase compilan con 'mse' y solo reportan MAE
    como metrica secundaria; la propia diapositiva de teoria del taller
    ("REAL PROBLEM", pag. 11-12) usa en cambio 'minimize MAE' como funcion
    de aprendizaje del problema real que motiva el taller (y los resultados
    de esa diapositiva se reportan en las mismas unidades que el target,
    Kelvin — la ventaja practica de entrenar con MAE). Con un target de
    colas pesadas como un retorno diario (ver notebook 02), entrenar con
    MAE es ademas mas robusto a los pocos dias de retorno extremo que con
    MSE. Por defecto se deja 'mse' (fiel a los notebooks de clase); el
    notebook 04 lo cambia explicitamente a 'mae' con esta justificacion."""
    model = keras.Sequential(name="predictor_dense")
    model.add(layers.Flatten(input_shape=(window_x, n_tickers)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(output_dim))
    model.compile(optimizer="adam", loss=loss, metrics=["mae", "mse"])
    return model


def build_predictor_cnn(
    window_x: int,
    n_tickers: int,
    output_dim: int,
    conv_filters=(64,),
    kernel_size: int = 3,
    dense_units: int = 100,
    loss: str = "mse",
):
    """CNN1D. Con `conv_filters=(64,)` reproduce `cnn_model` (clase); con
    `conv_filters=(64,128,128)` reproduce `cnn_model_2`, la arquitectura que
    se usa en el taller para comparar los generadores (GAN/Gaussiano/RBIG/
    Ruido). Cada bloque es Conv1D(relu) + MaxPooling1D(2).

    `loss`: ver docstring de `build_predictor_dense` — por defecto 'mse'
    (fiel a clase), pero la propia teoria del taller recomienda 'mae' para
    el problema real que motiva el ejercicio."""
    model = keras.Sequential(name="predictor_cnn")
    model.add(
        layers.Conv1D(
            filters=conv_filters[0],
            kernel_size=kernel_size,
            activation="relu",
            input_shape=(window_x, n_tickers),
        )
    )
    model.add(layers.MaxPooling1D(pool_size=2))
    for filters in conv_filters[1:]:
        model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"))
        model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(output_dim))
    model.compile(optimizer="adam", loss=loss, metrics=["mae", "mse"])
    return model


def build_predictor_rnn(
    window_x: int,
    n_tickers: int,
    output_dim: int,
    lstm_units=(64,),
    dense_units: int = 100,
    loss: str = "mse",
):
    """RNN basada en LSTM. `lstm_units=(64,)` reproduce `rnn_model`;
    `lstm_units=(64,128)` reproduce `rnn_model_2` (2 capas LSTM apiladas).

    `loss`: ver docstring de `build_predictor_dense`."""
    model = keras.Sequential(name="predictor_rnn")
    if len(lstm_units) == 1:
        model.add(
            layers.LSTM(
                units=lstm_units[0],
                activation="relu",
                input_shape=(window_x, n_tickers),
            )
        )
    else:
        model.add(
            layers.LSTM(
                units=lstm_units[0],
                activation="relu",
                input_shape=(window_x, n_tickers),
                return_sequences=True,
            )
        )
        for units in lstm_units[1:-1]:
            model.add(layers.LSTM(units=units, activation="relu", return_sequences=True))
        model.add(layers.LSTM(units=lstm_units[-1], activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(output_dim))
    model.compile(optimizer="adam", loss=loss, metrics=["mae", "mse"])
    return model


# ---------------------------------------------------------------------------
# GAN (generador/discriminador densos sobre el vector aplanado XY)
# ---------------------------------------------------------------------------
def build_gan_generator(latent_dim: int, output_dim: int, hidden_units=(256, 512, 1024)):
    model = keras.Sequential(name="gan_generator")
    model.add(layers.Dense(hidden_units[0], input_shape=(latent_dim,), activation="relu"))
    for units in hidden_units[1:]:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(output_dim, activation="tanh"))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


def build_gan_discriminator(input_dim: int, hidden_units=(128, 64)):
    model = keras.Sequential(name="gan_discriminator")
    model.add(layers.Flatten(input_shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_gan_combined(generator, discriminator):
    """Modelo combinado generador+discriminador congelado, tal cual
    `Taller_GANs.ipynb`. `src.generators.GANGenerator` YA NO usa esta
    funcion para entrenar (ver la nota "Keras 3" en su docstring: el truco
    de congelar `discriminator.trainable=False` y confiar en que el
    compilado anterior "recuerde" las variables entrenables no funciona en
    Keras 3 sin recompilar en cada paso, ~20x mas lento) — se deja aqui por
    fidelidad al notebook de clase y por si se quiere inspeccionar/depurar
    el modelo combinado manualmente."""
    discriminator.trainable = False
    model = keras.Sequential(name="gan_combined")
    model.add(generator)
    model.add(discriminator)
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model
