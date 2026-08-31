"""
Arquitecturas de red (predictor de retornos y GAN), tal cual se vieron en
clase (Material_clase/*.ipynb), parametrizadas.

*** NO MODIFICAR HIPERPARAMETROS AQUI ***
Todas las funciones reciben unidades/filtros/dropout/l2/learning_rate/loss
como ARGUMENTOS con valores neutros por defecto. Los notebooks son los que
deciden esos valores al llamar a estas funciones (build_xxx(...)) y al
`.fit(...)`; este fichero solo define la FORMA de cada red (numero y tipo de
capas) y expone los mandos, nunca fija los valores.

Requiere tensorflow (tf.keras). Si no esta instalado, importar este modulo
lanzara un ImportError claro; en Google Colab funciona sin instalar nada.
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


def _optimizer(learning_rate: float | None):
    """Adam con el learning_rate que pida el notebook; `None` = Adam por
    defecto de Keras (1e-3), tal cual los notebooks de clase."""
    return keras.optimizers.Adam(learning_rate) if learning_rate else "adam"


def _reg(l2: float):
    """Regularizador L2 para las capas con pesos; `0` = sin regularizacion."""
    return keras.regularizers.l2(l2) if l2 else None


def build_predictor_dense(
    window_x: int,
    n_tickers: int,
    output_dim: int,
    hidden_units=(128, 64),
    loss: str = "mse",
    dropout: float = 0.0,
    l2: float = 0.0,
    learning_rate: float | None = None,
):
    """Red densa simple (Flatten + Dense...), como el "Modelo simple con
    capas densas" de clase.

    `loss`: los notebooks de clase compilan con 'mse' y solo reportan MAE
    como metrica secundaria; la diapositiva de teoria del taller ("REAL
    PROBLEM", pag. 11-12) usa en cambio 'minimize MAE' como funcion de
    aprendizaje, y reporta el error en las unidades del target. Con un
    target de colas pesadas como un retorno diario, entrenar con MAE es
    ademas mas robusto a los pocos dias de retorno extremo. Por defecto se
    deja 'mse'; el notebook 04 lo cambia a 'mae'.

    `dropout`, `l2`, `learning_rate`: mandos de regularizacion, con valores
    neutros por defecto (0/0/Adam-por-defecto = exactamente la red de
    clase). Con pocas muestras de entrenamiento y una entrada de
    window_x*n_tickers valores, esta red tiene ordenes de magnitud mas
    parametros que muestras y sobreajusta en pocas epocas si no se
    regulariza; el notebook 04 fija estos valores segun la busqueda de
    hiperparametros documentada en el README."""
    model = keras.Sequential(name="predictor_dense")
    model.add(layers.Flatten(input_shape=(window_x, n_tickers)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=_reg(l2)))
        if dropout:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_dim, kernel_regularizer=_reg(l2)))
    model.compile(optimizer=_optimizer(learning_rate), loss=loss, metrics=["mae", "mse"])
    return model


def build_predictor_cnn(
    window_x: int,
    n_tickers: int,
    output_dim: int,
    conv_filters=(64,),
    kernel_size: int = 3,
    dense_units: int = 100,
    loss: str = "mse",
    dropout: float = 0.0,
    l2: float = 0.0,
    learning_rate: float | None = None,
    global_pool: bool = False,
):
    """CNN1D. Con `conv_filters=(64,)` reproduce `cnn_model` (clase); con
    `conv_filters=(64,128,128)` reproduce `cnn_model_2`, la arquitectura que
    se usa en el taller para comparar los generadores (GAN/Gaussiano/RBIG/
    Ruido). Cada bloque es Conv1D(relu) + MaxPooling1D(2).

    `loss`, `dropout`, `l2`, `learning_rate`: ver docstring de
    `build_predictor_dense` (valores neutros por defecto = la red de clase).

    `global_pool`: si True sustituye el `Flatten()` final por un
    `GlobalAveragePooling1D()`. El Flatten deja un vector de
    (timesteps_restantes * filtros) que hace que la capa densa siguiente
    concentre la mayoria de los parametros de la red; el pooling global
    promedia sobre el tiempo y reduce esos parametros en uno o dos ordenes
    de magnitud, que es la palanca mas fuerte contra el sobreajuste cuando
    hay muy pocas muestras de entrenamiento."""
    model = keras.Sequential(name="predictor_cnn")
    model.add(
        layers.Conv1D(
            filters=conv_filters[0],
            kernel_size=kernel_size,
            activation="relu",
            input_shape=(window_x, n_tickers),
            kernel_regularizer=_reg(l2),
        )
    )
    model.add(layers.MaxPooling1D(pool_size=2))
    for filters in conv_filters[1:]:
        model.add(
            layers.Conv1D(
                filters=filters, kernel_size=kernel_size, activation="relu",
                kernel_regularizer=_reg(l2),
            )
        )
        model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.GlobalAveragePooling1D() if global_pool else layers.Flatten())
    if dropout:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense_units, activation="relu", kernel_regularizer=_reg(l2)))
    if dropout:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_dim, kernel_regularizer=_reg(l2)))
    model.compile(optimizer=_optimizer(learning_rate), loss=loss, metrics=["mae", "mse"])
    return model


def build_predictor_rnn(
    window_x: int,
    n_tickers: int,
    output_dim: int,
    lstm_units=(64,),
    dense_units: int = 100,
    loss: str = "mse",
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
    l2: float = 0.0,
    learning_rate: float | None = None,
):
    """RNN basada en LSTM. `lstm_units=(64,)` reproduce `rnn_model`;
    `lstm_units=(64,128)` reproduce `rnn_model_2` (2 capas LSTM apiladas).

    `loss`, `dropout`, `l2`, `learning_rate`: ver docstring de
    `build_predictor_dense`. `recurrent_dropout` aplica dropout tambien a
    la conexion recurrente de la LSTM (regularizacion especifica de RNN);
    todos con valores neutros por defecto = la red de clase."""
    model = keras.Sequential(name="predictor_rnn")
    lstm_kwargs = dict(
        activation="relu",
        kernel_regularizer=_reg(l2),
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
    )
    if len(lstm_units) == 1:
        model.add(
            layers.LSTM(
                units=lstm_units[0], input_shape=(window_x, n_tickers), **lstm_kwargs
            )
        )
    else:
        model.add(
            layers.LSTM(
                units=lstm_units[0], input_shape=(window_x, n_tickers),
                return_sequences=True, **lstm_kwargs
            )
        )
        for units in lstm_units[1:-1]:
            model.add(layers.LSTM(units=units, return_sequences=True, **lstm_kwargs))
        model.add(layers.LSTM(units=lstm_units[-1], **lstm_kwargs))
    model.add(layers.Flatten())
    if dropout:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense_units, activation="relu", kernel_regularizer=_reg(l2)))
    if dropout:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_dim, kernel_regularizer=_reg(l2)))
    model.compile(optimizer=_optimizer(learning_rate), loss=loss, metrics=["mae", "mse"])
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
    """Modelo combinado generador+discriminador congelado (congela
    `discriminator.trainable` antes de compilarlo junto al generador).
    `src.generators.GANGenerator` entrena con pasos manuales de
    `tf.GradientTape` en vez de este modelo combinado (ver su docstring);
    esta funcion se deja disponible para inspeccionar o depurar el grafo
    combinado manualmente si hace falta."""
    discriminator.trainable = False
    model = keras.Sequential(name="gan_combined")
    model.add(generator)
    model.add(discriminator)
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model
