"""
Los 4 modelos generativos del taller, con una interfaz comun:

    gen = XxxGenerator(**hiperparametros)
    gen.fit(XY_flat_real)          # XY_flat_real: (n_reales, d)
    XY_synth = gen.sample(n_muestras)   # (n_muestras, d)

Los 4 tipos (los "vistos en clase", ver Material_clase/):
  1. NoiseGenerator    -> datos reales + ruido gaussiano (el "ejemplo muy
                           tonto" de Taller_GANs.ipynb). Es el "modelo simple"
                           obligatorio del enunciado.
  2. GaussianGenerator -> ajusta una Normal multivariante (media+covarianza)
                           sobre el vector aplanado y muestrea de ella
                           (Taller_Gaussian_solution.ipynb). Se anade
                           shrinkage (Ledoit-Wolf) sobre la covarianza como
                           buena practica de estimacion robusta (mismo
                           modelo que clase, estimador mas fino de Sigma).
  3. RBIGGenerator     -> Rotation-Based Iterative Gaussianization (Laparra,
                           Camps-Valls & Malo, 2011), el metodo con el que la
                           propia diapositiva del taller compara los GANs.
                           No habia notebook de alumno para este, se
                           implementa aqui desde cero.
  4. GANGenerator      -> GAN densa condicion/incondicional identica en
                           espiritu a Taller_GANs.ipynb (generador+discrimi-
                           nador feed-forward sobre el vector aplanado).
                           Las arquitecturas viven en src/modelos.py; aqui
                           solo se orquesta el entrenamiento adversarial.

Todas las clases fijan una semilla (`random_state`) para reproducibilidad y
guardan el historial de "loss" de entrenamiento cuando aplica, para las
curvas de convergencia que pide el enunciado.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.covariance import LedoitWolf


class BaseGenerator:
    name = "base"

    def fit(self, XY_flat: np.ndarray) -> "BaseGenerator":
        raise NotImplementedError

    def sample(self, n_samples: int) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1) Modelo simple: ruido gaussiano sobre datos reales
# ---------------------------------------------------------------------------
class NoiseGenerator(BaseGenerator):
    """"Ejemplo muy tonto": toma muestras reales al azar y les suma ruido
    gaussiano iid de desviacion `sigma` (proporcional a la escala de cada
    columna si `relative=True`). No aprende ninguna distribucion, solo
    perturba: es el modelo simple/baseline pedido en el enunciado."""

    name = "noise"

    def __init__(self, sigma: float = 0.05, relative: bool = True, random_state: int = 42):
        self.sigma = sigma
        self.relative = relative
        self.random_state = random_state

    def fit(self, XY_flat: np.ndarray) -> "NoiseGenerator":
        self.pool_ = XY_flat
        self.scale_ = XY_flat.std(axis=0) if self.relative else 1.0
        self.rng_ = np.random.default_rng(self.random_state)
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        idx = self.rng_.integers(0, len(self.pool_), size=n_samples)
        base = self.pool_[idx]
        noise = self.rng_.normal(0, self.sigma, size=base.shape) * self.scale_
        return base + noise


# ---------------------------------------------------------------------------
# 2) Gaussiana multivariante (media + covarianza)
# ---------------------------------------------------------------------------
class GaussianGenerator(BaseGenerator):
    """Ajusta N(mu, Sigma) sobre el vector conjunto y muestrea de ella
    (`rng.multivariate_normal`), igual que Taller_Gaussian_solution.ipynb.

    Se usa el estimador con shrinkage de Ledoit-Wolf
    (`sklearn.covariance.LedoitWolf`) en vez del `np.cov` crudo de clase:
    con muestras suficientes (nuestro caso, d=5, decenas de miles de filas)
    el shrinkage aplicado es minimo (`shrinkage_` ~ 0), pero deja el
    estimador robusto por defecto si en el futuro se usa con menos datos o
    mas dimension. Sigue siendo "una Gaussiana multivariante"; solo cambia
    COMO se estima Sigma, no el modelo."""

    name = "gaussian"

    def __init__(self, shrinkage: bool = True, random_state: int = 42):
        self.shrinkage = shrinkage
        self.random_state = random_state

    def fit(self, XY_flat: np.ndarray) -> "GaussianGenerator":
        self.mean_ = XY_flat.mean(axis=0)
        if self.shrinkage:
            lw = LedoitWolf().fit(XY_flat)
            self.cov_ = lw.covariance_
            self.shrinkage_ = lw.shrinkage_
        else:
            self.cov_ = np.cov(XY_flat.T)
            self.shrinkage_ = 0.0
        self.rng_ = np.random.default_rng(self.random_state)
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        return self.rng_.multivariate_normal(self.mean_, self.cov_, size=n_samples)


# ---------------------------------------------------------------------------
# 3) RBIG: Rotation-Based Iterative Gaussianization
# ---------------------------------------------------------------------------
class RBIGGenerator(BaseGenerator):
    """Rotation-Based Iterative Gaussianization (Laparra, Camps-Valls & Malo,
    2011). Alterna, `n_iters` veces:
        (a) gaussianizacion marginal: cada columna se lleva a una uniforme
            via su funcion de distribucion empirica (rank-based ECDF) y
            luego a N(0,1) via la inversa de la normal estandar (probit).
        (b) una rotacion ortogonal aleatoria, para repartir la dependencia
            no lineal entre ejes antes de la siguiente gaussianizacion.

    Tras suficientes iteraciones los datos son (aproximadamente) N(0, I) en
    el espacio transformado, por lo que generar muestras nuevas es tan
    simple como muestrear N(0, I) y deshacer la cadena de transformaciones
    en orden inverso (las rotaciones se invierten trasponiendo, al ser
    ortogonales; las gaussianizaciones marginales se invierten con la
    funcion cuantil empirica).

    Para no disparar la memoria en alta dimension, la funcion de cuantiles
    empirica de cada iteracion/columna no se guarda muestra a muestra, sino
    en una rejilla de `grid_size` puntos (interpolacion lineal).
    """

    name = "rbig"

    def __init__(self, n_iters: int = 15, grid_size: int = 300, random_state: int = 42):
        self.n_iters = n_iters
        self.grid_size = grid_size
        self.random_state = random_state

    @staticmethod
    def _rank_uniform(col: np.ndarray) -> np.ndarray:
        n = len(col)
        order = np.argsort(col)
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        return (ranks - 0.5) / n

    def _random_orthogonal(self, d: int, rng: np.random.Generator) -> np.ndarray:
        A = rng.normal(size=(d, d))
        Q, R = np.linalg.qr(A)
        # Fija el signo para que la rotacion sea unica/estable
        Q = Q * np.sign(np.diag(R))
        return Q

    def fit(self, XY_flat: np.ndarray) -> "RBIGGenerator":
        rng = np.random.default_rng(self.random_state)
        n, d = XY_flat.shape
        p_grid = (np.arange(self.grid_size) + 0.5) / self.grid_size

        Z = XY_flat.astype(float).copy()
        self.rotations_ = []
        self.quantile_grids_ = []  # cada elemento: (grid_size, d)
        # Diagnostico de convergencia (no hay "loss" iterativa en RBIG, pero
        # si un indicador claro de que cada iteracion acerca los datos a una
        # Normal: el exceso de curtosis medio |kurtosis-3| por dimension,
        # que tiende a 0 a medida que la gaussianizacion avanza).
        from scipy.stats import kurtosis

        self.excess_kurtosis_history_ = []

        for _ in range(self.n_iters):
            grid_vals = np.quantile(Z, p_grid, axis=0)  # (grid_size, d)
            self.quantile_grids_.append(grid_vals)

            U = np.empty_like(Z)
            for j in range(d):
                U[:, j] = self._rank_uniform(Z[:, j])
            U = np.clip(U, 1e-6, 1 - 1e-6)
            G = norm.ppf(U)

            R = self._random_orthogonal(d, rng)
            self.rotations_.append(R)
            Z = G @ R

            self.excess_kurtosis_history_.append(float(np.mean(np.abs(kurtosis(Z, axis=0)))))

        self.d_ = d
        self.p_grid_ = p_grid
        self.rng_ = rng
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        Z = self.rng_.normal(size=(n_samples, self.d_))
        for it in reversed(range(self.n_iters)):
            R = self.rotations_[it]
            G = Z @ R.T  # rotacion ortogonal: inversa = transpuesta
            U = norm.cdf(G)
            grid_vals = self.quantile_grids_[it]
            X = np.empty_like(G)
            for j in range(self.d_):
                X[:, j] = np.interp(U[:, j], self.p_grid_, grid_vals[:, j])
            Z = X
        return Z


# ---------------------------------------------------------------------------
# 4) GAN densa (arquitectura en src/modelos.py)
# ---------------------------------------------------------------------------
class GANGenerator(BaseGenerator):
    """GAN totalmente conectada sobre el vector aplanado, replicando
    Taller_GANs.ipynb (generador/discriminador densos, entrenamiento
    adversarial por batches con ratio adaptativo D/G). La arquitectura de
    las redes vive en `src.modelos.build_gan_generator/build_gan_discriminator`;
    esta clase solo orquesta fit/sample y guarda el historial de losses.

    *** Nota Keras 3 *** `Taller_GANs.ipynb` entrena con el truco clasico
    de Keras 1/2: congelar el discriminador (`discriminator.trainable =
    False`) y compilar un modelo combinado generador+discriminador aparte
    (`modelos.build_gan_combined`), confiando en que el discriminador ya
    compilado (con `trainable=True`) "recuerda" sus variables entrenables
    aunque luego se ponga `trainable=False`. Ese truco depende de que Keras
    fije la lista de variables entrenables EN EL MOMENTO DE COMPILAR y no
    la actualice despues — comprobado empiricamente en este proyecto
    (Keras 3.15 / TF 2.21): ya NO es asi, `trainable_variables` se evalua
    de forma dinamica, así que sin recompilar en cada paso (~20x mas lento)
    el discriminador deja de aprender del todo.

    Por eso aqui NO se usa `build_gan_combined` ni `train_on_batch`: se
    entrena con pasos manuales de `tf.GradientTape`, pidiendo gradientes
    SOLO de `disc_.trainable_variables` (paso D) o SOLO de
    `gen_.trainable_variables` (paso G, con `disc_(..., training=False)`
    dentro de la cinta pero sin pedirle gradientes) — mismo resultado
    conceptual que el truco de clase (GAN adversarial por lotes con ratio
    D/G adaptativo), pero sin depender de la gestion interna de
    `.trainable` de cada version de Keras."""

    name = "gan"

    def __init__(
        self,
        latent_dim: int = 150,
        epochs: int = 500,
        batch_size: int = 10,
        gen_hidden=(256, 512, 1024),
        disc_hidden=(128, 64),
        learning_rate: float = 1e-4,
        d_steps_per_g: int = 2,
        random_state: int = 42,
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.gen_hidden = gen_hidden
        self.disc_hidden = disc_hidden
        self.learning_rate = learning_rate
        self.d_steps_per_g = d_steps_per_g
        self.random_state = random_state

    def fit(self, XY_flat: np.ndarray) -> "GANGenerator":
        from . import modelos  # import perezoso: requiere tensorflow
        import tensorflow as tf

        np.random.seed(self.random_state)
        d = XY_flat.shape[1]
        XY_flat = XY_flat.astype("float32")

        self.gen_ = modelos.build_gan_generator(self.latent_dim, d, self.gen_hidden)
        self.disc_ = modelos.build_gan_discriminator(d, self.disc_hidden)

        # learning_rate=1e-4 (Adam por defecto usa 1e-3): en el barrido de
        # hiperparametros de este proyecto, bajar el learning rate fue lo
        # que mas mejoro el colapso de modo del GAN vainilla en este
        # problema de baja dimension (d=5) — de una distancia de Frobenius
        # real-vs-sintetico de ~3.0 (colapso severo, ver notebook 02) a
        # ~1.3 con la misma arquitectura y el resto de hiperparametros
        # igual. Sigue sin iguialar a RBIG/Gaussiana/Ruido en ese
        # diagnostico: es una limitacion conocida y documentada de los GAN
        # vainilla en problemas de pocas dimensiones (ver README, seccion
        # de limitaciones), no un fallo de implementacion.
        bce = tf.keras.losses.BinaryCrossentropy()
        opt_d = tf.keras.optimizers.Adam(self.learning_rate)
        opt_g = tf.keras.optimizers.Adam(self.learning_rate)

        def d_train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                preds = self.disc_(x_batch, training=True)
                loss = bce(y_batch, preds)
            grads = tape.gradient(loss, self.disc_.trainable_variables)
            opt_d.apply_gradients(zip(grads, self.disc_.trainable_variables))
            return float(loss.numpy())

        def g_train_step(noise, y_mislabeled):
            with tf.GradientTape() as tape:
                synth = self.gen_(noise, training=True)
                preds = self.disc_(synth, training=False)
                loss = bce(y_mislabeled, preds)
            grads = tape.gradient(loss, self.gen_.trainable_variables)
            opt_g.apply_gradients(zip(grads, self.gen_.trainable_variables))
            return float(loss.numpy())

        d_loss_hist = np.zeros(self.epochs)
        g_loss_hist = np.zeros(self.epochs)
        ratio = 1.0
        batch = self.batch_size

        for epoch in range(self.epochs):
            # d_steps_per_g pasos de discriminador por cada paso de
            # generador: en el barrido de hiperparametros, dar al
            # discriminador mas pasos (>1) ademas del learning rate bajo
            # redujo aun mas el colapso de modo (ver notebook 02 / README).
            for _ in range(self.d_steps_per_g):
                batch_discr = max(1, int(round(ratio * batch)))
                idx = np.random.randint(0, len(XY_flat) - batch_discr) if len(XY_flat) > batch_discr else 0
                legit = XY_flat[idx : idx + batch_discr]

                noise = np.random.normal(0, 1, (batch_discr, self.latent_dim)).astype("float32")
                synth = self.gen_.predict(noise, verbose=0)

                x_batch = np.concatenate((legit, synth)).astype("float32")
                y_batch = np.concatenate(
                    (np.ones((batch_discr, 1)), np.zeros((batch_discr, 1)))
                ).astype("float32")
                d_loss_val = d_train_step(x_batch, y_batch)

            batch_gen = max(1, int(round(batch / ratio)))
            noise = np.random.normal(0, 1, (2 * batch_gen, self.latent_dim)).astype("float32")
            y_mislabeled = np.ones((2 * batch_gen, 1)).astype("float32")
            g_loss_val = g_train_step(noise, y_mislabeled)

            d_loss_hist[epoch] = d_loss_val
            g_loss_hist[epoch] = g_loss_val
            ratio = (d_loss_val + 1) / (g_loss_val + 1)

        self.history_ = {"d_loss": d_loss_hist, "g_loss": g_loss_hist}
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        return self.gen_.predict(noise, verbose=0)


GENERATOR_REGISTRY = {
    "noise": NoiseGenerator,
    "gaussian": GaussianGenerator,
    "rbig": RBIGGenerator,
    "gan": GANGenerator,
}
