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

    def __init__(self, sigma: float = 0.05, relative: bool = True,
                 noise_dist: str = "normal", df: float = 4.0,
                 random_state: int = 42):
        """`noise_dist`: forma del ruido que se suma.
          - 'normal'    : ruido gaussiano (el de Taller_GANs.ipynb).
          - 'student_t' : ruido t-Student con `df` grados de libertad,
            escalado a desviacion `sigma`. Los retornos financieros tienen
            colas mucho mas pesadas que una Normal (ver notebook 02), asi
            que perturbar con una t reproduce mejor la frecuencia de dias
            extremos que perturbar con una Normal."""
        self.sigma = sigma
        self.relative = relative
        self.noise_dist = noise_dist
        self.df = df
        self.random_state = random_state

    def fit(self, XY_flat: np.ndarray) -> "NoiseGenerator":
        self.pool_ = XY_flat
        self.scale_ = XY_flat.std(axis=0) if self.relative else 1.0
        self.rng_ = np.random.default_rng(self.random_state)
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        idx = self.rng_.integers(0, len(self.pool_), size=n_samples)
        base = self.pool_[idx]
        if self.noise_dist == "student_t":
            raw = self.rng_.standard_t(self.df, size=base.shape)
            # una t con df g.l. tiene varianza df/(df-2): se reescala para
            # que `sigma` signifique lo mismo que en el caso normal
            raw /= np.sqrt(self.df / (self.df - 2.0)) if self.df > 2 else 1.0
        else:
            raw = self.rng_.normal(0, 1, size=base.shape)
        return base + raw * self.sigma * self.scale_


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

    def __init__(self, shrinkage: bool = True, shrinkage_alpha: float | None = None,
                 shrinkage_target: str = "identity", marginal: str = "gaussian",
                 random_state: int = 42):
        """`shrinkage_alpha`: si se da (0..1), se usa shrinkage MANUAL con
        esa intensidad hacia una diagonal escalada,
        `Sigma = (1-a)*S + a*media(diag(S))*I`, en vez de dejar que
        Ledoit-Wolf elija la intensidad. Permite explorar el grado de
        regularizacion de la covarianza en vez de aceptar un unico valor
        automatico.

        `marginal`: espacio en el que se ajusta la Normal.
          - 'gaussian': se ajusta directamente sobre los datos (el modelo
            de Taller_Gaussian_solution.ipynb).
          - 'rank_gauss': cada columna se lleva primero a una N(0,1) por su
            distribucion empirica (transformada rank-gauss), se ajusta ahi
            la Normal multivariante y se deshace la transformacion al
            muestrear. Sigue siendo "una Gaussiana multivariante", pero
            aplicada a la COPULA en vez de a los datos crudos: conserva las
            marginales reales (colas pesadas incluidas) y modela solo la
            dependencia, que es justo lo que una Normal si puede capturar."""
        self.shrinkage = shrinkage
        self.shrinkage_alpha = shrinkage_alpha
        self.shrinkage_target = shrinkage_target
        self.marginal = marginal
        self.random_state = random_state

    def fit(self, XY_flat: np.ndarray) -> "GaussianGenerator":
        Z = XY_flat
        if self.marginal == "rank_gauss":
            # guarda los valores ordenados de cada columna para poder
            # deshacer la transformacion al muestrear
            self.sorted_cols_ = [np.sort(XY_flat[:, j]) for j in range(XY_flat.shape[1])]
            Z = np.empty_like(XY_flat, dtype=float)
            n = len(XY_flat)
            for j in range(XY_flat.shape[1]):
                order = np.argsort(XY_flat[:, j])
                ranks = np.empty(n)
                ranks[order] = np.arange(1, n + 1)
                Z[:, j] = norm.ppf((ranks - 0.5) / n)

        self.mean_ = Z.mean(axis=0)
        if self.shrinkage_alpha is not None:
            S = np.cov(Z.T)
            a = float(self.shrinkage_alpha)
            # dos objetivos de shrinkage estandar: identidad escalada (todas
            # las varianzas iguales, encoge tambien las varianzas) o la
            # diagonal de S (conserva las varianzas, encoge solo las
            # correlaciones hacia cero)
            if self.shrinkage_target == "diagonal":
                target = np.diag(np.diag(S))
            else:
                target = np.mean(np.diag(S)) * np.eye(S.shape[0])
            self.cov_ = (1 - a) * S + a * target
            self.shrinkage_ = a
        elif self.shrinkage:
            lw = LedoitWolf().fit(Z)
            self.cov_ = lw.covariance_
            self.shrinkage_ = lw.shrinkage_
        else:
            self.cov_ = np.cov(Z.T)
            self.shrinkage_ = 0.0
        self.rng_ = np.random.default_rng(self.random_state)
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        Z = self.rng_.multivariate_normal(self.mean_, self.cov_, size=n_samples)
        if self.marginal != "rank_gauss":
            return Z
        # deshace la transformacion rank-gauss: N(0,1) -> uniforme -> cuantil
        # empirico de la columna real correspondiente
        out = np.empty_like(Z)
        for j, col_sorted in enumerate(self.sorted_cols_):
            u = np.clip(norm.cdf(Z[:, j]), 1e-9, 1 - 1e-9)
            out[:, j] = np.quantile(col_sorted, u)
        return out


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

    def __init__(self, n_iters: int = 15, grid_size: int = 300,
                 rotation: str = "random", random_state: int = 42):
        """`rotation`: que rotacion se aplica entre gaussianizaciones.
          - 'random': rotacion ortogonal aleatoria (via QR).
          - 'pca'   : rotacion a los componentes principales de los datos
            en esa iteracion. Es la variante clasica del articulo original
            de RBIG: alinear con los ejes de maxima varianza suele
            gaussianizar en menos iteraciones que rotar al azar."""
        self.n_iters = n_iters
        self.grid_size = grid_size
        self.rotation = rotation
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

            if self.rotation == "pca":
                # rotacion a componentes principales: los autovectores de la
                # covarianza forman una matriz ortogonal, asi que se invierte
                # igual (transpuesta) que la rotacion aleatoria
                cov = np.cov(G.T)
                _, eigvecs = np.linalg.eigh(cov)
                R = eigvecs[:, ::-1]  # de mayor a menor varianza
            else:
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
    """GAN totalmente conectada sobre el vector aplanado, en el mismo
    espiritu que Taller_GANs.ipynb (generador/discriminador densos,
    entrenamiento adversarial por batches con ratio adaptativo D/G). La
    arquitectura de las redes vive en
    `src.modelos.build_gan_generator/build_gan_discriminator`; esta clase
    orquesta el entrenamiento y guarda el historial de losses.

    El entrenamiento usa pasos manuales de `tf.GradientTape` en vez del
    truco de congelar el discriminador (`discriminator.trainable = False`
    + modelo combinado) de Keras 1/2: ese truco depende de que Keras fije
    la lista de variables entrenables al compilar y no la actualice
    despues, algo que Keras 3 ya no garantiza (`trainable_variables` se
    evalua de forma dinamica). Con `GradientTape` se piden gradientes SOLO
    de `disc_.trainable_variables` (paso D) o SOLO de
    `gen_.trainable_variables` (paso G, con `disc_(..., training=False)`
    dentro de la cinta pero sin pedirle gradientes) — el mismo
    entrenamiento adversarial con ratio D/G adaptativo, sin depender de
    como cada version de Keras gestione `.trainable` internamente."""

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

        # El generador termina en activation='tanh', con rango util en
        # [-1, 1]. Nuestras columnas (retorno diario, volatilidad
        # realizada, ...) valen tipicamente 0.01-0.03 en magnitud: sin
        # reescalar, el generador solo usaria una rebanada minuscula del
        # rango de tanh cerca de 0. Se reescala cada columna por su
        # percentil 99.5 de |valor| antes de entrenar (asi se aprovecha el
        # rango completo de tanh) y se deshace al muestrear — practica
        # estandar en GANs, no cambia el modelo, solo la escala en la que
        # opera.
        self.scale_ = np.maximum(np.percentile(np.abs(XY_flat), 99.5, axis=0), 1e-6).astype("float32")
        XY_scaled = XY_flat / self.scale_

        self.gen_ = modelos.build_gan_generator(self.latent_dim, d, self.gen_hidden)
        self.disc_ = modelos.build_gan_discriminator(d, self.disc_hidden)

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
            # generador: darle al discriminador varios pasos por cada
            # actualizacion del generador ayuda a que la señal de
            # gradiente que recibe el generador sea mas informativa.
            for _ in range(self.d_steps_per_g):
                batch_discr = max(1, int(round(ratio * batch)))
                idx = np.random.randint(0, len(XY_scaled) - batch_discr) if len(XY_scaled) > batch_discr else 0
                legit = XY_scaled[idx : idx + batch_discr]

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
        synth_scaled = self.gen_.predict(noise, verbose=0)
        return synth_scaled * self.scale_  # deshace el reescalado de fit()


GENERATOR_REGISTRY = {
    "noise": NoiseGenerator,
    "gaussian": GaussianGenerator,
    "rbig": RBIGGenerator,
    "gan": GANGenerator,
}
