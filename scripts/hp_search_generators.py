"""
Busqueda de hiperparametros para los 4 modelos generativos (Ruido,
Gaussiana, RBIG, GAN).

Objetivo: que la distribucion CONJUNTA sintetica
`[retorno, realized_vol, open_30m_ret, close_30m_ret, hl_range]` se parezca
lo mas posible a la real. Se mide con tres metricas complementarias sobre
un conjunto de referencia REAL que el generador no ha visto (`pool_val`,
el mismo split que usa el notebook 02):

  - MMD (Maximum Mean Discrepancy, kernel RBF): metrica estandar para
    comparar dos distribuciones multivariantes. Captura marginales Y
    dependencia a la vez, asi que es la metrica PRINCIPAL de esta busqueda.
  - Wasserstein-1 medio sobre las 5 marginales: cuanto se desplaza cada
    variable por separado.
  - Distancia de Frobenius entre matrices de correlacion: la misma que ya
    reporta el notebook 02, para poder comparar con lo que hay.

REGLA: el generador se ajusta sobre `pool_train` y se evalua sobre
`pool_val`. Nada de val/test del predictor entra aqui (el pool ya excluye
`VAL_START_DATE` en adelante, ver notebook 02).

Uso:
    python scripts/hp_search_generators.py --minutes 1000 --worker 0
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

_THREADS = os.environ.get("HPSEARCH_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _THREADS)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, features as feat, generators as gen  # noqa: E402

N_SYNTH = 20_000        # muestras sinteticas por configuracion
MMD_SUBSAMPLE = 1500    # MMD es O(n^2): se submuestrea para que sea viable


# ---------------------------------------------------------------------------
# Metricas de fidelidad distribucional
# ---------------------------------------------------------------------------
def mmd_rbf(real: np.ndarray, synth: np.ndarray, rng, subsample=MMD_SUBSAMPLE) -> float:
    """MMD^2 con kernel RBF entre dos muestras multivariantes. Ambas se
    estandarizan con la media/desviacion de los datos REALES (asi la
    metrica no premia a un generador solo por acertar la escala) y el
    ancho de banda del kernel se fija con la heuristica de la mediana."""
    mu, sd = real.mean(axis=0), real.std(axis=0) + 1e-12
    R = (real - mu) / sd
    S = (synth - mu) / sd

    if len(R) > subsample:
        R = R[rng.choice(len(R), subsample, replace=False)]
    if len(S) > subsample:
        S = S[rng.choice(len(S), subsample, replace=False)]

    def sqdist(A, B):
        return (
            np.sum(A**2, axis=1)[:, None]
            + np.sum(B**2, axis=1)[None, :]
            - 2.0 * A @ B.T
        )

    d_rr, d_ss, d_rs = sqdist(R, R), sqdist(S, S), sqdist(R, S)
    # heuristica de la mediana sobre las distancias reales-reales
    med = np.median(d_rr[d_rr > 0]) if np.any(d_rr > 0) else 1.0
    gamma = 1.0 / max(med, 1e-12)

    k_rr, k_ss, k_rs = np.exp(-gamma * d_rr), np.exp(-gamma * d_ss), np.exp(-gamma * d_rs)
    n, m = len(R), len(S)
    # estimadores insesgados (se excluye la diagonal)
    mmd2 = (
        (k_rr.sum() - np.trace(k_rr)) / (n * (n - 1))
        + (k_ss.sum() - np.trace(k_ss)) / (m * (m - 1))
        - 2.0 * k_rs.mean()
    )
    return float(max(mmd2, 0.0))


def evaluate_synth(real: np.ndarray, synth: np.ndarray, rng) -> dict:
    ok = np.isfinite(synth).all(axis=1)
    synth = synth[ok]
    if len(synth) < 100:
        raise ValueError("el generador produjo casi todo NaN/inf")

    mu, sd = real.mean(axis=0), real.std(axis=0) + 1e-12
    w1 = float(np.mean([
        wasserstein_distance((real[:, j] - mu[j]) / sd[j], (synth[:, j] - mu[j]) / sd[j])
        for j in range(real.shape[1])
    ]))
    frob = float(np.linalg.norm(np.corrcoef(real.T) - np.corrcoef(synth.T)))
    return {
        "mmd": mmd_rbf(real, synth, rng),
        "wasserstein_mean": w1,
        "frobenius_corr": frob,
        "frac_valida": float(ok.mean()),
        "std_ratio_mean": float(np.mean(synth.std(axis=0) / (real.std(axis=0) + 1e-12))),
    }


# ---------------------------------------------------------------------------
# Espacios de busqueda por generador
# ---------------------------------------------------------------------------
def sample_config(rng, family: str) -> dict:
    # Los espacios se dimensionan para tener >=100 configuraciones DISTINTAS
    # por familia: con menos, la busqueda aleatoria agota el espacio en
    # minutos y solo reevalua lo mismo (se observo 130x de redundancia en la
    # gaussiana con su espacio original de 2 configuraciones).
    if family == "noise":
        # `sigma` es CONTINUO: se muestrea log-uniforme en [0.005, 0.5] en
        # vez de sobre una rejilla fija. Para un parametro continuo esto es
        # mejor practica en busqueda aleatoria (no se desperdicia el
        # presupuesto repitiendo los mismos valores) y hace que el espacio
        # no se agote.
        sigma = float(np.exp(rng.uniform(np.log(0.005), np.log(0.5))))
        dist = str(rng.choice(["normal", "student_t"]))
        cfg = {"family": family, "sigma": round(sigma, 5),
               "relative": bool(rng.integers(2)), "noise_dist": dist}
        cfg["df"] = float(rng.choice([3, 4, 5, 8, 10])) if dist == "student_t" else 0.0
        return cfg
    if family == "gaussian":
        # `shrinkage_alpha` tambien continuo en [0, 1] (o None = que lo
        # elija Ledoit-Wolf), cruzado con 2 objetivos de shrinkage y 2
        # espacios de marginal.
        use_lw = rng.random() < 0.15   # 15% de las veces, Ledoit-Wolf automatico
        return {
            "family": family,
            "shrinkage": bool(rng.integers(2)),
            "shrinkage_alpha": None if use_lw else round(float(rng.uniform(0.0, 1.0)), 5),
            "shrinkage_target": str(rng.choice(["identity", "diagonal"])),
            "marginal": str(rng.choice(["gaussian", "rank_gauss"])),
        }
    if family == "rbig":
        # 11 n_iters x 8 grid_size x 2 rotaciones = 176 configs
        return {
            "family": family,
            "n_iters": int(rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100])),
            "grid_size": int(rng.choice([100, 200, 300, 400, 600, 800, 1000, 1500])),
            "rotation": str(rng.choice(["random", "pca"])),
        }
    if family == "gan":
        # Espacio acotado por COSTE y por lo observado: el GAN vainilla en
        # d=5 colapsa de modo, y alargar el entrenamiento lo EMPEORA en vez
        # de arreglarlo, asi que epochs>1500 y d_steps=5 solo gastan tiempo
        # (una config de 3000 epochs x d_steps=5 tarda ~20 min y no aporta).
        gen_h = [(32, 32), (64, 64), (64, 128, 64), (128, 128), (32, 64, 32), (16, 32, 16)]
        disc_h = [(16, 8), (32, 16), (64, 32), (128, 64), (64, 64)]
        # `d_steps_per_g` resulto ser el parametro MAS influyente (W1 medio
        # 1.20 -> 0.83 -> 0.68 al pasar de 1 a 2 a 3 pasos) y la tendencia
        # seguia subiendo en el borde de la rejilla anterior, asi que se
        # amplia hasta 6. Las epocas, en cambio, saturan hacia 1200-1500,
        # asi que ahi el rango se mantiene y se concentra donde funciona.
        return {
            "family": family,
            "latent_dim": int(rng.choice([8, 16, 32, 48])),
            "epochs": int(rng.choice([800, 1200, 1500, 2000])),
            "batch_size": int(rng.choice([64, 128, 256])),
            "gen_hidden": gen_h[rng.integers(len(gen_h))],
            "disc_hidden": disc_h[rng.integers(len(disc_h))],
            "learning_rate": float(rng.choice([1e-3, 5e-4, 3e-4, 1e-4])),
            "d_steps_per_g": int(rng.choice([2, 3, 4, 5, 6])),
        }
    raise ValueError(family)


def build_and_sample(cfg: dict, pool_train: np.ndarray, seed: int) -> np.ndarray:
    f = cfg["family"]
    if f == "noise":
        g = gen.NoiseGenerator(
            sigma=cfg["sigma"], relative=cfg["relative"],
            noise_dist=cfg.get("noise_dist", "normal"),
            df=cfg.get("df") or 4.0, random_state=seed,
        )
    elif f == "gaussian":
        g = gen.GaussianGenerator(
            shrinkage=cfg["shrinkage"], shrinkage_alpha=cfg.get("shrinkage_alpha"),
            shrinkage_target=cfg.get("shrinkage_target", "identity"),
            marginal=cfg.get("marginal", "gaussian"), random_state=seed,
        )
    elif f == "rbig":
        g = gen.RBIGGenerator(
            n_iters=cfg["n_iters"], grid_size=cfg["grid_size"],
            rotation=cfg.get("rotation", "random"), random_state=seed,
        )
    elif f == "gan":
        g = gen.GANGenerator(
            latent_dim=cfg["latent_dim"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            gen_hidden=cfg["gen_hidden"], disc_hidden=cfg["disc_hidden"],
            learning_rate=cfg["learning_rate"], d_steps_per_g=cfg["d_steps_per_g"],
            random_state=seed,
        )
    else:
        raise ValueError(f)
    g.fit(pool_train)
    # mismo postproceso que el notebook 02: las columnas fisicamente no
    # negativas (volatilidad, rango) se recortan a >= 0
    return feat.clip_nonnegative_pool_columns(g.sample(N_SYNTH))


def cfg_to_row(cfg: dict) -> dict:
    row = dict(cfg)
    for k in ("gen_hidden", "disc_hidden"):
        if k in row:
            row[k] = "x".join(str(v) for v in row[k])
    return row


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=1000)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--families", default="noise,gaussian,rbig,gan")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    families = tuple(args.families.split(","))
    out_path = config.TABLES_DIR / f"hpsearch_gen_w{args.worker}.csv"

    # Mismo split que el notebook 02: se ajusta en pool_train y se mide
    # contra pool_val (real, no visto por el generador).
    pool_full = np.load(config.INTERIM_DIR / "conditional_pool.npy")
    meta = pd.read_parquet(config.INTERIM_DIR / "conditional_pool_meta.parquet")
    pool = pool_full[~(meta["date"] >= pd.Timestamp(config.VAL_START_DATE)).values]
    rng0 = np.random.default_rng(42)
    shuffled = rng0.permutation(len(pool))
    n_val = max(int(0.1 * len(pool)), 500)
    pool_val, pool_train = pool[shuffled[:n_val]], pool[shuffled[n_val:]]
    print(f"[gen/w{args.worker}] train {pool_train.shape} | val {pool_val.shape} | "
          f"familias {list(families)} | presupuesto {args.minutes:.0f} min", flush=True)

    rng = np.random.default_rng(777 + 1000 * args.worker)
    metric_rng = np.random.default_rng(0)
    rows = []
    if out_path.exists():
        rows = pd.read_csv(out_path).to_dict("records")
        print(f"[gen/w{args.worker}] reanudando: {len(rows)} configs", flush=True)

    # Deduplicacion: los espacios de ruido/gaussiana/RBIG son pequenos y se
    # agotan enseguida, asi que sin esto los workers reevaluan una y otra
    # vez las mismas configuraciones (se observo 130x de redundancia en la
    # gaussiana). Se leen tambien los CSV de los OTROS workers para no
    # repetir entre procesos.
    def cfg_key(c: dict) -> tuple:
        return tuple(sorted((k, str(v)) for k, v in cfg_to_row(c).items()))

    seen = set()
    for f in (config.TABLES_DIR).glob("hpsearch_gen_w*.csv"):
        try:
            prev = pd.read_csv(f)
        except Exception:
            continue
        param_cols = [c for c in prev.columns
                      if c not in ("seconds",) and not c.endswith(("_mean", "_std"))]
        for _, r in prev[param_cols].iterrows():
            d = {k: v for k, v in r.items() if pd.notna(v)}
            seen.add(tuple(sorted((k, str(v)) for k, v in d.items())))
    print(f"[gen/w{args.worker}] {len(seen)} configuraciones ya evaluadas se saltaran", flush=True)

    t_end = time.time() + args.minutes * 60
    i = 0
    n_skip = 0
    while time.time() < t_end:
        family = families[i % len(families)]
        cfg = sample_config(rng, family)
        i += 1
        k = cfg_key(cfg)
        if k in seen:
            n_skip += 1
            if n_skip % 500 == 0:
                print(f"[gen/w{args.worker}] {n_skip} repetidas saltadas "
                      f"(espacio de '{family}' probablemente agotado)", flush=True)
            continue
        seen.add(k)
        try:
            t0 = time.time()
            per_seed = []
            for s in range(args.seeds):
                synth = build_and_sample(cfg, pool_train, seed=42 + s)
                per_seed.append(evaluate_synth(pool_val, synth, metric_rng))
                del synth
                gc.collect()
            df = pd.DataFrame(per_seed)
            metrics = {f"{c}_mean": float(df[c].mean()) for c in df.columns}
            metrics["mmd_std"] = float(df["mmd"].std())
            row = {**cfg_to_row(cfg), **metrics, "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            # guardado atomico: temporal + rename, para que morir a mitad
            # de escritura no corrompa el CSV bueno anterior
            tmp = out_path.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, out_path)
            print(
                f"[gen/w{args.worker}][{len(rows):4d}] {family:8s} "
                f"mmd={metrics['mmd_mean']:.5f} "
                f"w1={metrics['wasserstein_mean_mean']:.4f} "
                f"frob={metrics['frobenius_corr_mean']:.3f} "
                f"({row['seconds']:.0f}s)",
                flush=True,
            )
        except Exception as e:
            print(f"[gen/w{args.worker}] config fallida ({family}): "
                  f"{type(e).__name__}: {e}", flush=True)

    print(f"[gen/w{args.worker}] TERMINADO: {len(rows)} configs -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
