"""
Analiza los CSV de las busquedas de hiperparametros y elige la mejor
configuracion de cada familia.

*** Regla de seleccion: "un error estandar" ***
No se coge la configuracion con el mejor numero. Al evaluar cientos o miles
de configuraciones contra el mismo conjunto de validacion, el minimo esta
sesgado a la baja: parte de esa ventaja es ruido del propio conjunto de
validacion, no calidad real (problema clasico de seleccion multiple). La
regla estandar para corregirlo es:

  1. localizar el mejor valor de validacion `v*` y su error estandar `se`,
  2. quedarse con TODAS las configuraciones dentro de `v* + se`
     (estadisticamente empatadas con la mejor),
  3. de entre esas, elegir la MAS SIMPLE / MAS ESTABLE.

Asi se evita premiar a la configuracion que tuvo suerte y se prefiere la
que generalizara mejor. Para el predictor "mas simple" = menos parametros;
para los generadores, la de menor dispersion entre semillas.

Uso:
    python scripts/analizar_hpsearch.py                # todo lo que haya
    python scripts/analizar_hpsearch.py --que generadores
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

PARAM_COLS = [
    "family", "sigma", "relative", "noise_dist", "df", "shrinkage",
    "shrinkage_alpha", "shrinkage_target", "marginal", "n_iters", "grid_size",
    "rotation", "latent_dim", "epochs", "batch_size", "gen_hidden",
    "disc_hidden", "learning_rate", "d_steps_per_g", "dropout", "l2",
    "hidden_units", "conv_filters", "lstm_units", "dense_units",
    "global_pool", "kernel_size", "recurrent_dropout",
]


def _load(patron: str) -> pd.DataFrame:
    files = glob.glob(str(config.TABLES_DIR / patron))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _params_de(df: pd.DataFrame) -> list[str]:
    return [c for c in PARAM_COLS if c in df.columns]


def elegir_una_se(df: pd.DataFrame, metrica: str, simplicidad: str,
                  menor_mejor: bool = True) -> tuple[pd.Series, pd.DataFrame, float]:
    """Aplica la regla de un error estandar. Devuelve (elegida, empatadas,
    umbral)."""
    d = df.dropna(subset=[metrica]).copy()
    if menor_mejor:
        mejor = d[metrica].min()
        se = d[metrica].std() / max(len(d) ** 0.5, 1)
        umbral = mejor + se
        empatadas = d[d[metrica] <= umbral]
    else:
        mejor = d[metrica].max()
        se = d[metrica].std() / max(len(d) ** 0.5, 1)
        umbral = mejor - se
        empatadas = d[d[metrica] >= umbral]
    elegida = empatadas.sort_values([simplicidad, metrica]).iloc[0]
    return elegida, empatadas, umbral


def analizar_generadores():
    df = _load("hpsearch_gen_w*.csv")
    if df.empty:
        print("(sin resultados de generadores todavia)")
        return
    # cada fila es una evaluacion; se agrupa por configuracion distinta
    print(f"\n{'='*78}\nGENERADORES — {len(df)} evaluaciones\n{'='*78}")
    resumen = []
    for fam in ["noise", "gaussian", "rbig", "gan"]:
        sub = df[df.family == fam]
        if sub.empty:
            print(f"\n--- {fam.upper()}: sin evaluaciones ---")
            continue
        keys = [c for c in _params_de(sub) if c != "family" and sub[c].notna().any()]
        g = (sub.groupby(keys, dropna=False)
                .agg(n=("mmd_mean", "size"),
                     mmd=("mmd_mean", "mean"),
                     w1=("wasserstein_mean_mean", "mean"),
                     frob=("frobenius_corr_mean", "mean"),
                     std_ratio=("std_ratio_mean_mean", "mean"))
                .reset_index())
        # PUNTUACION COMPUESTA. El MMD por si solo no sirve para ordenar
        # las buenas: su estimador insesgado fluctua alrededor de 0 y se
        # recorta ahi, asi que decenas de configuraciones empatan en
        # 0.000000 y el desempate seria arbitrario. Se combinan las tres
        # metricas por RANGO (no por valor, porque estan en escalas muy
        # distintas): rango medio de MMD, Wasserstein y Frobenius.
        g["score"] = (g["mmd"].rank() + g["w1"].rank() + g["frob"].rank()) / 3.0
        elegida, empatadas, umbral = elegir_una_se(g, "score", "w1")
        print(f"\n--- {fam.upper()}: {len(g)} configuraciones distintas "
              f"| {len(empatadas)} empatadas dentro de 1 e.e. ---")
        print("  TOP 5 por puntuacion compuesta (rango medio de MMD/W1/Frobenius):")
        print(g.sort_values("score").head(5).to_string(index=False))
        print("  >>> ELEGIDA (regla 1 e.e., desempate por Wasserstein):")
        print("      " + " | ".join(f"{k}={elegida[k]}" for k in keys))
        print(f"      MMD={elegida['mmd']:.6f}  W1={elegida['w1']:.4f}  "
              f"Frobenius={elegida['frob']:.3f}  std_ratio={elegida['std_ratio']:.3f}")
        if fam == "noise":
            print("      NOTA: en el generador de Ruido estas metricas premian a "
                  "sigma->0,\n      que es memorizacion pura (copiar datos reales). "
                  "Es un artefacto\n      esperable del baseline trivial; la eleccion "
                  "final debe confirmarse\n      con el rendimiento aguas abajo, no solo "
                  "con fidelidad distribucional.")
        resumen.append({"generador": fam, **{k: elegida[k] for k in keys},
                        "mmd": elegida["mmd"], "w1": elegida["w1"],
                        "frobenius": elegida["frob"]})
    if resumen:
        out = config.TABLES_DIR / "hpsearch_mejores_generadores.csv"
        pd.DataFrame(resumen).to_csv(out, index=False)
        print(f"\nGuardado -> {out}")


def analizar_predictor():
    variantes = [
        ("hpsearch_A_real_single_w*.csv", "ETAPA A · corte unico"),
        ("hpsearch_A_real_w*.csv", "ETAPA A · corte unico (ficheros antiguos)"),
        ("hpsearch_A_real_wf_emb60_w*.csv", "ETAPA A · walk-forward CON purga"),
        ("hpsearch_A_real_wf_emb0_w*.csv", "ETAPA A · walk-forward SIN purga"),
        ("hpsearch_B_full_wf_emb60_w*.csv", "ETAPA B · walk-forward CON purga"),
        ("hpsearch_B_full_wf_emb0_w*.csv", "ETAPA B · walk-forward SIN purga"),
        ("hpsearch_B_full_single_w*.csv", "ETAPA B · corte unico"),
    ]
    resumen = []
    for patron, titulo in variantes:
        df = _load(patron)
        if df.empty:
            continue
        print(f"\n{'='*78}\n{titulo} — {len(df)} configuraciones\n{'='*78}")
        for fam in ["dense", "cnn", "rnn"]:
            sub = df[df.family == fam].dropna(subset=["best_val_mean"])
            if sub.empty:
                continue
            elegida, empatadas, umbral = elegir_una_se(
                sub, "best_val_mean", "n_params_mean"
            )
            keys = [c for c in _params_de(sub) if c != "family" and sub[c].notna().any()]
            estab = (f" | std entre cortes={elegida['best_val_std_folds']:.6f}"
                     if "best_val_std_folds" in elegida else "")
            print(f"\n  {fam.upper()}: {len(sub)} configs | {len(empatadas)} dentro de 1 e.e. "
                  f"(umbral {umbral:.6f})")
            print("    >>> " + " | ".join(f"{k}={elegida[k]}" for k in keys))
            print(f"        val={elegida['best_val_mean']:.6f} "
                  f"params={int(elegida['n_params_mean'])}{estab}")
            resumen.append({"protocolo": titulo, "familia": fam,
                            **{k: elegida[k] for k in keys},
                            "val": elegida["best_val_mean"],
                            "n_params": elegida["n_params_mean"],
                            "std_cortes": elegida.get("best_val_std_folds", float("nan"))})
    if resumen:
        out = config.TABLES_DIR / "hpsearch_mejores_predictor.csv"
        pd.DataFrame(resumen).to_csv(out, index=False)
        print(f"\nGuardado -> {out}")
        print("\nCOMPARATIVA DE PROTOCOLOS (mismo espacio de busqueda, distinta validacion):")
        piv = pd.DataFrame(resumen).pivot_table(
            index="familia", columns="protocolo", values="val"
        )
        print(piv.round(6).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--que", choices=["generadores", "predictor", "todo"], default="todo")
    args = ap.parse_args()
    if args.que in ("generadores", "todo"):
        analizar_generadores()
    if args.que in ("predictor", "todo"):
        analizar_predictor()


if __name__ == "__main__":
    main()
