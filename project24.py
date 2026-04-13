"""
Project 24: Extreme Learning Machine with L1 Minimisation.

Model:
    y = w sigma(W1 x)

W1 is a fixed random hidden-layer matrix, sigma is an elementwise activation,
and the output weights w are chosen by minimizing an L1 objective over the ELM
feature matrix. Two manual solvers are compared:
    (A1) Iteratively Reweighted Least Squares
    (A2) Smoothed Gradient continuation
"""

import numpy as np

from config import (
    ACTIVATION,
    ACTIVATION_NAME,
    DATASET_PARAMS,
    FIGURE_PATH,
    IRLS_PARAMS,
    N_HIDDEN,
    RNG_SEED,
    SCALE,
    SMOOTHED_GRADIENT_PARAMS,
)
from core.elm import ExtremeLearningMachine
from core.solvers import irls_l1, smoothed_gradient_l1
from utils.data import generate_dataset
from utils.metrics import l1_error
from viz.plots import plot_results


def run_experiment(output_path=FIGURE_PATH):
    X_raw, y, y_clean = generate_dataset(**DATASET_PARAMS)

    elm = ExtremeLearningMachine(
        n_hidden=N_HIDDEN,
        activation=ACTIVATION,
        rng=np.random.default_rng(RNG_SEED),
        scale=SCALE,
    )
    Phi = elm.fit_features(X_raw)

    w_irls, hist_irls = irls_l1(Phi, y, **IRLS_PARAMS)
    w_sg, hist_sg = smoothed_gradient_l1(Phi, y, **SMOOTHED_GRADIENT_PARAMS)
    w_l2, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)

    summary = {
        "activation": ACTIVATION_NAME,
        "n_hidden": N_HIDDEN,
        "scale": SCALE,
        "l1_irls": l1_error(Phi, w_irls, y),
        "l1_sg": l1_error(Phi, w_sg, y),
        "l1_l2": l1_error(Phi, w_l2, y),
        "irls_iterations": len(hist_irls),
        "sg_iterations": len(hist_sg),
        "dataset_size": len(y),
        "outlier_fraction": DATASET_PARAMS["outlier_frac"],
    }
    summary["irls_vs_l2_pct"] = 100.0 * (summary["l1_l2"] - summary["l1_irls"]) / summary["l1_l2"]
    summary["sg_vs_l2_pct"] = 100.0 * (summary["l1_l2"] - summary["l1_sg"]) / summary["l1_l2"]

    figure_path = plot_results(
        X_raw=X_raw,
        y=y,
        y_clean=y_clean,
        Phi=Phi,
        w_irls=w_irls,
        w_l2=w_l2,
        w_sg=w_sg,
        hist_irls=hist_irls,
        hist_sg=hist_sg,
        W1=elm.W1,
        activation=ACTIVATION,
        output_path=output_path,
    )
    summary["figure_path"] = figure_path

    return {
        "X_raw": X_raw,
        "y": y,
        "y_clean": y_clean,
        "Phi": Phi,
        "W1": elm.W1,
        "w_irls": w_irls,
        "w_sg": w_sg,
        "w_l2": w_l2,
        "hist_irls": hist_irls,
        "hist_sg": hist_sg,
        "summary": summary,
    }


def print_summary(summary):
    print("=" * 60)
    print("Project 24 — Extreme Learning Machine + L1 Minimisation")
    print("=" * 60)
    print(f"Activation     : {summary['activation']}")
    print(f"Hidden units   : {summary['n_hidden']}")
    print(f"Scale          : {summary['scale']}")
    print(f"Samples        : {summary['dataset_size']}")
    print(f"Outlier frac   : {summary['outlier_fraction']:.2f}")
    print()
    print(f"L1 IRLS        : {summary['l1_irls']:.6f}")
    print(f"L1 Smoothed    : {summary['l1_sg']:.6f}")
    print(f"L1 L2 baseline : {summary['l1_l2']:.6f}")
    print()
    print(f"IRLS iterations: {summary['irls_iterations']}")
    print(f"SG iterations  : {summary['sg_iterations']}")
    print()
    print(f"IRLS vs L2     : {summary['irls_vs_l2_pct']:.2f}% better")
    print(f"SG vs L2       : {summary['sg_vs_l2_pct']:.2f}% better")
    print(f"Figure         : {summary['figure_path']}")


def main():
    results = run_experiment()
    print_summary(results["summary"])


if __name__ == "__main__":
    main()
