import numpy as np

from core.activations import sigmoid

RNG_SEED = 42
RNG = np.random.default_rng(RNG_SEED)

ACTIVATION = sigmoid
ACTIVATION_NAME = "sigmoid"

N_HIDDEN = 60
SCALE = 5.0

DATASET_PARAMS = {
    "n": 120,
    "noise_std": 0.15,
    "outlier_frac": 0.12,
    "seed": 0,
}

IRLS_PARAMS = {
    "max_iter": 400,
    "tol": 1e-10,
    "thresh": 1e-6,
}

SMOOTHED_GRADIENT_PARAMS = {
    "mu_init": 0.5,
    "mu_decay": 0.5,
    "mu_min": 1e-6,
    "stage_max_iter": 1000,
    "tol": 1e-8,
}

FIGURE_PATH = "project24_results.png"
