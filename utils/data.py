import numpy as np


def generate_dataset(n=120, noise_std=0.15, outlier_frac=0.12, seed=0):
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 1, n)
    y_clean = np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x)

    noise = rng.normal(0, noise_std, n)

    n_out = int(outlier_frac * n)
    idx_out = rng.choice(n, n_out, replace=False)
    noise[idx_out] += rng.choice([-1, 1], n_out) * rng.uniform(1.5, 3.0, n_out)

    y = y_clean + noise

    return x[:, None], y, y_clean
