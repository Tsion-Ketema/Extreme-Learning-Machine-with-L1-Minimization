"""
utils/data.py
-------------
Loading and splitting the ML-CUP 2025 dataset.

CSV format (comment lines start with #):
    ID, feat_1, ..., feat_12, TARGET_1, TARGET_2, TARGET_3, TARGET_4

The four targets are independent regression problems.
One ELM model is trained per target.
The final prediction matrix is (n x 4) — four (n x 1) columns stacked.
"""

import numpy as np


def load_cup(filepath):
    """
    Load the ML-CUP 2025 training CSV.

    Skips comment lines. Drops the ID column. Returns standardized inputs
    and the 4 target columns as separate arrays.

    Parameters
    ----------
    filepath : str
        Path to training CSV (e.g., 'dataset/ML-CUP25-TR.csv').

    Returns
    -------
    Input : ndarray, shape (n, 12)
        Standardized input features (zero mean, unit variance).
    Y : ndarray, shape (n, 4)
        Target matrix. Each column = one independent regression problem.
    scaler_mean : ndarray, shape (12,)
        Feature means (needed to standardize test set the same way).
    scaler_std : ndarray, shape (12,)
        Feature standard deviations.
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([float(x) for x in line.split(',')])

    data = np.array(rows)    # shape (n, 17): ID + 12 inputs + 4 targets
    Input_raw = data[:, 1:13]     # columns 1..12
    Y = data[:, 13:]      # last 4 columns

    # standardize: zero mean, unit variance — computed on training set only
    mean = Input_raw.mean(axis=0)
    std = Input_raw.std(axis=0)
    std[std == 0] = 1.0           # guard against constant columns
    Input = (Input_raw - mean) / std

    return Input, Y, mean, std


def load_cup_test(filepath, scaler_mean, scaler_std):
    """
    Load the ML-CUP 2025 blind test CSV (no targets).

    Apply the same standardization as the training set.

    Parameters
    ----------
    filepath : str
        Path to blind test CSV (e.g., 'dataset/ML-CUP25-TS.csv').
    scaler_mean : ndarray, shape (12,)
    scaler_std  : ndarray, shape (12,)

    Returns
    -------
    Input : ndarray, shape (n_test, 12)
    ids   : ndarray, shape (n_test,)
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([float(x) for x in line.split(',')])

    data = np.array(rows)
    ids = data[:, 0].astype(int)
    Input_raw = data[:, 1:]
    Input = (Input_raw - scaler_mean) / scaler_std

    return Input, ids


def train_test_split(Input, Y, test_size=0.1, seed=42):
    """
    Randomly split data into train and test subsets.

    Parameters
    ----------
    Input     : ndarray, shape (n, 12)
    Y         : ndarray, shape (n, 4)
    test_size : float  — fraction held out for testing (default 0.1 = 10%)
    seed      : int

    Returns
    -------
    Input_tr, Input_te, Y_tr, Y_te : ndarrays
    """
    rng = np.random.default_rng(seed)
    n = Input.shape[0]
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return Input[train_idx], Input[test_idx], Y[train_idx], Y[test_idx]
