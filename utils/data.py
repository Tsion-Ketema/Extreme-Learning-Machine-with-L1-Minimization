"""
utils/data.py
-------------
Load and split the ML-CUP 2025 dataset.

CSV format (lines starting with # are comments):
    ID, feat_1, ..., feat_12, TARGET_1, TARGET_2, TARGET_3, TARGET_4

The 4 targets are treated as 4 independent regression problems.
One ELM is trained per target. Final predictions are (n x 4).
"""

import numpy as np


def load_cup(filepath):
    """
    Load ML-CUP 2025 training data.

    Skips comment lines, drops the ID column, standardizes inputs.

    Parameters
    ----------
    filepath : str — path to ML-CUP25-TR.csv

    Returns
    -------
    Input       : ndarray, shape (n, 12) — standardized features
    Y           : ndarray, shape (n, 4)  — 4 target columns
    scaler_mean : ndarray, shape (12,)   — feature means
    scaler_std  : ndarray, shape (12,)   — feature std devs
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([float(x) for x in line.split(',')])

    data = np.array(rows)   # shape (n, 17): ID + 12 inputs + 4 targets
    Input_raw = data[:, 1:13]    # columns 1-12: input features
    Y = data[:, 13:]     # last 4 columns: targets

    # Standardize inputs to zero mean and unit variance.
    # Statistics computed on training set only and reused on test set.
    mean = Input_raw.mean(axis=0)
    std = Input_raw.std(axis=0)
    std[std == 0] = 1.0          # prevent division by zero on constant cols
    Input = (Input_raw - mean) / std

    return Input, Y, mean, std


def load_cup_test(filepath, scaler_mean, scaler_std):
    """
    Load ML-CUP 2025 blind test data (no targets).

    Applies the same standardization as the training set.

    Parameters
    ----------
    filepath    : str           — path to ML-CUP25-TS.csv
    scaler_mean : ndarray (12,) — from load_cup
    scaler_std  : ndarray (12,) — from load_cup

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
    Randomly split into train and test subsets.

    Parameters
    ----------
    Input     : ndarray, shape (n, 12)
    Y         : ndarray, shape (n, 4)
    test_size : float — fraction held out (default 0.1 = 10%)
    seed      : int

    Returns
    -------
    Input_tr, Input_te, Y_tr, Y_te : ndarrays
    """
    rng = np.random.default_rng(seed)
    n = Input.shape[0]
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_size))
    return (Input[idx[n_test:]], Input[idx[:n_test]],
            Y[idx[n_test:]],     Y[idx[:n_test]])
