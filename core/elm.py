"""
core/elm.py
-----------
Extreme Learning Machine hidden-layer construction.

The ELM feature matrix is:
    X = sigma( Input @ W1.T )     shape (n, m)

W1 is generated once from Uniform(-1,1) and NEVER updated.
Only the output weight vector w is trained (see core/solvers.py).

Full pipeline:
    Input (n x p)
        --> [multiply by W1, apply sigma]
        --> X (n x m)
        --> [multiply by w]
        --> y_hat (n x 1)
"""

import numpy as np
from core.activations import get_activation


def random_hidden_weights(m, p, seed=None):
    """
    Generate W1 — the fixed random hidden-layer weight matrix.

    Entries drawn i.i.d. from Uniform(-1, 1).
    Generated once and never changed during training.

    Parameters
    ----------
    m    : int  — number of hidden neurons
    p    : int  — number of input features
    seed : int  — random seed for reproducibility

    Returns
    -------
    W1 : ndarray, shape (m, p)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(m, p))


def build_feature_matrix(Input, W1, activation='tanh'):
    """
    Compute the ELM feature matrix X = sigma(Input @ W1.T).

    Each row of X = one sample's activations across all m neurons.
    Each col of X = one neuron's output across all n samples.

    Parameters
    ----------
    Input      : ndarray, shape (n, p)  — raw input data
    W1         : ndarray, shape (m, p)  — fixed hidden weights
    activation : str                    — 'tanh', 'relu', or 'sigmoid'

    Returns
    -------
    X : ndarray, shape (n, m)
    """
    sigma = get_activation(activation)
    Z = Input @ W1.T      # shape (n, m): raw activations before sigma
    return sigma(Z)        # shape (n, m): ELM feature matrix


def predict(X, w):
    """
    Compute predictions: y_hat = X @ w.

    Parameters
    ----------
    X : ndarray, shape (n, m)
    w : ndarray, shape (m,)

    Returns
    -------
    y_hat : ndarray, shape (n,)
    """
    return X @ w


def predict_all_targets(X, weights):
    """
    Compute predictions for all 4 targets and stack into one matrix.

    Runs 4 separate dot products (one per target) and stacks the
    results column by column into a (n x 4) matrix.

    Parameters
    ----------
    X       : ndarray, shape (n, m)
    weights : list of 4 ndarrays each shape (m,)

    Returns
    -------
    Y_hat : ndarray, shape (n, 4)
    """
    return np.column_stack([X @ w for w in weights])
