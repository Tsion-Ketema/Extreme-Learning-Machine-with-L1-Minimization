"""
core/elm.py
-----------
Extreme Learning Machine (ELM) hidden-layer construction.

The ELM feature matrix is built as:

    X = sigma(Input @ W1.T)     shape: (n, m)

W1 is generated once from a random distribution and never updated.
Only the output weights w are trained (see core/solvers.py).

Pipeline:
    Input (n x p)  -->  [W1 + sigma]  -->  X (n x m)  -->  [w]  -->  y_hat (n x 1)
"""

import numpy as np
from core.activations import get_activation


def random_hidden_weights(m, p, seed=None):
    """
    Generate the fixed random hidden-layer weight matrix W1.

    Entries are drawn i.i.d. from Uniform(-1, 1).
    W1 is generated once and frozen for the entire training process.

    Parameters
    ----------
    m : int
        Number of hidden neurons.
        Determines the number of columns in the feature matrix X.
    p : int
        Number of input features (columns in Input matrix).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    W1 : ndarray, shape (m, p)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(m, p))


def build_feature_matrix(Input, W1, activation='tanh'):
    """
    Compute the ELM feature matrix X = sigma(Input @ W1.T).

    Each row of X is one sample seen through the eyes of m neurons.
    Each column of X is what one specific neuron says about all n samples.

    Parameters
    ----------
    Input : ndarray, shape (n, p)
        Raw input data. n samples, p features.
    W1 : ndarray, shape (m, p)
        Fixed random hidden-layer weight matrix.
    activation : str
        Activation function name. See core/activations.py.

    Returns
    -------
    X : ndarray, shape (n, m)
        ELM feature matrix.
        Shape: (n samples) x (m neurons).
    """
    sigma = get_activation(activation)
    Z = Input @ W1.T        # shape (n, m) — raw activations before sigma
    return sigma(Z)         # shape (n, m) — feature matrix


def predict(X, w):
    """
    Compute predictions for all samples given feature matrix X and weights w.

    For one sample:   y_hat_i = X[i, :] @ w   (dot product, one number)
    For all samples:  y_hat   = X @ w          (matrix-vector product, n numbers)

    Parameters
    ----------
    X : ndarray, shape (n, m)
        ELM feature matrix.
    w : ndarray, shape (m,)
        Output weight vector (one weight per neuron).

    Returns
    -------
    y_hat : ndarray, shape (n,)
        Predicted values. One prediction per sample.
    """
    return X @ w


def predict_all_targets(X, weights):
    """
    Compute predictions for all 4 targets by stacking 4 separate (n x 1) results.

    Each target has its own weight vector. The final matrix is (n x 4):
        column 0 = X @ w1  (predictions for TARGET_1)
        column 1 = X @ w2  (predictions for TARGET_2)
        column 2 = X @ w3  (predictions for TARGET_3)
        column 3 = X @ w4  (predictions for TARGET_4)

    Parameters
    ----------
    X : ndarray, shape (n, m)
        ELM feature matrix.
    weights : list of 4 ndarrays, each shape (m,)
        Output weight vectors, one per target.

    Returns
    -------
    Y_hat : ndarray, shape (n, 4)
        All predictions stacked column by column.
    """
    return np.column_stack([X @ w for w in weights])
