"""
core/activations.py
-------------------
Elementwise activation functions for the ELM hidden layer.

Each function takes a numpy array Z of shape (n, m) and returns
an array of the same shape with the activation applied elementwise.
"""

import numpy as np


def tanh(Z):
    """Hyperbolic tangent. Maps R -> (-1, +1)."""
    return np.tanh(Z)


def relu(Z):
    """Rectified Linear Unit. Maps R -> [0, +inf)."""
    return np.maximum(0.0, Z)


def sigmoid(Z):
    """Logistic sigmoid. Maps R -> (0, 1)."""
    return 1.0 / (1.0 + np.exp(-Z))


ACTIVATIONS = {
    'tanh': tanh,
    'relu': relu,
    'sigmoid': sigmoid,
}


def get_activation(name):
    """
    Return activation function by name.

    Parameters
    ----------
    name : str
        One of 'tanh', 'relu', 'sigmoid'.

    Returns
    -------
    fn : callable
    """
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]
