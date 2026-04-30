"""
core/activations.py
-------------------
Elementwise activation functions for the ELM hidden layer.

Each function takes Z of shape (n, m) and returns the same shape.
No library calls beyond numpy elementwise operations.
"""

import numpy as np


def tanh(Z):
    """tanh(z) — maps R -> (-1, +1). Smooth and bounded."""
    return np.tanh(Z)


def relu(Z):
    """ReLU(z) = max(0, z) — maps R -> [0, +inf)."""
    return np.maximum(0.0, Z)


def sigmoid(Z):
    """sigmoid(z) = 1/(1+e^{-z}) — maps R -> (0, 1)."""
    return 1.0 / (1.0 + np.exp(-Z))


ACTIVATIONS = {'tanh': tanh, 'relu': relu, 'sigmoid': sigmoid}


def get_activation(name):
    """Return activation function by name string."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. "
                         f"Choose from: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]
