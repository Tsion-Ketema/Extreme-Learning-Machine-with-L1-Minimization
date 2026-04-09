import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0.0, z)


def tanh_act(z):
    return np.tanh(z)
