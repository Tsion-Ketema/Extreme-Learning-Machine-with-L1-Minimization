import numpy as np


class ExtremeLearningMachine:
    def __init__(self, n_hidden, activation, rng, scale=1.0):
        self.n_hidden = n_hidden
        self.activation = activation
        self.scale = scale
        self.rng = rng
        self.W1 = None

    def fit_features(self, X):
        n_input = X.shape[1]
        self.W1 = self.rng.standard_normal(
            (self.n_hidden, n_input)) * self.scale
        return self.activation(X @ self.W1.T)

    def transform(self, X):
        return self.activation(X @ self.W1.T)
