import numpy as np


def l1_error(A, w, y):
    return np.sum(np.abs(A @ w - y))
