from core.elm import ExtremeLearningMachine
from core.solvers import irls_l1, smoothed_gradient_l1
from utils.data import generate_dataset
from utils.metrics import l1_error
from config import *


def main():
    # Data
    X, y, y_clean = generate_dataset()

    # ELM
    elm = ExtremeLearningMachine(
        n_hidden=N_HIDDEN,
        activation=ACTIVATION,
        rng=RNG,
        scale=SCALE
    )

    Phi = elm.fit_features(X)

    # IRLS
    w_irls, hist_irls = irls_l1(Phi, y)

    # Smoothed Gradient
    w_sg, hist_sg = smoothed_gradient_l1(Phi, y)

    # L2 baseline
    import numpy as np
    w_l2, *_ = np.linalg.lstsq(Phi, y, rcond=None)

    # Metrics
    print("L1 IRLS:", l1_error(Phi, w_irls, y))
    print("L1 SG  :", l1_error(Phi, w_sg, y))
    print("L1 L2  :", l1_error(Phi, w_l2, y))


if __name__ == "__main__":
    main()
