import numpy as np

# ─────────────────────────────
# Weighted Least Squares
# ─────────────────────────────


def weighted_least_squares(A, b, W_diag):
    WA = W_diag[:, None] * A
    Wb = W_diag * b
    w, _, _, _ = np.linalg.lstsq(WA, Wb, rcond=None)
    return w


# ─────────────────────────────
# IRLS (L1)
# ─────────────────────────────
def irls_l1(A, b, max_iter=200, tol=1e-8, thresh=1e-6):
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    history = []

    for _ in range(max_iter):
        r = A @ w - b
        history.append(np.sum(np.abs(r)))

        W_diag = np.maximum(np.abs(r), thresh) ** (-0.5)
        w_new = weighted_least_squares(A, b, W_diag)

        if np.linalg.norm(w_new - w) / (1 + np.linalg.norm(w)) < tol:
            w = w_new
            break

        w = w_new

    return w, np.array(history)


# ─────────────────────────────
# Smoothed Gradient (Pseudo-Huber)
# ─────────────────────────────
def smoothed_l1_grad(A, b, w, mu):
    r = A @ w - b
    return A.T @ (r / np.sqrt(r**2 + mu**2))


def smoothed_l1_obj(A, b, w, mu):
    r = A @ w - b
    return np.sum(np.sqrt(r**2 + mu**2) - mu)


def smoothed_gradient_l1(
    A, b,
    max_iter=500,
    tol=1e-8,
    mu_init=1.0,
    mu_decay=0.85,
    mu_min=1e-8,
    alpha0=1.0,
    beta=0.5,
    c=1e-4
):
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    mu = mu_init
    history = []

    for k in range(max_iter):
        history.append(np.sum(np.abs(A @ w - b)))

        grad = smoothed_l1_grad(A, b, w, mu)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < tol:
            break

        f0 = smoothed_l1_obj(A, b, w, mu)
        alpha = alpha0

        for _ in range(60):
            w_trial = w - alpha * grad
            if smoothed_l1_obj(A, b, w_trial, mu) <= f0 - c * alpha * grad_norm**2:
                break
            alpha *= beta

        w -= alpha * grad

        if (k + 1) % 10 == 0:
            mu = max(mu * mu_decay, mu_min)

    return w, np.array(history)
