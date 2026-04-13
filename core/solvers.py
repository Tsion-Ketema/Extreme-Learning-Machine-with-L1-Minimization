import numpy as np


def weighted_least_squares(A, b, W_diag):
    WA = W_diag[:, None] * A
    Wb = W_diag * b
    w, _, _, _ = np.linalg.lstsq(WA, Wb, rcond=None)
    return w


def irls_l1(A, b, max_iter=200, tol=1e-8, thresh=1e-6):
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    history = []

    for _ in range(max_iter):
        r = A @ w - b
        history.append(np.sum(np.abs(r)))

        W_diag = np.maximum(np.abs(r), thresh) ** (-0.5)
        w_new = weighted_least_squares(A, b, W_diag)

        if np.linalg.norm(w_new - w) / (1.0 + np.linalg.norm(w)) < tol:
            w = w_new
            history.append(np.sum(np.abs(A @ w - b)))
            break

        w = w_new

    return w, np.array(history)


def smoothed_l1_grad(A, b, w, mu):
    r = A @ w - b
    return A.T @ (r / np.sqrt(r**2 + mu**2))


def smoothed_l1_obj(A, b, w, mu):
    r = A @ w - b
    return np.sum(np.sqrt(r**2 + mu**2) - mu)


def smoothed_gradient_l1(
    A,
    b,
    mu_init=0.5,
    mu_decay=0.5,
    mu_min=1e-6,
    stage_max_iter=1000,
    tol=1e-8,
):
    """
    Continuation scheme for L1 minimization using a smoothed objective.

    Each stage solves a pseudo-Huber approximation with a smaller smoothing
    parameter mu and warm-starts from the previous stage. The step size comes
    from the smooth objective's Lipschitz constant, which keeps the updates
    stable without using an external optimizer.
    """
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    history = []

    gram = A.T @ A
    spectral_radius = np.linalg.eigvalsh(gram).max()
    mu = mu_init

    while True:
        step = mu / max(spectral_radius, 1e-12)
        t = 1.0
        z = w.copy()
        stage_history = []

        for iteration in range(stage_max_iter):
            grad = smoothed_l1_grad(A, b, z, mu)
            w_next = z - step * grad

            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = w_next + ((t - 1.0) / t_next) * (w_next - w)

            w = w_next
            t = t_next

            true_obj = np.sum(np.abs(A @ w - b))
            history.append(true_obj)
            stage_history.append(true_obj)

            if iteration >= 100:
                old_obj = stage_history[-101]
                if abs(old_obj - true_obj) / (1.0 + old_obj) < tol:
                    break

        if mu <= mu_min:
            break

        mu = max(mu * mu_decay, mu_min)

    history.append(np.sum(np.abs(A @ w - b)))
    return w, np.array(history)
