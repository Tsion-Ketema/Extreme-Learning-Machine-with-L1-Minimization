"""
core/solvers.py
---------------
A1: IRLS
A2: Smoothed gradient method
With optional debug tracing.
"""

import numpy as np
import time


def gaussian_elimination(A, b):
    m = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for col in range(m):
        pivot_row = col + np.argmax(np.abs(Ab[col:, col]))
        Ab[[col, pivot_row]] = Ab[[pivot_row, col]]

        if abs(Ab[col, col]) < 1e-14:
            return np.zeros(m)

        for row in range(col + 1, m):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]

    w = np.zeros(m)

    for row in range(m - 1, -1, -1):
        w[row] = Ab[row, m]

        for j in range(row + 1, m):
            w[row] -= Ab[row, j] * w[j]

        w[row] /= Ab[row, row]

    return w


def l1_loss(X, w, y):
    return float(np.sum(np.abs(X @ w - y)))


def irls(X, y, eps=1e-6, tol=1e-6, max_iter=500,
         debug=False, name="A1"):

    n, m = X.shape
    w = np.zeros(m)

    history = {
        'loss': [],
        'gap': [],
        'w_change': [],
        'n_iter': 0,
        'converged': False,
        'time_sec': 0.0,
    }

    if debug:
        print(f"      [{name}] start")
        print(f"        X shape     = {X.shape}")
        print(f"        y shape     = {y.shape}")
        print(f"        eps         = {eps}")
        print(f"        tol         = {tol}")
        print(f"        max_iter    = {max_iter}")

    t_start = time.perf_counter()

    for k in range(max_iter):
        r = X @ w - y
        loss = float(np.sum(np.abs(r)))
        history['loss'].append(loss)

        d = 1.0 / np.sqrt(np.maximum(np.abs(r), eps))

        X_w = d[:, None] * X
        y_w = d * y

        A = X_w.T @ X_w
        b = X_w.T @ y_w

        w_new = gaussian_elimination(A, b)

        delta = float(np.linalg.norm(w_new - w))
        history['w_change'].append(delta)

        w = w_new

        if debug and (k == 0 or (k + 1) % 50 == 0):
            print(f"        iter={k + 1:5d} "
                  f"loss={loss:.6f} "
                  f"delta={delta:.2e} "
                  f"min|r|={np.min(np.abs(r)):.2e} "
                  f"max|r|={np.max(np.abs(r)):.2e}")

        if delta < tol:
            history['converged'] = True
            history['n_iter'] = k + 1

            if debug:
                print(f"      [{name}] converged at iter {k + 1}")
                print(f"        delta={delta:.2e} < tol={tol}")

            break
    else:
        history['n_iter'] = max_iter

        if debug:
            print(f"      [{name}] reached max_iter={max_iter}")

    final_loss = float(np.sum(np.abs(X @ w - y)))
    history['loss'].append(final_loss)
    history['time_sec'] = time.perf_counter() - t_start

    if debug:
        print(f"      [{name}] finished")
        print(f"        final loss  = {final_loss:.6f}")
        print(f"        iterations  = {history['n_iter']}")
        print(f"        converged   = {history['converged']}")
        print(f"        time        = {history['time_sec']:.3f}s")

    return w, history


def smoothed_gradient(X, y, mu=1e-3, lr=None, tol=1e-6,
                      max_iter=500, debug=False, name="A2"):

    n, m = X.shape
    w = np.zeros(m)

    if lr is None:
        frob_sq = float(np.linalg.norm(X, 'fro') ** 2)
        frob_sq = max(frob_sq, 1e-14)
        lr = mu / frob_sq

    history = {
        'loss': [],
        'gap': [],
        'w_change': [],
        'n_iter': 0,
        'converged': False,
        'time_sec': 0.0,
    }

    if debug:
        print(f"      [{name}] start")
        print(f"        X shape     = {X.shape}")
        print(f"        y shape     = {y.shape}")
        print(f"        mu          = {mu}")
        print(f"        lr          = {lr:.2e}")
        print(f"        tol         = {tol}")
        print(f"        max_iter    = {max_iter}")

    t_start = time.perf_counter()

    for k in range(max_iter):
        r = X @ w - y
        loss = float(np.sum(np.abs(r)))
        history['loss'].append(loss)

        smooth_deriv = r / np.sqrt(r ** 2 + mu)
        grad = X.T @ smooth_deriv

        w_new = w - lr * grad

        delta = float(np.linalg.norm(w_new - w))
        history['w_change'].append(delta)

        w = w_new

        if debug and (k == 0 or (k + 1) % 50 == 0):
            grad_norm = float(np.linalg.norm(grad))
            print(f"        iter={k + 1:5d} "
                  f"loss={loss:.6f} "
                  f"delta={delta:.2e} "
                  f"grad_norm={grad_norm:.2e} "
                  f"min|r|={np.min(np.abs(r)):.2e} "
                  f"max|r|={np.max(np.abs(r)):.2e}")

        if delta < tol:
            history['converged'] = True
            history['n_iter'] = k + 1

            if debug:
                print(f"      [{name}] converged at iter {k + 1}")
                print(f"        delta={delta:.2e} < tol={tol}")

            break
    else:
        history['n_iter'] = max_iter

        if debug:
            print(f"      [{name}] reached max_iter={max_iter}")

    final_loss = float(np.sum(np.abs(X @ w - y)))
    history['loss'].append(final_loss)
    history['time_sec'] = time.perf_counter() - t_start

    if debug:
        print(f"      [{name}] finished")
        print(f"        final loss  = {final_loss:.6f}")
        print(f"        iterations  = {history['n_iter']}")
        print(f"        converged   = {history['converged']}")
        print(f"        time        = {history['time_sec']:.3f}s")

    return w, history


def compute_gap(history, f_star):
    f_star_abs = max(abs(f_star), 1e-14)

    history['gap'] = [
        (loss - f_star) / f_star_abs
        for loss in history['loss']
    ]

    return history
