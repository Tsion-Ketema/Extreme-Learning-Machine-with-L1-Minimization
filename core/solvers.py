"""
core/solvers.py
---------------
A1: IRLS
A2: Smoothed gradient method
A2-FGM: Nesterov fast gradient method for the smoothed L1 objective.
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
    """Original nonsmooth L1 objective: sum_i |x_i^T w - y_i|."""
    return float(np.sum(np.abs(X @ w - y)))


def smooth_l1_loss(X, w, y, mu):
    """Smoothed L1 objective used by A2: sum_i sqrt(r_i^2 + mu)."""
    r = X @ w - y
    return float(np.sum(np.sqrt(r ** 2 + mu)))


def compute_nesterov_mu(n, target_eps):
    """
    Theory-guided smoothing parameter for sqrt smoothing.

    A2 replaces |r_i| by sqrt(r_i^2 + mu). For every residual,
        0 <= sqrt(r_i^2 + mu) - |r_i| <= sqrt(mu).
    Summing over n samples gives total smoothing bias <= n * sqrt(mu).

    To make the smoothing bias at most target_eps / 2, choose
        n * sqrt(mu) <= target_eps / 2
    hence
        mu = (target_eps / (2*n))^2.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if target_eps <= 0:
        raise ValueError("target_eps must be positive")
    return (target_eps / (2.0 * n)) ** 2


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

        if tol is not None and delta < tol:
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
    """
    Non-accelerated smoothed gradient method.

    This is kept for practical comparison and parameter-sensitivity plots.
    For sqrt smoothing, grad f_mu has Lipschitz constant bounded by
        L <= ||X||_F^2 / sqrt(mu),
    so the default safe step size is
        lr = sqrt(mu) / ||X||_F^2.
    """
    n, m = X.shape
    w = np.zeros(m)

    if lr is None:
        frob_sq = float(np.linalg.norm(X, 'fro') ** 2)
        frob_sq = max(frob_sq, 1e-14)
        lr = np.sqrt(mu) / frob_sq

    history = {
        'loss': [],
        'smooth_loss': [],
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
        sm_loss = float(np.sum(np.sqrt(r ** 2 + mu)))
        history['loss'].append(loss)
        history['smooth_loss'].append(sm_loss)

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
                  f"smooth_loss={sm_loss:.6f} "
                  f"delta={delta:.2e} "
                  f"grad_norm={grad_norm:.2e}")

        if tol is not None and delta < tol:
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

    final_loss = l1_loss(X, w, y)
    final_smooth_loss = smooth_l1_loss(X, w, y, mu)
    history['loss'].append(final_loss)
    history['smooth_loss'].append(final_smooth_loss)
    history['time_sec'] = time.perf_counter() - t_start

    if debug:
        print(f"      [{name}] finished")
        print(f"        final loss  = {final_loss:.6f}")
        print(f"        iterations  = {history['n_iter']}")
        print(f"        converged   = {history['converged']}")
        print(f"        time        = {history['time_sec']:.3f}s")

    return w, history


def fast_smoothed_gradient(X, y, mu, tol=None,
                           max_iter=500, debug=False, name="A2-FGM"):
    """
    A2-FGM: Nesterov fast gradient method for the smoothed L1 objective.

    Objective:
        f_mu(w) = sum_i sqrt((x_i^T w - y_i)^2 + mu)

    Pseudo-code:
        choose w_0 = 0, w_{-1} = 0, t_0 = 1
        L = ||X||_2^2 / sqrt(mu)
        alpha = 1/L
        for k = 0, 1, ..., K-1:
            t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
            beta_k  = (t_k - 1) / t_{k+1}
            z_k     = w_k + beta_k (w_k - w_{k-1})
            r_k     = X z_k - y
            q_k     = r_k / sqrt(r_k^2 + mu)
            grad    = X^T q_k
            w_{k+1} = z_k - alpha * grad

    Notes:
        * For this sqrt smoothing, the scalar second derivative is bounded
          by 1/sqrt(mu), hence L <= ||X||_2^2 / sqrt(mu).
        * The theory-guided choice for target tolerance eps is
              mu = (eps / (2*n))^2,
          because the total smoothing bias is at most n*sqrt(mu).
        * tol defaults to None because tiny theory-guided mu can make the
          step size very small; stopping only from ||w_new-w|| may stop too
          early even when the objective gap is still large.
    """
    n, m = X.shape
    op_norm_sq = float(np.linalg.norm(X, 2) ** 2)
    op_norm_sq = max(op_norm_sq, 1e-14)
    L = op_norm_sq / np.sqrt(mu)
    alpha = 1.0 / L

    w = np.zeros(m)
    w_prev = np.zeros(m)
    t = 1.0

    history = {
        'loss': [],
        'smooth_loss': [],
        'gap': [],
        'smooth_gap': [],
        'w_change': [],
        'n_iter': 0,
        'converged': False,
        'time_sec': 0.0,
        'L': L,
        'alpha': alpha,
    }

    if debug:
        print(f"      [{name}] start (fast gradient / Nesterov AGM)")
        print(f"        X shape     = {X.shape}")
        print(f"        mu          = {mu:.4e}")
        print(f"        L           = {L:.4e}")
        print(f"        alpha=1/L   = {alpha:.4e}")
        print(f"        tol         = {tol}")
        print(f"        max_iter    = {max_iter}")

    t_start = time.perf_counter()

    for k in range(max_iter):
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        beta = (t - 1.0) / t_next
        z_k = w + beta * (w - w_prev)

        r = X @ z_k - y
        loss = float(np.sum(np.abs(r)))
        sm_loss = float(np.sum(np.sqrt(r ** 2 + mu)))
        history['loss'].append(loss)
        history['smooth_loss'].append(sm_loss)

        q = r / np.sqrt(r ** 2 + mu)
        grad = X.T @ q

        w_new = z_k - alpha * grad

        delta = float(np.linalg.norm(w_new - w))
        history['w_change'].append(delta)

        w_prev = w
        w = w_new
        t = t_next

        if debug and (k == 0 or (k + 1) % max(1, max_iter // 10) == 0):
            print(f"        iter={k + 1:6d}  "
                  f"loss={loss:.6f}  "
                  f"smooth_loss={sm_loss:.6f}  "
                  f"delta={delta:.2e}")

        if tol is not None and delta < tol:
            history['converged'] = True
            history['n_iter'] = k + 1
            if debug:
                print(f"      [{name}] converged at iter {k + 1}")
            break
    else:
        history['n_iter'] = max_iter
        if debug:
            print(f"      [{name}] reached max_iter={max_iter}")

    final_loss = l1_loss(X, w, y)
    final_smooth_loss = smooth_l1_loss(X, w, y, mu)
    history['loss'].append(final_loss)
    history['smooth_loss'].append(final_smooth_loss)
    history['time_sec'] = time.perf_counter() - t_start

    if debug:
        print(f"      [{name}] finished")
        print(f"        final loss        = {final_loss:.6f}")
        print(f"        final smooth loss = {final_smooth_loss:.6f}")
        print(f"        iterations        = {history['n_iter']}")
        print(f"        converged         = {history['converged']}")
        print(f"        time              = {history['time_sec']:.3f}s")

    return w, history


def compute_gap(history, f_star):
    f_star_abs = max(abs(f_star), 1e-14)

    history['gap'] = [
        (loss - f_star) / f_star_abs
        for loss in history['loss']
    ]

    return history


def compute_smooth_gap(history, f_mu_ref=None):
    """
    Add a relative smooth-objective gap to history.

    If f_mu_ref is provided, the gap is measured against that reference value.
    For the theory experiment we pass f_mu(w_A1*) as an approximate smoothed optimum,
    instead of using the best observed A2 value, because that would make the final
    observed gap artificially become zero.
    """
    smooth_losses = np.array(history.get('smooth_loss', []), dtype=float)
    if smooth_losses.size == 0:
        history['smooth_gap'] = []
        return history

    if f_mu_ref is None:
        f_mu_ref = float(np.min(smooth_losses))

    denom = max(abs(f_mu_ref), 1e-14)
    history['smooth_gap'] = [
        max((float(v) - f_mu_ref) / denom, 0.0)
        for v in smooth_losses
    ]
    history['f_mu_ref'] = f_mu_ref
    return history