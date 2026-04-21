"""
core/solvers.py
---------------
Implements the two optimization algorithms required by the project:

    Algorithm A1 — Iteratively Reweighted Least Squares (IRLS)
    Algorithm A2 — Smoothed Gradient Descent

Both solve the same problem:
    min_w  f(w) = ||Xw - y||_1

They approach it differently:
    A1 replaces the L1 problem with a sequence of weighted L2 subproblems.
    A2 smooths the sharp corner of |r| with sqrt(r^2 + mu) and runs gradient descent.

Helper:
    gaussian_elimination(A, b) — solves A @ w = b from scratch (no library solver).
    l1_loss(X, w, y)           — evaluates the L1 objective.

Note on library usage:
    numpy.linalg.norm is used only for convergence checks and step size computation.
    This is explicitly permitted by the project guidelines as an "individual step".
    The core algorithmic logic is fully implemented from scratch.
"""

import numpy as np
import time


# =============================================================================
# HELPER: Gaussian Elimination (from scratch — no library solver)
# =============================================================================

def gaussian_elimination(A, b):
    """
    Solve A @ w = b using Gaussian elimination with partial pivoting.

    Implemented entirely from scratch. No numpy or scipy solver is used.
    Called inside IRLS at every iteration to solve the weighted normal equations:
        (X_w.T @ X_w) @ w = X_w.T @ y_w

    Parameters
    ----------
    A : ndarray, shape (m, m) — square coefficient matrix
    b : ndarray, shape (m,)   — right-hand side vector

    Returns
    -------
    w : ndarray, shape (m,) — solution to A @ w = b
    """
    m = len(b)

    # build augmented matrix [A | b] — we operate on both simultaneously
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    # ── forward elimination with partial pivoting ─────────────────────────
    for col in range(m):

        # find the row with the largest absolute value in this column
        # (from the current row downward) — this is partial pivoting.
        # Pivoting keeps numerical errors small when the diagonal is near zero.
        pivot_row = col + np.argmax(np.abs(Ab[col:, col]))

        # swap the current row with the pivot row
        Ab[[col, pivot_row]] = Ab[[pivot_row, col]]

        # if the pivot is effectively zero, the matrix is singular
        if abs(Ab[col, col]) < 1e-14:
            return np.zeros(m)   # return zeros safely

        # eliminate all entries below the diagonal in this column
        for row in range(col + 1, m):
            # how much of the pivot row do we subtract?
            factor = Ab[row, col] / Ab[col, col]
            # subtract: this zeros out Ab[row, col] and adjusts all other cols
            Ab[row, col:] -= factor * Ab[col, col:]

    # ── back substitution: solve upper triangular system ──────────────────
    w = np.zeros(m)
    for row in range(m - 1, -1, -1):
        # start from the right-hand side for this row
        w[row] = Ab[row, m]
        # subtract contributions from already-solved variables
        for j in range(row + 1, m):
            w[row] -= Ab[row, j] * w[j]
        # divide by diagonal to isolate w[row]
        w[row] /= Ab[row, row]

    return w


# =============================================================================
# HELPER: L1 loss
# =============================================================================

def l1_loss(X, w, y):
    """
    Compute the L1 objective: f(w) = ||Xw - y||_1 = sum_i |r_i|.

    Both A1 and A2 minimize this same function.
    Used to track and compare convergence of both algorithms.

    Parameters
    ----------
    X : ndarray, shape (n, m)
    w : ndarray, shape (m,)
    y : ndarray, shape (n,)

    Returns
    -------
    loss : float
    """
    return float(np.sum(np.abs(X @ w - y)))


# =============================================================================
# ALGORITHM A1 — Iteratively Reweighted Least Squares (IRLS)
# =============================================================================

def irls(X, y, eps=1e-6, tol=1e-6, max_iter=500):
    """
    Algorithm A1: solve min_w ||Xw - y||_1 via IRLS.

    Core idea — the algebraic trick:
        |r_i| = d_i^2 * r_i^2   where d_i = 1 / sqrt(|r_i|)

    This rewrites the L1 loss as a weighted sum of squared residuals.
    When d_i is frozen at its current value, the subproblem is smooth
    weighted least squares, solvable in closed form via normal equations:
        (X_w.T @ X_w) w = X_w.T @ y_w
    where X_w = diag(d) @ X and y_w = diag(d) @ y.

    The normal equations are solved with our own gaussian_elimination().
    The L1 objective decreases monotonically at every iteration (MM guarantee).

    Parameters
    ----------
    X        : ndarray, shape (n, m) — ELM feature matrix
    y        : ndarray, shape (n,)   — target vector (ONE target at a time)
    eps      : float — threshold for weights: d_i = 1/sqrt(max(|r_i|, eps))
                       prevents explosion when r_i ≈ 0
    tol      : float — stop when ||w_new - w||_2 < tol
    max_iter : int   — maximum iterations

    Returns
    -------
    w       : ndarray, shape (m,) — solution weight vector
    history : dict
        'loss'      — list of L1 loss values (one per iteration)
        'w_change'  — list of ||w_new - w|| (one per iteration)
        'n_iter'    — total iterations run
        'converged' — True if tol was reached
        'time_sec'  — wall-clock time in seconds
    """
    n, m = X.shape
    w = np.zeros(m)   # start from zero

    history = {'loss': [], 'w_change': [], 'n_iter': 0,
               'converged': False, 'time_sec': 0.0}
    t_start = time.perf_counter()

    for k in range(max_iter):

        # Step 1 — residuals: r_i = (Xw)_i - y_i
        r = X @ w - y                                       # shape (n,)

        # record L1 loss BEFORE updating w
        history['loss'].append(float(np.sum(np.abs(r))))

        # Step 2 — IRLS weights
        # big |r_i|  → small d_i → this sample has low influence this round
        # small |r_i| → big d_i  → this sample has high influence this round
        # eps guards against division by zero when r_i ≈ 0
        d = 1.0 / np.sqrt(np.maximum(np.abs(r), eps))      # shape (n,)

        # Step 3 — scale X and y by weights
        # Avoids forming the n×n diagonal matrix D_k explicitly.
        # X_w[i, :] = d_i * X[i, :]   and   y_w[i] = d_i * y[i]
        X_w = d[:, None] * X                                # shape (n, m)
        y_w = d * y                                         # shape (n,)

        # Step 4 — form normal equations
        # Minimizing ||X_w w - y_w||_2^2 → gradient = 0 gives:
        #     (X_w.T @ X_w) w = X_w.T @ y_w
        A = X_w.T @ X_w                                     # shape (m, m)
        A += 1e-8 * np.eye(m)     # ridge regularization for numerical stability
        b = X_w.T @ y_w                                     # shape (m,)

        # Step 5 — solve with our own Gaussian elimination
        w_new = gaussian_elimination(A, b)

        # Step 6 — convergence check (dual criterion)
        # Converge if EITHER weight change OR relative loss change is small.
        # numpy.linalg.norm is permitted as an individual sub-step
        delta = float(np.linalg.norm(w_new - w))
        history['w_change'].append(delta)
        w = w_new

        # check weight-change criterion
        w_converged = delta < tol
        # check relative loss change criterion
        loss_converged = False
        if len(history['loss']) >= 2:
            prev_loss = history['loss'][-2]
            cur_loss = history['loss'][-1]
            if prev_loss > 0:
                rel_change = abs(prev_loss - cur_loss) / prev_loss
                loss_converged = rel_change < tol

        if w_converged or loss_converged:
            history['converged'] = True
            history['n_iter'] = k + 1
            break
    else:
        history['n_iter'] = max_iter

    # record final L1 loss after last update
    history['loss'].append(float(np.sum(np.abs(X @ w - y))))
    history['time_sec'] = time.perf_counter() - t_start
    return w, history


# =============================================================================
# ALGORITHM A2 — Smoothed Gradient Descent
# =============================================================================

def smoothed_gradient(X, y, mu=1e-3, lr=None, tol=1e-6, max_iter=500):
    """
    Algorithm A2: solve min_w ||Xw - y||_1 via smoothed gradient descent
    with backtracking line search (Armijo rule).

    Core idea — smooth the non-differentiable |r_i| at r_i = 0:
        Replace |r_i| with phi_mu(r_i) = sqrt(r_i^2 + mu)

        As mu → 0: phi_mu(r_i) → |r_i|  (recovers true L1)
        At r_i = 0: phi_mu(0) = sqrt(mu) (no sharp corner — smooth!)

    The smoothed objective is:
        f_mu(w) = sum_i sqrt((Xw - y)_i^2 + mu)

    Its gradient with respect to w is (by chain rule):
        grad_w f_mu = X.T @ (r / sqrt(r^2 + mu))
        where r = Xw - y

    Step size is chosen by backtracking line search (Armijo condition):
        Start with lr_init and shrink by factor beta until sufficient decrease.
        This is much more adaptive than the conservative Lipschitz step size.

    Parameters
    ----------
    X        : ndarray, shape (n, m) — ELM feature matrix
    y        : ndarray, shape (n,)   — target vector (ONE target at a time)
    mu       : float — smoothing parameter
                       smaller mu = closer to true L1 but slower convergence
                       recommended range: 1e-2 to 1e-5
    lr       : float or None — initial step size for line search (default 1.0)
    tol      : float — stop when ||w_new - w||_2 < tol
    max_iter : int   — maximum iterations

    Returns
    -------
    w       : ndarray, shape (m,) — solution weight vector
    history : dict
        'loss'      — list of TRUE L1 loss (not smoothed) for fair comparison
        'w_change'  — list of ||w_new - w||
        'n_iter'    — total iterations run
        'converged' — True if tol was reached
        'time_sec'  — wall-clock time in seconds
    """
    n, m = X.shape
    w = np.zeros(m)   # start from zero

    # backtracking line search parameters
    lr_init = 1.0 if lr is None else lr   # starting step size
    beta = 0.5        # shrink factor
    c = 1e-4          # Armijo sufficient decrease parameter
    max_ls = 30       # maximum line search steps

    def _smoothed_loss(w_):
        """Evaluate the smoothed objective f_mu(w) = sum sqrt(r^2 + mu)."""
        r_ = X @ w_ - y
        return float(np.sum(np.sqrt(r_ ** 2 + mu)))

    history = {'loss': [], 'w_change': [], 'n_iter': 0,
               'converged': False, 'time_sec': 0.0}
    t_start = time.perf_counter()

    for k in range(max_iter):

        # Step 1 — residuals
        r = X @ w - y                                       # shape (n,)

        # record the TRUE L1 loss (not the smoothed version)
        # so we can compare A1 and A2 on the same objective fairly
        history['loss'].append(float(np.sum(np.abs(r))))

        # Step 2 — gradient of the smoothed objective
        # d/dr sqrt(r^2 + mu) = r / sqrt(r^2 + mu)
        # When r = 0: gradient = 0 (smooth, no undefined point)
        # When r >> sqrt(mu): gradient ≈ sign(r) (same as true L1)
        smooth_deriv = r / np.sqrt(r ** 2 + mu)            # shape (n,)

        # chain rule: d f_mu / d w = X.T @ (d f_mu / d r)
        grad = X.T @ smooth_deriv                          # shape (m,)

        # Step 3 — backtracking line search (Armijo rule)
        # Find step size that guarantees sufficient decrease in f_mu
        f_cur = _smoothed_loss(w)
        grad_norm_sq = float(np.dot(grad, grad))
        lr_k = lr_init
        for _ in range(max_ls):
            w_trial = w - lr_k * grad
            if _smoothed_loss(w_trial) <= f_cur - c * lr_k * grad_norm_sq:
                break
            lr_k *= beta
        else:
            # if line search exhausted, use the smallest step
            w_trial = w - lr_k * grad

        w_new = w_trial

        # Step 4 — convergence check (dual criterion)
        delta = float(np.linalg.norm(w_new - w))
        history['w_change'].append(delta)
        w = w_new

        # check weight-change criterion
        w_converged = delta < tol
        # check relative loss change criterion
        loss_converged = False
        if len(history['loss']) >= 2:
            prev_loss = history['loss'][-2]
            cur_loss = history['loss'][-1]
            if prev_loss > 0:
                rel_change = abs(prev_loss - cur_loss) / prev_loss
                loss_converged = rel_change < tol

        if w_converged or loss_converged:
            history['converged'] = True
            history['n_iter'] = k + 1
            break
    else:
        history['n_iter'] = max_iter

    # record final L1 loss
    history['loss'].append(float(np.sum(np.abs(X @ w - y))))
    history['time_sec'] = time.perf_counter() - t_start
    return w, history

