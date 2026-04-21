"""
utils/experiments.py
--------------------
Experiment runners for the report.

The project requires comparing two algorithms that both solve min_w ||Xw-y||_1:
    A1 — IRLS (iteratively reweighted least squares)
    A2 — Smoothed gradient descent

Experiments:
    1. experiment_convergence  — L1 loss vs iteration for A1 and A2
    2. experiment_epsilon      — effect of eps (A1) and mu (A2) on final loss
    3. experiment_scaling_m    — CPU time vs number of hidden neurons m
    4. experiment_a1_vs_a2     — head-to-head comparison on all 4 targets

No plotting happens here. Each function returns a plain dict for viz/plots.py.
"""

import numpy as np
import time

from core.elm import build_feature_matrix, random_hidden_weights
from core.solvers import irls, smoothed_gradient, l1_loss

TARGETS = ['TARGET_1', 'TARGET_2', 'TARGET_3', 'TARGET_4']


# =============================================================================
# 1. Convergence: L1 loss vs iteration for A1 and A2
# =============================================================================

def experiment_convergence(Input_tr, Y_tr, m=100, eps=1e-6, mu=1e-3,
                           tol=1e-6, max_iter=500, seed=0):
    """
    Run both A1 (IRLS) and A2 (smoothed gradient) for each of the 4 targets.
    Record the true L1 loss at every iteration.

    Used to verify:
        - Monotone decrease of L1 loss (A1 guaranteed, A2 expected)
        - Convergence rate (should be linear for A1 in semi-log plot)
        - Speed comparison: which algorithm converges faster?

    Returns
    -------
    dict with keys:
        'a1_losses'    — list of 4 arrays, L1 loss per iter for A1
        'a2_losses'    — list of 4 arrays, L1 loss per iter for A2
        'a1_n_iters'   — list of 4 ints
        'a2_n_iters'   — list of 4 ints
        'a1_converged' — list of 4 bools
        'a2_converged' — list of 4 bools
        'target_names' — list of 4 strings
        'm'            — int
    """
    # build X once — same for both algorithms and all 4 targets
    W1 = random_hidden_weights(m, Input_tr.shape[1], seed=seed)
    X = build_feature_matrix(Input_tr, W1)

    a1_losses, a1_niters, a1_conv = [], [], []
    a2_losses, a2_niters, a2_conv = [], [], []

    for t in range(4):
        y = Y_tr[:, t]

        # run A1 — IRLS
        _, h1 = irls(X, y, eps=eps, tol=tol, max_iter=max_iter)
        a1_losses.append(np.array(h1['loss']))
        a1_niters.append(h1['n_iter'])
        a1_conv.append(h1['converged'])

        # run A2 — smoothed gradient
        _, h2 = smoothed_gradient(X, y, mu=mu, tol=tol, max_iter=max_iter)
        a2_losses.append(np.array(h2['loss']))
        a2_niters.append(h2['n_iter'])
        a2_conv.append(h2['converged'])

    return {
        'a1_losses': a1_losses,
        'a2_losses': a2_losses,
        'a1_n_iters': a1_niters,
        'a2_n_iters': a2_niters,
        'a1_converged': a1_conv,
        'a2_converged': a2_conv,
        'target_names': TARGETS,
        'm': m,
    }


# =============================================================================
# 2. Parameter sensitivity: eps (A1) and mu (A2)
# =============================================================================

def experiment_params(Input_tr, Y_tr, m=100, seed=0,
                      eps_values=None, mu_values=None,
                      tol=1e-6, max_iter=500):
    """
    Vary the key parameter of each algorithm and record final L1 loss and
    iteration count. Helps choose good values of eps and mu.

    eps controls the IRLS threshold (A1).
    mu controls the smoothing level (A2).

    Returns
    -------
    dict with keys for A1 and A2 separately.
    """
    if eps_values is None:
        eps_values = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    if mu_values is None:
        mu_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    W1 = random_hidden_weights(m, Input_tr.shape[1], seed=seed)
    X = build_feature_matrix(Input_tr, W1)

    # A1: vary eps
    a1_losses = [[] for _ in range(4)]
    a1_niters = [[] for _ in range(4)]
    for eps in eps_values:
        for t in range(4):
            w, h = irls(X, Y_tr[:, t], eps=eps, tol=tol, max_iter=max_iter)
            a1_losses[t].append(h['loss'][-1])
            a1_niters[t].append(h['n_iter'])

    # A2: vary mu
    a2_losses = [[] for _ in range(4)]
    a2_niters = [[] for _ in range(4)]
    for mu in mu_values:
        for t in range(4):
            w, h = smoothed_gradient(X, Y_tr[:, t], mu=mu, tol=tol,
                                     max_iter=max_iter)
            a2_losses[t].append(h['loss'][-1])
            a2_niters[t].append(h['n_iter'])

    return {
        'eps_values': eps_values,
        'mu_values': mu_values,
        'a1_losses': [np.array(fl) for fl in a1_losses],
        'a1_niters': [np.array(ni) for ni in a1_niters],
        'a2_losses': [np.array(fl) for fl in a2_losses],
        'a2_niters': [np.array(ni) for ni in a2_niters],
        'target_names': TARGETS,
    }


# =============================================================================
# 3. Scaling: CPU time vs m for both algorithms
# =============================================================================

def experiment_scaling_m(Input_tr, Y_tr, m_values=None, seed=0,
                         eps=1e-6, mu=1e-3, tol=1e-6,
                         max_iter=500, n_repeat=3):
    """
    Measure wall-clock time for both A1 and A2 as the number of hidden
    neurons m increases. Repeated n_repeat times per configuration.

    Used to assess:
        - How does each algorithm scale with m?
        - Which is faster per iteration?
        - A1 costs O(nm^2 + m^3) per iteration. A2 costs O(nm) per iteration.

    Returns
    -------
    dict with timing and loss results for A1 and A2.
    """
    if m_values is None:
        m_values = [10, 25, 50, 100, 200]

    p = Input_tr.shape[1]

    a1_times, a1_times_std = [], []
    a2_times, a2_times_std = [], []
    a1_losses, a2_losses = [], []
    a1_niters, a2_niters = [], []

    for m in m_values:
        W1 = random_hidden_weights(m, p, seed=seed)
        X = build_feature_matrix(Input_tr, W1)

        # run A1 n_repeat times and average
        r1_times, r1_losses, r1_niters = [], [], []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            tgt_l, tgt_n = [], []
            for t in range(4):
                w, h = irls(X, Y_tr[:, t], eps=eps, tol=tol,
                            max_iter=max_iter)
                tgt_l.append(h['loss'][-1])
                tgt_n.append(h['n_iter'])
            r1_times.append(time.perf_counter() - t0)
            r1_losses.append(np.mean(tgt_l))
            r1_niters.append(np.mean(tgt_n))

        # run A2 n_repeat times and average
        r2_times, r2_losses, r2_niters = [], [], []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            tgt_l, tgt_n = [], []
            for t in range(4):
                w, h = smoothed_gradient(X, Y_tr[:, t], mu=mu, tol=tol,
                                         max_iter=max_iter)
                tgt_l.append(h['loss'][-1])
                tgt_n.append(h['n_iter'])
            r2_times.append(time.perf_counter() - t0)
            r2_losses.append(np.mean(tgt_l))
            r2_niters.append(np.mean(tgt_n))

        a1_times.append(np.mean(r1_times))
        a1_times_std.append(np.std(r1_times))
        a1_losses.append(np.mean(r1_losses))
        a1_niters.append(np.mean(r1_niters))

        a2_times.append(np.mean(r2_times))
        a2_times_std.append(np.std(r2_times))
        a2_losses.append(np.mean(r2_losses))
        a2_niters.append(np.mean(r2_niters))

    return {
        'm_values': m_values,
        'a1_times': np.array(a1_times),
        'a1_times_std': np.array(a1_times_std),
        'a2_times': np.array(a2_times),
        'a2_times_std': np.array(a2_times_std),
        'a1_losses': np.array(a1_losses),
        'a2_losses': np.array(a2_losses),
        'a1_niters': np.array(a1_niters),
        'a2_niters': np.array(a2_niters),
    }


# =============================================================================
# 4. Head-to-head: A1 vs A2 final results on all 4 targets
# =============================================================================

def experiment_a1_vs_a2(Input_tr, Y_tr, Input_te, Y_te,
                        m=100, seed=0, eps=1e-6, mu=1e-3,
                        tol=1e-6, max_iter=500):
    """
    Run both A1 and A2 on each of the 4 targets. Report:
        - Training L1 loss
        - Test L1 loss
        - Number of iterations
        - Wall-clock time
        - Whether convergence was reached

    This is the main comparison table for the report.
    Both algorithms solve the same problem — the comparison is purely
    about algorithmic efficiency and solution quality.

    Returns
    -------
    dict with key 'rows': list of 4 dicts (one per target).
    """
    W1 = random_hidden_weights(m, Input_tr.shape[1], seed=seed)
    X_tr = build_feature_matrix(Input_tr, W1)
    X_te = build_feature_matrix(Input_te, W1)

    rows = []
    for t in range(4):
        y_tr = Y_tr[:, t]
        y_te = Y_te[:, t]

        # A1 — IRLS
        w1, h1 = irls(X_tr, y_tr, eps=eps, tol=tol, max_iter=max_iter)
        a1_train = l1_loss(X_tr, w1, y_tr)
        a1_test = l1_loss(X_te, w1, y_te)

        # A2 — smoothed gradient
        w2, h2 = smoothed_gradient(X_tr, y_tr, mu=mu, tol=tol,
                                   max_iter=max_iter)
        a2_train = l1_loss(X_tr, w2, y_tr)
        a2_test = l1_loss(X_te, w2, y_te)

        rows.append({
            'target': TARGETS[t],
            'a1_train_l1': a1_train,
            'a1_test_l1': a1_test,
            'a1_n_iter': h1['n_iter'],
            'a1_converged': h1['converged'],
            'a1_time': h1['time_sec'],
            'a2_train_l1': a2_train,
            'a2_test_l1': a2_test,
            'a2_n_iter': h2['n_iter'],
            'a2_converged': h2['converged'],
            'a2_time': h2['time_sec'],
        })

    return {'rows': rows, 'm': m}
