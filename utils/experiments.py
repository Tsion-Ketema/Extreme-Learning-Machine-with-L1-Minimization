"""
utils/experiments.py
--------------------
Experiment runners with debug prints.
"""

import numpy as np
import time

from core.elm import build_feature_matrix, random_hidden_weights
from core.solvers import (irls, smoothed_gradient, fast_smoothed_gradient,
                          l1_loss, smooth_l1_loss, compute_gap,
                          compute_smooth_gap, compute_nesterov_mu)

TARGETS = ['TARGET_1', 'TARGET_2', 'TARGET_3', 'TARGET_4']


def experiment_convergence(Input_tr, Y_tr, m=100, eps=1e-6, mu=1e-3,
                           tol=1e-6, max_iter=500, seed=0, debug=True):

    if debug:
        print("\n[experiment_convergence]")
        print(f"  Building ELM feature matrix with m={m}, seed={seed}")

    W1 = random_hidden_weights(m, Input_tr.shape[1], seed=seed)
    X = build_feature_matrix(Input_tr, W1)

    if debug:
        print(f"  Feature matrix X shape: {X.shape}")

    a1_losses, a1_gaps, a1_niters, a1_conv = [], [], [], []
    a2_losses, a2_gaps, a2_niters, a2_conv = [], [], [], []
    f_stars = []

    for t in range(4):
        y = Y_tr[:, t]

        if debug:
            print(f"\n  Target {t + 1}/4: {TARGETS[t]}")
            print("    Step 1: Computing f* using long A1 run...")

        _, h_star = irls(
            X, y,
            eps=eps,
            tol=tol,
            max_iter=5000,
            debug=debug,
            name=f"f* {TARGETS[t]}"
        )

        f_star = h_star['loss'][-1]
        f_stars.append(f_star)

        if debug:
            print(f"    f* done: f* ≈ {f_star:.6f}")

        if debug:
            print("    Step 2: Running A1 — IRLS...")

        _, h1 = irls(
            X, y,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            debug=debug,
            name=f"A1 {TARGETS[t]}"
        )

        h1 = compute_gap(h1, f_star)

        a1_losses.append(np.array(h1['loss']))
        a1_gaps.append(np.array(h1['gap']))
        a1_niters.append(h1['n_iter'])
        a1_conv.append(h1['converged'])

        if debug:
            print(f"    A1 done:")
            print(f"      final loss = {h1['loss'][-1]:.6f}")
            print(f"      final gap  = {h1['gap'][-1]:.2e}")
            print(f"      iterations = {h1['n_iter']}")
            print(f"      converged  = {h1['converged']}")
            print(f"      time       = {h1['time_sec']:.3f}s")

        if debug:
            print("    Step 3: Running A2-FGM — Nesterov smoothed gradient...")

        _, h2 = fast_smoothed_gradient(
            X, y,
            mu=mu,
            tol=tol,
            max_iter=max_iter,
            debug=debug,
            name=f"A2-FGM {TARGETS[t]}"
        )

        h2 = compute_gap(h2, f_star)

        a2_losses.append(np.array(h2['loss']))
        a2_gaps.append(np.array(h2['gap']))
        a2_niters.append(h2['n_iter'])
        a2_conv.append(h2['converged'])

        if debug:
            print(f"    A2-FGM done:")
            print(f"      final loss = {h2['loss'][-1]:.6f}")
            print(f"      final gap  = {h2['gap'][-1]:.2e}")
            print(f"      iterations = {h2['n_iter']}")
            print(f"      converged  = {h2['converged']}")
            print(f"      time       = {h2['time_sec']:.3f}s")

    return {
        'a1_losses': a1_losses,
        'a2_losses': a2_losses,
        'a1_gaps': a1_gaps,
        'a2_gaps': a2_gaps,
        'a1_n_iters': a1_niters,
        'a2_n_iters': a2_niters,
        'a1_converged': a1_conv,
        'a2_converged': a2_conv,
        'f_star': f_stars,
        'f_stars': f_stars,
        'target_names': TARGETS,
        'm': m,
    }


def experiment_params(Input_tr, Y_tr, m=100, seed=0,
                      eps_values=None, mu_values=None,
                      tol=1e-6, max_iter=500, debug=True):

    if eps_values is None:
        eps_values = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    if mu_values is None:
        mu_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    if debug:
        print("\n[experiment_params]")
        print(f"  Building ELM feature matrix with m={m}")

    W1 = random_hidden_weights(m, Input_tr.shape[1], seed=seed)
    X = build_feature_matrix(Input_tr, W1)

    # Compute f* once per target
    f_stars = []
    for t in range(4):
        if debug:
            print(f"  Computing f* for {TARGETS[t]}...")

        _, h_star = irls(
            X, Y_tr[:, t],
            eps=1e-8,
            tol=tol,
            max_iter=5000,
            debug=False
        )

        f_stars.append(h_star['loss'][-1])

        if debug:
            print(f"    f* ≈ {f_stars[-1]:.6f}")

    a1_losses = [[] for _ in range(4)]
    a1_gaps = [[] for _ in range(4)]
    a1_niters = [[] for _ in range(4)]

    for eps in eps_values:
        if debug:
            print(f"\n  Testing A1 eps={eps}")

        for t in range(4):
            w, h = irls(
                X, Y_tr[:, t],
                eps=eps,
                tol=tol,
                max_iter=max_iter,
                debug=False
            )

            final_loss = h['loss'][-1]
            final_gap = (final_loss - f_stars[t]) / max(abs(f_stars[t]), 1e-14)

            a1_losses[t].append(final_loss)
            a1_gaps[t].append(final_gap)
            a1_niters[t].append(h['n_iter'])

            if debug:
                print(f"    {TARGETS[t]}: loss={final_loss:.6f}, "
                      f"gap={final_gap:.2e}, "
                      f"iters={h['n_iter']}")

    a2_losses = [[] for _ in range(4)]
    a2_gaps = [[] for _ in range(4)]
    a2_niters = [[] for _ in range(4)]

    for mu in mu_values:
        if debug:
            print(f"\n  Testing A2-FGM mu={mu}")

        for t in range(4):
            w, h = fast_smoothed_gradient(
                X, Y_tr[:, t],
                mu=mu,
                tol=tol,
                max_iter=max_iter,
                debug=False
            )

            final_loss = h['loss'][-1]
            final_gap = (final_loss - f_stars[t]) / max(abs(f_stars[t]), 1e-14)

            a2_losses[t].append(final_loss)
            a2_gaps[t].append(final_gap)
            a2_niters[t].append(h['n_iter'])

            if debug:
                print(f"    {TARGETS[t]}: loss={final_loss:.6f}, "
                      f"gap={final_gap:.2e}, "
                      f"iters={h['n_iter']}")

    return {
        'eps_values': eps_values,
        'mu_values': mu_values,

        'a1_losses': [np.array(x) for x in a1_losses],
        'a1_gaps': [np.array(x) for x in a1_gaps],
        'a1_niters': [np.array(x) for x in a1_niters],

        'a2_losses': [np.array(x) for x in a2_losses],
        'a2_gaps': [np.array(x) for x in a2_gaps],
        'a2_niters': [np.array(x) for x in a2_niters],

        'f_stars': f_stars,
        'target_names': TARGETS,
        'm': m,
    }


def experiment_scaling_m(Input_tr, Y_tr, m_values=None, seed=0,
                         eps=1e-6, mu=1e-3, tol=1e-6,
                         max_iter=500, n_repeat=3, debug=True):

    if debug:
        print("\n[experiment_scaling_m]")

    if m_values is None:
        m_values = [10, 25, 50, 100, 200]

    p = Input_tr.shape[1]

    a1_times, a1_times_std = [], []
    a2_times, a2_times_std = [], []
    a1_losses, a2_losses = [], []
    a1_niters, a2_niters = [], []

    for m in m_values:
        if debug:
            print(f"\n  Testing m={m}")

        W1 = random_hidden_weights(m, p, seed=seed)
        X = build_feature_matrix(Input_tr, W1)

        r1_times, r1_losses, r1_niters = [], [], []

        for rep in range(n_repeat):
            if debug:
                print(f"    A1 repeat {rep + 1}/{n_repeat}")

            t0 = time.perf_counter()
            tgt_l, tgt_n = [], []

            for t in range(4):
                w, h = irls(
                    X, Y_tr[:, t],
                    eps=eps,
                    tol=tol,
                    max_iter=max_iter,
                    debug=False
                )
                tgt_l.append(h['loss'][-1])
                tgt_n.append(h['n_iter'])

            elapsed = time.perf_counter() - t0
            r1_times.append(elapsed)
            r1_losses.append(np.mean(tgt_l))
            r1_niters.append(np.mean(tgt_n))

            if debug:
                print(f"      A1 repeat time={elapsed:.3f}s")

        r2_times, r2_losses, r2_niters = [], [], []

        for rep in range(n_repeat):
            if debug:
                print(f"    A2-FGM repeat {rep + 1}/{n_repeat}")

            t0 = time.perf_counter()
            tgt_l, tgt_n = [], []

            for t in range(4):
                w, h = fast_smoothed_gradient(
                    X, Y_tr[:, t],
                    mu=mu,
                    tol=tol,
                    max_iter=max_iter,
                    debug=False
                )
                tgt_l.append(h['loss'][-1])
                tgt_n.append(h['n_iter'])

            elapsed = time.perf_counter() - t0
            r2_times.append(elapsed)
            r2_losses.append(np.mean(tgt_l))
            r2_niters.append(np.mean(tgt_n))

            if debug:
                print(f"      A2-FGM repeat time={elapsed:.3f}s")

        a1_times.append(np.mean(r1_times))
        a1_times_std.append(np.std(r1_times))
        a1_losses.append(np.mean(r1_losses))
        a1_niters.append(np.mean(r1_niters))

        a2_times.append(np.mean(r2_times))
        a2_times_std.append(np.std(r2_times))
        a2_losses.append(np.mean(r2_losses))
        a2_niters.append(np.mean(r2_niters))

        if debug:
            print(f"  Finished m={m}")
            print(f"    A1 avg time={a1_times[-1]:.3f}s")
            print(f"    A2-FGM avg time={a2_times[-1]:.3f}s")

    return {
        'm_values': m_values,
        'a1_times': np.array(a1_times),
        'a1_times_std': np.array(a1_times_std),
        'a1_std': np.array(a1_times_std),
        'a2_times': np.array(a2_times),
        'a2_times_std': np.array(a2_times_std),
        'a2_std': np.array(a2_times_std),
        'a1_losses': np.array(a1_losses),
        'a2_losses': np.array(a2_losses),
        'a1_niters': np.array(a1_niters),
        'a2_niters': np.array(a2_niters),
    }


def experiment_a1_vs_a2(Input_tr, Y_tr, Input_te, Y_te,
                        m=100, seed=0, eps=1e-6, mu=1e-3,
                        tol=1e-6, max_iter=500, debug=True):

    if debug:
        print("\n[experiment_a1_vs_a2]")
        print(f"  Building train/test feature matrices with m={m}")

    W1 = random_hidden_weights(m, Input_tr.shape[1], seed=seed)
    X_tr = build_feature_matrix(Input_tr, W1)
    X_te = build_feature_matrix(Input_te, W1)

    rows = []

    for t in range(4):
        if debug:
            print(f"\n  Target {t + 1}/4: {TARGETS[t]}")

        y_tr = Y_tr[:, t]
        y_te = Y_te[:, t]

        if debug:
            print("    Running long A1 to estimate f*...")

        _, h_star = irls(
            X_tr, y_tr,
            eps=eps,
            tol=tol,
            max_iter=5000,
            debug=False
        )
        f_star = h_star['loss'][-1]

        if debug:
            print(f"    f* ≈ {f_star:.6f}")
            print("    Running A1...")

        w1, h1 = irls(
            X_tr, y_tr,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            debug=debug,
            name=f"A1 {TARGETS[t]}"
        )

        a1_train = l1_loss(X_tr, w1, y_tr)
        a1_test = l1_loss(X_te, w1, y_te)
        a1_gap = (a1_train - f_star) / max(abs(f_star), 1e-14)

        if debug:
            print("    Running A2-FGM...")

        w2, h2 = fast_smoothed_gradient(
            X_tr, y_tr,
            mu=mu,
            tol=tol,
            max_iter=max_iter,
            debug=debug,
            name=f"A2-FGM {TARGETS[t]}"
        )

        a2_train = l1_loss(X_tr, w2, y_tr)
        a2_test = l1_loss(X_te, w2, y_te)
        a2_gap = (a2_train - f_star) / max(abs(f_star), 1e-14)

        rows.append({
            'target': TARGETS[t],
            'f_star': f_star,

            'a1_train_l1': a1_train,
            'a1_test_l1': a1_test,
            'a1_gap': a1_gap,
            'a1_n_iter': h1['n_iter'],
            'a1_converged': h1['converged'],
            'a1_time': h1['time_sec'],

            'a2_train_l1': a2_train,
            'a2_test_l1': a2_test,
            'a2_gap': a2_gap,
            'a2_n_iter': h2['n_iter'],
            'a2_converged': h2['converged'],
            'a2_time': h2['time_sec'],
        })

        if debug:
            print("    Target summary:")
            print(
                f"      A1 train L1={a1_train:.6f}, test L1={a1_test:.6f}, gap={a1_gap:.2e}")
            print(
                f"      A2-FGM train L1={a2_train:.6f}, test L1={a2_test:.6f}, gap={a2_gap:.2e}")

    return {'rows': rows, 'm': m}


def experiment_fair_time_comparison(Input_tr, Y_tr, m=100, seed=0,
                                    eps=1e-6, mu=1e-3, tol=1e-8,
                                    max_iter_a1=500, debug=True):
    """
    Fair comparison: give A2-FGM the same wall-clock time as A1.

    This addresses the reviewer's concern that comparing algorithms
    at equal iteration count is unfair when per-iteration costs differ.
    """
    if debug:
        print("\n[experiment_fair_time_comparison]")
        print(f"  Building ELM feature matrix with m={m}, seed={seed}")

    W1 = random_hidden_weights(m, Input_tr.shape[1], seed=seed)
    X = build_feature_matrix(Input_tr, W1)

    results = []

    # Compute f* once for all targets using long A1 run
    f_stars = []
    if debug:
        print("  Computing f* for all targets...")

    for t in range(4):
        _, h_star = irls(X, Y_tr[:, t], eps=eps, tol=tol,
                         max_iter=5000, debug=False)
        f_stars.append(h_star['loss'][-1])
        if debug:
            print(f"    {TARGETS[t]}: f* = {f_stars[-1]:.6f}")

    for t in range(4):
        y = Y_tr[:, t]

        if debug:
            print(f"\n  Target {t+1}/4: {TARGETS[t]}")

        # Step 1: Run A1 and record its total time
        _, h_a1 = irls(X, y, eps=eps, tol=tol,
                       max_iter=max_iter_a1, debug=False)
        a1_time = h_a1['time_sec']
        a1_loss = h_a1['loss'][-1]
        a1_iters = h_a1['n_iter']

        if debug:
            print(
                f"    A1 finished: {a1_iters} iters in {a1_time:.3f}s, loss={a1_loss:.6f}")

        # Step 2: Estimate A2-FGM cost per iteration
        _, h_test = fast_smoothed_gradient(X, y, mu=mu, tol=None,
                                      max_iter=10, debug=False)
        cost_per_iter = h_test['time_sec'] / 10

        # Step 3: Run A2-FGM for as many iterations as fit in a1_time
        max_iter_a2 = max(1, int(a1_time / cost_per_iter))

        if debug:
            print(f"    A2-FGM cost per iter ~ {cost_per_iter:.6f}s")
            print(f"    Allowing A2-FGM up to {max_iter_a2} iterations")

        _, h_a2 = fast_smoothed_gradient(X, y, mu=mu, tol=None,
                                    max_iter=max_iter_a2, debug=False)
        a2_loss = h_a2['loss'][-1]
        a2_time = h_a2['time_sec']
        a2_iters = h_a2['n_iter']

        a1_gap = (a1_loss - f_stars[t]) / max(abs(f_stars[t]), 1e-14)
        a2_gap = (a2_loss - f_stars[t]) / max(abs(f_stars[t]), 1e-14)

        results.append({
            'target': TARGETS[t],
            'f_star': f_stars[t],
            'a1_time': a1_time,
            'a1_loss': a1_loss,
            'a1_gap': a1_gap,
            'a1_iters': a1_iters,
            'a2_time': a2_time,
            'a2_loss': a2_loss,
            'a2_gap': a2_gap,
            'a2_iters': a2_iters,
        })

        if debug:
            print(f"    Fair comparison result:")
            print(
                f"      A1 ({a1_iters} iters): gap={a1_gap:.2e}, time={a1_time:.3f}s")
            print(
                f"      A2-FGM ({a2_iters} iters): gap={a2_gap:.2e}, time={a2_time:.3f}s")

    return results




def experiment_momentum_comparison(Input_tr, Y_tr, m=100, seed=0,
                                   eps=1e-6, mu=1e-1, tol=1e-8,
                                   max_iter=500, debug=True):
    """
    Compare plain A2-GD and accelerated A2-FGM using the same feature matrix,
    same smoothing parameter, and same iteration budget.

    This isolates the effect of Nesterov momentum without changing the rest of
    the experimental setup.
    """
    if debug:
        print("\n[experiment_momentum_comparison]")
        print(f"  Building ELM feature matrix with m={m}, seed={seed}")
        print(f"  Comparing A2-GD and A2-FGM with the same mu={mu:.1e}")

    W1 = random_hidden_weights(m, Input_tr.shape[1], seed=seed)
    X = build_feature_matrix(Input_tr, W1)

    gd_gaps, fgm_gaps = [], []
    gd_niters, fgm_niters = [], []
    f_stars = []

    for t in range(4):
        y = Y_tr[:, t]

        if debug:
            print(f"\n  Target {t+1}/4: {TARGETS[t]}")
            print("    Computing f* using long A1 run...")

        _, h_star = irls(
            X, y,
            eps=eps,
            tol=tol,
            max_iter=5000,
            debug=False,
            name=f"f* {TARGETS[t]}"
        )
        f_star = h_star['loss'][-1]
        f_stars.append(f_star)

        if debug:
            print(f"    f* ≈ {f_star:.6f}")
            print("    Running A2-GD...")

        _, h_gd = smoothed_gradient(
            X, y,
            mu=mu,
            tol=tol,
            max_iter=max_iter,
            debug=False,
            name=f"A2-GD {TARGETS[t]}"
        )
        h_gd = compute_gap(h_gd, f_star)

        if debug:
            print("    Running A2-FGM...")

        _, h_fgm = fast_smoothed_gradient(
            X, y,
            mu=mu,
            tol=tol,
            max_iter=max_iter,
            debug=False,
            name=f"A2-FGM {TARGETS[t]}"
        )
        h_fgm = compute_gap(h_fgm, f_star)

        gd_gaps.append(np.array(h_gd['gap']))
        fgm_gaps.append(np.array(h_fgm['gap']))
        gd_niters.append(h_gd['n_iter'])
        fgm_niters.append(h_fgm['n_iter'])

        if debug:
            print("    Final gaps:")
            print(f"      A2-GD   gap={h_gd['gap'][-1]:.2e}, iters={h_gd['n_iter']}")
            print(f"      A2-FGM  gap={h_fgm['gap'][-1]:.2e}, iters={h_fgm['n_iter']}")

    return {
        'target_names': TARGETS,
        'm': m,
        'mu': mu,
        'f_stars': f_stars,
        'gd_gaps': gd_gaps,
        'fgm_gaps': fgm_gaps,
        'gd_n_iters': gd_niters,
        'fgm_n_iters': fgm_niters,
    }


def experiment_a2_theory(Input_tr, Y_tr, m=100, seed=0,
                         target_eps=1e-4, max_iter=100_000, debug=True):
    """
    Theory-guided A2 experiment using Nesterov fast gradient.

    This experiment is intentionally separated from the practical A1-vs-A2
    plots. Here mu is not tuned by hand. It is computed from the desired
    tolerance epsilon using

        mu = (epsilon / (2*n))^2,

    where n is the number of training samples. For the sqrt smoothing
    f_mu(w) = sum_i sqrt((x_i^T w - y_i)^2 + mu), this choice gives

        f_mu(w) - f(w) <= n*sqrt(mu) <= epsilon/2.

    The theorem applies to the smoothed objective f_mu, so this function
    records both:
        1. the original L1 gap, for project evaluation;
        2. the smoothed-objective gap, for checking the Nesterov trend.
    """
    if debug:
        print("\n[experiment_a2_theory]")
        print(f"  Building ELM feature matrix with m={m}, seed={seed}")

    W1 = random_hidden_weights(m, Input_tr.shape[1], seed=seed)
    X = build_feature_matrix(Input_tr, W1)
    n = X.shape[0]

    mu = compute_nesterov_mu(n, target_eps)
    op_norm_sq = float(np.linalg.norm(X, 2) ** 2)
    L = op_norm_sq / np.sqrt(mu)
    alpha = 1.0 / L
    bias_abs = n * np.sqrt(mu)

    if debug:
        print(f"  n={n}, target_eps={target_eps:.1e}, mu={mu:.4e}")
        print(f"  ||X||_2^2={op_norm_sq:.4e}, L=||X||_2^2/sqrt(mu)={L:.4e}")
        print(f"  alpha=1/L={alpha:.4e}")
        print(f"  Smoothing bias bound: n*sqrt(mu) = {bias_abs:.4e}")

    results = []

    for t in range(4):
        y = Y_tr[:, t]

        if debug:
            print(f"\n  Target {t+1}/4: {TARGETS[t]}")
            print("    Computing f* for original L1 problem using long A1 run...")

        w_star, h_star = irls(X, y, eps=1e-6, tol=1e-10,
                              max_iter=5000, debug=False)
        f_star = l1_loss(X, w_star, y)
        f_mu_at_w_star = smooth_l1_loss(X, w_star, y, mu)

        if debug:
            print(f"    f* ≈ {f_star:.6f}")
            print(f"    f_mu(w_A1*) ≈ {f_mu_at_w_star:.6f}")
            print(f"    Running A2-FGM, max_iter={max_iter}")

        _, h = fast_smoothed_gradient(
            X, y,
            mu=mu,
            tol=None,              # important: no premature delta-based stop
            max_iter=max_iter,
            debug=debug,
            name=f"A2-FGM {TARGETS[t]}"
        )

        h = compute_gap(h, f_star)
        h = compute_smooth_gap(h, f_mu_at_w_star)  # compare against f_mu at A1 approximate minimizer

        l1_gaps = np.array(h['gap'])
        smooth_gaps = np.array(h['smooth_gap'])
        smooth_losses = np.array(h['smooth_loss'])

        # Reference O(1/k^2) trend for visual comparison.
        if len(smooth_gaps) > 0:
            k_arr = np.arange(1, len(smooth_gaps) + 1)
            start = max(float(smooth_gaps[0]), 1e-16)
            theory_smooth_gaps = start / (k_arr ** 2)
        else:
            theory_smooth_gaps = np.array([])

        rel_bias_floor = bias_abs / max(abs(f_star), 1e-14)

        results.append({
            'target': TARGETS[t],
            'f_star': f_star,
            'f_mu_ref': h.get('f_mu_ref'),
            'f_mu_at_w_star': f_mu_at_w_star,
            'l1_gaps': l1_gaps,
            'gaps': l1_gaps,  # backward-compatible name for plotting
            'smooth_gaps': smooth_gaps,
            'smooth_losses': smooth_losses,
            'theory_smooth_gaps': theory_smooth_gaps,
            'n_iter': h['n_iter'],
            'mu': mu,
            'L': L,
            'alpha': alpha,
            'bias_abs': bias_abs,
            'relative_bias_floor': rel_bias_floor,
            'final_l1_gap': l1_gaps[-1] if len(l1_gaps) > 0 else float('nan'),
            'final_smooth_gap': smooth_gaps[-1] if len(smooth_gaps) > 0 else float('nan'),
        })

        if debug:
            print(f"    Final original L1 gap = {results[-1]['final_l1_gap']:.2e}")
            print(f"    Final smoothed gap    = {results[-1]['final_smooth_gap']:.2e}")
            print(f"    Relative smoothing floor = {rel_bias_floor:.2e}")

    return results