# """
# Project 24: Extreme Learning Machine with L1 Minimization
# ==========================================================

# (M)  Model: y = w @ sigma(W1 @ x)
#      W1 is fixed random, sigma is elementwise activation,
#      w is found by solving:  min_w ||Xw - y||_1

# (A1) Iteratively Reweighted Least Squares (IRLS)
#      Solves L1 via a sequence of weighted L2 problems:
#        w_{k+1} = (1/2) argmin_w  ||W_k (X w - y)||_2^2
#      where W_k is diagonal with entries |(X w_k - y)_i|^{-1/2}
#      (thresholded to avoid blow-up when residual ~ 0)

# (A2) Smoothed Gradient Method
#      Smooth the L1 objective:  f_mu(w) = sum_i sqrt(r_i^2 + mu^2) - mu
#      Gradient:  grad = X^T [ r / sqrt(r^2 + mu^2) ]
#      Use gradient descent with Armijo line search, shrinking mu over iterations.

# No off-the-shelf solvers used anywhere.
# """

# from matplotlib.ticker import MaxNLocator
# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')

# # ─────────────────────────────────────────────
# # Reproducibility
# # ─────────────────────────────────────────────
# RNG = np.random.default_rng(42)

# # ─────────────────────────────────────────────
# # (M) Extreme Learning Machine Setup
# # ─────────────────────────────────────────────


# def sigmoid(z):
#     """Elementwise sigmoid activation."""
#     return 1.0 / (1.0 + np.exp(-z))


# def relu(z):
#     return np.maximum(0.0, z)


# def tanh_act(z):
#     return np.tanh(z)


# # We use sigmoid as our activation of choice
# activation = sigmoid
# activation_name = "sigmoid"


# def build_elm_features(X_raw, n_hidden, scale=1.0):
#     """
#     Build the ELM feature matrix Phi = sigma(X_raw @ W1^T).
#     W1 is fixed random (n_hidden x n_input).
#     Returns Phi (n_samples x n_hidden) and W1.
#     """
#     n_input = X_raw.shape[1] if X_raw.ndim > 1 else 1
#     W1 = RNG.standard_normal((n_hidden, n_input)) * scale
#     Phi = activation(X_raw @ W1.T)      # (n_samples, n_hidden)
#     return Phi, W1


# # ─────────────────────────────────────────────
# # Helper: direct L2 least squares (used inside IRLS)
# # ─────────────────────────────────────────────

# def weighted_least_squares(A, b, W_diag):
#     """
#     Solve  min_w  ||W (A w - b)||_2^2
#     = min_w  (Aw-b)^T W^T W (Aw-b)
#     Normal equations:  (A^T W^2 A) w = A^T W^2 b
#     W_diag: 1-D array of diagonal weights.
#     Solved via numpy's lstsq (this is just linear algebra, not a convex solver).
#     """
#     WA = W_diag[:, None] * A    # element-wise row scaling
#     Wb = W_diag * b
#     # Normal equations via lstsq (pure linear algebra)
#     w, _, _, _ = np.linalg.lstsq(WA, Wb, rcond=None)
#     return w


# # ─────────────────────────────────────────────
# # (A1) IRLS  —  L1 via reweighted L2
# # ─────────────────────────────────────────────

# def irls_l1(A, b, max_iter=200, tol=1e-8, thresh=1e-6):
#     """
#     Iteratively Reweighted Least Squares for L1 minimisation.

#     Iteration:
#         W_k diagonal with  (W_k)_ii = |( A w_k - b )_i|^{-1/2}
#         (thresholded below at thresh to avoid division by ~0)

#         w_{k+1} = (1/2) argmin_w  ||W_k (A w - b)||_2^2

#     The (1/2) factor cancels inside the argmin and is irrelevant for the
#     minimiser; the argmin is just the weighted LS problem above.

#     Returns w*, history of L1 objective values.
#     """
#     n_samples, n_features = A.shape
#     # Initialise with ordinary least squares
#     w = np.linalg.lstsq(A, b, rcond=None)[0]

#     history = []
#     for k in range(max_iter):
#         r = A @ w - b                                   # residuals
#         obj = np.sum(np.abs(r))
#         history.append(obj)

#         # Build diagonal weights  |(r_i)|^{-1/2},  thresholded
#         abs_r = np.abs(r)
#         abs_r_thresh = np.maximum(abs_r, thresh)
#         W_diag = abs_r_thresh ** (-0.5)

#         w_new = weighted_least_squares(A, b, W_diag)

#         # Convergence check
#         if np.linalg.norm(w_new - w) / (1.0 + np.linalg.norm(w)) < tol:
#             w = w_new
#             history.append(np.sum(np.abs(A @ w - b)))
#             break
#         w = w_new

#     return w, np.array(history)


# # ─────────────────────────────────────────────
# # (A2) Smoothed Gradient Method
# # ─────────────────────────────────────────────

# def smoothed_l1_obj(A, b, w, mu):
#     """Smoothed L1:  sum_i ( sqrt(r_i^2 + mu^2) - mu )"""
#     r = A @ w - b
#     return np.sum(np.sqrt(r**2 + mu**2) - mu)


# def smoothed_l1_grad(A, b, w, mu):
#     """Gradient of smoothed L1 w.r.t. w."""
#     r = A @ w - b
#     # pseudo-Huber weights
#     g = r / np.sqrt(r**2 + mu**2)   # (n,)
#     return A.T @ g                   # (d,)


# def smoothed_gradient_l1(A, b, max_iter=500, tol=1e-8,
#                          mu_init=1.0, mu_decay=0.85, mu_min=1e-8,
#                          alpha0=1.0, beta=0.5, c=1e-4):
#     """
#     Smoothed gradient method for L1 minimisation.

#     Smoothing: f_mu(w) = sum_i sqrt(r_i^2 + mu^2) - mu   (Pseudo-Huber)
#     Gradient:  grad_w f_mu = A^T [ r / sqrt(r^2+mu^2) ]

#     Algorithm:
#       - Armijo backtracking line search at each iteration
#       - Gradually reduce mu -> better L1 approximation

#     Returns w*, history of TRUE L1 objective values.
#     """
#     # Initialise
#     w = np.linalg.lstsq(A, b, rcond=None)[0]
#     mu = mu_init

#     history = []
#     for k in range(max_iter):
#         true_obj = np.sum(np.abs(A @ w - b))
#         history.append(true_obj)

#         grad = smoothed_l1_grad(A, b, w, mu)
#         grad_norm = np.linalg.norm(grad)

#         if grad_norm < tol:
#             break

#         # Armijo backtracking line search
#         f0 = smoothed_l1_obj(A, b, w, mu)
#         alpha = alpha0
#         for _ in range(60):
#             w_trial = w - alpha * grad
#             f_trial = smoothed_l1_obj(A, b, w_trial, mu)
#             if f_trial <= f0 - c * alpha * grad_norm**2:
#                 break
#             alpha *= beta

#         w = w - alpha * grad

#         # Decrease smoothing parameter
#         if (k + 1) % 10 == 0:
#             mu = max(mu * mu_decay, mu_min)

#         # Convergence on step size
#         step = alpha * grad_norm
#         if step / (1.0 + np.linalg.norm(w)) < tol and mu <= mu_min * 10:
#             break

#     history.append(np.sum(np.abs(A @ w - b)))
#     return w, np.array(history)


# # ─────────────────────────────────────────────
# # Experiment Setup
# # ─────────────────────────────────────────────

# def generate_dataset(n=120, noise_std=0.15, outlier_frac=0.12, seed=0):
#     """
#     1-D regression with outliers — a good test for L1 robustness.
#     True function: sin(2*pi*x) + 0.3*cos(6*pi*x)
#     """
#     rng = np.random.default_rng(seed)
#     x = rng.uniform(0, 1, n)
#     y_clean = np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x)
#     noise = rng.normal(0, noise_std, n)
#     # Add heavy-tailed outliers
#     n_out = int(outlier_frac * n)
#     idx_out = rng.choice(n, n_out, replace=False)
#     noise[idx_out] += rng.choice([-1, 1], n_out) * rng.uniform(1.5, 3.0, n_out)
#     y = y_clean + noise
#     return x[:, None], y, y_clean


# def run_experiment():
#     print("=" * 60)
#     print("Project 24 — Extreme Learning Machine + L1 Minimisation")
#     print("=" * 60)

#     # ── Data ─────────────────────────────────
#     n_samples = 120
#     X_raw, y, y_clean = generate_dataset(n=n_samples, outlier_frac=0.12)
#     print(f"\nDataset: {n_samples} samples, ~{int(0.12*n_samples)} outliers")

#     # ── ELM feature map ──────────────────────
#     n_hidden = 60
#     Phi, W1 = build_elm_features(X_raw, n_hidden=n_hidden, scale=5.0)
#     print(f"ELM: {n_hidden} hidden units, activation = {activation_name}")
#     print(f"Feature matrix Phi: {Phi.shape}")

#     # ── (A1) IRLS ────────────────────────────
#     print("\n── (A1) IRLS ──────────────────────────")
#     w_irls, hist_irls = irls_l1(Phi, y, max_iter=300, tol=1e-10, thresh=1e-6)
#     l1_irls = np.sum(np.abs(Phi @ w_irls - y))
#     print(f"  Converged in {len(hist_irls)} iterations")
#     print(f"  Final L1 objective: {l1_irls:.6f}")

#     # ── (A2) Smoothed Gradient ───────────────
#     print("\n── (A2) Smoothed Gradient Method ──────")
#     w_sg, hist_sg = smoothed_gradient_l1(
#         Phi, y, max_iter=800, tol=1e-10,
#         mu_init=0.5, mu_decay=0.90, mu_min=1e-9,
#         alpha0=0.5, beta=0.5, c=1e-4
#     )
#     l1_sg = np.sum(np.abs(Phi @ w_sg - y))
#     print(f"  Converged in {len(hist_sg)} iterations")
#     print(f"  Final L1 objective: {l1_sg:.6f}")

#     # ── Baseline: L2 (OLS) for comparison ───
#     w_l2, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
#     l1_l2 = np.sum(np.abs(Phi @ w_l2 - y))
#     print(f"\n── Baseline L2 (OLS) ──────────────────")
#     print(f"  L1 objective with L2 weights: {l1_l2:.6f}")

#     print(f"\n  IRLS vs L2 improvement:  {(l1_l2-l1_irls)/l1_l2*100:.2f}%")
#     print(f"  SG   vs L2 improvement:  {(l1_l2-l1_sg)/l1_l2*100:.2f}%")
#     print(f"  IRLS vs SG  difference:  {abs(l1_irls-l1_sg):.6f}")

#     return X_raw, y, y_clean, Phi, w_irls, w_l2, w_sg, hist_irls, hist_sg


# # ─────────────────────────────────────────────
# # Plotting
# # ─────────────────────────────────────────────

# def plot_results(X_raw, y, y_clean, Phi, w_irls, w_l2, w_sg,
#                  hist_irls, hist_sg, W1):

#     # Dense grid for smooth prediction curves
#     x_grid = np.linspace(0, 1, 500)[:, None]
#     Phi_grid = activation(x_grid @ W1.T)
#     y_true_grid = np.sin(2 * np.pi * x_grid.ravel()) + \
#         0.3 * np.cos(6 * np.pi * x_grid.ravel())

#     pred_irls = Phi_grid @ w_irls
#     pred_sg = Phi_grid @ w_sg
#     pred_l2 = Phi_grid @ w_l2

#     # ── Figure setup ─────────────────────────────────────────────────────────
#     fig = plt.figure(figsize=(16, 11), facecolor='#0d1117')
#     fig.patch.set_facecolor('#0d1117')

#     gs = gridspec.GridSpec(
#         3, 3,
#         figure=fig,
#         hspace=0.50, wspace=0.38,
#         left=0.07, right=0.97, top=0.91, bottom=0.07
#     )

#     ax_main = fig.add_subplot(gs[0:2, :])   # top: regression fit
#     ax_conv = fig.add_subplot(gs[2, 0:2])   # bottom-left: convergence
#     ax_res = fig.add_subplot(gs[2, 2])     # bottom-right: residual box

#     dark_bg = '#0d1117'
#     panel_bg = '#161b22'
#     grid_col = '#21262d'
#     text_col = '#e6edf3'
#     muted = '#8b949e'

#     c_irls = '#58a6ff'   # blue
#     c_sg = '#3fb950'   # green
#     c_l2 = '#f78166'   # red-orange
#     c_true = '#ffa657'   # amber
#     c_data = '#8b949e'   # grey

#     for ax in [ax_main, ax_conv, ax_res]:
#         ax.set_facecolor(panel_bg)
#         ax.tick_params(colors=muted, labelsize=8)
#         for spine in ax.spines.values():
#             spine.set_edgecolor(grid_col)
#         ax.grid(True, color=grid_col, linewidth=0.6, linestyle='--', alpha=0.7)

#     x_flat = X_raw.ravel()

#     # ── Panel 1: Regression fit ───────────────────────────────────────────────
#     ax = ax_main

#     # Scatter: separate outliers visually
#     residuals_irls = np.abs(Phi @ w_irls - y)
#     thresh_out = np.percentile(residuals_irls, 88)
#     is_out = residuals_irls > thresh_out

#     ax.scatter(x_flat[~is_out], y[~is_out], s=18, color=c_data, alpha=0.65,
#                zorder=2, label='Data (inliers)', linewidths=0)
#     ax.scatter(x_flat[is_out],  y[is_out],  s=30, color='#ff7b72', alpha=0.85,
#                zorder=2, label='Outliers', marker='x', linewidths=1.2)

#     ax.plot(x_grid, y_true_grid, color=c_true, lw=1.8, ls='--',
#             alpha=0.8, zorder=3, label='True function')
#     ax.plot(x_grid, pred_l2,   color=c_l2,  lw=1.6, alpha=0.75,
#             zorder=4, label='L2 / OLS baseline')
#     ax.plot(x_grid, pred_irls, color=c_irls, lw=2.2, zorder=5,
#             label='(A1) IRLS — L1')
#     ax.plot(x_grid, pred_sg,   color=c_sg,  lw=2.2, ls=(0, (5, 3)),
#             zorder=5, label='(A2) Smoothed Gradient — L1')

#     ax.set_title('ELM Regression  ·  L1 vs L2 Minimisation',
#                  color=text_col, fontsize=13, fontweight='bold', pad=10)
#     ax.set_xlabel('x', color=muted, fontsize=9)
#     ax.set_ylabel('y', color=muted, fontsize=9)
#     leg = ax.legend(loc='upper right', fontsize=8, framealpha=0.25,
#                     facecolor=panel_bg, edgecolor=grid_col, labelcolor=text_col)

#     # Annotation box
#     l1_irls = np.sum(np.abs(Phi @ w_irls - y))
#     l1_sg = np.sum(np.abs(Phi @ w_sg - y))
#     l1_l2 = np.sum(np.abs(Phi @ w_l2 - y))
#     info = (f"L1 objectives\n"
#             f"IRLS : {l1_irls:.3f}\n"
#             f"SG   : {l1_sg:.3f}\n"
#             f"L2   : {l1_l2:.3f}")
#     ax.text(0.01, 0.97, info, transform=ax.transAxes, fontsize=7.5,
#             verticalalignment='top', color=text_col,
#             bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d',
#                       edgecolor=grid_col, alpha=0.9),
#             fontfamily='monospace')

#     # ── Panel 2: Convergence ──────────────────────────────────────────────────
#     ax = ax_conv

#     iters_irls = np.arange(1, len(hist_irls) + 1)
#     iters_sg = np.arange(1, len(hist_sg) + 1)

#     ax.semilogy(iters_irls, hist_irls, color=c_irls, lw=2.0, label='(A1) IRLS')
#     ax.semilogy(iters_sg,   hist_sg,   color=c_sg,
#                 lw=2.0, label='(A2) Smoothed Gradient')
#     ax.axhline(y=l1_irls, color=c_irls, ls=':', lw=1.0, alpha=0.5)
#     ax.axhline(y=l1_sg,   color=c_sg,   ls=':', lw=1.0, alpha=0.5)

#     ax.set_title('Convergence of L1 Objective', color=text_col,
#                  fontsize=10, fontweight='bold', pad=8)
#     ax.set_xlabel('Iteration', color=muted, fontsize=8)
#     ax.set_ylabel('||Xw − y||₁  (log)', color=muted, fontsize=8)
#     ax.legend(fontsize=8, framealpha=0.2, facecolor=panel_bg,
#               edgecolor=grid_col, labelcolor=text_col)
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

#     # ── Panel 3: Residual distributions ───────────────────────────────────────
#     ax = ax_res

#     res_irls = Phi @ w_irls - y
#     res_sg = Phi @ w_sg - y
#     res_l2 = Phi @ w_l2 - y

#     bp = ax.boxplot(
#         [res_l2, res_irls, res_sg],
#         labels=['L2', 'IRLS', 'SG'],
#         patch_artist=True,
#         medianprops=dict(color='white', linewidth=1.8),
#         whiskerprops=dict(color=muted),
#         capprops=dict(color=muted),
#         flierprops=dict(marker='o', color=muted, alpha=0.4, markersize=3)
#     )
#     colors_bp = [c_l2, c_irls, c_sg]
#     for patch, col in zip(bp['boxes'], colors_bp):
#         patch.set_facecolor(col)
#         patch.set_alpha(0.5)
#         patch.set_edgecolor(col)

#     ax.axhline(0, color=grid_col, lw=1.0)
#     ax.set_title('Residual Distribution', color=text_col,
#                  fontsize=10, fontweight='bold', pad=8)
#     ax.set_ylabel('Residual', color=muted, fontsize=8)
#     ax.tick_params(axis='x', colors=text_col, labelsize=8)

#     # ── Super title ──────────────────────────────────────────────────────────
#     fig.suptitle(
#         'Project 24  ·  Extreme Learning Machine  ·  L₁ Minimisation',
#         color=text_col, fontsize=15, fontweight='bold', y=0.975
#     )

#     out = 'project24_results.png'
#     fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=dark_bg)
#     plt.close(fig)
#     print(f"\nPlot saved → {out}")
#     return out


# # ─────────────────────────────────────────────
# # Monkey-patch to expose W1 from build_elm_features
# # ─────────────────────────────────────────────
# _orig_build = build_elm_features


# def build_elm_features(X_raw, n_hidden, scale=1.0):
#     n_input = X_raw.shape[1] if X_raw.ndim > 1 else 1
#     W1 = RNG.standard_normal((n_hidden, n_input)) * scale
#     Phi = activation(X_raw @ W1.T)
#     return Phi, W1


# # ─────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────

# if __name__ == '__main__':
#     # Reset RNG so W1 is deterministic for both training and grid plotting
#     RNG = np.random.default_rng(42)

#     X_raw, y, y_clean = generate_dataset(n=120, outlier_frac=0.12, seed=0)

#     n_hidden = 60
#     Phi, W1 = build_elm_features(X_raw, n_hidden=n_hidden, scale=5.0)

#     print("=" * 60)
#     print("Project 24 — Extreme Learning Machine + L1 Minimisation")
#     print("=" * 60)
#     print(f"\nDataset: 120 samples, ~14 outliers")
#     print(f"ELM: {n_hidden} hidden units, activation = {activation_name}")
#     print(f"Feature matrix Phi: {Phi.shape}")

#     # A1: IRLS
#     print("\n── (A1) IRLS ──────────────────────────")
#     w_irls, hist_irls = irls_l1(Phi, y, max_iter=600, tol=1e-12, thresh=1e-6)
#     l1_irls = np.sum(np.abs(Phi @ w_irls - y))
#     print(f"  Converged in {len(hist_irls)} iterations")
#     print(f"  Final L1 objective: {l1_irls:.6f}")

#     # A2: Smoothed Gradient
#     print("\n── (A2) Smoothed Gradient Method ──────")
#     w_sg, hist_sg = smoothed_gradient_l1(
#         Phi, y, max_iter=2000, tol=1e-12,
#         mu_init=0.1, mu_decay=0.95, mu_min=1e-10,
#         alpha0=1.0, beta=0.5, c=1e-4
#     )
#     l1_sg = np.sum(np.abs(Phi @ w_sg - y))
#     print(f"  Converged in {len(hist_sg)} iterations")
#     print(f"  Final L1 objective: {l1_sg:.6f}")

#     # Baseline L2
#     w_l2, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
#     l1_l2 = np.sum(np.abs(Phi @ w_l2 - y))
#     print(f"\n── Baseline L2 (OLS) ──────────────────")
#     print(f"  L1 objective with L2 weights: {l1_l2:.6f}")
#     print(f"  IRLS vs L2 improvement:  {(l1_l2-l1_irls)/l1_l2*100:.2f}%")
#     print(f"  SG   vs L2 improvement:  {(l1_l2-l1_sg)/l1_l2*100:.2f}%")
#     print(f"  IRLS vs SG  difference:  {abs(l1_irls-l1_sg):.6f}")

#     # Plot
#     plot_results(X_raw, y, y_clean, Phi, w_irls,
#                  w_l2, w_sg, hist_irls, hist_sg, W1)
#     print("\nDone.")
