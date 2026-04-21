"""
irls_pure.py
------------
Pure IRLS implementation following exactly the 14 steps
described in the personal understanding notes, verified
against Project 24 description.

Run from the project root:
    python irls_pure.py

No experiments framework. No A2. No comparison.
Just the exact pipeline from raw data to final prediction.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os

# =============================================================================
# CONFIGURATION — change these values to experiment
# =============================================================================

DATA_PATH = 'dataset/ML-CUP25-TR.csv'
M = 100       # number of hidden neurons
ACTIVATION = 'tanh'    # activation function: 'tanh', 'relu', 'sigmoid'
EPS = 1e-6      # threshold: prevents 1/sqrt(|r|) exploding near zero
TOL = 1e-6      # stop when ||w_new - w_old|| < TOL
MAX_ITER = 500      # maximum IRLS iterations
SEED = 42        # random seed for W1 (controls the random projection)
TEST_FRAC = 0.1       # fraction of data held out for testing

# =============================================================================
# PHASE 1 — SETUP
# =============================================================================

# ── Step 1: Load input data and true targets ──────────────────────────────────
# Read the CSV, skip comment lines, split into Input (features) and Y (targets)

print("=" * 60)
print("PHASE 1 — SETUP")
print("=" * 60)

print("\nStep 1: Loading data...")

rows = []
with open(DATA_PATH, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue                            # skip header comments
        rows.append([float(x) for x in line.split(',')])

data = np.array(rows)     # shape (500, 17): ID + 12 inputs + 4 targets
Input_raw = data[:, 1:13]      # 12 input features — your x_i values
Y_raw = data[:, 13:]       # 4 target columns  — your y_true values

n_total, p = Input_raw.shape   # n_total=500, p=12
n_targets = Y_raw.shape[1]    # 4 targets

print(f"  Total samples  : {n_total}")
print(f"  Input features : {p}")
print(f"  Targets        : {n_targets}  (TARGET_1 to TARGET_4)")

# standardize inputs: zero mean, unit variance
# this prevents features with large values from dominating
mean_input = Input_raw.mean(axis=0)
std_input = Input_raw.std(axis=0)
std_input[std_input == 0] = 1.0        # guard constant columns
Input = (Input_raw - mean_input) / std_input

# train / test split
rng = np.random.default_rng(SEED)
idx = rng.permutation(n_total)
n_test = max(1, int(n_total * TEST_FRAC))
test_idx = idx[:n_test]
train_idx = idx[n_test:]

Input_tr = Input[train_idx]     # shape (450, 12) — seen during training
Input_te = Input[test_idx]      # shape (50,  12) — held out for evaluation
Y_tr = Y_raw[train_idx]     # shape (450, 4)
Y_te = Y_raw[test_idx]      # shape (50,  4)

n_tr = Input_tr.shape[0]
n_te = Input_te.shape[0]
print(f"  Training       : {n_tr} samples")
print(f"  Test           : {n_te} samples")

# ── Step 2: Choose number of neurons and activation function ──────────────────

print(f"\nStep 2: Configuration")
print(f"  Hidden neurons : m = {M}")
print(f"  Activation     : {ACTIVATION}")
print(f"  IRLS threshold : eps = {EPS}")
print(f"  Tolerance      : tol = {TOL}")
print(f"  Max iterations : {MAX_ITER}")
print(f"  Random seed    : {SEED}")

# ── Step 3: Generate W1 — random, fixed, never changed ───────────────────────
# W1 shape: (m, p) = (100, 12)
# entries drawn from Uniform(-1, 1)
# this matrix is generated ONCE and never touched again

print(f"\nStep 3: Generating fixed random W1...")
W1 = rng.uniform(-1.0, 1.0, size=(M, p))   # shape (M, p) = (100, 12)
print(f"  W1 shape: {W1.shape}  (m x p = {M} x {p})")
print(f"  W1 is now frozen forever.")

# =============================================================================
# PHASE 2 — BUILD THE FEATURE MATRIX X
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 2 — BUILD FEATURE MATRIX X")
print("=" * 60)

# ── Step 4: Compute W1 @ x_i for each sample ─────────────────────────────────
# For each sample (row), multiply its p features by all m neuron weights.
# Result: n raw activation values, one per neuron.
# Done for all samples at once: Input_tr @ W1.T

print("\nStep 4: Computing W1 @ x_i for every sample...")
Z_tr = Input_tr @ W1.T     # shape (n_tr, M) = (450, 100)
Z_te = Input_te @ W1.T     # shape (n_te, M) = (50,  100)
print(f"  Z_tr shape: {Z_tr.shape}  (n_tr x m)")

# ── Step 5: Apply activation function σ elementwise ──────────────────────────
# σ(z) applied to every single number in Z.
# For tanh: maps any number into (-1, +1)
# For relu: if z < 0 → 0, else keep z

print(f"\nStep 5: Applying σ = {ACTIVATION} elementwise...")

if ACTIVATION == 'tanh':
    X_tr = np.tanh(Z_tr)           # shape (n_tr, M) = (450, 100)
    X_te = np.tanh(Z_te)
elif ACTIVATION == 'relu':
    X_tr = np.maximum(0.0, Z_tr)
    X_te = np.maximum(0.0, Z_te)
elif ACTIVATION == 'sigmoid':
    X_tr = 1.0 / (1.0 + np.exp(-Z_tr))
    X_te = 1.0 / (1.0 + np.exp(-Z_te))

# ── Step 6: X is now ready — shape (n_tr, M) ─────────────────────────────────
# Each row = one sample seen through m neurons
# Each column = what one specific neuron says about all samples

print(f"\nStep 6: Feature matrix X is ready.")
print(f"  X_tr shape: {X_tr.shape}  (n_tr x m = {n_tr} x {M})")
print(f"  Each row   = one sample's neuron outputs")
print(f"  Each col   = one neuron's output across all samples")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def gaussian_elimination(A, b):
    """
    Solve A @ w = b from scratch using Gaussian elimination with
    partial pivoting. No library solver used.

    Called at Step 8h of every IRLS iteration.

    Parameters
    ----------
    A : ndarray, shape (m, m)
    b : ndarray, shape (m,)

    Returns
    -------
    w : ndarray, shape (m,)
    """
    m = len(b)

    # build augmented matrix [A | b]
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    # forward elimination with partial pivoting
    for col in range(m):

        # find the largest value in this column from current row down
        # swap it to the top to avoid dividing by near-zero numbers
        pivot = col + np.argmax(np.abs(Ab[col:, col]))
        Ab[[col, pivot]] = Ab[[pivot, col]]

        # if diagonal is effectively zero — matrix is singular
        if abs(Ab[col, col]) < 1e-14:
            return np.zeros(m)

        # eliminate everything below the diagonal in this column
        for row in range(col + 1, m):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]

    # back substitution — solve from bottom row upward
    w = np.zeros(m)
    for row in range(m - 1, -1, -1):
        w[row] = Ab[row, m]
        for j in range(row + 1, m):
            w[row] -= Ab[row, j] * w[j]
        w[row] /= Ab[row, row]

    return w


# =============================================================================
# PHASE 3 — IRLS TO FIND OPTIMAL OUTPUT WEIGHTS w
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 3 — IRLS (one run per target)")
print("=" * 60)

# store results for all 4 targets
all_w = []          # final w vectors — one per target
all_losses = []          # L1 loss history — one array per target
all_niters = []          # iteration counts
all_conv = []          # convergence flags
all_times = []          # wall-clock times

for target_idx in range(n_targets):

    target_name = f"TARGET_{target_idx + 1}"
    y_tr = Y_tr[:, target_idx]      # shape (n_tr,) — y_true for this target

    print(f"\n--- {target_name} ---")

    # ── Step 7: Initialize w0 = [0, 0, ..., 0] ───────────────────────────────
    # shape (M,) — one zero per neuron
    # any starting point works for convex problems
    w = np.zeros(M)
    print(f"  Step 7: w0 initialized to zeros  shape={w.shape}")

    loss_history = []           # track L1 loss at each iteration
    w_change_history = []       # track ||w_new - w|| at each iteration
    converged = False
    t_start = time.perf_counter()

    # ── Step 8: Main IRLS loop ────────────────────────────────────────────────
    for k in range(MAX_ITER):

        # Step 8a — compute predictions Xw_k
        # shape (n_tr,) — one prediction per training sample
        predictions = X_tr @ w                             # shape (n_tr,)

        # Step 8b — compute residuals r_k = Xw_k - y_true
        # r_i > 0 means we predicted too high
        # r_i < 0 means we predicted too low
        r = predictions - y_tr                             # shape (n_tr,)

        # record L1 loss BEFORE updating w
        # L1 loss = sum of absolute errors = total euro-error in house analogy
        l1 = float(np.sum(np.abs(r)))
        loss_history.append(l1)

        # Step 8c — take absolute values |r_k|
        abs_r = np.abs(r)                                  # shape (n_tr,)

        # Step 8d — compute diagonal weights
        # (W_k)_ii = |r_i|^{-1/2} = 1 / sqrt(|r_i|)
        # NOTE: square root in denominator — confirmed from project description
        # threshold: use eps instead of |r_i| when |r_i| is near zero
        # this prevents the weight from exploding to infinity
        diag_w = 1.0 / np.sqrt(np.maximum(abs_r, EPS))    # shape (n_tr,)

        # Step 8e — build diagonal matrix W_k
        # We never form the full n x n diagonal matrix explicitly.
        # Instead we scale X and y row-by-row: equivalent and O(nm) not O(n^2).
        # X_scaled[i, :] = diag_w[i] * X_tr[i, :]
        # y_scaled[i]    = diag_w[i] * y_tr[i]
        X_scaled = diag_w[:, None] * X_tr                  # shape (n_tr, M)
        y_scaled = diag_w * y_tr                           # shape (n_tr,)

        # Step 8f — compute M = X^T W_k^2 X
        # X_scaled = W_k @ X  so  X_scaled.T @ X_scaled = X^T W_k^2 X
        M_mat = X_scaled.T @ X_scaled                      # shape (M, M)

        # Step 8g — compute b = X^T W_k^2 y_true
        b_vec = X_scaled.T @ y_scaled                      # shape (M,)

        # Step 8h — solve M w* = b using Gaussian elimination
        # No library solver — implemented from scratch above
        w_new = gaussian_elimination(M_mat, b_vec)         # shape (M,)

        # Step 8i — update w_{k+1} = w*
        # The 1/2 in the project formula is absorbed in the normal equations.
        # The actual update is simply w_{k+1} = w*  (no scaling factor).
        delta = float(np.linalg.norm(w_new - w))   # permitted sub-step
        w_change_history.append(delta)
        w = w_new

        # print progress every 100 iterations
        if (k + 1) % 100 == 0 or k == 0:
            print(f"  iter {k+1:4d}:  L1 loss = {l1:.4f}  "
                  f"||w_change|| = {delta:.2e}")

        # Step 9 — check convergence: stop when w barely changes
        if delta < TOL:
            converged = True
            print(f"  Converged at iteration {k+1}!")
            break

    elapsed = time.perf_counter() - t_start

    # record final loss after last update
    r_final = X_tr @ w - y_tr
    loss_history.append(float(np.sum(np.abs(r_final))))

    all_w.append(w)
    all_losses.append(loss_history)
    all_niters.append(k + 1)
    all_conv.append(converged)
    all_times.append(elapsed)

    print(f"  Final L1 loss  : {loss_history[-1]:.4f}")
    print(f"  Iterations     : {k+1}  |  Converged: {converged}")
    print(f"  Time           : {elapsed:.3f}s")

# =============================================================================
# PHASE 4 — PREDICT NEW SAMPLES (test set)
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 4 — PREDICT ON TEST SET")
print("=" * 60)

# ── Steps 10-13: for each target, compute predictions on unseen test data ─────

print("\nFor each target — using the w found by IRLS:")
print(f"\n  {'Target':<12} {'Train L1':>12} {'Test L1':>12} "
      f"{'Iters':>8} {'Conv':>6} {'Time':>8}")
print("  " + "-" * 62)

# shape (450, 4) — training predictions
Y_hat_tr = np.zeros((n_tr, n_targets))
Y_hat_te = np.zeros((n_te, n_targets))   # shape (50,  4) — test predictions

for t in range(n_targets):
    w_final = all_w[t]

    # Step 10-12: for test sample x_new:
    #   Step 11: W1 @ x_new  (already done — X_te = sigma(Input_te @ W1.T))
    #   Step 12: sigma applied  (already in X_te)
    #   Step 13: y_pred = w_final . h_new = X_te[i, :] @ w_final

    # Step 14: output predictions
    y_hat_tr = X_tr @ w_final              # shape (n_tr,)
    y_hat_te = X_te @ w_final              # shape (n_te,)

    Y_hat_tr[:, t] = y_hat_tr
    Y_hat_te[:, t] = y_hat_te

    train_l1 = float(np.sum(np.abs(y_hat_tr - Y_tr[:, t])))
    test_l1 = float(np.sum(np.abs(y_hat_te - Y_te[:, t])))
    conv_str = 'yes' if all_conv[t] else 'no'

    print(f"  TARGET_{t+1:<7} {train_l1:>12.4f} {test_l1:>12.4f} "
          f"{all_niters[t]:>8d} {conv_str:>6} {all_times[t]:>7.3f}s")

# Final prediction matrix shape
print(f"\n  One IRLS run  : X ({n_tr}x{M}) @ w ({M}x1) = y_hat ({n_tr}x1)")
print(f"  Stack 4 cols  : Y_hat = ({n_tr}x4)")
print(f"\n  Y_hat_tr shape: {Y_hat_tr.shape}  (training predictions)")
print(f"  Y_hat_te shape: {Y_hat_te.shape}  (test predictions)")

# =============================================================================
# PLOT — L1 loss vs iteration for all 4 targets
# =============================================================================

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

for t in range(n_targets):
    iters = np.arange(len(all_losses[t]))
    conv = 'converged' if all_conv[t] else f'{all_niters[t]} iters'
    ax.semilogy(iters, all_losses[t],
                color=colors[t], linewidth=1.5,
                label=f'TARGET_{t+1}  ({conv})')

ax.set_xlabel('Iteration  (k)')
ax.set_ylabel(r'L1 loss  $f(w^{(k)}) = \|Xw^{(k)} - y\|_1$')
ax.set_title(f'IRLS convergence  —  m={M}, activation={ACTIVATION}, '
             f'eps={EPS}, tol={TOL}')
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()

path = 'figures/irls_pure_convergence.pdf'
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"\nConvergence plot saved to {path}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
