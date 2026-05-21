"""
main.py
-------
Entry point for Project 24.

Run from the project root:
    python main.py

Loads ML-CUP 2025, compares A1 (IRLS) and A2, and adds a separate
Nesterov theory-guided A2 experiment.

Primary metric:
    gap^(k) = (f(w^(k)) - f*) / |f*|
where f* is approximated by running A1 for 5000 iterations.
"""

import numpy as np

from utils.data import load_cup, train_test_split
from core.solvers import compute_nesterov_mu
from utils.experiments import (experiment_convergence,
                               experiment_params,
                               experiment_scaling_m,
                               experiment_a1_vs_a2,
                               experiment_fair_time_comparison,
                               experiment_momentum_comparison,
                               experiment_a2_theory)
from viz.plots import (fig1_gap_convergence,
                       fig2_params,
                       fig3_scaling,
                       fig4_comparison,
                       fig5_fair_comparison,
                       fig6_a2_theory,
                       fig7_momentum_comparison)

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_PATH = 'dataset/ML-CUP25-TR.csv'
SEED = 42
M = 100
EPS = 1e-6
TOL = 1e-8
MAX_ITER = 1000
TEST_FRAC = 0.1

# Practical A2 parameter used for ordinary comparison plots.
# This is intentionally separate from the theory-guided Nesterov mu.
MU_PRACTICAL = 1e-1

# Theory-guided A2-FGM setup from Nesterov-style smoothing.
TARGET_EPS = 1e-4
A2_THEORY_MAX_ITER = 200_000

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading ML-CUP 2025...')
Input, Y, scaler_mean, scaler_std = load_cup(TRAIN_PATH)
print(f'  n={Input.shape[0]}  p={Input.shape[1]}  targets={Y.shape[1]}')

Input_tr, Input_te, Y_tr, Y_te = train_test_split(
    Input, Y, test_size=TEST_FRAC, seed=SEED)
print(f'  Train: {Input_tr.shape[0]}  Test: {Input_te.shape[0]}')

MU_THEORY = compute_nesterov_mu(Input_tr.shape[0], TARGET_EPS)
print(f'  Practical A2 mu={MU_PRACTICAL:.1e}')
print(f'  A2-FGM target_eps={TARGET_EPS:.1e}  theory-guided mu={MU_THEORY:.4e}')

# ── Experiment 1: Convergence with practical A2 mu ─────────────────────────────
print("\n[1/6] Convergence experiment started")
res_conv = experiment_convergence(
    Input_tr, Y_tr,
    m=M,
    eps=EPS,
    mu=MU_PRACTICAL,
    tol=TOL,
    max_iter=MAX_ITER,
    seed=0,
    debug=True
)
print("[1/6] Convergence experiment finished")

for t, name in enumerate(res_conv['target_names']):
    print(f'{name}: f*={res_conv["f_star"][t]:.2f}  '
          f'A1 gap={res_conv["a1_gaps"][t][-1]:.2e} '
          f'({res_conv["a1_n_iters"][t]} iters)  '
          f'A2-FGM gap={res_conv["a2_gaps"][t][-1]:.2e} '
          f'({res_conv["a2_n_iters"][t]} iters)')
fig1_gap_convergence(res_conv)

# ── Experiment 2: Parameter sensitivity ───────────────────────────────────────
print("\n[2/6] Parameter experiment started")
res_params = experiment_params(
    Input_tr, Y_tr,
    m=M,
    seed=0,
    tol=TOL,
    max_iter=MAX_ITER,
    debug=True
)
print("[2/6] Parameter experiment finished")
fig2_params(res_params)

# ── Experiment 3: Scaling with m ──────────────────────────────────────────────
print("\n[3/6] Scaling experiment started")
res_scaling = experiment_scaling_m(
    Input_tr, Y_tr,
    m_values=[10, 25, 50, 100, 200],
    seed=0,
    eps=EPS,
    mu=MU_PRACTICAL,
    tol=TOL,
    max_iter=MAX_ITER,
    n_repeat=3,
    debug=True
)
print("[3/6] Scaling experiment finished")
fig3_scaling(res_scaling)

# ── Experiment 4: A1 vs practical A2 ──────────────────────────────────────────
print("\n[4/6] A1 vs A2-FGM experiment started")
res_comp = experiment_a1_vs_a2(
    Input_tr, Y_tr,
    Input_te, Y_te,
    m=M,
    seed=0,
    eps=EPS,
    mu=MU_PRACTICAL,
    tol=TOL,
    max_iter=MAX_ITER,
    debug=True
)
print("[4/6] A1 vs A2-FGM experiment finished")
fig4_comparison(res_comp)

# ── Experiment 5: Fair time-based comparison with practical A2 ────────────────
print("\n[5/6] Fair time-based comparison started")
res_fair = experiment_fair_time_comparison(
    Input_tr, Y_tr,
    m=M,
    seed=0,
    eps=EPS,
    mu=MU_PRACTICAL,
    tol=TOL,
    max_iter_a1=MAX_ITER,
    debug=True
)
print("[5/6] Fair time-based comparison finished")
fig5_fair_comparison(res_fair)

print("\n" + "="*60)
print("FAIR COMPARISON: Equal wall-clock time budget")
print("="*60)
print(f"{'Target':<12} {'A1 gap':>12} {'A1 time':>10} {'A2-FGM gap':>12} {'A2-FGM iters':>14} {'A2-FGM time':>12}")
print("-"*70)
for r in res_fair:
    print(f"{r['target']:<12} {r['a1_gap']:>12.2e} {r['a1_time']:>10.3f}s "
          f"{r['a2_gap']:>12.2e} {r['a2_iters']:>12d} {r['a2_time']:>10.3f}s")



# ── Extra experiment: isolate Nesterov momentum effect ────────────────────────
print("\n[extra] Momentum comparison started")
res_momentum = experiment_momentum_comparison(
    Input_tr, Y_tr,
    m=M,
    seed=0,
    eps=EPS,
    mu=MU_PRACTICAL,
    tol=TOL,
    max_iter=MAX_ITER,
    debug=True
)
print("[extra] Momentum comparison finished")
fig7_momentum_comparison(res_momentum)

# ── Experiment 6: Theory-guided Nesterov A2-FGM ───────────────────────────────
print("\n[6/6] A2 theory-guided Nesterov experiment started")
res_a2_theory = experiment_a2_theory(
    Input_tr, Y_tr,
    m=M,
    seed=0,
    target_eps=TARGET_EPS,
    max_iter=A2_THEORY_MAX_ITER,
    debug=True
)
print("[6/6] A2 theory-guided Nesterov experiment finished")
fig6_a2_theory(res_a2_theory)

print('\nDone. Figures saved to ./figures/')
