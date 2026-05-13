"""
main.py
-------
Entry point for Project 24.

Run from the project root:
    python main.py

Loads ML-CUP 2025, runs all 4 experiments comparing A1 (IRLS) and
A2 (smoothed gradient descent), measures relative gap as the primary
metric, saves all figures to ./figures/.

Primary metric (as required by professors):
    gap^(k) = (f(w^(k)) - f*) / |f*|
where f* is approximated by running A1 for 5000 iterations.
Target: gap <= 1e-2 (acceptable), gap <= 1e-4 (good).
"""

from viz.plots import fig5_fair_comparison
from utils.experiments import experiment_fair_time_comparison
import numpy as np

from utils.data import load_cup, train_test_split
from utils.experiments import (experiment_convergence,
                               experiment_params,
                               experiment_scaling_m,
                               experiment_a1_vs_a2)
from viz.plots import (fig1_gap_convergence,
                       fig2_params,
                       fig3_scaling,
                       fig4_comparison,
                       print_comparison_table)

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_PATH = 'dataset/ML-CUP25-TR.csv'
SEED = 42
M = 100      # hidden neurons
EPS = 1e-6     # A1 threshold
MU = 1e-3     # A2 smoothing parameter
TOL = 1e-8     # convergence tolerance (both algorithms)
MAX_ITER = 500      # max iterations per run (both algorithms)
TEST_FRAC = 0.1      # 10% held out for evaluation

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading ML-CUP 2025...')
Input, Y, scaler_mean, scaler_std = load_cup(TRAIN_PATH)
print(f'  n={Input.shape[0]}  p={Input.shape[1]}  targets={Y.shape[1]}')

Input_tr, Input_te, Y_tr, Y_te = train_test_split(
    Input, Y, test_size=TEST_FRAC, seed=SEED)
print(f'  Train: {Input_tr.shape[0]}  Test: {Input_te.shape[0]}')

# ── Experiment 1: Convergence (gap vs iteration) ───────────────────────────────
print("\n[1/4] Convergence experiment started")
res_conv = experiment_convergence(
    Input_tr, Y_tr,
    m=100,
    eps=1e-6,
    mu=1e-3,
    tol=1e-8,
    max_iter=500,
    seed=0,
    debug=True
)
print("[1/4] Convergence experiment finished")

for t, name in enumerate(res_conv['target_names']):
    print(f'{name}: f*={res_conv["f_star"][t]:.2f}  '
          f'A1 gap={res_conv["a1_gaps"][t][-1]:.2e} '
          f'({res_conv["a1_n_iters"][t]} iters)  '
          f'A2 gap={res_conv["a2_gaps"][t][-1]:.2e} '
          f'({res_conv["a2_n_iters"][t]} iters)')
fig1_gap_convergence(res_conv)

print("\n[2/4] Parameter experiment started")
res_params = experiment_params(
    Input_tr, Y_tr,
    m=100,
    seed=0,
    tol=1e-8,
    max_iter=500,
    debug=True
)
print("[2/4] Parameter experiment finished")
fig2_params(res_params)

# ── Experiment 3: Scaling with m ──────────────────────────────────────────────
print("\n[3/4] Scaling experiment started")
res_scaling = experiment_scaling_m(
    Input_tr, Y_tr,
    m_values=[10, 25, 50, 100, 200],
    seed=0,
    eps=1e-6,
    mu=1e-3,
    tol=1e-8,
    max_iter=500,
    n_repeat=3,
    debug=True
)
print("[3/4] Scaling experiment finished")
fig3_scaling(res_scaling)

# ── Experiment 4: A1 vs A2 head-to-head ───────────────────────────────────────
print("\n[4/4] A1 vs A2 experiment started")
res_comp = experiment_a1_vs_a2(
    Input_tr, Y_tr,
    Input_te, Y_te,
    m=100,
    seed=0,
    eps=1e-6,
    mu=1e-3,
    tol=1e-8,
    max_iter=500,
    debug=True
)
print("[4/4] A1 vs A2 experiment finished")
fig4_comparison(res_comp)
# ── Experiment 5: Fair time-based comparison ───────────────────────────────────
print("\n[5/5] Fair time-based comparison started")

res_fair = experiment_fair_time_comparison(
    Input_tr, Y_tr,
    m=100, seed=0, eps=1e-6, mu=1e-3, tol=1e-8,
    max_iter_a1=500, debug=True
)
print("[5/5] Fair time-based comparison finished")
fig5_fair_comparison(res_fair)

# Print fair comparison table
print("\n" + "="*60)
print("FAIR COMPARISON: Equal wall-clock time budget")
print("="*60)
print(f"{'Target':<12} {'A1 gap':>12} {'A1 time':>10} {'A2 gap':>12} {'A2 iters':>12} {'A2 time':>10}")
print("-"*70)
for r in res_fair:
    print(f"{r['target']:<12} {r['a1_gap']:>12.2e} {r['a1_time']:>10.3f}s "
          f"{r['a2_gap']:>12.2e} {r['a2_iters']:>12d} {r['a2_time']:>10.3f}s")
print('\nDone. Figures saved to ./figures/')
