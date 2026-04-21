"""
main.py
-------
Entry point. Run from the project root:

    python main.py

Loads ML-CUP 2025 data, runs all 4 experiments comparing A1 (IRLS) and
A2 (smoothed gradient descent), saves all figures to ./figures/.

Both algorithms solve the same L1 problem:
    min_w  ||Xw - y||_1

The comparison is purely algorithmic: convergence speed, stability,
cost per iteration, and final solution quality.
"""

import numpy as np

from utils.data import load_cup, train_test_split
from utils.experiments import (experiment_convergence,
                               experiment_params,
                               experiment_scaling_m,
                               experiment_a1_vs_a2)
from viz.plots import (fig1_convergence,
                       fig2_params,
                       fig3_scaling,
                       fig4_comparison,
                       print_comparison_table)

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_PATH = 'dataset/ML-CUP25-TR.csv'
SEED = 42
M_DEFAULT = 100      # hidden neurons
EPS = 1e-6     # A1: IRLS threshold
MU = 1e-3     # A2: smoothing parameter
TOL = 1e-6     # convergence tolerance (both algorithms)
MAX_ITER = 1000      # maximum iterations (both algorithms)
TEST_FRAC = 0.1      # fraction of training data held out for evaluation

# ── Load data ─────────────────────────────────────────────────────────────────

print('Loading ML-CUP 2025 data...')
Input, Y, scaler_mean, scaler_std = load_cup(TRAIN_PATH)
print(
    f'  n={Input.shape[0]} samples  p={Input.shape[1]} features  {Y.shape[1]} targets')

Input_tr, Input_te, Y_tr, Y_te = train_test_split(Input, Y,
                                                  test_size=TEST_FRAC,
                                                  seed=SEED)
print(f'  Train: {Input_tr.shape[0]}  |  Test: {Input_te.shape[0]}')
print(f'  Both A1 and A2 run 4 times — once per target')

# ── Experiment 1: Convergence ─────────────────────────────────────────────────

print(
    f'\n[1/4] Convergence experiment  (m={M_DEFAULT}, eps={EPS}, mu={MU})...')
res_conv = experiment_convergence(Input_tr, Y_tr,
                                  m=M_DEFAULT, eps=EPS, mu=MU,
                                  tol=TOL, max_iter=MAX_ITER, seed=SEED)
for t, name in enumerate(res_conv['target_names']):
    a1c = 'yes' if res_conv['a1_converged'][t] else 'no'
    a2c = 'yes' if res_conv['a2_converged'][t] else 'no'
    print(f'  {name}:  A1={res_conv["a1_n_iters"][t]} iters (conv={a1c})  '
          f'A2={res_conv["a2_n_iters"][t]} iters (conv={a2c})')
fig1_convergence(res_conv)

# ── Experiment 2: Parameter sensitivity ──────────────────────────────────────

print('\n[2/4] Parameter sensitivity  (eps for A1, mu for A2)...')
res_params = experiment_params(Input_tr, Y_tr,
                               m=M_DEFAULT, seed=SEED,
                               eps_values=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
                               mu_values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                               tol=TOL, max_iter=MAX_ITER)
print('  A1 eps values:', res_params['eps_values'])
print('  A2 mu values: ', res_params['mu_values'])
fig2_params(res_params)

# ── Experiment 3: Scaling with m ─────────────────────────────────────────────

print('\n[3/4] CPU time scaling with m...')
res_scale = experiment_scaling_m(Input_tr, Y_tr,
                                 m_values=[10, 25, 50, 100, 200],
                                 seed=SEED, eps=EPS, mu=MU,
                                 tol=TOL, max_iter=MAX_ITER, n_repeat=3)
print(f'  m values  : {res_scale["m_values"]}')
print(f'  A1 time   : {res_scale["a1_times"].round(3)} s')
print(f'  A2 time   : {res_scale["a2_times"].round(3)} s')
fig3_scaling(res_scale)

# ── Experiment 4: A1 vs A2 head-to-head ──────────────────────────────────────

print('\n[4/4] A1 vs A2 head-to-head comparison...')
res_cmp = experiment_a1_vs_a2(Input_tr, Y_tr, Input_te, Y_te,
                              m=M_DEFAULT, seed=SEED,
                              eps=EPS, mu=MU, tol=TOL, max_iter=MAX_ITER)
print_comparison_table(res_cmp)
fig4_comparison(res_cmp)

# ── Shape summary ─────────────────────────────────────────────────────────────

print(f'\nShape summary:')
print(f'  Input  : {Input_tr.shape[0]} x {Input_tr.shape[1]}')
print(f'  W1     : {M_DEFAULT} x {Input_tr.shape[1]}  (random, frozen)')
print(f'  X      : {Input_tr.shape[0]} x {M_DEFAULT}')
print(f'  w      : {M_DEFAULT} x 1  (one per target, found by A1 or A2)')
print(f'  Y_hat  : {Input_tr.shape[0]} x 4  (4 columns stacked)')

print('\nDone. Figures saved to ./figures/')
