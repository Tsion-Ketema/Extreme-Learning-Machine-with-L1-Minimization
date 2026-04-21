"""
viz/plots.py
------------
Figure generation for the report. All figures compare A1 (IRLS) vs A2 (smoothed gradient).

Figures saved to ./figures/:
    fig1_convergence.pdf  — L1 loss vs iteration: A1 and A2 side by side
    fig2_params.pdf       — effect of eps (A1) and mu (A2) on final loss
    fig3_scaling.pdf      — CPU time vs m for A1 and A2 (log-log)
    fig4_comparison.pdf   — final test L1 loss: A1 vs A2 per target
"""

import os
import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# one color per target, consistent across all figures
TARGET_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
TARGETS = ['TARGET_1', 'TARGET_2', 'TARGET_3', 'TARGET_4']


def fig1_convergence(results, filename='fig1_convergence.pdf'):
    """
    Semi-log plot of TRUE L1 loss vs iteration for A1 and A2.

    Left panel: A1 (IRLS) — should show fast drop then linear convergence.
    Right panel: A2 (smoothed gradient) — typically slower, smoother curve.

    A straight line in semi-log scale = linear convergence rate.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for t in range(4):
        name = results['target_names'][t]
        c = TARGET_COLORS[t]

        iters1 = np.arange(len(results['a1_losses'][t]))
        ax1.semilogy(iters1, results['a1_losses'][t], color=c, linewidth=1.5,
                     label=f'{name}  ({results["a1_n_iters"][t]} iters)')

        iters2 = np.arange(len(results['a2_losses'][t]))
        ax2.semilogy(iters2, results['a2_losses'][t], color=c, linewidth=1.5,
                     linestyle='--',
                     label=f'{name}  ({results["a2_n_iters"][t]} iters)')

    for ax, title in zip([ax1, ax2],
                         [f'A1 — IRLS  (m={results["m"]})',
                          f'A2 — Smoothed gradient  (m={results["m"]})']):
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'L1 loss  $f(w^{(k)})$')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig2_params(results, filename='fig2_params.pdf'):
    """
    Two-panel figure showing parameter sensitivity.
    Left: A1 final L1 loss vs eps.
    Right: A2 final L1 loss vs mu.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # A1: eps sensitivity
    eps_x = np.arange(len(results['eps_values']))
    eps_labels = [f'1e{int(np.log10(e))}' for e in results['eps_values']]
    for t in range(4):
        ax1.plot(eps_x, results['a1_losses'][t], marker='o',
                 color=TARGET_COLORS[t], linewidth=1.5, label=TARGETS[t])
    ax1.set_xticks(eps_x)
    ax1.set_xticklabels(eps_labels, fontsize=8)
    ax1.set_xlabel(r'Threshold $\varepsilon$  (A1)')
    ax1.set_ylabel('Final L1 loss')
    ax1.set_title(r'A1: final loss vs $\varepsilon$')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # A2: mu sensitivity
    mu_x = np.arange(len(results['mu_values']))
    mu_labels = [f'1e{int(np.log10(e))}' for e in results['mu_values']]
    for t in range(4):
        ax2.plot(mu_x, results['a2_losses'][t], marker='s',
                 color=TARGET_COLORS[t], linewidth=1.5,
                 linestyle='--', label=TARGETS[t])
    ax2.set_xticks(mu_x)
    ax2.set_xticklabels(mu_labels, fontsize=8)
    ax2.set_xlabel(r'Smoothing parameter $\mu$  (A2)')
    ax2.set_ylabel('Final L1 loss')
    ax2.set_title(r'A2: final loss vs $\mu$')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig3_scaling(results, filename='fig3_scaling.pdf'):
    """
    Log-log plot of CPU time vs m for A1 and A2.
    A1 should scale as O(nm^2) — slope 2 in log-log.
    A2 should scale as O(nm)   — slope 1 in log-log.
    """
    m_vals = np.array(results['m_values'])
    a1_t = results['a1_times']
    a2_t = results['a2_times']

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.loglog(m_vals, a1_t, 'o-', color='#1f77b4', linewidth=1.5,
              label='A1 — IRLS')
    ax.fill_between(m_vals,
                    np.maximum(a1_t - results['a1_times_std'], 1e-9),
                    a1_t + results['a1_times_std'],
                    alpha=0.15, color='#1f77b4')

    ax.loglog(m_vals, a2_t, 's--', color='#d62728', linewidth=1.5,
              label='A2 — Smoothed gradient')
    ax.fill_between(m_vals,
                    np.maximum(a2_t - results['a2_times_std'], 1e-9),
                    a2_t + results['a2_times_std'],
                    alpha=0.15, color='#d62728')

    # reference lines for theoretical scaling
    ref2 = a1_t[0] * (m_vals / m_vals[0]) ** 2
    ref1 = a2_t[0] * (m_vals / m_vals[0]) ** 1
    ax.loglog(m_vals, ref2, ':', color='gray', linewidth=1,
              label=r'slope 2  $\mathcal{O}(m^2)$')
    ax.loglog(m_vals, ref1, '-.', color='gray', linewidth=1,
              label=r'slope 1  $\mathcal{O}(m)$')

    ax.set_xlabel('Hidden neurons  $m$')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('CPU time scaling: A1 vs A2')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig4_comparison(results, filename='fig4_comparison.pdf'):
    """
    Grouped bar chart: A1 vs A2 test L1 loss per target.
    Both solve the same L1 problem — this shows which gets closer.
    """
    rows = results['rows']
    targets = [r['target'] for r in rows]
    a1_test = [r['a1_test_l1'] for r in rows]
    a2_test = [r['a2_test_l1'] for r in rows]

    x = np.arange(len(targets))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, a1_test, w, label='A1 — IRLS',
           color='#1f77b4', alpha=0.85)
    ax.bar(x + w/2, a2_test, w, label='A2 — Smoothed gradient',
           color='#d62728', alpha=0.85, hatch='///')

    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel('Test L1 loss')
    ax.set_title(f'A1 vs A2: test L1 loss  (m = {results["m"]})')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def print_comparison_table(results):
    """Print formatted A1 vs A2 comparison table to stdout."""
    rows = results['rows']
    print(f"\n  {'Target':<10} "
          f"{'A1 train':>10} {'A1 test':>10} {'A1 iters':>9} {'A1 time':>8} "
          f"{'A2 train':>10} {'A2 test':>10} {'A2 iters':>9} {'A2 time':>8}")
    print('  ' + '-' * 90)
    for r in rows:
        print(f"  {r['target']:<10} "
              f"{r['a1_train_l1']:>10.3f} {r['a1_test_l1']:>10.3f} "
              f"{r['a1_n_iter']:>9d} {r['a1_time']:>8.3f}s "
              f"{r['a2_train_l1']:>10.3f} {r['a2_test_l1']:>10.3f} "
              f"{r['a2_n_iter']:>9d} {r['a2_time']:>8.3f}s")
