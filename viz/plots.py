"""
viz/plots.py
------------
All figures for the report. Primary metric is the RELATIVE GAP.

Figures:
    fig1_gap_convergence.pdf  — gap vs iteration (semi-log) for A1 and A2
    fig2_params.pdf           — final gap vs eps (A1) and mu (A2)
    fig3_scaling.pdf          — CPU time vs m, log-log with reference slopes
    fig4_comparison.pdf       — final gap: A1 vs A2 per target (bar chart)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
TARGETS = ['TARGET_1', 'TARGET_2', 'TARGET_3', 'TARGET_4']


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def fig1_gap_convergence(results, filename='fig1_gap_convergence.pdf'):
    """
    Semi-log plot of relative gap vs iteration for A1 (left) and A2 (right).

    A straight line on a semi-log scale confirms the linear convergence
    rate predicted by Daubechies et al. 2010, Theorem 3.1 (for A1).
    Monotone decrease for A1 confirms the MM descent guarantee.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for t in range(4):
        c = COLORS[t]
        name = results['target_names'][t]
        f_s = results['f_stars'][t]

        # A1 gaps
        gaps1 = np.array(results['a1_gaps'][t])
        # clip negative gaps to a small positive value for log scale
        gaps1 = np.maximum(gaps1, 1e-12)
        ax1.semilogy(np.arange(len(gaps1)), gaps1,
                     color=c, linewidth=1.5,
                     label=f'{name} (f*≈{f_s:.1f}, '
                     f'{results["a1_n_iters"][t]} iters)')

        # A2 gaps
        gaps2 = np.array(results['a2_gaps'][t])
        gaps2 = np.maximum(gaps2, 1e-12)
        ax2.semilogy(np.arange(len(gaps2)), gaps2,
                     color=c, linewidth=1.5, linestyle='--',
                     label=f'{name} ({results["a2_n_iters"][t]} iters)')

    for ax, title in zip([ax1, ax2],
                         [f'A1 — IRLS  (m={results["m"]})',
                          f'A2 — Smoothed gradient  (m={results["m"]})']):
        ax.set_xlabel('Iteration $k$')
        ax.set_ylabel(r'Relative gap  $(f(w^{(k)})-f^*)/|f^*|$')
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, which='both', alpha=0.3)
        ax.axhline(1e-2, color='gray', linestyle=':', linewidth=0.8,
                   label='gap = 1e-2')
        ax.axhline(1e-4, color='gray', linestyle='-.', linewidth=0.8,
                   label='gap = 1e-4')

    _save(fig, filename)


def fig2_params(results, filename='fig2_params.pdf'):
    """
    Final relative gap vs eps (A1, left) and mu (A2, right).
    Shows parameter sensitivity for both algorithms.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    eps_x = np.arange(len(results['eps_values']))
    mu_x = np.arange(len(results['mu_values']))
    eps_lb = [f'1e{int(np.log10(e))}' for e in results['eps_values']]
    mu_lb = [f'1e{int(np.log10(e))}' for e in results['mu_values']]

    for t in range(4):
        ax1.semilogy(eps_x, np.maximum(results['a1_gaps'][t], 1e-12),
                     marker='o', color=COLORS[t], linewidth=1.5,
                     label=TARGETS[t])
        ax2.semilogy(mu_x, np.maximum(results['a2_gaps'][t], 1e-12),
                     marker='s', color=COLORS[t], linewidth=1.5,
                     linestyle='--', label=TARGETS[t])

    ax1.set_xticks(eps_x)
    ax1.set_xticklabels(eps_lb, fontsize=8)
    ax1.set_xlabel(r'Threshold $\varepsilon$ (A1)')
    ax1.set_ylabel('Final relative gap')
    ax1.set_title(r'A1: final gap vs $\varepsilon$')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xticks(mu_x)
    ax2.set_xticklabels(mu_lb, fontsize=8)
    ax2.set_xlabel(r'Smoothing parameter $\mu$ (A2)')
    ax2.set_ylabel('Final relative gap')
    ax2.set_title(r'A2: final gap vs $\mu$')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    _save(fig, filename)


def fig3_scaling(results, filename='fig3_scaling.pdf'):
    """
    Log-log plot of CPU time vs m.
    A1 should follow slope-2 (O(nm^2)), A2 slope-1 (O(nm)).
    """
    m_vals = np.array(results['m_values'])
    a1_t = results['a1_times']
    a2_t = results['a2_times']

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.loglog(m_vals, a1_t, 'o-', color='#1f77b4', linewidth=1.5,
              label='A1 — IRLS')
    ax.fill_between(m_vals,
                    np.maximum(a1_t - results['a1_std'], 1e-9),
                    a1_t + results['a1_std'], alpha=0.15, color='#1f77b4')

    ax.loglog(m_vals, a2_t, 's--', color='#d62728', linewidth=1.5,
              label='A2 — Smoothed gradient')
    ax.fill_between(m_vals,
                    np.maximum(a2_t - results['a2_std'], 1e-9),
                    a2_t + results['a2_std'], alpha=0.15, color='#d62728')

    # Reference lines confirming theoretical scaling
    ref2 = a1_t[0] * (m_vals / m_vals[0]) ** 2
    ref1 = a2_t[0] * (m_vals / m_vals[0]) ** 1
    ax.loglog(m_vals, ref2, ':', color='gray', linewidth=1,
              label=r'slope 2  $\mathcal{O}(m^2)$')
    ax.loglog(m_vals, ref1, '-.', color='gray', linewidth=1,
              label=r'slope 1  $\mathcal{O}(m)$')

    ax.set_xlabel('Hidden neurons $m$')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('CPU time scaling: A1 vs A2')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    _save(fig, filename)


def fig4_comparison(results, filename='fig4_comparison.pdf'):
    """
    Bar chart of final relative gap per target for A1 and A2.
    Lower gap = better solution = shorter bar.
    """
    rows = results['rows']
    targets = [r['target'] for r in rows]
    a1_gaps = [max(r['a1_gap'], 1e-12) for r in rows]
    a2_gaps = [max(r['a2_gap'], 1e-12) for r in rows]

    x = np.arange(len(targets))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, a1_gaps, w, label='A1 — IRLS',
           color='#1f77b4', alpha=0.85)
    ax.bar(x + w/2, a2_gaps, w, label='A2 — Smoothed gradient',
           color='#d62728', alpha=0.85, hatch='///')

    ax.set_yscale('log')
    ax.axhline(1e-2, color='gray', linestyle=':', linewidth=0.8,
               label='gap = 1e-2 (acceptable)')
    ax.axhline(1e-4, color='gray', linestyle='-.', linewidth=0.8,
               label='gap = 1e-4 (good)')
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel('Final relative gap  (log scale)')
    ax.set_title(f'A1 vs A2: final relative gap  (m = {results["m"]})')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    _save(fig, filename)


def print_comparison_table(results):
    """Print the A1 vs A2 comparison table to stdout."""
    rows = results['rows']
    print(f"\n  {'Target':<10} {'f*':>10} "
          f"{'A1 gap':>10} {'A1 iters':>9} {'A1 time':>8} "
          f"{'A2 gap':>10} {'A2 iters':>9} {'A2 time':>8}")
    print('  ' + '-' * 80)
    for r in rows:
        print(f"  {r['target']:<10} {r['f_star']:>10.2f} "
              f"{r['a1_gap']:>10.2e} {r['a1_n_iter']:>9d} "
              f"{r['a1_time']:>8.3f}s "
              f"{r['a2_gap']:>10.2e} {r['a2_n_iter']:>9d} "
              f"{r['a2_time']:>8.3f}s")
