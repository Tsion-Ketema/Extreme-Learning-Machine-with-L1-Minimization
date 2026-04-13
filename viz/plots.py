import os

cache_dir = "/tmp/project24_mpl_cache"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(os.path.join(cache_dir, "fontconfig"), exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", cache_dir)
os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_results(
    X_raw,
    y,
    y_clean,
    Phi,
    w_irls,
    w_l2,
    w_sg,
    hist_irls,
    hist_sg,
    W1,
    activation,
    output_path="project24_results.png",
):
    x_grid = np.linspace(0.0, 1.0, 500)[:, None]
    Phi_grid = activation(x_grid @ W1.T)
    y_true_grid = np.sin(2.0 * np.pi * x_grid.ravel()) + 0.3 * np.cos(
        6.0 * np.pi * x_grid.ravel()
    )

    pred_irls = Phi_grid @ w_irls
    pred_sg = Phi_grid @ w_sg
    pred_l2 = Phi_grid @ w_l2

    fig = plt.figure(figsize=(16, 11), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        hspace=0.50,
        wspace=0.38,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.07,
    )

    ax_main = fig.add_subplot(gs[0:2, :])
    ax_conv = fig.add_subplot(gs[2, 0:2])
    ax_res = fig.add_subplot(gs[2, 2])

    dark_bg = "#0d1117"
    panel_bg = "#161b22"
    grid_col = "#21262d"
    text_col = "#e6edf3"
    muted = "#8b949e"

    c_irls = "#58a6ff"
    c_sg = "#3fb950"
    c_l2 = "#f78166"
    c_true = "#ffa657"
    c_data = "#8b949e"

    for ax in [ax_main, ax_conv, ax_res]:
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=muted, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_col)
        ax.grid(True, color=grid_col, linewidth=0.6, linestyle="--", alpha=0.7)

    x_flat = X_raw.ravel()

    residuals_irls = np.abs(Phi @ w_irls - y)
    thresh_out = np.percentile(residuals_irls, 88)
    is_out = residuals_irls > thresh_out

    ax_main.scatter(
        x_flat[~is_out],
        y[~is_out],
        s=18,
        color=c_data,
        alpha=0.65,
        zorder=2,
        label="Data (inliers)",
        linewidths=0,
    )
    ax_main.scatter(
        x_flat[is_out],
        y[is_out],
        s=30,
        color="#ff7b72",
        alpha=0.85,
        zorder=2,
        label="Outliers",
        marker="x",
        linewidths=1.2,
    )

    ax_main.plot(
        x_grid,
        y_true_grid,
        color=c_true,
        lw=1.8,
        ls="--",
        alpha=0.8,
        zorder=3,
        label="True function",
    )
    ax_main.plot(
        x_grid,
        pred_l2,
        color=c_l2,
        lw=1.6,
        alpha=0.75,
        zorder=4,
        label="L2 / OLS baseline",
    )
    ax_main.plot(
        x_grid,
        pred_irls,
        color=c_irls,
        lw=2.2,
        zorder=5,
        label="(A1) IRLS — L1",
    )
    ax_main.plot(
        x_grid,
        pred_sg,
        color=c_sg,
        lw=2.2,
        ls=(0, (5, 3)),
        zorder=5,
        label="(A2) Smoothed Gradient — L1",
    )

    ax_main.set_title(
        "ELM Regression  ·  L1 vs L2 Minimisation",
        color=text_col,
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax_main.set_xlabel("x", color=muted, fontsize=9)
    ax_main.set_ylabel("y", color=muted, fontsize=9)
    ax_main.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.25,
        facecolor=panel_bg,
        edgecolor=grid_col,
        labelcolor=text_col,
    )

    l1_irls = np.sum(np.abs(Phi @ w_irls - y))
    l1_sg = np.sum(np.abs(Phi @ w_sg - y))
    l1_l2 = np.sum(np.abs(Phi @ w_l2 - y))
    info = (
        "L1 objectives\n"
        f"IRLS : {l1_irls:.3f}\n"
        f"SG   : {l1_sg:.3f}\n"
        f"L2   : {l1_l2:.3f}"
    )
    ax_main.text(
        0.01,
        0.97,
        info,
        transform=ax_main.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        color=text_col,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#21262d",
            edgecolor=grid_col,
            alpha=0.9,
        ),
        fontfamily="monospace",
    )

    iters_irls = np.arange(1, len(hist_irls) + 1)
    iters_sg = np.arange(1, len(hist_sg) + 1)

    ax_conv.semilogy(iters_irls, hist_irls, color=c_irls, lw=2.0, label="(A1) IRLS")
    ax_conv.semilogy(
        iters_sg,
        hist_sg,
        color=c_sg,
        lw=2.0,
        label="(A2) Smoothed Gradient",
    )
    ax_conv.axhline(y=l1_irls, color=c_irls, ls=":", lw=1.0, alpha=0.5)
    ax_conv.axhline(y=l1_sg, color=c_sg, ls=":", lw=1.0, alpha=0.5)
    ax_conv.set_title(
        "Convergence of L1 Objective",
        color=text_col,
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax_conv.set_xlabel("Iteration", color=muted, fontsize=8)
    ax_conv.set_ylabel("||Xw − y||1  (log)", color=muted, fontsize=8)
    ax_conv.legend(
        fontsize=8,
        framealpha=0.2,
        facecolor=panel_bg,
        edgecolor=grid_col,
        labelcolor=text_col,
    )
    ax_conv.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    res_irls = Phi @ w_irls - y
    res_sg = Phi @ w_sg - y
    res_l2 = Phi @ w_l2 - y

    bp = ax_res.boxplot(
        [res_l2, res_irls, res_sg],
        labels=["L2", "IRLS", "SG"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=1.8),
        whiskerprops=dict(color=muted),
        capprops=dict(color=muted),
        flierprops=dict(marker="o", color=muted, alpha=0.4, markersize=3),
    )
    for patch, color in zip(bp["boxes"], [c_l2, c_irls, c_sg]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(color)

    ax_res.axhline(0, color=grid_col, lw=1.0)
    ax_res.set_title(
        "Residual Distribution",
        color=text_col,
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax_res.set_ylabel("Residual", color=muted, fontsize=8)
    ax_res.tick_params(axis="x", colors=text_col, labelsize=8)

    fig.suptitle(
        "Project 24  ·  Extreme Learning Machine  ·  L1 Minimisation",
        color=text_col,
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=dark_bg)
    plt.close(fig)
    return output_path
