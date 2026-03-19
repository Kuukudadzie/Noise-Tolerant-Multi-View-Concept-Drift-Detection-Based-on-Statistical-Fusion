"""
plot_results.py
===============
Reads results.pkl produced by run_experiment.py and saves six
publication-quality figures to the current directory.

Output files
------------
fig1_f1_per_dataset.png    — F1 vs noise, one panel per dataset (2×2)
fig2_overall_metrics.png   — F1 / EDDR / TPR / Precision vs noise (overall)
fig3_fp_per_dataset.png    — False positives vs noise, per dataset (2×2)
fig4_summary_bars.png      — TP/FP/FN + F1 + EDDR summary bars
fig5_f1_heatmap.png        — F1 heatmap: dataset × noise level
fig6_architecture.png      — MV-Fuse two-stage architecture diagram

Usage
-----
    python plot_results.py                   # reads results.pkl
    python plot_results.py my_results.pkl   # custom path

Requirements
------------
    numpy >= 1.21
    matplotlib >= 3.5
"""

import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ==============================================================================
# Style constants
# ==============================================================================

METHOD_NAMES = ["MV-Fuse", "DDM", "EDDM", "HDDM-A", "HDDM-W", "ADWIN"]
DATASETS     = ["SEA", "SINE", "HYPERPLANE", "AGRAWAL"]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]
NOISE_PCT    = [0, 5, 10, 15, 20]
KEYS         = ["TP", "FP", "FN", "TPR", "Precision", "F1", "EDDR", "D1", "D2"]

COLORS = {
    "MV-Fuse": "#1a6fad",
    "DDM":     "#e03e3e",
    "EDDM":    "#e07b00",
    "HDDM-A":  "#2ea84c",
    "HDDM-W":  "#8e44ad",
    "ADWIN":   "#7f8c8d",
}
MARKERS = {
    "MV-Fuse": "o", "DDM": "s", "EDDM": "^",
    "HDDM-A": "D", "HDDM-W": "v", "ADWIN": "p",
}
LINESTYLES = {
    "MV-Fuse": "-",  "DDM": "--", "EDDM": "--",
    "HDDM-A":  "--", "HDDM-W": "--", "ADWIN": "--",
}
LINEWIDTHS = {
    "MV-Fuse": 2.8, "DDM": 1.6, "EDDM": 1.6,
    "HDDM-A":  1.6, "HDDM-W": 1.6, "ADWIN": 1.6,
}


# ==============================================================================
# Helpers
# ==============================================================================

def _avg(all_results, dname, noise, method, metric):
    return np.mean([r[metric] for r in all_results[dname][noise][method]])


def _overall_avg(all_results, method, metric):
    return np.mean([
        r[metric]
        for d in DATASETS
        for n in NOISE_LEVELS
        for r in all_results[d][n][method]
    ])


def _legend_handles():
    return [mpatches.Patch(color=COLORS[nm], label=nm) for nm in METHOD_NAMES]


def _draw_lines(ax, all_results, metric, dname=None):
    """Plot one metric line per method. Averages over datasets if dname=None."""
    for nm in METHOD_NAMES:
        if dname:
            vals = [_avg(all_results, dname, n, nm, metric) for n in NOISE_LEVELS]
        else:
            vals = [
                np.mean([_avg(all_results, d, n, nm, metric) for d in DATASETS])
                for n in NOISE_LEVELS
            ]
        ax.plot(
            NOISE_PCT, vals,
            color=COLORS[nm], marker=MARKERS[nm],
            linestyle=LINESTYLES[nm], linewidth=LINEWIDTHS[nm],
            markersize=7 if nm == "MV-Fuse" else 5,
            label=nm, zorder=3 if nm == "MV-Fuse" else 2,
        )
    ax.set_xlabel("Noise level (%)", fontsize=10)
    ax.set_xticks(NOISE_PCT)
    ax.grid(True, alpha=0.3, linestyle=":")


def _add_legend(fig):
    fig.legend(
        handles=_legend_handles(),
        loc="lower center", ncol=6,
        fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )


# ==============================================================================
# Figure 1 — F1 per dataset (2×2)
# ==============================================================================

def plot1_f1_per_dataset(all_results, outpath="fig1_f1_per_dataset.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    fig.suptitle("F1 Score vs Noise Level — Per Dataset",
                 fontsize=13, fontweight="bold")
    for ax, dname in zip(axes.flat, DATASETS):
        _draw_lines(ax, all_results, "F1", dname)
        ax.set_title(dname, fontsize=11, fontweight="bold")
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_ylim(-0.05, 1.10)
    _add_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ==============================================================================
# Figure 2 — Overall metrics vs noise (2×2)
# ==============================================================================

def plot2_overall_metrics(all_results, outpath="fig2_overall_metrics.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    fig.suptitle("Overall Metrics vs Noise Level (All Datasets Average)",
                 fontsize=13, fontweight="bold")
    panels = [
        ("F1",        "F1 Score (↑)"),
        ("EDDR",      "EDDR (↑)"),
        ("TPR",       "TPR / Recall (↑)"),
        ("Precision", "Precision (↑)"),
    ]
    for ax, (metric, ylabel) in zip(axes.flat, panels):
        _draw_lines(ax, all_results, metric)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(-0.05, 1.10)
    _add_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ==============================================================================
# Figure 3 — False positives per dataset (2×2)
# ==============================================================================

def plot3_fp_per_dataset(all_results, outpath="fig3_fp_per_dataset.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    fig.suptitle("False Positives vs Noise Level — Per Dataset",
                 fontsize=13, fontweight="bold")
    for ax, dname in zip(axes.flat, DATASETS):
        _draw_lines(ax, all_results, "FP", dname)
        ax.set_title(dname, fontsize=11, fontweight="bold")
        ax.set_ylabel("False Positives", fontsize=10)
    _add_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ==============================================================================
# Figure 4 — Summary bars (TP/FP/FN + F1 + EDDR)
# ==============================================================================

def plot4_summary_bars(all_results, outpath="fig4_summary_bars.png"):
    fig = plt.figure(figsize=(14, 5), dpi=150)
    fig.suptitle(
        "Overall Performance Summary (All Datasets × All Noise Levels)",
        fontsize=12, fontweight="bold",
    )
    gs = GridSpec(1, 3, figure=fig, wspace=0.38)
    bar_colors = [COLORS[nm] for nm in METHOD_NAMES]
    x  = np.arange(len(METHOD_NAMES))
    w  = 0.25
    oa = {nm: {k: _overall_avg(all_results, nm, k) for k in KEYS}
          for nm in METHOD_NAMES}

    # TP / FP / FN
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(x - w, [oa[nm]["TP"] for nm in METHOD_NAMES], w,
            label="TP", color="#2ea84c", alpha=0.85)
    ax1.bar(x,     [oa[nm]["FP"] for nm in METHOD_NAMES], w,
            label="FP", color="#e03e3e", alpha=0.85)
    ax1.bar(x + w, [oa[nm]["FN"] for nm in METHOD_NAMES], w,
            label="FN", color="#e07b00", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(METHOD_NAMES, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Count")
    ax1.set_title("TP / FP / FN", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.axvspan(-0.5, 0.5, alpha=0.07, color="blue")

    # F1
    ax2 = fig.add_subplot(gs[1])
    f1_vals = [oa[nm]["F1"] for nm in METHOD_NAMES]
    bars = ax2.bar(METHOD_NAMES, f1_vals,
                   color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, f1_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Overall F1 (↑)", fontweight="bold")
    ax2.set_xticklabels(METHOD_NAMES, rotation=35, ha="right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # EDDR
    ax3 = fig.add_subplot(gs[2])
    eddr_vals = [oa[nm]["EDDR"] for nm in METHOD_NAMES]
    bars3 = ax3.bar(METHOD_NAMES, eddr_vals,
                    color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars3, eddr_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("EDDR")
    ax3.set_title("Overall EDDR (↑)", fontweight="bold")
    ax3.set_xticklabels(METHOD_NAMES, rotation=35, ha="right", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ==============================================================================
# Figure 5 — F1 heatmap
# ==============================================================================

def plot5_f1_heatmap(all_results, outpath="fig5_f1_heatmap.png"):
    fig, axes = plt.subplots(1, len(METHOD_NAMES), figsize=(16, 3.5), dpi=150)
    fig.suptitle("F1 Score Heatmap — Dataset × Noise Level",
                 fontsize=12, fontweight="bold")
    noise_labels = [f"{int(n*100)}%" for n in NOISE_LEVELS]
    im = None
    for ax, nm in zip(axes, METHOD_NAMES):
        mat = np.array([
            [_avg(all_results, d, n, nm, "F1") for n in NOISE_LEVELS]
            for d in DATASETS
        ])
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(NOISE_LEVELS)))
        ax.set_xticklabels(noise_labels, fontsize=7)
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels(DATASETS, fontsize=8)
        ax.set_title(nm, fontsize=9, fontweight="bold",
                     color="#1a6fad" if nm == "MV-Fuse" else "black")
        for i in range(len(DATASETS)):
            for j in range(len(NOISE_LEVELS)):
                ax.text(j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if mat[i, j] > 0.35 else "white")
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="F1")
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ==============================================================================
# Figure 6 — MV-Fuse architecture diagram
# ==============================================================================

def plot6_architecture(outpath="fig6_architecture.png"):
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, fc, ec, text, fontsize=9, bold=False, tc="white"):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.5, zorder=2))
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal",
                color=tc, zorder=3, multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.5))

    # Stream input
    box(0.2, 1.8, 1.4, 1.4, "#444444", "#222222",
        "Data\nStream\n(chunk Bₜ)", fontsize=8, bold=True)
    arrow(1.6, 2.5, 2.2, 2.5)

    # Stage I
    box(2.2, 1.4, 2.2, 2.2, "#185FA5", "#0C447C",
        "Stage I\nPrequential\nMonitoring\n(pᵢ, sᵢ, λ)", fontsize=8, bold=True)
    ax.text(3.3, 4.0, "Candidate?", ha="center", fontsize=8,
            color="#185FA5", style="italic")
    arrow(4.4, 2.5, 5.0, 2.5)
    ax.text(4.65, 2.65, "Yes", fontsize=7.5, color="#2ea84c", fontweight="bold")
    arrow(3.3, 1.4, 3.3, 0.7)
    ax.text(3.35, 0.5, "No → continue", fontsize=7.5, color="#888888")

    # Stage II bounding box
    box(5.0, 0.3, 3.8, 4.4, "#E6F1FB", "#185FA5", "")
    ax.text(6.9, 4.45, "Stage II  —  Multi-View Validation",
            ha="center", fontsize=8.5, fontweight="bold", color="#185FA5")

    # Three views
    for yv, label in [(3.2, "View 1\nMargins\n(KS → p₁)"),
                      (2.1, "View 2\nEntropy\n(KS → p₂)"),
                      (1.0, "View 3\nLabels\n(χ² → p₃)")]:
        box(5.15, yv, 1.1, 0.85, "#1a6fad", "#0C447C", label, fontsize=7)

    # Brown's fusion
    box(6.6, 1.8, 1.2, 1.5, "#0F6E56", "#085041",
        "Brown's\nMethod\np_fuse", fontsize=8, bold=True)
    for yv in [3.62, 2.52, 1.42]:
        arrow(6.25, yv, 6.6, 2.55 if yv != 1.42 else 2.0)

    # Decision
    box(8.1, 1.7, 1.3, 1.6, "#993C1D", "#712B13",
        "p_fuse < α\n?\n(τₚ checks)", fontsize=8, bold=True)
    arrow(7.8, 2.55, 8.1, 2.55)

    # Drift confirmed
    box(9.6, 2.0, 1.2, 1.1, "#2ea84c", "#1a5c2a",
        "Drift\nConfirmed\n✓", fontsize=8, bold=True)
    arrow(9.4, 2.55, 9.6, 2.55)
    ax.text(9.98, 3.2, "Yes", fontsize=7.5, color="#2ea84c",
            fontweight="bold", ha="center")
    arrow(8.75, 1.7, 8.75, 0.9)
    ax.text(8.4, 0.7, "No → cooldown", fontsize=7.5, color="#888888")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else "results.pkl"
    print(f"Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        all_results = pickle.load(f)

    print("Generating figures...")
    plot1_f1_per_dataset(all_results)
    plot2_overall_metrics(all_results)
    plot3_fp_per_dataset(all_results)
    plot4_summary_bars(all_results)
    plot5_f1_heatmap(all_results)
    plot6_architecture()
    print("\nDone. 6 figures saved.")
