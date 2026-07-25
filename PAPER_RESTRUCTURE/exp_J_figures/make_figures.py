#!/usr/bin/env python
"""
Tasks 1 & 2: render the four paper figures as VECTOR PDFs.

Task 1 (new data, cit-HepTh LCC sweep produced by run_cit_hepth_lcc_sweep.py):
  Paper_materials/figures/cit-HepTh_figure_A_modularity_no_alpha1.pdf
  Paper_materials/figures/cit-HepTh_figure_B_decomposition_no_alpha1.pdf

Task 2 (existing data, PAPER_EXPERIMENTS/results/exp1_3_lfr/, n=10000, alpha=0.8):
  Paper_materials/figures/plot1_delta_vs_mu.pdf
  Paper_materials/figures/plot3_dQ_vs_hub_ratio_n10000_r0.8.pdf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/md724/community_detection_spectral")
WORK = REPO / "PAPER_RESTRUCTURE/exp_J_figures"
FIGS = REPO / "Paper_materials/figures"
LFR = REPO / "PAPER_EXPERIMENTS/results/exp1_3_lfr"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "errorbar.capsize": 2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,   # embed TrueType, no Type-3
    "ps.fonttype": 42,
})

# Okabe-Ito colorblind-safe palette
C_FIXED = "#0072B2"   # blue
C_LEIDEN = "#D55E00"  # vermillion
C_DG = "#009E73"      # bluish green
LFR_COLORS = {"standard": "#0072B2", 1.0: "#56B4E9", 2.0: "#D55E00", 4.0: "#CC79A7"}

FIGSIZE = (3.6, 2.8)


def _finish(ax, legend_loc="best"):
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc=legend_loc, framealpha=0.9, borderpad=0.35,
              handlelength=1.8, labelspacing=0.3)


# ---------------------------------------------------------------- Task 1
def task1():
    df = pd.read_csv(WORK / "cit-HepTh_lcc_sweep.csv")
    df = df[~np.isclose(df["retention"], 1.0)]
    a = df["retention"]

    # Figure A
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(a, df["modularity_fixed_change_mean"], yerr=df["modularity_fixed_change_std"],
                fmt="o-", color=C_FIXED, capsize=2, markersize=3.5,
                label=r"$\Delta Q_{\mathrm{fixed}}$")
    ax.errorbar(a, df["modularity_leiden_change_mean"], yerr=df["modularity_leiden_change_std"],
                fmt="s--", color=C_LEIDEN, capsize=2, markersize=3.5,
                label=r"$\Delta Q_{\mathrm{Leiden}}^{(\mathrm{sp})}$")
    ax.set_xlabel(r"Nominal retention $\alpha$")
    ax.set_ylabel(r"Modularity change $\Delta Q$")
    _finish(ax)
    p = FIGS / "cit-HepTh_figure_A_modularity_no_alpha1.pdf"
    fig.savefig(p, format="pdf")
    fig.savefig(WORK / p.with_suffix(".png").name, format="png")
    plt.close(fig)
    print("wrote", p)

    # Figure B
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(a, df["modularity_fixed_change_mean"], yerr=df["modularity_fixed_change_std"],
                fmt="o-", color=C_FIXED, capsize=2, markersize=3.5,
                label=r"$\Delta Q_{\mathrm{fixed}}$")
    ax.errorbar(a, -df["dG_observed_mean"], yerr=df["dG_observed_std"],
                fmt="^-", color=C_DG, capsize=2, markersize=3.5,
                label=r"$-\Delta G_{\mathrm{obs}}$ (null-model relief)")
    ax.set_xlabel(r"Nominal retention $\alpha$")
    ax.set_ylabel(r"Contribution to $\Delta Q$")
    _finish(ax)
    p = FIGS / "cit-HepTh_figure_B_decomposition_no_alpha1.pdf"
    fig.savefig(p, format="pdf")
    fig.savefig(WORK / p.with_suffix(".png").name, format="png")
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------- Task 2
def task2(n_nodes=10000, retention=0.8):
    summary = pd.read_csv(LFR / "lfr_analysis_summary.csv")
    raw = pd.read_csv(LFR / "lfr_analysis_raw.csv")
    s = summary[(summary.n_nodes == n_nodes) & np.isclose(summary.retention, retention)]
    r = raw[(raw.n_nodes == n_nodes) & np.isclose(raw.retention, retention)]

    # --- plot 1: delta vs mu (planted partitions) ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    std = s[s.network_type == "standard_lfr"].sort_values("mu")
    ax.errorbar(std["mu"], std["delta_mean"], yerr=std["delta_std"],
                fmt="o-", color=LFR_COLORS["standard"], capsize=2, markersize=4,
                label=r"Standard LFR ($h=1$)")
    for h, mk in [(1.0, "s--"), (2.0, "^--"), (4.0, "D--")]:
        m = s[(s.network_type == "modified_lfr") & np.isclose(s.hub_strength, h)].sort_values("mu")
        if len(m) == 0:
            continue
        ax.errorbar(m["mu"], m["delta_mean"], yerr=m["delta_std"], fmt=mk,
                    color=LFR_COLORS[h], capsize=2, markersize=3.5,
                    label=rf"Hub-bridged, $h={h:g}$")
    ax.set_xlabel(r"Mixing parameter $\mu$")
    ax.set_ylabel(r"DSpar separation $\delta=\mu_{\mathrm{intra}}-\mu_{\mathrm{inter}}$")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.55 * (hi - lo))   # headroom for the legend
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=2, framealpha=0.9, borderpad=0.35,
              handlelength=1.6, labelspacing=0.25, columnspacing=1.0)
    p = FIGS / "plot1_delta_vs_mu.pdf"
    fig.savefig(p, format="pdf")
    fig.savefig(WORK / p.with_suffix(".png").name, format="png")
    plt.close(fig)
    print("wrote", p)

    # --- plot 3: dQ_fixed vs hub-bridging ratio (per-replicate scatter) ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    d = r[r.network_type == "standard_lfr"]
    ax.scatter(d["hub_bridge_ratio"], d["dQ_fixed"], c=LFR_COLORS["standard"],
               alpha=0.75, s=18, marker="o", linewidths=0, label=r"Standard LFR ($h=1$)")
    for h, mk in [(1.0, "s"), (2.0, "^"), (4.0, "D")]:
        d = r[(r.network_type == "modified_lfr") & np.isclose(r.hub_strength, h)]
        if len(d) == 0:
            continue
        ax.scatter(d["hub_bridge_ratio"], d["dQ_fixed"], c=LFR_COLORS[h],
                   alpha=0.75, s=18, marker=mk, linewidths=0,
                   label=rf"Hub-bridged, $h={h:g}$")
    rr = np.corrcoef(r["hub_bridge_ratio"], r["dQ_fixed"])[0, 1]
    ax.set_xlabel(r"Hub-bridging ratio  $E[d_ud_v\,|\,\mathrm{inter}]\,/\,E[d_ud_v\,|\,\mathrm{intra}]$",
                  fontsize=7.8)
    ax.set_ylabel(r"Modularity change $\Delta Q_{\mathrm{fixed}}$")
    ax.text(0.97, 0.06, rf"$r={rr:.3f}$", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.5))
    _finish(ax, legend_loc="upper left")
    p = FIGS / f"plot3_dQ_vs_hub_ratio_n{n_nodes}_r{retention}.pdf"
    fig.savefig(p, format="pdf")
    fig.savefig(WORK / p.with_suffix(".png").name, format="png")
    plt.close(fig)
    print("wrote", p, f"(pearson r = {rr:.4f}, N = {len(r)})")


if __name__ == "__main__":
    task1()
    task2()
