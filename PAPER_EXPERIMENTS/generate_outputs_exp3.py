#!/usr/bin/env python3
"""
Generate outputs (summary CSV, LaTeX tables, plots) from raw experiment data.

Usage:
    python generate_outputs_exp3.py
    python generate_outputs_exp3.py --input results/exp3_scalability/scalability_raw.csv
    python generate_outputs_exp3.py --alpha 0.6  # Generate tables for different alpha

This script reads the raw CSV from experiment 3 and generates:

CSVs:
  - scalability_summary.csv: Full summary with mean/std for all metrics
  - scalability_table_alpha{X}.csv: Filtered CSV for specific alpha (for LaTeX)

LaTeX Tables:
  - scalability_table_alpha{X}.tex: Main table (m, m_α, T_orig, T_Leiden, Speedup_L, ΔQ, NMI)
  - scalability_table_appendix_alpha{X}.tex: Appendix table (pipeline overhead details)

Derived Metrics:
  - speedup (pipeline): T_leiden_orig / (T_sparsify + T_leiden_sparse)
  - speedup_leiden:     T_leiden_orig / T_leiden_sparse (pure Leiden speedup)
  - sparsify_frac:      T_sparsify / T_pipeline (sparsification time fraction)

Main Plots:
  - Plot 1: scaling_sparsify_time - Sparsification time vs graph size
  - Plot 2: scaling_pipeline_time - Pipeline time vs graph size
  - Plot 3: quality_vs_alpha (per dataset) - ΔQ_leiden vs retention α
  - Plot 4: speedup_vs_quality (per dataset) - Pipeline speedup vs ΔQ_leiden
  - Plot 5: speedup_leiden_vs_quality (per dataset) - Leiden speedup vs ΔQ_leiden

Appendix Plots:
  - Plot D: scaling_leiden_time - Leiden time vs graph size (with baseline)
  - Plot E: nmi_vs_alpha (per dataset) - NMI(P0, Pα) vs retention α

Useful for:
  - Regenerating outputs from partial results (if experiment crashed)
  - Regenerating after modifying plot styles
  - Generating outputs without re-running experiments
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT = Path(__file__).parent / "results" / "exp3_scalability" / "scalability_raw.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "exp3_scalability"

# Publication plot settings (matching exp1_3_lfr_analysis.py style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Colorblind-friendly palette
COLORS = {
    'dspar': '#0072B2',           # Blue
    'uniform_random': '#D55E00',  # Orange
    'degree_sampling': '#009E73', # Green
    'spectral': '#CC79A7',        # Pink
}

MARKERS = {
    'dspar': 'o',
    'uniform_random': 's',
    'degree_sampling': '^',
    'spectral': 'D',
}

LINESTYLES = {
    'dspar': '-',
    'uniform_random': '--',
    'degree_sampling': '-.',
    'spectral': ':',
}


# =============================================================================
# OUTPUT GENERATION (copied from exp3_scalability.py)
# =============================================================================

def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics grouped by dataset, method, alpha."""

    # Calculate derived columns from existing columns
    df = df.copy()
    df['speedup_leiden'] = df['T_leiden_orig_sec'] / df['T_leiden_sparse_sec']
    df['sparsify_frac'] = df['T_sparsify_sec'] / df['T_pipeline_sec']

    agg_cols = [
        # Timing
        'T_sparsify_sec', 'T_leiden_sparse_sec', 'T_leiden_orig_sec', 'T_pipeline_sec',
        # Speedups
        'speedup', 'speedup_leiden', 'sparsify_frac',
        # Quality
        'dQ_fixed', 'dQ_leiden', 'Q_sparse_leiden',
        # Structure
        'nmi_P0_Palpha', 'n_communities_sparse',
        # Compression
        'm_sparse', 'retention_actual',
    ]

    # Filter to columns that exist
    agg_cols = [c for c in agg_cols if c in df.columns]

    summary = df.groupby(['dataset', 'method', 'alpha'])[agg_cols].agg(['mean', 'std'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    # Add baseline Q0 (single value per dataset)
    baseline_Q = df.groupby('dataset')['Q0'].first().reset_index()
    summary = summary.merge(baseline_Q, on='dataset')

    # Add graph size info (single value per dataset)
    graph_info = df.groupby('dataset').agg({
        'n_nodes': 'first',
        'm_edges': 'first'
    }).reset_index()
    summary = summary.merge(graph_info, on='dataset')

    return summary


def generate_filtered_csv(df: pd.DataFrame, output_dir: Path, alpha: float = 0.8) -> Path:
    """
    Generate filtered CSV for a specific alpha value (for easy LaTeX table creation).
    """
    df_alpha = df[np.isclose(df['alpha'], alpha)]

    if len(df_alpha) == 0:
        print(f"  [WARNING] No data for alpha={alpha}")
        return None

    # Calculate derived columns
    df_alpha = df_alpha.copy()
    df_alpha['speedup_leiden'] = df_alpha['T_leiden_orig_sec'] / df_alpha['T_leiden_sparse_sec']

    # Aggregate by dataset and method
    agg = df_alpha.groupby(['dataset', 'method']).agg({
        'n_nodes': 'first',
        'm_edges': 'first',
        'm_sparse': ['mean', 'std'],
        'T_leiden_orig_sec': 'first',
        'T_sparsify_sec': ['mean', 'std'],
        'T_leiden_sparse_sec': ['mean', 'std'],
        'T_pipeline_sec': ['mean', 'std'],
        'speedup': ['mean', 'std'],
        'speedup_leiden': ['mean', 'std'],
        'dQ_fixed': ['mean', 'std'],
        'dQ_leiden': ['mean', 'std'],
        'nmi_P0_Palpha': ['mean', 'std'],
    })
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
    agg = agg.reset_index()

    output_path = output_dir / f"scalability_table_alpha{alpha}.csv"
    agg.to_csv(output_path, index=False)

    return output_path


def generate_latex_table(df: pd.DataFrame, output_dir: Path, alpha: float = 0.8) -> Path:
    """
    Generate LaTeX table for a specific alpha value (main text table).
    Columns: Dataset, Method, m, m_α, T_leiden_orig, T_leiden_sparse, Speedup_Leiden, ΔQ_fixed, ΔQ_leiden, NMI
    """
    df_alpha = df[np.isclose(df['alpha'], alpha)]

    if len(df_alpha) == 0:
        print(f"  [WARNING] No data for alpha={alpha}")
        return None

    # Calculate speedup_leiden from existing columns
    df_alpha = df_alpha.copy()
    df_alpha['speedup_leiden'] = df_alpha['T_leiden_orig_sec'] / df_alpha['T_leiden_sparse_sec']

    # Aggregate by dataset and method
    agg = df_alpha.groupby(['dataset', 'method']).agg({
        'm_edges': 'first',
        'm_sparse': ['mean', 'std'],
        'T_leiden_orig_sec': 'first',
        'T_leiden_sparse_sec': ['mean', 'std'],
        'speedup_leiden': ['mean', 'std'],
        'dQ_fixed': ['mean', 'std'],
        'dQ_leiden': ['mean', 'std'],
        'nmi_P0_Palpha': ['mean', 'std'],
    })
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
    agg = agg.reset_index()

    def fmt_val(mean, std, decimals=3):
        if pd.isna(std) or std == 0:
            return f"${mean:.{decimals}f}$"
        return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"

    def fmt_time(mean, std=None):
        if std is None or pd.isna(std):
            if mean < 1:
                return f"${mean*1000:.0f}$ ms"
            return f"${mean:.2f}$ s"
        if mean < 1:
            return f"${mean*1000:.0f} \\pm {std*1000:.0f}$ ms"
        return f"${mean:.2f} \\pm {std:.2f}$ s"

    def fmt_int(val, with_std=None):
        if with_std is None or pd.isna(with_std):
            return f"{int(val):,}"
        return f"${int(val):,} \\pm {int(with_std):,}$"

    def escape_latex(s):
        return s.replace("_", "\\_").replace("-", "-")

    method_names = {
        'dspar': 'DSpar',
        'uniform_random': 'Uniform',
        'degree_sampling': 'Degree',
        'spectral': 'Spectral',
    }

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{Scalability results at $\\alpha = {alpha}$.}}",
        r"\label{tab:exp3_scalability}",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Dataset & Method & $m$ & $m_\alpha$ & $T_{\mathrm{orig}}$ & $T_{\mathrm{Leiden}}$ & Speedup$_{\mathrm{L}}$ & $\Delta Q_{\mathrm{fix}}$ & $\Delta Q_{\mathrm{L}}$ & NMI \\",
        r"\midrule",
    ]

    for dataset in agg['dataset'].unique():
        df_d = agg[agg['dataset'] == dataset]
        first_row = True

        for _, row in df_d.iterrows():
            if first_row:
                ds_str = escape_latex(dataset)
                m_str = f"{int(row['m_edges_first']):,}"
                t_orig = fmt_time(row['T_leiden_orig_sec_first'])
                first_row = False
            else:
                ds_str = ""
                m_str = ""
                t_orig = ""

            method_str = method_names.get(row['method'], row['method'])
            m_sparse_str = fmt_int(row['m_sparse_mean'], row['m_sparse_std'])
            t_leiden = fmt_time(row['T_leiden_sparse_sec_mean'], row['T_leiden_sparse_sec_std'])
            speedup_leiden_str = fmt_val(row['speedup_leiden_mean'], row['speedup_leiden_std'], 2)
            dQ_fixed = fmt_val(row['dQ_fixed_mean'], row['dQ_fixed_std'], 4)
            dQ_leiden = fmt_val(row['dQ_leiden_mean'], row['dQ_leiden_std'], 4)
            nmi_str = fmt_val(row['nmi_P0_Palpha_mean'], row['nmi_P0_Palpha_std'], 3)

            lines.append(
                f"{ds_str} & {method_str} & {m_str} & {m_sparse_str} & "
                f"{t_orig} & {t_leiden} & {speedup_leiden_str} & {dQ_fixed} & {dQ_leiden} & {nmi_str} \\\\"
            )

        lines.append(r"\midrule")

    # Remove last midrule
    lines[-1] = r"\bottomrule"

    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path = output_dir / f"scalability_table_alpha{alpha}.tex"
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


def generate_latex_table_appendix(df: pd.DataFrame, output_dir: Path, alpha: float = 0.8) -> Path:
    """
    Generate LaTeX table for appendix (pipeline overhead details).
    Includes: T_sparsify, T_pipeline, Speedup_pipeline
    """
    df_alpha = df[np.isclose(df['alpha'], alpha)]

    if len(df_alpha) == 0:
        print(f"  [WARNING] No data for alpha={alpha}")
        return None

    # Calculate derived columns
    df_alpha = df_alpha.copy()
    df_alpha['speedup_leiden'] = df_alpha['T_leiden_orig_sec'] / df_alpha['T_leiden_sparse_sec']
    df_alpha['sparsify_frac'] = df_alpha['T_sparsify_sec'] / df_alpha['T_pipeline_sec']

    # Aggregate by dataset and method
    agg = df_alpha.groupby(['dataset', 'method']).agg({
        'm_edges': 'first',
        'T_sparsify_sec': ['mean', 'std'],
        'T_leiden_sparse_sec': ['mean', 'std'],
        'T_pipeline_sec': ['mean', 'std'],
        'speedup': ['mean', 'std'],
        'speedup_leiden': ['mean', 'std'],
        'sparsify_frac': ['mean', 'std'],
    })
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
    agg = agg.reset_index()

    def fmt_val(mean, std, decimals=3):
        if pd.isna(std) or std == 0:
            return f"${mean:.{decimals}f}$"
        return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"

    def fmt_time(mean, std):
        if pd.isna(std):
            std = 0
        if mean < 1:
            return f"${mean*1000:.0f} \\pm {std*1000:.0f}$ ms"
        return f"${mean:.2f} \\pm {std:.2f}$ s"

    def escape_latex(s):
        return s.replace("_", "\\_").replace("-", "-")

    method_names = {
        'dspar': 'DSpar',
        'uniform_random': 'Uniform',
        'degree_sampling': 'Degree',
        'spectral': 'Spectral',
    }

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{Pipeline overhead details at $\\alpha = {alpha}$.}}",
        r"\label{tab:exp3_pipeline_overhead}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Method & $T_{\mathrm{spar}}$ & $T_{\mathrm{Leiden}}$ & $T_{\mathrm{pipe}}$ & Speedup$_{\mathrm{pipe}}$ & Speedup$_{\mathrm{L}}$ & Spar\% \\",
        r"\midrule",
    ]

    for dataset in agg['dataset'].unique():
        df_d = agg[agg['dataset'] == dataset]
        first_row = True

        for _, row in df_d.iterrows():
            if first_row:
                ds_str = escape_latex(dataset)
                first_row = False
            else:
                ds_str = ""

            method_str = method_names.get(row['method'], row['method'])
            t_spar = fmt_time(row['T_sparsify_sec_mean'], row['T_sparsify_sec_std'])
            t_leiden = fmt_time(row['T_leiden_sparse_sec_mean'], row['T_leiden_sparse_sec_std'])
            t_pipe = fmt_time(row['T_pipeline_sec_mean'], row['T_pipeline_sec_std'])
            speedup_pipe_str = fmt_val(row['speedup_mean'], row['speedup_std'], 2)
            speedup_leiden_str = fmt_val(row['speedup_leiden_mean'], row['speedup_leiden_std'], 2)
            spar_frac_str = fmt_val(row['sparsify_frac_mean'] * 100, row['sparsify_frac_std'] * 100, 1)

            lines.append(
                f"{ds_str} & {method_str} & {t_spar} & {t_leiden} & "
                f"{t_pipe} & {speedup_pipe_str} & {speedup_leiden_str} & {spar_frac_str} \\\\"
            )

        lines.append(r"\midrule")

    # Remove last midrule
    lines[-1] = r"\bottomrule"

    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path = output_dir / f"scalability_table_appendix_alpha{alpha}.tex"
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


# =============================================================================
# PLOTTING
# =============================================================================

def plot_scaling_sparsify_time(df: pd.DataFrame, output_dir: Path):
    """
    Plot 1: Sparsification runtime scaling with graph size.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Use alpha=0.8 data
    df_plot = df[np.isclose(df['alpha'], 0.8)]

    # Get methods from data
    methods_in_data = df_plot['method'].unique()

    for method in methods_in_data:
        df_m = df_plot[df_plot['method'] == method]

        if len(df_m) == 0:
            continue

        # Aggregate by dataset
        agg = df_m.groupby('dataset').agg({
            'm_edges': 'first',
            'T_sparsify_sec': ['mean', 'std']
        }).reset_index()
        agg.columns = ['dataset', 'm_edges', 'T_mean', 'T_std']
        agg = agg.sort_values('m_edges')

        ax.errorbar(
            agg['m_edges'], agg['T_mean'], yerr=agg['T_std'],
            fmt=MARKERS.get(method, 'o') + LINESTYLES.get(method, '-'),
            color=COLORS.get(method, '#666666'),
            label=method.replace('_', ' ').title(),
            capsize=3, markersize=6
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of edges $m$')
    ax.set_ylabel(r'Sparsification time $T_{\mathrm{spar}}$ (s)')
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'plot1_scaling_sparsify_time.pdf', format='pdf')
    fig.savefig(output_dir / 'plot1_scaling_sparsify_time.png', format='png')
    plt.close(fig)


def plot_scaling_pipeline_time(df: pd.DataFrame, output_dir: Path):
    """
    Plot 2: End-to-end pipeline time scaling.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Use alpha=0.8 data
    df_plot = df[np.isclose(df['alpha'], 0.8)]

    # Get methods from data
    methods_in_data = df_plot['method'].unique()

    for method in methods_in_data:
        df_m = df_plot[df_plot['method'] == method]

        if len(df_m) == 0:
            continue

        agg = df_m.groupby('dataset').agg({
            'm_edges': 'first',
            'T_pipeline_sec': ['mean', 'std']
        }).reset_index()
        agg.columns = ['dataset', 'm_edges', 'T_mean', 'T_std']
        agg = agg.sort_values('m_edges')

        ax.errorbar(
            agg['m_edges'], agg['T_mean'], yerr=agg['T_std'],
            fmt=MARKERS.get(method, 'o') + LINESTYLES.get(method, '-'),
            color=COLORS.get(method, '#666666'),
            label=method.replace('_', ' ').title(),
            capsize=3, markersize=6
        )

    # Also plot baseline Leiden time
    baseline = df_plot.groupby('dataset').agg({
        'm_edges': 'first',
        'T_leiden_orig_sec': 'first'
    }).reset_index()
    baseline = baseline.sort_values('m_edges')

    ax.plot(baseline['m_edges'], baseline['T_leiden_orig_sec'],
            'k--', linewidth=1.5, label='Leiden (original)', alpha=0.7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of edges $m$')
    ax.set_ylabel(r'Pipeline time $T_{\mathrm{pipe}}$ (s)')
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'plot2_scaling_pipeline_time.pdf', format='pdf')
    fig.savefig(output_dir / 'plot2_scaling_pipeline_time.png', format='png')
    plt.close(fig)


def plot_quality_vs_alpha(df: pd.DataFrame, output_dir: Path):
    """
    Plot 3: Quality (dQ_leiden) vs retention alpha, per dataset.
    """
    for dataset in df['dataset'].unique():
        df_d = df[df['dataset'] == dataset]

        fig, ax = plt.subplots(figsize=(6, 4.5))

        # Get methods from data
        methods_in_data = df_d['method'].unique()

        for method in methods_in_data:
            df_m = df_d[df_d['method'] == method]

            if len(df_m) == 0:
                continue

            agg = df_m.groupby('alpha').agg({
                'dQ_leiden': ['mean', 'std']
            }).reset_index()
            agg.columns = ['alpha', 'dQ_mean', 'dQ_std']

            ax.errorbar(
                agg['alpha'], agg['dQ_mean'], yerr=agg['dQ_std'],
                fmt=MARKERS.get(method, 'o') + LINESTYLES.get(method, '-'),
                color=COLORS.get(method, '#666666'),
                label=method.replace('_', ' ').title(),
                capsize=3, markersize=6
            )

        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_xlabel(r'Retention $\alpha$')
        ax.set_ylabel(r'$\Delta Q_{\mathrm{Leiden}}$')
        ax.legend(loc='best', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        safe_name = dataset.replace('-', '_').replace('.', '_')
        fig.savefig(output_dir / f'plot3_quality_vs_alpha_{safe_name}.pdf', format='pdf')
        fig.savefig(output_dir / f'plot3_quality_vs_alpha_{safe_name}.png', format='png')
        plt.close(fig)


def plot_speedup_vs_quality(df: pd.DataFrame, output_dir: Path):
    """
    Plot 4: Speedup vs quality tradeoff, per dataset.
    """
    for dataset in df['dataset'].unique():
        df_d = df[df['dataset'] == dataset]

        fig, ax = plt.subplots(figsize=(6, 4.5))

        # Get methods from data
        methods_in_data = df_d['method'].unique()

        for method in methods_in_data:
            df_m = df_d[df_d['method'] == method]

            if len(df_m) == 0:
                continue

            agg = df_m.groupby('alpha').agg({
                'speedup': 'mean',
                'dQ_leiden': 'mean'
            }).reset_index()

            # Plot as connected points (each point is an alpha level)
            ax.plot(
                agg['speedup'], agg['dQ_leiden'],
                marker=MARKERS.get(method, 'o'), linestyle=LINESTYLES.get(method, '-'),
                color=COLORS.get(method, '#666666'),
                label=method.replace('_', ' ').title(),
                markersize=5
            )

            # Annotate alpha values on DSpar line
            if method == 'dspar':
                for _, row in agg.iterrows():
                    if row['alpha'] < 1.0:
                        ax.annotate(
                            f"$\\alpha$={row['alpha']:.1f}",
                            (row['speedup'], row['dQ_leiden']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, alpha=0.7
                        )

        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.axvline(1, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Speedup')
        ax.set_ylabel(r'$\Delta Q_{\mathrm{Leiden}}$')
        ax.legend(loc='best', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        safe_name = dataset.replace('-', '_').replace('.', '_')
        fig.savefig(output_dir / f'plot4_speedup_vs_quality_{safe_name}.pdf', format='pdf')
        fig.savefig(output_dir / f'plot4_speedup_vs_quality_{safe_name}.png', format='png')
        plt.close(fig)


def plot_speedup_leiden_vs_quality(df: pd.DataFrame, output_dir: Path):
    """
    Plot 5: Speedup_leiden (T_orig/T_leiden_sparse) vs quality tradeoff, per dataset.
    This shows the pure Leiden speedup ignoring sparsification cost.
    """
    # Calculate speedup_leiden from existing columns
    df = df.copy()
    df['speedup_leiden'] = df['T_leiden_orig_sec'] / df['T_leiden_sparse_sec']

    for dataset in df['dataset'].unique():
        df_d = df[df['dataset'] == dataset]

        fig, ax = plt.subplots(figsize=(6, 4.5))

        # Get methods from data
        methods_in_data = df_d['method'].unique()

        for method in methods_in_data:
            df_m = df_d[df_d['method'] == method]

            if len(df_m) == 0:
                continue

            agg = df_m.groupby('alpha').agg({
                'speedup_leiden': 'mean',
                'dQ_leiden': 'mean'
            }).reset_index()

            # Plot as connected points (each point is an alpha level)
            ax.plot(
                agg['speedup_leiden'], agg['dQ_leiden'],
                marker=MARKERS.get(method, 'o'), linestyle=LINESTYLES.get(method, '-'),
                color=COLORS.get(method, '#666666'),
                label=method.replace('_', ' ').title(),
                markersize=5
            )

            # Annotate alpha values on DSpar line
            if method == 'dspar':
                for _, row in agg.iterrows():
                    if row['alpha'] < 1.0:
                        ax.annotate(
                            f"$\\alpha$={row['alpha']:.1f}",
                            (row['speedup_leiden'], row['dQ_leiden']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, alpha=0.7
                        )

        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.axvline(1, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_xlabel(r'Speedup$_{\mathrm{Leiden}}$ ($T_{\mathrm{orig}} / T_{\mathrm{Leiden,sparse}}$)')
        ax.set_ylabel(r'$\Delta Q_{\mathrm{Leiden}}$')
        ax.legend(loc='best', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        safe_name = dataset.replace('-', '_').replace('.', '_')
        fig.savefig(output_dir / f'plot5_speedup_leiden_vs_quality_{safe_name}.pdf', format='pdf')
        fig.savefig(output_dir / f'plot5_speedup_leiden_vs_quality_{safe_name}.png', format='png')
        plt.close(fig)


def plot_scaling_leiden_time(df: pd.DataFrame, output_dir: Path):
    """
    Plot D (appendix): Leiden time scaling with graph size.
    Shows T_leiden_sparse vs m_edges with baseline T_leiden_orig reference.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Use alpha=0.8 data
    df_plot = df[np.isclose(df['alpha'], 0.8)]

    # Get methods from data
    methods_in_data = df_plot['method'].unique()

    for method in methods_in_data:
        df_m = df_plot[df_plot['method'] == method]

        if len(df_m) == 0:
            continue

        agg = df_m.groupby('dataset').agg({
            'm_edges': 'first',
            'T_leiden_sparse_sec': ['mean', 'std']
        }).reset_index()
        agg.columns = ['dataset', 'm_edges', 'T_mean', 'T_std']
        agg = agg.sort_values('m_edges')

        ax.errorbar(
            agg['m_edges'], agg['T_mean'], yerr=agg['T_std'],
            fmt=MARKERS.get(method, 'o') + LINESTYLES.get(method, '-'),
            color=COLORS.get(method, '#666666'),
            label=method.replace('_', ' ').title(),
            capsize=3, markersize=6
        )

    # Also plot baseline Leiden time on original graph
    baseline = df_plot.groupby('dataset').agg({
        'm_edges': 'first',
        'T_leiden_orig_sec': 'first'
    }).reset_index()
    baseline = baseline.sort_values('m_edges')

    ax.plot(baseline['m_edges'], baseline['T_leiden_orig_sec'],
            'k--', linewidth=1.5, label='Leiden (original)', alpha=0.7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of edges $m$')
    ax.set_ylabel(r'Leiden time $T_{\mathrm{Leiden}}$ (s)')
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'plot_scaling_leiden_time.pdf', format='pdf')
    fig.savefig(output_dir / 'plot_scaling_leiden_time.png', format='png')
    plt.close(fig)


def plot_nmi_vs_alpha(df: pd.DataFrame, output_dir: Path):
    """
    Plot E (appendix): NMI(P0, Pα) vs retention alpha, per dataset.
    Shows how partition stability changes with sparsification.
    """
    for dataset in df['dataset'].unique():
        df_d = df[df['dataset'] == dataset]

        fig, ax = plt.subplots(figsize=(6, 4.5))

        # Get methods from data
        methods_in_data = df_d['method'].unique()

        for method in methods_in_data:
            df_m = df_d[df_d['method'] == method]

            if len(df_m) == 0:
                continue

            agg = df_m.groupby('alpha').agg({
                'nmi_P0_Palpha': ['mean', 'std']
            }).reset_index()
            agg.columns = ['alpha', 'nmi_mean', 'nmi_std']

            ax.errorbar(
                agg['alpha'], agg['nmi_mean'], yerr=agg['nmi_std'],
                fmt=MARKERS.get(method, 'o') + LINESTYLES.get(method, '-'),
                color=COLORS.get(method, '#666666'),
                label=method.replace('_', ' ').title(),
                capsize=3, markersize=6
            )

        ax.set_xlabel(r'Retention $\alpha$')
        ax.set_ylabel(r'NMI$(P_0, P_\alpha)$')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        safe_name = dataset.replace('-', '_').replace('.', '_')
        fig.savefig(output_dir / f'plot_nmi_vs_alpha_{safe_name}.pdf', format='pdf')
        fig.savefig(output_dir / f'plot_nmi_vs_alpha_{safe_name}.png', format='png')
        plt.close(fig)


def generate_all_plots(df: pd.DataFrame, output_dir: Path):
    """Generate all publication plots."""
    print("\nGenerating plots...")

    # Main plots
    plot_scaling_sparsify_time(df, output_dir)
    print("  Plot 1: scaling_sparsify_time")

    plot_scaling_pipeline_time(df, output_dir)
    print("  Plot 2: scaling_pipeline_time")

    plot_quality_vs_alpha(df, output_dir)
    print(f"  Plot 3: quality_vs_alpha (per dataset)")

    plot_speedup_vs_quality(df, output_dir)
    print(f"  Plot 4: speedup_vs_quality (per dataset)")

    plot_speedup_leiden_vs_quality(df, output_dir)
    print(f"  Plot 5: speedup_leiden_vs_quality (per dataset)")

    # Appendix plots
    plot_scaling_leiden_time(df, output_dir)
    print("  Plot D: scaling_leiden_time (appendix)")

    plot_nmi_vs_alpha(df, output_dir)
    print(f"  Plot E: nmi_vs_alpha (per dataset, appendix)")


def print_key_results(df: pd.DataFrame, alpha: float = 0.8):
    """Print key results table for a specific alpha."""
    print(f"\n{'='*120}")
    print(f"KEY RESULTS AT alpha = {alpha}")
    print(f"{'='*120}")

    df_alpha = df[np.isclose(df['alpha'], alpha)].copy()

    if len(df_alpha) == 0:
        print("No data available.")
        return

    # Calculate speedup_leiden from existing columns
    df_alpha['speedup_leiden'] = df_alpha['T_leiden_orig_sec'] / df_alpha['T_leiden_sparse_sec']

    print(f"\n{'Dataset':<15} {'Method':<18} {'T_spar(s)':<12} {'T_pipe(s)':<12} "
          f"{'Speedup_pipe':<14} {'Speedup_Leiden':<16} {'dQ_fixed':<12} {'dQ_Leiden':<12}")
    print("-" * 120)

    # Get methods from data
    methods_in_data = df_alpha['method'].unique()

    for dataset in df_alpha['dataset'].unique():
        df_d = df_alpha[df_alpha['dataset'] == dataset]
        T_orig = df_d['T_leiden_orig_sec'].iloc[0]

        for method in methods_in_data:
            df_m = df_d[df_d['method'] == method]

            if len(df_m) == 0:
                continue

            T_spar = df_m['T_sparsify_sec'].mean()
            T_pipe = df_m['T_pipeline_sec'].mean()
            speedup = df_m['speedup'].mean()
            speedup_leiden = df_m['speedup_leiden'].mean()
            dQ_fixed = df_m['dQ_fixed'].mean()
            dQ_leiden = df_m['dQ_leiden'].mean()

            print(f"{dataset:<15} {method:<18} {T_spar:<12.4f} {T_pipe:<12.4f} "
                  f"{speedup:<14.2f} {speedup_leiden:<16.2f} {dQ_fixed:<+12.6f} {dQ_leiden:<+12.6f}")

        print(f"{'':<15} {'(baseline)':<18} {'':<12} {T_orig:<12.4f} "
              f"{'1.00':<14} {'1.00':<16} {'0.0':<12} {'0.0':<12}")
        print("-" * 120)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate outputs from raw experiment 3 data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Path to raw CSV file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Alpha value for LaTeX table (default: 0.8)"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--no_table",
        action="store_true",
        help="Skip LaTeX table generation"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"

    # Check input file exists
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print(f"\nRun exp3_scalability.py first to generate raw data.")
        sys.exit(1)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATE OUTPUTS FROM EXPERIMENT 3 RAW DATA")
    print("=" * 80)

    print(f"\nInput file: {input_file}")
    print(f"Output directory: {output_dir}")

    # Load data
    print(f"\nLoading data...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} rows")
    print(f"  Datasets: {df['dataset'].unique().tolist()}")
    print(f"  Methods: {df['method'].unique().tolist()}")
    print(f"  Alphas: {sorted(df['alpha'].unique().tolist())}")

    # Generate summary
    print(f"\nGenerating summary CSV...")
    summary = generate_summary(df)
    summary_file = output_dir / "scalability_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"  Saved: {summary_file}")

    # Generate filtered CSV for specific alpha
    print(f"\nGenerating filtered CSV (alpha={args.alpha})...")
    filtered_csv = generate_filtered_csv(df, output_dir, alpha=args.alpha)
    if filtered_csv:
        print(f"  Saved: {filtered_csv}")

    # Generate LaTeX tables
    if not args.no_table:
        print(f"\nGenerating LaTeX tables (alpha={args.alpha})...")

        # Main table
        latex_file = generate_latex_table(df, output_dir, alpha=args.alpha)
        if latex_file:
            print(f"  Saved: {latex_file}")

        # Appendix table (pipeline overhead)
        latex_appendix = generate_latex_table_appendix(df, output_dir, alpha=args.alpha)
        if latex_appendix:
            print(f"  Saved: {latex_appendix}")

    # Generate plots
    if not args.no_plots:
        generate_all_plots(df, figures_dir)
        print(f"\nPlots saved to: {figures_dir}/")

    # Print key results
    print_key_results(df, alpha=args.alpha)

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
