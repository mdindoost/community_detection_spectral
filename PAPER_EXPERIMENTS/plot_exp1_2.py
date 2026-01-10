#!/usr/bin/env python3
"""
Publication-quality plotting script for Experiment 1.2

Generates two figures per dataset:
    Figure A: Modularity improvement vs retention (ΔQ_fixed and ΔQ_Leiden)
    Figure B: Modularity decomposition (ΔQ_fixed and -ΔG_observed)

Output formats: PDF (vector) and PNG (raster)

Usage:
    python plot_exp1_2.py [--datasets DATASET1,DATASET2,...] [--output-dir DIR]
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (5, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'errorbar.capsize': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Color palette (colorblind-friendly)
COLORS = {
    'fixed': '#0072B2',      # Blue
    'leiden': '#D55E00',     # Orange
    'dG': '#009E73',         # Green
    'predicted': '#CC79A7',  # Pink
}

# Default datasets
DEFAULT_DATASETS = [
    "ca-AstroPh",
    "ca-CondMat",
    "ca-GrQc",
    "ca-HepPh",
    "ca-HepTh",
    "cit-HepPh",
    "cit-HepTh",
    "email-Enron",
    "facebook-combined",
    "ego-Facebook",
    "wiki-Vote",
    "email-Eu-core",
]

SCRIPT_DIR = Path(__file__).parent
RESULTS_BASE = SCRIPT_DIR / "results" / "exp1_2_theoretical"


def load_summary(dataset: str) -> pd.DataFrame:
    """Load summary CSV for a dataset."""
    summary_file = RESULTS_BASE / f"{dataset}_summary_FIXED.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    return pd.read_csv(summary_file)


def plot_figure_a(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Figure A: Modularity improvement vs retention

    Shows ΔQ_fixed (theory-aligned) and ΔQ_Leiden (downstream performance)
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    retention = df['retention']

    # ΔQ fixed (theory-aligned mechanism)
    ax.errorbar(
        retention,
        df['modularity_fixed_change_mean'],
        yerr=df['modularity_fixed_change_std'],
        fmt='o-',
        color=COLORS['fixed'],
        label=r'$\Delta Q_{\mathrm{fixed}}$ (theory)',
        capsize=2,
        markersize=4,
    )

    # ΔQ Leiden (downstream performance)
    ax.errorbar(
        retention,
        df['modularity_leiden_change_mean'],
        yerr=df['modularity_leiden_change_std'],
        fmt='s--',
        color=COLORS['leiden'],
        label=r'$\Delta Q_{\mathrm{Leiden}}$ (pipeline)',
        capsize=2,
        markersize=4,
    )

    # Reference line at y=0
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

    ax.set_xlabel(r'Retention $\alpha$')
    ax.set_ylabel(r'Modularity change $\Delta Q$')
    ax.legend(loc='best', framealpha=0.9)

    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{dataset}_figure_A_modularity.pdf"
    png_path = output_dir / f"{dataset}_figure_A_modularity.png"

    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png')
    plt.close(fig)

    return pdf_path, png_path


def plot_figure_b(df: pd.DataFrame, dataset: str, output_dir: Path):
    """
    Figure B: Modularity decomposition

    Shows ΔQ_fixed and -ΔG_observed (note: minus sign to show positive contribution)
    Supports the identity: ΔQ = ΔF + (-ΔG)
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    retention = df['retention']

    # ΔQ fixed (total)
    ax.errorbar(
        retention,
        df['modularity_fixed_change_mean'],
        yerr=df['modularity_fixed_change_std'],
        fmt='o-',
        color=COLORS['fixed'],
        label=r'$\Delta Q_{\mathrm{fixed}}$',
        capsize=2,
        markersize=4,
    )

    # -ΔG (plot as positive contribution)
    # Note: minus sign converts penalty reduction to positive contribution
    ax.errorbar(
        retention,
        -df['dG_observed_mean'],
        yerr=df['dG_observed_std'],
        fmt='^-',
        color=COLORS['dG'],
        label=r'$-\Delta G$ (null-model relief)',
        capsize=2,
        markersize=4,
    )

    # Reference line at y=0
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

    ax.set_xlabel(r'Retention $\alpha$')
    ax.set_ylabel(r'Contribution to $\Delta Q$')
    ax.legend(loc='best', framealpha=0.9)

    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{dataset}_figure_B_decomposition.pdf"
    png_path = output_dir / f"{dataset}_figure_B_decomposition.png"

    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png')
    plt.close(fig)

    return pdf_path, png_path


def plot_combined_figure(all_data: dict, output_dir: Path):
    """
    Create a combined multi-panel figure for all datasets (optional).
    """
    n_datasets = len(all_data)
    if n_datasets == 0:
        return None

    # Determine grid layout
    n_cols = min(4, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    axes = np.atleast_2d(axes)

    for idx, (dataset, df) in enumerate(all_data.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        retention = df['retention']

        # ΔQ fixed
        ax.errorbar(
            retention,
            df['modularity_fixed_change_mean'],
            yerr=df['modularity_fixed_change_std'],
            fmt='o-',
            color=COLORS['fixed'],
            label=r'$\Delta Q_{\mathrm{fixed}}$',
            capsize=1,
            markersize=3,
            linewidth=1,
        )

        # ΔQ Leiden
        ax.errorbar(
            retention,
            df['modularity_leiden_change_mean'],
            yerr=df['modularity_leiden_change_std'],
            fmt='s--',
            color=COLORS['leiden'],
            label=r'$\Delta Q_{\mathrm{Leiden}}$',
            capsize=1,
            markersize=3,
            linewidth=1,
        )

        ax.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_title(dataset, fontsize=9)

        if row == n_rows - 1:
            ax.set_xlabel(r'$\alpha$', fontsize=9)
        if col == 0:
            ax.set_ylabel(r'$\Delta Q$', fontsize=9)

        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for idx in range(len(all_data), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    fig.subplots_adjust(top=0.93)

    pdf_path = output_dir / "all_datasets_modularity_combined.pdf"
    png_path = output_dir / "all_datasets_modularity_combined.png"

    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', bbox_inches='tight')
    plt.close(fig)

    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for Experiment 1.2"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets (default: all available)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: results/exp1_2_theoretical/figures)"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also generate combined multi-panel figure"
    )

    args = parser.parse_args()

    # Determine datasets
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        # Find all available summary files
        datasets = []
        for d in DEFAULT_DATASETS:
            summary_file = RESULTS_BASE / f"{d}_summary_FIXED.csv"
            if summary_file.exists():
                datasets.append(d)

    if not datasets:
        print("No datasets found. Run experiments first with run_exp1_2_all.py")
        return

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_BASE / "figures"

    print("="*80)
    print("EXPERIMENT 1.2: GENERATING PUBLICATION FIGURES")
    print("="*80)
    print(f"\nDatasets: {len(datasets)}")
    print(f"Output directory: {output_dir}")

    all_data = {}
    generated_files = []

    for dataset in datasets:
        print(f"\nProcessing {dataset}...")

        try:
            df = load_summary(dataset)
            all_data[dataset] = df

            # Generate Figure A
            pdf_a, png_a = plot_figure_a(df, dataset, output_dir)
            print(f"  Figure A: {pdf_a.name}")
            generated_files.extend([pdf_a, png_a])

            # Generate Figure B
            pdf_b, png_b = plot_figure_b(df, dataset, output_dir)
            print(f"  Figure B: {pdf_b.name}")
            generated_files.extend([pdf_b, png_b])

        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    # Combined figure
    if args.combined and len(all_data) > 1:
        print("\nGenerating combined figure...")
        result = plot_combined_figure(all_data, output_dir)
        if result:
            print(f"  Combined: {result[0].name}")
            generated_files.extend(result)

    print("\n" + "="*80)
    print(f"Generated {len(generated_files)} files in {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
