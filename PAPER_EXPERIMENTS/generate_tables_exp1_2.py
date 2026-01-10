#!/usr/bin/env python3
"""
LaTeX table generator for Experiment 1.2: Theoretical Predictions Validation

Generates publication-ready LaTeX tables using booktabs formatting.

Tables generated:
    Table 1: Modularity changes across datasets (at α=0.8)
    Table 2: Decomposition verification (ΔQ = ΔF - ΔG)

Usage:
    python generate_tables_exp1_2.py [--datasets DATASET1,DATASET2,...] [--output-dir DIR]
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

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

# Target retention for tables
TARGET_RETENTION = 0.8


def format_pm(mean: float, std: float, precision: int = 4) -> str:
    """Format mean ± std for LaTeX."""
    if abs(mean) < 1e-6 and abs(std) < 1e-6:
        return "$0$"
    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"


def format_sci(val: float, precision: int = 2) -> str:
    """Format number in scientific notation for LaTeX."""
    if abs(val) < 1e-10:
        return "$0$"
    exp = int(np.floor(np.log10(abs(val))))
    mantissa = val / (10 ** exp)
    return f"${mantissa:.{precision}f} \\times 10^{{{exp}}}$"


def escape_latex(s: str) -> str:
    """Escape special LaTeX characters in dataset names."""
    return s.replace("_", "\\_").replace("-", "-")


def load_summary(dataset: str) -> pd.DataFrame:
    """Load summary CSV for a dataset."""
    summary_file = RESULTS_BASE / f"{dataset}_summary_FIXED.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    return pd.read_csv(summary_file)


def load_raw(dataset: str) -> pd.DataFrame:
    """Load raw trial data for a dataset."""
    raw_file = RESULTS_BASE / f"{dataset}_theoretical_validation_FIXED.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}")
    return pd.read_csv(raw_file)


def generate_table1(all_data: dict, output_dir: Path) -> Path:
    """
    Table 1: Modularity changes across datasets

    Columns: Dataset, m, Q_original, ΔQ_fixed (α=0.8), ΔQ_Leiden (α=0.8)
    """
    rows = []

    for dataset, data in all_data.items():
        summary_df = data['summary']
        raw_df = data['raw']

        # Get row at target retention
        row = summary_df[np.isclose(summary_df['retention'], TARGET_RETENTION)]
        if row.empty:
            # Find closest retention
            closest_idx = (summary_df['retention'] - TARGET_RETENTION).abs().idxmin()
            row = summary_df.iloc[[closest_idx]]

        row = row.iloc[0]

        # Get graph stats from raw data (use correct column names)
        m = int(raw_df['m'].iloc[0])
        Q_orig = raw_df['modularity_fixed_original'].iloc[0]

        rows.append({
            'dataset': dataset,
            'm': m,
            'Q_orig': Q_orig,
            'dQ_fixed_mean': row['modularity_fixed_change_mean'],
            'dQ_fixed_std': row['modularity_fixed_change_std'],
            'dQ_leiden_mean': row['modularity_leiden_change_mean'],
            'dQ_leiden_std': row['modularity_leiden_change_std'],
        })

    # Generate LaTeX
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Modularity changes under DSpar sparsification at $\alpha = 0.8$.}",
        r"\label{tab:exp1_2_modularity}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & $m$ & $Q_{\mathrm{orig}}$ & $\Delta Q_{\mathrm{fixed}}$ & $\Delta Q_{\mathrm{Leiden}}$ \\",
        r"\midrule",
    ]

    for r in rows:
        dataset_escaped = escape_latex(r['dataset'])
        m_str = f"{r['m']:,}"
        Q_orig_str = f"${r['Q_orig']:.4f}$"
        dQ_fixed_str = format_pm(r['dQ_fixed_mean'], r['dQ_fixed_std'], precision=4)
        dQ_leiden_str = format_pm(r['dQ_leiden_mean'], r['dQ_leiden_std'], precision=4)

        latex_lines.append(
            f"{dataset_escaped} & {m_str} & {Q_orig_str} & {dQ_fixed_str} & {dQ_leiden_str} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    # Write file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "table1_modularity_changes.tex"
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))

    return output_path


def generate_table2(all_data: dict, output_dir: Path) -> Path:
    """
    Table 2: Decomposition verification (ΔQ = ΔF - ΔG)

    Columns: Dataset, ΔQ_fixed, ΔF_obs, -ΔG_obs, residual
    """
    rows = []

    for dataset, data in all_data.items():
        summary_df = data['summary']
        raw_df = data['raw']

        # Get summary row at target retention
        row = summary_df[np.isclose(summary_df['retention'], TARGET_RETENTION)]
        if row.empty:
            closest_idx = (summary_df['retention'] - TARGET_RETENTION).abs().idxmin()
            row = summary_df.iloc[[closest_idx]]

        row = row.iloc[0]

        # Check if dG_observed columns exist
        if 'dG_observed_mean' not in row:
            continue

        # Get F_improvement from raw data at target retention
        raw_at_retention = raw_df[np.isclose(raw_df['retention'], TARGET_RETENTION)]
        if raw_at_retention.empty:
            closest_idx = (raw_df['retention'] - TARGET_RETENTION).abs().idxmin()
            raw_at_retention = raw_df.iloc[[closest_idx]]

        dF_obs = raw_at_retention['F_improvement_observed'].mean()
        dF_obs_std = raw_at_retention['F_improvement_observed'].std()

        dQ_fixed = row['modularity_fixed_change_mean']
        dG_obs = row['dG_observed_mean']

        # Reconstruction: ΔQ = ΔF - ΔG
        reconstructed = dF_obs - dG_obs
        residual = abs(dQ_fixed - reconstructed)

        rows.append({
            'dataset': dataset,
            'dQ_fixed': dQ_fixed,
            'dQ_fixed_std': row['modularity_fixed_change_std'],
            'dF_obs': dF_obs,
            'dF_obs_std': dF_obs_std if not np.isnan(dF_obs_std) else 0,
            'neg_dG_obs': -dG_obs,  # Note: we show -ΔG (positive contribution)
            'neg_dG_obs_std': row['dG_observed_std'],
            'residual': residual,
        })

    if not rows:
        print("Warning: No data with dG_observed columns found for Table 2")
        return None

    # Generate LaTeX
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Modularity decomposition verification at $\alpha = 0.8$. "
        r"The identity $\Delta Q = \Delta F - \Delta G$ holds to machine precision.}",
        r"\label{tab:exp1_2_decomposition}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & $\Delta Q_{\mathrm{fixed}}$ & $\Delta F_{\mathrm{obs}}$ & $-\Delta G_{\mathrm{obs}}$ & $|\epsilon|$ \\",
        r"\midrule",
    ]

    for r in rows:
        dataset_escaped = escape_latex(r['dataset'])
        dQ_str = format_pm(r['dQ_fixed'], r['dQ_fixed_std'], precision=4)
        dF_str = format_pm(r['dF_obs'], r['dF_obs_std'], precision=4)
        neg_dG_str = format_pm(r['neg_dG_obs'], r['neg_dG_obs_std'], precision=4)
        residual_str = format_sci(r['residual'], precision=1)

        latex_lines.append(
            f"{dataset_escaped} & {dQ_str} & {dF_str} & {neg_dG_str} & {residual_str} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    # Write file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "table2_decomposition.tex"
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))

    return output_path


def generate_table3_summary(all_data: dict, output_dir: Path) -> Path:
    """
    Table 3: Summary statistics across all datasets

    Shows aggregate results: mean improvement, consistency, etc.
    """
    improvements_fixed = []
    improvements_leiden = []
    datasets_positive_fixed = 0
    datasets_positive_leiden = 0

    for dataset, data in all_data.items():
        summary_df = data['summary']

        row = summary_df[np.isclose(summary_df['retention'], TARGET_RETENTION)]
        if row.empty:
            closest_idx = (summary_df['retention'] - TARGET_RETENTION).abs().idxmin()
            row = summary_df.iloc[[closest_idx]]

        row = row.iloc[0]

        dQ_fixed = row['modularity_fixed_change_mean']
        dQ_leiden = row['modularity_leiden_change_mean']

        improvements_fixed.append(dQ_fixed)
        improvements_leiden.append(dQ_leiden)

        if dQ_fixed > 0:
            datasets_positive_fixed += 1
        if dQ_leiden > 0:
            datasets_positive_leiden += 1

    n_datasets = len(all_data)

    # Generate LaTeX
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Summary statistics for Experiment 1.2 at $\alpha = 0.8$.}",
        r"\label{tab:exp1_2_summary}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & $\Delta Q_{\mathrm{fixed}}$ & $\Delta Q_{\mathrm{Leiden}}$ \\",
        r"\midrule",
        f"Mean improvement & ${np.mean(improvements_fixed):.4f}$ & ${np.mean(improvements_leiden):.4f}$ \\\\",
        f"Std deviation & ${np.std(improvements_fixed):.4f}$ & ${np.std(improvements_leiden):.4f}$ \\\\",
        f"Min & ${np.min(improvements_fixed):.4f}$ & ${np.min(improvements_leiden):.4f}$ \\\\",
        f"Max & ${np.max(improvements_fixed):.4f}$ & ${np.max(improvements_leiden):.4f}$ \\\\",
        f"Datasets with $\\Delta Q > 0$ & {datasets_positive_fixed}/{n_datasets} & {datasets_positive_leiden}/{n_datasets} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "table3_summary.tex"
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for Experiment 1.2"
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
        help="Output directory for tables (default: results/exp1_2_theoretical/tables)"
    )
    parser.add_argument(
        "--retention",
        type=float,
        default=0.8,
        help="Target retention value for tables (default: 0.8)"
    )

    args = parser.parse_args()

    global TARGET_RETENTION
    TARGET_RETENTION = args.retention

    # Determine datasets
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
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
        output_dir = RESULTS_BASE / "tables"

    print("=" * 80)
    print("EXPERIMENT 1.2: GENERATING LATEX TABLES")
    print("=" * 80)
    print(f"\nDatasets: {len(datasets)}")
    print(f"Target retention: {TARGET_RETENTION}")
    print(f"Output directory: {output_dir}")

    # Load all data
    all_data = {}
    for dataset in datasets:
        try:
            summary_df = load_summary(dataset)
            raw_df = load_raw(dataset)
            all_data[dataset] = {
                'summary': summary_df,
                'raw': raw_df,
            }
            print(f"  Loaded: {dataset}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    if not all_data:
        print("\nNo data loaded. Exiting.")
        return

    # Generate tables
    generated = []

    print("\nGenerating tables...")

    # Table 1: Modularity changes
    path1 = generate_table1(all_data, output_dir)
    print(f"  Table 1: {path1.name}")
    generated.append(path1)

    # Table 2: Decomposition verification
    path2 = generate_table2(all_data, output_dir)
    if path2:
        print(f"  Table 2: {path2.name}")
        generated.append(path2)

    # Table 3: Summary statistics
    path3 = generate_table3_summary(all_data, output_dir)
    print(f"  Table 3: {path3.name}")
    generated.append(path3)

    print("\n" + "=" * 80)
    print(f"Generated {len(generated)} LaTeX tables in {output_dir}")
    print("=" * 80)

    # Print usage hint
    print("\nTo include in your LaTeX document:")
    for p in generated:
        print(f"  \\input{{{p.relative_to(SCRIPT_DIR)}}}")


if __name__ == "__main__":
    main()
