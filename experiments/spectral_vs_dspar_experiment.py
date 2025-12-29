#!/usr/bin/env python3
"""
Spectral Sparsification vs DSpar Comparison Experiment

Compares Spielman-Srivastava spectral sparsification (using effective resistance)
with DSpar (degree-based approximation) to verify they produce similar Ratio values.

Hypothesis: Both methods should show Ratio > 1 (inter-community edges removed faster)
because effective resistance is bounded by degree-based quantities.

Usage:
    python experiments/spectral_vs_dspar_experiment.py
    python experiments/spectral_vs_dspar_experiment.py --datasets cit-HepTh cit-HepPh
    python experiments/spectral_vs_dspar_experiment.py --epsilon 0.5 1.0 2.0
"""

import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import (
    PROJECT_ROOT,
    load_snap_dataset,
    spectral_sparsify,
    get_dataset_dir,
)

# Try imports
try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False
    print("Warning: leidenalg not available")

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Configuration
# =============================================================================

PARAMETERS = {
    # Epsilon values to test (controls sparsification level)
    # Lower epsilon = more edges kept = denser graph
    # Higher epsilon = fewer edges kept = sparser graph
    'epsilon_values': [0.5, 1.0, 1.5, 2.0, 2.5],

    # Additional epsilon values for larger datasets (more aggressive sparsification)
    'epsilon_values_extended': [3.0, 4.0, 5.0, 6.0, 7.0],
    'datasets_extended': ['com-DBLP', 'com-Youtube'],

    # Datasets (ordered smallest to largest)
    'datasets': ['cit-HepTh', 'cit-HepPh', 'com-DBLP', 'com-Youtube'],

    # Leiden parameters (same as DSpar experiments)
    'leiden_resolution': 1.0,

    # Random seed
    'seed': 42,
}

# DSpar results from previous experiments (for comparison)
DSPAR_RESULTS = {
    'cit-HepTh': {
        'keep_25%': {'ratio': 1.0565, 'keep_ratio': 0.25},
        'keep_50%': {'ratio': 1.0924, 'keep_ratio': 0.50},
        'keep_75%': {'ratio': 1.1252, 'keep_ratio': 0.75},
    },
    'cit-HepPh': {
        'keep_25%': {'ratio': 1.0190, 'keep_ratio': 0.25},
        'keep_50%': {'ratio': 1.0291, 'keep_ratio': 0.50},
        'keep_75%': {'ratio': 1.0355, 'keep_ratio': 0.75},
    },
    'com-DBLP': {
        'keep_25%': {'ratio': 1.1134, 'keep_ratio': 0.25},
        'keep_50%': {'ratio': 1.2111, 'keep_ratio': 0.50},
        'keep_75%': {'ratio': 1.2961, 'keep_ratio': 0.75},
    },
    'com-Youtube': {
        'keep_25%': {'ratio': 1.1465, 'keep_ratio': 0.25},
        'keep_50%': {'ratio': 1.2462, 'keep_ratio': 0.50},
        'keep_75%': {'ratio': 1.3091, 'keep_ratio': 0.75},
    },
}

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results" / "spectral_vs_dspar"


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_leiden_nx(G, resolution=1.0, seed=42):
    """Run Leiden community detection on NetworkX graph."""
    if not HAS_LEIDEN:
        from networkx.algorithms.community import louvain_communities
        partition = louvain_communities(G, resolution=resolution, seed=seed)
        communities = {}
        for comm_id, members in enumerate(partition):
            for node in members:
                communities[node] = comm_id
        return communities

    node_list = list(G.nodes())
    node_map = {n: i for i, n in enumerate(node_list)}
    edges_ig = [(node_map[u], node_map[v]) for u, v in G.edges()]

    ig_graph = ig.Graph(n=len(node_list), edges=edges_ig, directed=False)
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed
    )

    communities = {}
    for comm_id, members in enumerate(partition):
        for idx in members:
            communities[node_list[idx]] = comm_id

    return communities


def communities_to_partition(G, communities):
    """Convert communities dict to partition for modularity computation."""
    comm_sets = defaultdict(set)
    uncovered = set()

    for node in G.nodes():
        if node in communities:
            comm_sets[communities[node]].add(node)
        else:
            uncovered.add(node)

    partition = list(comm_sets.values())
    if uncovered:
        partition.append(uncovered)

    return partition


def safe_modularity(G, communities):
    """Compute modularity safely."""
    try:
        partition = communities_to_partition(G, communities)
        return nx.algorithms.community.modularity(G, partition)
    except Exception as e:
        print(f"    Warning: modularity computation failed: {e}")
        return np.nan


def load_dataset_with_communities(dataset_name, resolution=1.0):
    """Load dataset and compute Leiden communities."""
    print(f"Loading dataset: {dataset_name}")
    edges, n_nodes, _ = load_snap_dataset(dataset_name)

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    edge_set = set()
    for s, d in edges:
        if s != d:
            edge_set.add((min(s, d), max(s, d)))
    G.add_edges_from(edge_set)

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    # Run Leiden
    print(f"  Running Leiden (resolution={resolution})...")
    communities = run_leiden_nx(G, resolution=resolution)
    n_communities = len(set(communities.values()))
    print(f"  Communities: {n_communities}")

    return G, communities, edges


# =============================================================================
# Core Experiment Functions
# =============================================================================

def count_edge_types(G, communities):
    """Count intra and inter community edges."""
    intra = 0
    inter = 0
    for u, v in G.edges():
        if communities.get(u, -1) == communities.get(v, -1) and communities.get(u, -1) != -1:
            intra += 1
        else:
            inter += 1
    return intra, inter


def compute_ratio_metric(original_intra, original_inter, sparse_intra, sparse_inter):
    """
    Compute Ratio = inter_removed_frac / intra_removed_frac

    Ratio > 1: Inter-community edges removed faster
    Ratio = 1: No preference
    Ratio < 1: Intra-community edges removed faster
    """
    intra_removed_frac = (original_intra - sparse_intra) / original_intra if original_intra > 0 else 0
    inter_removed_frac = (original_inter - sparse_inter) / original_inter if original_inter > 0 else 0

    if intra_removed_frac > 0:
        ratio = inter_removed_frac / intra_removed_frac
    else:
        ratio = float('inf') if inter_removed_frac > 0 else 1.0

    return ratio, intra_removed_frac, inter_removed_frac


def run_spectral_sparsification_experiment(G, communities, edges, epsilon, dataset_name):
    """
    Run spectral sparsification and compute Ratio metric.

    Args:
        G: NetworkX graph
        communities: dict mapping node -> community_id
        edges: list of (src, dst) tuples (both directions)
        epsilon: sparsification parameter
        dataset_name: name for caching

    Returns:
        dict with all metrics
    """
    n_nodes = G.number_of_nodes()

    # Count original edge types
    original_intra, original_inter = count_edge_types(G, communities)
    original_edges = G.number_of_edges()

    print(f"    Original: {original_edges:,} edges ({original_intra:,} intra, {original_inter:,} inter)")

    # Run spectral sparsification
    print(f"    Running spectral sparsification (epsilon={epsilon})...")
    start_time = time.time()

    try:
        sparse_edges_list, sparsify_time = spectral_sparsify(edges, n_nodes, epsilon, dataset_name)
        elapsed = time.time() - start_time

        # Build sparse graph
        G_sparse = nx.Graph()
        G_sparse.add_nodes_from(G.nodes())
        sparse_edge_set = set()
        for s, d in sparse_edges_list:
            if s != d:
                sparse_edge_set.add((min(s, d), max(s, d)))
        G_sparse.add_edges_from(sparse_edge_set)

        # Count sparse edge types
        sparse_intra, sparse_inter = count_edge_types(G_sparse, communities)
        sparse_edges = G_sparse.number_of_edges()

        print(f"    Sparse: {sparse_edges:,} edges ({sparse_intra:,} intra, {sparse_inter:,} inter)")

        # Compute ratio
        ratio, intra_removed_frac, inter_removed_frac = compute_ratio_metric(
            original_intra, original_inter, sparse_intra, sparse_inter
        )

        keep_ratio = sparse_edges / original_edges if original_edges > 0 else 0

        print(f"    Keep ratio: {keep_ratio*100:.1f}%, Ratio: {ratio:.4f}")

        # Compute modularity
        modularity_original = safe_modularity(G, communities)
        modularity_sparse = safe_modularity(G_sparse, communities)
        mod_change = ((modularity_sparse - modularity_original) / modularity_original * 100
                      if modularity_original > 0 else 0)

        return {
            'dataset': dataset_name,
            'epsilon': epsilon,
            'original_edges': original_edges,
            'sparse_edges': sparse_edges,
            'keep_ratio': keep_ratio,
            'original_intra': original_intra,
            'original_inter': original_inter,
            'sparse_intra': sparse_intra,
            'sparse_inter': sparse_inter,
            'intra_removed_frac': intra_removed_frac,
            'inter_removed_frac': inter_removed_frac,
            'ratio': ratio,
            'modularity_original': modularity_original,
            'modularity_sparse': modularity_sparse,
            'modularity_change_pct': mod_change,
            'time_seconds': elapsed,
            'error': None,
        }

    except Exception as e:
        print(f"    ERROR: {e}")
        return {
            'dataset': dataset_name,
            'epsilon': epsilon,
            'error': str(e),
        }


# =============================================================================
# Comparison Functions
# =============================================================================

def find_closest_dspar_result(dataset, keep_ratio):
    """Find DSpar result with closest keep_ratio."""
    if dataset not in DSPAR_RESULTS:
        return None

    best_match = None
    best_diff = float('inf')

    for key, result in DSPAR_RESULTS[dataset].items():
        diff = abs(result['keep_ratio'] - keep_ratio)
        if diff < best_diff:
            best_diff = diff
            best_match = result.copy()
            best_match['key'] = key

    return best_match


def create_comparison_table(spectral_results_df):
    """Create comparison table between Spectral and DSpar."""
    comparison_rows = []

    for dataset in spectral_results_df['dataset'].unique():
        dataset_df = spectral_results_df[spectral_results_df['dataset'] == dataset]

        for _, row in dataset_df.iterrows():
            if row.get('error'):
                continue

            keep_ratio = row['keep_ratio']
            spectral_ratio = row['ratio']

            # Find closest DSpar result
            dspar_match = find_closest_dspar_result(dataset, keep_ratio)

            if dspar_match:
                comparison_rows.append({
                    'dataset': dataset,
                    'epsilon': row['epsilon'],
                    'keep_ratio_spectral': keep_ratio,
                    'keep_ratio_dspar': dspar_match['keep_ratio'],
                    'ratio_spectral': spectral_ratio,
                    'ratio_dspar': dspar_match['ratio'],
                    'ratio_difference': spectral_ratio - dspar_match['ratio'],
                    'ratio_relative_diff': (spectral_ratio - dspar_match['ratio']) / dspar_match['ratio'] * 100,
                })

    return pd.DataFrame(comparison_rows)


def compute_correlation(comparison_df):
    """Compute correlation between Spectral and DSpar ratios."""
    if not HAS_SCIPY or len(comparison_df) < 3:
        return None

    spectral_ratios = comparison_df['ratio_spectral'].values
    dspar_ratios = comparison_df['ratio_dspar'].values

    # Remove any inf/nan
    valid_mask = np.isfinite(spectral_ratios) & np.isfinite(dspar_ratios)
    if valid_mask.sum() < 3:
        return None

    spectral_valid = spectral_ratios[valid_mask]
    dspar_valid = dspar_ratios[valid_mask]

    pearson_r, pearson_p = pearsonr(spectral_valid, dspar_valid)
    spearman_r, spearman_p = spearmanr(spectral_valid, dspar_valid)

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_samples': len(spectral_valid),
    }


# =============================================================================
# Visualization
# =============================================================================

def create_comparison_plot(spectral_results_df, comparison_df, results_dir):
    """Create comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Spectral Ratio vs Epsilon by dataset
    ax1 = axes[0, 0]
    for dataset in spectral_results_df['dataset'].unique():
        df = spectral_results_df[spectral_results_df['dataset'] == dataset]
        df = df[df['error'].isna()]
        if len(df) > 0:
            ax1.plot(df['epsilon'], df['ratio'], 'o-', label=dataset, markersize=8)

    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No preference')
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Ratio (inter_removed / intra_removed)')
    ax1.set_title('Spectral Sparsification: Ratio vs Epsilon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Keep Ratio vs Epsilon
    ax2 = axes[0, 1]
    for dataset in spectral_results_df['dataset'].unique():
        df = spectral_results_df[spectral_results_df['dataset'] == dataset]
        df = df[df['error'].isna()]
        if len(df) > 0:
            ax2.plot(df['epsilon'], df['keep_ratio'] * 100, 'o-', label=dataset, markersize=8)

    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Keep Ratio (%)')
    ax2.set_title('Edge Retention vs Epsilon')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Spectral vs DSpar Ratio scatter
    ax3 = axes[1, 0]
    if len(comparison_df) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_df['dataset'].unique())))
        for i, dataset in enumerate(comparison_df['dataset'].unique()):
            df = comparison_df[comparison_df['dataset'] == dataset]
            ax3.scatter(df['ratio_dspar'], df['ratio_spectral'], label=dataset,
                       color=colors[i], s=100, alpha=0.7)

        # Add diagonal line
        all_ratios = list(comparison_df['ratio_dspar']) + list(comparison_df['ratio_spectral'])
        min_r, max_r = min(all_ratios), max(all_ratios)
        ax3.plot([min_r, max_r], [min_r, max_r], 'k--', alpha=0.5, label='y=x')

        ax3.set_xlabel('DSpar Ratio')
        ax3.set_ylabel('Spectral Ratio')
        ax3.set_title('Spectral vs DSpar Ratio Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Ratio comparison bar chart
    ax4 = axes[1, 1]
    if len(comparison_df) > 0:
        # Get one comparison per dataset (closest to 50% keep ratio)
        summary_rows = []
        for dataset in comparison_df['dataset'].unique():
            df = comparison_df[comparison_df['dataset'] == dataset]
            # Find row closest to 50% keep ratio
            df = df.copy()
            df['keep_diff'] = abs(df['keep_ratio_spectral'] - 0.5)
            best_row = df.loc[df['keep_diff'].idxmin()]
            summary_rows.append(best_row)

        summary_df = pd.DataFrame(summary_rows)

        x = np.arange(len(summary_df))
        width = 0.35

        ax4.bar(x - width/2, summary_df['ratio_spectral'], width, label='Spectral', color='steelblue')
        ax4.bar(x + width/2, summary_df['ratio_dspar'], width, label='DSpar', color='coral')
        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Ratio')
        ax4.set_title('Spectral vs DSpar Ratio (at ~50% keep ratio)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(summary_df['dataset'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved to {results_dir / 'comparison_plot.png'}")


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(datasets, epsilon_values, resolution=1.0):
    """Run the full spectral vs DSpar comparison experiment."""

    ensure_dir(RESULTS_DIR)

    print("=" * 80)
    print("SPECTRAL SPARSIFICATION vs DSPAR COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"\nDatasets: {datasets}")
    print(f"Epsilon values: {epsilon_values}")
    print(f"Leiden resolution: {resolution}")
    print(f"Results directory: {RESULTS_DIR}")

    all_results = []

    # Run experiments
    for dataset_name in datasets:
        print("\n" + "#" * 80)
        print(f"# DATASET: {dataset_name}")
        print("#" * 80)

        try:
            # Load dataset and communities
            G, communities, edges = load_dataset_with_communities(dataset_name, resolution)

            # Determine epsilon values for this dataset
            dataset_epsilons = list(epsilon_values)
            if dataset_name in PARAMETERS.get('datasets_extended', []):
                dataset_epsilons = dataset_epsilons + PARAMETERS.get('epsilon_values_extended', [])
                print(f"  Using extended epsilon values: {dataset_epsilons}")

            # Run spectral sparsification at each epsilon
            for epsilon in dataset_epsilons:
                print(f"\n  Epsilon = {epsilon}:")
                result = run_spectral_sparsification_experiment(
                    G, communities, edges, epsilon, dataset_name
                )
                all_results.append(result)

        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            import traceback
            traceback.print_exc()

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save detailed results
    results_df.to_csv(RESULTS_DIR / "spectral_sparsification_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'spectral_sparsification_results.csv'}")

    # Print results table
    print("\n" + "=" * 100)
    print("SPECTRAL SPARSIFICATION RESULTS")
    print("=" * 100)

    for dataset in results_df['dataset'].unique():
        df = results_df[results_df['dataset'] == dataset]
        print(f"\nDATASET: {dataset}")
        print("-" * 90)
        print(f"{'Epsilon':>8} {'Keep%':>8} {'Ratio':>10} {'Mod_orig':>10} {'Mod_sparse':>11} {'Mod_change':>11}")
        print("-" * 90)

        for _, row in df.iterrows():
            if row.get('error'):
                print(f"{row['epsilon']:>8} {'ERROR':>8} {row['error'][:50]}")
            else:
                mod_change_str = f"{row['modularity_change_pct']:+.1f}%"
                print(f"{row['epsilon']:>8.1f} {row['keep_ratio']*100:>7.1f}% {row['ratio']:>10.4f} "
                      f"{row['modularity_original']:>10.4f} {row['modularity_sparse']:>11.4f} {mod_change_str:>11}")

    # Create comparison table
    comparison_df = create_comparison_table(results_df)

    if len(comparison_df) > 0:
        comparison_df.to_csv(RESULTS_DIR / "comparison_summary.csv", index=False)

        print("\n" + "=" * 100)
        print("COMPARISON: SPECTRAL vs DSPAR")
        print("=" * 100)
        print(f"\n{'Dataset':<12} | {'Keep%':>8} | {'Spectral':>10} | {'DSpar':>10} | {'Diff':>10} | {'Rel Diff':>10}")
        print("-" * 75)

        for _, row in comparison_df.iterrows():
            print(f"{row['dataset']:<12} | {row['keep_ratio_spectral']*100:>7.1f}% | "
                  f"{row['ratio_spectral']:>10.4f} | {row['ratio_dspar']:>10.4f} | "
                  f"{row['ratio_difference']:>+10.4f} | {row['ratio_relative_diff']:>+9.1f}%")

        # Compute correlation
        corr = compute_correlation(comparison_df)
        if corr:
            print(f"\nCorrelation Analysis (n={corr['n_samples']}):")
            print(f"  Pearson r = {corr['pearson_r']:.4f} (p = {corr['pearson_p']:.4e})")
            print(f"  Spearman r = {corr['spearman_r']:.4f} (p = {corr['spearman_p']:.4e})")

        # Create plots
        print("\nGenerating comparison plots...")
        create_comparison_plot(results_df, comparison_df, RESULTS_DIR)

    # Print conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    valid_results = results_df[results_df['error'].isna()]
    if len(valid_results) > 0:
        avg_ratio = valid_results['ratio'].mean()
        ratio_gt_1 = (valid_results['ratio'] > 1).sum()
        total = len(valid_results)

        print(f"\nSpectral Sparsification Results:")
        print(f"  Average Ratio: {avg_ratio:.4f}")
        print(f"  Ratio > 1 in {ratio_gt_1}/{total} cases ({ratio_gt_1/total*100:.1f}%)")

        if avg_ratio > 1:
            print(f"\n  --> Spectral sparsification DOES preferentially remove inter-community edges")
        else:
            print(f"\n  --> Spectral sparsification does NOT preferentially remove inter-community edges")

        if len(comparison_df) > 0:
            spectral_ratios = comparison_df['ratio_spectral'].mean()
            dspar_ratios = comparison_df['ratio_dspar'].mean()
            print(f"\nComparison with DSpar:")
            print(f"  Average Spectral Ratio: {spectral_ratios:.4f}")
            print(f"  Average DSpar Ratio: {dspar_ratios:.4f}")

            if corr and corr['pearson_r'] > 0.5:
                print(f"\n  --> Spectral and DSpar ratios are POSITIVELY CORRELATED (r={corr['pearson_r']:.3f})")
                print(f"      This supports the theory that DSpar approximates spectral sparsification")
            elif corr and corr['pearson_r'] < -0.5:
                print(f"\n  --> Spectral and DSpar ratios are NEGATIVELY CORRELATED (r={corr['pearson_r']:.3f})")
            else:
                print(f"\n  --> No strong correlation found between Spectral and DSpar ratios")

    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'epsilon_values': epsilon_values,
            'datasets': datasets,
            'leiden_resolution': resolution,
        },
        'results': results_df.to_dict('records'),
        'comparison': comparison_df.to_dict('records') if len(comparison_df) > 0 else [],
        'correlation': corr if 'corr' in dir() and corr else None,
    }

    with open(RESULTS_DIR / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll results saved to {RESULTS_DIR}")
    print("=" * 80)

    return results_df, comparison_df


def main():
    parser = argparse.ArgumentParser(description='Spectral vs DSpar Comparison Experiment')
    parser.add_argument('--datasets', nargs='+',
                        default=PARAMETERS['datasets'],
                        help='Datasets to analyze')
    parser.add_argument('--epsilon', nargs='+', type=float,
                        default=PARAMETERS['epsilon_values'],
                        help='Epsilon values to test')
    parser.add_argument('--resolution', type=float,
                        default=PARAMETERS['leiden_resolution'],
                        help='Leiden resolution parameter')

    args = parser.parse_args()

    run_experiment(
        datasets=args.datasets,
        epsilon_values=args.epsilon,
        resolution=args.resolution
    )


if __name__ == '__main__':
    main()
