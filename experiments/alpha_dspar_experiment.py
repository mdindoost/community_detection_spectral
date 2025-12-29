#!/usr/bin/env python3
"""
Alpha-DSpar Experiment

Tests whether amplifying DSpar's degree-based sampling improves community preservation.
The hypothesis is that raising the DSpar score to a power α > 1 will more aggressively
keep intra-community edges and remove inter-community edges.

Standard DSpar (α = 1): p_e ∝ 1/d_u + 1/d_v
Alpha-DSpar (α > 1):    p_e ∝ (1/d_u + 1/d_v)^α

When α > 1:
- Low-degree edges get even higher probability (kept more)
- High-degree edges get even lower probability (removed more)
- The gap between intra and inter-community edges is amplified

Usage:
    python experiments/alpha_dspar_experiment.py
    python experiments/alpha_dspar_experiment.py --datasets cit-HepTh cit-HepPh
    python experiments/alpha_dspar_experiment.py --alpha 1.0 1.5 2.0 2.5
"""

import sys
import argparse
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import (
    PROJECT_ROOT,
    load_snap_dataset,
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
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available")

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Configuration
# =============================================================================

PARAMETERS = {
    # Alpha values to test
    # alpha = 0.0: Uniform random (baseline)
    # alpha = 1.0: Standard DSpar
    # alpha > 1.0: Amplified DSpar (our hypothesis)
    'alpha_values': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],

    # Keep ratios to test
    'keep_ratios': [0.25, 0.50, 0.75],

    # Random seeds for multiple runs (statistical significance)
    'seeds': [42, 123, 456],

    # Datasets (ordered smallest to largest)
    'datasets': ['cit-HepTh', 'cit-HepPh', 'com-DBLP', 'com-Youtube'],

    # Leiden parameters
    'leiden_resolution': 1.0,
}

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results" / "alpha_dspar"


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

    return G, communities


# =============================================================================
# Alpha-DSpar Algorithm
# =============================================================================

def alpha_dspar_sparsify(G, keep_ratio, alpha=1.0, seed=42):
    """
    Alpha-DSpar: Degree-based sparsification with amplification.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    keep_ratio : float
        Fraction of edges to keep (0 < keep_ratio <= 1)
    alpha : float
        Amplification exponent (alpha=1 is standard DSpar)
        alpha > 1: more aggressive degree-based removal
        alpha < 1: less aggressive (approaches random)
        alpha = 0: uniform random sampling
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    G_sparse : networkx.Graph
        Sparsified graph
    """
    np.random.seed(seed)

    m = G.number_of_edges()
    Q = int(keep_ratio * m)  # Number of edges to sample

    if Q >= m:
        return G.copy()

    edges = list(G.edges())

    # Compute sampling probabilities
    probs = []
    for u, v in edges:
        d_u, d_v = G.degree(u), G.degree(v)
        dspar_score = 1.0 / d_u + 1.0 / d_v

        # Apply alpha amplification
        if alpha == 0:
            prob = 1.0  # Uniform
        else:
            prob = dspar_score ** alpha

        probs.append(prob)

    # Normalize probabilities
    probs = np.array(probs)
    probs = probs / probs.sum()

    # Sample edges with replacement (like original DSpar)
    sampled_indices = np.random.choice(len(edges), size=Q, replace=True, p=probs)

    # Count occurrences for edge weights
    edge_counts = Counter(sampled_indices)

    # Build sparsified graph
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes())

    for idx, count in edge_counts.items():
        u, v = edges[idx]
        # Weight = count / (Q * p_e) for unbiased estimator
        weight = count / (Q * probs[idx])
        if G_sparse.has_edge(u, v):
            G_sparse[u][v]['weight'] += weight
        else:
            G_sparse.add_edge(u, v, weight=weight)

    return G_sparse


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


def run_alpha_dspar_experiment(G, communities_original, alpha, keep_ratio, seed, dataset_name):
    """
    Run single Alpha-DSpar experiment and collect all metrics.
    """
    # Count original edge types
    original_intra, original_inter = count_edge_types(G, communities_original)
    original_edges = G.number_of_edges()

    # Apply alpha-DSpar
    G_sparse = alpha_dspar_sparsify(G, keep_ratio, alpha, seed)

    # Count sparse edge types
    sparse_intra, sparse_inter = count_edge_types(G_sparse, communities_original)
    sparse_edges = G_sparse.number_of_edges()

    # Compute Ratio
    intra_removed_frac = (original_intra - sparse_intra) / original_intra if original_intra > 0 else 0
    inter_removed_frac = (original_inter - sparse_inter) / original_inter if original_inter > 0 else 0

    if intra_removed_frac > 0:
        ratio = inter_removed_frac / intra_removed_frac
    else:
        ratio = float('inf') if inter_removed_frac > 0 else 1.0

    # Run Leiden on sparsified graph
    communities_sparse = run_leiden_nx(G_sparse, seed=seed)

    # Compute NMI and ARI
    if HAS_SKLEARN:
        nodes = sorted(set(communities_original.keys()) & set(communities_sparse.keys()))
        if len(nodes) > 0:
            orig_labels = [communities_original[n] for n in nodes]
            sparse_labels = [communities_sparse[n] for n in nodes]
            nmi = normalized_mutual_info_score(orig_labels, sparse_labels)
            ari = adjusted_rand_score(orig_labels, sparse_labels)
        else:
            nmi, ari = np.nan, np.nan
    else:
        nmi, ari = np.nan, np.nan

    # Compute modularity
    modularity_original = safe_modularity(G, communities_original)
    modularity_preserved = safe_modularity(G_sparse, communities_original)
    modularity_new = safe_modularity(G_sparse, communities_sparse)

    # Check connectivity
    n_components = nx.number_connected_components(G_sparse)
    if n_components > 0:
        largest_cc = len(max(nx.connected_components(G_sparse), key=len)) / G.number_of_nodes()
    else:
        largest_cc = 0

    return {
        'dataset': dataset_name,
        'alpha': alpha,
        'keep_ratio': keep_ratio,
        'seed': seed,
        'original_edges': original_edges,
        'sparse_edges': sparse_edges,
        'actual_keep_ratio': sparse_edges / original_edges if original_edges > 0 else 0,
        'original_intra': original_intra,
        'original_inter': original_inter,
        'sparse_intra': sparse_intra,
        'sparse_inter': sparse_inter,
        'intra_removed_frac': intra_removed_frac,
        'inter_removed_frac': inter_removed_frac,
        'ratio': ratio,
        'nmi': nmi,
        'ari': ari,
        'modularity_original': modularity_original,
        'modularity_preserved': modularity_preserved,
        'modularity_new': modularity_new,
        'n_components': n_components,
        'largest_cc_fraction': largest_cc,
    }


# =============================================================================
# Visualization
# =============================================================================

def create_plots(results_df, summary_df, results_dir):
    """Create all visualization plots."""

    print("\nGenerating plots...")

    # Plot 1: Ratio vs Alpha (one line per dataset)
    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset in results_df['dataset'].unique():
        df = summary_df[(summary_df['dataset'] == dataset) & (summary_df['keep_ratio'] == 0.50)]
        if len(df) > 0:
            ax.errorbar(df['alpha'], df['ratio_mean'], yerr=df['ratio_std'],
                       marker='o', label=dataset, capsize=3, markersize=8)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No preference')
    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel('Ratio (inter_removed / intra_removed)', fontsize=12)
    ax.set_title('Ratio vs Alpha (keep_ratio=50%)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "ratio_vs_alpha.png", dpi=150)
    plt.close()

    # Plot 2: NMI vs Alpha
    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset in results_df['dataset'].unique():
        df = summary_df[(summary_df['dataset'] == dataset) & (summary_df['keep_ratio'] == 0.50)]
        if len(df) > 0:
            ax.errorbar(df['alpha'], df['nmi_mean'], yerr=df['nmi_std'],
                       marker='o', label=dataset, capsize=3, markersize=8)

    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel('NMI (vs original communities)', fontsize=12)
    ax.set_title('Community Preservation (NMI) vs Alpha (keep_ratio=50%)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "nmi_vs_alpha.png", dpi=150)
    plt.close()

    # Plot 3: Ratio vs NMI Scatter (colored by alpha)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(results_df['ratio'], results_df['nmi'],
                        c=results_df['alpha'], cmap='viridis',
                        alpha=0.6, s=50)
    plt.colorbar(scatter, label='Alpha (α)')
    ax.set_xlabel('Ratio', fontsize=12)
    ax.set_ylabel('NMI', fontsize=12)
    ax.set_title('Ratio vs NMI (colored by Alpha)', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "ratio_vs_nmi_scatter.png", dpi=150)
    plt.close()

    # Plot 4: Connectivity vs Alpha
    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset in results_df['dataset'].unique():
        df = summary_df[(summary_df['dataset'] == dataset) & (summary_df['keep_ratio'] == 0.50)]
        if len(df) > 0:
            ax.plot(df['alpha'], df['largest_cc_mean'], marker='o', label=dataset, markersize=8)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Fully connected')
    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel('Largest CC Fraction', fontsize=12)
    ax.set_title('Graph Connectivity vs Alpha (keep_ratio=50%)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(results_dir / "connectivity_vs_alpha.png", dpi=150)
    plt.close()

    # Plot 5: Heatmap of Ratio by Dataset and Alpha
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, keep_ratio in enumerate([0.25, 0.50, 0.75]):
        df = summary_df[summary_df['keep_ratio'] == keep_ratio]
        if len(df) > 0:
            pivot = df.pivot(index='dataset', columns='alpha', values='ratio_mean')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[idx],
                       center=1.0, vmin=0.9, vmax=1.5)
            axes[idx].set_title(f'Ratio (keep_ratio={keep_ratio:.0%})')
            axes[idx].set_xlabel('Alpha')
            axes[idx].set_ylabel('Dataset')

    plt.tight_layout()
    plt.savefig(results_dir / "ratio_heatmap.png", dpi=150)
    plt.close()

    # Plot 6: NMI Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, keep_ratio in enumerate([0.25, 0.50, 0.75]):
        df = summary_df[summary_df['keep_ratio'] == keep_ratio]
        if len(df) > 0:
            pivot = df.pivot(index='dataset', columns='alpha', values='nmi_mean')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'NMI (keep_ratio={keep_ratio:.0%})')
            axes[idx].set_xlabel('Alpha')
            axes[idx].set_ylabel('Dataset')

    plt.tight_layout()
    plt.savefig(results_dir / "nmi_heatmap.png", dpi=150)
    plt.close()

    # Plot 7: Improvement over Random (alpha=0) and Standard DSpar (alpha=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # NMI improvement over random
    for dataset in results_df['dataset'].unique():
        df = summary_df[(summary_df['dataset'] == dataset) & (summary_df['keep_ratio'] == 0.50)]
        if len(df) > 0:
            baseline = df[df['alpha'] == 0.0]['nmi_mean'].values
            if len(baseline) > 0:
                baseline = baseline[0]
                improvement = (df['nmi_mean'] - baseline) / baseline * 100
                axes[0].plot(df['alpha'], improvement, marker='o', label=dataset, markersize=8)

    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Alpha (α)', fontsize=12)
    axes[0].set_ylabel('NMI Improvement over Random (%)', fontsize=12)
    axes[0].set_title('NMI Improvement over Random (α=0)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # NMI improvement over standard DSpar (alpha=1)
    for dataset in results_df['dataset'].unique():
        df = summary_df[(summary_df['dataset'] == dataset) & (summary_df['keep_ratio'] == 0.50)]
        if len(df) > 0:
            baseline = df[df['alpha'] == 1.0]['nmi_mean'].values
            if len(baseline) > 0:
                baseline = baseline[0]
                improvement = (df['nmi_mean'] - baseline) / baseline * 100
                axes[1].plot(df['alpha'], improvement, marker='o', label=dataset, markersize=8)

    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Alpha (α)', fontsize=12)
    axes[1].set_ylabel('NMI Improvement over DSpar (%)', fontsize=12)
    axes[1].set_title('NMI Improvement over Standard DSpar (α=1)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "nmi_improvement.png", dpi=150)
    plt.close()

    print(f"  Plots saved to {results_dir}")


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(datasets, alpha_values, keep_ratios, seeds, resolution=1.0):
    """Run the full Alpha-DSpar experiment."""

    ensure_dir(RESULTS_DIR)

    print("=" * 80)
    print("ALPHA-DSPAR EXPERIMENT")
    print("=" * 80)
    print(f"\nDatasets: {datasets}")
    print(f"Alpha values: {alpha_values}")
    print(f"Keep ratios: {keep_ratios}")
    print(f"Seeds: {seeds}")
    print(f"Leiden resolution: {resolution}")
    print(f"Results directory: {RESULTS_DIR}")

    total_experiments = len(datasets) * len(alpha_values) * len(keep_ratios) * len(seeds)
    print(f"\nTotal experiments: {total_experiments}")

    all_results = []
    exp_count = 0

    # Run experiments
    for dataset_name in datasets:
        print("\n" + "#" * 80)
        print(f"# DATASET: {dataset_name}")
        print("#" * 80)

        try:
            # Load dataset and communities
            G, communities = load_dataset_with_communities(dataset_name, resolution)

            for keep_ratio in keep_ratios:
                print(f"\n  Keep Ratio: {keep_ratio:.0%}")
                print(f"  {'-'*60}")

                for alpha in alpha_values:
                    for seed in seeds:
                        exp_count += 1
                        print(f"    [{exp_count}/{total_experiments}] α={alpha}, seed={seed}...", end=" ")

                        try:
                            result = run_alpha_dspar_experiment(
                                G, communities, alpha, keep_ratio, seed, dataset_name
                            )
                            all_results.append(result)
                            print(f"Ratio={result['ratio']:.4f}, NMI={result['nmi']:.4f}")
                        except Exception as e:
                            print(f"ERROR: {e}")
                            all_results.append({
                                'dataset': dataset_name,
                                'alpha': alpha,
                                'keep_ratio': keep_ratio,
                                'seed': seed,
                                'error': str(e),
                            })

        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            import traceback
            traceback.print_exc()

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Filter out errors for analysis
    valid_df = results_df[~results_df.get('error', pd.Series([None]*len(results_df))).notna()].copy()

    # Save raw results
    results_df.to_csv(RESULTS_DIR / "alpha_dspar_results.csv", index=False)

    # Compute summary statistics
    summary_rows = []
    for dataset in valid_df['dataset'].unique():
        for alpha in valid_df['alpha'].unique():
            for keep_ratio in valid_df['keep_ratio'].unique():
                subset = valid_df[(valid_df['dataset'] == dataset) &
                                  (valid_df['alpha'] == alpha) &
                                  (valid_df['keep_ratio'] == keep_ratio)]
                if len(subset) > 0:
                    summary_rows.append({
                        'dataset': dataset,
                        'alpha': alpha,
                        'keep_ratio': keep_ratio,
                        'n_runs': len(subset),
                        'ratio_mean': subset['ratio'].mean(),
                        'ratio_std': subset['ratio'].std(),
                        'nmi_mean': subset['nmi'].mean(),
                        'nmi_std': subset['nmi'].std(),
                        'ari_mean': subset['ari'].mean(),
                        'ari_std': subset['ari'].std(),
                        'modularity_preserved_mean': subset['modularity_preserved'].mean(),
                        'modularity_preserved_std': subset['modularity_preserved'].std(),
                        'n_components_mean': subset['n_components'].mean(),
                        'largest_cc_mean': subset['largest_cc_fraction'].mean(),
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "alpha_dspar_summary.csv", index=False)

    # Print results
    print("\n" + "=" * 100)
    print("ALPHA-DSPAR EXPERIMENT RESULTS")
    print("=" * 100)

    for dataset in summary_df['dataset'].unique():
        print(f"\nDATASET: {dataset}")

        for keep_ratio in sorted(summary_df['keep_ratio'].unique()):
            print(f"\n  Keep Ratio: {keep_ratio:.0%}")
            print(f"  {'-'*80}")
            print(f"  {'Alpha':<8} {'Ratio (mean±std)':<18} {'NMI (mean±std)':<18} {'Modularity':<12} {'Components':<12}")
            print(f"  {'-'*80}")

            df = summary_df[(summary_df['dataset'] == dataset) & (summary_df['keep_ratio'] == keep_ratio)]
            df = df.sort_values('alpha')

            for _, row in df.iterrows():
                ratio_str = f"{row['ratio_mean']:.3f} ± {row['ratio_std']:.3f}"
                nmi_str = f"{row['nmi_mean']:.3f} ± {row['nmi_std']:.3f}"
                mod_str = f"{row['modularity_preserved_mean']:.3f}"
                comp_str = f"{row['n_components_mean']:.1f}"
                print(f"  {row['alpha']:<8.1f} {ratio_str:<18} {nmi_str:<18} {mod_str:<12} {comp_str:<12}")

    # Find optimal alpha for each dataset
    print("\n" + "=" * 80)
    print("OPTIMAL ALPHA BY DATASET (maximizing NMI)")
    print("=" * 80)

    optimal_results = []
    for dataset in summary_df['dataset'].unique():
        for keep_ratio in sorted(summary_df['keep_ratio'].unique()):
            df = summary_df[(summary_df['dataset'] == dataset) & (summary_df['keep_ratio'] == keep_ratio)]
            if len(df) > 0:
                best_row = df.loc[df['nmi_mean'].idxmax()]
                optimal_results.append({
                    'dataset': dataset,
                    'keep_ratio': keep_ratio,
                    'optimal_alpha': best_row['alpha'],
                    'nmi_at_optimal': best_row['nmi_mean'],
                    'ratio_at_optimal': best_row['ratio_mean'],
                })
                print(f"  {dataset} (keep={keep_ratio:.0%}): α={best_row['alpha']:.1f} → NMI={best_row['nmi_mean']:.4f}")

    optimal_df = pd.DataFrame(optimal_results)
    optimal_df.to_csv(RESULTS_DIR / "optimal_alpha_by_dataset.csv", index=False)

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # 1. Ratio vs Alpha correlation
    if HAS_SCIPY and len(valid_df) > 10:
        corr, pval = pearsonr(valid_df['alpha'], valid_df['ratio'])
        print(f"\n1. Ratio vs Alpha Correlation:")
        print(f"   Pearson r = {corr:.4f} (p = {pval:.2e})")
        if corr > 0.5:
            print(f"   → Higher alpha leads to higher Ratio (as expected)")

    # 2. NMI improvement
    print(f"\n2. NMI Improvement over Standard DSpar (α=1):")
    for dataset in summary_df['dataset'].unique():
        df = summary_df[(summary_df['dataset'] == dataset) & (summary_df['keep_ratio'] == 0.50)]
        if len(df) > 0:
            baseline = df[df['alpha'] == 1.0]['nmi_mean'].values
            best = df['nmi_mean'].max()
            best_alpha = df.loc[df['nmi_mean'].idxmax(), 'alpha']
            if len(baseline) > 0:
                improvement = (best - baseline[0]) / baseline[0] * 100
                print(f"   {dataset}: {improvement:+.2f}% (optimal α={best_alpha})")

    # 3. Connectivity warning
    print(f"\n3. Connectivity Analysis:")
    fragmented = summary_df[summary_df['n_components_mean'] > 1.5]
    if len(fragmented) > 0:
        min_frag_alpha = fragmented['alpha'].min()
        print(f"   Graph fragmentation starts at α > {min_frag_alpha}")
        print(f"   Safe alpha range: [0, {min_frag_alpha}]")
    else:
        print(f"   No significant fragmentation observed")

    # Create plots
    if len(valid_df) > 0:
        create_plots(valid_df, summary_df, RESULTS_DIR)

    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'alpha_values': alpha_values,
            'keep_ratios': keep_ratios,
            'seeds': seeds,
            'datasets': datasets,
            'leiden_resolution': resolution,
        },
        'total_experiments': total_experiments,
        'successful_experiments': len(valid_df),
    }

    with open(RESULTS_DIR / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll results saved to {RESULTS_DIR}")
    print("=" * 80)

    return results_df, summary_df


def main():
    parser = argparse.ArgumentParser(description='Alpha-DSpar Experiment')
    parser.add_argument('--datasets', nargs='+',
                        default=PARAMETERS['datasets'],
                        help='Datasets to analyze')
    parser.add_argument('--alpha', nargs='+', type=float,
                        default=PARAMETERS['alpha_values'],
                        help='Alpha values to test')
    parser.add_argument('--keep-ratios', nargs='+', type=float,
                        default=PARAMETERS['keep_ratios'],
                        help='Keep ratios to test')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=PARAMETERS['seeds'],
                        help='Random seeds for multiple runs')
    parser.add_argument('--resolution', type=float,
                        default=PARAMETERS['leiden_resolution'],
                        help='Leiden resolution parameter')

    args = parser.parse_args()

    run_experiment(
        datasets=args.datasets,
        alpha_values=args.alpha,
        keep_ratios=args.keep_ratios,
        seeds=args.seeds,
        resolution=args.resolution
    )


if __name__ == '__main__':
    main()
