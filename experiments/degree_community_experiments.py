#!/usr/bin/env python3
"""
Degree-Community Correlation Experiments

Tests the hypothesis: Inter-community edges tend to connect higher-degree nodes (hubs),
which explains why spectral sparsification preferentially removes inter-community edges.

Specifically:
    E[1/d_u + 1/d_v | inter-community] < E[1/d_u + 1/d_v | intra-community]
    E[(d_u * d_v) | inter-community] > E[(d_u * d_v) | intra-community]

Experiments:
1. Degree-Community Correlation Analysis
2. Distribution Visualization
3. Hub Analysis
4. Sparsification Simulation (DSpar vs Random)
5. Modularity-Preserving Analysis
6. Eigenvector Analysis

Usage:
    python experiments/degree_community_experiments.py --datasets com-Youtube com-DBLP
    python experiments/degree_community_experiments.py --experiments 1 2 3
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import (
    PROJECT_ROOT,
    load_snap_dataset,
    edges_to_adjacency,
    adjacency_to_igraph,
)

# Try sklearn imports
try:
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, some metrics will be skipped")

# Try igraph/leidenalg
try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False
    print("Warning: leidenalg not available")


# =============================================================================
# Configuration
# =============================================================================

PARAMETERS = {
    # Datasets (ordered from smallest to largest)
    'datasets': ['cit-HepTh', 'cit-HepPh', 'com-DBLP', 'com-Youtube'],

    # Leiden parameters
    'leiden_resolution': 1.0,
    'leiden_n_iterations': -1,  # Until convergence

    # Hub analysis
    'hub_percentiles': [1, 5, 10],  # Top X% considered hubs

    # Sparsification simulation
    'keep_ratios': [0.25, 0.50, 0.75],
    'sparsification_methods': ['dspar', 'random'],

    # Statistical tests
    'significance_level': 0.05,

    # For large graphs, consider sampling
    'max_edges_for_full_analysis': 10_000_000,
    'sample_size': 1_000_000,
}

# Results directory
RESULTS_BASE = PROJECT_ROOT / "results" / "degree_community"


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset_with_communities(dataset_name):
    """
    Load dataset and get community assignments.
    Always uses Leiden for community detection (consistent across all datasets).

    Returns:
        G: NetworkX graph
        communities: dict mapping node -> community_id
        has_ground_truth: bool (always False since we use Leiden)
    """
    print(f"Loading dataset: {dataset_name}")
    edges, n_nodes, ground_truth = load_snap_dataset(dataset_name)

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

    # Always use Leiden for community detection (consistent across datasets)
    print(f"  Running Leiden for community detection (resolution={PARAMETERS['leiden_resolution']})...")
    communities = run_leiden_nx(G, resolution=PARAMETERS['leiden_resolution'])

    n_communities = len(set(communities.values()))
    print(f"  Communities: {n_communities}")

    return G, communities, False


def run_leiden_nx(G, resolution=1.0, seed=42):
    """
    Run Leiden community detection on NetworkX graph.

    Args:
        G: NetworkX graph
        resolution: Resolution parameter (1.0 = standard modularity, higher = more communities)
        seed: Random seed for reproducibility

    Returns dict: node -> community_id
    """
    if not HAS_LEIDEN:
        # Fallback to NetworkX Louvain
        from networkx.algorithms.community import louvain_communities
        partition = louvain_communities(G, resolution=resolution, seed=seed)
        communities = {}
        for comm_id, members in enumerate(partition):
            for node in members:
                communities[node] = comm_id
        return communities

    # Convert to igraph
    node_list = list(G.nodes())
    node_map = {n: i for i, n in enumerate(node_list)}
    edges_ig = [(node_map[u], node_map[v]) for u, v in G.edges()]

    ig_graph = ig.Graph(n=len(node_list), edges=edges_ig, directed=False)

    # Use RBConfigurationVertexPartition to support resolution parameter
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


# =============================================================================
# Experiment 1: Degree-Community Correlation Analysis
# =============================================================================

def compute_edge_metrics(G, communities):
    """
    Compute degree-based metrics for each edge.

    Returns:
        DataFrame with edge metrics
    """
    m = G.number_of_edges()

    records = []
    for u, v in G.edges():
        d_u = G.degree(u)
        d_v = G.degree(v)

        # DSpar score (higher = more likely to be kept)
        dspar_score = 1/d_u + 1/d_v

        # Newman's null model expectation
        null_model_expected = (d_u * d_v) / (2 * m)

        # Modularity contribution (for existing edge A_ij = 1)
        modularity_contribution = 1 - null_model_expected

        # Degree product
        degree_product = d_u * d_v

        # Harmonic mean of degrees
        harmonic_mean_degree = 2 * d_u * d_v / (d_u + d_v) if (d_u + d_v) > 0 else 0

        # Edge type
        comm_u = communities.get(u, -1)
        comm_v = communities.get(v, -1)
        is_intra = (comm_u == comm_v) and (comm_u != -1)

        records.append({
            'u': u,
            'v': v,
            'd_u': d_u,
            'd_v': d_v,
            'dspar_score': dspar_score,
            'null_model_expected': null_model_expected,
            'modularity_contribution': modularity_contribution,
            'degree_product': degree_product,
            'harmonic_mean_degree': harmonic_mean_degree,
            'is_intra': is_intra,
            'edge_type': 'intra' if is_intra else 'inter'
        })

    return pd.DataFrame(records)


def compute_statistics(intra_values, inter_values, metric_name):
    """
    Compute statistical comparison between intra and inter community edges.
    """
    intra = np.array(intra_values)
    inter = np.array(inter_values)

    result = {
        'metric': metric_name,
        'intra_mean': np.mean(intra),
        'intra_std': np.std(intra),
        'intra_median': np.median(intra),
        'inter_mean': np.mean(inter),
        'inter_std': np.std(inter),
        'inter_median': np.median(inter),
        'n_intra': len(intra),
        'n_inter': len(inter),
    }

    # Mann-Whitney U test (non-parametric)
    if len(intra) > 0 and len(inter) > 0:
        try:
            stat, pvalue = stats.mannwhitneyu(intra, inter, alternative='two-sided')
            result['mannwhitney_stat'] = stat
            result['pvalue'] = pvalue
        except Exception:
            result['mannwhitney_stat'] = np.nan
            result['pvalue'] = np.nan
    else:
        result['mannwhitney_stat'] = np.nan
        result['pvalue'] = np.nan

    # Effect size (Cohen's d)
    if len(intra) > 0 and len(inter) > 0:
        pooled_std = np.sqrt((np.var(intra) + np.var(inter)) / 2)
        if pooled_std > 0:
            result['cohens_d'] = (np.mean(intra) - np.mean(inter)) / pooled_std
        else:
            result['cohens_d'] = 0
    else:
        result['cohens_d'] = np.nan

    # AUC: Can this metric predict edge type?
    if HAS_SKLEARN and len(intra) > 0 and len(inter) > 0:
        try:
            labels = [1] * len(intra) + [0] * len(inter)
            scores = list(intra) + list(inter)
            result['auc'] = roc_auc_score(labels, scores)
        except Exception:
            result['auc'] = np.nan
    else:
        result['auc'] = np.nan

    return result


def run_experiment1(G, communities, dataset_name, results_dir):
    """
    Experiment 1: Degree-Community Correlation Analysis
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Degree-Community Correlation Analysis")
    print("="*80)

    exp_dir = ensure_dir(results_dir / "experiment1_degree_community_correlation")

    # Compute edge metrics
    print("Computing edge metrics...")
    df = compute_edge_metrics(G, communities)

    # Save detailed data
    df.to_csv(exp_dir / f"detailed_stats_{dataset_name}.csv", index=False)

    # Compute statistics for each metric
    metrics_to_analyze = [
        'dspar_score',
        'degree_product',
        'null_model_expected',
        'modularity_contribution',
        'harmonic_mean_degree'
    ]

    intra_df = df[df['is_intra'] == True]
    inter_df = df[df['is_intra'] == False]

    print(f"\nEdge counts: {len(intra_df):,} intra, {len(inter_df):,} inter")

    stats_results = []
    for metric in metrics_to_analyze:
        stat = compute_statistics(
            intra_df[metric].values,
            inter_df[metric].values,
            metric
        )
        stat['dataset'] = dataset_name
        stats_results.append(stat)

    stats_df = pd.DataFrame(stats_results)

    # Print results table
    print("\n" + "-"*120)
    print(f"{'Metric':<25} | {'Intra-Mean':>12} | {'Intra-Std':>10} | {'Inter-Mean':>12} | {'Inter-Std':>10} | {'p-value':>12} | {'Effect Size':>11} | {'AUC':>6}")
    print("-"*120)

    for _, row in stats_df.iterrows():
        pval_str = f"{row['pvalue']:.2e}" if not np.isnan(row['pvalue']) else "N/A"
        auc_str = f"{row['auc']:.4f}" if not np.isnan(row['auc']) else "N/A"
        print(f"{row['metric']:<25} | {row['intra_mean']:>12.6f} | {row['intra_std']:>10.6f} | {row['inter_mean']:>12.6f} | {row['inter_std']:>10.6f} | {pval_str:>12} | {row['cohens_d']:>11.4f} | {auc_str:>6}")

    # Hypothesis test summary
    print("\n" + "-"*80)
    print("HYPOTHESIS TEST SUMMARY:")
    print("-"*80)

    dspar_row = stats_df[stats_df['metric'] == 'dspar_score'].iloc[0]
    print(f"\nDSpar Score (1/d_u + 1/d_v):")
    print(f"  Intra-community mean: {dspar_row['intra_mean']:.6f}")
    print(f"  Inter-community mean: {dspar_row['inter_mean']:.6f}")
    hypothesis_supported = dspar_row['intra_mean'] > dspar_row['inter_mean']
    print(f"  Hypothesis (intra > inter): {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}")
    print(f"  Effect size (Cohen's d): {dspar_row['cohens_d']:.4f}")
    print(f"  p-value: {dspar_row['pvalue']:.2e}")

    dp_row = stats_df[stats_df['metric'] == 'degree_product'].iloc[0]
    print(f"\nDegree Product (d_u * d_v):")
    print(f"  Intra-community mean: {dp_row['intra_mean']:.2f}")
    print(f"  Inter-community mean: {dp_row['inter_mean']:.2f}")
    hypothesis_supported = dp_row['inter_mean'] > dp_row['intra_mean']
    print(f"  Hypothesis (inter > intra): {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}")

    return stats_df, df


# =============================================================================
# Experiment 2: Distribution Visualization
# =============================================================================

def run_experiment2(df, dataset_name, results_dir):
    """
    Experiment 2: Distribution Visualization
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Distribution Visualization")
    print("="*80)

    exp_dir = ensure_dir(results_dir / "experiment2_distributions")

    intra_df = df[df['is_intra'] == True]
    inter_df = df[df['is_intra'] == False]

    # Sample if too large for plotting
    max_samples = 100000
    if len(intra_df) > max_samples:
        intra_sample = intra_df.sample(n=max_samples, random_state=42)
    else:
        intra_sample = intra_df
    if len(inter_df) > max_samples:
        inter_sample = inter_df.sample(n=max_samples, random_state=42)
    else:
        inter_sample = inter_df

    # 1. DSpar score histogram
    print("  Generating DSpar score histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(intra_sample['dspar_score'], bins=50, alpha=0.6, label='Intra-community', density=True)
    ax.hist(inter_sample['dspar_score'], bins=50, alpha=0.6, label='Inter-community', density=True)
    ax.set_xlabel('DSpar Score (1/d_u + 1/d_v)')
    ax.set_ylabel('Density')
    ax.set_title(f'{dataset_name}: DSpar Score Distribution')
    ax.legend()
    ax.axvline(intra_sample['dspar_score'].mean(), color='blue', linestyle='--', alpha=0.7, label='Intra mean')
    ax.axvline(inter_sample['dspar_score'].mean(), color='orange', linestyle='--', alpha=0.7, label='Inter mean')
    plt.tight_layout()
    plt.savefig(exp_dir / f"{dataset_name}_dspar_score_histogram.png", dpi=150)
    plt.close()

    # 2. Degree product histogram (log scale)
    print("  Generating degree product histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use log scale for degree product
    intra_log = np.log10(intra_sample['degree_product'] + 1)
    inter_log = np.log10(inter_sample['degree_product'] + 1)
    ax.hist(intra_log, bins=50, alpha=0.6, label='Intra-community', density=True)
    ax.hist(inter_log, bins=50, alpha=0.6, label='Inter-community', density=True)
    ax.set_xlabel('log10(Degree Product + 1)')
    ax.set_ylabel('Density')
    ax.set_title(f'{dataset_name}: Degree Product Distribution (log scale)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / f"{dataset_name}_degree_product_histogram.png", dpi=150)
    plt.close()

    # 3. Null model expected histogram (log scale)
    print("  Generating null model expected histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    intra_log = np.log10(intra_sample['null_model_expected'] + 1e-10)
    inter_log = np.log10(inter_sample['null_model_expected'] + 1e-10)
    ax.hist(intra_log, bins=50, alpha=0.6, label='Intra-community', density=True)
    ax.hist(inter_log, bins=50, alpha=0.6, label='Inter-community', density=True)
    ax.set_xlabel('log10(Null Model Expected)')
    ax.set_ylabel('Density')
    ax.set_title(f'{dataset_name}: Null Model Expected Distribution (log scale)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / f"{dataset_name}_null_model_histogram.png", dpi=150)
    plt.close()

    # 4. ROC curve using DSpar score to predict intra-community
    if HAS_SKLEARN:
        print("  Generating ROC curve...")
        fig, ax = plt.subplots(figsize=(8, 8))

        labels = [1] * len(intra_sample) + [0] * len(inter_sample)
        scores = list(intra_sample['dspar_score']) + list(inter_sample['dspar_score'])

        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)

        ax.plot(fpr, tpr, label=f'DSpar Score (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{dataset_name}: ROC Curve for Predicting Intra-Community Edges')
        ax.legend()
        plt.tight_layout()
        plt.savefig(exp_dir / f"{dataset_name}_roc_curve.png", dpi=150)
        plt.close()

    # 5. Box plots
    print("  Generating box plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['dspar_score', 'degree_product', 'harmonic_mean_degree']
    titles = ['DSpar Score', 'Degree Product', 'Harmonic Mean Degree']

    plot_df = pd.concat([intra_sample, inter_sample])

    for ax, metric, title in zip(axes, metrics, titles):
        data_to_plot = [
            plot_df[plot_df['is_intra'] == True][metric].values,
            plot_df[plot_df['is_intra'] == False][metric].values
        ]
        bp = ax.boxplot(data_to_plot, labels=['Intra', 'Inter'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightsalmon')
        ax.set_ylabel(metric)
        ax.set_title(title)
        if metric == 'degree_product':
            ax.set_yscale('log')

    plt.suptitle(f'{dataset_name}: Metric Comparison by Edge Type')
    plt.tight_layout()
    plt.savefig(exp_dir / f"{dataset_name}_boxplots.png", dpi=150)
    plt.close()

    print(f"  Plots saved to {exp_dir}")


# =============================================================================
# Experiment 3: Hub Analysis
# =============================================================================

def run_experiment3(G, communities, dataset_name, results_dir, hub_percentiles=[1, 5, 10]):
    """
    Experiment 3: Hub Analysis
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Hub Analysis")
    print("="*80)

    exp_dir = ensure_dir(results_dir / "experiment3_hub_analysis")

    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    results = []

    for percentile in hub_percentiles:
        threshold = np.percentile(degree_values, 100 - percentile)
        hubs = set(node for node, deg in degrees.items() if deg >= threshold)

        print(f"\n  Hub threshold: top {percentile}% (degree >= {threshold:.0f})")
        print(f"  Number of hubs: {len(hubs):,}")

        # Categorize edges
        hub_hub = []
        hub_peripheral = []
        peripheral_peripheral = []

        for u, v in G.edges():
            u_is_hub = u in hubs
            v_is_hub = v in hubs

            comm_u = communities.get(u, -1)
            comm_v = communities.get(v, -1)
            is_inter = (comm_u != comm_v) or (comm_u == -1) or (comm_v == -1)

            if u_is_hub and v_is_hub:
                hub_hub.append(is_inter)
            elif u_is_hub or v_is_hub:
                hub_peripheral.append(is_inter)
            else:
                peripheral_peripheral.append(is_inter)

        # Compute statistics
        def pct_inter(lst):
            return 100 * sum(lst) / len(lst) if len(lst) > 0 else 0

        hh_inter = pct_inter(hub_hub)
        hp_inter = pct_inter(hub_peripheral)
        pp_inter = pct_inter(peripheral_peripheral)

        print(f"\n  {'Edge Type':<25} {'Total':>10} {'% Inter-Comm':>12} {'% Intra-Comm':>12}")
        print(f"  {'-'*60}")
        print(f"  {'Hub-Hub':<25} {len(hub_hub):>10,} {hh_inter:>11.1f}% {100-hh_inter:>11.1f}%")
        print(f"  {'Hub-Peripheral':<25} {len(hub_peripheral):>10,} {hp_inter:>11.1f}% {100-hp_inter:>11.1f}%")
        print(f"  {'Peripheral-Peripheral':<25} {len(peripheral_peripheral):>10,} {pp_inter:>11.1f}% {100-pp_inter:>11.1f}%")

        results.append({
            'dataset': dataset_name,
            'hub_percentile': percentile,
            'hub_threshold_degree': threshold,
            'n_hubs': len(hubs),
            'hub_hub_total': len(hub_hub),
            'hub_hub_inter_pct': hh_inter,
            'hub_peripheral_total': len(hub_peripheral),
            'hub_peripheral_inter_pct': hp_inter,
            'peripheral_peripheral_total': len(peripheral_peripheral),
            'peripheral_peripheral_inter_pct': pp_inter,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(exp_dir / f"hub_edge_type_breakdown_{dataset_name}.csv", index=False)

    # Create visualization
    print("\n  Generating hub analysis plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(hub_percentiles))
    width = 0.25

    hh_vals = [r['hub_hub_inter_pct'] for r in results]
    hp_vals = [r['hub_peripheral_inter_pct'] for r in results]
    pp_vals = [r['peripheral_peripheral_inter_pct'] for r in results]

    ax.bar(x - width, hh_vals, width, label='Hub-Hub', color='red', alpha=0.7)
    ax.bar(x, hp_vals, width, label='Hub-Peripheral', color='orange', alpha=0.7)
    ax.bar(x + width, pp_vals, width, label='Peripheral-Peripheral', color='green', alpha=0.7)

    ax.set_xlabel('Hub Definition (Top X%)')
    ax.set_ylabel('% Inter-Community Edges')
    ax.set_title(f'{dataset_name}: Inter-Community Edge Rate by Edge Type')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top {p}%' for p in hub_percentiles])
    ax.legend()

    plt.tight_layout()
    plt.savefig(exp_dir / f"{dataset_name}_hub_analysis.png", dpi=150)
    plt.close()

    return results_df


# =============================================================================
# Experiment 4: Sparsification Simulation
# =============================================================================

def dspar_sparsify(G, keep_ratio, seed=42):
    """
    DSpar (Degree-based Sparsification).
    Sample edges with probability proportional to 1/d_u + 1/d_v.
    """
    np.random.seed(seed)

    m = G.number_of_edges()
    Q = int(keep_ratio * m)

    if Q >= m:
        return G.copy()

    # Compute sampling probabilities
    edges = list(G.edges())
    probs = []
    for u, v in edges:
        p = 1/G.degree(u) + 1/G.degree(v)
        probs.append(p)

    probs = np.array(probs)
    probs = probs / probs.sum()

    # Sample with replacement
    sampled_indices = np.random.choice(len(edges), size=Q, replace=True, p=probs)
    edge_counts = Counter(sampled_indices)

    # Build sparsified graph with edge weights
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes())

    for idx, count in edge_counts.items():
        u, v = edges[idx]
        weight = count / (Q * probs[idx])  # Unbiased estimator
        if G_sparse.has_edge(u, v):
            G_sparse[u][v]['weight'] += weight
        else:
            G_sparse.add_edge(u, v, weight=weight)

    return G_sparse


def random_sparsify(G, keep_ratio, seed=42):
    """Random edge sampling."""
    np.random.seed(seed)

    edges = list(G.edges())
    m = len(edges)
    Q = int(keep_ratio * m)

    if Q >= m:
        return G.copy()

    indices = np.random.choice(m, size=Q, replace=False)
    sampled_edges = [edges[i] for i in indices]

    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes())
    G_sparse.add_edges_from(sampled_edges)

    return G_sparse


def compute_ratio_metric(G_original, G_sparse, communities):
    """
    Compute Ratio = (fraction inter removed) / (fraction intra removed)
    Ratio < 1 means inter-community edges removed faster
    """
    orig_edges = set((min(u,v), max(u,v)) for u, v in G_original.edges())
    sparse_edges = set((min(u,v), max(u,v)) for u, v in G_sparse.edges())

    intra_orig = 0
    inter_orig = 0
    intra_kept = 0
    inter_kept = 0

    for u, v in orig_edges:
        is_intra = communities.get(u, -1) == communities.get(v, -1) and communities.get(u, -1) != -1
        if is_intra:
            intra_orig += 1
            if (u, v) in sparse_edges:
                intra_kept += 1
        else:
            inter_orig += 1
            if (u, v) in sparse_edges:
                inter_kept += 1

    intra_removed_frac = (intra_orig - intra_kept) / intra_orig if intra_orig > 0 else 0
    inter_removed_frac = (inter_orig - inter_kept) / inter_orig if inter_orig > 0 else 0

    ratio = inter_removed_frac / intra_removed_frac if intra_removed_frac > 0 else float('inf')

    return {
        'intra_orig': intra_orig,
        'inter_orig': inter_orig,
        'intra_kept': intra_kept,
        'inter_kept': inter_kept,
        'intra_removed_frac': intra_removed_frac,
        'inter_removed_frac': inter_removed_frac,
        'ratio': ratio
    }


def communities_to_partition(G, communities):
    """
    Convert communities dict to a proper partition covering all nodes.
    Nodes not in communities dict are assigned to a special 'uncovered' community.
    """
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
    """Compute modularity safely, handling partial community coverage."""
    try:
        partition = communities_to_partition(G, communities)
        return nx.algorithms.community.modularity(G, partition)
    except Exception as e:
        print(f"    Warning: modularity computation failed: {e}")
        return np.nan


def run_experiment4(G, communities, dataset_name, results_dir,
                    keep_ratios=[0.25, 0.5, 0.75], methods=['dspar', 'random']):
    """
    Experiment 4: Sparsification Simulation
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: Sparsification Simulation")
    print("="*80)

    exp_dir = ensure_dir(results_dir / "experiment4_sparsification")

    results = []

    # Get original community quality
    original_modularity = safe_modularity(G, communities)

    for method in methods:
        print(f"\n  Method: {method.upper()}")

        for keep_ratio in keep_ratios:
            print(f"    Keep ratio: {keep_ratio:.0%}")

            # Sparsify
            if method == 'dspar':
                G_sparse = dspar_sparsify(G, keep_ratio)
            else:
                G_sparse = random_sparsify(G, keep_ratio)

            # Compute ratio metric
            ratio_info = compute_ratio_metric(G, G_sparse, communities)

            # Run community detection on sparse graph
            sparse_communities = run_leiden_nx(G_sparse)

            # Compute modularity and NMI
            sparse_modularity = safe_modularity(G_sparse, sparse_communities)

            if HAS_SKLEARN:
                # Compare to original communities
                nodes = sorted(set(communities.keys()) & set(sparse_communities.keys()))
                orig_labels = [communities[n] for n in nodes]
                sparse_labels = [sparse_communities[n] for n in nodes]
                nmi = normalized_mutual_info_score(orig_labels, sparse_labels)
                ari = adjusted_rand_score(orig_labels, sparse_labels)
            else:
                nmi = np.nan
                ari = np.nan

            print(f"      Ratio: {ratio_info['ratio']:.4f}")
            print(f"      NMI vs original: {nmi:.4f}")
            print(f"      Modularity: {sparse_modularity:.4f}")

            results.append({
                'dataset': dataset_name,
                'method': method,
                'keep_ratio': keep_ratio,
                'ratio': ratio_info['ratio'],
                'intra_removed_frac': ratio_info['intra_removed_frac'],
                'inter_removed_frac': ratio_info['inter_removed_frac'],
                'n_edges_sparse': G_sparse.number_of_edges(),
                'modularity_original': original_modularity,
                'modularity_sparse': sparse_modularity,
                'nmi': nmi,
                'ari': ari,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(exp_dir / f"sparsification_results_{dataset_name}.csv", index=False)

    # Print summary table
    print("\n" + "-"*100)
    print(f"{'Method':<10} {'Keep%':>8} {'Ratio':>8} {'NMI':>8} {'Mod_orig':>10} {'Mod_sparse':>10}")
    print("-"*100)
    for _, row in results_df.iterrows():
        print(f"{row['method']:<10} {row['keep_ratio']*100:>7.0f}% {row['ratio']:>8.4f} {row['nmi']:>8.4f} {row['modularity_original']:>10.4f} {row['modularity_sparse']:>10.4f}")

    # Create visualization
    print("\n  Generating sparsification comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Ratio by method and keep_ratio
    for method in methods:
        method_df = results_df[results_df['method'] == method]
        axes[0].plot(method_df['keep_ratio'], method_df['ratio'], 'o-', label=method.upper())
    axes[0].axhline(y=1.0, color='gray', linestyle='--', label='No preference')
    axes[0].set_xlabel('Keep Ratio')
    axes[0].set_ylabel('Ratio (inter removed / intra removed)')
    axes[0].set_title(f'{dataset_name}: Sparsification Ratio')
    axes[0].legend()

    # Plot 2: NMI by method and keep_ratio
    for method in methods:
        method_df = results_df[results_df['method'] == method]
        axes[1].plot(method_df['keep_ratio'], method_df['nmi'], 'o-', label=method.upper())
    axes[1].set_xlabel('Keep Ratio')
    axes[1].set_ylabel('NMI (vs original communities)')
    axes[1].set_title(f'{dataset_name}: Community Detection Quality')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(exp_dir / f"{dataset_name}_sparsification_comparison.png", dpi=150)
    plt.close()

    return results_df


# =============================================================================
# Experiment 5: Modularity-Preserving Analysis
# =============================================================================

def run_experiment5(G, communities, dataset_name, results_dir, removal_fractions=None):
    """
    Experiment 5: Modularity-Preserving Analysis
    Remove edges by degree product vs random, compare modularity preservation.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 5: Modularity-Preserving Analysis")
    print("="*80)

    exp_dir = ensure_dir(results_dir / "experiment5_modularity_preservation")

    if removal_fractions is None:
        removal_fractions = np.arange(0.1, 1.0, 0.1)

    # Convert communities to partition format (covering all nodes)
    partition = communities_to_partition(G, communities)

    # Original modularity
    original_mod = safe_modularity(G, communities)
    print(f"  Original modularity: {original_mod:.4f}")

    # Compute degree product for all edges
    edges = list(G.edges())
    degree_products = [(u, v, G.degree(u) * G.degree(v)) for u, v in edges]
    degree_products.sort(key=lambda x: -x[2])  # Sort descending

    results = []

    for frac in removal_fractions:
        n_remove = int(frac * len(edges))

        # Degree-based removal (remove highest degree product edges)
        edges_to_keep_degree = set((min(u,v), max(u,v)) for u, v, _ in degree_products[n_remove:])
        G_degree = nx.Graph()
        G_degree.add_nodes_from(G.nodes())
        G_degree.add_edges_from(edges_to_keep_degree)

        # Random removal
        np.random.seed(42)
        indices = np.random.choice(len(edges), size=len(edges)-n_remove, replace=False)
        edges_to_keep_random = set((min(edges[i][0], edges[i][1]), max(edges[i][0], edges[i][1])) for i in indices)
        G_random = nx.Graph()
        G_random.add_nodes_from(G.nodes())
        G_random.add_edges_from(edges_to_keep_random)

        # Compute modularity on remaining graph with original partition
        try:
            partition_degree = communities_to_partition(G_degree, communities)
            mod_degree = nx.algorithms.community.modularity(G_degree, partition_degree)
        except Exception:
            mod_degree = np.nan

        try:
            partition_random = communities_to_partition(G_random, communities)
            mod_random = nx.algorithms.community.modularity(G_random, partition_random)
        except Exception:
            mod_random = np.nan

        print(f"  Removed {frac*100:.0f}%: Degree-based mod={mod_degree:.4f}, Random mod={mod_random:.4f}")

        results.append({
            'dataset': dataset_name,
            'fraction_removed': frac,
            'modularity_degree_removal': mod_degree,
            'modularity_random_removal': mod_random,
            'original_modularity': original_mod,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(exp_dir / f"modularity_preservation_{dataset_name}.csv", index=False)

    # Create visualization
    print("\n  Generating modularity preservation plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['fraction_removed'] * 100, results_df['modularity_degree_removal'],
            'o-', label='Degree-based removal', color='blue')
    ax.plot(results_df['fraction_removed'] * 100, results_df['modularity_random_removal'],
            'o-', label='Random removal', color='red')
    ax.axhline(y=original_mod, color='green', linestyle='--', label=f'Original ({original_mod:.4f})')

    ax.set_xlabel('Fraction of Edges Removed (%)')
    ax.set_ylabel('Modularity')
    ax.set_title(f'{dataset_name}: Modularity vs Edge Removal Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / f"{dataset_name}_modularity_preservation.png", dpi=150)
    plt.close()

    return results_df


# =============================================================================
# Experiment 6: Eigenvector Analysis
# =============================================================================

def run_experiment6(G, communities, dataset_name, results_dir):
    """
    Experiment 6: Eigenvector Analysis
    Compute IPR and correlation with ground truth for original and sparsified.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 6: Eigenvector Analysis")
    print("="*80)

    exp_dir = ensure_dir(results_dir / "experiment6_eigenvector_analysis")

    # Limit to smaller graphs for eigenvector analysis
    if G.number_of_nodes() > 50000:
        print(f"  Skipping eigenvector analysis (graph too large: {G.number_of_nodes()} nodes)")
        return None

    def compute_eigenvector_metrics(graph, name):
        """Compute normalized Laplacian eigenvectors and metrics."""
        n = graph.number_of_nodes()

        # Get largest connected component
        if not nx.is_connected(graph):
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()
            n = graph.number_of_nodes()

        if n < 10:
            return None

        # Build normalized Laplacian
        A = nx.adjacency_matrix(graph).astype(float)
        degrees = np.array(A.sum(axis=1)).flatten()

        # Handle zero degrees
        degrees[degrees == 0] = 1

        D_inv_sqrt = csr_matrix(np.diag(1.0 / np.sqrt(degrees)))
        L_norm = csr_matrix(np.eye(n)) - D_inv_sqrt @ A @ D_inv_sqrt

        # Compute eigenvectors (find smallest eigenvalues)
        try:
            eigenvalues, eigenvectors = eigsh(L_norm, k=min(10, n-1), which='SM')
        except Exception as e:
            print(f"    Eigenvalue computation failed: {e}")
            return None

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Second eigenvector (Fiedler vector) - index 1 since 0 is constant
        v2 = eigenvectors[:, 1]

        # IPR (Inverse Participation Ratio)
        # IPR close to 1 = localized (bad)
        # IPR close to 1/n = delocalized (good)
        ipr = np.sum(v2**4) / (np.sum(v2**2)**2)

        print(f"    {name}: IPR = {ipr:.6f} (1/n = {1/n:.6f})")

        return {
            'name': name,
            'n_nodes': n,
            'ipr': ipr,
            'inv_n': 1/n,
            'spectral_gap': eigenvalues[1] if len(eigenvalues) > 1 else 0,
        }

    results = []

    # Original graph
    print("  Computing eigenvectors for original graph...")
    orig_metrics = compute_eigenvector_metrics(G, 'original')
    if orig_metrics:
        orig_metrics['dataset'] = dataset_name
        results.append(orig_metrics)

    # DSpar sparsified at different levels
    for keep_ratio in [0.5, 0.75]:
        print(f"  Computing eigenvectors for DSpar (keep={keep_ratio})...")
        G_sparse = dspar_sparsify(G, keep_ratio)
        sparse_metrics = compute_eigenvector_metrics(G_sparse, f'dspar_{keep_ratio}')
        if sparse_metrics:
            sparse_metrics['dataset'] = dataset_name
            results.append(sparse_metrics)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(exp_dir / f"eigenvector_analysis_{dataset_name}.csv", index=False)

        # Print summary
        print("\n" + "-"*60)
        print(f"{'Graph':<20} {'Nodes':>10} {'IPR':>12} {'1/n':>12} {'Gap':>10}")
        print("-"*60)
        for _, row in results_df.iterrows():
            print(f"{row['name']:<20} {row['n_nodes']:>10} {row['ipr']:>12.6f} {row['inv_n']:>12.6f} {row['spectral_gap']:>10.4f}")

        return results_df

    return None


# =============================================================================
# Main Runner
# =============================================================================

def run_all_experiments(dataset_name, experiments_to_run=None):
    """Run all experiments for a single dataset."""

    if experiments_to_run is None:
        experiments_to_run = [1, 2, 3, 4, 5, 6]

    results_dir = ensure_dir(RESULTS_BASE / dataset_name)

    print("\n" + "#"*80)
    print(f"# DATASET: {dataset_name}")
    print("#"*80)

    # Load data
    G, communities, has_gt = load_dataset_with_communities(dataset_name)

    all_results = {
        'dataset': dataset_name,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_communities': len(set(communities.values())),
        'has_ground_truth': has_gt,
        'timestamp': datetime.now().isoformat(),
    }

    # Run experiments
    edge_df = None

    if 1 in experiments_to_run:
        stats_df, edge_df = run_experiment1(G, communities, dataset_name, results_dir)
        all_results['experiment1'] = stats_df.to_dict('records')

    if 2 in experiments_to_run:
        if edge_df is None:
            edge_df = compute_edge_metrics(G, communities)
        run_experiment2(edge_df, dataset_name, results_dir)

    if 3 in experiments_to_run:
        hub_df = run_experiment3(G, communities, dataset_name, results_dir,
                                 PARAMETERS['hub_percentiles'])
        all_results['experiment3'] = hub_df.to_dict('records')

    if 4 in experiments_to_run:
        sparse_df = run_experiment4(G, communities, dataset_name, results_dir,
                                    PARAMETERS['keep_ratios'],
                                    PARAMETERS['sparsification_methods'])
        all_results['experiment4'] = sparse_df.to_dict('records')

    if 5 in experiments_to_run:
        mod_df = run_experiment5(G, communities, dataset_name, results_dir)
        all_results['experiment5'] = mod_df.to_dict('records')

    if 6 in experiments_to_run:
        eig_df = run_experiment6(G, communities, dataset_name, results_dir)
        if eig_df is not None:
            all_results['experiment6'] = eig_df.to_dict('records')

    # Save combined results
    with open(results_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def generate_summary_report(all_results, results_dir):
    """Generate markdown summary report."""

    report_path = results_dir / "summary_report.md"

    with open(report_path, 'w') as f:
        f.write("# Degree-Community Correlation Experiments - Summary Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Datasets Analyzed\n\n")
        f.write("| Dataset | Nodes | Edges | Communities | Ground Truth |\n")
        f.write("|---------|-------|-------|-------------|---------------|\n")
        for result in all_results:
            f.write(f"| {result['dataset']} | {result['n_nodes']:,} | {result['n_edges']:,} | {result['n_communities']} | {'Yes' if result['has_ground_truth'] else 'No'} |\n")

        f.write("\n## Experiment 1: Degree-Community Correlation\n\n")
        f.write("Testing hypothesis: Inter-community edges have lower DSpar scores (higher degree products).\n\n")

        for result in all_results:
            if 'experiment1' in result:
                f.write(f"\n### {result['dataset']}\n\n")
                exp1 = result['experiment1']
                dspar = next((r for r in exp1 if r['metric'] == 'dspar_score'), None)
                if dspar:
                    f.write(f"- DSpar Score: Intra mean = {dspar['intra_mean']:.6f}, Inter mean = {dspar['inter_mean']:.6f}\n")
                    f.write(f"- Effect size (Cohen's d): {dspar['cohens_d']:.4f}\n")
                    f.write(f"- AUC: {dspar['auc']:.4f}\n")
                    supported = dspar['intra_mean'] > dspar['inter_mean']
                    f.write(f"- Hypothesis supported: **{'YES' if supported else 'NO'}**\n")

        f.write("\n## Key Findings\n\n")
        f.write("1. **Hypothesis validation**: [To be filled based on results]\n")
        f.write("2. **Effect sizes**: [To be filled based on results]\n")
        f.write("3. **Hub behavior**: [To be filled based on results]\n")

    print(f"\nSummary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Degree-Community Correlation Experiments')
    parser.add_argument('--datasets', nargs='+',
                        default=['cit-HepTh', 'cit-HepPh', 'com-DBLP', 'com-Youtube'],
                        help='Datasets to analyze (ordered smallest to largest)')
    parser.add_argument('--experiments', nargs='+', type=int, default=None,
                        help='Experiments to run (1-6, default: all)')

    args = parser.parse_args()

    experiments = args.experiments if args.experiments else [1, 2, 3, 4, 5, 6]

    print("="*80)
    print("DEGREE-COMMUNITY CORRELATION EXPERIMENTS")
    print("="*80)
    print(f"\nDatasets: {args.datasets}")
    print(f"Experiments: {experiments}")
    print(f"Results will be saved to: {RESULTS_BASE}")

    ensure_dir(RESULTS_BASE)

    all_results = []
    for dataset in args.datasets:
        try:
            result = run_all_experiments(dataset, experiments)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR processing {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary report
    if all_results:
        generate_summary_report(all_results, RESULTS_BASE)

        # Save combined results
        with open(RESULTS_BASE / "all_datasets_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved to: {RESULTS_BASE}")


if __name__ == '__main__':
    main()
