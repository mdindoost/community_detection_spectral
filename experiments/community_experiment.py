#!/usr/bin/env python3
"""
Community Detection Sparsification Experiment

Tests whether spectral graph sparsification preserves community structure
using the Leiden algorithm on SNAP datasets.

Hypothesis: Spectral sparsification preserves inter-community edges at higher
rates than intra-community edges, maintaining community structure.

Usage:
    python experiments/community_experiment.py --datasets email-Eu-core
    python experiments/community_experiment.py --datasets all
"""

import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import leidenalg
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from utils import (
    SNAP_DATASETS,
    RESULTS_DIR,
    load_snap_dataset,
    edges_to_adjacency,
    adjacency_to_igraph,
    count_connected_components,
    spectral_sparsify,
    get_results_dir,
    analyze_edge_preservation,
    analyze_ground_truth_edge_preservation,
    analyze_misclassification,
    compute_ground_truth_modularity,
)


# =============================================================================
# Configuration
# =============================================================================

# Epsilon values to test for all datasets
EPSILON_VALUES = [0.5, 1.0, 2.0, 3.0, 4.0]


def get_epsilon_values(dataset_name):
    """Get epsilon values for a dataset."""
    return EPSILON_VALUES


# =============================================================================
# Community Detection
# =============================================================================

def run_leiden(graph, seed=42):
    """
    Run Leiden community detection.

    Returns:
        tuple: (labels, n_communities, elapsed_time)
            - labels: list of community assignments (one per node)
            - n_communities: number of communities found
            - elapsed_time: time in seconds
    """
    start_time = time.time()
    partition = leidenalg.find_partition(
        graph,
        leidenalg.ModularityVertexPartition,
        seed=seed
    )
    elapsed_time = time.time() - start_time

    # Convert to list of labels
    labels = [0] * graph.vcount()
    for comm_id, community in enumerate(partition):
        for node in community:
            labels[node] = comm_id

    return labels, len(partition), elapsed_time


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(pred_labels, true_labels, pred_graph, original_graph):
    """
    Compute all evaluation metrics.

    Args:
        pred_labels: predicted community labels
        true_labels: ground truth labels (can be None)
        pred_graph: igraph Graph used for prediction
        original_graph: original igraph Graph

    Returns:
        dict with metrics
    """
    results = {}

    # Number of communities
    results['n_communities'] = len(set(pred_labels))

    # Modularity on the graph used for prediction
    results['modularity_pred_graph'] = pred_graph.modularity(pred_labels)

    # Modularity on ORIGINAL graph (key metric!)
    results['modularity_original'] = original_graph.modularity(pred_labels)

    # NMI and ARI (if ground truth available)
    if true_labels is not None:
        valid_nodes = [i for i in range(len(pred_labels)) if i in true_labels]
        if len(valid_nodes) > 0:
            pred_valid = [pred_labels[i] for i in valid_nodes]
            true_valid = [true_labels[i] for i in valid_nodes]

            results['nmi'] = normalized_mutual_info_score(true_valid, pred_valid)
            results['ari'] = adjusted_rand_score(true_valid, pred_valid)
        else:
            results['nmi'] = None
            results['ari'] = None
    else:
        results['nmi'] = None
        results['ari'] = None

    return results


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(dataset_name, epsilon_values=None):
    """Run full experiment on one dataset."""

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print('='*70)

    # Get appropriate epsilon values for this dataset
    if epsilon_values is None:
        epsilon_values = get_epsilon_values(dataset_name)

    results_dir = get_results_dir(dataset_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n[1] Loading dataset...")
    edges, n_nodes, ground_truth = load_snap_dataset(dataset_name)
    n_edges_original = len(edges) // 2

    print(f"    Nodes: {n_nodes}")
    print(f"    Edges (undirected): {n_edges_original}")
    print(f"    Ground truth: {'Yes' if ground_truth else 'No'}")

    # Build original graph
    A_original = edges_to_adjacency(edges, n_nodes)
    g_original = adjacency_to_igraph(A_original)

    n_cc_original = count_connected_components(A_original)
    print(f"    Connected components: {n_cc_original}")

    results = {
        'dataset': dataset_name,
        'n_nodes': n_nodes,
        'n_edges_original': n_edges_original,
        'n_cc_original': n_cc_original,
        'has_ground_truth': ground_truth is not None,
        'configs': {}
    }

    # Run Leiden on original to get baseline metrics AND community labels for edge analysis
    print("\n[2] Running Leiden on original graph...")
    labels_original, n_comm_original, leiden_time_original = run_leiden(g_original)
    metrics_original = compute_metrics(labels_original, ground_truth, g_original, g_original)

    # Compute ground truth modularity on original graph
    gt_modularity_original = compute_ground_truth_modularity(g_original, ground_truth)

    results['configs']['original'] = {
        'n_edges': n_edges_original,
        'edge_ratio': 1.0,
        'n_cc': n_cc_original,
        'metrics': metrics_original,
        'leiden_time': leiden_time_original,
        'gt_modularity': gt_modularity_original
    }

    print(f"    Communities: {metrics_original['n_communities']}")
    print(f"    Modularity: {metrics_original['modularity_original']:.4f}")
    print(f"    Leiden time: {leiden_time_original:.4f}s")
    if metrics_original['nmi'] is not None:
        print(f"    NMI: {metrics_original['nmi']:.4f}")
        print(f"    ARI: {metrics_original['ari']:.4f}")
    if gt_modularity_original is not None:
        print(f"    GT Modularity: {gt_modularity_original:.4f}")

    # Spectral sparsification with edge preservation analysis
    print(f"\n[3] Spectral sparsification (epsilon values: {epsilon_values})...")
    for eps in epsilon_values:
        print(f"\n  epsilon = {eps}:")

        try:
            sparsified_edges, sparsify_time = spectral_sparsify(edges, n_nodes, eps, dataset_name)
            n_edges_sparse = len(sparsified_edges) // 2
            edge_ratio = n_edges_sparse / n_edges_original

            A_sparse = edges_to_adjacency(sparsified_edges, n_nodes)
            g_sparse = adjacency_to_igraph(A_sparse)
            n_cc_sparse = count_connected_components(A_sparse)

            print(f"    Edges: {n_edges_sparse} ({edge_ratio*100:.1f}%)")
            print(f"    Connected components: {n_cc_sparse}")
            if sparsify_time is not None:
                print(f"    Sparsification time: {sparsify_time:.4f}s")

            # Edge preservation analysis using original Leiden community labels
            edge_pres = analyze_edge_preservation(edges, sparsified_edges, labels_original)
            print(f"    [Leiden labels] Intra preserved: {edge_pres['intra_preservation_rate']*100:.1f}%, Inter preserved: {edge_pres['inter_preservation_rate']*100:.1f}%, Ratio: {edge_pres['preservation_ratio']:.3f}")

            # Ground truth edge preservation analysis (if available)
            gt_edge_pres = analyze_ground_truth_edge_preservation(edges, sparsified_edges, ground_truth)
            if gt_edge_pres:
                print(f"    [GT labels] Intra preserved: {gt_edge_pres['gt_intra_preservation_rate']*100:.1f}%, Inter preserved: {gt_edge_pres['gt_inter_preservation_rate']*100:.1f}%, Ratio: {gt_edge_pres['gt_preservation_ratio']:.3f}")

            # Run Leiden on sparsified graph
            labels_sparse, _, leiden_time_sparse = run_leiden(g_sparse)
            metrics_sparse = compute_metrics(labels_sparse, ground_truth, g_sparse, g_original)

            # Community similarity: how different are Leiden communities on sparse vs original?
            community_similarity = normalized_mutual_info_score(labels_original, labels_sparse)
            print(f"    Community similarity (NMI original vs sparse): {community_similarity:.4f}")

            # Misclassification analysis (if ground truth available)
            misclass = analyze_misclassification(edges, sparsified_edges, labels_original, ground_truth)
            if misclass:
                print(f"    Misclassification: {misclass['removed_misclassification_rate']*100:.1f}% of removed 'inter' edges were actually GT-intra")

            # Ground truth modularity on sparse graph
            gt_modularity_sparse = compute_ground_truth_modularity(g_sparse, ground_truth)
            if gt_modularity_sparse is not None:
                print(f"    GT Modularity on sparse: {gt_modularity_sparse:.4f} (original: {gt_modularity_original:.4f})")

            results['configs'][f'spectral_eps{eps}'] = {
                'n_edges': n_edges_sparse,
                'edge_ratio': edge_ratio,
                'n_cc': n_cc_sparse,
                'metrics': metrics_sparse,
                'sparsify_time': sparsify_time,
                'leiden_time': leiden_time_sparse,
                'edge_preservation': edge_pres,
                'gt_edge_preservation': gt_edge_pres,
                'community_similarity': community_similarity,
                'misclassification': misclass,
                'gt_modularity': gt_modularity_sparse
            }

            print(f"    Communities: {metrics_sparse['n_communities']}")
            print(f"    Modularity (on original): {metrics_sparse['modularity_original']:.4f}")
            if metrics_sparse['nmi'] is not None:
                print(f"    NMI: {metrics_sparse['nmi']:.4f}")
                print(f"    ARI: {metrics_sparse['ari']:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            results['configs'][f'spectral_eps{eps}'] = {'error': str(e)}

    # Save results
    results_file = results_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    # Print dataset summary table
    print_dataset_summary(results)

    return results


def print_dataset_summary(result):
    """Print summary table for a single dataset."""

    dataset = result['dataset']
    has_gt = result.get('has_ground_truth', False)

    print(f"\n{'='*160}")
    print(f"SUMMARY: {dataset}")
    print("="*160)

    # Header - basic metrics
    print(f"\n{'Config':<20} {'Edge%':>8} {'CC':>6} {'Mod':>8} {'NMI':>8} {'ARI':>8} {'CommSim':>8} {'Intra%':>8} {'Inter%':>8} {'Ratio':>8}")
    print("-"*110)

    for config_name, config in result['configs'].items():
        if 'error' in config:
            continue

        edge_pct = config['edge_ratio'] * 100
        cc = config.get('n_cc', '-')
        mod = config['metrics'].get('modularity_original', '-')
        nmi = config['metrics'].get('nmi')
        ari = config['metrics'].get('ari')
        comm_sim = config.get('community_similarity')

        # Edge preservation stats (Leiden labels)
        edge_pres = config.get('edge_preservation', {})
        intra_pres = edge_pres.get('intra_preservation_rate')
        inter_pres = edge_pres.get('inter_preservation_rate')
        pres_ratio = edge_pres.get('preservation_ratio')

        cc_str = f"{cc:.0f}" if isinstance(cc, (int, float)) else str(cc)
        mod_str = f"{mod:.4f}" if isinstance(mod, float) else str(mod)
        nmi_str = f"{nmi:.4f}" if nmi is not None else "-"
        ari_str = f"{ari:.4f}" if ari is not None else "-"
        comm_sim_str = f"{comm_sim:.4f}" if comm_sim is not None else "-"
        intra_str = f"{intra_pres*100:.1f}%" if intra_pres is not None else "-"
        inter_str = f"{inter_pres*100:.1f}%" if inter_pres is not None else "-"
        ratio_str = f"{pres_ratio:.3f}" if pres_ratio is not None else "-"

        print(f"{config_name:<20} {edge_pct:>7.1f}% {cc_str:>6} {mod_str:>8} {nmi_str:>8} {ari_str:>8} {comm_sim_str:>8} {intra_str:>8} {inter_str:>8} {ratio_str:>8}")

    # Ground truth analysis table (only if has ground truth)
    if has_gt:
        print(f"\n--- Ground Truth Analysis ---")
        print(f"{'Config':<20} {'GT_Intra%':>10} {'GT_Inter%':>10} {'GT_Ratio':>10} {'Misclass%':>10} {'GT_Mod':>10}")
        print("-"*80)

        for config_name, config in result['configs'].items():
            if 'error' in config:
                continue

            gt_pres = config.get('gt_edge_preservation', {})
            gt_intra = gt_pres.get('gt_intra_preservation_rate') if gt_pres else None
            gt_inter = gt_pres.get('gt_inter_preservation_rate') if gt_pres else None
            gt_ratio = gt_pres.get('gt_preservation_ratio') if gt_pres else None

            misclass = config.get('misclassification', {})
            misclass_rate = misclass.get('removed_misclassification_rate') if misclass else None

            gt_mod = config.get('gt_modularity')

            gt_intra_str = f"{gt_intra*100:.1f}%" if gt_intra is not None else "-"
            gt_inter_str = f"{gt_inter*100:.1f}%" if gt_inter is not None else "-"
            gt_ratio_str = f"{gt_ratio:.3f}" if gt_ratio is not None else "-"
            misclass_str = f"{misclass_rate*100:.1f}%" if misclass_rate is not None else "-"
            gt_mod_str = f"{gt_mod:.4f}" if gt_mod is not None else "-"

            print(f"{config_name:<20} {gt_intra_str:>10} {gt_inter_str:>10} {gt_ratio_str:>10} {misclass_str:>10} {gt_mod_str:>10}")

    print("="*160)


def print_summary(all_results):
    """Print summary table across all datasets."""

    print("\n" + "="*160)
    print("SUMMARY (ALL DATASETS)")
    print("="*160)

    # Header
    print(f"\n{'Dataset':<18} {'Config':<20} {'Edge%':>8} {'CC':>6} {'Mod':>8} {'NMI':>8} {'ARI':>8} {'CommSim':>8} {'Intra%':>8} {'Inter%':>8} {'Ratio':>8}")
    print("-"*130)

    for result in all_results:
        dataset = result['dataset']

        for config_name, config in result['configs'].items():
            if 'error' in config:
                continue

            edge_pct = config['edge_ratio'] * 100
            cc = config.get('n_cc', '-')
            mod = config['metrics'].get('modularity_original', '-')
            nmi = config['metrics'].get('nmi')
            ari = config['metrics'].get('ari')
            comm_sim = config.get('community_similarity')

            # Edge preservation stats
            edge_pres = config.get('edge_preservation', {})
            intra_pres = edge_pres.get('intra_preservation_rate')
            inter_pres = edge_pres.get('inter_preservation_rate')
            pres_ratio = edge_pres.get('preservation_ratio')

            cc_str = f"{cc:.0f}" if isinstance(cc, (int, float)) else str(cc)
            mod_str = f"{mod:.4f}" if isinstance(mod, float) else str(mod)
            nmi_str = f"{nmi:.4f}" if nmi is not None else "-"
            ari_str = f"{ari:.4f}" if ari is not None else "-"
            comm_sim_str = f"{comm_sim:.4f}" if comm_sim is not None else "-"
            intra_str = f"{intra_pres*100:.1f}%" if intra_pres is not None else "-"
            inter_str = f"{inter_pres*100:.1f}%" if inter_pres is not None else "-"
            ratio_str = f"{pres_ratio:.3f}" if pres_ratio is not None else "-"

            print(f"{dataset:<18} {config_name:<20} {edge_pct:>7.1f}% {cc_str:>6} {mod_str:>8} {nmi_str:>8} {ari_str:>8} {comm_sim_str:>8} {intra_str:>8} {inter_str:>8} {ratio_str:>8}")

        print("-"*130)

    print("="*160)


def main():
    parser = argparse.ArgumentParser(description='Community Detection Sparsification Experiment')
    parser.add_argument('--datasets', nargs='+', default=['email-Eu-core'],
                        help='Datasets to run (or "all")')
    parser.add_argument('--epsilon', nargs='+', type=float, default=None,
                        help='Override epsilon values (default: auto based on dataset)')

    args = parser.parse_args()

    epsilon_values = args.epsilon  # None means auto-select

    if 'all' in args.datasets:
        datasets = list(SNAP_DATASETS.keys())
    else:
        datasets = args.datasets

    # Validate datasets
    for d in datasets:
        if d not in SNAP_DATASETS:
            print(f"Unknown dataset: {d}")
            print(f"Available: {list(SNAP_DATASETS.keys())}")
            sys.exit(1)

    print("Community Detection Sparsification Experiment")
    print("="*50)
    print(f"Datasets: {datasets}")
    if epsilon_values:
        print(f"Epsilon values (override): {epsilon_values}")
    else:
        print("Epsilon values: auto (sparse datasets get more values)")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []
    for dataset in datasets:
        result = run_experiment(dataset, epsilon_values)
        all_results.append(result)

    # Print combined summary only if multiple datasets
    if len(all_results) > 1:
        print_summary(all_results)

    # Save combined results
    summary_file = RESULTS_DIR / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {summary_file}")


if __name__ == '__main__':
    main()
