#!/usr/bin/env python3
"""
LFR Benchmark Experiment for Spectral Sparsification Study

Tests how spectral sparsification affects community detection on synthetic
LFR benchmark graphs with known ground truth communities.

Key hypotheses:
1. Sparsification helps more when original GT_Mod is low (high μ)
2. Sparsification helps more on denser graphs (high k_avg)
3. GT_Mod increases with sparsification until graph fragments
4. Optimal ε depends on graph density

Usage:
    python experiments/lfr_experiment.py
    python experiments/lfr_experiment.py --n 5000 --repeats 3
    python experiments/lfr_experiment.py --mu 0.3 --k_avg 25
"""

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import leidenalg
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from utils import (
    generate_lfr,
    edges_to_adjacency,
    adjacency_to_igraph,
    count_connected_components,
    spectral_sparsify_direct,
    analyze_ground_truth_edge_preservation,
    compute_ground_truth_modularity,
    PROJECT_ROOT,
)


# =============================================================================
# Configuration
# =============================================================================

# LFR fixed parameters
TAU1 = 2.5  # Degree distribution exponent
TAU2 = 1.5  # Community size distribution exponent

# Parameter sweeps
MU_VALUES = [0.1, 0.3, 0.5, 0.7]  # Mixing parameter
K_AVG_VALUES = [15, 25, 50]  # Average degree (15 instead of 10 for stability)

# Community size bounds (adjusted per k_avg for constraint compatibility)
def get_community_bounds(k_avg):
    """Get min/max community size based on average degree."""
    # For lower k_avg, need smaller min_community
    if k_avg <= 15:
        return 10, 50
    elif k_avg <= 30:
        return 20, 100
    else:
        return 30, 150
EPSILON_VALUES = [0.5, 1.0, 2.0, 3.0, 4.0]

# Experiment settings
DEFAULT_N = 1000  # Number of nodes
DEFAULT_REPEATS = 5  # Repetitions per configuration
BASE_SEED = 42

# Output directories
LFR_RESULTS_DIR = PROJECT_ROOT / "results" / "lfr"


# =============================================================================
# Community Detection
# =============================================================================

def run_leiden(graph, seed=42):
    """Run Leiden community detection."""
    start_time = time.time()
    partition = leidenalg.find_partition(
        graph,
        leidenalg.ModularityVertexPartition,
        seed=seed
    )
    elapsed_time = time.time() - start_time

    labels = [0] * graph.vcount()
    for comm_id, community in enumerate(partition):
        for node in community:
            labels[node] = comm_id

    return labels, len(partition), elapsed_time


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(pred_labels, ground_truth, graph):
    """Compute evaluation metrics."""
    results = {}

    # Number of communities
    results['n_communities'] = len(set(pred_labels))

    # Modularity of predicted communities
    results['modularity'] = graph.modularity(pred_labels)

    # NMI and ARI vs ground truth
    gt_labels = [ground_truth[i] for i in range(len(pred_labels))]
    results['nmi'] = normalized_mutual_info_score(gt_labels, pred_labels)
    results['ari'] = adjusted_rand_score(gt_labels, pred_labels)

    return results


# =============================================================================
# Single Configuration Experiment
# =============================================================================

def run_single_experiment(n, mu, k_avg, seed):
    """
    Run experiment for a single LFR configuration.

    Returns dict with results for original and all epsilon values.
    """
    results = {
        'n': n,
        'mu': mu,
        'k_avg': k_avg,
        'seed': seed,
        'configs': {}
    }

    # Get community bounds based on k_avg
    min_community, max_community = get_community_bounds(k_avg)

    # Generate LFR graph
    try:
        edges, n_nodes, ground_truth = generate_lfr(
            n=n,
            tau1=TAU1,
            tau2=TAU2,
            mu=mu,
            average_degree=k_avg,
            min_community=min_community,
            max_community=max_community,
            seed=seed
        )
    except Exception as e:
        results['error'] = f"LFR generation failed: {e}"
        return results

    n_edges_original = len(edges) // 2
    results['n_edges'] = n_edges_original
    results['n_communities_gt'] = len(set(ground_truth.values()))

    # Build original graph
    A_original = edges_to_adjacency(edges, n_nodes)
    g_original = adjacency_to_igraph(A_original)
    n_cc_original = count_connected_components(A_original)

    # Ground truth modularity on original graph
    gt_mod_original = compute_ground_truth_modularity(g_original, ground_truth)
    results['gt_modularity_original'] = gt_mod_original

    # Run Leiden on original
    labels_original, n_comm, leiden_time = run_leiden(g_original, seed=seed)
    metrics_original = compute_metrics(labels_original, ground_truth, g_original)

    results['configs']['original'] = {
        'n_edges': n_edges_original,
        'edge_ratio': 1.0,
        'n_cc': n_cc_original,
        'gt_modularity': gt_mod_original,
        'metrics': metrics_original,
        'leiden_time': leiden_time
    }

    # Spectral sparsification for each epsilon
    for eps in EPSILON_VALUES:
        try:
            sparsified_edges, sparsify_time = spectral_sparsify_direct(edges, n_nodes, eps)
            n_edges_sparse = len(sparsified_edges) // 2
            edge_ratio = n_edges_sparse / n_edges_original

            A_sparse = edges_to_adjacency(sparsified_edges, n_nodes)
            g_sparse = adjacency_to_igraph(A_sparse)
            n_cc_sparse = count_connected_components(A_sparse)

            # Ground truth modularity on sparse graph
            gt_mod_sparse = compute_ground_truth_modularity(g_sparse, ground_truth)

            # Ground truth edge preservation
            gt_edge_pres = analyze_ground_truth_edge_preservation(edges, sparsified_edges, ground_truth)

            # Run Leiden on sparse graph
            labels_sparse, _, leiden_time_sparse = run_leiden(g_sparse, seed=seed)
            metrics_sparse = compute_metrics(labels_sparse, ground_truth, g_sparse)

            # Community similarity (original Leiden vs sparse Leiden)
            comm_sim = normalized_mutual_info_score(labels_original, labels_sparse)

            results['configs'][f'spectral_eps{eps}'] = {
                'n_edges': n_edges_sparse,
                'edge_ratio': edge_ratio,
                'n_cc': n_cc_sparse,
                'gt_modularity': gt_mod_sparse,
                'metrics': metrics_sparse,
                'sparsify_time': sparsify_time,
                'leiden_time': leiden_time_sparse,
                'gt_edge_preservation': gt_edge_pres,
                'community_similarity': comm_sim
            }

        except Exception as e:
            results['configs'][f'spectral_eps{eps}'] = {'error': str(e)}

    return results


# =============================================================================
# Aggregation Functions
# =============================================================================

def aggregate_results(all_results):
    """Aggregate results across repetitions for same (mu, k_avg) configuration."""
    aggregated = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        if 'error' in result:
            continue

        key = (result['mu'], result['k_avg'])

        for config_name, config in result['configs'].items():
            if 'error' in config:
                continue

            aggregated[key][config_name].append({
                'edge_ratio': config['edge_ratio'],
                'gt_modularity': config['gt_modularity'],
                'nmi': config['metrics']['nmi'],
                'ari': config['metrics']['ari'],
                'n_cc': config['n_cc'],
                'community_similarity': config.get('community_similarity'),
                'gt_ratio': config.get('gt_edge_preservation', {}).get('gt_preservation_ratio')
            })

    # Compute mean and std for each metric
    summary = {}
    for key, configs in aggregated.items():
        mu, k_avg = key
        summary[(mu, k_avg)] = {}

        for config_name, runs in configs.items():
            n_runs = len(runs)
            if n_runs == 0:
                continue

            summary[(mu, k_avg)][config_name] = {
                'n_runs': n_runs,
                'edge_ratio': np.mean([r['edge_ratio'] for r in runs]),
                'gt_modularity_mean': np.mean([r['gt_modularity'] for r in runs if r['gt_modularity'] is not None]),
                'gt_modularity_std': np.std([r['gt_modularity'] for r in runs if r['gt_modularity'] is not None]),
                'nmi_mean': np.mean([r['nmi'] for r in runs]),
                'nmi_std': np.std([r['nmi'] for r in runs]),
                'ari_mean': np.mean([r['ari'] for r in runs]),
                'ari_std': np.std([r['ari'] for r in runs]),
                'n_cc_mean': np.mean([r['n_cc'] for r in runs]),
            }

            # Optional metrics
            comm_sims = [r['community_similarity'] for r in runs if r['community_similarity'] is not None]
            if comm_sims:
                summary[(mu, k_avg)][config_name]['comm_sim_mean'] = np.mean(comm_sims)
                summary[(mu, k_avg)][config_name]['comm_sim_std'] = np.std(comm_sims)

            gt_ratios = [r['gt_ratio'] for r in runs if r['gt_ratio'] is not None]
            if gt_ratios:
                summary[(mu, k_avg)][config_name]['gt_ratio_mean'] = np.mean(gt_ratios)
                summary[(mu, k_avg)][config_name]['gt_ratio_std'] = np.std(gt_ratios)

    return summary


# =============================================================================
# Output Formatting
# =============================================================================

def print_config_table(mu, k_avg, configs, n_nodes):
    """Print results table for a single (mu, k_avg) configuration."""
    print(f"\nLFR RESULTS: n={n_nodes}, μ={mu}, k_avg={k_avg}")
    print("=" * 100)
    print(f"{'Config':<15} {'Edge%':>8} {'GT_Mod':>10} {'NMI':>12} {'ARI':>12} {'GT_Ratio':>10} {'CommSim':>10}")
    print("-" * 100)

    for config_name in ['original'] + [f'spectral_eps{eps}' for eps in EPSILON_VALUES]:
        if config_name not in configs:
            continue

        c = configs[config_name]
        edge_pct = c['edge_ratio'] * 100

        gt_mod_str = f"{c['gt_modularity_mean']:.3f}±{c['gt_modularity_std']:.3f}"
        nmi_str = f"{c['nmi_mean']:.3f}±{c['nmi_std']:.3f}"
        ari_str = f"{c['ari_mean']:.3f}±{c['ari_std']:.3f}"

        gt_ratio_str = "-"
        if 'gt_ratio_mean' in c:
            gt_ratio_str = f"{c['gt_ratio_mean']:.3f}"

        comm_sim_str = "-"
        if 'comm_sim_mean' in c:
            comm_sim_str = f"{c['comm_sim_mean']:.3f}"

        print(f"{config_name:<15} {edge_pct:>7.1f}% {gt_mod_str:>10} {nmi_str:>12} {ari_str:>12} {gt_ratio_str:>10} {comm_sim_str:>10}")

    print("=" * 100)


def print_summary_heatmap(summary, metric='nmi', n_nodes=1000):
    """Print summary heatmap for a metric across (mu, k_avg) combinations."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {metric.upper()} improvement (best ε vs original) - n={n_nodes}")
    print("=" * 80)

    # Header
    print(f"{'μ \\ k_avg':<10}", end="")
    for k_avg in K_AVG_VALUES:
        print(f"{k_avg:>15}", end="")
    print()
    print("-" * (10 + 15 * len(K_AVG_VALUES)))

    for mu in MU_VALUES:
        print(f"{mu:<10}", end="")
        for k_avg in K_AVG_VALUES:
            key = (mu, k_avg)
            if key not in summary:
                print(f"{'N/A':>15}", end="")
                continue

            configs = summary[key]
            if 'original' not in configs:
                print(f"{'N/A':>15}", end="")
                continue

            original_val = configs['original'][f'{metric}_mean']

            # Find best epsilon
            best_improvement = 0
            best_eps = None
            for eps in EPSILON_VALUES:
                config_name = f'spectral_eps{eps}'
                if config_name in configs:
                    val = configs[config_name][f'{metric}_mean']
                    improvement = val - original_val
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_eps = eps

            if best_eps is not None:
                print(f"{best_improvement:>+.3f} (ε={best_eps})", end="")
            else:
                print(f"{'0.000':>15}", end="")
        print()

    print("=" * 80)


def print_optimal_epsilon_heatmap(summary, n_nodes=1000):
    """Print heatmap of optimal epsilon for each (mu, k_avg) combination."""
    print(f"\n{'='*60}")
    print(f"OPTIMAL ε (best NMI) - n={n_nodes}")
    print("=" * 60)

    # Header
    print(f"{'μ \\ k_avg':<10}", end="")
    for k_avg in K_AVG_VALUES:
        print(f"{k_avg:>12}", end="")
    print()
    print("-" * (10 + 12 * len(K_AVG_VALUES)))

    for mu in MU_VALUES:
        print(f"{mu:<10}", end="")
        for k_avg in K_AVG_VALUES:
            key = (mu, k_avg)
            if key not in summary:
                print(f"{'N/A':>12}", end="")
                continue

            configs = summary[key]

            # Find best epsilon by NMI
            best_nmi = 0
            best_eps = "original"
            for config_name, c in configs.items():
                if c['nmi_mean'] > best_nmi:
                    best_nmi = c['nmi_mean']
                    if config_name == 'original':
                        best_eps = "original"
                    else:
                        best_eps = config_name.replace('spectral_eps', 'ε=')

            print(f"{best_eps:>12}", end="")
        print()

    print("=" * 60)


def generate_report(summary, all_results, n_nodes):
    """Generate full text report."""
    lines = []
    lines.append("=" * 120)
    lines.append("LFR BENCHMARK EXPERIMENT - SPECTRAL SPARSIFICATION STUDY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"Configuration:")
    lines.append(f"  Nodes: {n_nodes}")
    lines.append(f"  tau1 (degree exponent): {TAU1}")
    lines.append(f"  tau2 (community exponent): {TAU2}")
    lines.append(f"  Community bounds: adjusted per k_avg (see get_community_bounds())")
    lines.append(f"  μ values: {MU_VALUES}")
    lines.append(f"  k_avg values: {K_AVG_VALUES}")
    lines.append(f"  ε values: {EPSILON_VALUES}")
    lines.append("")

    # Per-configuration tables
    for mu in MU_VALUES:
        for k_avg in K_AVG_VALUES:
            key = (mu, k_avg)
            if key not in summary:
                continue

            configs = summary[key]
            lines.append("")
            lines.append(f"{'='*100}")
            lines.append(f"LFR: n={n_nodes}, μ={mu}, k_avg={k_avg}")
            lines.append("=" * 100)
            lines.append(f"{'Config':<15} {'Edge%':>8} {'GT_Mod':>12} {'NMI':>12} {'ARI':>12} {'GT_Ratio':>10} {'CommSim':>10}")
            lines.append("-" * 100)

            for config_name in ['original'] + [f'spectral_eps{eps}' for eps in EPSILON_VALUES]:
                if config_name not in configs:
                    continue

                c = configs[config_name]
                edge_pct = c['edge_ratio'] * 100

                gt_mod_str = f"{c['gt_modularity_mean']:.3f}±{c['gt_modularity_std']:.3f}"
                nmi_str = f"{c['nmi_mean']:.3f}±{c['nmi_std']:.3f}"
                ari_str = f"{c['ari_mean']:.3f}±{c['ari_std']:.3f}"

                gt_ratio_str = "-"
                if 'gt_ratio_mean' in c:
                    gt_ratio_str = f"{c['gt_ratio_mean']:.3f}"

                comm_sim_str = "-"
                if 'comm_sim_mean' in c:
                    comm_sim_str = f"{c['comm_sim_mean']:.3f}"

                lines.append(f"{config_name:<15} {edge_pct:>7.1f}% {gt_mod_str:>12} {nmi_str:>12} {ari_str:>12} {gt_ratio_str:>10} {comm_sim_str:>10}")

    # Summary heatmaps
    lines.append("")
    lines.append("=" * 120)
    lines.append("SUMMARY: NMI IMPROVEMENT (best ε vs original)")
    lines.append("=" * 120)
    lines.append("")

    # NMI improvement heatmap
    lines.append(f"{'μ \\ k_avg':<10}" + "".join(f"{k:>15}" for k in K_AVG_VALUES))
    lines.append("-" * (10 + 15 * len(K_AVG_VALUES)))

    for mu in MU_VALUES:
        row = f"{mu:<10}"
        for k_avg in K_AVG_VALUES:
            key = (mu, k_avg)
            if key not in summary or 'original' not in summary[key]:
                row += f"{'N/A':>15}"
                continue

            configs = summary[key]
            original_nmi = configs['original']['nmi_mean']

            best_improvement = 0
            best_eps = None
            for eps in EPSILON_VALUES:
                config_name = f'spectral_eps{eps}'
                if config_name in configs:
                    improvement = configs[config_name]['nmi_mean'] - original_nmi
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_eps = eps

            if best_eps is not None:
                row += f"{best_improvement:>+.3f} (ε={best_eps})"
            else:
                row += f"{'0.000':>15}"
        lines.append(row)

    # Optimal epsilon heatmap
    lines.append("")
    lines.append("=" * 120)
    lines.append("OPTIMAL ε (highest NMI)")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"{'μ \\ k_avg':<10}" + "".join(f"{k:>12}" for k in K_AVG_VALUES))
    lines.append("-" * (10 + 12 * len(K_AVG_VALUES)))

    for mu in MU_VALUES:
        row = f"{mu:<10}"
        for k_avg in K_AVG_VALUES:
            key = (mu, k_avg)
            if key not in summary:
                row += f"{'N/A':>12}"
                continue

            configs = summary[key]
            best_nmi = 0
            best_config = "original"
            for config_name, c in configs.items():
                if c['nmi_mean'] > best_nmi:
                    best_nmi = c['nmi_mean']
                    best_config = config_name.replace('spectral_eps', 'ε=') if 'spectral' in config_name else config_name

            row += f"{best_config:>12}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 120)
    lines.append("END OF REPORT")

    return "\n".join(lines)


# =============================================================================
# Main Experiment
# =============================================================================

def run_full_experiment(n_nodes, repeats, mu_values=None, k_avg_values=None):
    """Run full LFR experiment across all parameter combinations."""

    if mu_values is None:
        mu_values = MU_VALUES
    if k_avg_values is None:
        k_avg_values = K_AVG_VALUES

    LFR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_configs = len(mu_values) * len(k_avg_values) * repeats
    current = 0

    print("=" * 70)
    print("LFR BENCHMARK EXPERIMENT")
    print("=" * 70)
    print(f"Nodes: {n_nodes}")
    print(f"μ values: {mu_values}")
    print(f"k_avg values: {k_avg_values}")
    print(f"ε values: {EPSILON_VALUES}")
    print(f"Repetitions: {repeats}")
    print(f"Total configurations: {total_configs}")
    print("=" * 70)

    for mu in mu_values:
        for k_avg in k_avg_values:
            print(f"\n{'='*60}")
            print(f"Running: μ={mu}, k_avg={k_avg}")
            print("=" * 60)

            for rep in range(repeats):
                current += 1
                seed = BASE_SEED + rep * 1000 + int(mu * 100) + k_avg

                print(f"\n  Repetition {rep+1}/{repeats} (seed={seed}) [{current}/{total_configs}]")

                result = run_single_experiment(n_nodes, mu, k_avg, seed)
                all_results.append(result)

                if 'error' not in result:
                    # Quick summary
                    orig = result['configs']['original']
                    print(f"    Original: NMI={orig['metrics']['nmi']:.3f}, GT_Mod={orig['gt_modularity']:.3f}")

                    # Best sparsified
                    best_nmi = orig['metrics']['nmi']
                    best_eps = None
                    for eps in EPSILON_VALUES:
                        config_name = f'spectral_eps{eps}'
                        if config_name in result['configs'] and 'error' not in result['configs'][config_name]:
                            nmi = result['configs'][config_name]['metrics']['nmi']
                            if nmi > best_nmi:
                                best_nmi = nmi
                                best_eps = eps

                    if best_eps is not None:
                        print(f"    Best: ε={best_eps}, NMI={best_nmi:.3f}")
                else:
                    print(f"    ERROR: {result['error']}")

            # Print aggregated results for this (mu, k_avg)
            config_results = [r for r in all_results if r.get('mu') == mu and r.get('k_avg') == k_avg]
            if config_results:
                summary = aggregate_results(config_results)
                if (mu, k_avg) in summary:
                    print_config_table(mu, k_avg, summary[(mu, k_avg)], n_nodes)

    # Final aggregation and summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    summary = aggregate_results(all_results)

    print_summary_heatmap(summary, 'nmi', n_nodes)
    print_optimal_epsilon_heatmap(summary, n_nodes)

    # Save results
    results_file = LFR_RESULTS_DIR / f"lfr_n{n_nodes}_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {results_file}")

    # Save summary
    summary_serializable = {f"{k[0]}_{k[1]}": v for k, v in summary.items()}
    summary_file = LFR_RESULTS_DIR / f"lfr_n{n_nodes}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_serializable, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    # Generate and save report
    report = generate_report(summary, all_results, n_nodes)
    report_file = LFR_RESULTS_DIR / f"lfr_n{n_nodes}_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_file}")

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description='LFR Benchmark Experiment')
    parser.add_argument('--n', type=int, default=DEFAULT_N,
                        help=f'Number of nodes (default: {DEFAULT_N})')
    parser.add_argument('--repeats', type=int, default=DEFAULT_REPEATS,
                        help=f'Number of repetitions (default: {DEFAULT_REPEATS})')
    parser.add_argument('--mu', type=float, nargs='+', default=None,
                        help=f'Mixing parameter values (default: {MU_VALUES})')
    parser.add_argument('--k_avg', type=int, nargs='+', default=None,
                        help=f'Average degree values (default: {K_AVG_VALUES})')

    args = parser.parse_args()

    run_full_experiment(
        n_nodes=args.n,
        repeats=args.repeats,
        mu_values=args.mu,
        k_avg_values=args.k_avg
    )


if __name__ == '__main__':
    main()
