#!/usr/bin/env python3
"""
Noisy LFR Benchmark Experiment for Spectral Sparsification Study

Tests the hypothesis that adding random noise to clean LFR graphs makes them
behave like real-world networks under sparsification.

Core Hypothesis:
- Clean LFR: Sparsification hurts (removes signal)
- Noisy LFR: Sparsification helps (removes noise) <- like real data

If this is true, it proves the mechanism is noise removal.

Usage:
    python experiments/noisy_lfr_experiment.py
    python experiments/noisy_lfr_experiment.py --n 5000 --repeats 3
    python experiments/noisy_lfr_experiment.py --noise 0.2 0.3
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
    add_noise_edges,
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

# LFR fixed parameters (for cleaner comparison)
MU = 0.3           # Moderate community structure
K_AVG = 25         # Medium density
TAU1 = 2.5         # Degree distribution exponent
TAU2 = 1.5         # Community size distribution exponent
MIN_COMMUNITY = 20
MAX_COMMUNITY = 100

# Noise levels to test
NOISE_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]  # 0%, 10%, 20%, 30%, 50% extra edges

# Sparsification parameters
EPSILON_VALUES = [0.5, 1.0, 2.0, 3.0, 4.0]

# Experiment settings
DEFAULT_N = 1000
DEFAULT_REPEATS = 5
BASE_SEED = 42

# Output directories
NOISY_LFR_RESULTS_DIR = PROJECT_ROOT / "results" / "noisy_lfr"


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

    # Modularity of predicted communities (Leiden modularity)
    results['leiden_modularity'] = graph.modularity(pred_labels)

    # NMI and ARI vs ground truth
    gt_labels = [ground_truth[i] for i in range(len(pred_labels))]
    results['nmi'] = normalized_mutual_info_score(gt_labels, pred_labels)
    results['ari'] = adjusted_rand_score(gt_labels, pred_labels)

    return results


# =============================================================================
# Single Configuration Experiment
# =============================================================================

def run_single_experiment(n, noise_ratio, seed):
    """
    Run experiment for a single (n, noise_ratio) configuration.

    Returns dict with results for original noisy graph and all epsilon values.
    """
    results = {
        'n': n,
        'noise_ratio': noise_ratio,
        'mu': MU,
        'k_avg': K_AVG,
        'seed': seed,
        'configs': {}
    }

    # Generate clean LFR graph
    try:
        clean_edges, n_nodes, ground_truth = generate_lfr(
            n=n,
            tau1=TAU1,
            tau2=TAU2,
            mu=MU,
            average_degree=K_AVG,
            min_community=MIN_COMMUNITY,
            max_community=MAX_COMMUNITY,
            seed=seed
        )
    except Exception as e:
        results['error'] = f"LFR generation failed: {e}"
        return results

    n_clean_edges = len(clean_edges) // 2
    results['n_clean_edges'] = n_clean_edges
    results['n_communities_gt'] = len(set(ground_truth.values()))

    # Add noise to the graph
    noisy_edges, noise_stats = add_noise_edges(
        clean_edges, n_nodes, ground_truth, noise_ratio, seed=seed
    )
    results['noise_stats'] = noise_stats

    n_noisy_edges = len(noisy_edges) // 2
    results['n_noisy_edges'] = n_noisy_edges

    # Build noisy graph
    A_noisy = edges_to_adjacency(noisy_edges, n_nodes)
    g_noisy = adjacency_to_igraph(A_noisy)
    n_cc_noisy = count_connected_components(A_noisy)

    # Ground truth modularity on noisy graph
    gt_mod_noisy = compute_ground_truth_modularity(g_noisy, ground_truth)
    results['gt_modularity_noisy'] = gt_mod_noisy

    # Run Leiden on noisy graph (baseline)
    labels_noisy, n_comm, leiden_time = run_leiden(g_noisy, seed=seed)
    metrics_noisy = compute_metrics(labels_noisy, ground_truth, g_noisy)

    # Spurious gap = Leiden_Mod - GT_Mod (measures fake structure from noise)
    spurious_gap = metrics_noisy['leiden_modularity'] - gt_mod_noisy if gt_mod_noisy else None

    results['configs']['original'] = {
        'n_edges': n_noisy_edges,
        'edge_ratio': 1.0,
        'n_cc': n_cc_noisy,
        'gt_modularity': gt_mod_noisy,
        'leiden_modularity': metrics_noisy['leiden_modularity'],
        'spurious_gap': spurious_gap,
        'metrics': metrics_noisy,
        'leiden_time': leiden_time
    }

    # Spectral sparsification for each epsilon
    for eps in EPSILON_VALUES:
        try:
            sparsified_edges, sparsify_time = spectral_sparsify_direct(noisy_edges, n_nodes, eps)
            n_edges_sparse = len(sparsified_edges) // 2
            edge_ratio = n_edges_sparse / n_noisy_edges

            A_sparse = edges_to_adjacency(sparsified_edges, n_nodes)
            g_sparse = adjacency_to_igraph(A_sparse)
            n_cc_sparse = count_connected_components(A_sparse)

            # Ground truth modularity on sparse graph
            gt_mod_sparse = compute_ground_truth_modularity(g_sparse, ground_truth)

            # Ground truth edge preservation
            gt_edge_pres = analyze_ground_truth_edge_preservation(noisy_edges, sparsified_edges, ground_truth)

            # Run Leiden on sparse graph
            labels_sparse, _, leiden_time_sparse = run_leiden(g_sparse, seed=seed)
            metrics_sparse = compute_metrics(labels_sparse, ground_truth, g_sparse)

            # Spurious gap on sparse graph
            spurious_gap_sparse = metrics_sparse['leiden_modularity'] - gt_mod_sparse if gt_mod_sparse else None

            # Community similarity (noisy Leiden vs sparse Leiden)
            comm_sim = normalized_mutual_info_score(labels_noisy, labels_sparse)

            results['configs'][f'spectral_eps{eps}'] = {
                'n_edges': n_edges_sparse,
                'edge_ratio': edge_ratio,
                'n_cc': n_cc_sparse,
                'gt_modularity': gt_mod_sparse,
                'leiden_modularity': metrics_sparse['leiden_modularity'],
                'spurious_gap': spurious_gap_sparse,
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
    """Aggregate results across repetitions for same (n, noise_ratio) configuration."""
    aggregated = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        if 'error' in result:
            continue

        key = (result['n'], result['noise_ratio'])

        for config_name, config in result['configs'].items():
            if 'error' in config:
                continue

            aggregated[key][config_name].append({
                'edge_ratio': config['edge_ratio'],
                'gt_modularity': config['gt_modularity'],
                'leiden_modularity': config['leiden_modularity'],
                'spurious_gap': config.get('spurious_gap'),
                'nmi': config['metrics']['nmi'],
                'ari': config['metrics']['ari'],
                'n_cc': config['n_cc'],
                'community_similarity': config.get('community_similarity'),
                'gt_ratio': config.get('gt_edge_preservation', {}).get('gt_preservation_ratio')
            })

    # Compute mean and std for each metric
    summary = {}
    for key, configs in aggregated.items():
        n, noise_ratio = key
        summary[(n, noise_ratio)] = {}

        for config_name, runs in configs.items():
            n_runs = len(runs)
            if n_runs == 0:
                continue

            s = summary[(n, noise_ratio)][config_name] = {
                'n_runs': n_runs,
                'edge_ratio': np.mean([r['edge_ratio'] for r in runs]),
            }

            # GT modularity
            gt_mods = [r['gt_modularity'] for r in runs if r['gt_modularity'] is not None]
            if gt_mods:
                s['gt_modularity_mean'] = np.mean(gt_mods)
                s['gt_modularity_std'] = np.std(gt_mods)

            # Leiden modularity
            leiden_mods = [r['leiden_modularity'] for r in runs]
            s['leiden_modularity_mean'] = np.mean(leiden_mods)
            s['leiden_modularity_std'] = np.std(leiden_mods)

            # Spurious gap
            gaps = [r['spurious_gap'] for r in runs if r['spurious_gap'] is not None]
            if gaps:
                s['spurious_gap_mean'] = np.mean(gaps)
                s['spurious_gap_std'] = np.std(gaps)

            # NMI and ARI
            s['nmi_mean'] = np.mean([r['nmi'] for r in runs])
            s['nmi_std'] = np.std([r['nmi'] for r in runs])
            s['ari_mean'] = np.mean([r['ari'] for r in runs])
            s['ari_std'] = np.std([r['ari'] for r in runs])

            # CC
            s['n_cc_mean'] = np.mean([r['n_cc'] for r in runs])

            # Community similarity
            comm_sims = [r['community_similarity'] for r in runs if r['community_similarity'] is not None]
            if comm_sims:
                s['comm_sim_mean'] = np.mean(comm_sims)
                s['comm_sim_std'] = np.std(comm_sims)

            # GT ratio
            gt_ratios = [r['gt_ratio'] for r in runs if r['gt_ratio'] is not None]
            if gt_ratios:
                s['gt_ratio_mean'] = np.mean(gt_ratios)
                s['gt_ratio_std'] = np.std(gt_ratios)

    return summary


# =============================================================================
# Output Formatting
# =============================================================================

def print_config_table(n, noise_ratio, configs):
    """Print results table for a single (n, noise_ratio) configuration."""
    print(f"\nNOISY LFR RESULTS: n={n}, μ={MU}, k_avg={K_AVG}, noise={noise_ratio*100:.0f}%")
    print("=" * 120)
    print(f"{'Config':<15} {'Edge%':>8} {'GT_Mod':>12} {'Leiden_Mod':>12} {'Gap':>10} {'NMI':>12} {'ARI':>12} {'GT_Ratio':>10}")
    print("-" * 120)

    for config_name in ['original'] + [f'spectral_eps{eps}' for eps in EPSILON_VALUES]:
        if config_name not in configs:
            continue

        c = configs[config_name]
        edge_pct = c['edge_ratio'] * 100

        gt_mod_str = f"{c['gt_modularity_mean']:.3f}±{c['gt_modularity_std']:.3f}" if 'gt_modularity_mean' in c else "-"
        leiden_mod_str = f"{c['leiden_modularity_mean']:.3f}±{c['leiden_modularity_std']:.3f}"
        gap_str = f"{c['spurious_gap_mean']:.3f}" if 'spurious_gap_mean' in c else "-"
        nmi_str = f"{c['nmi_mean']:.3f}±{c['nmi_std']:.3f}"
        ari_str = f"{c['ari_mean']:.3f}±{c['ari_std']:.3f}"
        gt_ratio_str = f"{c['gt_ratio_mean']:.3f}" if 'gt_ratio_mean' in c else "-"

        print(f"{config_name:<15} {edge_pct:>7.1f}% {gt_mod_str:>12} {leiden_mod_str:>12} {gap_str:>10} {nmi_str:>12} {ari_str:>12} {gt_ratio_str:>10}")

    print("=" * 120)


def print_noise_comparison(summary, n_nodes):
    """Print comparison across noise levels."""
    print(f"\n{'='*100}")
    print(f"NOISE LEVEL COMPARISON: n={n_nodes}, μ={MU}, k_avg={K_AVG}")
    print(f"{'='*100}")
    print(f"\n{'Noise%':<10} {'Config':<15} {'GT_Mod':>10} {'Leiden_Mod':>10} {'Gap':>8} {'NMI':>10} {'GT_Ratio':>10}")
    print("-" * 80)

    for noise_ratio in NOISE_RATIOS:
        key = (n_nodes, noise_ratio)
        if key not in summary:
            continue

        configs = summary[key]

        # Show original and best epsilon (eps=2.0 typically)
        for config_name in ['original', 'spectral_eps2.0']:
            if config_name not in configs:
                continue

            c = configs[config_name]
            gt_mod = c.get('gt_modularity_mean', 0)
            leiden_mod = c.get('leiden_modularity_mean', 0)
            gap = c.get('spurious_gap_mean', 0)
            nmi = c.get('nmi_mean', 0)
            gt_ratio = c.get('gt_ratio_mean')

            gt_ratio_str = f"{gt_ratio:.3f}" if gt_ratio else "-"

            print(f"{noise_ratio*100:>5.0f}%    {config_name:<15} {gt_mod:>10.3f} {leiden_mod:>10.3f} {gap:>8.3f} {nmi:>10.3f} {gt_ratio_str:>10}")

        print("-" * 80)

    print("=" * 100)


def print_improvement_summary(summary, n_nodes):
    """Print summary of improvements from sparsification across noise levels."""
    print(f"\n{'='*90}")
    print(f"IMPROVEMENT SUMMARY (eps=2.0 vs original): n={n_nodes}")
    print(f"{'='*90}")
    print(f"\n{'Noise%':<10} {'GT_Mod Δ':>12} {'NMI Δ':>12} {'Gap Δ':>12} {'GT_Ratio':>12} {'Helps?':>10}")
    print("-" * 70)

    for noise_ratio in NOISE_RATIOS:
        key = (n_nodes, noise_ratio)
        if key not in summary:
            continue

        configs = summary[key]
        if 'original' not in configs or 'spectral_eps2.0' not in configs:
            continue

        orig = configs['original']
        sparse = configs['spectral_eps2.0']

        gt_mod_delta = sparse.get('gt_modularity_mean', 0) - orig.get('gt_modularity_mean', 0)
        nmi_delta = sparse.get('nmi_mean', 0) - orig.get('nmi_mean', 0)
        gap_delta = sparse.get('spurious_gap_mean', 0) - orig.get('spurious_gap_mean', 0)
        gt_ratio = sparse.get('gt_ratio_mean')

        gt_ratio_str = f"{gt_ratio:.3f}" if gt_ratio else "-"

        # Does sparsification help?
        helps = "YES" if nmi_delta > 0.01 else ("~" if abs(nmi_delta) < 0.01 else "NO")

        print(f"{noise_ratio*100:>5.0f}%    {gt_mod_delta:>+12.3f} {nmi_delta:>+12.3f} {gap_delta:>+12.3f} {gt_ratio_str:>12} {helps:>10}")

    print("-" * 70)
    print("\nExpected pattern:")
    print("  - Clean (0%): GT_Mod decreases, NMI stable/decreases, GT_Ratio > 1")
    print("  - Noisy (20%+): GT_Mod increases, NMI improves, GT_Ratio < 1")
    print("=" * 90)


def generate_report(summary, all_results, n_nodes):
    """Generate full text report."""
    lines = []
    lines.append("=" * 120)
    lines.append("NOISY LFR BENCHMARK EXPERIMENT - SPECTRAL SPARSIFICATION STUDY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 120)
    lines.append("")
    lines.append("HYPOTHESIS:")
    lines.append("  Clean LFR: Sparsification hurts (removes signal)")
    lines.append("  Noisy LFR: Sparsification helps (removes noise) <- like real data")
    lines.append("")
    lines.append(f"Configuration:")
    lines.append(f"  Nodes: {n_nodes}")
    lines.append(f"  μ (mixing parameter): {MU}")
    lines.append(f"  k_avg (average degree): {K_AVG}")
    lines.append(f"  tau1 (degree exponent): {TAU1}")
    lines.append(f"  tau2 (community exponent): {TAU2}")
    lines.append(f"  Noise ratios: {[f'{r*100:.0f}%' for r in NOISE_RATIOS]}")
    lines.append(f"  ε values: {EPSILON_VALUES}")
    lines.append("")

    # Per-noise-level tables
    for noise_ratio in NOISE_RATIOS:
        key = (n_nodes, noise_ratio)
        if key not in summary:
            continue

        configs = summary[key]
        lines.append("")
        lines.append(f"{'='*120}")
        lines.append(f"NOISE LEVEL: {noise_ratio*100:.0f}%")
        lines.append("=" * 120)
        lines.append(f"{'Config':<15} {'Edge%':>8} {'GT_Mod':>14} {'Leiden_Mod':>14} {'Gap':>10} {'NMI':>14} {'ARI':>14} {'GT_Ratio':>10}")
        lines.append("-" * 120)

        for config_name in ['original'] + [f'spectral_eps{eps}' for eps in EPSILON_VALUES]:
            if config_name not in configs:
                continue

            c = configs[config_name]
            edge_pct = c['edge_ratio'] * 100

            gt_mod_str = f"{c['gt_modularity_mean']:.3f}±{c['gt_modularity_std']:.3f}" if 'gt_modularity_mean' in c else "-"
            leiden_mod_str = f"{c['leiden_modularity_mean']:.3f}±{c['leiden_modularity_std']:.3f}"
            gap_str = f"{c['spurious_gap_mean']:.3f}" if 'spurious_gap_mean' in c else "-"
            nmi_str = f"{c['nmi_mean']:.3f}±{c['nmi_std']:.3f}"
            ari_str = f"{c['ari_mean']:.3f}±{c['ari_std']:.3f}"
            gt_ratio_str = f"{c['gt_ratio_mean']:.3f}" if 'gt_ratio_mean' in c else "-"

            lines.append(f"{config_name:<15} {edge_pct:>7.1f}% {gt_mod_str:>14} {leiden_mod_str:>14} {gap_str:>10} {nmi_str:>14} {ari_str:>14} {gt_ratio_str:>10}")

    # Improvement summary
    lines.append("")
    lines.append("=" * 120)
    lines.append("IMPROVEMENT SUMMARY (eps=2.0 vs original)")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"{'Noise%':<10} {'GT_Mod Δ':>12} {'NMI Δ':>12} {'Gap Δ':>12} {'GT_Ratio':>12} {'Verdict':>15}")
    lines.append("-" * 75)

    for noise_ratio in NOISE_RATIOS:
        key = (n_nodes, noise_ratio)
        if key not in summary:
            continue

        configs = summary[key]
        if 'original' not in configs or 'spectral_eps2.0' not in configs:
            continue

        orig = configs['original']
        sparse = configs['spectral_eps2.0']

        gt_mod_delta = sparse.get('gt_modularity_mean', 0) - orig.get('gt_modularity_mean', 0)
        nmi_delta = sparse.get('nmi_mean', 0) - orig.get('nmi_mean', 0)
        gap_delta = sparse.get('spurious_gap_mean', 0) - orig.get('spurious_gap_mean', 0)
        gt_ratio = sparse.get('gt_ratio_mean')

        gt_ratio_str = f"{gt_ratio:.3f}" if gt_ratio else "-"

        if nmi_delta > 0.01:
            verdict = "HELPS (like real)"
        elif nmi_delta < -0.01:
            verdict = "HURTS (clean LFR)"
        else:
            verdict = "NEUTRAL"

        lines.append(f"{noise_ratio*100:>5.0f}%    {gt_mod_delta:>+12.3f} {nmi_delta:>+12.3f} {gap_delta:>+12.3f} {gt_ratio_str:>12} {verdict:>15}")

    lines.append("")
    lines.append("=" * 120)
    lines.append("INTERPRETATION:")
    lines.append("  - GT_Mod Δ > 0: Sparsification reveals true community structure")
    lines.append("  - NMI Δ > 0: Detection accuracy improves with sparsification")
    lines.append("  - Gap Δ < 0: Spurious modularity from noise is removed")
    lines.append("  - GT_Ratio < 1: Inter-community edges removed faster (like real data)")
    lines.append("")
    lines.append("If noisy LFR shows positive NMI Δ and GT_Ratio < 1, it confirms:")
    lines.append("  'Sparsification helps real networks by removing noise'")
    lines.append("=" * 120)
    lines.append("")
    lines.append("END OF REPORT")

    return "\n".join(lines)


# =============================================================================
# Main Experiment
# =============================================================================

def run_full_experiment(n_nodes, repeats, noise_ratios=None):
    """Run full noisy LFR experiment across all noise levels."""

    if noise_ratios is None:
        noise_ratios = NOISE_RATIOS

    NOISY_LFR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_configs = len(noise_ratios) * repeats
    current = 0

    print("=" * 70)
    print("NOISY LFR BENCHMARK EXPERIMENT")
    print("=" * 70)
    print(f"Nodes: {n_nodes}")
    print(f"μ (mixing): {MU}")
    print(f"k_avg: {K_AVG}")
    print(f"Noise ratios: {[f'{r*100:.0f}%' for r in noise_ratios]}")
    print(f"ε values: {EPSILON_VALUES}")
    print(f"Repetitions: {repeats}")
    print(f"Total configurations: {total_configs}")
    print("=" * 70)

    for noise_ratio in noise_ratios:
        print(f"\n{'='*60}")
        print(f"Running: noise_ratio={noise_ratio*100:.0f}%")
        print("=" * 60)

        for rep in range(repeats):
            current += 1
            seed = BASE_SEED + rep * 1000 + int(noise_ratio * 100)

            print(f"\n  Repetition {rep+1}/{repeats} (seed={seed}) [{current}/{total_configs}]")

            result = run_single_experiment(n_nodes, noise_ratio, seed)
            all_results.append(result)

            if 'error' not in result:
                orig = result['configs']['original']
                print(f"    Original: NMI={orig['metrics']['nmi']:.3f}, GT_Mod={orig['gt_modularity']:.3f}, Gap={orig['spurious_gap']:.3f}")

                # Best sparsified (typically eps=2.0)
                if 'spectral_eps2.0' in result['configs'] and 'error' not in result['configs']['spectral_eps2.0']:
                    sparse = result['configs']['spectral_eps2.0']
                    nmi_delta = sparse['metrics']['nmi'] - orig['metrics']['nmi']
                    print(f"    eps=2.0: NMI={sparse['metrics']['nmi']:.3f} (Δ={nmi_delta:+.3f}), GT_Ratio={sparse['gt_edge_preservation']['gt_preservation_ratio']:.3f}")
            else:
                print(f"    ERROR: {result['error']}")

        # Print aggregated results for this noise level
        noise_results = [r for r in all_results if r.get('noise_ratio') == noise_ratio]
        if noise_results:
            summary = aggregate_results(noise_results)
            if (n_nodes, noise_ratio) in summary:
                print_config_table(n_nodes, noise_ratio, summary[(n_nodes, noise_ratio)])

    # Final aggregation and summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    summary = aggregate_results(all_results)

    print_noise_comparison(summary, n_nodes)
    print_improvement_summary(summary, n_nodes)

    # Save results
    results_file = NOISY_LFR_RESULTS_DIR / f"noisy_lfr_n{n_nodes}_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {results_file}")

    # Save summary
    summary_serializable = {f"{k[0]}_{k[1]}": v for k, v in summary.items()}
    summary_file = NOISY_LFR_RESULTS_DIR / f"noisy_lfr_n{n_nodes}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_serializable, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    # Generate and save report
    report = generate_report(summary, all_results, n_nodes)
    report_file = NOISY_LFR_RESULTS_DIR / f"noisy_lfr_n{n_nodes}_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_file}")

    return all_results, summary


def main():
    parser = argparse.ArgumentParser(description='Noisy LFR Benchmark Experiment')
    parser.add_argument('--n', type=int, default=DEFAULT_N,
                        help=f'Number of nodes (default: {DEFAULT_N})')
    parser.add_argument('--repeats', type=int, default=DEFAULT_REPEATS,
                        help=f'Number of repetitions (default: {DEFAULT_REPEATS})')
    parser.add_argument('--noise', type=float, nargs='+', default=None,
                        help=f'Noise ratios to test (default: {NOISE_RATIOS})')

    args = parser.parse_args()

    run_full_experiment(
        n_nodes=args.n,
        repeats=args.repeats,
        noise_ratios=args.noise
    )


if __name__ == '__main__':
    main()
