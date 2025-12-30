"""
EXPERIMENT E9: CM Pipeline Speedup Evaluation

CM Pipeline:
1. Run Leiden → initial communities
2. For each community, recursively refine:
   - Compute min-cut
   - If min-cut > log(|C|): WCC, stop
   - Else: Split and recurse
3. Output: Final refined communities (all WCC or size ≤ 2)

Pipelines to Compare:
- CM-Original: Leiden → CM refinement on original graph
- CM-Spectral: Spectral(ε=2.0) → Leiden → CM refinement on sparse graph
- CM-DSpar: DSpar(75%) → Leiden → CM refinement on sparse graph
"""

import argparse
import numpy as np
import pandas as pd
import math
import time
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import (
    PROJECT_ROOT, RESULTS_DIR,
    spectral_sparsify_direct
)

# Try to import community detection libraries
try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False
    print("Warning: leidenalg not available")

try:
    from sklearn.metrics import normalized_mutual_info_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available for NMI")


# =============================================================================
# Configuration
# =============================================================================

E1_RESULTS_DIR = RESULTS_DIR / "E1_baseline"
E9_RESULTS_DIR = RESULTS_DIR / "E9_pipeline"

# Datasets from E1
E1_DATASETS = ['email-Eu-core', 'cit-HepTh', 'cit-HepPh', 'com-DBLP', 'com-Youtube', 'test_network_1']

# Ground truth datasets
GT_DATASETS = {'email-Eu-core', 'test_network_1'}

# Pipeline parameters
SPECTRAL_EPSILON = 2.0
DSPAR_KEEP_RATIO = 0.75

# Number of runs for timing stability
N_RUNS = 1

RANDOM_SEED = 42


# =============================================================================
# Data Loading
# =============================================================================

def load_e1_data(dataset_name):
    """Load preprocessed data from E1 experiment."""
    dataset_dir = E1_RESULTS_DIR / dataset_name

    edgelist_path = dataset_dir / f"{dataset_name}_lcc.edgelist"
    if not edgelist_path.exists():
        raise FileNotFoundError(f"E1 data not found: {edgelist_path}")

    edges = []
    with open(edgelist_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                edges.append((v, u))  # Bidirectional

    node_set = set()
    for u, v in edges:
        node_set.add(u)
        node_set.add(v)
    n_nodes = max(node_set) + 1 if node_set else 0

    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
    n_edges = len(edge_set)

    return edges, n_nodes, n_edges


def load_ground_truth(dataset_name):
    """Load ground truth from E1's saved file."""
    gt_path = E1_RESULTS_DIR / dataset_name / f"{dataset_name}_ground_truth.tsv"
    if not gt_path.exists():
        return None

    ground_truth = {}
    with open(gt_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    node, comm = int(parts[0]), int(parts[1])
                    ground_truth[node] = comm
                except ValueError:
                    continue

    return ground_truth if ground_truth else None


# =============================================================================
# Sparsification Methods
# =============================================================================

def dspar_sparsify(edges, n_nodes, keep_ratio, seed=42):
    """DSpar sparsification with replacement sampling."""
    from collections import Counter

    np.random.seed(seed)

    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
    edge_list = list(edge_set)

    if not edge_list:
        return []

    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1

    probs = []
    for u, v in edge_list:
        p = 1.0 / degree[u] + 1.0 / degree[v]
        probs.append(p)
    probs = np.array(probs)
    probs = probs / probs.sum()

    Q = int(len(edge_list) * keep_ratio)
    if Q == 0:
        Q = 1

    sampled_indices = np.random.choice(len(edge_list), size=Q, replace=True, p=probs)
    unique_indices = set(sampled_indices)

    sparsified_edges = []
    for idx in unique_indices:
        u, v = edge_list[idx]
        sparsified_edges.append((u, v))
        sparsified_edges.append((v, u))

    return sparsified_edges


# =============================================================================
# Leiden Community Detection
# =============================================================================

def run_leiden(edges, n_nodes, resolution=1.0, seed=42):
    """Run Leiden algorithm (includes graph construction time)."""
    if not HAS_LEIDEN:
        return [], 0.0, 0.0

    start = time.time()  # Include graph construction in timing

    g = ig.Graph(n=n_nodes, directed=False)
    edge_list = list(set((min(u,v), max(u,v)) for u, v in edges))
    g.add_edges(edge_list)

    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        seed=seed
    )
    elapsed = time.time() - start

    # Return list of communities (each is a set of nodes)
    communities = [set(members) for members in partition]
    modularity = partition.modularity

    return communities, modularity, elapsed


# =============================================================================
# CM Refinement (Recursive Min-Cut)
# =============================================================================

def cm_refine(G, community_nodes, depth=0, max_depth=50):
    """
    Recursively refine a community until WCC or size <= 2.

    WCC condition: min-cut > log(|C|)

    Returns:
        refined_communities: list of sets (final communities)
        mincut_calls: number of min-cut computations
        wcc_count: number of communities that satisfied WCC condition
    """
    n = len(community_nodes)

    # Base case: too small to split
    if n <= 2:
        return [set(community_nodes)], 0, 0

    # Prevent infinite recursion
    if depth > max_depth:
        return [set(community_nodes)], 0, 0

    # Extract subgraph
    subgraph = G.subgraph(community_nodes).copy()

    # Check if connected
    if not nx.is_connected(subgraph):
        # Handle disconnected: process each component separately
        components = list(nx.connected_components(subgraph))
        all_refined = []
        total_calls = 0
        total_wcc = 0
        for comp in components:
            refined, calls, wcc = cm_refine(G, comp, depth + 1, max_depth)
            all_refined.extend(refined)
            total_calls += calls
            total_wcc += wcc
        return all_refined, total_calls, total_wcc

    # Compute min-cut using Stoer-Wagner
    try:
        min_cut_value, partition = nx.stoer_wagner(subgraph)
        mincut_calls = 1
    except Exception as e:
        # If min-cut fails, return as-is
        return [set(community_nodes)], 0, 0

    # WCC condition: min-cut > log(n)
    wcc_threshold = math.log(n)

    if min_cut_value > wcc_threshold:
        # Well-Connected Community, stop recursion
        return [set(community_nodes)], mincut_calls, 1
    else:
        # Not WCC: split and recurse
        part1, part2 = partition

        # Recurse on both parts
        refined1, calls1, wcc1 = cm_refine(G, part1, depth + 1, max_depth)
        refined2, calls2, wcc2 = cm_refine(G, part2, depth + 1, max_depth)

        return refined1 + refined2, mincut_calls + calls1 + calls2, wcc1 + wcc2


def run_cm_refinement(G, initial_communities):
    """
    Run CM refinement on all initial communities.

    Returns:
        final_communities: list of sets
        total_mincut_calls: total number of min-cut computations
        total_wcc_count: number of WCC communities
        elapsed_time: time for CM refinement
    """
    start = time.time()

    final_communities = []
    total_mincut_calls = 0
    total_wcc_count = 0

    for comm in initial_communities:
        if len(comm) <= 2:
            final_communities.append(comm)
            continue

        refined, calls, wcc = cm_refine(G, comm)
        final_communities.extend(refined)
        total_mincut_calls += calls
        total_wcc_count += wcc

    elapsed = time.time() - start

    return final_communities, total_mincut_calls, total_wcc_count, elapsed


# =============================================================================
# Full CM Pipeline
# =============================================================================

def cm_pipeline(edges, n_nodes, sparsify_method=None, seed=42):
    """
    Full CM pipeline with optional sparsification.

    Args:
        edges: list of (u, v) tuples
        n_nodes: number of nodes
        sparsify_method: None, 'spectral', or 'dspar'
        seed: random seed

    Returns:
        final_communities: list of sets
        timings: dict with timing breakdown
        stats: dict with statistics
    """
    # Start end-to-end timer
    end_to_end_start = time.time()

    timings = {}
    stats = {}

    # Step 0: Optional sparsification
    if sparsify_method == 'spectral':
        start = time.time()
        sparse_edges, _ = spectral_sparsify_direct(edges, n_nodes, SPECTRAL_EPSILON)
        timings['sparsify'] = time.time() - start
        work_edges = sparse_edges
    elif sparsify_method == 'dspar':
        start = time.time()
        work_edges = dspar_sparsify(edges, n_nodes, DSPAR_KEEP_RATIO, seed)
        timings['sparsify'] = time.time() - start
    else:
        timings['sparsify'] = 0.0
        work_edges = edges

    # Build NetworkX graph for CM refinement (timed)
    start = time.time()
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    edge_set = set((min(u,v), max(u,v)) for u, v in work_edges)
    G.add_edges_from(edge_set)
    timings['graph_build'] = time.time() - start

    stats['edges_used'] = len(edge_set)

    # Step 1: Leiden (includes igraph construction)
    initial_communities, modularity, leiden_time = run_leiden(work_edges, n_nodes, seed=seed)
    timings['leiden'] = leiden_time
    stats['initial_communities'] = len(initial_communities)

    # Step 2: CM refinement
    final_communities, mincut_calls, wcc_count, cm_time = run_cm_refinement(G, initial_communities)
    timings['cm_refine'] = cm_time

    # End-to-end time (captures everything)
    timings['end_to_end'] = time.time() - end_to_end_start

    # Sum of components (for comparison)
    timings['sum_components'] = timings['sparsify'] + timings['graph_build'] + timings['leiden'] + timings['cm_refine']

    # Statistics
    stats['final_communities'] = len(final_communities)
    stats['mincut_calls'] = mincut_calls
    stats['wcc_count'] = wcc_count
    stats['modularity'] = modularity
    stats['avg_calls_per_comm'] = mincut_calls / len(initial_communities) if initial_communities else 0

    return final_communities, timings, stats


# =============================================================================
# Quality Metrics
# =============================================================================

def compute_nmi(communities, ground_truth, n_nodes):
    """Compute NMI between detected communities and ground truth."""
    if not HAS_SKLEARN or ground_truth is None:
        return None

    # Build node to community mapping
    node_to_comm = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = comm_id

    common_nodes = set(node_to_comm.keys()) & set(ground_truth.keys())
    if len(common_nodes) < 10:
        return None

    labels_true = [ground_truth[n] for n in sorted(common_nodes)]
    labels_pred = [node_to_comm[n] for n in sorted(common_nodes)]

    return normalized_mutual_info_score(labels_true, labels_pred)


def compute_modularity(G, communities):
    """Compute modularity of a partition."""
    # Convert to list of frozensets for NetworkX
    partition = [frozenset(c) for c in communities if len(c) > 0]
    try:
        return nx.algorithms.community.modularity(G, partition)
    except:
        return 0.0


# =============================================================================
# Main Experiment
# =============================================================================

def run_e9_experiment():
    """Run E9 CM Pipeline Speedup Evaluation."""
    print("=" * 70)
    print("EXPERIMENT E9: CM Pipeline Speedup Evaluation")
    print("=" * 70)
    print(f"Parameters: Spectral ε={SPECTRAL_EPSILON}, DSpar keep={DSPAR_KEEP_RATIO}")
    print(f"Runs per pipeline: {N_RUNS}")

    E9_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_timing = []
    results_mincut = []
    results_quality = []
    results_speedup = []
    results_recommendation = []

    for dataset_name in E1_DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}...")
        print('='*60)

        try:
            edges, n_nodes, n_edges = load_e1_data(dataset_name)
            ground_truth = load_ground_truth(dataset_name) if dataset_name in GT_DATASETS else None

            print(f"  Nodes: {n_nodes}, Edges: {n_edges}")

            # Store results for each pipeline
            pipeline_results = {}

            for pipeline_name, sparsify_method in [
                ('Original', None),
                ('Spectral', 'spectral'),
                ('DSpar', 'dspar')
            ]:
                print(f"\n  Running CM-{pipeline_name} ({N_RUNS} runs)...")

                all_timings = []
                all_stats = []

                for run in range(N_RUNS):
                    communities, timings, stats = cm_pipeline(
                        edges, n_nodes, sparsify_method, seed=RANDOM_SEED + run
                    )
                    all_timings.append(timings)
                    all_stats.append(stats)

                    if run == 0:
                        final_communities = communities
                        final_stats = stats

                # Average timings
                avg_timings = {
                    'sparsify': np.mean([t['sparsify'] for t in all_timings]),
                    'graph_build': np.mean([t['graph_build'] for t in all_timings]),
                    'leiden': np.mean([t['leiden'] for t in all_timings]),
                    'cm_refine': np.mean([t['cm_refine'] for t in all_timings]),
                    'end_to_end': np.mean([t['end_to_end'] for t in all_timings]),
                    'sum_components': np.mean([t['sum_components'] for t in all_timings])
                }

                # Compute quality metrics
                G = nx.Graph()
                G.add_nodes_from(range(n_nodes))
                G.add_edges_from(set((min(u,v), max(u,v)) for u, v in edges))

                final_modularity = compute_modularity(G, final_communities)
                nmi = compute_nmi(final_communities, ground_truth, n_nodes)

                pipeline_results[pipeline_name] = {
                    'timings': avg_timings,
                    'stats': final_stats,
                    'communities': final_communities,
                    'modularity': final_modularity,
                    'nmi': nmi
                }

                # Calculate edge retention percentage
                edges_retained_pct = (final_stats['edges_used'] / n_edges * 100) if n_edges > 0 else 100

                print(f"    End-to-End: {avg_timings['end_to_end']:.4f}s | Edges: {final_stats['edges_used']}/{n_edges} ({edges_retained_pct:.1f}%)")
                print(f"    Breakdown: Sparsify={avg_timings['sparsify']:.4f}s, GraphBuild={avg_timings['graph_build']:.4f}s, Leiden={avg_timings['leiden']:.4f}s, CM={avg_timings['cm_refine']:.4f}s")
                print(f"    Communities: {final_stats['initial_communities']} → {final_stats['final_communities']}, Min-cut calls: {final_stats['mincut_calls']}, WCC: {final_stats['wcc_count']}")

            # Compute speedups (using end-to-end time)
            orig_e2e = pipeline_results['Original']['timings']['end_to_end']
            orig_cm = pipeline_results['Original']['timings']['cm_refine']

            for pipeline_name in ['Original', 'Spectral', 'DSpar']:
                res = pipeline_results[pipeline_name]
                timings = res['timings']
                stats = res['stats']

                speedup = orig_e2e / timings['end_to_end'] if timings['end_to_end'] > 0 else 0

                # Edge retention
                edges_used = stats.get('edges_used', n_edges)
                edge_pct = (edges_used / n_edges * 100) if n_edges > 0 else 100

                # E9.1: Timing
                results_timing.append({
                    'Dataset': dataset_name,
                    'Pipeline': pipeline_name,
                    'Edges': edges_used,
                    'Edge %': f"{edge_pct:.1f}%",
                    'Sparsify (s)': f"{timings['sparsify']:.4f}",
                    'Graph Build (s)': f"{timings['graph_build']:.4f}",
                    'Leiden (s)': f"{timings['leiden']:.4f}",
                    'CM Refine (s)': f"{timings['cm_refine']:.4f}",
                    'End-to-End (s)': f"{timings['end_to_end']:.4f}",
                    'Speedup': f"{speedup:.2f}x"
                })

                # E9.2: Min-Cut Analysis
                results_mincut.append({
                    'Dataset': dataset_name,
                    'Pipeline': pipeline_name,
                    'Initial Communities': stats['initial_communities'],
                    'Min-Cut Calls': stats['mincut_calls'],
                    'Final Communities': stats['final_communities'],
                    'WCC Count': stats['wcc_count'],
                    'Avg Calls/Comm': f"{stats['avg_calls_per_comm']:.2f}"
                })

                # E9.3: Quality
                mod_change = ((res['modularity'] - pipeline_results['Original']['modularity']) /
                             pipeline_results['Original']['modularity'] * 100) if pipeline_results['Original']['modularity'] > 0 else 0

                orig_nmi = pipeline_results['Original']['nmi']
                nmi_change = ((res['nmi'] - orig_nmi) / orig_nmi * 100) if orig_nmi and res['nmi'] else None

                results_quality.append({
                    'Dataset': dataset_name,
                    'Pipeline': pipeline_name,
                    'Final Communities': stats['final_communities'],
                    'Modularity': f"{res['modularity']:.4f}",
                    'Mod vs Original': f"{mod_change:+.2f}%" if pipeline_name != 'Original' else '-',
                    'NMI': f"{res['nmi']:.4f}" if res['nmi'] else 'N/A',
                    'NMI vs Original': f"{nmi_change:+.2f}%" if nmi_change is not None and pipeline_name != 'Original' else ('-' if pipeline_name == 'Original' else 'N/A'),
                    'WCC Count': stats['wcc_count']
                })

            # E9.4: Speedup Breakdown
            spectral_cm = pipeline_results['Spectral']['timings']['cm_refine']
            dspar_cm = pipeline_results['DSpar']['timings']['cm_refine']
            spectral_e2e = pipeline_results['Spectral']['timings']['end_to_end']
            dspar_e2e = pipeline_results['DSpar']['timings']['end_to_end']

            cm_speedup_spectral = orig_cm / spectral_cm if spectral_cm > 0 else 0
            cm_speedup_dspar = orig_cm / dspar_cm if dspar_cm > 0 else 0

            e2e_speedup_spectral = orig_e2e / spectral_e2e if spectral_e2e > 0 else 0
            e2e_speedup_dspar = orig_e2e / dspar_e2e if dspar_e2e > 0 else 0

            results_speedup.append({
                'Dataset': dataset_name,
                'Original E2E (s)': f"{orig_e2e:.4f}",
                'Spectral E2E (s)': f"{spectral_e2e:.4f}",
                'DSpar E2E (s)': f"{dspar_e2e:.4f}",
                'CM Original (s)': f"{orig_cm:.4f}",
                'CM Spectral (s)': f"{spectral_cm:.4f}",
                'CM DSpar (s)': f"{dspar_cm:.4f}",
                'CM Speedup (Spectral)': f"{cm_speedup_spectral:.2f}x",
                'CM Speedup (DSpar)': f"{cm_speedup_dspar:.2f}x",
                'E2E Speedup (Spectral)': f"{e2e_speedup_spectral:.2f}x",
                'E2E Speedup (DSpar)': f"{e2e_speedup_dspar:.2f}x"
            })

            # E9.5: Recommendation
            best_speedup = max(e2e_speedup_spectral, e2e_speedup_dspar)
            best_pipeline = 'Spectral' if e2e_speedup_spectral >= e2e_speedup_dspar else 'DSpar'
            best_res = pipeline_results[best_pipeline]

            mod_preserved = abs(best_res['modularity'] - pipeline_results['Original']['modularity']) < 0.05

            if best_speedup > 1.2 and mod_preserved:
                recommendation = "Use sparsification"
            elif best_speedup < 0.8:
                recommendation = "Use original"
            else:
                recommendation = "Marginal benefit"

            results_recommendation.append({
                'Dataset': dataset_name,
                'Nodes': n_nodes,
                'Edges': f"{n_edges/1000:.0f}K" if n_edges >= 1000 else str(n_edges),
                'Best Pipeline': best_pipeline,
                'E2E Speedup': f"{best_speedup:.2f}x",
                'Quality Preserved?': 'Yes' if mod_preserved else 'No',
                'Recommendation': recommendation
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    tables = [
        (results_timing, "E9_1_timing_comparison.csv"),
        (results_mincut, "E9_2_mincut_analysis.csv"),
        (results_quality, "E9_3_quality_comparison.csv"),
        (results_speedup, "E9_4_speedup_breakdown.csv"),
        (results_recommendation, "E9_5_recommendation.csv")
    ]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for results, filename in tables:
        if results:
            df = pd.DataFrame(results)
            df.to_csv(E9_RESULTS_DIR / filename, index=False)
            print(f"\n{filename}:")
            print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("E9 COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {E9_RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="E9: CM Pipeline Speedup Evaluation"
    )
    parser.add_argument(
        '-n', '--n-runs',
        type=int,
        default=3,
        help='Number of runs for timing (default: 3)'
    )

    args = parser.parse_args()

    global N_RUNS
    N_RUNS = args.n_runs

    run_e9_experiment()


if __name__ == "__main__":
    main()
