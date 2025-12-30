"""
Experiments E2, E3, E4: Sparsification Quality Evaluation

E2: Spectral Sparsification - Tests Spielman-Srivastava spectral sparsification
E3: DSpar Sparsification - Tests degree-based sampling sparsification
E3-Control: Random Sparsification - Baseline comparison
E4: Spectral vs DSpar Comparison - Quantifies how well DSpar approximates spectral

Prerequisites:
- Preprocessed LCC graphs from E1
- Baseline Leiden communities from E1
"""

import numpy as np
import networkx as nx
import pandas as pd
import json
import sys
import time
from pathlib import Path
from collections import Counter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import (
    PROJECT_ROOT, DATASETS_DIR, RESULTS_DIR,
    spectral_sparsify_direct, load_edges_from_file, save_edges_to_file
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
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available for NMI/ARI")


# =============================================================================
# Configuration
# =============================================================================

E1_RESULTS_DIR = RESULTS_DIR / "E1_baseline"

# Datasets from E1
E1_DATASETS = ['email-Eu-core', 'cit-HepTh', 'cit-HepPh', 'com-DBLP', 'com-Youtube', 'test_network_1']

# Ground truth datasets
GT_DATASETS = {'email-Eu-core', 'test_network_1'}

# Spectral sparsification parameters (E2)
SPECTRAL_EPSILONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# DSpar keep ratios (E3)
DSPAR_KEEP_RATIOS = [0.25, 0.40, 0.50, 0.60, 0.75, 0.90]

# Random baseline keep ratios (E3-Control)
RANDOM_KEEP_RATIOS = [0.25, 0.50, 0.75]

RANDOM_SEED = 42


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_e1_data(dataset_name):
    """Load preprocessed data from E1 experiment."""
    dataset_dir = E1_RESULTS_DIR / dataset_name

    # Load LCC edgelist
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
                edges.append((v, u))  # Make bidirectional

    # Get number of nodes
    node_set = set()
    for u, v in edges:
        node_set.add(u)
        node_set.add(v)
    n_nodes = max(node_set) + 1 if node_set else 0

    # Load baseline Leiden communities
    comm_path = dataset_dir / f"{dataset_name}_leiden_communities.tsv"
    baseline_communities = []
    with open(comm_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                baseline_communities.append(int(parts[1]))

    # Load baseline stats
    stats_path = dataset_dir / f"{dataset_name}_baseline_stats.json"
    with open(stats_path, 'r') as f:
        baseline_stats = json.load(f)

    # Load ground truth if available
    ground_truth = None
    if dataset_name in GT_DATASETS:
        if 'ground_truth' in baseline_stats:
            # Ground truth was recorded in E1
            # Need to reload from original source
            ground_truth = load_ground_truth(dataset_name, n_nodes)

    return edges, n_nodes, baseline_communities, baseline_stats, ground_truth


def load_ground_truth(dataset_name, n_nodes):
    """
    Load ground truth communities from E1's saved file.

    E1 saves ground truth with node IDs already remapped to match the LCC graph,
    so we can load it directly without any remapping.
    """
    # Load from E1's saved ground truth file (already remapped to LCC node IDs)
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
    """
    DSpar sparsification: Sample edges with probability proportional to
    1/degree(u) + 1/degree(v).

    Uses sampling WITH replacement (like theoretical DSpar algorithm).

    Returns:
        sparsified_edges: list of (src, dst) tuples (bidirectional)
        elapsed_time: time in seconds
    """
    from collections import Counter

    start_time = time.time()
    np.random.seed(seed)

    # Get unique undirected edges
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
    edge_list = list(edge_set)

    if not edge_list:
        return [], time.time() - start_time

    m = len(edge_list)
    Q = int(m * keep_ratio)  # Number of samples to draw

    if Q >= m:
        # Keep all edges
        sparsified_edges = []
        for u, v in edge_list:
            sparsified_edges.append((u, v))
            sparsified_edges.append((v, u))
        return sparsified_edges, time.time() - start_time

    # Compute degrees
    degree = np.zeros(n_nodes)
    for u, v in edge_list:
        degree[u] += 1
        degree[v] += 1

    # Compute DSpar scores for each edge: 1/d_u + 1/d_v
    probs = []
    for u, v in edge_list:
        p = 1.0 / degree[u] + 1.0 / degree[v]
        probs.append(p)

    probs = np.array(probs)
    probs = probs / probs.sum()  # Normalize to probabilities

    # Sample WITH replacement (like existing DSpar implementation)
    sampled_indices = np.random.choice(len(edge_list), size=Q, replace=True, p=probs)
    edge_counts = Counter(sampled_indices)

    # Create bidirectional edge list from unique sampled edges
    # (edge_counts contains indices of edges that were sampled at least once)
    sparsified_edges = []
    for idx in edge_counts.keys():
        u, v = edge_list[idx]
        sparsified_edges.append((u, v))
        sparsified_edges.append((v, u))

    elapsed_time = time.time() - start_time
    return sparsified_edges, elapsed_time


def random_sparsify(edges, n_nodes, keep_ratio, seed=42):
    """
    Random sparsification: Sample edges uniformly at random.

    Returns:
        sparsified_edges: list of (src, dst) tuples (bidirectional)
        elapsed_time: time in seconds
    """
    start_time = time.time()
    np.random.seed(seed)

    # Get unique undirected edges
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
    edge_list = list(edge_set)

    if not edge_list:
        return [], time.time() - start_time

    # Sample edges uniformly
    n_keep = int(len(edge_list) * keep_ratio)
    n_keep = max(1, min(n_keep, len(edge_list)))

    selected_indices = np.random.choice(
        len(edge_list),
        size=n_keep,
        replace=False
    )

    # Create bidirectional edge list
    sparsified_edges = []
    for idx in selected_indices:
        u, v = edge_list[idx]
        sparsified_edges.append((u, v))
        sparsified_edges.append((v, u))

    elapsed_time = time.time() - start_time
    return sparsified_edges, elapsed_time


# =============================================================================
# Graph Analysis Functions
# =============================================================================

def check_connectivity(edges, n_nodes):
    """Check if graph is connected and return LCC if not."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
    G.add_edges_from(edge_set)

    if nx.is_connected(G):
        return edges, n_nodes, True, 1.0

    # Extract LCC
    components = list(nx.connected_components(G))
    lcc_nodes = max(components, key=len)

    # Remap nodes
    lcc_node_list = sorted(lcc_nodes)
    lcc_map = {old: new for new, old in enumerate(lcc_node_list)}
    n_lcc = len(lcc_node_list)

    # Filter and remap edges
    lcc_edges = []
    for u, v in edges:
        if u in lcc_map and v in lcc_map:
            lcc_edges.append((lcc_map[u], lcc_map[v]))

    lcc_ratio = n_lcc / n_nodes if n_nodes > 0 else 0

    return lcc_edges, n_lcc, False, lcc_ratio


def run_leiden(edges, n_nodes, resolution=1.0, seed=42):
    """Run Leiden algorithm and return communities, modularity, and timing."""
    if not HAS_LEIDEN:
        raise RuntimeError("leidenalg not available")

    start_time = time.time()

    # Build igraph
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))

    g = ig.Graph(n=n_nodes, edges=list(edge_set), directed=False)

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        seed=seed
    )

    communities = partition.membership
    modularity = partition.modularity
    n_communities = len(set(communities))

    elapsed_time = time.time() - start_time

    return communities, modularity, n_communities, elapsed_time


def compute_metrics(communities, ground_truth, n_nodes):
    """Compute NMI and ARI against ground truth."""
    if not HAS_SKLEARN or ground_truth is None:
        return None, None

    pred_labels = []
    true_labels = []

    for node in range(n_nodes):
        if node in ground_truth:
            pred_labels.append(communities[node])
            true_labels.append(ground_truth[node])

    if len(pred_labels) < 2:
        return None, None

    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    return nmi, ari


def count_edges(edges):
    """Count unique undirected edges."""
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
    return len(edge_set)


def save_communities(communities, filepath):
    """Save community assignments to TSV file."""
    with open(filepath, 'w') as f:
        f.write("node_id\tcommunity_id\n")
        for node, comm in enumerate(communities):
            f.write(f"{node}\t{comm}\n")


# =============================================================================
# E2: Spectral Sparsification Experiment
# =============================================================================

def run_e2_spectral(datasets=None):
    """Run E2: Spectral Sparsification experiment."""

    print("\n" + "=" * 80)
    print("EXPERIMENT E2: SPECTRAL SPARSIFICATION")
    print("=" * 80)

    if datasets is None:
        datasets = E1_DATASETS

    output_dir = RESULTS_DIR / "E2_spectral"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    edge_reduction_results = []
    quality_results = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print("=" * 60)

        try:
            # Load E1 data
            edges, n_nodes, baseline_comm, baseline_stats, ground_truth = load_e1_data(dataset_name)
            n_edges_orig = count_edges(edges)
            baseline_mod = baseline_stats['community_detection']['modularity']

            # Get baseline NMI/ARI if ground truth available
            baseline_nmi, baseline_ari = None, None
            if ground_truth is not None:
                baseline_nmi, baseline_ari = compute_metrics(baseline_comm, ground_truth, n_nodes)

            print(f"  Original: {n_nodes} nodes, {n_edges_orig} edges")
            print(f"  Baseline modularity: {baseline_mod:.4f}")
            if baseline_nmi is not None:
                print(f"  Baseline NMI: {baseline_nmi:.4f}, ARI: {baseline_ari:.4f}")

            dataset_output_dir = output_dir / dataset_name
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

            for epsilon in SPECTRAL_EPSILONS:
                print(f"\n  ε = {epsilon}:")

                try:
                    # Run spectral sparsification
                    sparse_edges, sparsify_time = spectral_sparsify_direct(edges, n_nodes, epsilon)
                    n_edges_sparse = count_edges(sparse_edges)
                    edge_pct = 100 * n_edges_sparse / n_edges_orig if n_edges_orig > 0 else 0

                    print(f"    Edges: {n_edges_sparse} ({edge_pct:.1f}%) [sparsify: {sparsify_time:.2f}s]")

                    # Check connectivity
                    sparse_edges_lcc, n_nodes_lcc, is_connected, lcc_ratio = check_connectivity(
                        sparse_edges, n_nodes
                    )

                    if not is_connected:
                        print(f"    WARNING: Disconnected, using LCC ({100*lcc_ratio:.1f}%)")
                        sparse_edges = sparse_edges_lcc
                        n_nodes_use = n_nodes_lcc
                    else:
                        n_nodes_use = n_nodes

                    # Run Leiden
                    communities, modularity, n_comm, leiden_time = run_leiden(sparse_edges, n_nodes_use, seed=RANDOM_SEED)
                    mod_delta = 100 * (modularity - baseline_mod) / baseline_mod if baseline_mod != 0 else 0

                    print(f"    Modularity: {modularity:.4f} (Δ = {mod_delta:+.2f}%) [leiden: {leiden_time:.2f}s]")
                    print(f"    Communities: {n_comm}")

                    # Compute ground truth metrics
                    nmi, ari = None, None
                    nmi_delta, ari_delta = None, None
                    if ground_truth is not None and is_connected:
                        nmi, ari = compute_metrics(communities, ground_truth, n_nodes_use)
                        if nmi is not None and baseline_nmi is not None and baseline_nmi != 0:
                            nmi_delta = 100 * (nmi - baseline_nmi) / baseline_nmi
                        if ari is not None and baseline_ari is not None and baseline_ari != 0:
                            ari_delta = 100 * (ari - baseline_ari) / baseline_ari
                        if nmi is not None:
                            print(f"    NMI: {nmi:.4f} (Δ = {nmi_delta:+.2f}%)" if nmi_delta else f"    NMI: {nmi:.4f}")

                    # Save communities
                    save_communities(communities, dataset_output_dir / f"{dataset_name}_spectral_eps{epsilon}_communities.tsv")

                    # Record results
                    edge_reduction_results.append({
                        'Dataset': dataset_name,
                        'Epsilon': epsilon,
                        'Edges (orig)': n_edges_orig,
                        'Edges (sparse)': n_edges_sparse,
                        'Edge %': f"{edge_pct:.1f}",
                        'Connected': 'Yes' if is_connected else 'No',
                        'Sparsify Time (s)': f"{sparsify_time:.2f}" if sparsify_time else "N/A"
                    })

                    quality_results.append({
                        'Dataset': dataset_name,
                        'Epsilon': epsilon,
                        'Edge %': f"{edge_pct:.1f}",
                        'Communities': n_comm,
                        'Modularity': f"{modularity:.4f}",
                        'Mod Delta %': f"{mod_delta:+.2f}",
                        'NMI (GT)': f"{nmi:.4f}" if nmi is not None else "N/A",
                        'NMI Delta %': f"{nmi_delta:+.2f}" if nmi_delta is not None else "N/A",
                        'ARI (GT)': f"{ari:.4f}" if ari is not None else "N/A",
                        'ARI Delta %': f"{ari_delta:+.2f}" if ari_delta is not None else "N/A",
                        'Leiden Time (s)': f"{leiden_time:.2f}"
                    })

                except Exception as e:
                    print(f"    ERROR: {e}")
                    edge_reduction_results.append({
                        'Dataset': dataset_name,
                        'Epsilon': epsilon,
                        'Edges (orig)': n_edges_orig,
                        'Edges (sparse)': 'ERROR',
                        'Edge %': 'ERROR',
                        'Connected': 'ERROR'
                    })

        except Exception as e:
            print(f"ERROR loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    print(f"\n{'='*60}")
    print("Saving E2 results...")
    print("=" * 60)

    df_edges = pd.DataFrame(edge_reduction_results)
    df_edges.to_csv(output_dir / "E2_1_edge_reduction.csv", index=False)
    print("\nTable E2.1: Edge Reduction")
    print(df_edges.to_string(index=False))

    df_quality = pd.DataFrame(quality_results)
    df_quality.to_csv(output_dir / "E2_2_community_quality.csv", index=False)
    print("\nTable E2.2: Community Quality")
    print(df_quality.to_string(index=False))

    # Compute best configuration per dataset
    best_config = compute_best_config_spectral(quality_results, datasets)
    df_best = pd.DataFrame(best_config)
    df_best.to_csv(output_dir / "E2_3_best_config.csv", index=False)
    print("\nTable E2.3: Best Configuration per Dataset")
    print(df_best.to_string(index=False))

    return edge_reduction_results, quality_results


def compute_best_config_spectral(quality_results, datasets):
    """Compute best spectral configuration per dataset."""
    best_config = []

    for dataset in datasets:
        dataset_results = [r for r in quality_results if r['Dataset'] == dataset and 'ERROR' not in str(r.get('Modularity', 'ERROR'))]

        if not dataset_results:
            best_config.append({
                'Dataset': dataset,
                'Best Epsilon (by Mod)': 'N/A',
                'Edge %': 'N/A',
                'Mod Delta %': 'N/A',
                'Best Epsilon (by NMI)': 'N/A',
                'NMI Delta %': 'N/A'
            })
            continue

        # Best by modularity
        best_mod = max(dataset_results, key=lambda x: float(x['Modularity']))

        # Best by NMI (if available)
        nmi_results = [r for r in dataset_results if r['NMI (GT)'] != 'N/A']
        if nmi_results:
            best_nmi = max(nmi_results, key=lambda x: float(x['NMI (GT)']))
            best_nmi_eps = best_nmi['Epsilon']
            best_nmi_delta = best_nmi['NMI Delta %']
        else:
            best_nmi_eps = 'N/A'
            best_nmi_delta = 'N/A'

        best_config.append({
            'Dataset': dataset,
            'Best Epsilon (by Mod)': best_mod['Epsilon'],
            'Edge %': best_mod['Edge %'],
            'Mod Delta %': best_mod['Mod Delta %'],
            'Best Epsilon (by NMI)': best_nmi_eps,
            'NMI Delta %': best_nmi_delta
        })

    return best_config


# =============================================================================
# E3: DSpar Sparsification Experiment
# =============================================================================

def run_e3_dspar(datasets=None):
    """Run E3: DSpar Sparsification experiment."""

    print("\n" + "=" * 80)
    print("EXPERIMENT E3: DSPAR SPARSIFICATION")
    print("=" * 80)

    if datasets is None:
        datasets = E1_DATASETS

    output_dir = RESULTS_DIR / "E3_dspar"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    edge_reduction_results = []
    quality_results = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print("=" * 60)

        try:
            # Load E1 data
            edges, n_nodes, baseline_comm, baseline_stats, ground_truth = load_e1_data(dataset_name)
            n_edges_orig = count_edges(edges)
            baseline_mod = baseline_stats['community_detection']['modularity']

            # Get baseline NMI/ARI
            baseline_nmi, baseline_ari = None, None
            if ground_truth is not None:
                baseline_nmi, baseline_ari = compute_metrics(baseline_comm, ground_truth, n_nodes)

            print(f"  Original: {n_nodes} nodes, {n_edges_orig} edges")
            print(f"  Baseline modularity: {baseline_mod:.4f}")

            dataset_output_dir = output_dir / dataset_name
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

            for keep_ratio in DSPAR_KEEP_RATIOS:
                print(f"\n  Keep ratio = {keep_ratio*100:.0f}%:")

                try:
                    # Run DSpar sparsification
                    sparse_edges, sparsify_time = dspar_sparsify(edges, n_nodes, keep_ratio, seed=RANDOM_SEED)
                    n_edges_sparse = count_edges(sparse_edges)
                    actual_keep_pct = 100 * n_edges_sparse / n_edges_orig if n_edges_orig > 0 else 0

                    print(f"    Edges: {n_edges_sparse} ({actual_keep_pct:.1f}%) [dspar: {sparsify_time:.2f}s]")

                    # Check connectivity
                    sparse_edges_lcc, n_nodes_lcc, is_connected, lcc_ratio = check_connectivity(
                        sparse_edges, n_nodes
                    )

                    if not is_connected:
                        print(f"    WARNING: Disconnected, using LCC ({100*lcc_ratio:.1f}%)")
                        sparse_edges = sparse_edges_lcc
                        n_nodes_use = n_nodes_lcc
                    else:
                        n_nodes_use = n_nodes

                    # Run Leiden
                    communities, modularity, n_comm, leiden_time = run_leiden(sparse_edges, n_nodes_use, seed=RANDOM_SEED)
                    mod_delta = 100 * (modularity - baseline_mod) / baseline_mod if baseline_mod != 0 else 0

                    print(f"    Modularity: {modularity:.4f} (Δ = {mod_delta:+.2f}%) [leiden: {leiden_time:.2f}s]")

                    # Compute ground truth metrics
                    nmi, ari = None, None
                    nmi_delta, ari_delta = None, None
                    if ground_truth is not None and is_connected:
                        nmi, ari = compute_metrics(communities, ground_truth, n_nodes_use)
                        if nmi is not None and baseline_nmi is not None and baseline_nmi != 0:
                            nmi_delta = 100 * (nmi - baseline_nmi) / baseline_nmi
                        if ari is not None and baseline_ari is not None and baseline_ari != 0:
                            ari_delta = 100 * (ari - baseline_ari) / baseline_ari
                        if nmi is not None:
                            print(f"    NMI: {nmi:.4f}")

                    # Save communities
                    save_communities(communities, dataset_output_dir / f"{dataset_name}_dspar_keep{int(keep_ratio*100)}_communities.tsv")

                    # Record results
                    edge_reduction_results.append({
                        'Dataset': dataset_name,
                        'Keep %': f"{keep_ratio*100:.0f}%",
                        'Edges (orig)': n_edges_orig,
                        'Edges (kept)': n_edges_sparse,
                        'Actual Keep %': f"{actual_keep_pct:.1f}%",
                        'Connected': 'Yes' if is_connected else 'No',
                        'DSpar Time (s)': f"{sparsify_time:.2f}"
                    })

                    quality_results.append({
                        'Dataset': dataset_name,
                        'Keep %': f"{keep_ratio*100:.0f}%",
                        'Communities': n_comm,
                        'Modularity': f"{modularity:.4f}",
                        'Mod Delta %': f"{mod_delta:+.2f}",
                        'NMI (GT)': f"{nmi:.4f}" if nmi is not None else "N/A",
                        'NMI Delta %': f"{nmi_delta:+.2f}" if nmi_delta is not None else "N/A",
                        'ARI (GT)': f"{ari:.4f}" if ari is not None else "N/A",
                        'ARI Delta %': f"{ari_delta:+.2f}" if ari_delta is not None else "N/A",
                        'Leiden Time (s)': f"{leiden_time:.2f}"
                    })

                except Exception as e:
                    print(f"    ERROR: {e}")

        except Exception as e:
            print(f"ERROR loading {dataset_name}: {e}")
            continue

    # Save results
    print(f"\n{'='*60}")
    print("Saving E3 results...")
    print("=" * 60)

    df_edges = pd.DataFrame(edge_reduction_results)
    df_edges.to_csv(output_dir / "E3_1_edge_reduction.csv", index=False)
    print("\nTable E3.1: Edge Reduction")
    print(df_edges.to_string(index=False))

    df_quality = pd.DataFrame(quality_results)
    df_quality.to_csv(output_dir / "E3_2_community_quality.csv", index=False)
    print("\nTable E3.2: Community Quality")
    print(df_quality.to_string(index=False))

    # Compute best configuration
    best_config = compute_best_config_dspar(quality_results, datasets)
    df_best = pd.DataFrame(best_config)
    df_best.to_csv(output_dir / "E3_3_best_config.csv", index=False)
    print("\nTable E3.3: Best Configuration per Dataset")
    print(df_best.to_string(index=False))

    return edge_reduction_results, quality_results


def compute_best_config_dspar(quality_results, datasets):
    """Compute best DSpar configuration per dataset."""
    best_config = []

    for dataset in datasets:
        dataset_results = [r for r in quality_results if r['Dataset'] == dataset]

        if not dataset_results:
            continue

        # Best by modularity
        best_mod = max(dataset_results, key=lambda x: float(x['Modularity']))

        # Best by NMI
        nmi_results = [r for r in dataset_results if r['NMI (GT)'] != 'N/A']
        if nmi_results:
            best_nmi = max(nmi_results, key=lambda x: float(x['NMI (GT)']))
            best_nmi_keep = best_nmi['Keep %']
            best_nmi_delta = best_nmi['NMI Delta %']
        else:
            best_nmi_keep = 'N/A'
            best_nmi_delta = 'N/A'

        best_config.append({
            'Dataset': dataset,
            'Best Keep % (by Mod)': best_mod['Keep %'],
            'Mod Delta %': best_mod['Mod Delta %'],
            'Best Keep % (by NMI)': best_nmi_keep,
            'NMI Delta %': best_nmi_delta
        })

    return best_config


# =============================================================================
# E3-Control: Random Sparsification Baseline
# =============================================================================

def run_e3_control_random(datasets=None):
    """Run E3-Control: Random Sparsification baseline."""

    print("\n" + "=" * 80)
    print("EXPERIMENT E3-CONTROL: RANDOM SPARSIFICATION BASELINE")
    print("=" * 80)

    if datasets is None:
        datasets = E1_DATASETS

    output_dir = RESULTS_DIR / "E3_dspar"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    random_results = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print("=" * 60)

        try:
            # Load E1 data
            edges, n_nodes, baseline_comm, baseline_stats, ground_truth = load_e1_data(dataset_name)
            n_edges_orig = count_edges(edges)
            baseline_mod = baseline_stats['community_detection']['modularity']

            baseline_nmi, baseline_ari = None, None
            if ground_truth is not None:
                baseline_nmi, baseline_ari = compute_metrics(baseline_comm, ground_truth, n_nodes)

            for keep_ratio in RANDOM_KEEP_RATIOS:
                print(f"\n  Keep ratio = {keep_ratio*100:.0f}%:")

                try:
                    # Run random sparsification
                    sparse_edges, sparsify_time = random_sparsify(edges, n_nodes, keep_ratio, seed=RANDOM_SEED)

                    # Check connectivity and get LCC
                    sparse_edges_lcc, n_nodes_lcc, is_connected, lcc_ratio = check_connectivity(
                        sparse_edges, n_nodes
                    )

                    if not is_connected:
                        sparse_edges = sparse_edges_lcc
                        n_nodes_use = n_nodes_lcc
                    else:
                        n_nodes_use = n_nodes

                    # Run Leiden
                    communities, modularity, n_comm, leiden_time = run_leiden(sparse_edges, n_nodes_use, seed=RANDOM_SEED)
                    mod_delta = 100 * (modularity - baseline_mod) / baseline_mod if baseline_mod != 0 else 0

                    print(f"    Modularity: {modularity:.4f} (Δ = {mod_delta:+.2f}%) [random: {sparsify_time:.2f}s, leiden: {leiden_time:.2f}s]")

                    # Ground truth metrics
                    nmi, ari = None, None
                    nmi_delta, ari_delta = None, None
                    if ground_truth is not None and is_connected:
                        nmi, ari = compute_metrics(communities, ground_truth, n_nodes_use)
                        if nmi is not None and baseline_nmi is not None and baseline_nmi != 0:
                            nmi_delta = 100 * (nmi - baseline_nmi) / baseline_nmi

                    random_results.append({
                        'Dataset': dataset_name,
                        'Keep %': f"{keep_ratio*100:.0f}%",
                        'Modularity': f"{modularity:.4f}",
                        'Mod Delta %': f"{mod_delta:+.2f}",
                        'NMI (GT)': f"{nmi:.4f}" if nmi is not None else "N/A",
                        'NMI Delta %': f"{nmi_delta:+.2f}" if nmi_delta is not None else "N/A",
                        'Random Time (s)': f"{sparsify_time:.2f}",
                        'Leiden Time (s)': f"{leiden_time:.2f}"
                    })

                except Exception as e:
                    print(f"    ERROR: {e}")

        except Exception as e:
            print(f"ERROR loading {dataset_name}: {e}")
            continue

    # Save results
    df_random = pd.DataFrame(random_results)
    df_random.to_csv(output_dir / "E3_4_random_baseline.csv", index=False)
    print("\nTable E3.4: Random Sparsification Baseline")
    print(df_random.to_string(index=False))

    return random_results


# =============================================================================
# E4: Spectral vs DSpar Comparison
# =============================================================================

def run_e4_comparison(e2_quality, e3_quality, random_results, datasets=None):
    """Run E4: Spectral vs DSpar comparison."""

    print("\n" + "=" * 80)
    print("EXPERIMENT E4: SPECTRAL VS DSPAR COMPARISON")
    print("=" * 80)

    if datasets is None:
        datasets = E1_DATASETS

    output_dir = RESULTS_DIR / "E4_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table E4.1: Matched comparison
    matched_results = []

    # Table E4.2: Ground truth comparison
    gt_results = []

    # Table E4.3: Method summary
    summary_results = []

    # Table E4.4: vs Random
    vs_random_results = []

    for dataset in datasets:
        print(f"\nProcessing: {dataset}")

        # Get spectral and DSpar results for this dataset
        spectral_data = [r for r in e2_quality if r['Dataset'] == dataset and 'ERROR' not in str(r.get('Modularity', 'ERROR'))]
        dspar_data = [r for r in e3_quality if r['Dataset'] == dataset]
        random_data = [r for r in random_results if r['Dataset'] == dataset]

        if not spectral_data or not dspar_data:
            continue

        # Find matched edge levels (within ±5%)
        matched_pairs = []

        for dspar_r in dspar_data:
            dspar_edge_pct = float(dspar_r['Keep %'].replace('%', ''))

            for spectral_r in spectral_data:
                spectral_edge_pct = float(spectral_r['Edge %'])

                # Check if within ±5%
                if abs(spectral_edge_pct - dspar_edge_pct) <= 5:
                    matched_pairs.append((spectral_r, dspar_r, (spectral_edge_pct + dspar_edge_pct) / 2))

        # Record matched results
        dspar_spectral_ratios_mod = []
        dspar_spectral_ratios_nmi = []

        for spectral_r, dspar_r, avg_edge_pct in matched_pairs:
            spectral_mod = float(spectral_r['Modularity'])
            dspar_mod = float(dspar_r['Modularity'])
            ratio_mod = dspar_mod / spectral_mod if spectral_mod != 0 else 0
            dspar_spectral_ratios_mod.append(ratio_mod)

            matched_results.append({
                'Dataset': dataset,
                'Edge %': f"~{avg_edge_pct:.0f}%",
                'Spectral Config': f"ε={spectral_r['Epsilon']}",
                'DSpar Config': dspar_r['Keep %'],
                'Spectral Mod': spectral_r['Modularity'],
                'DSpar Mod': dspar_r['Modularity'],
                'DSpar/Spectral': f"{ratio_mod:.3f}"
            })

            # Ground truth comparison
            if spectral_r['NMI (GT)'] != 'N/A' and dspar_r['NMI (GT)'] != 'N/A':
                spectral_nmi = float(spectral_r['NMI (GT)'])
                dspar_nmi = float(dspar_r['NMI (GT)'])
                spectral_ari = float(spectral_r['ARI (GT)'])
                dspar_ari = float(dspar_r['ARI (GT)'])

                ratio_nmi = dspar_nmi / spectral_nmi if spectral_nmi != 0 else 0
                ratio_ari = dspar_ari / spectral_ari if spectral_ari != 0 else 0
                dspar_spectral_ratios_nmi.append(ratio_nmi)

                gt_results.append({
                    'Dataset': dataset,
                    'Edge %': f"~{avg_edge_pct:.0f}%",
                    'Spectral NMI': spectral_r['NMI (GT)'],
                    'DSpar NMI': dspar_r['NMI (GT)'],
                    'DSpar/Spectral (NMI)': f"{ratio_nmi:.3f}",
                    'Spectral ARI': spectral_r['ARI (GT)'],
                    'DSpar ARI': dspar_r['ARI (GT)'],
                    'DSpar/Spectral (ARI)': f"{ratio_ari:.3f}"
                })

        # Summary for this dataset
        avg_ratio_mod = np.mean(dspar_spectral_ratios_mod) if dspar_spectral_ratios_mod else None
        avg_ratio_nmi = np.mean(dspar_spectral_ratios_nmi) if dspar_spectral_ratios_nmi else None

        conclusion = "N/A"
        if avg_ratio_mod is not None:
            if avg_ratio_mod >= 0.85:
                conclusion = "Effective"
            else:
                conclusion = "Partial"

        summary_results.append({
            'Dataset': dataset,
            'Avg DSpar/Spectral (Mod)': f"{avg_ratio_mod:.3f}" if avg_ratio_mod else "N/A",
            'Avg DSpar/Spectral (NMI)': f"{avg_ratio_nmi:.3f}" if avg_ratio_nmi else "N/A (no GT)",
            'Conclusion': conclusion
        })

        # vs Random comparison
        for dspar_r in dspar_data:
            dspar_keep = dspar_r['Keep %']

            # Find matching random result
            random_match = [r for r in random_data if r['Keep %'] == dspar_keep]
            if not random_match:
                continue
            random_r = random_match[0]

            # Find matching spectral result
            dspar_edge_pct = float(dspar_keep.replace('%', ''))
            spectral_match = None
            for s in spectral_data:
                if abs(float(s['Edge %']) - dspar_edge_pct) <= 5:
                    spectral_match = s
                    break

            dspar_mod = float(dspar_r['Modularity'])
            random_mod = float(random_r['Modularity'])
            spectral_mod = float(spectral_match['Modularity']) if spectral_match else None

            dspar_vs_random = f"{dspar_mod - random_mod:+.4f}"
            spectral_vs_random = f"{spectral_mod - random_mod:+.4f}" if spectral_mod else "N/A"

            vs_random_results.append({
                'Dataset': dataset,
                'Edge %': dspar_keep,
                'Random Mod': random_r['Modularity'],
                'DSpar Mod': dspar_r['Modularity'],
                'Spectral Mod': spectral_match['Modularity'] if spectral_match else "N/A",
                'DSpar vs Random': dspar_vs_random,
                'Spectral vs Random': spectral_vs_random
            })

    # Save results
    print(f"\n{'='*60}")
    print("Saving E4 results...")
    print("=" * 60)

    df_matched = pd.DataFrame(matched_results)
    df_matched.to_csv(output_dir / "E4_1_matched_comparison.csv", index=False)
    print("\nTable E4.1: Matched Comparison (Spectral vs DSpar)")
    print(df_matched.to_string(index=False))

    if gt_results:
        df_gt = pd.DataFrame(gt_results)
        df_gt.to_csv(output_dir / "E4_2_ground_truth_comparison.csv", index=False)
        print("\nTable E4.2: Ground Truth Comparison")
        print(df_gt.to_string(index=False))

    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(output_dir / "E4_3_method_summary.csv", index=False)
    print("\nTable E4.3: Method Comparison Summary")
    print(df_summary.to_string(index=False))

    if vs_random_results:
        df_vs_random = pd.DataFrame(vs_random_results)
        df_vs_random.to_csv(output_dir / "E4_4_vs_random.csv", index=False)
        print("\nTable E4.4: Method Comparison vs Random")
        print(df_vs_random.to_string(index=False))

    return matched_results, gt_results, summary_results, vs_random_results


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_experiments(datasets=None):
    """Run all E2, E3, E4 experiments."""

    print("=" * 80)
    print("EXPERIMENTS E2, E3, E4: SPARSIFICATION QUALITY EVALUATION")
    print("=" * 80)

    if datasets is None:
        datasets = E1_DATASETS

    # Run E2: Spectral Sparsification
    e2_edges, e2_quality = run_e2_spectral(datasets)

    # Run E3: DSpar Sparsification
    e3_edges, e3_quality = run_e3_dspar(datasets)

    # Run E3-Control: Random Baseline
    random_results = run_e3_control_random(datasets)

    # Run E4: Comparison
    run_e4_comparison(e2_quality, e3_quality, random_results, datasets)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - {RESULTS_DIR / 'E2_spectral'}")
    print(f"  - {RESULTS_DIR / 'E3_dspar'}")
    print(f"  - {RESULTS_DIR / 'E4_comparison'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run E2, E3, E4 sparsification experiments")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to process (default: all)')
    parser.add_argument('--e2-only', action='store_true', help='Run only E2 (Spectral)')
    parser.add_argument('--e3-only', action='store_true', help='Run only E3 (DSpar)')
    parser.add_argument('--e4-only', action='store_true', help='Run only E4 (Comparison)')

    args = parser.parse_args()

    datasets = args.datasets if args.datasets else E1_DATASETS

    if args.e2_only:
        run_e2_spectral(datasets)
    elif args.e3_only:
        run_e3_dspar(datasets)
        run_e3_control_random(datasets)
    elif args.e4_only:
        # Need to load previous results
        print("E4 requires E2 and E3 results. Running all experiments...")
        run_all_experiments(datasets)
    else:
        run_all_experiments(datasets)
