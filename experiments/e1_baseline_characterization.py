"""
Experiment E1: Baseline Characterization

Establishes network properties and baseline community detection quality for all datasets
before any sparsification. This creates the foundation for all subsequent experiments.

Datasets:
- email-Eu-core: with ground truth (department labels)
- cit-HepTh: no ground truth
- cit-HepPh: no ground truth
- com-DBLP: no ground truth (top5000 communities not useful for NMI)
- com-Youtube: no ground truth (top5000 communities not useful for NMI)
- test_network_1: with ground truth

Outputs:
- E1_network_statistics.csv (Table E1.1)
- E1_degree_distribution.csv (Table E1.2)
- E1_community_detection.csv (Table E1.3)
- E1_ground_truth_comparison.csv (Table E1.4)
- E1_edge_classification.csv (Table E1.5)
- Per-dataset files: {dataset}_lcc.edgelist, {dataset}_leiden_communities.tsv, {dataset}_baseline_stats.json
"""

import numpy as np
import networkx as nx
import pandas as pd
import json
import sys
from pathlib import Path
from collections import Counter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import (
    PROJECT_ROOT, DATASETS_DIR, RESULTS_DIR,
    load_snap_dataset, edges_to_adjacency, get_dataset_dir
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
# Dataset Configuration
# =============================================================================

E1_DATASETS = {
    'email-Eu-core': {
        'has_ground_truth': True,
        'source': 'snap'
    },
    'cit-HepTh': {
        'has_ground_truth': False,
        'source': 'snap'
    },
    'cit-HepPh': {
        'has_ground_truth': False,
        'source': 'snap'
    },
    'com-DBLP': {
        'has_ground_truth': False,  # top5000 not useful for full graph NMI
        'source': 'snap'
    },
    'com-Youtube': {
        'has_ground_truth': False,  # top5000 not useful for full graph NMI
        'source': 'snap'
    },
    'test_network_1': {
        'has_ground_truth': True,
        'source': 'local'
    }
}


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_local_dataset(name):
    """Load test_network_1 from local TSV files."""
    edge_file = DATASETS_DIR / f"{name}.tsv"
    gt_file = DATASETS_DIR / f"test_clustering_1.tsv"

    # Load edges
    edges = []
    node_set = set()
    with open(edge_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    src, dst = int(parts[0]), int(parts[1])
                    edges.append((src, dst))
                    node_set.add(src)
                    node_set.add(dst)
                except ValueError:
                    continue

    # Remap to 0-indexed
    node_list = sorted(node_set)
    node_map = {old: new for new, old in enumerate(node_list)}
    reverse_map = {new: old for old, new in node_map.items()}
    n_nodes = len(node_list)

    edges = [(node_map[s], node_map[d]) for s, d in edges]

    # Make undirected, remove self-loops
    edge_set = set()
    for s, d in edges:
        if s != d:
            edge_set.add((min(s, d), max(s, d)))

    edges = []
    for s, d in edge_set:
        edges.append((s, d))
        edges.append((d, s))

    # Load ground truth
    ground_truth = {}
    if gt_file.exists():
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        node, comm = int(parts[0]), int(parts[1])
                        if node in node_map:
                            ground_truth[node_map[node]] = comm
                    except ValueError:
                        continue

    print(f"  Loaded: {n_nodes} nodes, {len(edges)//2} undirected edges")
    if ground_truth:
        print(f"  Ground truth: {len(ground_truth)} nodes, {len(set(ground_truth.values()))} communities")

    return edges, n_nodes, ground_truth if ground_truth else None


def load_dataset(name, config):
    """Load dataset based on configuration."""
    print(f"\nLoading {name}...")

    if config['source'] == 'local':
        return load_local_dataset(name)
    else:
        return load_snap_dataset(name)


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_graph(edges, n_nodes, ground_truth=None):
    """
    Preprocess graph:
    1. Convert to undirected, unweighted
    2. Remove self-loops
    3. Remove duplicate edges
    4. Extract LCC if disconnected

    Returns:
        edges_lcc: preprocessed edges
        n_nodes_lcc: number of nodes in LCC
        ground_truth_lcc: ground truth remapped to LCC (or None)
        stats: dict with preprocessing statistics
    """
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # Add edges (already undirected from loading)
    edge_set = set()
    for s, d in edges:
        if s != d:  # Remove self-loops
            edge_set.add((min(s, d), max(s, d)))

    G.add_edges_from(edge_set)

    stats = {
        'nodes_raw': n_nodes,
        'edges_raw': len(edge_set)
    }

    # Check connectivity and extract LCC
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        lcc_nodes = max(components, key=len)
        G_lcc = G.subgraph(lcc_nodes).copy()
        print(f"  Graph disconnected: {len(components)} components")
        print(f"  Extracted LCC: {G_lcc.number_of_nodes()} nodes ({100*G_lcc.number_of_nodes()/n_nodes:.1f}%)")
    else:
        G_lcc = G
        lcc_nodes = set(G.nodes())

    # Remap LCC nodes to 0-indexed
    lcc_node_list = sorted(lcc_nodes)
    lcc_map = {old: new for new, old in enumerate(lcc_node_list)}
    reverse_lcc_map = {new: old for old, new in lcc_map.items()}
    n_nodes_lcc = len(lcc_node_list)

    # Convert edges
    edges_lcc = []
    for u, v in G_lcc.edges():
        new_u, new_v = lcc_map[u], lcc_map[v]
        edges_lcc.append((new_u, new_v))
        edges_lcc.append((new_v, new_u))

    stats['nodes_lcc'] = n_nodes_lcc
    stats['edges_lcc'] = G_lcc.number_of_edges()
    stats['lcc_pct'] = 100 * n_nodes_lcc / n_nodes

    # Remap ground truth
    ground_truth_lcc = None
    if ground_truth is not None:
        ground_truth_lcc = {}
        for old_node, comm in ground_truth.items():
            if old_node in lcc_map:
                ground_truth_lcc[lcc_map[old_node]] = comm

    return edges_lcc, n_nodes_lcc, ground_truth_lcc, stats, lcc_map


# =============================================================================
# Network Statistics Functions
# =============================================================================

def compute_network_statistics(edges, n_nodes):
    """Compute network statistics for Table E1.1."""
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    edge_set = set()
    for s, d in edges:
        if s < d:
            edge_set.add((s, d))
    G.add_edges_from(edge_set)

    n_edges = len(edge_set)

    # Degree statistics
    degrees = [d for n, d in G.degree()]
    avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
    median_degree = np.median(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0

    # Density
    density = 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

    # Clustering coefficient (can be slow for large graphs)
    if n_nodes < 100000:
        clustering_coef = nx.transitivity(G)
    else:
        # Sample for large graphs
        sample_nodes = np.random.choice(list(G.nodes()), size=min(10000, n_nodes), replace=False)
        clustering_coef = nx.average_clustering(G, nodes=sample_nodes)

    return {
        'avg_degree': avg_degree,
        'median_degree': median_degree,
        'max_degree': max_degree,
        'density': density,
        'clustering_coef': clustering_coef
    }


def compute_degree_distribution(edges, n_nodes):
    """Compute degree distribution statistics for Table E1.2."""
    # Build degree dict
    degree = {i: 0 for i in range(n_nodes)}
    for s, d in edges:
        if s != d:
            degree[s] += 1

    # Divide by 2 since edges are stored bidirectionally
    degrees = [d // 2 for d in degree.values()]

    if not degrees:
        return {pct: 0 for pct in ['min', '25th', 'median', '75th', '90th', '95th', '99th', 'max']}

    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    values = np.percentile(degrees, percentiles)

    return {
        'min': int(values[0]),
        '25th': int(values[1]),
        'median': int(values[2]),
        '75th': int(values[3]),
        '90th': int(values[4]),
        '95th': int(values[5]),
        '99th': int(values[6]),
        'max': int(values[7])
    }


# =============================================================================
# Community Detection Functions
# =============================================================================

def run_leiden(edges, n_nodes, resolution=1.0, seed=42):
    """
    Run Leiden algorithm for community detection.

    Returns:
        communities: list of community assignments (one per node)
        stats: dict with community detection statistics
    """
    if not HAS_LEIDEN:
        raise RuntimeError("leidenalg not available")

    # Build igraph graph
    edge_set = set()
    for s, d in edges:
        if s < d:
            edge_set.add((s, d))

    g = ig.Graph(n=n_nodes, edges=list(edge_set), directed=False)

    # Run Leiden with modularity optimization
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        seed=seed
    )

    communities = partition.membership
    modularity = partition.modularity

    # Compute statistics
    comm_sizes = Counter(communities)
    n_communities = len(comm_sizes)
    singletons = sum(1 for size in comm_sizes.values() if size == 1)
    large_communities = sum(1 for size in comm_sizes.values() if size >= 10)

    # Coverage: fraction of nodes in communities with >= 10 nodes
    nodes_in_large = sum(size for size in comm_sizes.values() if size >= 10)
    coverage = nodes_in_large / n_nodes if n_nodes > 0 else 0

    stats = {
        'n_communities': n_communities,
        'singletons': singletons,
        'large_communities': large_communities,
        'modularity': modularity,
        'coverage': coverage
    }

    return communities, stats


def compute_nmi_ari(communities, ground_truth, n_nodes):
    """Compute NMI and ARI against ground truth."""
    if not HAS_SKLEARN or ground_truth is None:
        return None, None

    # Get labels for nodes that have ground truth
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


# =============================================================================
# Edge Classification Functions
# =============================================================================

def classify_edges(edges, communities):
    """
    Classify edges as intra-community or inter-community based on Leiden communities.

    Returns:
        dict with edge classification statistics
    """
    edge_set = set()
    for s, d in edges:
        if s < d:
            edge_set.add((s, d))

    intra = 0
    inter = 0

    for u, v in edge_set:
        if communities[u] == communities[v]:
            intra += 1
        else:
            inter += 1

    total = intra + inter

    return {
        'total_edges': total,
        'intra_edges': intra,
        'inter_edges': inter,
        'intra_pct': 100 * intra / total if total > 0 else 0,
        'inter_pct': 100 * inter / total if total > 0 else 0
    }


# =============================================================================
# File I/O Functions
# =============================================================================

def save_edgelist(edges, filepath):
    """Save edge list to file."""
    edge_set = set()
    for s, d in edges:
        if s < d:
            edge_set.add((s, d))

    with open(filepath, 'w') as f:
        for u, v in sorted(edge_set):
            f.write(f"{u}\t{v}\n")


def save_communities(communities, filepath):
    """Save community assignments to TSV file."""
    with open(filepath, 'w') as f:
        f.write("node_id\tcommunity_id\n")
        for node, comm in enumerate(communities):
            f.write(f"{node}\t{comm}\n")


def save_stats_json(stats, filepath):
    """Save statistics to JSON file."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(filepath, 'w') as f:
        json.dump(convert(stats), f, indent=2)


# =============================================================================
# Main Experiment
# =============================================================================

def run_e1_experiment():
    """Run the complete E1 baseline characterization experiment."""

    print("=" * 80)
    print("EXPERIMENT E1: BASELINE CHARACTERIZATION")
    print("=" * 80)

    # Output directory
    output_dir = RESULTS_DIR / "E1_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    network_stats = []
    degree_stats = []
    community_stats = []
    gt_comparison = []
    edge_classification = []

    # Process each dataset
    for dataset_name, config in E1_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print("=" * 60)

        try:
            # Load dataset
            edges, n_nodes, ground_truth = load_dataset(dataset_name, config)

            # Preprocess
            print("\nPreprocessing...")
            edges_lcc, n_lcc, gt_lcc, preprocess_stats, lcc_map = preprocess_graph(
                edges, n_nodes, ground_truth
            )

            # Compute network statistics
            print("Computing network statistics...")
            net_stats = compute_network_statistics(edges_lcc, n_lcc)

            network_stats.append({
                'Dataset': dataset_name,
                'Nodes (raw)': preprocess_stats['nodes_raw'],
                'Edges (raw)': preprocess_stats['edges_raw'],
                'Nodes (LCC)': preprocess_stats['nodes_lcc'],
                'Edges (LCC)': preprocess_stats['edges_lcc'],
                'LCC %': f"{preprocess_stats['lcc_pct']:.1f}",
                'Avg Degree': f"{net_stats['avg_degree']:.2f}",
                'Median Degree': int(net_stats['median_degree']),
                'Max Degree': int(net_stats['max_degree']),
                'Density': f"{net_stats['density']:.6f}",
                'Clustering Coef': f"{net_stats['clustering_coef']:.4f}"
            })

            # Compute degree distribution
            print("Computing degree distribution...")
            deg_dist = compute_degree_distribution(edges_lcc, n_lcc)

            degree_stats.append({
                'Dataset': dataset_name,
                'Min': deg_dist['min'],
                '25th %ile': deg_dist['25th'],
                'Median': deg_dist['median'],
                '75th %ile': deg_dist['75th'],
                '90th %ile': deg_dist['90th'],
                '95th %ile': deg_dist['95th'],
                '99th %ile': deg_dist['99th'],
                'Max': deg_dist['max']
            })

            # Run Leiden community detection
            print("Running Leiden community detection...")
            communities, comm_stats = run_leiden(edges_lcc, n_lcc, resolution=1.0, seed=42)

            community_stats.append({
                'Dataset': dataset_name,
                'Communities': comm_stats['n_communities'],
                'Singletons': comm_stats['singletons'],
                'Communities (>=10 nodes)': comm_stats['large_communities'],
                'Modularity': f"{comm_stats['modularity']:.4f}",
                'Coverage': f"{comm_stats['coverage']:.4f}"
            })

            # Ground truth comparison
            if config['has_ground_truth'] and gt_lcc is not None:
                print("Computing ground truth comparison...")
                nmi, ari = compute_nmi_ari(communities, gt_lcc, n_lcc)
                n_gt_communities = len(set(gt_lcc.values()))

                gt_comparison.append({
                    'Dataset': dataset_name,
                    'GT Communities': n_gt_communities,
                    'Leiden Communities': comm_stats['n_communities'],
                    'NMI': f"{nmi:.4f}" if nmi is not None else "N/A",
                    'ARI': f"{ari:.4f}" if ari is not None else "N/A"
                })

                print(f"  NMI: {nmi:.4f}, ARI: {ari:.4f}")

            # Edge classification
            print("Classifying edges...")
            edge_class = classify_edges(edges_lcc, communities)

            edge_classification.append({
                'Dataset': dataset_name,
                'Total Edges': edge_class['total_edges'],
                'Intra-Community Edges': edge_class['intra_edges'],
                'Inter-Community Edges': edge_class['inter_edges'],
                'Intra %': f"{edge_class['intra_pct']:.2f}",
                'Inter %': f"{edge_class['inter_pct']:.2f}"
            })

            # Save per-dataset files
            dataset_output_dir = output_dir / dataset_name
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nSaving outputs to {dataset_output_dir}...")

            # Save LCC edgelist
            save_edgelist(edges_lcc, dataset_output_dir / f"{dataset_name}_lcc.edgelist")

            # Save Leiden communities
            save_communities(communities, dataset_output_dir / f"{dataset_name}_leiden_communities.tsv")

            # Save ground truth (remapped to LCC node IDs) if available
            if gt_lcc is not None:
                gt_path = dataset_output_dir / f"{dataset_name}_ground_truth.tsv"
                with open(gt_path, 'w') as f:
                    f.write("node_id\tcommunity_id\n")
                    for node_id, comm_id in sorted(gt_lcc.items()):
                        f.write(f"{node_id}\t{comm_id}\n")
                print(f"  Saved: {dataset_name}_ground_truth.tsv")

            # Save all statistics to JSON
            all_stats = {
                'preprocessing': preprocess_stats,
                'network': net_stats,
                'degree_distribution': deg_dist,
                'community_detection': comm_stats,
                'edge_classification': edge_class
            }
            if config['has_ground_truth'] and gt_lcc is not None:
                all_stats['ground_truth'] = {
                    'n_gt_communities': n_gt_communities,
                    'nmi': nmi,
                    'ari': ari
                }

            save_stats_json(all_stats, dataset_output_dir / f"{dataset_name}_baseline_stats.json")

            print(f"  Saved: {dataset_name}_lcc.edgelist")
            print(f"  Saved: {dataset_name}_leiden_communities.tsv")
            print(f"  Saved: {dataset_name}_baseline_stats.json")

        except Exception as e:
            print(f"ERROR processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary tables
    print(f"\n{'='*60}")
    print("Saving summary tables...")
    print("=" * 60)

    # Table E1.1: Network Statistics
    df_network = pd.DataFrame(network_stats)
    df_network.to_csv(output_dir / "E1_network_statistics.csv", index=False)
    print(f"\nTable E1.1: Network Statistics")
    print(df_network.to_string(index=False))

    # Table E1.2: Degree Distribution
    df_degree = pd.DataFrame(degree_stats)
    df_degree.to_csv(output_dir / "E1_degree_distribution.csv", index=False)
    print(f"\nTable E1.2: Degree Distribution Summary")
    print(df_degree.to_string(index=False))

    # Table E1.3: Community Detection
    df_community = pd.DataFrame(community_stats)
    df_community.to_csv(output_dir / "E1_community_detection.csv", index=False)
    print(f"\nTable E1.3: Baseline Community Detection (Leiden)")
    print(df_community.to_string(index=False))

    # Table E1.4: Ground Truth Comparison
    if gt_comparison:
        df_gt = pd.DataFrame(gt_comparison)
        df_gt.to_csv(output_dir / "E1_ground_truth_comparison.csv", index=False)
        print(f"\nTable E1.4: Ground Truth Comparison")
        print(df_gt.to_string(index=False))

    # Table E1.5: Edge Classification
    df_edges = pd.DataFrame(edge_classification)
    df_edges.to_csv(output_dir / "E1_edge_classification.csv", index=False)
    print(f"\nTable E1.5: Edge Classification (using Leiden communities)")
    print(df_edges.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print("=" * 60)

    # Validation checks
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    for row in community_stats:
        dataset = row['Dataset']
        mod = float(row['Modularity'])
        if mod <= 0:
            print(f"WARNING: {dataset} has modularity <= 0: {mod}")
        else:
            print(f"OK: {dataset} modularity = {mod}")

    if gt_comparison:
        for row in gt_comparison:
            dataset = row['Dataset']
            nmi_str = row['NMI']
            if nmi_str != "N/A":
                nmi = float(nmi_str)
                if nmi < 0.3:
                    print(f"WARNING: {dataset} has NMI < 0.3: {nmi}")
                else:
                    print(f"OK: {dataset} NMI = {nmi}")

    return output_dir


if __name__ == "__main__":
    run_e1_experiment()
