"""
CM-Mimicking Experiment: Test Sparsification for CM Preprocessing

Tests whether spectral sparsification preserves the well-connected property (WCC)
of communities, so CM decisions should be identical on original and sparsified graphs.

Datasets:
- cit-HepPh: 34,546 nodes, 421,578 edges (Arxiv HEP-Phenomenology citations)
- cit-HepTh: 27,770 nodes, 352,807 edges (Arxiv HEP-Theory citations)
- cit-Patents: 3,774,768 nodes, 16,518,948 edges (US Patent citations)

For each dataset:
1. Load graph (convert directed to undirected)
2. Run Leiden to get initial communities
3. For each community with >= 10 nodes:
   a. Extract subgraph
   b. Compute min-cut on ORIGINAL
   c. Apply sparsification (ε = 0.5, 1.0, 2.0)
   d. Compute min-cut on SPARSIFIED
   e. Compare WCC decisions and metrics

Success Criteria:
- WCC Agreement Rate >= 95%
- WCC Lost Rate <= 2%
- Connectivity Rate = 100%
"""

import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
import json
import time
import sys
import urllib.request
import gzip
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import spectral_sparsify_direct, PROJECT_ROOT

# Try to import community detection
try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False
    print("Warning: leidenalg not available, will use NetworkX Louvain")

from scipy.stats import pearsonr


# =============================================================================
# DATASET LOADING
# =============================================================================

CITATION_DATASETS = {
    'cit-HepPh': {
        'url': 'https://snap.stanford.edu/data/cit-HepPh.txt.gz',
        'nodes': 34546,
        'edges': 421578,
        'description': 'Arxiv HEP-Phenomenology citations'
    },
    'cit-HepTh': {
        'url': 'https://snap.stanford.edu/data/cit-HepTh.txt.gz',
        'nodes': 27770,
        'edges': 352807,
        'description': 'Arxiv HEP-Theory citations'
    },
    'cit-Patents': {
        'url': 'https://snap.stanford.edu/data/cit-Patents.txt.gz',
        'nodes': 3774768,
        'edges': 16518948,
        'description': 'US Patent citations'
    }
}


def download_dataset(name, data_dir):
    """Download citation dataset if not already present."""
    if name not in CITATION_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    info = CITATION_DATASETS[name]
    url = info['url']
    filename = f"{name}.txt.gz"
    filepath = data_dir / filename
    txt_filepath = data_dir / f"{name}.txt"

    if txt_filepath.exists():
        return txt_filepath

    if not filepath.exists():
        print(f"Downloading {name} from {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"  Failed to download: {e}")
            return None

    # Decompress
    if filepath.exists() and not txt_filepath.exists():
        print(f"Decompressing {filename}...")
        with gzip.open(filepath, 'rt') as f_in:
            with open(txt_filepath, 'w') as f_out:
                f_out.write(f_in.read())

    return txt_filepath


def load_citation_network(filepath):
    """
    Load citation network from file.
    Convert directed edges to undirected.
    Returns NetworkX graph.
    """
    G = nx.Graph()  # Undirected
    edge_set = set()

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    src, dst = int(parts[0]), int(parts[1])
                    if src != dst:  # Skip self-loops
                        edge = (min(src, dst), max(src, dst))
                        edge_set.add(edge)
                except ValueError:
                    continue

    G.add_edges_from(edge_set)
    return G


# =============================================================================
# COMMUNITY DETECTION
# =============================================================================

def run_leiden(G):
    """
    Run Leiden community detection.
    Returns dict: node -> community_id
    """
    if HAS_LEIDEN:
        # Convert to igraph
        node_list = list(G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}
        edges = [(node_map[u], node_map[v]) for u, v in G.edges()]

        ig_graph = ig.Graph(n=len(node_list), edges=edges)
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

        # Convert back to node -> community mapping
        node_to_comm = {}
        for comm_id, members in enumerate(partition):
            for idx in members:
                node_to_comm[node_list[idx]] = comm_id
        return node_to_comm
    else:
        # Fallback to NetworkX Louvain
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G, seed=42)
        node_to_comm = {}
        for comm_id, members in enumerate(communities):
            for node in members:
                node_to_comm[node] = comm_id
        return node_to_comm


def get_communities_dict(node_to_comm):
    """Convert node->community to community->nodes dict."""
    comm_to_nodes = defaultdict(set)
    for node, comm in node_to_comm.items():
        comm_to_nodes[comm].add(node)
    return dict(comm_to_nodes)


# =============================================================================
# EFFECTIVE RESISTANCE
# =============================================================================

def compute_effective_resistance(G):
    """
    Compute effective resistance for all edges in graph G.
    Returns dict: edge (canonical form) -> effective resistance
    """
    nodes = list(G.nodes())
    n = len(nodes)

    if n == 0:
        return {}

    node_map = {node: i for i, node in enumerate(nodes)}

    # Build adjacency matrix
    A = np.zeros((n, n))
    for u, v in G.edges():
        i, j = node_map[u], node_map[v]
        A[i, j] = 1
        A[j, i] = 1

    # Compute Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Compute pseudoinverse
    try:
        L_pinv = np.linalg.pinv(L)
    except np.linalg.LinAlgError:
        return {}

    # Compute ER for each edge
    er_dict = {}
    for u, v in G.edges():
        i, j = node_map[u], node_map[v]
        r_eff = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
        canonical_edge = (min(u, v), max(u, v))
        er_dict[canonical_edge] = max(0, r_eff)

    return er_dict


# =============================================================================
# SPARSIFICATION
# =============================================================================

def sparsify_subgraph(G, epsilon):
    """
    Apply spectral sparsification to subgraph.
    Returns: sparse_G, kept_edges_set
    """
    nodes = list(G.nodes())
    edges = list(G.edges())
    n = len(nodes)

    if len(edges) == 0:
        return G.copy(), set()

    # Remap nodes to 0-indexed
    node_map = {old: new for new, old in enumerate(nodes)}
    reverse_map = {new: old for new, old in enumerate(nodes)}

    # Convert to directed format for sparsification
    directed_edges = []
    for u, v in edges:
        i, j = node_map[u], node_map[v]
        directed_edges.append((i, j))
        directed_edges.append((j, i))

    try:
        sparse_edges, _ = spectral_sparsify_direct(directed_edges, n, epsilon)

        # Convert back to original node IDs and undirected
        kept_edges = set()
        for i, j in sparse_edges:
            u, v = reverse_map[i], reverse_map[j]
            kept_edges.add((min(u, v), max(u, v)))

        # Build sparse graph
        sparse_G = nx.Graph()
        sparse_G.add_nodes_from(nodes)
        sparse_G.add_edges_from(kept_edges)

        return sparse_G, kept_edges
    except Exception as e:
        print(f"    Sparsification failed: {e}")
        return None, None


# =============================================================================
# MIN-CUT AND WCC
# =============================================================================

def compute_mincut(G):
    """
    Compute minimum edge cut.
    Returns: min_cut_edges (set), min_cut_size
    """
    if not nx.is_connected(G):
        return set(), 0

    try:
        min_cut_edges = nx.minimum_edge_cut(G)
        min_cut_edges = set((min(u, v), max(u, v)) for u, v in min_cut_edges)
        return min_cut_edges, len(min_cut_edges)
    except nx.NetworkXError:
        return set(), 0


def is_well_connected(n_nodes, min_cut_size):
    """Check if community is well-connected: min_cut > log(n)"""
    if n_nodes < 2:
        return False
    log_n = np.log(n_nodes)
    return min_cut_size > log_n


# =============================================================================
# RATIO METRIC (Inter/Intra edges)
# =============================================================================

def compute_ratio_metric(G, node_to_comm, kept_edges=None):
    """
    Compute Ratio = inter_edges / intra_edges

    If kept_edges is provided, compute on sparse graph.
    Otherwise compute on original graph.
    """
    if kept_edges is None:
        edges = set((min(u, v), max(u, v)) for u, v in G.edges())
    else:
        edges = kept_edges

    intra_count = 0
    inter_count = 0

    for u, v in edges:
        if node_to_comm.get(u) == node_to_comm.get(v):
            intra_count += 1
        else:
            inter_count += 1

    if intra_count == 0:
        return float('inf')
    return inter_count / intra_count


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_cm_mimic_experiment(G, node_to_comm, epsilon_values=[0.5, 1.0, 2.0], min_community_size=10):
    """
    Mimic CM pipeline analysis on original vs sparsified graphs.

    Returns list of result dicts for each community.
    """
    communities = get_communities_dict(node_to_comm)
    results = []

    # Filter communities
    valid_communities = {cid: nodes for cid, nodes in communities.items()
                         if len(nodes) >= min_community_size}

    print(f"  Analyzing {len(valid_communities)} communities (>= {min_community_size} nodes)")

    for comm_id, nodes in valid_communities.items():
        # Extract community subgraph
        subgraph = G.subgraph(nodes).copy()
        n_nodes = len(nodes)
        n_edges = subgraph.number_of_edges()
        log_n = np.log(n_nodes)

        # Skip disconnected subgraphs
        if not nx.is_connected(subgraph):
            continue

        # Skip trivial subgraphs
        if n_edges < 2:
            continue

        # Original min-cut
        orig_mincut_edges, orig_mincut_size = compute_mincut(subgraph)
        orig_is_wcc = is_well_connected(n_nodes, orig_mincut_size)

        # Compute effective resistance (for communities with reasonable size)
        er_dict = {}
        if n_nodes <= 1000:  # Skip ER for very large communities (too slow)
            er_dict = compute_effective_resistance(subgraph)

        for eps in epsilon_values:
            # Sparsify
            sparse_subgraph, kept_edges = sparsify_subgraph(subgraph, eps)

            if sparse_subgraph is None:
                continue

            sparse_n_edges = sparse_subgraph.number_of_edges()

            # Check connectivity
            still_connected = nx.is_connected(sparse_subgraph)

            if not still_connected:
                sparse_mincut_size = 0
                sparse_is_wcc = False
                sparse_mincut_edges = set()
            else:
                sparse_mincut_edges, sparse_mincut_size = compute_mincut(sparse_subgraph)
                sparse_is_wcc = is_well_connected(n_nodes, sparse_mincut_size)

            # Compute metrics
            edge_reduction = (n_edges - sparse_n_edges) / n_edges if n_edges > 0 else 0
            mincut_ratio = sparse_mincut_size / orig_mincut_size if orig_mincut_size > 0 else 0
            mincut_error = abs(sparse_mincut_size - orig_mincut_size) / orig_mincut_size if orig_mincut_size > 0 else 0

            # ER analysis (if available)
            mincut_edge_er_avg = 0
            other_edge_er_avg = 0
            if er_dict and orig_mincut_edges:
                mincut_ers = [er_dict.get(e, 0) for e in orig_mincut_edges]
                other_edges = set(er_dict.keys()) - orig_mincut_edges
                other_ers = [er_dict.get(e, 0) for e in other_edges]
                if mincut_ers:
                    mincut_edge_er_avg = np.mean(mincut_ers)
                if other_ers:
                    other_edge_er_avg = np.mean(other_ers)

            result = {
                'community': comm_id,
                'n_nodes': n_nodes,
                'n_edges_orig': n_edges,
                'n_edges_sparse': sparse_n_edges,
                'edge_reduction': edge_reduction,
                'epsilon': eps,
                'log_n': log_n,
                'orig_mincut': orig_mincut_size,
                'sparse_mincut': sparse_mincut_size,
                'mincut_ratio': mincut_ratio,
                'mincut_error': mincut_error,
                'orig_is_wcc': orig_is_wcc,
                'sparse_is_wcc': sparse_is_wcc,
                'wcc_decision_match': orig_is_wcc == sparse_is_wcc,
                'wcc_preserved': orig_is_wcc and sparse_is_wcc,
                'wcc_lost': orig_is_wcc and not sparse_is_wcc,
                'wcc_gained': not orig_is_wcc and sparse_is_wcc,
                'still_connected': still_connected,
                'mincut_edge_er_avg': mincut_edge_er_avg,
                'other_edge_er_avg': other_edge_er_avg,
            }
            results.append(result)

    return results


def compute_aggregate_metrics(results, epsilon):
    """Compute aggregate metrics for a specific epsilon."""
    eps_results = [r for r in results if r['epsilon'] == epsilon]

    if not eps_results:
        return None

    n_communities = len(eps_results)
    n_wcc_match = sum(1 for r in eps_results if r['wcc_decision_match'])
    n_wcc_preserved = sum(1 for r in eps_results if r['wcc_preserved'])
    n_wcc_lost = sum(1 for r in eps_results if r['wcc_lost'])
    n_wcc_gained = sum(1 for r in eps_results if r['wcc_gained'])
    n_connected = sum(1 for r in eps_results if r['still_connected'])
    n_orig_wcc = sum(1 for r in eps_results if r['orig_is_wcc'])

    avg_edge_reduction = np.mean([r['edge_reduction'] for r in eps_results])
    avg_mincut_ratio = np.mean([r['mincut_ratio'] for r in eps_results if r['orig_mincut'] > 0])

    # Min-cut correlation
    orig_mincuts = [r['orig_mincut'] for r in eps_results if r['orig_mincut'] > 0]
    sparse_mincuts = [r['sparse_mincut'] for r in eps_results if r['orig_mincut'] > 0]
    if len(orig_mincuts) >= 2:
        # Check if arrays are constant (correlation undefined)
        if np.std(orig_mincuts) > 0 and np.std(sparse_mincuts) > 0:
            mincut_corr, _ = pearsonr(orig_mincuts, sparse_mincuts)
        else:
            mincut_corr = float('nan')  # Constant input, correlation undefined
    else:
        mincut_corr = float('nan')

    return {
        'epsilon': epsilon,
        'n_communities': n_communities,
        'n_orig_wcc': n_orig_wcc,
        'wcc_agreement_rate': n_wcc_match / n_communities if n_communities > 0 else 0,
        'wcc_preserved': n_wcc_preserved,
        'wcc_lost': n_wcc_lost,
        'wcc_lost_rate': n_wcc_lost / n_orig_wcc if n_orig_wcc > 0 else 0,
        'wcc_gained': n_wcc_gained,
        'connectivity_rate': n_connected / n_communities if n_communities > 0 else 0,
        'avg_edge_reduction': avg_edge_reduction,
        'avg_mincut_ratio': avg_mincut_ratio,
        'mincut_correlation': mincut_corr,
    }


def print_tables(dataset_name, results, epsilon_values):
    """Print formatted result tables."""

    print("\n" + "=" * 100)
    print(f"RESULTS FOR {dataset_name}")
    print("=" * 100)

    # Table 2: Per-Epsilon Analysis
    print("\n--- Table: Per-Epsilon Analysis ---")
    print(f"{'ε':>5} | {'Communities':>12} | {'WCC Agree%':>10} | {'WCC Lost':>8} | {'WCC Lost%':>9} | {'Edge Red%':>9} | {'MinCut Corr':>11} | {'Connected%':>10}")
    print("-" * 100)

    for eps in epsilon_values:
        metrics = compute_aggregate_metrics(results, eps)
        if metrics:
            print(f"{eps:>5.1f} | {metrics['n_communities']:>12} | {100*metrics['wcc_agreement_rate']:>9.1f}% | {metrics['wcc_lost']:>8} | {100*metrics['wcc_lost_rate']:>8.1f}% | {100*metrics['avg_edge_reduction']:>8.1f}% | {metrics['mincut_correlation']:>11.3f} | {100*metrics['connectivity_rate']:>9.1f}%")

    # Table 3: Community-Level Details (sample of failures and edge cases)
    print("\n--- Table: Community-Level Details (WCC Lost cases) ---")
    wcc_lost_cases = [r for r in results if r['wcc_lost']]

    if wcc_lost_cases:
        print(f"{'Comm':>6} | {'Nodes':>6} | {'Edges':>6} | {'ε':>4} | {'Orig MC':>7} | {'Sparse MC':>9} | {'log(n)':>6} | {'Edge Red%':>9} | {'Connected':>9}")
        print("-" * 95)
        for r in wcc_lost_cases[:20]:  # Show up to 20
            print(f"{r['community']:>6} | {r['n_nodes']:>6} | {r['n_edges_orig']:>6} | {r['epsilon']:>4.1f} | {r['orig_mincut']:>7} | {r['sparse_mincut']:>9} | {r['log_n']:>6.2f} | {100*r['edge_reduction']:>8.1f}% | {'Yes' if r['still_connected'] else 'NO':>9}")
    else:
        print("  No WCC Lost cases!")

    # Table 4: WCC Preserved cases (sample)
    print("\n--- Table: WCC Preserved Sample ---")
    wcc_preserved_cases = [r for r in results if r['wcc_preserved']][:10]

    if wcc_preserved_cases:
        print(f"{'Comm':>6} | {'Nodes':>6} | {'Edges':>6} | {'ε':>4} | {'Orig MC':>7} | {'Sparse MC':>9} | {'log(n)':>6} | {'Edge Red%':>9}")
        print("-" * 85)
        for r in wcc_preserved_cases:
            print(f"{r['community']:>6} | {r['n_nodes']:>6} | {r['n_edges_orig']:>6} | {r['epsilon']:>4.1f} | {r['orig_mincut']:>7} | {r['sparse_mincut']:>9} | {r['log_n']:>6.2f} | {100*r['edge_reduction']:>8.1f}%")

    # ER Analysis
    print("\n--- ER Analysis (Min-cut edges vs Other edges) ---")
    er_results = [r for r in results if r['mincut_edge_er_avg'] > 0]
    if er_results:
        avg_mincut_er = np.mean([r['mincut_edge_er_avg'] for r in er_results])
        avg_other_er = np.mean([r['other_edge_er_avg'] for r in er_results])
        print(f"  Average ER of min-cut edges: {avg_mincut_er:.4f}")
        print(f"  Average ER of other edges: {avg_other_er:.4f}")
        if avg_other_er > 0:
            print(f"  Min-cut / Other ER ratio: {avg_mincut_er / avg_other_er:.2f}x")


def run_dataset_experiment(dataset_name, data_dir, epsilon_values):
    """Run full experiment on a single dataset."""

    print(f"\n{'#' * 80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'#' * 80}")

    info = CITATION_DATASETS[dataset_name]
    print(f"Description: {info['description']}")
    print(f"Expected: {info['nodes']:,} nodes, {info['edges']:,} edges")

    # Download and load
    filepath = download_dataset(dataset_name, data_dir)
    if filepath is None:
        print("  Failed to download dataset, skipping.")
        return None, None

    print(f"\nLoading graph from {filepath}...")
    start = time.time()
    G = load_citation_network(filepath)
    print(f"  Loaded in {time.time() - start:.1f}s")
    print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

    # Get largest connected component
    if not nx.is_connected(G):
        print("  Graph is disconnected, using largest connected component...")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"  LCC: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Run community detection
    print("\nRunning Leiden community detection...")
    start = time.time()
    node_to_comm = run_leiden(G)
    print(f"  Done in {time.time() - start:.1f}s")

    communities = get_communities_dict(node_to_comm)
    print(f"  Found {len(communities)} communities")

    # Compute global Ratio metric
    print("\nComputing global Ratio metric (inter/intra edges)...")
    global_ratio = compute_ratio_metric(G, node_to_comm)
    print(f"  Original graph Ratio (inter/intra): {global_ratio:.4f}")

    # Run CM-mimic experiment
    print("\nRunning CM-mimic experiment...")
    start = time.time()
    results = run_cm_mimic_experiment(G, node_to_comm, epsilon_values)
    print(f"  Done in {time.time() - start:.1f}s")
    print(f"  Analyzed {len(results) // len(epsilon_values)} communities across {len(epsilon_values)} epsilon values")

    # Print tables
    print_tables(dataset_name, results, epsilon_values)

    # Return results for aggregation
    return results, {
        'dataset': dataset_name,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_communities': len(communities),
        'global_ratio': global_ratio,
    }


def print_summary_table(all_results, dataset_infos, epsilon_values):
    """Print Table 1: Per-Dataset Summary."""

    print("\n" + "=" * 120)
    print("SUMMARY TABLE: ALL DATASETS")
    print("=" * 120)

    print(f"\n{'Dataset':<15} | {'Nodes':>10} | {'Edges':>12} | {'Communities':>12} | {'Ratio':>8} | ", end="")
    for eps in epsilon_values:
        print(f"ε={eps} WCC% | ε={eps} Lost | ", end="")
    print()
    print("-" * 120)

    for info in dataset_infos:
        if info is None:
            continue
        dataset = info['dataset']
        results = all_results.get(dataset, [])

        print(f"{dataset:<15} | {info['n_nodes']:>10,} | {info['n_edges']:>12,} | {info['n_communities']:>12} | {info['global_ratio']:>8.4f} | ", end="")

        for eps in epsilon_values:
            metrics = compute_aggregate_metrics(results, eps)
            if metrics:
                print(f"{100*metrics['wcc_agreement_rate']:>7.1f}% | {metrics['wcc_lost']:>8} | ", end="")
            else:
                print(f"{'N/A':>8} | {'N/A':>8} | ", end="")
        print()


def check_success_criteria(all_results, epsilon_values):
    """Check if experiment meets success criteria."""

    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)

    criteria = {
        'WCC Agreement Rate >= 95%': True,
        'WCC Lost Rate <= 2%': True,
        'Connectivity Rate = 100%': True,
    }

    for eps in epsilon_values:
        all_eps_results = []
        for results in all_results.values():
            all_eps_results.extend([r for r in results if r['epsilon'] == eps])

        if not all_eps_results:
            continue

        n_total = len(all_eps_results)
        n_wcc_match = sum(1 for r in all_eps_results if r['wcc_decision_match'])
        n_orig_wcc = sum(1 for r in all_eps_results if r['orig_is_wcc'])
        n_wcc_lost = sum(1 for r in all_eps_results if r['wcc_lost'])
        n_connected = sum(1 for r in all_eps_results if r['still_connected'])

        wcc_agreement = n_wcc_match / n_total if n_total > 0 else 0
        wcc_lost_rate = n_wcc_lost / n_orig_wcc if n_orig_wcc > 0 else 0
        connectivity = n_connected / n_total if n_total > 0 else 0

        print(f"\nEpsilon = {eps}:")
        print(f"  WCC Agreement Rate: {100*wcc_agreement:.1f}% {'✓' if wcc_agreement >= 0.95 else '✗'}")
        print(f"  WCC Lost Rate: {100*wcc_lost_rate:.1f}% {'✓' if wcc_lost_rate <= 0.02 else '✗'}")
        print(f"  Connectivity Rate: {100*connectivity:.1f}% {'✓' if connectivity >= 1.0 else '✗'}")

        if wcc_agreement < 0.95:
            criteria['WCC Agreement Rate >= 95%'] = False
        if wcc_lost_rate > 0.02:
            criteria['WCC Lost Rate <= 2%'] = False
        if connectivity < 1.0:
            criteria['Connectivity Rate = 100%'] = False

    print("\n" + "-" * 40)
    print("OVERALL VERDICT:")
    all_pass = all(criteria.values())
    for criterion, passed in criteria.items():
        print(f"  {criterion}: {'PASS ✓' if passed else 'FAIL ✗'}")

    if all_pass:
        print("\n*** SPARSIFICATION IS VIABLE FOR CM PREPROCESSING ***")
    else:
        print("\n*** SPARSIFICATION MAY NOT BE SUITABLE FOR CM PREPROCESSING ***")


def main():
    """Main entry point."""

    print("=" * 80)
    print("CM PIPELINE EXPERIMENT: SPARSIFICATION FOR CM PREPROCESSING")
    print("=" * 80)
    print("\nHypothesis: Spectral sparsification preserves the well-connected")
    print("property (WCC) of communities, so CM decisions should be identical")
    print("on original and sparsified graphs.")

    # Setup
    data_dir = PROJECT_ROOT / "datasets"
    data_dir.mkdir(exist_ok=True)

    epsilon_values = [1.0]

    # Datasets to run
    datasets_to_run = ['cit-HepTh', 'cit-HepPh', 'cit-Patents']

    # Check command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=datasets_to_run,
                        help='Datasets to run (default: cit-HepTh cit-HepPh cit-Patents)')
    parser.add_argument('--epsilon', nargs='+', type=float, default=epsilon_values,
                        help='Epsilon values to test (default: 1.0)')
    args = parser.parse_args()

    datasets_to_run = args.datasets
    epsilon_values = args.epsilon

    print(f"\nDatasets: {datasets_to_run}")
    print(f"Epsilon values: {epsilon_values}")

    # Run experiments
    all_results = {}
    dataset_infos = []

    for dataset in datasets_to_run:
        results, info = run_dataset_experiment(dataset, data_dir, epsilon_values)
        if results:
            all_results[dataset] = results
            dataset_infos.append(info)

    # Print summary
    print_summary_table(all_results, dataset_infos, epsilon_values)

    # Check success criteria
    check_success_criteria(all_results, epsilon_values)

    # Save results
    results_dir = PROJECT_ROOT / "results" / "cm_pipeline"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "cm_experiment_results.json"
    with open(output_file, 'w') as f:
        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, set):
                    return list(obj)
                return super().default(obj)

        json.dump({
            'results': all_results,
            'dataset_infos': dataset_infos,
            'epsilon_values': epsilon_values,
        }, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
