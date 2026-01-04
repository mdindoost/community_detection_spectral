"""
Experiment: Compare DSpar vs Spectral sparsification on citation networks

Measures:
- Connected components (original and after sparsification)
- Edges after sparsification (count and percentage)
- Sparsification time
- Leiden clustering time (original and sparsified)
- NMI/ARI comparing Leiden(original) vs Leiden(sparsified)

Run: python experiments/cit_hepph_experiment.py [dataset]

Datasets: cit-HepPh, cit-HepTh, citeseer
"""

import sys
import time
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from pathlib import Path

from dspar import dspar_sparsify
from utils import spectral_sparsify_direct

# Parameters
EPSILON_VALUES = [0.25, 0.5, 0.75, 1.0]
RETENTION_VALUES = [0.90, 0.75, 0.50]
DSPAR_METHODS = ["paper", "probabilistic_no_replace", "deterministic"]
CPM_RESOLUTIONS = [0.1, 0.01, 0.001]


def load_citation_dataset(name):
    """Load citation dataset by name."""
    if name == "cit-HepPh":
        edge_file = Path("datasets/cit-HepPh/cit-HepPh.txt")
    elif name == "cit-HepTh":
        edge_file = Path("datasets/cit-HepTh/cit-HepTh.txt")
    elif name == "citeseer":
        edge_file = Path("datasets/citeseer/edges_original.txt")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if not edge_file.exists():
        raise FileNotFoundError(f"Dataset not found: {edge_file}")

    G = nx.Graph()
    with open(edge_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:  # Skip self-loops
                    G.add_edge(u, v)

    # Relabel nodes to 0-indexed
    G = nx.convert_node_labels_to_integers(G)
    return G


def nx_to_igraph(G):
    """Convert NetworkX graph to igraph."""
    edges = list(G.edges())
    ig_graph = ig.Graph(n=G.number_of_nodes(), edges=edges, directed=False)
    return ig_graph


def run_leiden(G, resolution=None):
    """Run Leiden clustering and return membership + time.

    If resolution is None, uses ModularityVertexPartition.
    If resolution is specified, uses CPMVertexPartition with that resolution.
    """
    ig_graph = nx_to_igraph(G)

    start = time.time()
    if resolution is None:
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    else:
        partition = leidenalg.find_partition(ig_graph, leidenalg.CPMVertexPartition,
                                              resolution_parameter=resolution)
    elapsed = time.time() - start

    membership = partition.membership
    modularity = partition.modularity
    n_communities = len(set(membership))

    return membership, modularity, n_communities, elapsed


def spectral_sparsify(G, epsilon):
    """Run Julia spectral sparsification."""
    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))

    n_nodes = G.number_of_nodes()

    start = time.time()
    sparsified_edges, _ = spectral_sparsify_direct(edges, n_nodes, epsilon)
    elapsed = time.time() - start

    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(range(n_nodes))
    for u, v in sparsified_edges:
        if u < v:
            G_sparse.add_edge(u, v)

    return G_sparse, elapsed


def run_dspar(G, retention, method):
    """Run DSpar sparsification."""
    start = time.time()
    G_sparse = dspar_sparsify(G, retention=retention, method=method, seed=42)
    elapsed = time.time() - start
    return G_sparse, elapsed


def compute_metrics(G_original, G_sparse, membership_original, membership_sparse):
    """Compute comparison metrics."""
    # Only compare nodes that exist in both
    nodes_original = set(G_original.nodes())
    nodes_sparse = set(G_sparse.nodes())
    common_nodes = sorted(nodes_original & nodes_sparse)

    # Extract memberships for common nodes
    mem_orig = [membership_original[n] for n in common_nodes]
    mem_sparse = [membership_sparse[n] for n in common_nodes]

    nmi = normalized_mutual_info_score(mem_orig, mem_sparse)
    ari = adjusted_rand_score(mem_orig, mem_sparse)

    return nmi, ari


def main():
    # Get dataset name from command line
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cit-HepPh"

    print("=" * 90)
    print(f"{dataset.upper()} EXPERIMENT: DSpar vs Spectral Sparsification")
    print("=" * 90)

    # Load dataset
    print(f"\nLoading {dataset} dataset...")
    G = load_citation_dataset(dataset)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_cc = nx.number_connected_components(G)

    print(f"Original graph: {n_nodes:,} nodes, {n_edges:,} edges, {n_cc} connected components")

    # Run Leiden on original
    print("\nRunning Leiden on original graph...")
    mem_original, mod_original, n_comm_original, leiden_time_original = run_leiden(G)
    print(f"  Communities: {n_comm_original}, Modularity: {mod_original:.4f}, Time: {leiden_time_original:.2f}s")

    # Results storage
    results = []

    # Header
    print("\n" + "=" * 100)
    print(f"{'Method':<35} {'Param':<8} {'Edges':<12} {'%':<8} {'CC':<6} {'Comm':<6} {'Mod':<8} {'NMI':<8} {'ARI':<8} {'Spar(s)':<8} {'Leid(s)':<8}")
    print("-" * 100)
    print(f"{'Original':<35} {'-':<8} {n_edges:<12,} {'100%':<8} {n_cc:<6} {n_comm_original:<6} {mod_original:<8.4f} {'-':<8} {'-':<8} {'-':<8} {leiden_time_original:<8.2f}")
    print("-" * 100)

    # =========================================================================
    # SPECTRAL SPARSIFICATION
    # =========================================================================
    print("\n>>> SPECTRAL (Julia Laplacians.jl)")
    print("-" * 100)

    for epsilon in EPSILON_VALUES:
        try:
            G_sparse, spar_time = spectral_sparsify(G, epsilon)

            n_edges_sparse = G_sparse.number_of_edges()
            edge_pct = 100.0 * n_edges_sparse / n_edges
            n_cc_sparse = nx.number_connected_components(G_sparse)

            # Run Leiden on sparsified
            mem_sparse, mod_sparse, n_comm_sparse, leiden_time_sparse = run_leiden(G_sparse)

            # Compute NMI/ARI
            nmi, ari = compute_metrics(G, G_sparse, mem_original, mem_sparse)

            print(f"{'Spectral':<35} {'ε='+str(epsilon):<8} {n_edges_sparse:<12,} {edge_pct:<8.1f} {n_cc_sparse:<6} {n_comm_sparse:<6} {mod_sparse:<8.4f} {nmi:<8.4f} {ari:<8.4f} {spar_time:<8.2f} {leiden_time_sparse:<8.2f}")

            results.append({
                'method': 'Spectral',
                'param': f'ε={epsilon}',
                'edges': n_edges_sparse,
                'edge_pct': edge_pct,
                'cc': n_cc_sparse,
                'communities': n_comm_sparse,
                'modularity': mod_sparse,
                'nmi': nmi,
                'ari': ari,
                'spar_time': spar_time,
                'leiden_time': leiden_time_sparse
            })

        except Exception as e:
            print(f"{'Spectral':<35} {'ε='+str(epsilon):<8} ERROR: {e}")

    # =========================================================================
    # DSPAR SPARSIFICATION
    # =========================================================================
    for method in DSPAR_METHODS:
        print(f"\n>>> DSPAR ({method})")
        print("-" * 100)

        for retention in RETENTION_VALUES:
            try:
                G_sparse, spar_time = run_dspar(G, retention, method)

                n_edges_sparse = G_sparse.number_of_edges()
                edge_pct = 100.0 * n_edges_sparse / n_edges
                n_cc_sparse = nx.number_connected_components(G_sparse)

                # Run Leiden on sparsified
                mem_sparse, mod_sparse, n_comm_sparse, leiden_time_sparse = run_leiden(G_sparse)

                # Compute NMI/ARI
                nmi, ari = compute_metrics(G, G_sparse, mem_original, mem_sparse)

                method_name = f"DSpar ({method})"
                print(f"{method_name:<35} {'r='+str(retention):<8} {n_edges_sparse:<12,} {edge_pct:<8.1f} {n_cc_sparse:<6} {n_comm_sparse:<6} {mod_sparse:<8.4f} {nmi:<8.4f} {ari:<8.4f} {spar_time:<8.2f} {leiden_time_sparse:<8.2f}")

                results.append({
                    'method': method_name,
                    'param': f'r={retention}',
                    'edges': n_edges_sparse,
                    'edge_pct': edge_pct,
                    'cc': n_cc_sparse,
                    'communities': n_comm_sparse,
                    'modularity': mod_sparse,
                    'nmi': nmi,
                    'ari': ari,
                    'spar_time': spar_time,
                    'leiden_time': leiden_time_sparse
                })

            except Exception as e:
                print(f"{'DSpar ('+method+')':<35} {'r='+str(retention):<8} ERROR: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"\nOriginal: {n_nodes:,} nodes, {n_edges:,} edges, {n_cc} CC, {n_comm_original} communities, Modularity={mod_original:.4f}")
    print(f"Leiden time on original: {leiden_time_original:.2f}s")

    if results:
        best_nmi = max(results, key=lambda x: x['nmi'])
        best_ari = max(results, key=lambda x: x['ari'])
        best_mod = max(results, key=lambda x: x['modularity'])
        fastest_spar = min(results, key=lambda x: x['spar_time'])

        print(f"\nBest NMI: {best_nmi['method']} {best_nmi['param']} -> NMI={best_nmi['nmi']:.4f}")
        print(f"Best ARI: {best_ari['method']} {best_ari['param']} -> ARI={best_ari['ari']:.4f}")
        print(f"Best Modularity: {best_mod['method']} {best_mod['param']} -> Mod={best_mod['modularity']:.4f} (original={mod_original:.4f})")
        print(f"Fastest sparsification: {fastest_spar['method']} {fastest_spar['param']} -> {fastest_spar['spar_time']:.2f}s")

    # =========================================================================
    # CPM RESOLUTION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 100)
    print("CPM RESOLUTION ANALYSIS (Constant Potts Model)")
    print("=" * 100)
    print("\nComparing community detection at different resolutions on original vs best sparsified graph")
    print("Lower resolution = larger communities, Higher resolution = smaller communities\n")

    # Find best sparsification (by modularity)
    if results:
        best_result = max(results, key=lambda x: x['modularity'])
        best_method = best_result['method']
        best_param = best_result['param']

        # Re-run the best sparsification to get the graph
        if 'Spectral' in best_method:
            epsilon = float(best_param.split('=')[1])
            G_best_sparse, _ = spectral_sparsify(G, epsilon)
        else:
            retention = float(best_param.split('=')[1])
            method = best_method.split('(')[1].split(')')[0]
            G_best_sparse, _ = run_dspar(G, retention, method)

        print(f"Best sparsification: {best_method} {best_param} (Mod={best_result['modularity']:.4f})")
        print(f"Edges: {G_best_sparse.number_of_edges():,} ({100*G_best_sparse.number_of_edges()/n_edges:.1f}%)\n")

        print(f"{'Resolution':<12} {'Original Comm':<15} {'Original Mod':<15} {'Sparse Comm':<15} {'Sparse Mod':<15}")
        print("-" * 75)

        for res in CPM_RESOLUTIONS:
            # Run on original
            mem_orig_cpm, mod_orig_cpm, n_comm_orig_cpm, _ = run_leiden(G, resolution=res)

            # Run on best sparsified
            mem_sparse_cpm, mod_sparse_cpm, n_comm_sparse_cpm, _ = run_leiden(G_best_sparse, resolution=res)

            print(f"{res:<12} {n_comm_orig_cpm:<15} {mod_orig_cpm:<15.4f} {n_comm_sparse_cpm:<15} {mod_sparse_cpm:<15.4f}")

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("""
METRICS:
- Mod (Modularity): Quality of clustering on that graph (higher = better separated communities)
- NMI/ARI: Similarity to original clustering (higher = more consistent with original)
- CPM Resolution: Controls community granularity (lower = larger communities)

KEY INSIGHTS:
- Modularity can INCREASE after sparsification if noise edges are removed
- High NMI/ARI + similar Modularity = sparsification preserves structure well
- High Modularity but low NMI/ARI = found different but valid communities
- Spectral preserves connectivity (keeps bridge edges)
- DSpar is faster but may disconnect graph (removes hub edges)
- CPM at different resolutions shows if sparsification affects multi-scale structure
""")


if __name__ == "__main__":
    main()
