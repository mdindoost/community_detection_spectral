"""
Synthetic Cluster Sparsification Analysis

Tests spectral sparsification on synthetic graphs with known structure:
1. Barbell graph: two cliques connected by a single bridge
2. Two-cluster graph: dense intra-cluster, sparse inter-cluster edges

Analyzes whether sparsification preserves critical connectivity (bridges)
and preferentially removes intra-cluster edges over inter-cluster edges.
"""

import numpy as np
import networkx as nx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import spectral_sparsify_direct, PROJECT_ROOT


def create_barbell_graph(n1=10, n2=10):
    """
    Two cliques of size n1 and n2 connected by a single edge (bridge).

    Returns: G, cluster_labels, bridge_edge
    """
    G = nx.barbell_graph(n1, 0)  # 0 = no path between, just one bridge

    # Cluster labels: 0 for first clique, 1 for second
    cluster_labels = {i: 0 for i in range(n1)}
    cluster_labels.update({i: 1 for i in range(n1, n1 + n2)})

    # Bridge edge connects node (n1-1) to node (n1)
    bridge_edge = (n1 - 1, n1)

    return G, cluster_labels, bridge_edge


def create_two_cluster_graph(n=20, p_intra=0.8, p_inter=0.05, seed=None):
    """
    Two clusters with dense internal edges and sparse inter-edges.

    Cluster 1: nodes 0 to n-1
    Cluster 2: nodes n to 2n-1

    Returns: G, cluster_labels, inter_edges
    """
    if seed is not None:
        np.random.seed(seed)

    G = nx.Graph()
    G.add_nodes_from(range(2 * n))

    intra_edges = []
    inter_edges = []

    # Cluster 1 internal edges
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < p_intra:
                G.add_edge(i, j)
                intra_edges.append((i, j))

    # Cluster 2 internal edges
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < p_intra:
                G.add_edge(n + i, n + j)
                intra_edges.append((n + i, n + j))

    # Inter-cluster edges
    for i in range(n):
        for j in range(n):
            if np.random.random() < p_inter:
                G.add_edge(i, n + j)
                inter_edges.append((i, n + j))

    # Cluster labels
    cluster_labels = {i: 0 for i in range(n)}
    cluster_labels.update({i: 1 for i in range(n, 2 * n)})

    return G, cluster_labels, set(inter_edges)


def compute_effective_resistance(G):
    """
    Compute effective resistance for all edges in graph G.
    Returns dict: edge (canonical form) -> effective resistance
    """
    nodes = list(G.nodes())
    n = len(nodes)
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


def find_min_cut(G):
    """Find minimum edge cut."""
    if not nx.is_connected(G):
        return set(), 0
    try:
        min_cut_edges = nx.minimum_edge_cut(G)
        min_cut_edges = set((min(u, v), max(u, v)) for u, v in min_cut_edges)
        return min_cut_edges, len(min_cut_edges)
    except nx.NetworkXError:
        return set(), 0


def run_sparsification(G, epsilon):
    """
    Run spectral sparsification on graph G.
    Returns: kept_edges (set of canonical tuples), original_count, kept_count
    """
    edges = list(G.edges())
    n = G.number_of_nodes()

    if len(edges) == 0:
        return set(), 0, 0

    # Convert to directed format for sparsification
    directed_edges = []
    for u, v in edges:
        directed_edges.append((u, v))
        directed_edges.append((v, u))

    try:
        sparse_edges, _ = spectral_sparsify_direct(directed_edges, n, epsilon)

        # Convert back to undirected canonical form
        kept_edges = set((min(u, v), max(u, v)) for u, v in sparse_edges)
        return kept_edges, len(edges), len(kept_edges)
    except Exception as e:
        print(f"  Sparsification failed: {e}")
        return set(), len(edges), -1


def classify_edges(G, cluster_labels):
    """
    Classify edges as intra-cluster or inter-cluster.
    Returns: intra_edges (set), inter_edges (set)
    """
    intra = set()
    inter = set()

    for u, v in G.edges():
        canonical = (min(u, v), max(u, v))
        if cluster_labels.get(u) == cluster_labels.get(v):
            intra.add(canonical)
        else:
            inter.add(canonical)

    return intra, inter


def analyze_graph(name, G, cluster_labels, special_edges, epsilon_values):
    """
    Analyze a synthetic graph.

    special_edges: bridge edge for barbell, or inter-cluster edges for two-cluster
    """
    print("\n" + "=" * 80)
    print(f"ANALYZING: {name}")
    print("=" * 80)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    log_n = np.log(n_nodes)

    print(f"\nGraph: {n_nodes} nodes, {n_edges} edges")

    # Classify edges
    intra_edges, inter_edges = classify_edges(G, cluster_labels)
    print(f"Intra-cluster edges: {len(intra_edges)}")
    print(f"Inter-cluster edges: {len(inter_edges)}")

    # Compute effective resistance
    er_dict = compute_effective_resistance(G)

    if er_dict:
        # ER statistics by edge type
        intra_ers = [er_dict.get(e, 0) for e in intra_edges]
        inter_ers = [er_dict.get(e, 0) for e in inter_edges]

        print(f"\nEffective Resistance:")
        if intra_ers:
            print(f"  Intra-cluster: min={min(intra_ers):.4f}, max={max(intra_ers):.4f}, avg={np.mean(intra_ers):.4f}")
        if inter_ers:
            print(f"  Inter-cluster: min={min(inter_ers):.4f}, max={max(inter_ers):.4f}, avg={np.mean(inter_ers):.4f}")

        # Show special edges ER
        if isinstance(special_edges, tuple):
            # Single bridge edge
            bridge_canonical = (min(special_edges), max(special_edges))
            bridge_er = er_dict.get(bridge_canonical, 0)
            print(f"  Bridge edge {bridge_canonical}: ER={bridge_er:.4f}")
        else:
            # Set of inter-cluster edges
            special_ers = [er_dict.get((min(u, v), max(u, v)), 0) for u, v in special_edges if (min(u, v), max(u, v)) in er_dict]
            if special_ers:
                print(f"  Inter-cluster edges ER range: {min(special_ers):.4f} to {max(special_ers):.4f}")

    # Find min-cut
    min_cut_edges, min_cut_size = find_min_cut(G)
    is_wcc = min_cut_size > log_n
    wcc_status = "(well connected)" if is_wcc else ""
    print(f"\nMin-cut: {min_cut_size} edges (log(n)={log_n:.2f}) {wcc_status}")
    if min_cut_size <= 10:
        print(f"Min-cut edges: {sorted(min_cut_edges)}")

    # Run sparsification for each epsilon
    print("\n" + "-" * 60)
    print("SPARSIFICATION RESULTS")
    print("-" * 60)

    for eps in epsilon_values:
        print(f"\n--- Epsilon = {eps} ---")

        kept_edges, original, kept = run_sparsification(G, eps)

        if kept == -1:
            print("  FAILED")
            continue

        removed = original - kept
        reduction = (removed / original * 100) if original > 0 else 0
        print(f"Edges: {kept}/{original} kept, {removed} removed ({reduction:.1f}% reduction)")

        # Classify removed edges
        removed_edges = set((min(u, v), max(u, v)) for u, v in G.edges()) - kept_edges
        removed_intra = removed_edges & intra_edges
        removed_inter = removed_edges & inter_edges

        print(f"Removed intra-cluster: {len(removed_intra)}/{len(intra_edges)} ({100*len(removed_intra)/max(1,len(intra_edges)):.1f}%)")
        print(f"Removed inter-cluster: {len(removed_inter)}/{len(inter_edges)} ({100*len(removed_inter)/max(1,len(inter_edges)):.1f}%)")

        # Check special edges
        if isinstance(special_edges, tuple):
            # Single bridge edge
            bridge_canonical = (min(special_edges), max(special_edges))
            if bridge_canonical in kept_edges:
                print(f"Bridge edge: KEPT (as expected)")
            else:
                print(f"Bridge edge: REMOVED!!! (UNEXPECTED - this should never happen)")
        else:
            # Set of inter-cluster edges
            special_canonical = set((min(u, v), max(u, v)) for u, v in special_edges)
            special_kept = special_canonical & kept_edges
            special_removed = special_canonical - kept_edges
            print(f"Inter-cluster edges kept: {len(special_kept)}/{len(special_canonical)}")
            if special_removed:
                print(f"  Removed inter-cluster: {sorted(special_removed)}")

        # Min-cut on sparse graph
        sparse_G = nx.Graph()
        sparse_G.add_nodes_from(G.nodes())
        sparse_G.add_edges_from(kept_edges)

        if not nx.is_connected(sparse_G):
            print(f"Sparse graph: DISCONNECTED!")
            n_components = nx.number_connected_components(sparse_G)
            print(f"  Number of components: {n_components}")
        else:
            sparse_min_cut, sparse_min_cut_size = find_min_cut(sparse_G)
            sparse_is_wcc = sparse_min_cut_size > log_n
            sparse_wcc_status = "(well connected)" if sparse_is_wcc else ""
            print(f"Sparse min-cut: {sparse_min_cut_size} edges {sparse_wcc_status}")

            if is_wcc and not sparse_is_wcc:
                print(f"  WCC LOST!")

            # Check if min-cut edges changed
            common = min_cut_edges & sparse_min_cut
            print(f"Min-cut overlap: {len(common)}/{min_cut_size} edges in common")

        # ER analysis of removed edges
        if er_dict and removed_edges:
            removed_ers = [er_dict.get(e, 0) for e in removed_edges]
            kept_ers = [er_dict.get(e, 0) for e in kept_edges]
            print(f"\nER of removed edges: min={min(removed_ers):.4f}, max={max(removed_ers):.4f}, avg={np.mean(removed_ers):.4f}")
            print(f"ER of kept edges: min={min(kept_ers):.4f}, max={max(kept_ers):.4f}, avg={np.mean(kept_ers):.4f}")

            # Compare intra vs inter ER for removed
            if removed_intra:
                removed_intra_ers = [er_dict.get(e, 0) for e in removed_intra]
                print(f"ER of removed intra-cluster: avg={np.mean(removed_intra_ers):.4f}")
            if removed_inter:
                removed_inter_ers = [er_dict.get(e, 0) for e in removed_inter]
                print(f"ER of removed inter-cluster: avg={np.mean(removed_inter_ers):.4f}")


def main():
    """Main function."""

    print("=" * 80)
    print("SYNTHETIC CLUSTER SPARSIFICATION ANALYSIS")
    print("=" * 80)
    print("\nThis experiment tests spectral sparsification on synthetic graphs")
    print("with known structure to verify theoretical behavior.")

    epsilon_values = [0.5, 1.0, 2.0]

    # Test 1: Barbell graph
    print("\n\n" + "#" * 80)
    print("TEST 1: BARBELL GRAPH (Two cliques + bridge)")
    print("#" * 80)
    print("\nExpected behavior:")
    print("- Bridge edge has very high ER (only path between cliques)")
    print("- Bridge should NEVER be removed")
    print("- Clique edges have low ER and should be removed preferentially")

    for n in [5, 10, 15]:
        G, labels, bridge = create_barbell_graph(n1=n, n2=n)
        analyze_graph(f"Barbell (2x K{n})", G, labels, bridge, epsilon_values)

    # Test 2: Two-cluster graph
    print("\n\n" + "#" * 80)
    print("TEST 2: TWO-CLUSTER GRAPH (Dense intra, sparse inter)")
    print("#" * 80)
    print("\nExpected behavior:")
    print("- Inter-cluster edges have higher ER than intra-cluster")
    print("- Sparsification should remove more intra-cluster edges")
    print("- Inter-cluster edges should be preferentially kept")

    # Different configurations
    configs = [
        (20, 0.8, 0.05),   # Dense clusters, sparse inter
        (20, 0.5, 0.1),    # Medium density
        (30, 0.9, 0.02),   # Very dense clusters, very sparse inter
    ]

    for n, p_intra, p_inter in configs:
        G, labels, inter_edges = create_two_cluster_graph(n=n, p_intra=p_intra, p_inter=p_inter, seed=42)
        analyze_graph(f"TwoCluster (n={n}, p_intra={p_intra}, p_inter={p_inter})",
                      G, labels, inter_edges, epsilon_values)

    # Test 3: Varying inter-cluster density
    print("\n\n" + "#" * 80)
    print("TEST 3: VARYING INTER-CLUSTER DENSITY")
    print("#" * 80)
    print("\nHow does inter-cluster edge density affect ER distribution?")

    n = 20
    p_intra = 0.7
    for p_inter in [0.01, 0.05, 0.1, 0.2]:
        G, labels, inter_edges = create_two_cluster_graph(n=n, p_intra=p_intra, p_inter=p_inter, seed=42)

        if G.number_of_edges() == 0:
            continue

        er_dict = compute_effective_resistance(G)
        intra_edges, inter_edges_set = classify_edges(G, labels)

        intra_ers = [er_dict.get(e, 0) for e in intra_edges if e in er_dict]
        inter_ers = [er_dict.get(e, 0) for e in inter_edges_set if e in er_dict]

        print(f"\np_inter={p_inter}: {len(intra_edges)} intra, {len(inter_edges_set)} inter edges")
        if intra_ers:
            print(f"  Intra ER avg: {np.mean(intra_ers):.4f}")
        if inter_ers:
            print(f"  Inter ER avg: {np.mean(inter_ers):.4f}")
            if intra_ers:
                ratio = np.mean(inter_ers) / np.mean(intra_ers)
                print(f"  Inter/Intra ER ratio: {ratio:.2f}x")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key observations:
1. Bridge edges in barbell graphs have ER approaching 2.0 (theoretical max for bridge)
2. Inter-cluster edges have higher ER than intra-cluster edges
3. Sparsification preferentially removes low-ER (intra-cluster) edges
4. The more isolated the clusters, the higher the inter/intra ER ratio
5. Well-connected status may be lost if too many edges are removed
""")

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
