"""
Cluster Subgraph Sparsification Analysis

Loads a graph and clustering, extracts subgraph for each cluster,
and runs spectral sparsification to see reduction per cluster.
Also finds minimum cut edges and checks if they were removed.
"""

import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import spectral_sparsify_direct, PROJECT_ROOT


def load_network(filepath):
    """Load edge list from TSV file (src\tdst format). Graph is undirected."""
    edge_set = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    src, dst = int(parts[0]), int(parts[1])
                    if src != dst:  # Skip self-loops
                        edge_set.add((min(src, dst), max(src, dst)))  # Undirected: store canonical form
                except ValueError:
                    continue
    # Return as list of tuples
    return list(edge_set)


def load_clustering(filepath):
    """Load node to cluster mapping from TSV file (node\tcluster format)."""
    node_to_cluster = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    node, cluster = int(parts[0]), int(parts[1])
                    node_to_cluster[node] = cluster
                except ValueError:
                    continue
    return node_to_cluster


def extract_cluster_subgraph(edges, node_to_cluster, cluster_id):
    """
    Extract subgraph for a specific cluster.
    Only includes edges where BOTH endpoints belong to the cluster.
    Edges are undirected (stored as (min, max) tuples).
    """
    cluster_nodes = set(node for node, c in node_to_cluster.items() if c == cluster_id)

    subgraph_edges = []
    for src, dst in edges:
        if src in cluster_nodes and dst in cluster_nodes:
            subgraph_edges.append((min(src, dst), max(src, dst)))

    # Remove duplicates (should already be unique, but just in case)
    subgraph_edges = list(set(subgraph_edges))

    return cluster_nodes, subgraph_edges


def find_min_cut_edges(nodes, edges):
    """
    Find minimum cut edges in the subgraph.
    Returns the set of edges that form the minimum edge cut.
    """
    if len(edges) == 0 or len(nodes) < 2:
        return set(), 0

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Check if graph is connected
    if not nx.is_connected(G):
        # Graph is already disconnected, min cut is 0
        return set(), 0

    # Find minimum edge cut
    try:
        min_cut_edges = nx.minimum_edge_cut(G)
        # Convert to canonical form (min, max)
        min_cut_edges = set((min(u, v), max(u, v)) for u, v in min_cut_edges)
        return min_cut_edges, len(min_cut_edges)
    except nx.NetworkXError:
        return set(), 0


def compute_effective_resistance(nodes, edges):
    """
    Compute effective resistance for all edges.
    Returns dict: edge (canonical form) -> effective resistance
    """
    if len(edges) == 0:
        return {}

    # Remap nodes to 0-indexed
    node_list = sorted(nodes)
    node_map = {old: new for new, old in enumerate(node_list)}
    reverse_map = {new: old for new, old in enumerate(node_list)}
    n = len(node_list)

    # Build adjacency matrix
    A = np.zeros((n, n))
    for u, v in edges:
        if u in node_map and v in node_map:
            i, j = node_map[u], node_map[v]
            A[i, j] = 1
            A[j, i] = 1

    # Compute Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Compute pseudoinverse of Laplacian
    try:
        L_pinv = np.linalg.pinv(L)
    except np.linalg.LinAlgError:
        return {}

    # Compute effective resistance for each edge
    er_dict = {}
    for u, v in edges:
        if u in node_map and v in node_map:
            i, j = node_map[u], node_map[v]
            # R_eff(i,j) = L^+(i,i) + L^+(j,j) - 2*L^+(i,j)
            r_eff = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
            canonical_edge = (min(u, v), max(u, v))
            er_dict[canonical_edge] = max(0, r_eff)  # Ensure non-negative

    return er_dict


def count_degree_one_nodes(nodes, edges):
    """
    Count nodes with degree 1 in the subgraph.
    Returns the count of degree-1 nodes.
    """
    if len(edges) == 0:
        return 0

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Count nodes with degree 1
    degree_one_count = sum(1 for node in G.nodes() if G.degree(node) == 1)
    return degree_one_count


def get_connected_components(nodes, edges):
    """
    Get connected components of a subgraph.
    Returns list of (cc_nodes, cc_edges) tuples.
    """
    if len(edges) == 0:
        return []

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Get connected components
    components = []
    for cc_nodes in nx.connected_components(G):
        cc_nodes = set(cc_nodes)
        # Get edges within this component
        cc_edges = [(u, v) for u, v in edges if u in cc_nodes and v in cc_nodes]
        if len(cc_edges) > 0:  # Only include CCs with edges
            components.append((cc_nodes, cc_edges))

    return components


def run_sparsification_on_subgraph(nodes, edges, epsilon):
    """
    Run spectral sparsification on a subgraph.
    Input edges are undirected (stored as (min, max) tuples).
    """
    if len(edges) == 0:
        return 0, 0, []

    # Remap nodes to 0-indexed
    node_list = sorted(nodes)
    node_map = {old: new for new, old in enumerate(node_list)}
    n_nodes = len(node_list)

    # Convert undirected edges to directed format for sparsification
    # (sparsification expects both directions)
    mapped_edges = []
    for src, dst in edges:
        if src in node_map and dst in node_map:
            i, j = node_map[src], node_map[dst]
            mapped_edges.append((i, j))
            mapped_edges.append((j, i))  # Add both directions

    if len(mapped_edges) == 0:
        return 0, 0, []

    original_count = len(edges)  # Original undirected edge count

    try:
        sparse_edges, _ = spectral_sparsify_direct(mapped_edges, n_nodes, epsilon)

        # Convert back to undirected and count
        sparse_unique = set((min(s, d), max(s, d)) for s, d in sparse_edges)
        kept_count = len(sparse_unique)

        return original_count, kept_count, sparse_unique
    except Exception as e:
        print(f"    Sparsification failed: {e}")
        return original_count, -1, []


def main():
    """Main function."""

    # File paths
    network_file = PROJECT_ROOT / "datasets" / "test_network_1.tsv"
    clustering_file = PROJECT_ROOT / "datasets" / "test_clustering_1.tsv"

    # Epsilon value
    epsilon_values = [1.0]

    print("=" * 80)
    print("CLUSTER SUBGRAPH SPARSIFICATION ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading network...")
    edges = load_network(network_file)
    print(f"  Total edges: {len(edges)}")

    print("Loading clustering...")
    node_to_cluster = load_clustering(clustering_file)
    print(f"  Total nodes with cluster assignment: {len(node_to_cluster)}")

    # Get unique clusters
    clusters = sorted(set(node_to_cluster.values()))
    print(f"  Number of clusters: {len(clusters)}")

    # Analyze each cluster
    print("\n" + "=" * 80)
    print("RESULTS BY CLUSTER")
    print("=" * 80)

    # Store results for summary
    results = []

    for cluster_id in clusters:
        cluster_nodes, cluster_edges = extract_cluster_subgraph(edges, node_to_cluster, cluster_id)

        n_nodes = len(cluster_nodes)
        n_edges = len(cluster_edges)  # Already unique undirected edges

        if n_edges < 2:
            # Skip clusters with too few edges
            continue

        # Get connected components
        components = get_connected_components(cluster_nodes, cluster_edges)
        n_components = len(components)

        print(f"\nCluster {cluster_id}: {n_nodes} nodes, {n_edges} edges, {n_components} connected component(s)")

        # Process each connected component
        for cc_idx, (cc_nodes, cc_edges) in enumerate(components):
            cc_n_nodes = len(cc_nodes)
            cc_n_edges = len(cc_edges)

            if n_components > 1:
                print(f"  [CC {cc_idx + 1}/{n_components}]: {cc_n_nodes} nodes, {cc_n_edges} edges")
                prefix = "    "
                label = f"{cluster_id}.{cc_idx + 1}"
            else:
                prefix = "  "
                label = str(cluster_id)

            if cc_n_edges < 2:
                print(f"{prefix}(skipped - too few edges)")
                continue

            # Compute effective resistance for all edges
            er_dict = compute_effective_resistance(cc_nodes, cc_edges)

            # Find minimum cut edges
            min_cut_edges, min_cut_size = find_min_cut_edges(cc_nodes, cc_edges)
            log_n = np.log(cc_n_nodes)
            is_wcc = min_cut_size > log_n
            wcc_status = "(well connected)" if is_wcc else ""
            print(f"{prefix}Min cut size: {min_cut_size} (log(n)={log_n:.2f}) {wcc_status}")
            if min_cut_size > 0 and min_cut_size <= 5:  # Only show if small
                print(f"{prefix}Min cut edges: {sorted(min_cut_edges)}")

            # Show ER statistics for min-cut edges
            if min_cut_size > 0 and er_dict:
                min_cut_ers = [er_dict.get(e, 0) for e in min_cut_edges]
                all_ers = list(er_dict.values())
                print(f"{prefix}Min-cut edges ER: min={min(min_cut_ers):.4f}, max={max(min_cut_ers):.4f}, avg={np.mean(min_cut_ers):.4f}")
                print(f"{prefix}All edges ER: min={min(all_ers):.4f}, max={max(all_ers):.4f}, avg={np.mean(all_ers):.4f}")

            # Count degree-1 nodes in original CC
            original_deg1 = count_degree_one_nodes(cc_nodes, cc_edges)
            print(f"{prefix}Degree-1 nodes (original): {original_deg1}")

            cluster_results = {
                'cluster_id': label,
                'nodes': cc_n_nodes,
                'original_edges': cc_n_edges,
                'min_cut_size': min_cut_size,
                'min_cut_edges': min_cut_edges,
                'original_deg1': original_deg1,
                'er_dict': er_dict,
                'log_n': log_n,
                'is_wcc': is_wcc,
            }

            for eps in epsilon_values:
                original, kept, kept_edges_set = run_sparsification_on_subgraph(cc_nodes, cc_edges, eps)

                if kept == -1:
                    print(f"{prefix}ε={eps}: FAILED")
                    cluster_results[f'eps_{eps}'] = {'kept': -1, 'removed': -1, 'reduction': -1, 'min_cut_removed': -1, 'sparse_deg1': -1}
                else:
                    removed = original - kept
                    reduction = (removed / original * 100) if original > 0 else 0

                    # Convert kept_edges_set back to original node IDs
                    # kept_edges_set uses remapped (0-indexed) nodes, need to map back
                    node_list = sorted(cc_nodes)
                    reverse_map = {new: old for new, old in enumerate(node_list)}
                    kept_edges_original = set((min(reverse_map[s], reverse_map[d]), max(reverse_map[s], reverse_map[d])) for s, d in kept_edges_set)

                    # Check if min cut edges were removed
                    if min_cut_size > 0:
                        min_cut_removed = min_cut_edges - kept_edges_original
                        min_cut_kept = min_cut_edges & kept_edges_original
                        n_min_cut_removed = len(min_cut_removed)
                        min_cut_status = f"{n_min_cut_removed}/{min_cut_size} min-cut edges removed"
                        if n_min_cut_removed > 0:
                            min_cut_status += " ⚠️"
                            # Show ER of removed vs kept min-cut edges
                            if er_dict:
                                removed_ers = [er_dict.get(e, 0) for e in min_cut_removed]
                                kept_mc_ers = [er_dict.get(e, 0) for e in min_cut_kept] if min_cut_kept else []
                                print(f"{prefix}  → Removed min-cut edges ER: {[f'{er:.4f}' for er in sorted(removed_ers)]}")
                                if kept_mc_ers:
                                    print(f"{prefix}  → Kept min-cut edges ER: {[f'{er:.4f}' for er in sorted(kept_mc_ers)]}")
                    else:
                        n_min_cut_removed = 0
                        min_cut_status = "N/A (disconnected)"
                        min_cut_removed = set()

                    # Count degree-1 nodes after sparsification
                    sparse_deg1 = count_degree_one_nodes(cc_nodes, list(kept_edges_original))
                    deg1_change = sparse_deg1 - original_deg1

                    # Run min-cut on sparsified graph
                    sparse_min_cut_edges, sparse_min_cut_size = find_min_cut_edges(cc_nodes, list(kept_edges_original))
                    sparse_is_wcc = sparse_min_cut_size > log_n

                    # Compare with original min-cut
                    if min_cut_size > 0 and sparse_min_cut_size > 0:
                        common_edges = min_cut_edges & sparse_min_cut_edges
                        n_common = len(common_edges)
                        sparse_wcc_status = "(well connected)" if sparse_is_wcc else ""
                        print(f"{prefix}ε={eps}: {kept}/{original} edges kept, {removed} removed ({reduction:.1f}% reduction) | {min_cut_status} | deg1: {original_deg1}→{sparse_deg1} ({deg1_change:+d})")
                        print(f"{prefix}  Sparse min-cut: {sparse_min_cut_size} edges (was {min_cut_size}) | {n_common} in common {sparse_wcc_status}")
                        if is_wcc and not sparse_is_wcc:
                            print(f"{prefix}  ⚠️ WCC LOST: was well-connected, now NOT")
                            print(f"{prefix}  --- WCC LOSS INVESTIGATION ---")
                            print(f"{prefix}  Original graph: {cc_n_nodes} nodes, {cc_n_edges} edges")
                            print(f"{prefix}  Original min-cut ({min_cut_size} edges): {sorted(min_cut_edges)}")
                            print(f"{prefix}  Sparse graph: {cc_n_nodes} nodes, {len(kept_edges_original)} edges")
                            print(f"{prefix}  Sparse min-cut ({sparse_min_cut_size} edges): {sorted(sparse_min_cut_edges)}")
                            removed_edges = set((min(u,v), max(u,v)) for u,v in cc_edges) - kept_edges_original
                            print(f"{prefix}  Removed edges ({len(removed_edges)}): {sorted(removed_edges)}")
                            # Show which min-cut edges were removed
                            orig_mincut_removed = min_cut_edges - kept_edges_original
                            if orig_mincut_removed:
                                print(f"{prefix}  Original min-cut edges REMOVED: {sorted(orig_mincut_removed)}")
                            # Show if sparse min-cut edges existed in original
                            new_mincut_edges = sparse_min_cut_edges - min_cut_edges
                            if new_mincut_edges:
                                print(f"{prefix}  NEW min-cut edges (not in original min-cut): {sorted(new_mincut_edges)}")
                            print(f"{prefix}  --- END INVESTIGATION ---")
                        elif not is_wcc and sparse_is_wcc:
                            print(f"{prefix}  ✓ WCC GAINED: was not well-connected, now IS")
                        if sparse_min_cut_size <= 5:
                            print(f"{prefix}  Sparse min-cut edges: {sorted(sparse_min_cut_edges)}")
                    else:
                        n_common = 0
                        sparse_is_wcc = False
                        print(f"{prefix}ε={eps}: {kept}/{original} edges kept, {removed} removed ({reduction:.1f}% reduction) | {min_cut_status} | deg1: {original_deg1}→{sparse_deg1} ({deg1_change:+d})")
                        if sparse_min_cut_size == 0:
                            print(f"{prefix}  Sparse graph is disconnected!")
                            if is_wcc:
                                print(f"{prefix}  ⚠️ WCC LOST: was well-connected, now DISCONNECTED")

                    cluster_results[f'eps_{eps}'] = {
                        'kept': kept,
                        'removed': removed,
                        'reduction': reduction,
                        'min_cut_removed': n_min_cut_removed,
                        'min_cut_removed_edges': min_cut_removed,
                        'sparse_deg1': sparse_deg1,
                        'sparse_min_cut_size': sparse_min_cut_size,
                        'sparse_min_cut_edges': sparse_min_cut_edges,
                        'common_min_cut': n_common,
                        'sparse_is_wcc': sparse_is_wcc
                    }

            results.append(cluster_results)

    # Summary table
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)

    print(f"\n{'Cluster':>10} | {'Nodes':>6} | {'Edges':>6} | {'MinCut':>6} | {'Deg1':>5} | ", end="")
    for eps in epsilon_values:
        print(f"ε={eps} Rem | ε={eps} Red% | ε={eps} MCut | ε={eps} D1 | ", end="")
    print()
    print("-" * (60 + 45 * len(epsilon_values)))

    total_original = 0
    total_kept = {eps: 0 for eps in epsilon_values}
    total_removed = {eps: 0 for eps in epsilon_values}
    total_mincut_removed = {eps: 0 for eps in epsilon_values}
    total_orig_deg1 = 0
    total_sparse_deg1 = {eps: 0 for eps in epsilon_values}

    for r in results:
        orig_deg1 = r.get('original_deg1', 0)
        print(f"{r['cluster_id']:>10} | {r['nodes']:>6} | {r['original_edges']:>6} | {r['min_cut_size']:>6} | {orig_deg1:>5} | ", end="")
        total_original += r['original_edges']
        total_orig_deg1 += orig_deg1

        for eps in epsilon_values:
            data = r.get(f'eps_{eps}', {})
            kept = data.get('kept', -1)
            removed = data.get('removed', -1)
            reduction = data.get('reduction', -1)
            mincut_rem = data.get('min_cut_removed', -1)
            sparse_deg1 = data.get('sparse_deg1', -1)

            if kept >= 0:
                total_kept[eps] += kept
                total_removed[eps] += removed
                if mincut_rem >= 0:
                    total_mincut_removed[eps] += mincut_rem
                if sparse_deg1 >= 0:
                    total_sparse_deg1[eps] += sparse_deg1
                mincut_str = f"{mincut_rem}" if mincut_rem >= 0 else "N/A"
                deg1_str = f"{sparse_deg1}" if sparse_deg1 >= 0 else "N/A"
                print(f"{removed:>8} | {reduction:>8.1f}% | {mincut_str:>8} | {deg1_str:>7} | ", end="")
            else:
                print(f"{'FAIL':>8} | {'FAIL':>8} | {'FAIL':>8} | {'FAIL':>7} | ", end="")
        print()

    # Totals
    print("-" * (60 + 45 * len(epsilon_values)))
    print(f"{'TOTAL':>10} | {'-':>6} | {total_original:>6} | {'-':>6} | {total_orig_deg1:>5} | ", end="")
    for eps in epsilon_values:
        kept = total_kept[eps]
        removed = total_removed[eps]
        reduction = (removed / total_original * 100) if total_original > 0 else 0
        mincut_rem = total_mincut_removed[eps]
        sparse_deg1 = total_sparse_deg1[eps]
        print(f"{removed:>8} | {reduction:>8.1f}% | {mincut_rem:>8} | {sparse_deg1:>7} | ", end="")
    print()

    # Min cut summary
    print("\n" + "=" * 100)
    print("MIN CUT ANALYSIS WITH EFFECTIVE RESISTANCE")
    print("=" * 100)
    print("\nClusters where min-cut edges were removed:")
    for r in results:
        cluster_id = r['cluster_id']
        min_cut_size = r['min_cut_size']
        min_cut_edges = r.get('min_cut_edges', set())
        er_dict = r.get('er_dict', {})
        if min_cut_size == 0:
            continue
        for eps in epsilon_values:
            data = r.get(f'eps_{eps}', {})
            mincut_rem = data.get('min_cut_removed', 0)
            min_cut_removed_edges = data.get('min_cut_removed_edges', set())
            if mincut_rem > 0:
                print(f"\n  Cluster {cluster_id} (ε={eps}): {mincut_rem}/{min_cut_size} min-cut edges removed")
                if er_dict:
                    # ER statistics
                    all_ers = list(er_dict.values())
                    min_cut_ers = [er_dict.get(e, 0) for e in min_cut_edges]
                    removed_ers = [er_dict.get(e, 0) for e in min_cut_removed_edges]
                    kept_mc_edges = min_cut_edges - min_cut_removed_edges
                    kept_mc_ers = [er_dict.get(e, 0) for e in kept_mc_edges]

                    print(f"    All edges ER: min={min(all_ers):.10f}, max={max(all_ers):.10f}, avg={np.mean(all_ers):.10f}")
                    print(f"    All min-cut edges ER: min={min(min_cut_ers):.10f}, max={max(min_cut_ers):.10f}, avg={np.mean(min_cut_ers):.10f}")
                    print(f"    REMOVED min-cut edges ER: {[f'{er:.4f}' for er in sorted(removed_ers)]}")
                    if kept_mc_ers:
                        print(f"    KEPT min-cut edges ER: avg={np.mean(kept_mc_ers):.4f}, min={min(kept_mc_ers):.4f}")

                    # Percentile of removed edges
                    for edge in sorted(min_cut_removed_edges):
                        er = er_dict.get(edge, 0)
                        percentile = 100 * sum(1 for e in all_ers if e <= er) / len(all_ers)
                        print(f"    Edge {edge}: ER={er:.10f} (percentile: {percentile:.1f}%)")

    # Sparse min-cut comparison summary
    print("\n" + "=" * 100)
    print("SPARSE MIN-CUT COMPARISON")
    print("=" * 100)
    print("\nComparing min-cut before and after sparsification:")
    for r in results:
        cluster_id = r['cluster_id']
        orig_min_cut_size = r['min_cut_size']
        is_wcc = r.get('is_wcc', False)
        log_n = r.get('log_n', 0)
        if orig_min_cut_size == 0:
            continue
        for eps in epsilon_values:
            data = r.get(f'eps_{eps}', {})
            sparse_min_cut_size = data.get('sparse_min_cut_size', 0)
            sparse_is_wcc = data.get('sparse_is_wcc', False)
            n_common = data.get('common_min_cut', 0)
            if sparse_min_cut_size > 0:
                wcc_orig = "WCC" if is_wcc else "not-WCC"
                wcc_sparse = "WCC" if sparse_is_wcc else "not-WCC"
                print(f"  Cluster {cluster_id}: original={orig_min_cut_size} ({wcc_orig}), sparse={sparse_min_cut_size} ({wcc_sparse}), common={n_common}")
                if is_wcc and not sparse_is_wcc:
                    print(f"    ⚠️ WCC LOST")
            else:
                print(f"  Cluster {cluster_id}: original={orig_min_cut_size}, sparse=DISCONNECTED")
                if is_wcc:
                    print(f"    ⚠️ WCC LOST (disconnected)")

    # WCC preservation summary
    print("\n" + "-" * 50)
    print("WCC PRESERVATION SUMMARY:")
    total_wcc = sum(1 for r in results if r.get('is_wcc', False))
    wcc_preserved = 0
    wcc_lost = 0
    for r in results:
        if r.get('is_wcc', False):
            for eps in epsilon_values:
                data = r.get(f'eps_{eps}', {})
                if data.get('sparse_is_wcc', False):
                    wcc_preserved += 1
                else:
                    wcc_lost += 1
    print(f"  Total WCC clusters: {total_wcc}")
    print(f"  WCC preserved after sparsification: {wcc_preserved}")
    print(f"  WCC lost after sparsification: {wcc_lost}")

    # Degree-1 summary
    print("\n" + "=" * 100)
    print("DEGREE-1 NODE ANALYSIS")
    print("=" * 100)
    print("\nClusters where degree-1 nodes increased after sparsification:")
    for r in results:
        cluster_id = r['cluster_id']
        orig_deg1 = r.get('original_deg1', 0)
        for eps in epsilon_values:
            data = r.get(f'eps_{eps}', {})
            sparse_deg1 = data.get('sparse_deg1', -1)
            if sparse_deg1 > orig_deg1:
                change = sparse_deg1 - orig_deg1
                print(f"  Cluster {cluster_id} (ε={eps}): {orig_deg1}→{sparse_deg1} (+{change} degree-1 nodes)")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
