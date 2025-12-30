"""
Test balanced min-cut approaches on synthetic graphs.
"""

import networkx as nx


def balanced_mincut(G, min_fraction=0.25):
    """Find min-cut where both sides have >= min_fraction of nodes.

    Approach: Iteratively contract the smallest partition and find new min-cut.
    """
    n = G.number_of_nodes()
    min_size = max(2, int(n * min_fraction))

    H = G.copy()

    for iteration in range(n):
        if H.number_of_nodes() < 2:
            break

        try:
            cut_val, (A, B) = nx.stoer_wagner(H)
        except:
            break

        # Check if balanced
        if min(len(A), len(B)) >= min_size:
            return cut_val, (A, B)

        # Not balanced - merge the smaller side into one node
        smaller = A if len(A) < len(B) else B
        larger = B if len(A) < len(B) else A

        if len(smaller) == 0:
            break

        # Contract smaller side into a single node
        new_node = f"contracted_{iteration}"
        H.add_node(new_node)

        for node in smaller:
            for neighbor in list(H.neighbors(node)):
                if neighbor in larger:
                    if not H.has_edge(new_node, neighbor):
                        H.add_edge(new_node, neighbor)
            H.remove_node(node)

    return None, ([], [])


def fiedler_bisection(G):
    """Spectral bisection using Fiedler vector."""
    fiedler = nx.fiedler_vector(G)
    nodes = list(G.nodes())
    partition_a = [n for n, val in zip(nodes, fiedler) if val < 0]
    partition_b = [n for n, val in zip(nodes, fiedler) if val >= 0]

    # Find cut edges
    set_a = set(partition_a)
    set_b = set(partition_b)
    cut_edges = [(u, v) for u, v in G.edges()
                 if (u in set_a and v in set_b) or (u in set_b and v in set_a)]

    return len(cut_edges), (partition_a, partition_b), cut_edges


def create_test_graph():
    """Create test graph: 2 cliques + 3 bridges + tail of 5 nodes."""
    G = nx.Graph()

    # Clique 1: nodes 0,1,2,3
    G.add_edges_from([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])

    # Clique 2: nodes 4,5,6,7
    G.add_edges_from([(4,5), (4,6), (4,7), (5,6), (5,7), (6,7)])

    # 3 bridges between cliques
    G.add_edge(3, 4)
    G.add_edge(2, 5)
    G.add_edge(1, 6)

    # Tail: path of 5 nodes attached to node 0
    G.add_edge(0, 8)
    G.add_edge(8, 9)
    G.add_edge(9, 10)
    G.add_edge(10, 11)
    G.add_edge(11, 12)

    return G


def main():
    G = create_test_graph()

    print("Graph: 2 cliques (4 nodes each) + 3 bridges + tail (5 nodes)")
    print(f"Total: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # Standard stoer_wagner
    cut_val, partition = nx.stoer_wagner(G)
    print("=" * 60)
    print("Standard Stoer-Wagner:")
    print("=" * 60)
    print(f"  Min-cut: {cut_val}")
    print(f"  Partition sizes: {len(partition[0])}, {len(partition[1])}")
    print(f"  Partition: {partition}\n")

    # Constrained balanced min-cut
    print("=" * 60)
    print("Constrained Balanced Min-Cut:")
    print("=" * 60)
    for frac in [0.25, 0.20, 0.15]:
        min_size = max(2, int(G.number_of_nodes() * frac))
        print(f"\nmin_fraction={frac} (requires each side >= {min_size} nodes):")

        cut_val, partition = balanced_mincut(G, frac)
        if cut_val is not None:
            print(f"  Min-cut: {cut_val}")
            print(f"  Partition sizes: {len(partition[0])}, {len(partition[1])}")
            print(f"  Partition: {partition}")
        else:
            print("  No balanced cut found")

    # Fiedler vector
    print("\n" + "=" * 60)
    print("Fiedler Vector (Spectral Bisection):")
    print("=" * 60)
    cut_size, (partition_a, partition_b), cut_edges = fiedler_bisection(G)
    print(f"  Partition sizes: {len(partition_a)}, {len(partition_b)}")
    print(f"  Partition A: {partition_a}")
    print(f"  Partition B: {partition_b}")
    print(f"  Cut edges: {cut_edges}")
    print(f"  Cut size: {cut_size}")


if __name__ == "__main__":
    main()
