"""
Test networks for understanding spectral sparsification.

Creates synthetic networks and visualizes:
- Original graph with effective resistance (ER) labels
- Sparsified graph showing which edges were kept/removed
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import linalg
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import spectral_sparsify_direct, PROJECT_ROOT


def compute_effective_resistances(G):
    """
    Compute effective resistance for each edge in the graph.

    R_eff(u,v) = L^+(u,u) + L^+(v,v) - 2*L^+(u,v)
    where L^+ is the Moore-Penrose pseudoinverse of the Laplacian.

    For MultiGraphs, parallel edges are treated as separate edges with same ER.
    """
    # Convert to simple graph for Laplacian computation
    if isinstance(G, nx.MultiGraph):
        G_simple = nx.Graph(G)
    else:
        G_simple = G

    n = G_simple.number_of_nodes()
    nodes = list(G_simple.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Compute Laplacian matrix
    L = nx.laplacian_matrix(G_simple).toarray().astype(float)

    # Compute pseudoinverse
    L_pinv = linalg.pinv(L)

    # Compute effective resistance for each unique node pair
    er = {}
    if isinstance(G, nx.MultiGraph):
        for u, v, key in G.edges(keys=True):
            i, j = node_to_idx[u], node_to_idx[v]
            r = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
            er[(u, v, key)] = r
            er[(v, u, key)] = r  # Symmetric
    else:
        for u, v in G.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            r = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
            er[(u, v)] = r
            er[(v, u)] = r  # Symmetric

    return er


def create_two_triangles_bridge():
    """
    Two triangles connected by one bridge.

        0---1         3---4
         \ /    ---    \ /
          2             5
    """
    G = nx.Graph()
    # Triangle 1: nodes 0, 1, 2
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Triangle 2: nodes 3, 4, 5
    G.add_edges_from([(3, 4), (4, 5), (5, 3)])
    # Bridge
    G.add_edge(2, 3)

    # Position for visualization
    pos = {
        0: (-2, 1), 1: (-1, 1), 2: (-1.5, 0),
        3: (1.5, 0), 4: (1, 1), 5: (2, 1)
    }

    return G, pos, "Two Triangles with Bridge"


def create_two_stars_one_bridge():
    """
    Two stars (1 center + 9 leaves each) connected by one bridge between centers.
    """
    G = nx.Graph()

    # Star 1: center=0, leaves=1-9
    center1 = 0
    for i in range(1, 10):
        G.add_edge(center1, i)

    # Star 2: center=10, leaves=11-19
    center2 = 10
    for i in range(11, 20):
        G.add_edge(center2, i)

    # Bridge between centers
    G.add_edge(center1, center2)

    # Position for visualization
    pos = {}
    # Star 1 on left
    pos[0] = (-3, 0)
    for i in range(1, 10):
        angle = 2 * np.pi * (i - 1) / 9
        pos[i] = (-3 + 1.5 * np.cos(angle), 1.5 * np.sin(angle))

    # Star 2 on right
    pos[10] = (3, 0)
    for i in range(11, 20):
        angle = 2 * np.pi * (i - 11) / 9
        pos[i] = (3 + 1.5 * np.cos(angle), 1.5 * np.sin(angle))

    return G, pos, "Two Stars with One Bridge"


def create_two_stars_two_bridges():
    """
    Two stars (1 center + 9 leaves each) connected by THREE bridges:
    - 2 direct bridges between centers (parallel edges)
    - 1 bridge through intermediate node 20

    Uses MultiGraph to support parallel edges.
    """
    G = nx.MultiGraph()

    # Star 1: center=0, leaves=1-9
    center1 = 0
    for i in range(1, 10):
        G.add_edge(center1, i)

    # Star 2: center=10, leaves=11-19
    center2 = 10
    for i in range(11, 20):
        G.add_edge(center2, i)

    # Bridge 1: direct connection
    G.add_edge(center1, center2)

    # Bridge 2: another direct connection (parallel edge)
    G.add_edge(center1, center2)

    # Bridge 3: through an intermediate node
    G.add_edge(center1, 20)
    G.add_edge(20, center2)

    # Position for visualization
    pos = {}
    # Star 1 on left
    pos[0] = (-3, 0)
    for i in range(1, 10):
        angle = 2 * np.pi * (i - 1) / 9
        pos[i] = (-3 + 1.5 * np.cos(angle), 1.5 * np.sin(angle))

    # Star 2 on right
    pos[10] = (3, 0)
    for i in range(11, 20):
        angle = 2 * np.pi * (i - 11) / 9
        pos[i] = (3 + 1.5 * np.cos(angle), 1.5 * np.sin(angle))

    # Bridge node in the middle
    pos[20] = (0, 0.8)

    return G, pos, "Two Stars with Three Bridges"


def create_k4_with_tail():
    """
    K4 (complete graph on 4 nodes) with a tail of 5 nodes.

    K4: nodes 0, 1, 2, 3 (all connected)
    Tail: 4-5-6-7-8 (chain attached to node 3)
    """
    G = nx.Graph()

    # K4: nodes 0, 1, 2, 3
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(i, j)

    # Tail: 4-5-6-7-8 attached to node 3
    G.add_edge(3, 4)
    G.add_edge(4, 5)
    G.add_edge(5, 6)
    G.add_edge(6, 7)
    G.add_edge(7, 8)

    # Position for visualization
    pos = {
        0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0),  # K4 as square
        4: (0, -1), 5: (0, -2), 6: (0, -3), 7: (0, -4), 8: (0, -5)  # Tail going down
    }

    return G, pos, "K4 with Tail (5 nodes)"


def run_sparsification(G, epsilon=1.0):
    """Run spectral sparsification and return kept edges."""
    # Convert MultiGraph to simple Graph for sparsification
    if isinstance(G, nx.MultiGraph):
        G_simple = nx.Graph(G)
    else:
        G_simple = G

    nodes = list(G_simple.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    n_nodes = len(nodes)

    # Convert to edge list format
    edges = []
    for u, v in G_simple.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        edges.append((i, j))
        edges.append((j, i))

    try:
        sparse_edges, _ = spectral_sparsify_direct(edges, n_nodes, epsilon)

        # Convert back to original node labels
        kept_edges = set()
        for i, j in sparse_edges:
            u, v = idx_to_node[i], idx_to_node[j]
            kept_edges.add((min(u, v), max(u, v)))

        return kept_edges
    except Exception as e:
        print(f"Sparsification failed: {e}")
        return None


def visualize_network(G, pos, title, er, kept_edges=None, filename=None):
    """
    Visualize network with effective resistance labels.

    If kept_edges is provided, show which edges were kept (green) vs removed (red dashed).
    Handles both Graph and MultiGraph.
    """
    is_multigraph = isinstance(G, nx.MultiGraph)

    fig, axes = plt.subplots(1, 2 if kept_edges is not None else 1, figsize=(14 if kept_edges else 8, 6))

    if kept_edges is None:
        axes = [axes]

    # Original graph
    ax = axes[0]
    ax.set_title(f"{title}\nOriginal Graph with Effective Resistance")

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue',
                          node_size=500, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    # Draw edges with ER labels
    if is_multigraph:
        # Track edge counts for offset of parallel edges
        edge_counts = {}
        for u, v, key in G.edges(keys=True):
            edge_pair = (min(u, v), max(u, v))
            if edge_pair not in edge_counts:
                edge_counts[edge_pair] = 0
            offset = edge_counts[edge_pair] * 0.15
            edge_counts[edge_pair] += 1

            x1, y1 = pos[u]
            x2, y2 = pos[v]
            # Offset for parallel edges
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                nx_off, ny_off = -dy/length * offset, dx/length * offset
            else:
                nx_off, ny_off = 0, offset

            ax.plot([x1 + nx_off, x2 + nx_off], [y1 + ny_off, y2 + ny_off], 'k-', linewidth=2, zorder=1)

            # Edge label (ER value)
            mid_x, mid_y = (x1 + x2) / 2 + nx_off, (y1 + y2) / 2 + ny_off
            r = er.get((u, v, key), er.get((v, u, key), 0))
            ax.annotate(f'{r:.3f}', (mid_x, mid_y), fontsize=8,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    else:
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)

            # Edge label (ER value)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            r = er.get((u, v), er.get((v, u), 0))
            ax.annotate(f'{r:.3f}', (mid_x, mid_y), fontsize=8,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

    ax.set_aspect('equal')
    ax.axis('off')

    # Sparsified graph
    if kept_edges is not None:
        ax = axes[1]

        # Get unique edges from original graph
        if is_multigraph:
            original_edges = set((min(u, v), max(u, v)) for u, v, _ in G.edges(keys=True))
        else:
            original_edges = set((min(u, v), max(u, v)) for u, v in G.edges())

        removed_edges = original_edges - kept_edges

        ax.set_title(f"After Sparsification\nKept: {len(kept_edges)}, Removed: {len(removed_edges)}")

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue',
                              node_size=500, edgecolors='black', linewidths=2)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

        # Draw kept edges (green)
        for u, v in kept_edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], 'g-', linewidth=3, zorder=1)

            # ER label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            if is_multigraph:
                # Get ER from any key for this edge pair
                r = 0
                for key in range(10):
                    if (u, v, key) in er:
                        r = er[(u, v, key)]
                        break
                    if (v, u, key) in er:
                        r = er[(v, u, key)]
                        break
            else:
                r = er.get((u, v), er.get((v, u), 0))
            ax.annotate(f'{r:.3f}', (mid_x, mid_y), fontsize=8,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))

        # Draw removed edges (red dashed)
        for u, v in removed_edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], 'r--', linewidth=2, alpha=0.5, zorder=1)

            # ER label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            if is_multigraph:
                r = 0
                for key in range(10):
                    if (u, v, key) in er:
                        r = er[(u, v, key)]
                        break
                    if (v, u, key) in er:
                        r = er[(v, u, key)]
                        break
            else:
                r = er.get((u, v), er.get((v, u), 0))
            ax.annotate(f'{r:.3f}', (mid_x, mid_y), fontsize=8,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))

        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")

    plt.close()


def print_edge_table(G, er, kept_edges=None):
    """Print table of edges with effective resistance."""
    is_multigraph = isinstance(G, nx.MultiGraph)

    print("\n  Edge             | Eff. Resistance | Status")
    print("  " + "-" * 50)

    if is_multigraph:
        edges = sorted(G.edges(keys=True), key=lambda x: (x[0], x[1], x[2]))
        for u, v, key in edges:
            r = er.get((u, v, key), er.get((v, u, key), 0))
            if kept_edges is not None:
                status = "KEPT" if (min(u, v), max(u, v)) in kept_edges else "REMOVED"
            else:
                status = "-"
            print(f"  ({u:2d}, {v:2d}) [k={key}] |     {r:.6f}    | {status}")
    else:
        edges = sorted(G.edges())
        for u, v in edges:
            r = er.get((u, v), er.get((v, u), 0))
            if kept_edges is not None:
                status = "KEPT" if (min(u, v), max(u, v)) in kept_edges else "REMOVED"
            else:
                status = "-"
            print(f"  ({u:2d}, {v:2d})        |     {r:.6f}    | {status}")


def main():
    """Main function to run all test networks."""

    output_dir = PROJECT_ROOT / "results" / "test_networks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define test networks
    networks = [
        create_two_triangles_bridge,
        create_two_stars_one_bridge,
        create_two_stars_two_bridges,
        create_k4_with_tail,
    ]

    epsilon = 1.0  # Sparsification parameter (change this value to test different levels)

    print("=" * 60)
    print("TEST NETWORKS - Spectral Sparsification Analysis")
    print(f"Epsilon: {epsilon}")
    print("=" * 60)

    for create_func in networks:
        G, pos, title = create_func()

        print(f"\n{'=' * 60}")
        print(f"Network: {title}")
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print("=" * 60)

        # Compute effective resistances
        er = compute_effective_resistances(G)

        # Print edge table (before sparsification)
        print("\nBefore Sparsification:")
        print_edge_table(G, er)

        # Run sparsification
        print(f"\nRunning spectral sparsification (epsilon={epsilon})...")
        kept_edges = run_sparsification(G, epsilon)

        if kept_edges is not None:
            print(f"\nAfter Sparsification:")
            print(f"  Kept edges: {len(kept_edges)} / {G.number_of_edges()}")
            print_edge_table(G, er, kept_edges)

            # Identify which edges were removed
            removed = set((min(u, v), max(u, v)) for u, v in G.edges()) - kept_edges
            if removed:
                print(f"\n  Removed edges: {sorted(removed)}")
                avg_er_removed = np.mean([er.get((u, v), er.get((v, u), 0)) for u, v in removed])
                avg_er_kept = np.mean([er.get((u, v), er.get((v, u), 0)) for u, v in kept_edges])
                print(f"  Avg ER of removed edges: {avg_er_removed:.4f}")
                print(f"  Avg ER of kept edges: {avg_er_kept:.4f}")

        # Visualize
        safe_title = title.replace(" ", "_").replace("(", "").replace(")", "")
        filename = output_dir / f"{safe_title}.png"
        visualize_network(G, pos, title, er, kept_edges, filename)

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
