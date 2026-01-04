"""
DSpar Demo: Visualize how DSpar and Spectral sparsification differ.

Compares:
1. DSpar (degree-based): score = 1/d_u + 1/d_v
2. Spectral (effective resistance): Julia Laplacians.jl sparsification

Run: python experiments/dspar_demo.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import linalg
from dspar import dspar_sparsify, compute_dspar_scores

# Import Julia spectral sparsification
try:
    from utils import spectral_sparsify_direct
    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False
    print("Warning: Julia spectral sparsification not available, using Python approximation")


def create_two_cliques_graph():
    """
    Create two cliques connected by a hub node.

    This graph shows where DSpar and Spectral DISAGREE:
    - Hub node 8 is a BRIDGE between cliques
    - DSpar: removes bridge edges (low score due to hub degree)
    - Spectral: keeps bridge edges (high ER, only path between cliques)
    """
    G = nx.Graph()

    # Left clique (nodes 0,1,2,3) - all connected (K4)
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

    # Right clique (nodes 4,5,6,7) - all connected (K4)
    G.add_edges_from([(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)])

    # Hub node 8 connects both cliques (high degree BRIDGE)
    G.add_edges_from([(8, 0), (8, 1), (8, 4), (8, 5)])

    return G


def compute_effective_resistance(G):
    """
    Compute effective resistance for each edge using the Laplacian pseudoinverse.

    Effective Resistance R(u,v) = L†[u,u] + L†[v,v] - 2*L†[u,v]
    where L† is the Moore-Penrose pseudoinverse of the Laplacian.

    Higher ER = edge is more important (fewer alternate paths)
    Lower ER = edge is less important (many alternate paths)
    """
    L = nx.laplacian_matrix(G).toarray().astype(float)
    L_pinv = linalg.pinv(L)

    er_scores = {}
    for u, v in G.edges():
        er = L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v]
        edge = (min(u, v), max(u, v))
        er_scores[edge] = er

    return er_scores


def spectral_sparsify_julia(G, epsilon=1.0):
    """
    Call Julia Laplacians.jl for spectral sparsification.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    epsilon : float
        Sparsification parameter. Smaller = more edges retained.
        Typical values: 0.3 (keep ~80%), 0.5 (keep ~60%), 1.0 (keep ~40%)

    Returns
    -------
    G_sparse : nx.Graph
        Sparsified graph
    """
    if not JULIA_AVAILABLE:
        raise RuntimeError("Julia spectral sparsification not available. Run setup_julia.sh first.")

    # Convert graph to edge list (both directions for Julia)
    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))

    n_nodes = G.number_of_nodes()

    # Call Julia
    sparsified_edges, elapsed = spectral_sparsify_direct(edges, n_nodes, epsilon)

    # Convert back to NetworkX graph
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes())

    # Add edges (sparsified_edges contains both directions)
    for u, v in sparsified_edges:
        if u < v:  # Avoid duplicates
            G_sparse.add_edge(u, v)

    return G_sparse


def draw_graph_with_scores(G, scores, ax, title, kept_edges=None, weights=None, pos=None):
    """Draw graph with scores on edges, highlighting kept/removed edges."""

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    if kept_edges:
        kept_edges_norm = {(min(u, v), max(u, v)) for u, v in kept_edges}
    else:
        kept_edges_norm = None

    # Draw nodes
    degrees = dict(G.degree())
    node_sizes = [300 + degrees[n] * 200 for n in G.nodes()]
    node_colors = ['red' if degrees[n] >= 4 else 'lightblue' for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='black')

    labels = {n: f"{n}\n(d={degrees[n]})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

    # Draw edges
    for u, v in G.edges():
        edge_norm = (min(u, v), max(u, v))
        score = scores.get(edge_norm, 0)

        if kept_edges_norm is None:
            color = plt.cm.RdYlGn(score / 1.5)
            width = 1 + score * 2
            style = 'solid'
        elif edge_norm in kept_edges_norm:
            color = 'green'
            width = 2.5
            style = 'solid'
        else:
            color = 'red'
            width = 1.5
            style = 'dashed'

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                               edge_color=[color], width=width, style=style)

        # Add score label
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        offset_x = (pos[v][1] - pos[u][1]) * 0.1
        offset_y = (pos[u][0] - pos[v][0]) * 0.1

        if weights and edge_norm in weights:
            label = f"{score:.2f}\nw={weights[edge_norm]:.1f}"
        else:
            label = f"{score:.2f}"

        ax.text(mid_x + offset_x, mid_y + offset_y, label,
                fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')


def visualize_dspar_vs_spectral(G, pos, retention=0.5, epsilon=1.0, seed=42):
    """
    Create visualization comparing DSpar Paper method vs Julia Spectral.

    Parameters
    ----------
    retention : float
        DSpar retention parameter (fraction of samples to draw)
    epsilon : float
        Spectral epsilon parameter (smaller = more edges kept)
    """
    dspar_scores = compute_dspar_scores(G)
    er_scores = compute_effective_resistance(G)
    degrees = dict(G.degree())

    # DSpar Paper method
    G_dspar, dspar_weights = dspar_sparsify(
        G, retention=retention, method='paper', seed=seed, return_weights=True
    )
    dspar_edges = {(min(u, v), max(u, v)) for u, v in G_dspar.edges()}

    # Spectral method (Julia Laplacians.jl)
    G_spectral = spectral_sparsify_julia(G, epsilon=epsilon)
    spectral_edges = {(min(u, v), max(u, v)) for u, v in G_spectral.edges()}

    # Print comparison
    print("=" * 80)
    print("DSpar vs SPECTRAL COMPARISON")
    print("=" * 80)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"""
PARAMETERS:
  DSpar:    retention = {retention} (draw {int(np.ceil(retention * G.number_of_edges()))} samples)
  Spectral: epsilon = {epsilon}

METHODS:
  DSpar Paper: Sample WITH replacement, P(e) ~ 1/d_u + 1/d_v, then reweight
  Julia Spectral: Spielman-Srivastava algorithm (Laplacians.jl)
""")

    sorted_by_dspar = sorted(dspar_scores.items(), key=lambda x: -x[1])

    print(f"{'Edge':<8} {'Deg':<8} {'DSpar':<8} {'ER':<8} {'DSpar Paper':<20} {'Spectral'}")
    print("-" * 80)

    for (u, v), dspar in sorted_by_dspar:
        edge = (min(u, v), max(u, v))
        er = er_scores[edge]
        du, dv = degrees[u], degrees[v]

        if edge in dspar_edges:
            w = dspar_weights.get(edge, 1.0)
            dspar_status = f"KEPT (w={w:.2f})"
        else:
            dspar_status = "NOT SAMPLED"

        spectral_status = "KEPT" if edge in spectral_edges else "NOT SAMPLED"

        if (edge in dspar_edges) != (edge in spectral_edges):
            marker = " <-- DISAGREE!"
        else:
            marker = ""

        print(f"({u}, {v})    ({du},{dv})    {dspar:.3f}    {er:.3f}    {dspar_status:<20} {spectral_status}{marker}")

    # Summary
    common = dspar_edges & spectral_edges
    only_dspar = dspar_edges - spectral_edges
    only_spectral = spectral_edges - dspar_edges

    print(f"\nDSpar edges: {len(dspar_edges)}")
    print(f"Spectral edges: {len(spectral_edges)}")
    print(f"Common: {len(common)}")
    print(f"Only DSpar: {sorted(only_dspar)}")
    print(f"Only Spectral: {sorted(only_spectral)}")

    print(f"\nConnectivity:")
    print(f"  DSpar: {nx.is_connected(G_dspar)}")
    print(f"  Spectral: {nx.is_connected(G_spectral)}")

    # Correlation
    edges = list(dspar_scores.keys())
    x = [dspar_scores[e] for e in edges]
    y = [er_scores[e] for e in edges]
    corr = np.corrcoef(x, y)[0, 1]
    print(f"\nCorrelation (DSpar vs ER): r = {corr:.3f}")

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 0: Original scores
    draw_graph_with_scores(G, dspar_scores, axes[0, 0],
                          "DSpar Scores\ns(e) = 1/d_u + 1/d_v", pos=pos)

    draw_graph_with_scores(G, er_scores, axes[0, 1],
                          "Effective Resistance\nR(e) from Laplacian", pos=pos)

    # Correlation plot
    ax = axes[0, 2]
    colors = ['red' if 8 in e else 'blue' for e in edges]
    ax.scatter(x, y, s=100, c=colors, alpha=0.7)
    for i, e in enumerate(edges):
        ax.annotate(f"{e}", (x[i], y[i]), fontsize=7, ha='center', va='bottom')
    ax.set_xlabel("DSpar Score (degree-based)")
    ax.set_ylabel("Effective Resistance")
    ax.set_title(f"Correlation: r = {corr:.3f}\nRed = bridge edges to node 8")
    ax.grid(True, alpha=0.3)

    # Row 1: Sparsification results
    draw_graph_with_scores(G, dspar_scores, axes[1, 0],
                          f"DSpar PAPER (retention={retention})\n{len(dspar_edges)} edges kept",
                          kept_edges=dspar_edges, weights=dspar_weights, pos=pos)

    draw_graph_with_scores(G, er_scores, axes[1, 1],
                          f"SPECTRAL (epsilon={epsilon})\n{len(spectral_edges)} edges kept",
                          kept_edges=spectral_edges, pos=pos)

    # Difference visualization
    ax = axes[1, 2]
    node_colors = ['red' if n == 8 else ('orange' if degrees[n] >= 4 else 'lightblue') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500,
                           node_color=node_colors, edgecolors='black', linewidths=2)
    labels = {n: f"{n}\nd={degrees[n]}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

    for u, v in G.edges():
        edge = (min(u, v), max(u, v))
        in_dspar = edge in dspar_edges
        in_spectral = edge in spectral_edges

        if in_dspar and in_spectral:
            color, style, width = 'green', 'solid', 3
        elif in_dspar and not in_spectral:
            color, style, width = 'blue', 'solid', 3
        elif not in_dspar and in_spectral:
            color, style, width = 'orange', 'solid', 3
        else:
            color, style, width = 'lightgray', 'dashed', 1.5

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                               edge_color=[color], width=width, style=style)

    ax.set_title("DIFFERENCE\nGreen=both, Blue=DSpar only\nOrange=Spectral only, Gray=neither")
    ax.axis('off')

    fig.text(0.5, 0.02,
             "KEY: DSpar removes bridge edges (high degree hub -> low score). "
             "Spectral KEEPS bridge edges (high ER, critical for connectivity).",
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.suptitle("Two Cliques + Hub Bridge: DSpar vs Spectral", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    output_path = "experiments/dspar_vs_spectral.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    return fig


def main():
    print("""
################################################################################
#                    DSpar vs SPECTRAL SPARSIFICATION DEMO                     #
################################################################################

FORMULAS:
  DSpar Score:           s(e) = 1/d_u + 1/d_v      (degree-based)
  Effective Resistance:  R(e) = L+[u,u] + L+[v,v] - 2*L+[u,v]  (Laplacian-based)

INTERPRETATION:
  DSpar: Higher score = LOW degree nodes = peripheral edges = KEEP
  ER:    Higher ER = FEW alternate paths = bridge edges = KEEP

KEY INSIGHT:
  DSpar score != Effective Resistance
  They DISAGREE on high-degree bridges (hub connecting components)
""")

    # Two Cliques + Bridge graph
    G = create_two_cliques_graph()
    pos = {
        # Left clique
        0: (-2, 1), 1: (-2, -1), 2: (-3, 0), 3: (-1, 0),
        # Right clique
        4: (2, 1), 5: (2, -1), 6: (3, 0), 7: (1, 0),
        # Bridge hub
        8: (0, 0)
    }

    # Parameters:
    # - DSpar: retention=0.5 means draw 50% of edges as samples
    # - Spectral: epsilon=1.0 controls approximation quality
    visualize_dspar_vs_spectral(
        G, pos,
        retention=0.5,
        epsilon=1.0,
        seed=42
    )

    plt.show()


if __name__ == "__main__":
    main()
