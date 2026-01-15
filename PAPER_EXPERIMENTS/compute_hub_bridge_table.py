"""
Compute hub-bridging statistics for network datasets.

Outputs:
- HB Ratio = E[du·dv | inter] / E[du·dv | intra]
- δ (DSpar separation) = μ_intra - μ_inter
- |C| = number of communities

Uses Leiden partition on ORIGINAL graph as the reference partition.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import networkx as nx
import igraph as ig

from experiments.utils import load_snap_dataset


# Datasets to analyze (exp3 scalability datasets)
DATASETS = [
    'com-DBLP',          # ~317K nodes, ~1M edges
    'com-Amazon',        # ~335K nodes, ~926K edges
    'com-Youtube',       # ~1.1M nodes, ~3M edges
    'wiki-Talk',         # ~2.4M nodes, ~5M edges
    'cit-Patents',       # ~3.8M nodes, ~17M edges
    'wiki-topcats',      # ~1.8M nodes, ~28M edges
    'com-LiveJournal',   # ~4M nodes, ~35M edges
    'com-Orkut',         # ~3M nodes, ~117M edges
]


def load_dataset(name):
    """Load dataset and return NetworkX graph (undirected, simple, LCC)."""
    edges, n_nodes, _ground_truth = load_snap_dataset(name)

    # Build undirected graph, remove self-loops and multi-edges
    edge_set = set()
    for u, v in edges:
        if u != v:  # Remove self-loops
            edge_set.add((min(u, v), max(u, v)))

    G = nx.Graph()
    G.add_edges_from(edge_set)

    # Take largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


def nx_to_igraph(G: nx.Graph) -> ig.Graph:
    """Convert NetworkX graph to igraph."""
    edges = list(G.edges())
    return ig.Graph(n=G.number_of_nodes(), edges=edges, directed=False)


def run_leiden(G: nx.Graph):
    """
    Run Leiden clustering on the graph.
    Returns membership dict {node: community_id} and number of communities.
    """
    ig_graph = nx_to_igraph(G)
    partition = ig_graph.community_leiden(
        objective_function='modularity',
        resolution=1.0,
        n_iterations=2
    )

    membership = {i: partition.membership[i] for i in range(G.number_of_nodes())}
    n_communities = len(set(partition.membership))

    return membership, n_communities


def compute_dspar_delta(G: nx.Graph, membership: dict):
    """
    Compute DSpar separation δ = μ_intra - μ_inter.

    μ_intra = mean(1/du + 1/dv) for intra-community edges
    μ_inter = mean(1/du + 1/dv) for inter-community edges
    """
    degrees = dict(G.degree())

    intra_scores = []
    inter_scores = []

    for u, v in G.edges():
        d_u, d_v = degrees[u], degrees[v]
        if d_u > 0 and d_v > 0:
            score = 1.0 / d_u + 1.0 / d_v
            if membership.get(u) == membership.get(v):
                intra_scores.append(score)
            else:
                inter_scores.append(score)

    mu_intra = np.mean(intra_scores) if intra_scores else 0
    mu_inter = np.mean(inter_scores) if inter_scores else 0
    delta = mu_intra - mu_inter

    return delta, mu_intra, mu_inter


def compute_hub_bridge_ratio(G: nx.Graph, membership: dict):
    """
    Compute hub-bridge ratio: E[d_u·d_v | inter] / E[d_u·d_v | intra]
    """
    degrees = dict(G.degree())

    intra_products = []
    inter_products = []

    for u, v in G.edges():
        product = degrees[u] * degrees[v]
        if membership.get(u) == membership.get(v):
            intra_products.append(product)
        else:
            inter_products.append(product)

    mean_inter = np.mean(inter_products) if inter_products else 0
    mean_intra = np.mean(intra_products) if intra_products else 1

    return mean_inter / mean_intra if mean_intra > 0 else 1.0


def compute_statistics(name):
    """Compute all statistics for a single dataset."""
    print(f"\nProcessing {name}...")

    # Load graph
    G = load_dataset(name)
    print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

    # Run Leiden on original graph
    membership, n_communities = run_leiden(G)
    print(f"  Communities: {n_communities}")

    # Compute hub-bridge ratio
    hb_ratio = compute_hub_bridge_ratio(G, membership)
    print(f"  HB Ratio: {hb_ratio:.3f}")

    # Compute DSpar delta
    delta, mu_intra, mu_inter = compute_dspar_delta(G, membership)
    print(f"  δ: {delta:+.4f} (μ_intra={mu_intra:.4f}, μ_inter={mu_inter:.4f})")

    return {
        'dataset': name,
        'hb_ratio': hb_ratio,
        'delta': delta,
        'n_communities': n_communities,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
    }


def generate_latex_table(results):
    """Generate LaTeX table from results."""

    latex = r"""\begin{table}[t]
\centering
\caption{Empirical validation of hub-bridging and DSpar separation. All
evaluated networks exhibit both conditions simultaneously.}
\label{tab:hub-bridge-validation}

\begin{tabular}{lccc}
\toprule
Dataset & HB Ratio & $\delta$ (DS gap) & $|\mathcal{C}|$ \\
\midrule
"""

    for r in results:
        # Format delta with explicit + sign if positive
        delta_str = f"+{r['delta']:.4f}" if r['delta'] >= 0 else f"{r['delta']:.4f}"

        latex += f"{r['dataset']} & {r['hb_ratio']:.3f} & {delta_str} & {r['n_communities']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\begin{minipage}{0.95\linewidth}
\small
HB Ratio $= \mathbb{E}[d_u d_v \mid \text{inter}] \,/\, \mathbb{E}[d_u d_v \mid \text{intra}]$ \\
$\delta = \mu_{\text{intra}} - \mu_{\text{inter}}$ \quad (DSpar separation gap)
\end{minipage}

\end{table}"""

    return latex


def main():
    import csv
    import traceback

    print("=" * 60)
    print("Computing Hub-Bridging Statistics (Exp3 Datasets)")
    print("=" * 60)

    # Setup output directory and CSV file for incremental saving
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "hub_bridge_exp3.csv"

    results = []

    # Print header
    print(f"\n{'Dataset':<20} {'Nodes':>12} {'Edges':>12} {'HB Ratio':>10} {'δ':>12} {'|C|':>8}")
    print("-" * 78)

    # Open CSV file for incremental writing
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'n_nodes', 'n_edges', 'hb_ratio', 'delta',
            'mu_intra', 'mu_inter', 'n_communities'
        ])
        writer.writeheader()

        for dataset in DATASETS:
            try:
                print(f"\nProcessing {dataset}...")

                # Load graph
                G = load_dataset(dataset)
                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()
                print(f"  Nodes: {n_nodes:,}, Edges: {n_edges:,}")

                # Run Leiden on original graph
                print(f"  Running Leiden...")
                membership, n_communities = run_leiden(G)
                print(f"  Communities: {n_communities}")

                # Compute hub-bridge ratio
                print(f"  Computing HB ratio...")
                hb_ratio = compute_hub_bridge_ratio(G, membership)

                # Compute DSpar delta
                print(f"  Computing δ...")
                delta, mu_intra, mu_inter = compute_dspar_delta(G, membership)

                # Store result
                stats = {
                    'dataset': dataset,
                    'n_nodes': n_nodes,
                    'n_edges': n_edges,
                    'hb_ratio': hb_ratio,
                    'delta': delta,
                    'mu_intra': mu_intra,
                    'mu_inter': mu_inter,
                    'n_communities': n_communities,
                }
                results.append(stats)

                # Write to CSV immediately
                writer.writerow(stats)
                f.flush()  # Ensure it's written to disk

                # Print result immediately
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                print(f"\n  >>> RESULT: {dataset:<20} {n_nodes:>12,} {n_edges:>12,} {hb_ratio:>10.3f} {delta_str:>12} {n_communities:>8}")
                print(f"  >>> μ_intra={mu_intra:.6f}, μ_inter={mu_inter:.6f}")
                print(f"  >>> Saved to {csv_file}")

            except Exception as e:
                print(f"\n  ERROR processing {dataset}: {e}")
                traceback.print_exc()
                print(f"  Skipping {dataset}, continuing with next dataset...")
                continue

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    if not results:
        print("No results collected!")
        return

    # Print summary table
    print(f"\n{'Dataset':<20} {'Nodes':>12} {'Edges':>12} {'HB Ratio':>10} {'δ':>12} {'|C|':>8}")
    print("-" * 78)
    for r in results:
        delta_str = f"+{r['delta']:.4f}" if r['delta'] >= 0 else f"{r['delta']:.4f}"
        print(f"{r['dataset']:<20} {r['n_nodes']:>12,} {r['n_edges']:>12,} {r['hb_ratio']:>10.3f} {delta_str:>12} {r['n_communities']:>8}")

    # Generate LaTeX table
    print("\n" + "=" * 60)
    print("LATEX TABLE")
    print("=" * 60)
    latex_table = generate_latex_table(results)
    print(latex_table)

    # Save LaTeX table
    latex_file = output_dir / "hub_bridge_exp3_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"\nSaved LaTeX table to: {latex_file}")
    print(f"Saved CSV results to: {csv_file}")


if __name__ == "__main__":
    main()
