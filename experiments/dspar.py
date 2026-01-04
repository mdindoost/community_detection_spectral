"""
DSpar: Degree-Based Graph Sparsification

This module implements DSpar sparsification matching the original paper:
- Probabilistic sampling WITH replacement
- Edge reweighting to preserve spectral properties

DSpar Score Formula:
    s(e) = 1/d_u + 1/d_v

    Higher score = edge connects low-degree nodes (more important to keep)
    Lower score = edge connects high-degree hubs (less important)

Three methods available:

1. "paper" - Original DSpar paper method
   - Sampling: WITH replacement (same edge can be sampled multiple times)
   - Output: Weighted graph (weight = count / (q * probability))
   - retention parameter: Number of SAMPLES to draw = ceil(retention * m)
   - Actual edges kept: LESS than retention (duplicates merged into weights)
   - Theory: Preserves spectral properties, unbiased graph estimation
   - Example: retention=0.75, m=100 edges -> draw 75 samples -> ~50-60 unique edges

2. "probabilistic_no_replace" - Probabilistic WITHOUT replacement
   - Sampling: Each edge sampled independently with probability ~ score
   - Output: Unweighted graph
   - retention parameter: EXPECTED fraction of edges to keep
   - Actual edges kept: Approximately retention (varies each run)
   - Theory: Biased toward high-score edges, no spectral guarantees
   - Example: retention=0.75, m=100 edges -> ~70-80 edges (varies)

3. "deterministic" - Top-k by score (no randomness)
   - Sampling: Sort edges by score, keep top-k
   - Output: Unweighted graph
   - retention parameter: EXACT fraction of edges to keep
   - Actual edges kept: Exactly ceil(retention * m)
   - Theory: No spectral guarantees, but reproducible
   - Example: retention=0.75, m=100 edges -> exactly 75 edges

Usage:
    from dspar import dspar_sparsify

    # Original paper method (recommended for theory)
    G_sparse = dspar_sparsify(G, retention=0.75, method="paper")

    # Probabilistic without replacement
    G_sparse = dspar_sparsify(G, retention=0.75, method="probabilistic_no_replace")

    # Simplified deterministic (for reproducibility)
    G_sparse = dspar_sparsify(G, retention=0.75, method="deterministic")
"""

import numpy as np
import networkx as nx
from typing import Union, Tuple, Dict, Optional
from collections import Counter


def compute_dspar_scores(G: nx.Graph) -> Dict[Tuple[int, int], float]:
    """
    Compute DSpar score for each edge.
    
    DSpar score: s(e) = 1/d_u + 1/d_v
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    scores : dict
        Edge tuple -> DSpar score
    """
    degrees = dict(G.degree())
    scores = {}
    
    for u, v in G.edges():
        score = 1.0 / degrees[u] + 1.0 / degrees[v]
        edge = (min(u, v), max(u, v))
        scores[edge] = score
    
    return scores


def dspar_sparsify(
    G: nx.Graph,
    retention: float = 0.75,
    method: str = "paper",
    seed: Optional[int] = None,
    return_weights: bool = False
) -> Union[nx.Graph, Tuple[nx.Graph, Dict]]:
    """
    Sparsify graph using DSpar.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    retention : float
        Fraction of edges to keep (controls number of samples)
        
    method : str
        - "paper": Original paper method (probabilistic, WITH replacement, reweighted)
        - "probabilistic_no_replace": Probabilistic WITHOUT replacement
        - "deterministic": Top-k edges by score (no randomness)
        
    seed : int, optional
        Random seed
        
    return_weights : bool
        If True, also return edge weights dictionary
        
    Returns
    -------
    G_sparse : nx.Graph
        Sparsified graph (with edge weights if method="paper")
        
    weights : dict (only if return_weights=True)
        Edge weights after reweighting
    """
    
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Compute DSpar scores
    scores = compute_dspar_scores(G)
    edges = list(scores.keys())
    score_values = np.array([scores[e] for e in edges])
    
    m = len(edges)  # Original number of edges
    
    if method == "paper":
        # ============================================================
        # ORIGINAL PAPER METHOD: WITH Replacement + Reweighting
        # ============================================================
        
        # Step 1: Compute sampling probabilities
        probs = score_values / score_values.sum()
        
        # Step 2: Number of samples to draw
        q = int(np.ceil(retention * m))
        
        # Step 3: Sample WITH replacement
        # Each sample independently draws an edge according to probs
        sampled_indices = np.random.choice(
            len(edges), 
            size=q, 
            replace=True,  # WITH replacement
            p=probs
        )
        
        # Step 4: Count how many times each edge was sampled
        edge_counts = Counter(sampled_indices)
        
        # Step 5: Compute new weights
        # w'_e = k_e / (q * p_e)
        # This ensures unbiased estimation of original graph
        weights = {}
        for idx, count in edge_counts.items():
            edge = edges[idx]
            p_e = probs[idx]
            w_e = count / (q * p_e)
            weights[edge] = w_e
        
        # Step 6: Build weighted graph
        G_sparse = nx.Graph()
        G_sparse.add_nodes_from(G.nodes(data=True))
        
        for edge, weight in weights.items():
            u, v = edge
            G_sparse.add_edge(u, v, weight=weight)
        
        if return_weights:
            return G_sparse, weights
        return G_sparse
    
    elif method == "probabilistic_no_replace":
        # ============================================================
        # WITHOUT Replacement (my previous implementation)
        # ============================================================
        
        n_keep = int(np.ceil(retention * m))
        
        # Probabilities scaled to expected n_keep edges
        probs = score_values / score_values.sum() * n_keep
        probs = np.clip(probs, 0, 1)
        
        # Sample each edge independently
        random_vals = np.random.random(len(edges))
        edges_to_keep = [e for e, p, r in zip(edges, probs, random_vals) if r < p]
        
        # Build graph
        G_sparse = nx.Graph()
        G_sparse.add_nodes_from(G.nodes(data=True))
        for u, v in edges_to_keep:
            G_sparse.add_edge(u, v)
        
        if return_weights:
            weights = {e: 1.0 for e in edges_to_keep}
            return G_sparse, weights
        return G_sparse
    
    elif method == "deterministic":
        # ============================================================
        # DETERMINISTIC: Top-k by score
        # ============================================================
        
        n_keep = int(np.ceil(retention * m))
        
        # Sort by score (highest first)
        sorted_indices = np.argsort(score_values)[::-1]
        keep_indices = sorted_indices[:n_keep]
        edges_to_keep = [edges[i] for i in keep_indices]
        
        # Build graph
        G_sparse = nx.Graph()
        G_sparse.add_nodes_from(G.nodes(data=True))
        for u, v in edges_to_keep:
            G_sparse.add_edge(u, v)
        
        if return_weights:
            weights = {e: 1.0 for e in edges_to_keep}
            return G_sparse, weights
        return G_sparse
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compare_methods(G: nx.Graph, retention: float = 0.75, seed: int = 42):
    """
    Compare different sparsification methods.
    """
    print(f"\nOriginal graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Retention: {retention:.0%}")
    print("-" * 60)
    
    # Method 1: Paper (with replacement)
    G1, weights1 = dspar_sparsify(G, retention, method="paper", seed=seed, return_weights=True)
    total_weight = sum(weights1.values())
    print(f"Paper method (WITH replacement):")
    print(f"  Edges kept: {G1.number_of_edges()}")
    print(f"  Total weight: {total_weight:.2f}")
    print(f"  Weight range: [{min(weights1.values()):.3f}, {max(weights1.values()):.3f}]")
    
    # Method 2: Probabilistic without replacement
    G2 = dspar_sparsify(G, retention, method="probabilistic_no_replace", seed=seed)
    print(f"\nProbabilistic (WITHOUT replacement):")
    print(f"  Edges kept: {G2.number_of_edges()}")
    
    # Method 3: Deterministic
    G3 = dspar_sparsify(G, retention, method="deterministic")
    print(f"\nDeterministic (top-k):")
    print(f"  Edges kept: {G3.number_of_edges()}")
    
    return G1, G2, G3


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("DSpar Method Comparison")
    print("=" * 60)
    
    # Test with Karate Club
    G = nx.karate_club_graph()
    
    G1, G2, G3 = compare_methods(G, retention=0.75, seed=42)
    
    print("\n" + "=" * 60)
    print("Method Comparison Table")
    print("=" * 60)
    print(f"{'Method':<30} {'Edges':<10} {'Notes'}")
    print("-" * 60)
    print(f"{'Original':<30} {G.number_of_edges():<10} {'All edges'}")
    print(f"{'Paper (with replacement)':<30} {G1.number_of_edges():<10} {'Weighted, theoretical'}")
    print(f"{'Probabilistic (no replace)':<30} {G2.number_of_edges():<10} {'Unweighted'}")
    print(f"{'Deterministic':<30} {G3.number_of_edges():<10} {'Reproducible'}")
    
    # Show some edge weights from paper method
    print("\n" + "=" * 60)
    print("Edge Weights (Paper Method)")
    print("=" * 60)
    
    G1_weighted, weights = dspar_sparsify(G, retention=0.75, method="paper", seed=42, return_weights=True)
    
    # Sort by weight
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Edge':<15} {'Weight':<10} {'Interpretation'}")
    print("-" * 45)
    for (u, v), w in sorted_weights[:5]:
        interpretation = "Sampled multiple times" if w > 1.5 else "Sampled ~once" if w > 0.7 else "Downweighted"
        print(f"({u:2d}, {v:2d})       {w:<10.3f} {interpretation}")
    print("...")
    for (u, v), w in sorted_weights[-3:]:
        interpretation = "Sampled multiple times" if w > 1.5 else "Sampled ~once" if w > 0.7 else "Downweighted"
        print(f"({u:2d}, {v:2d})       {w:<10.3f} {interpretation}")
