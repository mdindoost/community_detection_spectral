"""
DSpar graph sparsification algorithm.

Implements DSpar scoring (s(e) = 1/d_u + 1/d_v) and three sparsification methods:
- paper: Original method with replacement and reweighting
- probabilistic_no_replace: Probabilistic without replacement  
- deterministic: Top-k edges by score
"""
import time
import numpy as np
import igraph as ig
from typing import Union, Tuple, Optional


def dspar_sparsify(
    G: ig.Graph,
    retention: float = 0.75,
    method: str = "paper",
    seed: Optional[int] = None,
    return_weights: bool = False
) -> Union[ig.Graph, Tuple[ig.Graph, np.ndarray]]:
    """
    Sparsify graph using DSpar.
    
    Parameters
    ----------
    G : ig.Graph
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
        If True, also return edge weights array
        
    Returns
    -------
    G_sparse : ig.Graph
        Sparsified graph (with edge weights if method="paper")
        
    weights : np.ndarray (only if return_weights=True)
        Edge weights after reweighting
    """
    
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Compute DSpar scores using numpy (vectorized edge access)
    degrees = np.array(G.degree())
    edge_list = np.array(G.get_edgelist())
    sources = edge_list[:, 0]
    targets = edge_list[:, 1]
    score_values = 1.0 / degrees[sources] + 1.0 / degrees[targets]
    
    m = len(score_values)  # Original number of edges
    n_nodes = G.vcount()
    
    if method == "paper":
        # ============================================================
        # ORIGINAL PAPER METHOD: WITH Replacement + Reweighting
        # ============================================================
        
        # Step 1: Compute sampling probabilities
        probs = score_values / score_values.sum()
        
        # Step 2: Number of samples to draw
        q = int(np.ceil(retention * m))
        
        # Step 3: Sample WITH replacement
        sampled_indices = np.random.choice(m, size=q, replace=True, p=probs)
        
        # Step 4: Count how many times each edge was sampled (vectorized)
        unique_indices, counts = np.unique(sampled_indices, return_counts=True)
        
        # Step 5: Compute new weights (vectorized)
        # w'_e = k_e / (q * p_e)
        weights = counts / (q * probs[unique_indices])
        
        # Step 6: Build weighted graph
        edge_list = np.column_stack([sources[unique_indices], targets[unique_indices]]).tolist()
        G_sparse = ig.Graph(n=n_nodes, edges=edge_list, directed=False)
        G_sparse.es['weight'] = weights
        
        if return_weights:
            return G_sparse, weights
        return G_sparse
    
    elif method == "probabilistic_no_replace":
        # ============================================================
        # WITHOUT Replacement (vectorized)
        # ============================================================
        
        n_keep = int(np.ceil(retention * m))
        
        # Probabilities scaled to expected n_keep edges
        probs = score_values / score_values.sum() * n_keep
        probs = np.clip(probs, 0, 1)
        
        # Sample each edge independently (vectorized)
        keep_mask = np.random.random(m) < probs
        keep_indices = np.where(keep_mask)[0]
        
        # Build graph
        edge_list = np.column_stack([sources[keep_indices], targets[keep_indices]]).tolist()
        G_sparse = ig.Graph(n=n_nodes, edges=edge_list, directed=False)
        
        if return_weights:
            weights = np.ones(len(keep_indices))
            return G_sparse, weights
        return G_sparse
    
    elif method == "deterministic":
        # ============================================================
        # DETERMINISTIC: Top-k by score (matches original paper code)
        # ============================================================
        
        n_keep = int(np.ceil(retention * m))
        
        # Sort by score (highest first) - use argsort for exact reproducibility
        # (argpartition is faster but doesn't preserve order among top-k)
        sorted_indices = np.argsort(score_values)[::-1]
        keep_indices = sorted_indices[:n_keep]
        
        # Build graph
        edge_list = np.column_stack([sources[keep_indices], targets[keep_indices]]).tolist()
        G_sparse = ig.Graph(n=n_nodes, edges=edge_list, directed=False)
        
        if return_weights:
            weights = np.ones(len(keep_indices))
            return G_sparse, weights
        return G_sparse
    
    else:
        raise ValueError(f"Unknown method: {method}")


def dspar_sparsify_timed(
    G: ig.Graph,
    retention: float = 0.75,
    method: str = "paper",
    seed: Optional[int] = None
) -> Tuple[ig.Graph, float]:
    """
    Sparsify graph using DSpar with timing.
    
    Parameters
    ----------
    G : ig.Graph
        Input graph
    retention : float
        Fraction of edges to keep
    method : str
        Sparsification method
    seed : int, optional
        Random seed
        
    Returns
    -------
    G_sparse : ig.Graph
        Sparsified graph
    elapsed : float
        Time in seconds
    """
    start = time.perf_counter()
    G_sparse = dspar_sparsify(G, retention=retention, method=method, seed=seed)
    elapsed = time.perf_counter() - start
    return G_sparse, elapsed

