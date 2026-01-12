"""
Baseline sparsification methods for comparison.

All methods use numpy vectorization for performance - no Python for loops.
Uses igraph for graph representation.
"""
import time
import numpy as np
import igraph as ig
from typing import Tuple, Optional


def uniform_random_sparsify(
    G: ig.Graph,
    retention: float = 0.75,
    seed: Optional[int] = None
) -> ig.Graph:
    """
    Uniform random edge sampling without replacement.
    Each edge has equal probability of being kept.
    
    Parameters
    ----------
    G : ig.Graph
        Input graph
    retention : float
        Fraction of edges to keep
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G_sparse : ig.Graph
        Sparsified graph
    """
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    
    if seed is not None:
        np.random.seed(seed)
    
    edge_list = np.array(G.get_edgelist())
    m = len(edge_list)
    n_nodes = G.vcount()
    n_keep = int(np.ceil(retention * m))
    
    if n_keep >= m:
        return G.copy()
    
    # Sample without replacement (vectorized)
    keep_indices = np.random.choice(m, size=n_keep, replace=False)
    
    # Build graph
    kept_edges = edge_list[keep_indices].tolist()
    G_sparse = ig.Graph(n=n_nodes, edges=kept_edges, directed=False)
    
    return G_sparse


def uniform_random_sparsify_timed(
    G: ig.Graph,
    retention: float = 0.75,
    seed: Optional[int] = None
) -> Tuple[ig.Graph, float]:
    """Uniform random sparsification with timing."""
    start = time.perf_counter()
    G_sparse = uniform_random_sparsify(G, retention, seed)
    elapsed = time.perf_counter() - start
    return G_sparse, elapsed


def degree_sampling_sparsify(
    G: ig.Graph,
    retention: float = 0.75,
    seed: Optional[int] = None
) -> ig.Graph:
    """
    Degree-aware edge sampling.
    Edge (u,v) kept with probability proportional to deg(u) + deg(v).
    Normalized to achieve approximately retention * m edges.
    
    Parameters
    ----------
    G : ig.Graph
        Input graph
    retention : float
        Expected fraction of edges to keep
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G_sparse : ig.Graph
        Sparsified graph
    """
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Get edge list and degrees (vectorized)
    edge_list = np.array(G.get_edgelist())
    m = len(edge_list)
    n_nodes = G.vcount()
    degrees = np.array(G.degree())
    
    sources = edge_list[:, 0]
    targets = edge_list[:, 1]
    
    # Compute sampling weights proportional to deg(u) + deg(v) (vectorized)
    weights = degrees[sources] + degrees[targets]
    weights = weights.astype(np.float64)
    
    # Normalize to get expected retention * m edges
    n_keep = int(np.ceil(retention * m))
    probs = weights / weights.sum() * n_keep
    probs = np.clip(probs, 0, 1)
    
    # Sample each edge independently (vectorized Bernoulli)
    keep_mask = np.random.random(m) < probs
    
    # Build graph
    kept_edges = edge_list[keep_mask].tolist()
    G_sparse = ig.Graph(n=n_nodes, edges=kept_edges, directed=False)
    
    return G_sparse


def degree_sampling_sparsify_timed(
    G: ig.Graph,
    retention: float = 0.75,
    seed: Optional[int] = None
) -> Tuple[ig.Graph, float]:
    """Degree sampling sparsification with timing."""
    start = time.perf_counter()
    G_sparse = degree_sampling_sparsify(G, retention, seed)
    elapsed = time.perf_counter() - start
    return G_sparse, elapsed


def sparsify(
    G: ig.Graph,
    method: str,
    retention: float = 0.75,
    seed: Optional[int] = None
) -> ig.Graph:
    """
    Dispatch to appropriate baseline sparsification method.
    
    Parameters
    ----------
    G : ig.Graph
        Input graph
    method : str
        'uniform_random' or 'degree_sampling'
    retention : float
        Fraction/expected fraction of edges to keep
    seed : int, optional
        Random seed
        
    Returns
    -------
    G_sparse : ig.Graph
        Sparsified graph
    """
    if method == 'uniform_random':
        return uniform_random_sparsify(G, retention, seed)
    elif method == 'degree_sampling':
        return degree_sampling_sparsify(G, retention, seed)
    else:
        raise ValueError(f"Unknown baseline method: {method}. Use 'uniform_random' or 'degree_sampling'")


def sparsify_timed(
    G: ig.Graph,
    method: str,
    retention: float = 0.75,
    seed: Optional[int] = None
) -> Tuple[ig.Graph, float]:
    """
    Baseline sparsification with timing.
    
    Parameters
    ----------
    G : ig.Graph
        Input graph
    method : str
        'uniform_random' or 'degree_sampling'
    retention : float
        Fraction/expected fraction of edges to keep
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
    G_sparse = sparsify(G, method, retention, seed)
    elapsed = time.perf_counter() - start
    return G_sparse, elapsed
