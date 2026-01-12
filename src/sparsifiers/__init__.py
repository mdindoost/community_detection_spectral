"""Graph sparsification algorithms."""
import time
from typing import Tuple, Optional
import igraph as ig

from .dspar import dspar_sparsify, dspar_sparsify_timed
from .baseline import (
    uniform_random_sparsify,
    uniform_random_sparsify_timed,
    degree_sampling_sparsify,
    degree_sampling_sparsify_timed,
    sparsify as baseline_sparsify,
    sparsify_timed as baseline_sparsify_timed
)

# Try to import spectral sparsification (requires Julia)
try:
    from .spectral import spectral_sparsify, spectral_sparsify_timed
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False
    spectral_sparsify = None
    spectral_sparsify_timed = None


def sparsify(
    G: ig.Graph,
    method: str,
    alpha: float,
    seed: Optional[int] = None,
    epsilon_map: Optional[dict] = None
) -> Optional[ig.Graph]:
    """
    Unified sparsification dispatcher.
    
    Parameters
    ----------
    G : ig.Graph
        Input graph
    method : str
        Sparsification method: 'dspar', 'uniform_random', 'degree_sampling', 'spectral'
    alpha : float
        Retention fraction (0, 1]
    seed : int, optional
        Random seed for reproducibility
    epsilon_map : dict, optional
        Mapping from alpha to epsilon for spectral sparsification
        
    Returns
    -------
    G_sparse : ig.Graph or None
        Sparsified graph. None if sparsification failed.
    """
    if method == 'dspar':
        G_sparse = dspar_sparsify(G, retention=alpha, method='paper', seed=seed)
        # Convert weighted to unweighted (topology only)
        if G_sparse.is_weighted():
            edges = G_sparse.get_edgelist()
            G_sparse = ig.Graph(n=G_sparse.vcount(), edges=edges, directed=False)
        return G_sparse
    
    elif method == 'uniform_random':
        return uniform_random_sparsify(G, retention=alpha, seed=seed)
    
    elif method == 'degree_sampling':
        return degree_sampling_sparsify(G, retention=alpha, seed=seed)
    
    elif method == 'spectral':
        if not SPECTRAL_AVAILABLE:
            raise RuntimeError("Spectral sparsification not available (Julia not configured)")
        
        # Get epsilon from map or use default
        if epsilon_map is not None:
            epsilon = epsilon_map.get(alpha)
        else:
            # Default epsilon mapping
            epsilon = {0.2: 3.0, 0.4: 1.5, 0.6: 0.8, 0.8: 0.3, 1.0: 0.0}.get(alpha, 0.5)
        
        if epsilon is None or epsilon <= 0:
            return G.copy()
        
        try:
            return spectral_sparsify(G, epsilon)
        except Exception as e:
            print(f"\n    [SPECTRAL ERROR] {e}")
            return None
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'dspar', 'uniform_random', 'degree_sampling', or 'spectral'")


def sparsify_timed(
    G: ig.Graph,
    method: str,
    alpha: float,
    seed: Optional[int] = None,
    epsilon_map: Optional[dict] = None
) -> Tuple[Optional[ig.Graph], float]:
    """
    Unified sparsification dispatcher with timing.
    
    Parameters
    ----------
    G : ig.Graph
        Input graph
    method : str
        Sparsification method
    alpha : float
        Retention fraction (0, 1]
    seed : int, optional
        Random seed
    epsilon_map : dict, optional
        Mapping from alpha to epsilon for spectral
        
    Returns
    -------
    G_sparse : ig.Graph or None
        Sparsified graph
    elapsed : float
        Time in seconds
    """
    start = time.perf_counter()
    G_sparse = sparsify(G, method, alpha, seed, epsilon_map)
    elapsed = time.perf_counter() - start
    return G_sparse, elapsed
