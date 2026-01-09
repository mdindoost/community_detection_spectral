"""
Leiden result caching for original graph community detection.

Caches results to avoid recomputing Leiden on the same graph multiple times.
Results are stored in: results/original_leiden/{dataset}_{objective}[_{resolution}].txt
"""
import time
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import igraph as ig

# Use absolute imports for running as script
import sys
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import ORIGINAL_LEIDEN_DIR
    from src.clustering.run_leiden import run_leiden
else:
    from ..config import ORIGINAL_LEIDEN_DIR
    from .run_leiden import run_leiden


def _get_cache_filename(dataset_name: str, objective: str, resolution: Optional[float] = None) -> str:
    """
    Generate cache filename for Leiden result.
    
    Format: {dataset}_{objective}[_{resolution}].txt
    """
    if objective == "modularity":
        return f"{dataset_name}_modularity.txt"
    else:
        # Format resolution to avoid issues with decimal points in filename
        res_str = str(resolution).replace(".", "_")
        return f"{dataset_name}_CPM_{res_str}.txt"


def _get_cache_path(dataset_name: str, objective: str, resolution: Optional[float] = None) -> Path:
    """Get full path to cache file."""
    filename = _get_cache_filename(dataset_name, objective, resolution)
    return ORIGINAL_LEIDEN_DIR / filename


def save_leiden_result(
    membership: List[int],
    dataset_name: str,
    objective: str,
    resolution: Optional[float] = None
) -> Path:
    """
    Save Leiden membership to cache file.
    
    Parameters
    ----------
    membership : list
        Community membership for each node
    dataset_name : str
        Name of the dataset
    objective : str
        "modularity" or "CPM"
    resolution : float, optional
        Resolution parameter (for CPM)
        
    Returns
    -------
    filepath : Path
        Path to saved file
    """
    ORIGINAL_LEIDEN_DIR.mkdir(parents=True, exist_ok=True)
    filepath = _get_cache_path(dataset_name, objective, resolution)
    
    # Save using numpy for speed (node_id is implicit from row index)
    mem_arr = np.array(membership, dtype=np.int64)
    node_ids = np.arange(len(mem_arr), dtype=np.int64)
    data = np.column_stack([node_ids, mem_arr])
    np.savetxt(filepath, data, fmt='%d', delimiter='\t')
    
    return filepath


def load_leiden_result(
    dataset_name: str,
    objective: str,
    resolution: Optional[float] = None
) -> Optional[List[int]]:
    """
    Load Leiden membership from cache file if exists.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    objective : str
        "modularity" or "CPM"
    resolution : float, optional
        Resolution parameter (for CPM)
        
    Returns
    -------
    membership : list or None
        Community membership if cache exists, None otherwise
    """
    filepath = _get_cache_path(dataset_name, objective, resolution)
    
    if not filepath.exists():
        return None
    
    # Load membership from file using numpy
    try:
        data = np.loadtxt(filepath, dtype=np.int64, delimiter='\t')
        if data.ndim == 1:
            # Single row case
            membership = [int(data[1])]
        else:
            # Multiple rows: second column is community_id
            membership = data[:, 1].tolist()
        return membership if membership else None
    except Exception:
        return None


def is_leiden_cached(
    dataset_name: str,
    objective: str,
    resolution: Optional[float] = None
) -> bool:
    """Check if Leiden result is already cached."""
    filepath = _get_cache_path(dataset_name, objective, resolution)
    return filepath.exists()


def run_leiden_cached(
    G: ig.Graph,
    dataset_name: str,
    objective: str = "modularity",
    resolution: Optional[float] = None
) -> Tuple[List[int], float, int, float, bool]:
    """
    Run Leiden with caching - load from cache if available, compute otherwise.
    
    Parameters
    ----------
    G : ig.Graph
        Input graph
    dataset_name : str
        Name of the dataset (for cache key)
    objective : str
        "modularity" or "CPM"
    resolution : float, optional
        Resolution parameter (for CPM)
        
    Returns
    -------
    membership : list
        Community membership for each node
    modularity : float
        Modularity score
    n_communities : int
        Number of communities
    elapsed : float
        Time in seconds (0 if loaded from cache)
    from_cache : bool
        True if result was loaded from cache
    """
    # Try to load from cache
    cached_membership = load_leiden_result(dataset_name, objective, resolution)
    
    if cached_membership is not None:
        # Verify the membership length matches graph size
        if len(cached_membership) == G.vcount():
            modularity = G.modularity(cached_membership)
            n_communities = len(set(cached_membership))
            return cached_membership, modularity, n_communities, 0.0, True
    
    # Compute Leiden
    start = time.perf_counter()
    membership, modularity, n_communities = run_leiden(G, objective, resolution)
    elapsed = time.perf_counter() - start
    
    # Save to cache
    save_leiden_result(membership, dataset_name, objective, resolution)
    
    return membership, modularity, n_communities, elapsed, False
