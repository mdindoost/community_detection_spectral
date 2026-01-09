"""
Leiden community detection using igraph's built-in implementation.
"""
import time
from typing import Tuple, Optional, List
import igraph as ig


def run_leiden(
    g: ig.Graph,
    objective: str = "modularity",
    resolution: Optional[float] = None
) -> Tuple[List[int], float, int]:
    """
    Run Leiden clustering using igraph's built-in method.
    
    Parameters
    ----------
    g : ig.Graph
        Input graph
    objective : str
        Objective function: "modularity" or "CPM"
    resolution : float, optional
        Resolution parameter for CPM. Ignored if objective="modularity".
        
    Returns
    -------
    membership : list
        Community membership for each node
    modularity : float
        Modularity score of the partition
    n_communities : int
        Number of communities found
    """
    if objective == "modularity":
        partition = g.community_leiden(objective_function="modularity")
    else:
        partition = g.community_leiden(
            objective_function="CPM",
            resolution=resolution
        )
    
    membership = partition.membership
    modularity = partition.modularity  # Use pre-computed modularity from partition
    n_communities = len(partition)
    
    return membership, modularity, n_communities


def run_leiden_timed(
    g: ig.Graph,
    objective: str = "modularity",
    resolution: Optional[float] = None
) -> Tuple[List[int], float, int, float]:
    """
    Run Leiden clustering with timing.
    
    Parameters
    ----------
    g : ig.Graph
        Input graph
    objective : str
        Objective function: "modularity" or "CPM"
    resolution : float, optional
        Resolution parameter for CPM
        
    Returns
    -------
    membership : list
        Community membership for each node
    modularity : float
        Modularity score of the partition
    n_communities : int
        Number of communities found
    elapsed : float
        Time in seconds
    """
    start = time.perf_counter()
    membership, modularity, n_communities = run_leiden(g, objective, resolution)
    elapsed = time.perf_counter() - start
    
    return membership, modularity, n_communities, elapsed
