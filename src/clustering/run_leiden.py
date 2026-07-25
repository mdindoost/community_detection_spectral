"""
Leiden community detection using igraph's built-in implementation.
"""
import time
from typing import Tuple, Optional, List
import igraph as ig


def run_leiden(
    g: ig.Graph,
    objective: str = "modularity",
    resolution: float = 1.0,
    n_iterations: int = 2,
    initial_membership: Optional[List[int]] = None
) -> Tuple[List[int], float, int]:
    """
    Run Leiden clustering using igraph's built-in method.

    Parameters
    ----------
    g : ig.Graph
        Input graph
    objective : str
        Objective function: "modularity" or "CPM"
    resolution : float
        Resolution parameter. Default 1.0 to match paper experiments.
    n_iterations : int
        Number of iterations. Default 2 to match paper experiments.
    initial_membership : list of int, optional
        Initial community assignment to seed Leiden with.

    Returns
    -------
    membership : list
        Community membership for each node
    modularity : float
        Modularity score of the partition
    n_communities : int
        Number of communities found
    """
    kwargs = dict(
        objective_function=objective,
        resolution=resolution,
        n_iterations=n_iterations,
    )
    if initial_membership is not None:
        kwargs['initial_membership'] = initial_membership

    partition = g.community_leiden(**kwargs)

    membership = partition.membership
    modularity = partition.modularity
    n_communities = len(partition)

    return membership, modularity, n_communities


def run_leiden_timed(
    g: ig.Graph,
    objective: str = "modularity",
    resolution: float = 1.0,
    n_iterations: int = 2
) -> Tuple[List[int], float, int, float]:
    """
    Run Leiden clustering with timing.
    
    Parameters
    ----------
    g : ig.Graph
        Input graph
    objective : str
        Objective function: "modularity" or "CPM"
    resolution : float
        Resolution parameter. Default 1.0 to match paper experiments.
    n_iterations : int
        Number of iterations. Default 2 to match paper experiments.
        
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
    membership, modularity, n_communities = run_leiden(g, objective, resolution, n_iterations)
    elapsed = time.perf_counter() - start
    
    return membership, modularity, n_communities, elapsed
