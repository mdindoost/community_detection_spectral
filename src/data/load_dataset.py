"""
Dataset loading utilities using polars for fast I/O.
"""
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Support both package and script imports
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.datasets import DATASET_PATHS, DATASET_URLS, SNAP_DATASETS
    from src.data.snap_downloader import ensure_file_exists
    from src.config import SEED_BASE
else:
    from .datasets import DATASET_PATHS, DATASET_URLS, SNAP_DATASETS
    from .snap_downloader import ensure_file_exists
    from ..config import SEED_BASE

import numpy as np
import polars as pl
import igraph as ig


def load_edges(name: str) -> np.ndarray:
    """
    Load dataset edges as numpy array.
    
    Parameters
    ----------
    name : str
        Dataset name (cit-HepPh, cit-HepTh, citeseer, soc-LiveJournal1)
        
    Returns
    -------
    edges : np.ndarray
        Edge array of shape (n_edges, 2) with columns [source, target]
    """
    try:
        edge_file = DATASET_PATHS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_PATHS.keys())}")

    if not edge_file.exists():
        url = DATASET_URLS.get(name)
        if url is not None:
            ensure_file_exists(edge_file, url)
        else:
            raise FileNotFoundError(f"Dataset not found and no download URL: {edge_file}")

    # Determine separator based on dataset (wiki-topcats uses space, others use tab)
    separator = " " if name in ["wiki-topcats"] else "\t"
    
    df = pl.read_csv(
        edge_file,
        comment_prefix="#",
        separator=separator,
        has_header=False,
        new_columns=["source", "target"],
        schema_overrides={"source": pl.Int64, "target": pl.Int64},
    )

    df = df.filter(pl.col("source") != pl.col("target")).select(["source", "target"])
    
    unique_nodes = pl.concat([df["source"], df["target"]]).unique().sort()
    node_id_map = dict(zip(unique_nodes.to_list(), range(len(unique_nodes))))
    
    df = df.with_columns([
        pl.col("source").replace(node_id_map).alias("source"),
        pl.col("target").replace(node_id_map).alias("target")
    ])
    
    return df.to_numpy()


def load_graph(name: str) -> ig.Graph:
    """
    Load dataset as igraph Graph.
    
    Parameters
    ----------
    name : str
        Dataset name
        
    Returns
    -------
    G : ig.Graph
        Undirected igraph graph
    """
    edges = load_edges(name)
    G = ig.Graph(edges=edges.tolist(), directed=False)
    G.simplify()
    return G


# Keep backward compatibility
def load_dataset(name: str) -> np.ndarray:
    """Backward compatible function - returns edges array."""
    return load_edges(name)


def load_large_dataset(
    name: str, 
    max_edges: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Optional[ig.Graph], Optional[Dict[str, Any]]]:
    """
    Load a large dataset as igraph Graph.
    Ensures undirected, simple, largest connected component.
    
    Parameters
    ----------
    name : str
        Dataset name from SNAP_DATASETS registry
    max_edges : int, optional
        Maximum edges to keep (for testing). If None, keep all.
    verbose : bool
        Whether to print loading progress
        
    Returns
    -------
    G : ig.Graph or None
        Undirected, simple, connected graph. None if loading failed.
    info : dict or None
        Dataset statistics. None if loading failed.
    """
    if verbose:
        print(f"\n  Loading {name}...")
    
    if name not in SNAP_DATASETS:
        if verbose:
            print(f"    [WARNING] Dataset {name} not in registry, skipping")
        return None, None
    
    try:
        edges = load_edges(name)
    except Exception as e:
        if verbose:
            print(f"    [ERROR] Failed to load {name}: {e}")
        return None, None
    
    # Build igraph Graph (vectorized)
    sources = edges[:, 0]
    targets = edges[:, 1]
    
    # Remove self-loops (vectorized)
    mask = sources != targets
    self_loops_removed = int((~mask).sum())
    sources = sources[mask]
    targets = targets[mask]
    
    # Create graph
    n_nodes_raw = max(sources.max(), targets.max()) + 1
    edge_list = list(zip(sources.tolist(), targets.tolist()))
    G = ig.Graph(n=n_nodes_raw, edges=edge_list, directed=False)
    
    # Simplify (remove multi-edges)
    m_before = G.ecount()
    G.simplify()
    multi_edges_collapsed = m_before - G.ecount()
    
    # Take largest connected component
    components = G.components()
    if len(components) > 1:
        largest_cc_idx = np.argmax(components.sizes())
        G = components.subgraph(largest_cc_idx)
    
    n_nodes = G.vcount()
    m_edges = G.ecount()
    
    # Apply max_edges limit if specified
    if max_edges is not None and m_edges > max_edges:
        if verbose:
            print(f"    Subsampling from {m_edges:,} to {max_edges:,} edges for testing...")
        np.random.seed(SEED_BASE)
        keep_indices = np.random.choice(m_edges, size=max_edges, replace=False)
        edges_arr = np.array(G.get_edgelist())
        kept_edges = edges_arr[keep_indices].tolist()
        G = ig.Graph(n=n_nodes, edges=kept_edges, directed=False)
        
        # Take LCC again
        components = G.components()
        if len(components) > 1:
            largest_cc_idx = np.argmax(components.sizes())
            G = components.subgraph(largest_cc_idx)
        
        n_nodes = G.vcount()
        m_edges = G.ecount()
    
    info = {
        'name': name,
        'n_nodes': n_nodes,
        'm_edges': m_edges,
        'directed': False,
        'self_loops_removed': self_loops_removed,
        'multi_edges_collapsed': multi_edges_collapsed,
        'is_connected': True,
    }
    
    if verbose:
        print(f"    Nodes: {n_nodes:,}")
        print(f"    Edges: {m_edges:,}")
        print(f"    Self-loops removed: {self_loops_removed:,}")
        print(f"    Multi-edges collapsed: {multi_edges_collapsed:,}")
        print(f"    Status: undirected, simple, connected")
    
    return G, info

