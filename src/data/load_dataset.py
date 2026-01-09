"""
Dataset loading utilities using polars for fast I/O.
"""
import sys
from pathlib import Path

# Support both package and script imports
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.datasets import DATASET_PATHS, DATASET_URLS
    from src.data.snap_downloader import ensure_file_exists
else:
    from .datasets import DATASET_PATHS, DATASET_URLS
    from .snap_downloader import ensure_file_exists

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

    df = pl.read_csv(
        edge_file,
        comment_prefix="#",
        separator="\t",
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

