"""
Spectral sparsification using Julia's Laplacians.jl library.
"""
import os
import time
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import igraph as ig

# Support both package and script imports
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import JULIA_PROJECT, JULIA_DEPOT, SPARSIFY_SCRIPT, TEMP_DIR, PROJECT_ROOT, JULIA_VERSION
else:
    from ..config import JULIA_PROJECT, JULIA_DEPOT, SPARSIFY_SCRIPT, TEMP_DIR, PROJECT_ROOT, JULIA_VERSION


def get_julia_path() -> str:
    """Find Julia executable."""
    # Check local installation first
    local_julia = PROJECT_ROOT / f"julia-{JULIA_VERSION}" / "bin" / "julia"
    if local_julia.exists():
        return str(local_julia)
    
    # Check system PATH
    julia_in_path = shutil.which("julia")
    if julia_in_path:
        return julia_in_path
    
    raise RuntimeError(
        f"Julia not found. Run ./setup_julia.sh to install Julia {JULIA_VERSION}"
    )


def spectral_sparsify(G: ig.Graph, epsilon: float) -> ig.Graph:
    """
    Run Julia spectral sparsification on igraph Graph.
    
    Parameters
    ----------
    G : ig.Graph
        Input graph
    epsilon : float
        Sparsification parameter (smaller = more edges retained)
        
    Returns
    -------
    G_sparse : ig.Graph
        Sparsified graph
    """
    n_nodes = G.vcount()
    
    # Get edges as numpy array (vectorized)
    edge_list = G.get_edgelist()
    edges_arr = np.array(edge_list, dtype=np.int64)
    
    # Create bidirectional edge list for Julia
    edges_bi = np.vstack([edges_arr, edges_arr[:, ::-1]])
    
    sparsified_edges = _run_julia_sparsify(edges_bi, n_nodes, epsilon)
    
    # Build sparse graph - filter to u < v to avoid duplicates (vectorized)
    mask = sparsified_edges[:, 0] < sparsified_edges[:, 1]
    edges_filtered = sparsified_edges[mask]
    
    G_sparse = ig.Graph(n=n_nodes, edges=edges_filtered.tolist(), directed=False)
    
    return G_sparse


def spectral_sparsify_timed(G: ig.Graph, epsilon: float) -> Tuple[ig.Graph, float]:
    """
    Run Julia spectral sparsification with timing.
    
    Parameters
    ----------
    G : ig.Graph
        Input graph
    epsilon : float
        Sparsification parameter
        
    Returns
    -------
    G_sparse : ig.Graph
        Sparsified graph
    elapsed : float
        Time in seconds
    """
    start = time.perf_counter()
    G_sparse = spectral_sparsify(G, epsilon)
    elapsed = time.perf_counter() - start
    return G_sparse, elapsed


def _run_julia_sparsify(edges: np.ndarray, n_nodes: int, epsilon: float) -> np.ndarray:
    """
    Run Julia spectral sparsification.
    
    Parameters
    ----------
    edges : np.ndarray
        Bidirectional edge array of shape (n_edges*2, 2)
    n_nodes : int
        Number of nodes
    epsilon : float
        Sparsification parameter
        
    Returns
    -------
    sparsified_edges : np.ndarray
        Array of sparsified edges
    """
    # Create temp directory if needed
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file for edges
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', dir=TEMP_DIR, delete=False) as f:
        edges_file = Path(f.name)
        # Write edges efficiently using numpy
        np.savetxt(f, edges, fmt='%d %d')
    
    try:
        julia_path = get_julia_path()
        
        env = os.environ.copy()
        env['JULIA_DEPOT_PATH'] = str(JULIA_DEPOT)
        
        cmd = [
            julia_path,
            f"--project={JULIA_PROJECT}",
            str(SPARSIFY_SCRIPT),
            str(edges_file),
            str(n_nodes),
            str(epsilon)
        ]
        
        result = subprocess.run(
            cmd,
            cwd=TEMP_DIR,
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Spectral sparsification failed: {result.stderr}")
        
        # Load output
        output_file = TEMP_DIR / f"edges_sparsified_eps{epsilon}.txt"
        sparsified_edges = _load_edges_from_file(output_file)
        
        # Clean up output file
        output_file.unlink(missing_ok=True)
        
        return sparsified_edges
        
    finally:
        # Clean up input file
        edges_file.unlink(missing_ok=True)


def _load_edges_from_file(filepath: Path) -> np.ndarray:
    """Load edges from file as numpy array using polars."""
    import polars as pl
    
    df = pl.read_csv(
        filepath,
        comment_prefix="#",
        separator=" ",
        has_header=False,
        new_columns=["source", "target"],
        schema_overrides={"source": pl.Int64, "target": pl.Int64},
    )
    
    return df.select(["source", "target"]).to_numpy()

