"""
Metrics for comparing community detection results.
Uses numpy for vectorized computation - no Python loops.
"""
import numpy as np
import igraph as ig
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from typing import Dict, List, Tuple


def compute_nmi_ari(
    membership_original: List[int],
    membership_sparse: List[int]
) -> Tuple[float, float]:
    """
    Compute NMI and ARI between two community partitions.
    
    Parameters
    ----------
    membership_original : list
        Community membership from original graph
    membership_sparse : list
        Community membership from sparsified graph
        
    Returns
    -------
    nmi : float
        Normalized Mutual Information
    ari : float
        Adjusted Rand Index
    """
    nmi = normalized_mutual_info_score(membership_original, membership_sparse)
    ari = adjusted_rand_score(membership_original, membership_sparse)
    return nmi, ari


def calculate_edge_preservation_ratio(
    G: ig.Graph,
    G_sparse: ig.Graph,
    membership: List[int]
) -> Dict[str, float]:
    """
    Calculate edge preservation ratio using vectorized operations.
    
    Ratio = inter_rate / intra_rate
    Ratio < 1 means inter-community edges removed faster (desired)
    
    Parameters
    ----------
    G : ig.Graph
        Original graph
    G_sparse : ig.Graph
        Sparsified graph
    membership : list
        Community membership for each node
        
    Returns
    -------
    stats : dict
        Dictionary with preservation statistics
    """
    mem = np.array(membership)
    
    # Get original edges as numpy array (vectorized)
    orig_edges = np.array(G.get_edgelist())
    orig_sources = orig_edges[:, 0]
    orig_targets = orig_edges[:, 1]
    
    # Classify original edges (vectorized)
    is_intra = mem[orig_sources] == mem[orig_targets]
    total_intra = is_intra.sum()
    total_inter = len(orig_edges) - total_intra
    
    # Get sparse edges as numpy array and create normalized version (fully vectorized)
    sparse_edges = np.array(G_sparse.get_edgelist())
    if len(sparse_edges) > 0:
        sparse_normalized = np.column_stack([
            np.minimum(sparse_edges[:, 0], sparse_edges[:, 1]),
            np.maximum(sparse_edges[:, 0], sparse_edges[:, 1])
        ])
        # Convert to set of tuples for fast lookup
        sparse_edge_set = set(map(tuple, sparse_normalized))
    else:
        sparse_edge_set = set()
    
    # Create normalized original edges for lookup (vectorized)
    orig_normalized = np.column_stack([
        np.minimum(orig_sources, orig_targets),
        np.maximum(orig_sources, orig_targets)
    ])
    
    # Check which edges are preserved using vectorized approach
    # Convert sparse_edge_set to numpy for vectorized comparison
    if sparse_edge_set:
        sparse_arr = np.array(list(sparse_edge_set))
        # Use broadcasting to check membership (create unique edge keys)
        orig_keys = orig_normalized[:, 0].astype(np.int64) * (orig_normalized[:, 1].max() + 1) + orig_normalized[:, 1]
        sparse_keys = sparse_arr[:, 0].astype(np.int64) * (orig_normalized[:, 1].max() + 1) + sparse_arr[:, 1]
        preserved_mask = np.isin(orig_keys, sparse_keys)
    else:
        preserved_mask = np.zeros(len(orig_normalized), dtype=bool)
    
    # Count preserved intra/inter edges (vectorized)
    preserved_intra = (preserved_mask & is_intra).sum()
    preserved_inter = (preserved_mask & ~is_intra).sum()
    
    # Calculate rates and ratio
    intra_rate = preserved_intra / total_intra if total_intra > 0 else 1.0
    inter_rate = preserved_inter / total_inter if total_inter > 0 else 1.0
    ratio = inter_rate / intra_rate if intra_rate > 0 else float('inf')
    
    return {
        'total_intra': int(total_intra),
        'total_inter': int(total_inter),
        'preserved_intra': int(preserved_intra),
        'preserved_inter': int(preserved_inter),
        'intra_rate': intra_rate,
        'inter_rate': inter_rate,
        'ratio': ratio
    }


def count_intra_inter_edges(
    edges: np.ndarray,
    membership: List[int]
) -> Tuple[int, int]:
    """
    Count intra and inter community edges using vectorized operations.
    
    Parameters
    ----------
    edges : np.ndarray
        Edge array of shape (n_edges, 2)
    membership : list
        Community membership for each node
        
    Returns
    -------
    total_intra : int
        Number of intra-community edges
    total_inter : int
        Number of inter-community edges
    """
    mem = np.array(membership)
    sources = edges[:, 0]
    targets = edges[:, 1]
    total_intra = int((mem[sources] == mem[targets]).sum())
    total_inter = len(edges) - total_intra
    return total_intra, total_inter
