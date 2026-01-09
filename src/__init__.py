"""
Community Detection with Spectral Sparsification

A modular package for comparing DSpar and Spectral sparsification methods
for community detection on large graphs.
"""

from .config import (
    EPSILON_VALUES,
    RETENTION_VALUES,
    DSPAR_METHODS,
    CPM_RESOLUTIONS,
    RESULTS_DIR,
    DATASETS_DIR,
    ORIGINAL_LEIDEN_DIR,
)
from .data import load_edges, load_graph, load_dataset
from .clustering import run_leiden, run_leiden_timed
from .clustering.leiden_cache import run_leiden_cached, save_leiden_result, load_leiden_result, is_leiden_cached
from .sparsifiers import dspar_sparsify, spectral_sparsify
from .sparsifiers.dspar import dspar_sparsify_timed
from .sparsifiers.spectral import spectral_sparsify_timed
from .eval.metrics import compute_nmi_ari, calculate_edge_preservation_ratio, count_intra_inter_edges
from .io import ResultsManager

__all__ = [
    # Config
    'EPSILON_VALUES',
    'RETENTION_VALUES', 
    'DSPAR_METHODS',
    'CPM_RESOLUTIONS',
    'RESULTS_DIR',
    'DATASETS_DIR',
    'ORIGINAL_LEIDEN_DIR',
    # Data loading
    'load_edges',
    'load_graph',
    'load_dataset',
    # Leiden
    'run_leiden',
    'run_leiden_timed',
    'run_leiden_cached',
    'save_leiden_result',
    'load_leiden_result',
    'is_leiden_cached',
    # Sparsification
    'dspar_sparsify',
    'dspar_sparsify_timed',
    'spectral_sparsify',
    'spectral_sparsify_timed',
    # Metrics
    'compute_nmi_ari',
    'calculate_edge_preservation_ratio',
    'count_intra_inter_edges',
    # Results
    'ResultsManager',
]
