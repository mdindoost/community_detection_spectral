"""
Community Detection with Spectral Sparsification

A modular package for comparing DSpar and baseline sparsification methods
for community detection on large graphs.
"""

from .config import (
    ALPHAS,
    METHODS,
    N_REPLICATES,
    SEED_BASE,
    SPECTRAL_EPSILON_MAP,
    SPECTRAL_TIMEOUT,
    DEFAULT_DATASETS,
    DATASETS_DIR,
    RESULTS_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    ORIGINAL_LEIDEN_DIR,
)
from .data import load_edges, load_graph, load_dataset
from .clustering import run_leiden, run_leiden_timed
from .clustering.leiden_cache import run_leiden_cached, save_leiden_result, load_leiden_result, is_leiden_cached
from .sparsifiers import dspar_sparsify
from .sparsifiers.dspar import dspar_sparsify_timed
from .sparsifiers.baseline import uniform_random_sparsify, degree_sampling_sparsify
from .eval.metrics import compute_nmi, compute_modularity_fixed
from .io import ResultsManager

__all__ = [
    # Config
    'ALPHAS',
    'METHODS',
    'N_REPLICATES',
    'SEED_BASE',
    'SPECTRAL_EPSILON_MAP',
    'SPECTRAL_TIMEOUT',
    'DEFAULT_DATASETS',
    'RESULTS_DIR',
    'DATASETS_DIR',
    'OUTPUT_DIR',
    'FIGURES_DIR',
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
    'uniform_random_sparsify',
    'degree_sampling_sparsify',
    # Metrics
    'compute_nmi',
    'compute_modularity_fixed',
    # Results
    'ResultsManager',
]
