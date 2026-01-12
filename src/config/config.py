"""
Configuration constants for scalability experiments.
Matches PAPER_EXPERIMENTS/exp3_scalability.py configuration.
"""
import os
from pathlib import Path

# Project root directory  
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# =============================================================================
# EXPERIMENT CONFIGURATION (matching exp3_scalability.py)
# =============================================================================

# Retention levels to test
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Number of replicates per configuration
N_REPLICATES = 3

# Sparsification methods
METHODS = ['dspar', 'uniform_random', 'degree_sampling']

# Seed base for reproducibility
SEED_BASE = 42

# Spectral epsilon values corresponding to approximate retention levels
SPECTRAL_EPSILON_MAP = {
    0.2: 3.0,   # Very aggressive sparsification
    0.4: 1.5,   # Aggressive
    0.6: 0.8,   # Moderate
    0.8: 0.3,   # Conservative
    1.0: 0.0,   # No sparsification (will skip)
}

# Maximum time (seconds) to wait for spectral sparsification
SPECTRAL_TIMEOUT = 300  # 5 minutes

# Default datasets (ordered by approximate edge count)
DEFAULT_DATASETS = [
    'com-DBLP',          # ~317K nodes, ~1M edges
    'com-Amazon',        # ~335K nodes, ~926K edges
    'com-Youtube',       # ~1.1M nodes, ~3M edges
    'wiki-Talk',         # ~2.4M nodes, ~5M edges
    'cit-Patents',       # ~3.8M nodes, ~17M edges
    'wiki-topcats',      # ~1.8M nodes, ~28M edges
    'com-LiveJournal',   # ~4M nodes, ~35M edges
    'com-Orkut',         # ~3M nodes, ~117M edges
]

# =============================================================================
# DIRECTORY CONFIGURATIONS
# =============================================================================

DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "exp3_scalability"
FIGURES_DIR = OUTPUT_DIR / "figures"
TEMP_DIR = PROJECT_ROOT / "temp"
ORIGINAL_LEIDEN_DIR = RESULTS_DIR / "original_leiden"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
ORIGINAL_LEIDEN_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# JULIA CONFIGURATION (for spectral sparsification)
# =============================================================================

JULIA_VERSION = "1.10.2"
JULIA_PROJECT = PROJECT_ROOT / "JuliaProject"
JULIA_DEPOT = Path(os.environ.get('JULIA_DEPOT_PATH', str(PROJECT_ROOT / "julia_depot")))
SPARSIFY_SCRIPT = PROJECT_ROOT / "sparsify_graph.jl"

