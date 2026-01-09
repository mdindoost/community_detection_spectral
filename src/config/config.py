"""
Configuration constants for the community detection experiment.
"""
import os
from pathlib import Path

# Project root directory  
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Experiment parameters
EPSILON_VALUES = [1.0, 1.25]
RETENTION_VALUES = [0.90, 0.75, 0.50]
DSPAR_METHODS = ["paper", "probabilistic_no_replace", "deterministic"]
CPM_RESOLUTIONS = [0.1, 0.01, 0.001]

# Directory configurations
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
ORIGINAL_LEIDEN_DIR = RESULTS_DIR / "original_leiden"

# Julia configuration
JULIA_VERSION = "1.10.2"
JULIA_PROJECT = PROJECT_ROOT / "JuliaProject"
# Use JULIA_DEPOT_PATH from environment if set, otherwise use local directory
JULIA_DEPOT = Path(os.environ.get('JULIA_DEPOT_PATH', str(PROJECT_ROOT / "julia_depot")))
SPARSIFY_SCRIPT = PROJECT_ROOT / "sparsify_graph.jl"
TEMP_DIR = PROJECT_ROOT / "temp_lfr"
