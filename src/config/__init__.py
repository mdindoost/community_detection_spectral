"""Configuration module."""
from .env import (
    PROJECT_ROOT, 
    print_config,
    JULIA_VERSION,
    JULIA_DEPOT,
    JULIA_PROJECT,
    SPARSIFY_SCRIPT,
    JULIA_NUM_THREADS,
    JULIA_OPTIMIZE,
    JULIA_CHECK_BOUNDS,
    JULIA_MATH_MODE,
    JULIA_STARTUP_FILE,
    JULIA_HISTORY_FILE,
    JULIA_COMPILE,
)
from .config import *
