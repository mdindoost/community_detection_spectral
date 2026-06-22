"""
Environment configuration loader.

Loads settings from .env file for portable configuration across devices.
All paths can be absolute or relative to PROJECT_ROOT.
"""
import os
from pathlib import Path
from typing import Optional

# Project root directory (fixed, everything else is relative to this)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def load_env_file(env_path: Optional[Path] = None) -> dict:
    """
    Load environment variables from .env file.
    
    Parameters
    ----------
    env_path : Path, optional
        Path to .env file. Defaults to PROJECT_ROOT/.env
        
    Returns
    -------
    dict
        Dictionary of loaded environment variables
    """
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"
    
    loaded = {}
    
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse key=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Don't override existing environment variables
                    if key not in os.environ:
                        os.environ[key] = value
                    loaded[key] = value
    
    return loaded


def resolve_path(path_str: str, default: str = None) -> Path:
    """
    Resolve a path string to an absolute Path.
    
    Handles:
    - Absolute paths: returned as-is
    - Relative paths (starting with ./): resolved relative to PROJECT_ROOT
    - Other relative paths: resolved relative to PROJECT_ROOT
    
    Parameters
    ----------
    path_str : str
        Path string from environment variable
    default : str, optional
        Default value if path_str is empty
        
    Returns
    -------
    Path
        Resolved absolute path
    """
    if not path_str and default:
        path_str = default
    
    if not path_str:
        return None
    
    path = Path(path_str)
    
    # If absolute, return as-is
    if path.is_absolute():
        return path
    
    # Otherwise resolve relative to PROJECT_ROOT
    return (PROJECT_ROOT / path).resolve()


def get_env(key: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    return os.environ.get(key, default)


def get_env_path(key: str, default: str = None) -> Path:
    """Get environment variable as resolved Path."""
    value = os.environ.get(key, default)
    return resolve_path(value, default)


# =============================================================================
# Load .env on module import
# =============================================================================
_loaded_env = load_env_file()


# =============================================================================
# Julia Configuration (loaded from .env)
# =============================================================================

JULIA_VERSION = get_env('JULIA_VERSION', '1.10.2')
JULIA_DEPOT = get_env_path('JULIA_DEPOT_PATH', './julia_depot')
JULIA_PROJECT = get_env_path('JULIA_PROJECT_PATH', './JuliaProject')
SPARSIFY_SCRIPT = get_env_path('JULIA_SPARSIFY_SCRIPT', './sparsify_graph.jl')

# Julia optimization flags
JULIA_NUM_THREADS = get_env('JULIA_NUM_THREADS', 'auto')
JULIA_OPTIMIZE = get_env('JULIA_OPTIMIZE', '3')
JULIA_CHECK_BOUNDS = get_env('JULIA_CHECK_BOUNDS', 'no')
JULIA_MATH_MODE = get_env('JULIA_MATH_MODE', 'fast')
JULIA_STARTUP_FILE = get_env('JULIA_STARTUP_FILE', 'no')
JULIA_HISTORY_FILE = get_env('JULIA_HISTORY_FILE', 'no')
JULIA_COMPILE = get_env('JULIA_COMPILE', 'min')


# =============================================================================
# Directory Configuration (with defaults, overridable via .env)
# =============================================================================

DATASETS_DIR = get_env_path('DATASETS_DIR', './datasets')
RESULTS_DIR = get_env_path('RESULTS_DIR', './results')
TEMP_DIR = get_env_path('TEMP_DIR', './temp')
OUTPUT_DIR = RESULTS_DIR / "exp3_scalability"
FIGURES_DIR = OUTPUT_DIR / "figures"
ORIGINAL_LEIDEN_DIR = RESULTS_DIR / "original_leiden"


def ensure_directories():
    """Create required directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    ORIGINAL_LEIDEN_DIR.mkdir(parents=True, exist_ok=True)


# Create directories on import
ensure_directories()


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("ENVIRONMENT CONFIGURATION")
    print("=" * 60)
    print(f"PROJECT_ROOT:     {PROJECT_ROOT}")
    print()
    print("Julia Configuration:")
    print(f"  JULIA_VERSION:  {JULIA_VERSION}")
    print(f"  JULIA_DEPOT:    {JULIA_DEPOT}")
    print(f"  JULIA_PROJECT:  {JULIA_PROJECT}")
    print(f"  SPARSIFY_SCRIPT:{SPARSIFY_SCRIPT}")
    print()
    print("Julia Optimization:")
    print(f"  JULIA_NUM_THREADS:  {JULIA_NUM_THREADS}")
    print(f"  JULIA_OPTIMIZE:     {JULIA_OPTIMIZE}")
    print(f"  JULIA_CHECK_BOUNDS: {JULIA_CHECK_BOUNDS}")
    print(f"  JULIA_MATH_MODE:    {JULIA_MATH_MODE}")
    print()
    print("Directories:")
    print(f"  DATASETS_DIR:   {DATASETS_DIR}")
    print(f"  RESULTS_DIR:    {RESULTS_DIR}")
    print(f"  TEMP_DIR:       {TEMP_DIR}")
    print(f"  OUTPUT_DIR:     {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
