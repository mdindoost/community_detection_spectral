"""
Dataset path and URL configurations.

This file is kept separate from config.py since dataset lists can grow large.
"""
import sys
from pathlib import Path

# Support both package and script imports
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import DATASETS_DIR
else:
    from ..config import DATASETS_DIR

# Dictionary mapping dataset names to their edge file paths
DATASET_PATHS = {
    "cit-HepPh": DATASETS_DIR / "cit-HepPh" / "cit-HepPh.txt",
    "cit-HepTh": DATASETS_DIR / "cit-HepTh" / "cit-HepTh.txt",
    "citeseer": DATASETS_DIR / "citeseer" / "edges_original.txt",
    "soc-LiveJournal1": DATASETS_DIR / "soc-LiveJournal1" / "soc-LiveJournal1.txt",
    "ca-CondMat": DATASETS_DIR / "ca-CondMat" / "ca-CondMat.txt",
}

# Dictionary mapping dataset names to their download URLs (SNAP)
DATASET_URLS = {
    "cit-HepPh": "https://snap.stanford.edu/data/cit-HepPh.txt.gz",
    "cit-HepTh": "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
    "soc-LiveJournal1": "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
    "ca-CondMat": "https://snap.stanford.edu/data/ca-CondMat.txt.gz",
    # Citeseer is not a SNAP edge list, so leave blank or add if available
}
