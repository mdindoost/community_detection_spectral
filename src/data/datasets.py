"""
Dataset configurations for scalability experiments.
Matches PAPER_EXPERIMENTS/exp3_scalability.py DEFAULT_DATASETS.
"""
import sys
from pathlib import Path

# Support both package and script imports
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import DATASETS_DIR
else:
    from ..config import DATASETS_DIR


# Default datasets for scalability experiments (ordered by edge count)
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


# SNAP dataset URLs and ground truth info
# Format: name -> (url, has_ground_truth, ground_truth_url)
SNAP_DATASETS = {
    'com-DBLP': (
        'https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz',
        True,
        'https://snap.stanford.edu/data/bigdata/communities/com-dblp.top5000.cmty.txt.gz'
    ),
    'com-Amazon': (
        'https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz',
        True,
        'https://snap.stanford.edu/data/bigdata/communities/com-amazon.top5000.cmty.txt.gz'
    ),
    'com-Youtube': (
        'https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz',
        True,
        'https://snap.stanford.edu/data/bigdata/communities/com-youtube.top5000.cmty.txt.gz'
    ),
    'wiki-Talk': (
        'https://snap.stanford.edu/data/wiki-Talk.txt.gz',
        False,
        None
    ),
    'cit-Patents': (
        'https://snap.stanford.edu/data/cit-Patents.txt.gz',
        False,
        None
    ),
    'wiki-topcats': (
        'https://snap.stanford.edu/data/wiki-topcats.txt.gz',
        True,
        'https://snap.stanford.edu/data/wiki-topcats-categories.txt.gz'
    ),
    'com-LiveJournal': (
        'https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz',
        True,
        'https://snap.stanford.edu/data/bigdata/communities/com-lj.top5000.cmty.txt.gz'
    ),
    'com-Orkut': (
        'https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz',
        True,
        'https://snap.stanford.edu/data/bigdata/communities/com-orkut.top5000.cmty.txt.gz'
    ),
}


# Dataset file paths
DATASET_PATHS = {
    'com-DBLP': DATASETS_DIR / 'com-DBLP' / 'com-dblp.ungraph.txt',
    'com-Amazon': DATASETS_DIR / 'com-Amazon' / 'com-amazon.ungraph.txt',
    'com-Youtube': DATASETS_DIR / 'com-Youtube' / 'com-youtube.ungraph.txt',
    'wiki-Talk': DATASETS_DIR / 'wiki-Talk' / 'wiki-Talk.txt',
    'cit-Patents': DATASETS_DIR / 'cit-Patents' / 'cit-Patents.txt',
    'wiki-topcats': DATASETS_DIR / 'wiki-topcats' / 'wiki-topcats.txt',
    'com-LiveJournal': DATASETS_DIR / 'com-LiveJournal' / 'com-lj.ungraph.txt',
    'com-Orkut': DATASETS_DIR / 'com-Orkut' / 'com-orkut.ungraph.txt',
}


# Dataset URLs (for backward compatibility)
DATASET_URLS = {name: info[0] for name, info in SNAP_DATASETS.items()}


# Ground truth URLs
GROUND_TRUTH_URLS = {
    name: info[2] for name, info in SNAP_DATASETS.items() if info[1] and info[2]
}


