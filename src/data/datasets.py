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
    # Smaller SNAP datasets
    'ca-AstroPh': (
        'https://snap.stanford.edu/data/ca-AstroPh.txt.gz',
        False, None
    ),
    'ca-HepPh': (
        'https://snap.stanford.edu/data/ca-HepPh.txt.gz',
        False, None
    ),
    'cit-HepPh': (
        'https://snap.stanford.edu/data/cit-HepPh.txt.gz',
        False, None
    ),
    'email-Enron': (
        'https://snap.stanford.edu/data/email-Enron.txt.gz',
        False, None
    ),
    'facebook-combined': (
        'https://snap.stanford.edu/data/facebook_combined.txt.gz',
        False, None
    ),
    'ca-GrQc': (
        'https://snap.stanford.edu/data/ca-GrQc.txt.gz',
        False, None
    ),
    'ca-CondMat': (
        'https://snap.stanford.edu/data/ca-CondMat.txt.gz',
        False, None
    ),
    'cit-HepTh': (
        'https://snap.stanford.edu/data/cit-HepTh.txt.gz',
        False, None
    ),
    'wiki-Vote': (
        'https://snap.stanford.edu/data/wiki-Vote.txt.gz',
        False, None
    ),
    'soc-Epinions1': (
        'https://snap.stanford.edu/data/soc-Epinions1.txt.gz',
        False, None
    ),
    'ca-HepTh': (
        'https://snap.stanford.edu/data/ca-HepTh.txt.gz',
        False, None
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
    'ca-AstroPh': DATASETS_DIR / 'ca-AstroPh' / 'ca-AstroPh.txt',
    'ca-HepPh': DATASETS_DIR / 'ca-HepPh' / 'ca-HepPh.txt',
    'cit-HepPh': DATASETS_DIR / 'cit-HepPh' / 'cit-HepPh.txt',
    'email-Enron': DATASETS_DIR / 'email-Enron' / 'email-Enron.txt',
    'facebook-combined': DATASETS_DIR / 'facebook-combined' / 'facebook-combined.txt',
    'ca-GrQc': DATASETS_DIR / 'ca-GrQc' / 'ca-GrQc.txt',
    'ca-CondMat': DATASETS_DIR / 'ca-CondMat' / 'ca-CondMat.txt',
    'cit-HepTh': DATASETS_DIR / 'cit-HepTh' / 'cit-HepTh.txt',
    'wiki-Vote': DATASETS_DIR / 'wiki-Vote' / 'wiki-Vote.txt',
    'soc-Epinions1': DATASETS_DIR / 'soc-Epinions1' / 'soc-Epinions1.txt',
    'ca-HepTh': DATASETS_DIR / 'ca-HepTh' / 'ca-HepTh.txt',
}


# Dataset URLs (for backward compatibility)
DATASET_URLS = {name: info[0] for name, info in SNAP_DATASETS.items()}


# Ground truth URLs
GROUND_TRUTH_URLS = {
    name: info[2] for name, info in SNAP_DATASETS.items() if info[1] and info[2]
}


