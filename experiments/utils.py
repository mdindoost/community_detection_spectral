"""
Utility functions for community detection experiments.

Handles:
- Dataset downloading and loading
- Graph construction and conversion
- Spectral and random sparsification
- File I/O for edges and graphs
"""

import os
import subprocess
import time
import urllib.request
import gzip
import shutil
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
import networkx as nx
import igraph as ig


# =============================================================================
# Configuration
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Dataset configurations
SNAP_DATASETS = {
    # name: (url, has_ground_truth, ground_truth_url)
    # ----- Small datasets -----
    'email-Eu-core': (
        'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
        True,
        'https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz'
    ),
    'wiki-Vote': (
        'https://snap.stanford.edu/data/wiki-Vote.txt.gz',
        False,
        None
    ),
    'ca-HepPh': (
        'https://snap.stanford.edu/data/ca-HepPh.txt.gz',
        False,
        None
    ),
    'soc-Epinions1': (
        'https://snap.stanford.edu/data/soc-Epinions1.txt.gz',
        False,
        None
    ),
    # ----- Medium datasets -----
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
    # ----- Large datasets -----
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
    'com-Friendster': (
        'https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz',
        True,
        'https://snap.stanford.edu/data/bigdata/communities/com-friendster.top5000.cmty.txt.gz'
    ),
    # ----- Citation networks -----
    'cora': (
        'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
        True,
        None  # Labels included in tgz
    ),
    'citeseer': (
        'https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz',
        True,
        None  # Labels included in tgz
    ),
    # ----- PPI networks -----
    'yeast-ppi': (
        'http://nrvis.com/download/data/bio/bio-yeast.zip',
        False,
        None
    ),
    'human-ppi': (
        'http://nrvis.com/download/data/bio/bio-human-gene1.zip',
        False,
        None
    ),
}

# Datasets requiring special parsing
SPECIAL_DATASETS = {'cora', 'citeseer', 'yeast-ppi', 'human-ppi'}

# Julia paths
JULIA_VERSION = "1.10.2"
JULIA_PROJECT = PROJECT_ROOT / "JuliaProject"
JULIA_DEPOT = PROJECT_ROOT / "julia_depot"
SPARSIFY_SCRIPT = PROJECT_ROOT / "sparsify_graph.jl"

# Directories
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_dataset_dir(dataset_name):
    """Get the directory for a specific dataset."""
    return DATASETS_DIR / dataset_name


def get_results_dir(dataset_name):
    """Get the results directory for a specific dataset."""
    return RESULTS_DIR / dataset_name


# =============================================================================
# Julia Setup
# =============================================================================

def get_julia_path():
    """Find Julia executable."""
    # Check local installation first
    local_julia = PROJECT_ROOT / f"julia-{JULIA_VERSION}" / "bin" / "julia"
    if local_julia.exists():
        return str(local_julia)

    # Check system PATH
    julia_in_path = shutil.which("julia")
    if julia_in_path:
        return julia_in_path

    raise RuntimeError(
        f"Julia not found. Run ./setup_julia.sh to install Julia {JULIA_VERSION}"
    )


# =============================================================================
# Dataset Downloading and Loading
# =============================================================================

def download_file(url, dest_path):
    """Download a file from URL."""
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, dest_path)


def _load_special_dataset(name):
    """Load datasets with special formats (Cora, Citeseer, PPI networks)."""
    import tarfile
    import zipfile

    dataset_dir = get_dataset_dir(name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    url, has_gt, _ = SNAP_DATASETS[name]

    if name in ('cora', 'citeseer'):
        return _load_citation_dataset(name, url, dataset_dir)
    elif name in ('yeast-ppi', 'human-ppi'):
        return _load_ppi_dataset(name, url, dataset_dir)
    else:
        raise ValueError(f"Unknown special dataset: {name}")


def _load_citation_dataset(name, url, dataset_dir):
    """Load Cora or Citeseer citation network."""
    import tarfile

    tgz_path = dataset_dir / f"{name}.tgz"
    extract_dir = dataset_dir / name

    # Download and extract
    if not extract_dir.exists():
        if not tgz_path.exists():
            download_file(url, tgz_path)
        print(f"  Extracting {tgz_path}...")
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(dataset_dir)

    # Find the cites file (edges) and content file (labels)
    cites_file = extract_dir / f"{name}.cites"
    content_file = extract_dir / f"{name}.content"

    if not cites_file.exists():
        raise FileNotFoundError(f"Could not find {cites_file}")

    # Parse content file to get node IDs and labels
    print(f"  Parsing {content_file}...")
    node_to_label = {}
    node_ids = []
    with open(content_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                node_id = parts[0]
                label = parts[-1]  # Last column is the label
                node_ids.append(node_id)
                node_to_label[node_id] = label

    # Create node mapping
    node_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
    n_nodes = len(node_ids)

    # Create label to int mapping
    unique_labels = sorted(set(node_to_label.values()))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Ground truth
    ground_truth = {node_map[nid]: label_map[label] for nid, label in node_to_label.items() if nid in node_map}

    # Parse edges from cites file
    print(f"  Parsing edges from {cites_file}...")
    edges = []
    for line in open(cites_file, 'r'):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            src, dst = parts[0], parts[1]
            if src in node_map and dst in node_map:
                s, d = node_map[src], node_map[dst]
                if s != d:
                    edges.append((s, d))
                    edges.append((d, s))

    # Remove duplicates
    edge_set = set((min(s, d), max(s, d)) for s, d in edges)
    edges = []
    for s, d in edge_set:
        edges.append((s, d))
        edges.append((d, s))

    print(f"  Loaded: {n_nodes} nodes, {len(edges)//2} undirected edges, {len(unique_labels)} classes")
    return edges, n_nodes, ground_truth


def _load_ppi_dataset(name, url, dataset_dir):
    """Load PPI network from NetworkRepository."""
    import zipfile

    zip_path = dataset_dir / f"{name}.zip"

    # Download
    if not zip_path.exists():
        download_file(url, zip_path)

    # Extract
    print(f"  Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dataset_dir)

    # Find the edges file (usually .edges or .mtx)
    edges_file = None
    for f in dataset_dir.iterdir():
        if f.suffix == '.edges' or (f.suffix == '.mtx' and 'bio' in f.name):
            edges_file = f
            break

    if edges_file is None:
        # Try to find any file with edges
        for f in dataset_dir.iterdir():
            if f.is_file() and f.suffix not in ('.zip',):
                edges_file = f
                break

    if edges_file is None:
        raise FileNotFoundError(f"Could not find edges file in {dataset_dir}")

    print(f"  Parsing edges from {edges_file}...")
    edges = []
    node_set = set()

    with open(edges_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    src, dst = int(parts[0]), int(parts[1])
                    edges.append((src, dst))
                    node_set.add(src)
                    node_set.add(dst)
                except ValueError:
                    continue

    # Remap to 0-indexed
    node_list = sorted(node_set)
    node_map = {old: new for new, old in enumerate(node_list)}
    n_nodes = len(node_list)

    edges = [(node_map[s], node_map[d]) for s, d in edges]

    # Make undirected
    edge_set = set()
    for s, d in edges:
        if s != d:
            edge_set.add((min(s, d), max(s, d)))

    edges = []
    for s, d in edge_set:
        edges.append((s, d))
        edges.append((d, s))

    print(f"  Loaded: {n_nodes} nodes, {len(edges)//2} undirected edges")
    return edges, n_nodes, None  # No ground truth for PPI


def load_snap_dataset(name):
    """
    Load a SNAP dataset, downloading if necessary.

    Args:
        name: Dataset name (must be in SNAP_DATASETS)

    Returns:
        edges: list of (src, dst) tuples (0-indexed, both directions)
        n_nodes: number of nodes
        ground_truth: dict mapping node -> community (or None)
    """
    if name not in SNAP_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(SNAP_DATASETS.keys())}")

    # Handle special datasets with different formats
    if name in SPECIAL_DATASETS:
        return _load_special_dataset(name)

    dataset_dir = get_dataset_dir(name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    url, has_gt, gt_url = SNAP_DATASETS[name]

    # Download edge file
    gz_path = dataset_dir / f"{name}.txt.gz"
    txt_path = dataset_dir / f"{name}.txt"

    if not txt_path.exists():
        if not gz_path.exists():
            download_file(url, gz_path)
        print(f"  Extracting {gz_path}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(txt_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # Parse edges
    print(f"  Parsing edges from {txt_path}...")
    edges = []
    node_set = set()

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    src, dst = int(parts[0]), int(parts[1])
                    edges.append((src, dst))
                    node_set.add(src)
                    node_set.add(dst)
                except ValueError:
                    continue

    # Remap nodes to 0-indexed contiguous range
    node_list = sorted(node_set)
    node_map = {old: new for new, old in enumerate(node_list)}
    n_nodes = len(node_list)

    edges = [(node_map[s], node_map[d]) for s, d in edges]

    # Make undirected (add reverse edges, remove duplicates)
    edge_set = set()
    for s, d in edges:
        if s != d:  # Remove self-loops
            edge_set.add((min(s, d), max(s, d)))

    # Create both directions
    edges = []
    for s, d in edge_set:
        edges.append((s, d))
        edges.append((d, s))

    # Load ground truth if available
    ground_truth = None
    if has_gt and gt_url:
        gt_gz_path = dataset_dir / f"{name}_labels.txt.gz"
        gt_txt_path = dataset_dir / f"{name}_labels.txt"

        if not gt_txt_path.exists():
            if not gt_gz_path.exists():
                try:
                    download_file(gt_url, gt_gz_path)
                except Exception as e:
                    print(f"  Warning: Could not download ground truth: {e}")
                    gt_gz_path = None

            if gt_gz_path and gt_gz_path.exists():
                print(f"  Extracting {gt_gz_path}...")
                with gzip.open(gt_gz_path, 'rb') as f_in:
                    with open(gt_txt_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        if gt_txt_path.exists():
            ground_truth = _parse_ground_truth(gt_txt_path, node_map, name)

    print(f"  Loaded: {n_nodes} nodes, {len(edges)//2} undirected edges")

    return edges, n_nodes, ground_truth


def _parse_ground_truth(path, node_map, dataset_name):
    """
    Parse ground truth labels file.
    Format varies by dataset:
    - email-Eu-core: node_id label (one per line)
    - com-*: community file (each line = list of nodes in community)
    """
    ground_truth = {}

    if dataset_name == 'email-Eu-core':
        # Format: node_id label
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        node, label = int(parts[0]), int(parts[1])
                        if node in node_map:
                            ground_truth[node_map[node]] = label
                    except ValueError:
                        continue
    else:
        # Format: each line is a community (list of node ids)
        with open(path, 'r') as f:
            for comm_id, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                nodes = line.split()
                for n in nodes:
                    try:
                        node = int(n)
                        if node in node_map:
                            # Node can be in multiple communities; use first
                            if node_map[node] not in ground_truth:
                                ground_truth[node_map[node]] = comm_id
                    except ValueError:
                        continue

    return ground_truth if ground_truth else None


# =============================================================================
# Graph Construction and Conversion
# =============================================================================

def edges_to_adjacency(edges, n_nodes):
    """Convert edge list to sparse adjacency matrix."""
    src = np.array([e[0] for e in edges])
    dst = np.array([e[1] for e in edges])
    data = np.ones(len(edges))

    A = csr_matrix((data, (src, dst)), shape=(n_nodes, n_nodes))
    return A


def adjacency_to_igraph(A):
    """Convert scipy sparse matrix to igraph Graph."""
    A_coo = coo_matrix(A)
    edges = [(i, j) for i, j in zip(A_coo.row, A_coo.col) if i < j]

    g = ig.Graph(n=A.shape[0], edges=edges, directed=False)
    return g


def count_connected_components(A):
    """Count connected components in graph."""
    n = A.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))

    A_coo = coo_matrix(A)
    for i, j in zip(A_coo.row, A_coo.col):
        if i < j:
            g.add_edge(i, j)

    return nx.number_connected_components(g)


# =============================================================================
# Edge File I/O
# =============================================================================

def save_edges_to_file(edges, path):
    """Save edges to text file."""
    with open(path, 'w') as f:
        for s, d in edges:
            f.write(f"{s} {d}\n")


def load_edges_from_file(path):
    """Load edges from text file."""
    edges = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))
    return edges


# =============================================================================
# Sparsification Methods
# =============================================================================

def spectral_sparsify(edges, n_nodes, epsilon, dataset_name):
    """
    Run spectral sparsification using Julia's Laplacians.jl.

    Args:
        edges: list of (src, dst) tuples
        n_nodes: number of nodes
        epsilon: sparsification parameter (smaller = more edges retained)
        dataset_name: name of dataset (for caching)

    Returns:
        tuple: (sparsified_edges, elapsed_time)
            - sparsified_edges: list of (src, dst) tuples
            - elapsed_time: time in seconds (None if cached)
    """
    dataset_dir = get_dataset_dir(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_file = dataset_dir / f"edges_spectral_eps{epsilon}.txt"
    if cache_file.exists():
        print(f"    Using cached spectral sparsification (eps={epsilon})")
        return load_edges_from_file(cache_file), None

    # Write original edges to file if not exists
    edges_file = dataset_dir / "edges_original.txt"
    if not edges_file.exists():
        save_edges_to_file(edges, edges_file)

    # Run Julia sparsification
    print(f"    Running spectral sparsification (eps={epsilon})...")

    julia_path = get_julia_path()

    env = os.environ.copy()
    env['JULIA_DEPOT_PATH'] = str(JULIA_DEPOT)

    cmd = [
        julia_path,
        f"--project={JULIA_PROJECT}",
        str(SPARSIFY_SCRIPT),
        str(edges_file),
        str(n_nodes),
        str(epsilon)
    ]

    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=dataset_dir,
        capture_output=True,
        text=True,
        env=env
    )
    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print(f"    Julia error: {result.stderr}")
        raise RuntimeError(f"Spectral sparsification failed: {result.stderr}")

    # Load output (Julia writes to current directory)
    output_file = dataset_dir / f"edges_sparsified_eps{epsilon}.txt"
    sparsified_edges = load_edges_from_file(output_file)

    # Rename to cache name
    output_file.rename(cache_file)

    return sparsified_edges, elapsed_time


def random_sparsify(edges, target_edge_count, seed):
    """
    Random edge sampling to match spectral sparsification edge count.

    Args:
        edges: list of (src, dst) tuples (both directions)
        target_edge_count: target number of directed edges
        seed: random seed

    Returns:
        tuple: (sparsified_edges, elapsed_time)
            - sparsified_edges: list of (src, dst) tuples (both directions)
            - elapsed_time: time in seconds
    """
    start_time = time.time()

    rng = np.random.RandomState(seed)

    # Get unique undirected edges
    unique_edges = set()
    for s, d in edges:
        unique_edges.add((min(s, d), max(s, d)))
    unique_edges = list(unique_edges)

    # Target is for undirected edges
    target_undirected = target_edge_count // 2

    if target_undirected >= len(unique_edges):
        # Keep all edges
        selected = unique_edges
    else:
        # Random sample
        indices = rng.choice(len(unique_edges), size=target_undirected, replace=False)
        selected = [unique_edges[i] for i in indices]

    # Create both directions
    result = []
    for s, d in selected:
        result.append((s, d))
        result.append((d, s))

    elapsed_time = time.time() - start_time
    return result, elapsed_time


# =============================================================================
# Edge Preservation Analysis
# =============================================================================

def analyze_edge_preservation(original_edges, sparsified_edges, community_labels):
    """
    Analyze which edges are preserved by sparsification, categorized by type.

    Tests hypothesis: Spectral sparsification preserves inter-community edges
    at higher rates than intra-community edges.

    Args:
        original_edges: list of (src, dst) tuples from original graph
        sparsified_edges: list of (src, dst) tuples from sparsified graph
        community_labels: list of community assignments (one per node)

    Returns:
        dict with preservation statistics
    """
    # Convert to sets of undirected edges for efficient lookup
    def to_undirected_set(edges):
        return set((min(s, d), max(s, d)) for s, d in edges)

    original_set = to_undirected_set(original_edges)
    sparsified_set = to_undirected_set(sparsified_edges)

    # Categorize original edges
    intra_original = 0  # Both endpoints in same community
    inter_original = 0  # Endpoints in different communities
    intra_preserved = 0
    inter_preserved = 0

    for s, d in original_set:
        same_community = community_labels[s] == community_labels[d]
        preserved = (s, d) in sparsified_set

        if same_community:
            intra_original += 1
            if preserved:
                intra_preserved += 1
        else:
            inter_original += 1
            if preserved:
                inter_preserved += 1

    # Compute preservation rates
    intra_rate = intra_preserved / intra_original if intra_original > 0 else 0
    inter_rate = inter_preserved / inter_original if inter_original > 0 else 0

    # Compute ratio (inter / intra) - if > 1, hypothesis is supported
    preservation_ratio = inter_rate / intra_rate if intra_rate > 0 else float('inf')

    return {
        'intra_original': intra_original,
        'inter_original': inter_original,
        'intra_preserved': intra_preserved,
        'inter_preserved': inter_preserved,
        'intra_preservation_rate': intra_rate,
        'inter_preservation_rate': inter_rate,
        'preservation_ratio': preservation_ratio,
        'hypothesis_supported': preservation_ratio > 1.0
    }


def analyze_ground_truth_edge_preservation(original_edges, sparsified_edges, ground_truth):
    """
    Analyze edge preservation using GROUND TRUTH community labels.

    This reveals if spectral sparsification is removing "true noise" vs "true bridges".

    Args:
        original_edges: list of (src, dst) tuples from original graph
        sparsified_edges: list of (src, dst) tuples from sparsified graph
        ground_truth: dict mapping node -> ground truth community

    Returns:
        dict with ground truth preservation statistics, or None if ground_truth is None
    """
    if ground_truth is None:
        return None

    # Convert to sets of undirected edges
    def to_undirected_set(edges):
        return set((min(s, d), max(s, d)) for s, d in edges)

    original_set = to_undirected_set(original_edges)
    sparsified_set = to_undirected_set(sparsified_edges)

    # Categorize using ground truth labels
    gt_intra_original = 0
    gt_inter_original = 0
    gt_intra_preserved = 0
    gt_inter_preserved = 0

    for s, d in original_set:
        # Skip edges where one or both nodes don't have ground truth
        if s not in ground_truth or d not in ground_truth:
            continue

        same_gt_community = ground_truth[s] == ground_truth[d]
        preserved = (s, d) in sparsified_set

        if same_gt_community:
            gt_intra_original += 1
            if preserved:
                gt_intra_preserved += 1
        else:
            gt_inter_original += 1
            if preserved:
                gt_inter_preserved += 1

    # Compute preservation rates
    gt_intra_rate = gt_intra_preserved / gt_intra_original if gt_intra_original > 0 else 0
    gt_inter_rate = gt_inter_preserved / gt_inter_original if gt_inter_original > 0 else 0
    gt_ratio = gt_inter_rate / gt_intra_rate if gt_intra_rate > 0 else float('inf')

    return {
        'gt_intra_original': gt_intra_original,
        'gt_inter_original': gt_inter_original,
        'gt_intra_preserved': gt_intra_preserved,
        'gt_inter_preserved': gt_inter_preserved,
        'gt_intra_preservation_rate': gt_intra_rate,
        'gt_inter_preservation_rate': gt_inter_rate,
        'gt_preservation_ratio': gt_ratio
    }


def analyze_misclassification(original_edges, sparsified_edges, leiden_labels, ground_truth):
    """
    Analyze if removed "inter-community" edges (per Leiden) were actually misclassified.

    For edges Leiden calls "inter-community" that were REMOVED:
    - What fraction are actually INTRA-community in ground truth?

    This reveals if the edges being removed were misclassified by Leiden.

    Args:
        original_edges: list of (src, dst) tuples from original graph
        sparsified_edges: list of (src, dst) tuples from sparsified graph
        leiden_labels: list of Leiden community assignments
        ground_truth: dict mapping node -> ground truth community

    Returns:
        dict with misclassification statistics, or None if ground_truth is None
    """
    if ground_truth is None:
        return None

    # Convert to sets of undirected edges
    def to_undirected_set(edges):
        return set((min(s, d), max(s, d)) for s, d in edges)

    original_set = to_undirected_set(original_edges)
    sparsified_set = to_undirected_set(sparsified_edges)
    removed_edges = original_set - sparsified_set

    # Analyze removed edges that Leiden classified as inter-community
    leiden_inter_removed = 0
    leiden_inter_removed_but_gt_intra = 0  # Misclassified by Leiden

    # Analyze preserved edges that Leiden classified as intra-community
    leiden_intra_preserved = 0
    leiden_intra_preserved_but_gt_inter = 0  # Misclassified by Leiden

    for s, d in removed_edges:
        if s not in ground_truth or d not in ground_truth:
            continue

        leiden_same = leiden_labels[s] == leiden_labels[d]
        gt_same = ground_truth[s] == ground_truth[d]

        if not leiden_same:  # Leiden says inter-community
            leiden_inter_removed += 1
            if gt_same:  # But ground truth says intra
                leiden_inter_removed_but_gt_intra += 1

    preserved_edges = original_set & sparsified_set
    for s, d in preserved_edges:
        if s not in ground_truth or d not in ground_truth:
            continue

        leiden_same = leiden_labels[s] == leiden_labels[d]
        gt_same = ground_truth[s] == ground_truth[d]

        if leiden_same:  # Leiden says intra-community
            leiden_intra_preserved += 1
            if not gt_same:  # But ground truth says inter
                leiden_intra_preserved_but_gt_inter += 1

    # Compute misclassification rates
    removed_misclass_rate = (leiden_inter_removed_but_gt_intra / leiden_inter_removed
                            if leiden_inter_removed > 0 else 0)
    preserved_misclass_rate = (leiden_intra_preserved_but_gt_inter / leiden_intra_preserved
                              if leiden_intra_preserved > 0 else 0)

    return {
        'leiden_inter_removed': leiden_inter_removed,
        'leiden_inter_removed_but_gt_intra': leiden_inter_removed_but_gt_intra,
        'removed_misclassification_rate': removed_misclass_rate,
        'leiden_intra_preserved': leiden_intra_preserved,
        'leiden_intra_preserved_but_gt_inter': leiden_intra_preserved_but_gt_inter,
        'preserved_misclassification_rate': preserved_misclass_rate
    }


def compute_ground_truth_modularity(graph, ground_truth):
    """
    Compute modularity of ground truth communities on a graph.

    Args:
        graph: igraph Graph
        ground_truth: dict mapping node -> ground truth community

    Returns:
        modularity score, or None if ground_truth is None
    """
    if ground_truth is None:
        return None

    # Convert ground truth dict to list of labels
    n_nodes = graph.vcount()
    labels = []
    for i in range(n_nodes):
        if i in ground_truth:
            labels.append(ground_truth[i])
        else:
            # Assign nodes without ground truth to a special community
            labels.append(-1)

    # Filter to only include nodes with ground truth for modularity calculation
    # igraph modularity expects all nodes to have a community
    # We'll use the ground truth labels directly
    try:
        modularity = graph.modularity(labels)
        return modularity
    except Exception:
        return None


# =============================================================================
# LFR Benchmark Generation
# =============================================================================

def generate_lfr(n, tau1, tau2, mu, average_degree, min_community, max_community, seed, max_retries=20):
    """
    Generate LFR benchmark graph with ground truth communities.

    Args:
        n: number of nodes
        tau1: degree distribution power law exponent (typically 2-3)
        tau2: community size distribution power law exponent (typically 1-2)
        mu: mixing parameter (fraction of inter-community edges, 0-1)
        average_degree: average node degree
        min_community: minimum community size
        max_community: maximum community size
        seed: random seed for reproducibility
        max_retries: maximum attempts if generation fails

    Returns:
        tuple: (edges, n_nodes, ground_truth)
            - edges: list of (src, dst) tuples (both directions)
            - n_nodes: number of nodes
            - ground_truth: dict mapping node -> community_id
    """
    from networkx.generators.community import LFR_benchmark_graph

    # Set max_degree to prevent very high degree nodes
    max_degree = min(n - 1, int(average_degree * 3))

    for attempt in range(max_retries):
        try:
            G = LFR_benchmark_graph(
                n=n,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=average_degree,
                max_degree=max_degree,
                min_community=min_community,
                max_community=max_community,
                seed=seed + attempt,  # Different seed each retry
                max_iters=5000
            )

            # Extract ground truth communities
            # Each node has a 'community' attribute (a frozenset of nodes in its community)
            communities = {frozenset(G.nodes[v]["community"]) for v in G}
            ground_truth = {}
            for comm_id, comm in enumerate(sorted(communities, key=lambda x: min(x))):
                for node in comm:
                    ground_truth[node] = comm_id

            # Convert to edge list (both directions for undirected)
            edges = []
            for u, v in G.edges():
                edges.append((u, v))
                edges.append((v, u))

            return edges, n, ground_truth

        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"LFR generation failed after {max_retries} attempts: {e}")
            continue

    raise RuntimeError("LFR generation failed")


# =============================================================================
# Direct Sparsification (No Caching)
# =============================================================================

LFR_TEMP_DIR = PROJECT_ROOT / "temp_lfr"


def spectral_sparsify_direct(edges, n_nodes, epsilon):
    """
    Run spectral sparsification without caching (for synthetic graphs).

    Args:
        edges: list of (src, dst) tuples
        n_nodes: number of nodes
        epsilon: sparsification parameter (smaller = more edges retained)

    Returns:
        tuple: (sparsified_edges, elapsed_time)
            - sparsified_edges: list of (src, dst) tuples
            - elapsed_time: time in seconds
    """
    import tempfile

    # Create temp directory if needed
    LFR_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', dir=LFR_TEMP_DIR, delete=False) as f:
        edges_file = Path(f.name)
        for s, d in edges:
            f.write(f"{s} {d}\n")

    try:
        # Run Julia sparsification
        julia_path = get_julia_path()

        env = os.environ.copy()
        env['JULIA_DEPOT_PATH'] = str(JULIA_DEPOT)

        cmd = [
            julia_path,
            f"--project={JULIA_PROJECT}",
            str(SPARSIFY_SCRIPT),
            str(edges_file),
            str(n_nodes),
            str(epsilon)
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=LFR_TEMP_DIR,
            capture_output=True,
            text=True,
            env=env
        )
        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            raise RuntimeError(f"Spectral sparsification failed: {result.stderr}")

        # Load output (Julia writes to current directory with specific name)
        output_file = LFR_TEMP_DIR / f"edges_sparsified_eps{epsilon}.txt"
        sparsified_edges = load_edges_from_file(output_file)

        # Clean up output file
        output_file.unlink(missing_ok=True)

        return sparsified_edges, elapsed_time

    finally:
        # Clean up input file
        edges_file.unlink(missing_ok=True)


# =============================================================================
# Noise Addition for LFR Graphs
# =============================================================================

def add_noise_edges(edges, n_nodes, ground_truth, noise_ratio, seed=None):
    """
    Add random edges to a graph as noise.

    Args:
        edges: list of (src, dst) tuples (bidirectional)
        n_nodes: number of nodes
        ground_truth: dict mapping node -> community_id
        noise_ratio: fraction of original edges to add as noise (e.g., 0.1 = 10%)
        seed: random seed for reproducibility

    Returns:
        noisy_edges: edges + random noise edges
        noise_stats: dict with statistics about added noise
    """
    import random

    if seed is not None:
        random.seed(seed)

    # Get unique undirected edges
    original_edge_set = set((min(s, d), max(s, d)) for s, d in edges)
    n_original = len(original_edge_set)
    n_noise_to_add = int(n_original * noise_ratio)

    if n_noise_to_add == 0:
        return edges, {
            'n_original_edges': n_original,
            'n_noise_added': 0,
            'noise_ratio_actual': 0.0,
            'noise_inter': 0,
            'noise_intra': 0,
            'noise_inter_ratio': 0.0
        }

    # Add random edges (not already in graph)
    noise_edges = set()
    attempts = 0
    max_attempts = n_noise_to_add * 20

    while len(noise_edges) < n_noise_to_add and attempts < max_attempts:
        u = random.randint(0, n_nodes - 1)
        v = random.randint(0, n_nodes - 1)
        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in original_edge_set and edge not in noise_edges:
                noise_edges.add(edge)
        attempts += 1

    # Convert to bidirectional edge list
    noisy_edges = list(edges)  # Keep original
    for u, v in noise_edges:
        noisy_edges.append((u, v))
        noisy_edges.append((v, u))

    # Compute noise statistics
    # How many noise edges are inter vs intra community?
    noise_inter = 0
    noise_intra = 0
    for u, v in noise_edges:
        if u in ground_truth and v in ground_truth:
            if ground_truth[u] == ground_truth[v]:
                noise_intra += 1
            else:
                noise_inter += 1

    noise_stats = {
        'n_original_edges': n_original,
        'n_noise_added': len(noise_edges),
        'noise_ratio_actual': len(noise_edges) / n_original if n_original > 0 else 0,
        'noise_inter': noise_inter,
        'noise_intra': noise_intra,
        'noise_inter_ratio': noise_inter / len(noise_edges) if noise_edges else 0
    }

    return noisy_edges, noise_stats
