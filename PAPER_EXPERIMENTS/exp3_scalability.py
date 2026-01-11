#!/usr/bin/env python3
"""
Experiment 3: Scalability and Large-Graph Tradeoffs

Purpose: Evaluate runtime scalability, end-to-end speedup, and community quality
tradeoffs on large real-world graphs when using DSpar sparsification.

Key measurements:
  - Sparsification time and Leiden runtime (original vs sparsified)
  - End-to-end pipeline time = sparsifier + Leiden on sparsified graph
  - Speedup over running Leiden on the original graph
  - Community quality: ΔQ_fixed (theory-aligned) and ΔQ_Leiden (pipeline)

Baselines:
  - Uniform random edge sampling (no replacement)
  - Degree-aware sampling (probability ~ deg(u) + deg(v))

Datasets:
  - com-DBLP
  - web-Google
  - com-LiveJournal
  - cit-Patents (if feasible)

Usage:
    python exp3_scalability.py
    python exp3_scalability.py --datasets com-DBLP
    python exp3_scalability.py --datasets com-DBLP,web-Google --dry_run
    python exp3_scalability.py --max_edges 1000000

Outputs (in results/exp3_scalability/):
    - scalability_raw.csv
    - scalability_summary.csv
    - scalability_table_alpha0.8.tex
    - figures/plot1_scaling_sparsify_time.pdf/png
    - figures/plot2_scaling_pipeline_time.pdf/png
    - figures/plot3_quality_vs_alpha_{dataset}.pdf/png
    - figures/plot4_speedup_vs_quality_{dataset}.pdf/png
"""

import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import networkx as nx
import igraph as ig
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt

from experiments.dspar import dspar_sparsify
from experiments.utils import load_snap_dataset, SNAP_DATASETS

# Try to import spectral sparsification (requires Julia)
try:
    from experiments.utils import spectral_sparsify_direct
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False
    print("WARNING: Spectral sparsification not available (Julia not configured)")

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "results" / "exp3_scalability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Retention levels to test
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Number of replicates per configuration
N_REPLICATES = 3

# Sparsification methods
METHODS = ['dspar', 'uniform_random', 'degree_sampling']

# Spectral is included by default (excluded via --no_spectral flag)
INCLUDE_SPECTRAL = True  # Set via CLI argument

# Spectral epsilon values corresponding to approximate retention levels
# Smaller epsilon = more edges retained (better approximation)
# These are rough mappings: epsilon -> ~retention
SPECTRAL_EPSILON_MAP = {
    0.2: 3.0,   # Very aggressive sparsification
    0.4: 1.5,   # Aggressive
    0.6: 0.8,   # Moderate
    0.8: 0.3,   # Conservative
    1.0: 0.0,   # No sparsification (will skip)
}

# Maximum time (seconds) to wait for spectral sparsification per graph
SPECTRAL_TIMEOUT = 300  # 5 minutes

# Default datasets (large real-world graphs)
# Note: web-Google and cit-Patents not in current registry
DEFAULT_DATASETS = [
    'com-DBLP',
    'com-Amazon',
    'com-Youtube',
    'com-LiveJournal',
]

# Seed base for reproducibility
SEED_BASE = 42

# Publication plot settings (matching prior experiments exactly)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (5, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'errorbar.capsize': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Colorblind-friendly palette
COLORS = {
    'dspar': '#0072B2',           # Blue
    'uniform_random': '#D55E00',  # Orange
    'degree_sampling': '#009E73', # Green
    'spectral': '#CC79A7',        # Pink (if available)
}

MARKERS = {
    'dspar': 'o',
    'uniform_random': 's',
    'degree_sampling': '^',
    'spectral': 'D',
}

LINESTYLES = {
    'dspar': '-',
    'uniform_random': '--',
    'degree_sampling': '-.',
    'spectral': ':',
}


# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_large_dataset(name: str, max_edges: int = None):
    """
    Load a large dataset, ensuring it is undirected and simple.

    Args:
        name: Dataset name
        max_edges: Maximum edges to keep (for testing); None = no limit

    Returns:
        G: NetworkX graph (undirected, simple)
        info: dict with dataset statistics
    """
    print(f"\n  Loading {name}...")

    # Map to SNAP names if needed
    snap_name = name

    # Check if dataset is available
    if snap_name not in SNAP_DATASETS:
        print(f"    [WARNING] Dataset {name} not in registry, skipping")
        return None, None

    try:
        edges, n_nodes_raw, ground_truth = load_snap_dataset(snap_name)
    except Exception as e:
        print(f"    [ERROR] Failed to load {name}: {e}")
        return None, None

    # Build NetworkX graph
    G = nx.Graph()

    # Add edges (undirected, remove self-loops)
    edge_set = set()
    self_loops_removed = 0
    multi_edges_collapsed = 0

    for u, v in edges:
        if u == v:
            self_loops_removed += 1
            continue
        edge = (min(u, v), max(u, v))
        if edge in edge_set:
            multi_edges_collapsed += 1
        else:
            edge_set.add(edge)

    G.add_edges_from(edge_set)

    # Take largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    n_nodes = G.number_of_nodes()
    m_edges = G.number_of_edges()

    # Apply max_edges limit if specified
    if max_edges is not None and m_edges > max_edges:
        print(f"    Subsampling from {m_edges:,} to {max_edges:,} edges for testing...")
        edges_list = list(G.edges())
        np.random.seed(SEED_BASE)
        keep_indices = np.random.choice(len(edges_list), size=max_edges, replace=False)
        G_sub = nx.Graph()
        G_sub.add_nodes_from(range(n_nodes))
        G_sub.add_edges_from([edges_list[i] for i in keep_indices])

        # Take LCC again
        if not nx.is_connected(G_sub):
            largest_cc = max(nx.connected_components(G_sub), key=len)
            G_sub = G_sub.subgraph(largest_cc).copy()
            G_sub = nx.convert_node_labels_to_integers(G_sub)

        G = G_sub
        n_nodes = G.number_of_nodes()
        m_edges = G.number_of_edges()

    info = {
        'name': name,
        'n_nodes': n_nodes,
        'm_edges': m_edges,
        'directed': False,
        'self_loops_removed': self_loops_removed,
        'multi_edges_collapsed': multi_edges_collapsed,
        'is_connected': True,
    }

    print(f"    Nodes: {n_nodes:,}")
    print(f"    Edges: {m_edges:,}")
    print(f"    Self-loops removed: {self_loops_removed:,}")
    print(f"    Multi-edges collapsed: {multi_edges_collapsed:,}")
    print(f"    Status: undirected, simple, connected")

    return G, info


# =============================================================================
# SPARSIFICATION METHODS
# =============================================================================

def sparsify_dspar(G: nx.Graph, alpha: float, seed: int) -> nx.Graph:
    """
    DSpar sparsification using the original paper method.

    Uses sampling WITH replacement as described in the DSpar paper.
    Returns unweighted graph (topology only) for Leiden clustering.
    """
    G_sparse_weighted = dspar_sparsify(G, retention=alpha, method='paper', seed=seed)
    # Convert to unweighted graph (keep only topology)
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G_sparse_weighted.nodes())
    G_sparse.add_edges_from(G_sparse_weighted.edges())
    return G_sparse


def sparsify_uniform_random(G: nx.Graph, alpha: float, seed: int) -> nx.Graph:
    """
    Uniform random edge sampling without replacement.
    Each edge has equal probability of being kept.
    """
    np.random.seed(seed)

    edges = list(G.edges())
    m = len(edges)
    n_keep = int(np.ceil(alpha * m))

    if n_keep >= m:
        return G.copy()

    # Sample without replacement
    keep_indices = np.random.choice(m, size=n_keep, replace=False)

    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes())
    G_sparse.add_edges_from([edges[i] for i in keep_indices])

    return G_sparse


def sparsify_degree_sampling(G: nx.Graph, alpha: float, seed: int) -> nx.Graph:
    """
    Degree-aware edge sampling.
    Edge (u,v) kept with probability proportional to deg(u) + deg(v).
    Normalized to achieve approximately alpha * m edges.
    """
    np.random.seed(seed)

    edges = list(G.edges())
    m = len(edges)
    degrees = dict(G.degree())

    # Compute sampling probabilities proportional to deg(u) + deg(v)
    weights = np.array([degrees[u] + degrees[v] for u, v in edges], dtype=float)

    # Normalize to get expected alpha * m edges
    n_keep = int(np.ceil(alpha * m))
    probs = weights / weights.sum() * n_keep
    probs = np.clip(probs, 0, 1)

    # Sample each edge independently
    random_vals = np.random.random(m)
    keep_mask = random_vals < probs

    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes())
    G_sparse.add_edges_from([edges[i] for i in range(m) if keep_mask[i]])

    return G_sparse


def sparsify_spectral(G: nx.Graph, alpha: float, seed: int) -> nx.Graph:
    """
    Spectral sparsification using Julia's Laplacians.jl.

    Uses effective resistance sampling (Spielman-Srivastava algorithm).
    Maps alpha (retention) to epsilon (approximation quality).

    Returns:
        Sparsified graph, or None if spectral sparsification fails/times out
    """
    if not SPECTRAL_AVAILABLE:
        raise RuntimeError("Spectral sparsification not available (Julia not configured)")

    # Get epsilon for this alpha level
    epsilon = SPECTRAL_EPSILON_MAP.get(alpha)
    if epsilon is None or epsilon <= 0:
        # For alpha=1.0 or unknown, return copy
        return G.copy()

    # Convert to edge list
    edges = list(G.edges())
    n_nodes = G.number_of_nodes()

    # Need to map node labels to 0-indexed if not already
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    idx_to_node = {i: n for i, n in enumerate(node_list)}

    edges_indexed = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]

    try:
        # Run spectral sparsification with timeout
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Spectral sparsification timed out after {SPECTRAL_TIMEOUT}s")

        # Set timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(SPECTRAL_TIMEOUT)

        try:
            sparsified_edges, elapsed = spectral_sparsify_direct(edges_indexed, n_nodes, epsilon)
        finally:
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)

        # Build sparsified graph
        G_sparse = nx.Graph()
        G_sparse.add_nodes_from(G.nodes())

        # Convert back to original node labels
        for u_idx, v_idx in sparsified_edges:
            u = idx_to_node[u_idx]
            v = idx_to_node[v_idx]
            G_sparse.add_edge(u, v)

        return G_sparse

    except TimeoutError as e:
        print(f"\n    [SPECTRAL TIMEOUT] {e}")
        return None
    except Exception as e:
        print(f"\n    [SPECTRAL ERROR] {e}")
        return None


def sparsify(G: nx.Graph, method: str, alpha: float, seed: int) -> nx.Graph:
    """
    Dispatch to appropriate sparsification method.

    Returns:
        Sparsified graph, or None if method fails (e.g., spectral timeout)
    """
    if method == 'dspar':
        return sparsify_dspar(G, alpha, seed)
    elif method == 'uniform_random':
        return sparsify_uniform_random(G, alpha, seed)
    elif method == 'degree_sampling':
        return sparsify_degree_sampling(G, alpha, seed)
    elif method == 'spectral':
        return sparsify_spectral(G, alpha, seed)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# COMMUNITY DETECTION
# =============================================================================

def nx_to_igraph(G: nx.Graph) -> ig.Graph:
    """Convert NetworkX graph to igraph."""
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    return ig.Graph(n=len(node_list), edges=edges, directed=False)


def run_leiden(G: nx.Graph):
    """
    Run Leiden clustering using igraph's implementation.

    Returns:
        membership: dict {node: community_id}
        modularity: float
        runtime: float (seconds)
    """
    node_list = list(G.nodes())
    ig_graph = nx_to_igraph(G)

    start = time.perf_counter()
    partition = ig_graph.community_leiden(
        objective_function='modularity',
        resolution=1.0,
        n_iterations=2
    )
    runtime = time.perf_counter() - start

    membership = {node_list[i]: partition.membership[i] for i in range(len(node_list))}

    return membership, partition.modularity, runtime


def compute_modularity_fixed(G: nx.Graph, membership: dict) -> float:
    """
    Compute modularity for a FIXED partition on graph G.
    Uses igraph's modularity function.
    """
    node_list = list(G.nodes())
    ig_graph = nx_to_igraph(G)

    # Build membership list in igraph order
    membership_list = [membership.get(node_list[i], 0) for i in range(len(node_list))]

    return ig_graph.modularity(membership_list)


def compute_nmi(partition1: dict, partition2: dict, nodes: list) -> float:
    """Compute NMI between two partitions."""
    labels1 = [partition1.get(n, -1) for n in nodes]
    labels2 = [partition2.get(n, -1) for n in nodes]
    return normalized_mutual_info_score(labels1, labels2)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_single_trial(G: nx.Graph, baseline_partition: dict, Q0: float,
                     T_leiden_orig: float, method: str, alpha: float,
                     replicate: int, seed: int, dataset_info: dict) -> dict:
    """
    Run a single experimental trial.

    Returns:
        dict with all measurements
    """
    n_nodes = G.number_of_nodes()
    m_edges = G.number_of_edges()
    nodes = list(G.nodes())

    # Handle alpha = 1.0 (no sparsification)
    if alpha >= 1.0:
        G_sparse = G.copy()
        T_sparsify = 0.0
    else:
        # Sparsify
        start = time.perf_counter()
        G_sparse = sparsify(G, method, alpha, seed)
        T_sparsify = time.perf_counter() - start

    # Handle failed sparsification (e.g., spectral timeout)
    if G_sparse is None:
        return None

    m_sparse = G_sparse.number_of_edges()
    retention_actual = m_sparse / m_edges if m_edges > 0 else 1.0

    # Compute Q_sparse_fixed (modularity of baseline partition on sparsified graph)
    Q_sparse_fixed = compute_modularity_fixed(G_sparse, baseline_partition)
    dQ_fixed = Q_sparse_fixed - Q0

    # Run Leiden on sparsified graph
    partition_sparse, Q_sparse_leiden, T_leiden_sparse = run_leiden(G_sparse)
    dQ_leiden = Q_sparse_leiden - Q0

    # Pipeline time and speedup
    T_pipeline = T_sparsify + T_leiden_sparse
    speedup = T_leiden_orig / T_pipeline if T_pipeline > 0 else 1.0

    # Community counts
    n_communities_orig = len(set(baseline_partition.values()))
    n_communities_sparse = len(set(partition_sparse.values()))

    # Optional: NMI between baseline and sparse partitions
    nmi_P0_Palpha = compute_nmi(baseline_partition, partition_sparse, nodes)

    return {
        'dataset': dataset_info['name'],
        'method': method,
        'alpha': alpha,
        'replicate': replicate,
        'seed': seed,

        'n_nodes': n_nodes,
        'm_edges': m_edges,
        'm_sparse': m_sparse,
        'retention_actual': retention_actual,

        'T_sparsify_sec': T_sparsify,
        'T_leiden_orig_sec': T_leiden_orig,
        'T_leiden_sparse_sec': T_leiden_sparse,
        'T_pipeline_sec': T_pipeline,
        'speedup': speedup,

        'Q0': Q0,
        'Q_sparse_fixed': Q_sparse_fixed,
        'dQ_fixed': dQ_fixed,
        'Q_sparse_leiden': Q_sparse_leiden,
        'dQ_leiden': dQ_leiden,

        'n_communities_orig': n_communities_orig,
        'n_communities_sparse': n_communities_sparse,
        'nmi_P0_Palpha': nmi_P0_Palpha,
    }


def run_dataset_experiments(G: nx.Graph, dataset_info: dict,
                            methods: list, alphas: list,
                            n_replicates: int,
                            spectral_timeout_multiplier: float = 10.0) -> list:
    """
    Run all experiments for a single dataset.

    Strategy for spectral:
      1. Run DSpar first to measure baseline sparsification time
      2. Set spectral timeout = spectral_timeout_multiplier × max(DSpar time)
      3. Run spectral with dynamic timeout
    """
    global SPECTRAL_TIMEOUT

    results = []

    n_nodes = G.number_of_nodes()
    m_edges = G.number_of_edges()

    print(f"\n  Running Leiden on original graph...")
    baseline_partition, Q0, T_leiden_orig = run_leiden(G)
    n_communities_orig = len(set(baseline_partition.values()))

    print(f"    Baseline modularity Q0 = {Q0:.6f}")
    print(f"    Baseline communities = {n_communities_orig}")
    print(f"    Baseline Leiden time = {T_leiden_orig:.3f}s")

    # Separate methods: run non-spectral first, then spectral
    non_spectral_methods = [m for m in methods if m != 'spectral']
    has_spectral = 'spectral' in methods

    # Track DSpar times to set spectral timeout
    dspar_times = []

    # Phase 1: Run non-spectral methods (including DSpar)
    total_non_spectral = len(non_spectral_methods) * len(alphas) * n_replicates
    trial_count = 0

    print(f"\n  Phase 1: Running non-spectral methods...")

    for method in non_spectral_methods:
        for alpha in alphas:
            for rep in range(n_replicates):
                trial_count += 1

                seed = SEED_BASE + hash((dataset_info['name'], method, alpha, rep)) % (2**20)

                print(f"\r    [{trial_count}/{total_non_spectral}] {method}, α={alpha:.1f}, rep={rep+1}",
                      end='', flush=True)

                result = run_single_trial(
                    G, baseline_partition, Q0, T_leiden_orig,
                    method, alpha, rep, seed, dataset_info
                )
                if result is not None:
                    results.append(result)
                    # Track DSpar times
                    if method == 'dspar' and alpha < 1.0:
                        dspar_times.append(result['T_sparsify_sec'])
                else:
                    print(f" [SKIPPED]", end='')

    print()

    # Phase 2: Run spectral with dynamic timeout based on DSpar
    if has_spectral and len(dspar_times) > 0:
        max_dspar_time = max(dspar_times)
        dynamic_timeout = int(max_dspar_time * spectral_timeout_multiplier)
        dynamic_timeout = max(dynamic_timeout, 30)  # At least 30 seconds

        print(f"\n  Phase 2: Running spectral sparsification...")
        print(f"    Max DSpar time: {max_dspar_time:.2f}s")
        print(f"    Spectral timeout: {dynamic_timeout}s ({spectral_timeout_multiplier}× DSpar)")

        SPECTRAL_TIMEOUT = dynamic_timeout

        total_spectral = len(alphas) * n_replicates
        trial_count = 0

        for alpha in alphas:
            for rep in range(n_replicates):
                trial_count += 1

                seed = SEED_BASE + hash((dataset_info['name'], 'spectral', alpha, rep)) % (2**20)

                print(f"\r    [{trial_count}/{total_spectral}] spectral, α={alpha:.1f}, rep={rep+1}",
                      end='', flush=True)

                result = run_single_trial(
                    G, baseline_partition, Q0, T_leiden_orig,
                    'spectral', alpha, rep, seed, dataset_info
                )
                if result is not None:
                    results.append(result)
                else:
                    print(f" [SKIPPED]", end='')

        print()

    elif has_spectral:
        print(f"\n  [WARNING] No DSpar times recorded, skipping spectral")

    return results


def run_all_experiments(datasets: list, methods: list, alphas: list,
                        n_replicates: int, max_edges: int = None,
                        spectral_timeout_multiplier: float = 10.0) -> pd.DataFrame:
    """
    Run experiments on all datasets.
    """
    all_results = []

    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*80}")

        G, info = load_large_dataset(dataset_name, max_edges=max_edges)

        if G is None:
            print(f"  [SKIP] Could not load {dataset_name}")
            continue

        results = run_dataset_experiments(G, info, methods, alphas, n_replicates,
                                          spectral_timeout_multiplier=spectral_timeout_multiplier)
        all_results.extend(results)

    return pd.DataFrame(all_results)


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics grouped by dataset, method, alpha."""
    agg_cols = [
        'dQ_fixed', 'dQ_leiden',
        'T_sparsify_sec', 'T_leiden_sparse_sec', 'T_pipeline_sec',
        'speedup', 'nmi_P0_Palpha', 'retention_actual'
    ]

    summary = df.groupby(['dataset', 'method', 'alpha'])[agg_cols].agg(['mean', 'std'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    # Add baseline Leiden time (single value per dataset)
    baseline_times = df.groupby('dataset')['T_leiden_orig_sec'].first().reset_index()
    baseline_times.columns = ['dataset', 'T_leiden_orig_sec']
    summary = summary.merge(baseline_times, on='dataset')

    # Add baseline Q0
    baseline_Q = df.groupby('dataset')['Q0'].first().reset_index()
    summary = summary.merge(baseline_Q, on='dataset')

    return summary


def generate_latex_table(df: pd.DataFrame, output_dir: Path, alpha: float = 0.8) -> Path:
    """
    Generate LaTeX table for a specific alpha value.
    """
    df_alpha = df[np.isclose(df['alpha'], alpha)]

    if len(df_alpha) == 0:
        print(f"  [WARNING] No data for alpha={alpha}")
        return None

    # Aggregate by dataset and method
    agg = df_alpha.groupby(['dataset', 'method']).agg({
        'n_nodes': 'first',
        'm_edges': 'first',
        'T_sparsify_sec': ['mean', 'std'],
        'T_leiden_sparse_sec': ['mean', 'std'],
        'speedup': ['mean', 'std'],
        'dQ_fixed': ['mean', 'std'],
        'dQ_leiden': ['mean', 'std'],
    })
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
    agg = agg.reset_index()

    def fmt_val(mean, std, decimals=3):
        if std > 0:
            return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"
        return f"${mean:.{decimals}f}$"

    def fmt_time(mean, std):
        if mean < 1:
            return f"${mean*1000:.0f} \\pm {std*1000:.0f}$ ms"
        return f"${mean:.2f} \\pm {std:.2f}$ s"

    def escape_latex(s):
        return s.replace("_", "\\_").replace("-", "-")

    method_names = {
        'dspar': 'DSpar',
        'uniform_random': 'Uniform',
        'degree_sampling': 'Degree',
    }

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{Scalability results at $\\alpha = {alpha}$.}}",
        r"\label{tab:exp3_scalability}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Dataset & Method & $n$ & $m$ & $T_{\mathrm{spar}}$ & $T_{\mathrm{Leiden}}$ & Speedup & $\Delta Q_{\mathrm{fixed}}$ & $\Delta Q_{\mathrm{Leiden}}$ \\",
        r"\midrule",
    ]

    for dataset in agg['dataset'].unique():
        df_d = agg[agg['dataset'] == dataset]
        first_row = True

        for _, row in df_d.iterrows():
            if first_row:
                ds_str = escape_latex(dataset)
                n_str = f"{int(row['n_nodes_first']):,}"
                m_str = f"{int(row['m_edges_first']):,}"
                first_row = False
            else:
                ds_str = ""
                n_str = ""
                m_str = ""

            method_str = method_names.get(row['method'], row['method'])
            t_spar = fmt_time(row['T_sparsify_sec_mean'], row['T_sparsify_sec_std'])
            t_leiden = fmt_time(row['T_leiden_sparse_sec_mean'], row['T_leiden_sparse_sec_std'])
            speedup_str = fmt_val(row['speedup_mean'], row['speedup_std'], 2)
            dQ_fixed = fmt_val(row['dQ_fixed_mean'], row['dQ_fixed_std'], 4)
            dQ_leiden = fmt_val(row['dQ_leiden_mean'], row['dQ_leiden_std'], 4)

            lines.append(
                f"{ds_str} & {method_str} & {n_str} & {m_str} & "
                f"{t_spar} & {t_leiden} & {speedup_str} & {dQ_fixed} & {dQ_leiden} \\\\"
            )

        lines.append(r"\midrule")

    # Remove last midrule
    lines[-1] = r"\bottomrule"

    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path = output_dir / f"scalability_table_alpha{alpha}.tex"
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


# =============================================================================
# PLOTTING
# =============================================================================

def plot_scaling_sparsify_time(df: pd.DataFrame, output_dir: Path):
    """
    Plot 1: Sparsification runtime scaling with graph size.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # Use alpha=0.8 data
    df_plot = df[np.isclose(df['alpha'], 0.8)]

    # Get methods from data
    methods_in_data = df_plot['method'].unique()

    for method in methods_in_data:
        df_m = df_plot[df_plot['method'] == method]

        if len(df_m) == 0:
            continue

        # Aggregate by dataset
        agg = df_m.groupby('dataset').agg({
            'm_edges': 'first',
            'T_sparsify_sec': ['mean', 'std']
        }).reset_index()
        agg.columns = ['dataset', 'm_edges', 'T_mean', 'T_std']
        agg = agg.sort_values('m_edges')

        ax.errorbar(
            agg['m_edges'], agg['T_mean'], yerr=agg['T_std'],
            fmt=MARKERS.get(method, 'o') + LINESTYLES.get(method, '-'),
            color=COLORS.get(method, '#666666'),
            label=method.replace('_', ' ').title(),
            capsize=2, markersize=5
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of edges $m$')
    ax.set_ylabel(r'Sparsification time $T_{\mathrm{spar}}$ (s)')
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'plot1_scaling_sparsify_time.pdf', format='pdf')
    fig.savefig(output_dir / 'plot1_scaling_sparsify_time.png', format='png')
    plt.close(fig)


def plot_scaling_pipeline_time(df: pd.DataFrame, output_dir: Path):
    """
    Plot 2: End-to-end pipeline time scaling.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # Use alpha=0.8 data
    df_plot = df[np.isclose(df['alpha'], 0.8)]

    # Get methods from data
    methods_in_data = df_plot['method'].unique()

    for method in methods_in_data:
        df_m = df_plot[df_plot['method'] == method]

        if len(df_m) == 0:
            continue

        agg = df_m.groupby('dataset').agg({
            'm_edges': 'first',
            'T_pipeline_sec': ['mean', 'std']
        }).reset_index()
        agg.columns = ['dataset', 'm_edges', 'T_mean', 'T_std']
        agg = agg.sort_values('m_edges')

        ax.errorbar(
            agg['m_edges'], agg['T_mean'], yerr=agg['T_std'],
            fmt=MARKERS.get(method, 'o') + LINESTYLES.get(method, '-'),
            color=COLORS.get(method, '#666666'),
            label=method.replace('_', ' ').title(),
            capsize=2, markersize=5
        )

    # Also plot baseline Leiden time
    baseline = df_plot.groupby('dataset').agg({
        'm_edges': 'first',
        'T_leiden_orig_sec': 'first'
    }).reset_index()
    baseline = baseline.sort_values('m_edges')

    ax.plot(baseline['m_edges'], baseline['T_leiden_orig_sec'],
            'k--', linewidth=1.5, label='Leiden (original)', alpha=0.7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of edges $m$')
    ax.set_ylabel(r'Pipeline time $T_{\mathrm{pipe}}$ (s)')
    ax.legend(loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'plot2_scaling_pipeline_time.pdf', format='pdf')
    fig.savefig(output_dir / 'plot2_scaling_pipeline_time.png', format='png')
    plt.close(fig)


def plot_quality_vs_alpha(df: pd.DataFrame, output_dir: Path):
    """
    Plot 3: Quality (dQ_leiden) vs retention alpha, per dataset.
    """
    for dataset in df['dataset'].unique():
        df_d = df[df['dataset'] == dataset]

        fig, ax = plt.subplots(figsize=(5, 4))

        # Get methods from data
        methods_in_data = df_d['method'].unique()

        for method in methods_in_data:
            df_m = df_d[df_d['method'] == method]

            if len(df_m) == 0:
                continue

            agg = df_m.groupby('alpha').agg({
                'dQ_leiden': ['mean', 'std']
            }).reset_index()
            agg.columns = ['alpha', 'dQ_mean', 'dQ_std']

            ax.errorbar(
                agg['alpha'], agg['dQ_mean'], yerr=agg['dQ_std'],
                fmt=MARKERS.get(method, 'o') + LINESTYLES.get(method, '-'),
                color=COLORS.get(method, '#666666'),
                label=method.replace('_', ' ').title(),
                capsize=2, markersize=5
            )

        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_xlabel(r'Retention $\alpha$')
        ax.set_ylabel(r'$\Delta Q_{\mathrm{Leiden}}$')
        ax.legend(loc='best', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        safe_name = dataset.replace('-', '_').replace('.', '_')
        fig.savefig(output_dir / f'plot3_quality_vs_alpha_{safe_name}.pdf', format='pdf')
        fig.savefig(output_dir / f'plot3_quality_vs_alpha_{safe_name}.png', format='png')
        plt.close(fig)


def plot_speedup_vs_quality(df: pd.DataFrame, output_dir: Path):
    """
    Plot 4: Speedup vs quality tradeoff, per dataset.
    """
    for dataset in df['dataset'].unique():
        df_d = df[df['dataset'] == dataset]

        fig, ax = plt.subplots(figsize=(5, 4))

        # Get methods from data
        methods_in_data = df_d['method'].unique()

        for method in methods_in_data:
            df_m = df_d[df_d['method'] == method]

            if len(df_m) == 0:
                continue

            agg = df_m.groupby('alpha').agg({
                'speedup': 'mean',
                'dQ_leiden': 'mean'
            }).reset_index()

            # Plot as connected points (each point is an alpha level)
            ax.plot(
                agg['speedup'], agg['dQ_leiden'],
                marker=MARKERS.get(method, 'o'), linestyle=LINESTYLES.get(method, '-'),
                color=COLORS.get(method, '#666666'),
                label=method.replace('_', ' ').title(),
                markersize=5
            )

            # Annotate alpha values on DSpar line
            if method == 'dspar':
                for _, row in agg.iterrows():
                    if row['alpha'] < 1.0:
                        ax.annotate(
                            f"$\\alpha$={row['alpha']:.1f}",
                            (row['speedup'], row['dQ_leiden']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, alpha=0.7
                        )

        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.axvline(1, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Speedup')
        ax.set_ylabel(r'$\Delta Q_{\mathrm{Leiden}}$')
        ax.legend(loc='best', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        safe_name = dataset.replace('-', '_').replace('.', '_')
        fig.savefig(output_dir / f'plot4_speedup_vs_quality_{safe_name}.pdf', format='pdf')
        fig.savefig(output_dir / f'plot4_speedup_vs_quality_{safe_name}.png', format='png')
        plt.close(fig)


def generate_all_plots(df: pd.DataFrame, output_dir: Path):
    """Generate all publication plots."""
    print("\nGenerating plots...")

    plot_scaling_sparsify_time(df, output_dir)
    print("  Plot 1: scaling_sparsify_time")

    plot_scaling_pipeline_time(df, output_dir)
    print("  Plot 2: scaling_pipeline_time")

    plot_quality_vs_alpha(df, output_dir)
    print(f"  Plot 3: quality_vs_alpha (per dataset)")

    plot_speedup_vs_quality(df, output_dir)
    print(f"  Plot 4: speedup_vs_quality (per dataset)")


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def print_key_results(df: pd.DataFrame, alpha: float = 0.8):
    """Print key results table for a specific alpha."""
    print(f"\n{'='*100}")
    print(f"KEY RESULTS AT alpha = {alpha}")
    print(f"{'='*100}")

    df_alpha = df[np.isclose(df['alpha'], alpha)]

    if len(df_alpha) == 0:
        print("No data available.")
        return

    print(f"\n{'Dataset':<15} {'Method':<18} {'T_spar(s)':<12} {'T_pipe(s)':<12} "
          f"{'Speedup':<10} {'dQ_fixed':<12} {'dQ_Leiden':<12}")
    print("-" * 100)

    # Get methods from data
    methods_in_data = df_alpha['method'].unique()

    for dataset in df_alpha['dataset'].unique():
        df_d = df_alpha[df_alpha['dataset'] == dataset]
        T_orig = df_d['T_leiden_orig_sec'].iloc[0]

        for method in methods_in_data:
            df_m = df_d[df_d['method'] == method]

            if len(df_m) == 0:
                continue

            T_spar = df_m['T_sparsify_sec'].mean()
            T_pipe = df_m['T_pipeline_sec'].mean()
            speedup = df_m['speedup'].mean()
            dQ_fixed = df_m['dQ_fixed'].mean()
            dQ_leiden = df_m['dQ_leiden'].mean()

            print(f"{dataset:<15} {method:<18} {T_spar:<12.4f} {T_pipe:<12.4f} "
                  f"{speedup:<10.2f} {dQ_fixed:<+12.6f} {dQ_leiden:<+12.6f}")

        print(f"{'':<15} {'(baseline)':<18} {'':<12} {T_orig:<12.4f} "
              f"{'1.00':<10} {'0.0':<12} {'0.0':<12}")
        print("-" * 100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    global SPECTRAL_TIMEOUT  # Declare at start of function

    parser = argparse.ArgumentParser(
        description="Experiment 3: Scalability and Large-Graph Tradeoffs"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets (default: com-DBLP,web-Google,com-LiveJournal)"
    )
    parser.add_argument(
        "--max_edges",
        type=int,
        default=None,
        help="Maximum edges to keep per dataset (for testing)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configuration and exit without running"
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=N_REPLICATES,
        help=f"Number of replicates per configuration (default: {N_REPLICATES})"
    )
    parser.add_argument(
        "--no_spectral",
        action="store_true",
        help="Exclude spectral sparsification (spectral is included by default)"
    )
    parser.add_argument(
        "--spectral_timeout",
        type=int,
        default=SPECTRAL_TIMEOUT,
        help=f"Initial timeout in seconds for spectral (default: {SPECTRAL_TIMEOUT}), overridden by dynamic timeout"
    )
    parser.add_argument(
        "--spectral_multiplier",
        type=float,
        default=10.0,
        help="Spectral timeout = multiplier × max(DSpar time). Default: 10.0"
    )

    args = parser.parse_args()

    # Determine datasets
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = DEFAULT_DATASETS

    n_replicates = args.replicates

    # Determine methods (spectral included by default unless --no_spectral)
    methods = METHODS.copy()
    if not args.no_spectral:
        if SPECTRAL_AVAILABLE:
            methods.append('spectral')
            SPECTRAL_TIMEOUT = args.spectral_timeout
        else:
            print("WARNING: Spectral sparsification not available (Julia not configured), skipping")
    else:
        print("NOTE: Spectral sparsification excluded (--no_spectral flag)")

    print("=" * 100)
    print("EXPERIMENT 3: SCALABILITY AND LARGE-GRAPH TRADEOFFS")
    print("=" * 100)

    print(f"\nConfiguration:")
    print(f"  Datasets: {datasets}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Methods: {methods}")
    print(f"  Replicates: {n_replicates}")
    print(f"  Max edges: {args.max_edges if args.max_edges else 'unlimited'}")
    print(f"  Output directory: {OUTPUT_DIR}")

    if 'spectral' in methods:
        print(f"\n  NOTE: Spectral sparsification enabled")
        print(f"        Dynamic timeout: {args.spectral_multiplier}× max(DSpar time) per dataset")
        print(f"        May fail/timeout on large graphs (O(m log n) space, O(m log^2 n / eps^2) time)")
    else:
        print(f"\n  NOTE: Spectral sparsification not included.")
        print(f"        Remove --no_spectral flag to enable (requires Julia, may timeout on large graphs)")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return

    # Run experiments
    df = run_all_experiments(
        datasets=datasets,
        methods=methods,
        alphas=ALPHAS,
        n_replicates=n_replicates,
        max_edges=args.max_edges,
        spectral_timeout_multiplier=args.spectral_multiplier
    )

    if len(df) == 0:
        print("\nERROR: No experiments completed successfully!")
        return

    # Save raw results
    raw_file = OUTPUT_DIR / "scalability_raw.csv"
    df.to_csv(raw_file, index=False)
    print(f"\nSaved raw results: {raw_file}")

    # Generate summary
    summary = generate_summary(df)
    summary_file = OUTPUT_DIR / "scalability_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")

    # Generate LaTeX table
    latex_file = generate_latex_table(df, OUTPUT_DIR, alpha=0.8)
    if latex_file:
        print(f"Saved LaTeX table: {latex_file}")

    # Generate plots
    generate_all_plots(df, FIGURES_DIR)

    # Print key results
    print_key_results(df, alpha=0.8)

    # Final summary
    print(f"\n{'='*100}")
    print("EXPERIMENT 3 COMPLETE")
    print(f"{'='*100}")
    print(f"\nOutput files:")
    print(f"  Raw CSV: {raw_file}")
    print(f"  Summary CSV: {summary_file}")
    print(f"  LaTeX table: {latex_file}")
    print(f"  Figures: {FIGURES_DIR}/")

    # Summary statistics
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}")

    # Get methods from data
    methods_in_data = df['method'].unique()

    for dataset in df['dataset'].unique():
        df_d = df[(df['dataset'] == dataset) & np.isclose(df['alpha'], 0.8)]

        print(f"\n{dataset}:")

        for method in methods_in_data:
            df_m = df_d[df_d['method'] == method]
            if len(df_m) == 0:
                continue

            speedup = df_m['speedup'].mean()
            dQ = df_m['dQ_leiden'].mean()

            if dQ > 0:
                quality_note = "improved"
            elif dQ < -0.01:
                quality_note = "degraded"
            else:
                quality_note = "maintained"

            print(f"  {method}: {speedup:.2f}x speedup, quality {quality_note} (ΔQ={dQ:+.4f})")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
