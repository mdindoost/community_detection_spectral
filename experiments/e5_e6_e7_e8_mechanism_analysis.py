"""
Experiments E5, E6, E7, E8: Understanding Why Sparsification Works

E5: Edge Preservation Analysis (Ratio Metric)
E6: Hub-Community Correlation
E7: Effective Resistance Analysis
E8: Min-Cut Edge Preservation

Prerequisites:
- Preprocessed LCC graphs from E1
- Baseline Leiden communities from E1
"""

import argparse
import numpy as np
import pandas as pd
import json
import sys
import time
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import networkx as nx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import (
    PROJECT_ROOT, DATASETS_DIR, RESULTS_DIR,
    spectral_sparsify_direct
)

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

E1_RESULTS_DIR = RESULTS_DIR / "E1_baseline"
E5_RESULTS_DIR = RESULTS_DIR / "E5_ratio"
E6_RESULTS_DIR = RESULTS_DIR / "E6_hub_analysis"
E7_RESULTS_DIR = RESULTS_DIR / "E7_effective_resistance"
E8_RESULTS_DIR = RESULTS_DIR / "E8_mincut"

# Datasets from E1
E1_DATASETS = ['email-Eu-core', 'cit-HepTh', 'cit-HepPh', 'com-DBLP', 'com-Youtube', 'test_network_1']

# Spectral sparsification parameters
SPECTRAL_EPSILONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# DSpar keep ratios
DSPAR_KEEP_RATIOS = [0.25, 0.40, 0.50, 0.60, 0.75, 0.90]

# Random baseline keep ratios
RANDOM_KEEP_RATIOS = [0.25, 0.50, 0.75]

# Hub threshold (top X% by degree)
HUB_PERCENTILE = 90  # Top 10% = 90th percentile

# ER sampling for large graphs
ER_SAMPLE_SIZE = 2000

# Min-cut: minimum community size
MIN_COMMUNITY_SIZE = 10

# Max communities to process for min-cut (for large graphs)
MAX_COMMUNITIES_MINCUT = 20

RANDOM_SEED = 42


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_e1_data(dataset_name):
    """Load preprocessed data from E1 experiment."""
    dataset_dir = E1_RESULTS_DIR / dataset_name

    # Load LCC edgelist
    edgelist_path = dataset_dir / f"{dataset_name}_lcc.edgelist"
    if not edgelist_path.exists():
        raise FileNotFoundError(f"E1 data not found: {edgelist_path}")

    edges = []
    with open(edgelist_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))

    # Get unique undirected edges
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        elif v < u:
            edge_set.add((v, u))
    undirected_edges = list(edge_set)

    # Get number of nodes
    node_set = set()
    for u, v in undirected_edges:
        node_set.add(u)
        node_set.add(v)
    n_nodes = max(node_set) + 1 if node_set else 0

    # Load baseline Leiden communities
    comm_path = dataset_dir / f"{dataset_name}_leiden_communities.tsv"
    communities = {}
    with open(comm_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                node, comm = int(parts[0]), int(parts[1])
                communities[node] = comm

    return undirected_edges, n_nodes, communities


def classify_edges(edges, communities):
    """Classify edges as intra-community or inter-community."""
    intra_edges = []
    inter_edges = []

    for u, v in edges:
        if u in communities and v in communities:
            if communities[u] == communities[v]:
                intra_edges.append((u, v))
            else:
                inter_edges.append((u, v))

    return intra_edges, inter_edges


# =============================================================================
# Sparsification Methods
# =============================================================================

def dspar_sparsify(edges, n_nodes, keep_ratio, seed=42):
    """DSpar sparsification with replacement sampling."""
    np.random.seed(seed)

    if not edges:
        return []

    # Compute degrees
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    # Compute DSpar scores
    probs = []
    for u, v in edges:
        p = 1.0 / degree[u] + 1.0 / degree[v]
        probs.append(p)
    probs = np.array(probs)
    probs = probs / probs.sum()

    # Number of edges to sample
    Q = int(len(edges) * keep_ratio)
    if Q == 0:
        Q = 1

    # Sample WITH replacement
    sampled_indices = np.random.choice(len(edges), size=Q, replace=True, p=probs)
    unique_indices = set(sampled_indices)

    # Return unique sampled edges
    return [edges[i] for i in unique_indices]


def random_sparsify(edges, keep_ratio, seed=42):
    """Random uniform sparsification."""
    np.random.seed(seed)

    n_keep = int(len(edges) * keep_ratio)
    if n_keep == 0:
        n_keep = 1

    indices = np.random.choice(len(edges), size=n_keep, replace=False)
    return [edges[i] for i in indices]


def run_spectral_sparsify(edges, n_nodes, epsilon):
    """Run spectral sparsification via Julia."""
    # Convert to bidirectional for spectral
    bidir_edges = []
    for u, v in edges:
        bidir_edges.append((u, v))
        bidir_edges.append((v, u))

    sparse_edges, elapsed = spectral_sparsify_direct(bidir_edges, n_nodes, epsilon)

    # Convert back to undirected
    edge_set = set()
    for u, v in sparse_edges:
        if u < v:
            edge_set.add((u, v))
        elif v < u:
            edge_set.add((v, u))

    return list(edge_set)


# =============================================================================
# E5: Edge Preservation Analysis
# =============================================================================

def compute_preservation_ratio(orig_intra, orig_inter, sparse_edges):
    """Compute the edge preservation ratio."""
    sparse_set = set(sparse_edges)

    # Count preserved edges
    intra_kept = sum(1 for e in orig_intra if e in sparse_set)
    inter_kept = sum(1 for e in orig_inter if e in sparse_set)

    # Compute percentages
    intra_pct = (intra_kept / len(orig_intra) * 100) if orig_intra else 0
    inter_pct = (inter_kept / len(orig_inter) * 100) if orig_inter else 0

    # Compute ratio
    if intra_pct > 0:
        ratio = inter_pct / intra_pct
    else:
        ratio = float('inf') if inter_pct > 0 else 1.0

    return {
        'intra_orig': len(orig_intra),
        'intra_kept': intra_kept,
        'intra_pct': intra_pct,
        'inter_orig': len(orig_inter),
        'inter_kept': inter_kept,
        'inter_pct': inter_pct,
        'ratio': ratio
    }


def run_e5_experiment():
    """E5: Edge Preservation Analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT E5: Edge Preservation Analysis (Ratio Metric)")
    print("=" * 70)

    E5_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_spectral = []
    results_dspar = []
    results_random = []
    results_summary = []

    for dataset_name in E1_DATASETS:
        print(f"\nProcessing {dataset_name}...")

        try:
            edges, n_nodes, communities = load_e1_data(dataset_name)
            intra_edges, inter_edges = classify_edges(edges, communities)

            print(f"  Edges: {len(edges)} total, {len(intra_edges)} intra, {len(inter_edges)} inter")

            spectral_ratios = []
            dspar_ratios = []
            random_ratios = []

            # E5.1: Spectral sparsification
            print("  Running spectral sparsification...")
            for eps in SPECTRAL_EPSILONS:
                try:
                    sparse_edges = run_spectral_sparsify(edges, n_nodes, eps)
                    edge_pct = len(sparse_edges) / len(edges) * 100

                    stats = compute_preservation_ratio(intra_edges, inter_edges, sparse_edges)

                    results_spectral.append({
                        'Dataset': dataset_name,
                        'Epsilon': eps,
                        'Edge %': f"{edge_pct:.1f}",
                        'Intra Orig': stats['intra_orig'],
                        'Intra Kept': stats['intra_kept'],
                        'Intra %': f"{stats['intra_pct']:.2f}",
                        'Inter Orig': stats['inter_orig'],
                        'Inter Kept': stats['inter_kept'],
                        'Inter %': f"{stats['inter_pct']:.2f}",
                        'Ratio': f"{stats['ratio']:.4f}"
                    })
                    spectral_ratios.append(stats['ratio'])
                    print(f"    ε={eps}: Ratio={stats['ratio']:.4f}")
                except Exception as e:
                    print(f"    ε={eps}: Error - {e}")

            # E5.2: DSpar sparsification
            print("  Running DSpar sparsification...")
            for keep_ratio in DSPAR_KEEP_RATIOS:
                sparse_edges = dspar_sparsify(edges, n_nodes, keep_ratio, RANDOM_SEED)
                actual_pct = len(sparse_edges) / len(edges) * 100

                stats = compute_preservation_ratio(intra_edges, inter_edges, sparse_edges)

                results_dspar.append({
                    'Dataset': dataset_name,
                    'Keep %': f"{keep_ratio * 100:.0f}%",
                    'Actual Edge %': f"{actual_pct:.1f}",
                    'Intra Kept': stats['intra_kept'],
                    'Intra %': f"{stats['intra_pct']:.2f}",
                    'Inter Kept': stats['inter_kept'],
                    'Inter %': f"{stats['inter_pct']:.2f}",
                    'Ratio': f"{stats['ratio']:.4f}"
                })
                dspar_ratios.append(stats['ratio'])
                print(f"    Keep={keep_ratio*100:.0f}%: Ratio={stats['ratio']:.4f}")

            # E5.3: Random sparsification
            print("  Running random sparsification...")
            for keep_ratio in RANDOM_KEEP_RATIOS:
                sparse_edges = random_sparsify(edges, keep_ratio, RANDOM_SEED)

                stats = compute_preservation_ratio(intra_edges, inter_edges, sparse_edges)

                results_random.append({
                    'Dataset': dataset_name,
                    'Keep %': f"{keep_ratio * 100:.0f}%",
                    'Intra %': f"{stats['intra_pct']:.2f}",
                    'Inter %': f"{stats['inter_pct']:.2f}",
                    'Ratio': f"{stats['ratio']:.4f}"
                })
                random_ratios.append(stats['ratio'])
                print(f"    Random {keep_ratio*100:.0f}%: Ratio={stats['ratio']:.4f}")

            # Summary
            def get_pattern(avg_ratio):
                if avg_ratio < 0.95:
                    return "Inter-removed"
                elif avg_ratio > 1.05:
                    return "Intra-removed"
                else:
                    return "Neutral"

            avg_spectral = np.mean(spectral_ratios) if spectral_ratios else 1.0
            avg_dspar = np.mean(dspar_ratios) if dspar_ratios else 1.0
            avg_random = np.mean(random_ratios) if random_ratios else 1.0

            results_summary.append({
                'Dataset': dataset_name,
                'Spectral Avg Ratio': f"{avg_spectral:.4f}",
                'DSpar Avg Ratio': f"{avg_dspar:.4f}",
                'Random Avg Ratio': f"{avg_random:.4f}",
                'Pattern': get_pattern(min(avg_spectral, avg_dspar))
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    if results_spectral:
        df = pd.DataFrame(results_spectral)
        df.to_csv(E5_RESULTS_DIR / "E5_1_spectral_ratio.csv", index=False)
        print(f"\nSaved: E5_1_spectral_ratio.csv")

    if results_dspar:
        df = pd.DataFrame(results_dspar)
        df.to_csv(E5_RESULTS_DIR / "E5_2_dspar_ratio.csv", index=False)
        print(f"Saved: E5_2_dspar_ratio.csv")

    if results_random:
        df = pd.DataFrame(results_random)
        df.to_csv(E5_RESULTS_DIR / "E5_3_random_ratio.csv", index=False)
        print(f"Saved: E5_3_random_ratio.csv")

    if results_summary:
        df = pd.DataFrame(results_summary)
        df.to_csv(E5_RESULTS_DIR / "E5_4_ratio_summary.csv", index=False)
        print(f"Saved: E5_4_ratio_summary.csv")
        print("\nE5.4 Summary:")
        print(df.to_string(index=False))


# =============================================================================
# E6: Hub-Community Correlation
# =============================================================================

def compute_dspar_scores(edges, n_nodes):
    """Compute DSpar score for each edge."""
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    scores = []
    for u, v in edges:
        score = 1.0 / degree[u] + 1.0 / degree[v]
        scores.append(score)

    return scores, degree


def run_e6_experiment():
    """E6: Hub-Community Correlation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT E6: Hub-Community Correlation")
    print("=" * 70)

    E6_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_dspar_dist = []
    results_degree_product = []
    results_hub_class = []
    results_edge_type_hub = []
    results_hub_periph_ratio = []
    results_bridge_analysis = []

    for dataset_name in E1_DATASETS:
        print(f"\nProcessing {dataset_name}...")

        try:
            edges, n_nodes, communities = load_e1_data(dataset_name)
            intra_edges, inter_edges = classify_edges(edges, communities)

            # Compute DSpar scores and degrees
            dspar_scores, degree = compute_dspar_scores(edges, n_nodes)
            edge_to_score = {edges[i]: dspar_scores[i] for i in range(len(edges))}

            # E6.1: DSpar Score Distribution by Edge Type
            intra_scores = [edge_to_score.get(e, edge_to_score.get((e[1], e[0]), 0)) for e in intra_edges]
            inter_scores = [edge_to_score.get(e, edge_to_score.get((e[1], e[0]), 0)) for e in inter_edges]

            intra_mean = np.mean(intra_scores) if intra_scores else 0
            intra_std = np.std(intra_scores) if intra_scores else 0
            inter_mean = np.mean(inter_scores) if inter_scores else 0
            inter_std = np.std(inter_scores) if inter_scores else 0

            diff_pct = ((intra_mean - inter_mean) / inter_mean * 100) if inter_mean > 0 else 0

            # Cohen's d
            if intra_scores and inter_scores:
                pooled_std = np.sqrt(((len(intra_scores) - 1) * intra_std**2 + (len(inter_scores) - 1) * inter_std**2) /
                                     (len(intra_scores) + len(inter_scores) - 2))
                cohens_d = (intra_mean - inter_mean) / pooled_std if pooled_std > 0 else 0
            else:
                cohens_d = 0

            # Mann-Whitney U test
            if intra_scores and inter_scores and len(intra_scores) > 1 and len(inter_scores) > 1:
                try:
                    stat, p_value = stats.mannwhitneyu(intra_scores, inter_scores, alternative='two-sided')
                except:
                    p_value = 1.0
            else:
                p_value = 1.0

            results_dspar_dist.append({
                'Dataset': dataset_name,
                'Intra Count': len(intra_edges),
                'Intra Mean': f"{intra_mean:.6f}",
                'Intra Std': f"{intra_std:.6f}",
                'Inter Count': len(inter_edges),
                'Inter Mean': f"{inter_mean:.6f}",
                'Inter Std': f"{inter_std:.6f}",
                'Diff %': f"{diff_pct:.2f}",
                "Cohen's d": f"{cohens_d:.4f}",
                'p-value': f"{p_value:.2e}"
            })
            print(f"  DSpar scores: Intra mean={intra_mean:.6f}, Inter mean={inter_mean:.6f}, Cohen's d={cohens_d:.4f}")

            # E6.2: Degree Product by Edge Type
            intra_deg_prod = [degree[u] * degree[v] for u, v in intra_edges]
            inter_deg_prod = [degree[u] * degree[v] for u, v in inter_edges]

            intra_dp_mean = np.mean(intra_deg_prod) if intra_deg_prod else 0
            inter_dp_mean = np.mean(inter_deg_prod) if inter_deg_prod else 0
            dp_ratio = inter_dp_mean / intra_dp_mean if intra_dp_mean > 0 else 0

            results_degree_product.append({
                'Dataset': dataset_name,
                'Intra Mean (d_u × d_v)': f"{intra_dp_mean:.2f}",
                'Inter Mean (d_u × d_v)': f"{inter_dp_mean:.2f}",
                'Inter/Intra Ratio': f"{dp_ratio:.4f}"
            })
            print(f"  Degree product: Intra={intra_dp_mean:.2f}, Inter={inter_dp_mean:.2f}, Ratio={dp_ratio:.4f}")

            # E6.3: Hub Classification
            degrees = [degree[i] for i in range(n_nodes) if degree[i] > 0]
            if degrees:
                hub_threshold = np.percentile(degrees, HUB_PERCENTILE)
            else:
                hub_threshold = 0

            hubs = set(node for node in range(n_nodes) if degree[node] >= hub_threshold)
            periphery = set(node for node in range(n_nodes) if degree[node] > 0 and node not in hubs)

            results_hub_class.append({
                'Dataset': dataset_name,
                'Total Nodes': len(hubs) + len(periphery),
                'Hub Threshold (degree)': int(hub_threshold),
                'Num Hubs': len(hubs),
                'Num Periphery': len(periphery)
            })
            print(f"  Hubs: {len(hubs)} nodes (degree >= {hub_threshold})")

            # E6.4: Edge Type by Hub/Periphery Classification
            edge_types = {'Hub-Hub': [], 'Hub-Periphery': [], 'Periphery-Periphery': []}

            for u, v in edges:
                u_hub = u in hubs
                v_hub = v in hubs

                if u_hub and v_hub:
                    edge_types['Hub-Hub'].append((u, v))
                elif u_hub or v_hub:
                    edge_types['Hub-Periphery'].append((u, v))
                else:
                    edge_types['Periphery-Periphery'].append((u, v))

            intra_set = set(intra_edges)

            for edge_type_name, type_edges in edge_types.items():
                intra_count = sum(1 for e in type_edges if e in intra_set or (e[1], e[0]) in intra_set)
                inter_count = len(type_edges) - intra_count
                inter_pct = (inter_count / len(type_edges) * 100) if type_edges else 0

                results_edge_type_hub.append({
                    'Dataset': dataset_name,
                    'Edge Type': edge_type_name,
                    'Total Edges': len(type_edges),
                    'Intra-Comm': intra_count,
                    'Inter-Comm': inter_count,
                    'Inter %': f"{inter_pct:.2f}"
                })

            # E6.5: Hub/Periphery Ratio Summary
            hub_hub_inter_pct = 0
            periph_periph_inter_pct = 0

            for edge_type_name, type_edges in edge_types.items():
                intra_count = sum(1 for e in type_edges if e in intra_set or (e[1], e[0]) in intra_set)
                inter_count = len(type_edges) - intra_count
                inter_pct = (inter_count / len(type_edges) * 100) if type_edges else 0

                if edge_type_name == 'Hub-Hub':
                    hub_hub_inter_pct = inter_pct
                elif edge_type_name == 'Periphery-Periphery':
                    periph_periph_inter_pct = inter_pct

            hub_periph_ratio = hub_hub_inter_pct / periph_periph_inter_pct if periph_periph_inter_pct > 0 else 0

            results_hub_periph_ratio.append({
                'Dataset': dataset_name,
                'Hub-Hub Inter %': f"{hub_hub_inter_pct:.2f}",
                'Periph-Periph Inter %': f"{periph_periph_inter_pct:.2f}",
                'Hub/Periph Ratio': f"{hub_periph_ratio:.4f}"
            })
            print(f"  Hub/Periph Ratio: {hub_periph_ratio:.4f}")

            # E6.6: Bridge Node Analysis
            inter_set = set(inter_edges)
            bridge_nodes = set()
            internal_nodes = set()

            node_edges = defaultdict(list)
            for u, v in edges:
                node_edges[u].append((u, v))
                node_edges[v].append((u, v))

            for node in range(n_nodes):
                if degree[node] == 0:
                    continue

                has_inter = False
                for e in node_edges[node]:
                    if e in inter_set or (e[1], e[0]) in inter_set:
                        has_inter = True
                        break

                if has_inter:
                    bridge_nodes.add(node)
                else:
                    internal_nodes.add(node)

            avg_degree_bridge = np.mean([degree[n] for n in bridge_nodes]) if bridge_nodes else 0
            avg_degree_internal = np.mean([degree[n] for n in internal_nodes]) if internal_nodes else 0
            bridge_internal_ratio = avg_degree_bridge / avg_degree_internal if avg_degree_internal > 0 else 0

            total_nodes = len(bridge_nodes) + len(internal_nodes)
            bridge_pct = len(bridge_nodes) / total_nodes * 100 if total_nodes > 0 else 0

            results_bridge_analysis.append({
                'Dataset': dataset_name,
                'Total Nodes': total_nodes,
                'Bridge Nodes': len(bridge_nodes),
                'Bridge %': f"{bridge_pct:.2f}",
                'Avg Degree (Bridge)': f"{avg_degree_bridge:.2f}",
                'Avg Degree (Internal)': f"{avg_degree_internal:.2f}",
                'Bridge/Internal Ratio': f"{bridge_internal_ratio:.4f}"
            })
            print(f"  Bridge nodes: {len(bridge_nodes)} ({bridge_pct:.1f}%), degree ratio={bridge_internal_ratio:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    tables = [
        (results_dspar_dist, "E6_1_dspar_score_distribution.csv"),
        (results_degree_product, "E6_2_degree_product.csv"),
        (results_hub_class, "E6_3_hub_classification.csv"),
        (results_edge_type_hub, "E6_4_edge_type_by_hub.csv"),
        (results_hub_periph_ratio, "E6_5_hub_periph_ratio.csv"),
        (results_bridge_analysis, "E6_6_bridge_node_analysis.csv")
    ]

    for results, filename in tables:
        if results:
            df = pd.DataFrame(results)
            df.to_csv(E6_RESULTS_DIR / filename, index=False)
            print(f"\nSaved: {filename}")


# =============================================================================
# E7: Effective Resistance Analysis
# =============================================================================

def compute_effective_resistance_sample(edges, n_nodes, sample_size=None):
    """
    Compute effective resistance for edges.
    Uses pseudoinverse of Laplacian: ER(u,v) = L^+(u,u) + L^+(v,v) - 2*L^+(u,v)

    For large graphs, samples edges.
    """
    if not edges:
        return [], []

    # Build Laplacian
    L = lil_matrix((n_nodes, n_nodes), dtype=float)

    for u, v in edges:
        L[u, u] += 1
        L[v, v] += 1
        L[u, v] -= 1
        L[v, u] -= 1

    L = L.tocsr()

    # Sample edges if needed
    if sample_size and len(edges) > sample_size:
        np.random.seed(RANDOM_SEED)
        sample_indices = np.random.choice(len(edges), size=sample_size, replace=False)
        sampled_edges = [edges[i] for i in sample_indices]
    else:
        sampled_edges = edges

    # Compute pseudoinverse via solving linear systems
    # For ER(u,v), we need L^+ e_u and L^+ e_v where e_u is unit vector
    # ER(u,v) = (e_u - e_v)^T L^+ (e_u - e_v)

    # This is expensive, so we use an approximation:
    # Solve Lx = b for random probing vectors, then estimate ER

    # For practical purposes, we'll use NetworkX which has an efficient implementation
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges)

    er_values = []

    try:
        # Compute ER using NetworkX's resistance_distance
        for u, v in sampled_edges:
            try:
                er = nx.resistance_distance(G, u, v)
                er_values.append(er)
            except:
                er_values.append(np.nan)
    except Exception as e:
        print(f"    ER computation error: {e}")
        # Fall back to simple approximation: ER ≈ 1/min(d_u, d_v) for connected nodes
        degree = dict(G.degree())
        for u, v in sampled_edges:
            if degree.get(u, 0) > 0 and degree.get(v, 0) > 0:
                er_values.append(1.0 / min(degree[u], degree[v]))
            else:
                er_values.append(np.nan)

    return sampled_edges, er_values


def run_e7_experiment():
    """E7: Effective Resistance Analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT E7: Effective Resistance Analysis")
    print("=" * 70)

    E7_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_er_by_type = []
    results_correlation = []
    results_percentiles = []

    for dataset_name in E1_DATASETS:
        print(f"\nProcessing {dataset_name}...")

        try:
            edges, n_nodes, communities = load_e1_data(dataset_name)
            intra_edges, inter_edges = classify_edges(edges, communities)

            # Determine sample size
            if len(edges) > ER_SAMPLE_SIZE:
                sample_size = ER_SAMPLE_SIZE
                sample_label = str(ER_SAMPLE_SIZE)
            else:
                sample_size = None
                sample_label = "all"

            print(f"  Computing ER for {sample_label} edges...")

            # Compute ER
            sampled_edges, er_values = compute_effective_resistance_sample(edges, n_nodes, sample_size)

            if not er_values or all(np.isnan(v) for v in er_values):
                print(f"  Skipping - ER computation failed")
                continue

            # Create edge to ER mapping
            edge_to_er = {}
            for i, e in enumerate(sampled_edges):
                if not np.isnan(er_values[i]):
                    edge_to_er[e] = er_values[i]
                    edge_to_er[(e[1], e[0])] = er_values[i]

            # Compute DSpar scores
            dspar_scores, degree = compute_dspar_scores(edges, n_nodes)
            edge_to_dspar = {edges[i]: dspar_scores[i] for i in range(len(edges))}
            for i, e in enumerate(edges):
                edge_to_dspar[(e[1], e[0])] = dspar_scores[i]

            # E7.1: ER by Edge Type
            intra_er = [edge_to_er.get(e, edge_to_er.get((e[1], e[0]))) for e in intra_edges
                        if e in edge_to_er or (e[1], e[0]) in edge_to_er]
            inter_er = [edge_to_er.get(e, edge_to_er.get((e[1], e[0]))) for e in inter_edges
                        if e in edge_to_er or (e[1], e[0]) in edge_to_er]

            intra_er = [v for v in intra_er if v is not None and not np.isnan(v)]
            inter_er = [v for v in inter_er if v is not None and not np.isnan(v)]

            intra_mean = np.mean(intra_er) if intra_er else 0
            intra_std = np.std(intra_er) if intra_er else 0
            inter_mean = np.mean(inter_er) if inter_er else 0
            inter_std = np.std(inter_er) if inter_er else 0
            er_ratio = inter_mean / intra_mean if intra_mean > 0 else 0

            results_er_by_type.append({
                'Dataset': dataset_name,
                'Sample Size': sample_label,
                'Intra ER Mean': f"{intra_mean:.6f}",
                'Intra ER Std': f"{intra_std:.6f}",
                'Inter ER Mean': f"{inter_mean:.6f}",
                'Inter ER Std': f"{inter_std:.6f}",
                'Inter/Intra Ratio': f"{er_ratio:.4f}"
            })
            print(f"  ER: Intra mean={intra_mean:.6f}, Inter mean={inter_mean:.6f}, Ratio={er_ratio:.4f}")

            # E7.2: ER vs DSpar Correlation
            er_list = []
            dspar_list = []
            for e in sampled_edges:
                if e in edge_to_er and e in edge_to_dspar:
                    er_val = edge_to_er[e]
                    dspar_val = edge_to_dspar[e]
                    if not np.isnan(er_val):
                        er_list.append(er_val)
                        dspar_list.append(dspar_val)

            if len(er_list) > 2:
                pearson_r, pearson_p = stats.pearsonr(er_list, dspar_list)
                spearman_r, spearman_p = stats.spearmanr(er_list, dspar_list)
            else:
                pearson_r, pearson_p = 0, 1
                spearman_r, spearman_p = 0, 1

            results_correlation.append({
                'Dataset': dataset_name,
                'Sample Size': len(er_list),
                'Pearson r': f"{pearson_r:.4f}",
                'Spearman ρ': f"{spearman_r:.4f}",
                'p-value': f"{min(pearson_p, spearman_p):.2e}"
            })
            print(f"  ER-DSpar correlation: Pearson r={pearson_r:.4f}, Spearman ρ={spearman_r:.4f}")

            # E7.3: ER Percentiles
            for edge_type, er_vals in [('Intra', intra_er), ('Inter', inter_er)]:
                if er_vals:
                    results_percentiles.append({
                        'Dataset': dataset_name,
                        'Type': edge_type,
                        '10th %ile': f"{np.percentile(er_vals, 10):.6f}",
                        '25th %ile': f"{np.percentile(er_vals, 25):.6f}",
                        'Median': f"{np.percentile(er_vals, 50):.6f}",
                        '75th %ile': f"{np.percentile(er_vals, 75):.6f}",
                        '90th %ile': f"{np.percentile(er_vals, 90):.6f}"
                    })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    tables = [
        (results_er_by_type, "E7_1_er_by_edge_type.csv"),
        (results_correlation, "E7_2_er_dspar_correlation.csv"),
        (results_percentiles, "E7_3_er_percentiles.csv")
    ]

    for results, filename in tables:
        if results:
            df = pd.DataFrame(results)
            df.to_csv(E7_RESULTS_DIR / filename, index=False)
            print(f"\nSaved: {filename}")


# =============================================================================
# E8: Min-Cut Edge Preservation
# =============================================================================

def compute_community_mincut(G, community_nodes):
    """Compute min-cut for a community subgraph using Stoer-Wagner algorithm."""
    subgraph = G.subgraph(community_nodes).copy()

    if subgraph.number_of_nodes() < 2:
        return 0, []

    # Check if connected
    if not nx.is_connected(subgraph):
        return 0, []

    # Use Stoer-Wagner algorithm for global min-cut
    try:
        min_cut_value, partition = nx.stoer_wagner(subgraph)

        # Find the actual cut edges (edges crossing the partition)
        set_a, set_b = partition
        min_cut_edges = []
        for u in set_a:
            for v in subgraph.neighbors(u):
                if v in set_b:
                    if u < v:
                        min_cut_edges.append((u, v))
                    else:
                        min_cut_edges.append((v, u))
    except:
        min_cut_value = 0
        min_cut_edges = []

    return min_cut_value, min_cut_edges


def run_e8_experiment():
    """E8: Min-Cut Edge Preservation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT E8: Min-Cut Edge Preservation")
    print("=" * 70)

    E8_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_comm_stats = []
    results_mincut_er = []
    results_spectral_preservation = []
    results_dspar_preservation = []
    results_wcc_spectral = []
    results_wcc_dspar = []

    for dataset_name in E1_DATASETS:
        print(f"\nProcessing {dataset_name}...")

        try:
            edges, n_nodes, communities = load_e1_data(dataset_name)

            # Build graph
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))
            G.add_edges_from(edges)

            # Group nodes by community
            comm_to_nodes = defaultdict(list)
            for node, comm in communities.items():
                comm_to_nodes[comm].append(node)

            # Filter communities with >= MIN_COMMUNITY_SIZE nodes
            valid_communities = {c: nodes for c, nodes in comm_to_nodes.items()
                                 if len(nodes) >= MIN_COMMUNITY_SIZE}

            # Limit number of communities for large graphs
            if len(valid_communities) > MAX_COMMUNITIES_MINCUT:
                # Sort by size and take largest
                sorted_comms = sorted(valid_communities.items(), key=lambda x: len(x[1]), reverse=True)
                valid_communities = dict(sorted_comms[:MAX_COMMUNITIES_MINCUT])

            print(f"  Processing {len(valid_communities)} communities (>= {MIN_COMMUNITY_SIZE} nodes)")

            if not valid_communities:
                print("  No valid communities found")
                continue

            # E8.1: Community Statistics
            comm_sizes = [len(nodes) for nodes in valid_communities.values()]
            comm_edges = []
            comm_mincuts = []
            all_mincut_edges = set()
            all_other_edges = set()
            community_wcc_status = {}  # community -> WCC status

            for comm_id, comm_nodes in valid_communities.items():
                comm_node_set = set(comm_nodes)
                subgraph = G.subgraph(comm_nodes)
                n_edges = subgraph.number_of_edges()
                comm_edges.append(n_edges)

                # Compute min-cut
                mincut_val, mincut_edges = compute_community_mincut(G, comm_nodes)
                comm_mincuts.append(mincut_val)

                # WCC: min-cut > log(n)
                wcc_threshold = np.log(len(comm_nodes))
                community_wcc_status[comm_id] = mincut_val > wcc_threshold

                # Collect min-cut edges
                for e in mincut_edges:
                    if e[0] < e[1]:
                        all_mincut_edges.add(e)
                    else:
                        all_mincut_edges.add((e[1], e[0]))

                # Other edges in community
                for e in subgraph.edges():
                    if e[0] < e[1]:
                        edge = e
                    else:
                        edge = (e[1], e[0])
                    if edge not in all_mincut_edges:
                        all_other_edges.add(edge)

            results_comm_stats.append({
                'Dataset': dataset_name,
                'Communities (≥10 nodes)': len(valid_communities),
                'Avg Community Size': f"{np.mean(comm_sizes):.1f}",
                'Avg Community Edges': f"{np.mean(comm_edges):.1f}",
                'Avg Min-Cut Size': f"{np.mean(comm_mincuts):.2f}"
            })
            print(f"  Min-cuts per community: {comm_mincuts}")
            print(f"  Avg min-cut: {np.mean(comm_mincuts):.2f}")

            # E8.2: Min-Cut Edge ER Analysis
            if all_mincut_edges:
                print(f"  Computing ER for {len(all_mincut_edges)} min-cut edges...")

                # Compute ER for min-cut edges
                mc_er = []
                other_er = []

                for e in all_mincut_edges:
                    try:
                        er = nx.resistance_distance(G, e[0], e[1])
                        mc_er.append(er)
                    except:
                        pass

                # Sample other edges for comparison
                other_sample = list(all_other_edges)[:min(len(all_other_edges), len(all_mincut_edges) * 10)]
                for e in other_sample:
                    try:
                        er = nx.resistance_distance(G, e[0], e[1])
                        other_er.append(er)
                    except:
                        pass

                mc_mean = np.mean(mc_er) if mc_er else 0
                other_mean = np.mean(other_er) if other_er else 0
                mc_other_ratio = mc_mean / other_mean if other_mean > 0 else 0

                # Statistical test
                if mc_er and other_er:
                    try:
                        _, p_val = stats.mannwhitneyu(mc_er, other_er, alternative='two-sided')
                    except:
                        p_val = 1.0
                else:
                    p_val = 1.0

                results_mincut_er.append({
                    'Dataset': dataset_name,
                    'Communities': len(valid_communities),
                    'Min-Cut ER (avg)': f"{mc_mean:.6f}",
                    'Other ER (avg)': f"{other_mean:.6f}",
                    'MC/Other Ratio': f"{mc_other_ratio:.4f}",
                    'p-value': f"{p_val:.2e}"
                })
                print(f"  Min-Cut ER: {mc_mean:.6f}, Other ER: {other_mean:.6f}, Ratio: {mc_other_ratio:.4f}")

            # E8.3 & E8.4: Min-Cut Edge Preservation
            # Only test a subset of parameters to save time
            test_epsilons = [1.0, 2.0]
            test_keep_ratios = [0.75, 0.90]  # Higher ratios to avoid disconnection

            # Spectral preservation
            for eps in test_epsilons:
                try:
                    sparse_edges = run_spectral_sparsify(edges, n_nodes, eps)
                    sparse_set = set(sparse_edges)

                    mc_kept = sum(1 for e in all_mincut_edges if e in sparse_set)
                    other_kept = sum(1 for e in all_other_edges if e in sparse_set)

                    mc_keep_pct = mc_kept / len(all_mincut_edges) * 100 if all_mincut_edges else 0
                    other_keep_pct = other_kept / len(all_other_edges) * 100 if all_other_edges else 0
                    mc_other_pres = mc_keep_pct / other_keep_pct if other_keep_pct > 0 else 0

                    results_spectral_preservation.append({
                        'Dataset': dataset_name,
                        'ε': eps,
                        'Total MC Edges': len(all_mincut_edges),
                        'MC Kept': mc_kept,
                        'MC Keep %': f"{mc_keep_pct:.2f}",
                        'Other Keep %': f"{other_keep_pct:.2f}",
                        'MC/Other Preservation': f"{mc_other_pres:.4f}"
                    })

                    # WCC preservation
                    sparse_G = nx.Graph()
                    sparse_G.add_edges_from(sparse_edges)

                    wcc_agree = 0
                    wcc_lost = 0
                    wcc_gained = 0
                    orig_wcc = sum(community_wcc_status.values())

                    for comm_id, comm_nodes in valid_communities.items():
                        orig_wcc_status = community_wcc_status[comm_id]

                        # Recompute min-cut on sparse graph
                        sparse_mincut, _ = compute_community_mincut(sparse_G, comm_nodes)
                        sparse_wcc_status = sparse_mincut > np.log(len(comm_nodes))

                        if orig_wcc_status == sparse_wcc_status:
                            wcc_agree += 1
                        elif orig_wcc_status and not sparse_wcc_status:
                            wcc_lost += 1
                        else:
                            wcc_gained += 1

                    sparse_wcc = sum(1 for c in valid_communities if community_wcc_status.get(c, False))

                    results_wcc_spectral.append({
                        'Dataset': dataset_name,
                        'ε': eps,
                        'Communities': len(valid_communities),
                        'Orig WCC': orig_wcc,
                        'Sparse WCC': sparse_wcc,
                        'WCC Agree': wcc_agree,
                        'WCC Lost': wcc_lost,
                        'WCC Gained': wcc_gained
                    })

                except Exception as e:
                    print(f"    Spectral ε={eps}: Error - {e}")

            # DSpar preservation
            for keep_ratio in test_keep_ratios:
                sparse_edges = dspar_sparsify(edges, n_nodes, keep_ratio, RANDOM_SEED)
                sparse_set = set(sparse_edges)

                mc_kept = sum(1 for e in all_mincut_edges if e in sparse_set)
                other_kept = sum(1 for e in all_other_edges if e in sparse_set)

                mc_keep_pct = mc_kept / len(all_mincut_edges) * 100 if all_mincut_edges else 0
                other_keep_pct = other_kept / len(all_other_edges) * 100 if all_other_edges else 0
                mc_other_pres = mc_keep_pct / other_keep_pct if other_keep_pct > 0 else 0

                results_dspar_preservation.append({
                    'Dataset': dataset_name,
                    'Keep %': f"{keep_ratio * 100:.0f}%",
                    'Total MC Edges': len(all_mincut_edges),
                    'MC Kept': mc_kept,
                    'MC Keep %': f"{mc_keep_pct:.2f}",
                    'Other Keep %': f"{other_keep_pct:.2f}",
                    'MC/Other Preservation': f"{mc_other_pres:.4f}"
                })

                # WCC preservation for DSpar
                sparse_G = nx.Graph()
                sparse_G.add_edges_from(sparse_edges)

                wcc_agree = 0
                wcc_lost = 0
                orig_wcc = sum(community_wcc_status.values())

                for comm_id, comm_nodes in valid_communities.items():
                    orig_wcc_status = community_wcc_status[comm_id]
                    sparse_mincut, _ = compute_community_mincut(sparse_G, comm_nodes)
                    sparse_wcc_status = sparse_mincut > np.log(len(comm_nodes))

                    if orig_wcc_status == sparse_wcc_status:
                        wcc_agree += 1
                    elif orig_wcc_status and not sparse_wcc_status:
                        wcc_lost += 1

                sparse_wcc = sum(1 for c in valid_communities if community_wcc_status.get(c, False))

                results_wcc_dspar.append({
                    'Dataset': dataset_name,
                    'Keep %': f"{keep_ratio * 100:.0f}%",
                    'Communities': len(valid_communities),
                    'Orig WCC': orig_wcc,
                    'Sparse WCC': sparse_wcc,
                    'WCC Agree': wcc_agree,
                    'WCC Lost': wcc_lost
                })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    tables = [
        (results_comm_stats, "E8_1_community_stats.csv"),
        (results_mincut_er, "E8_2_mincut_er_analysis.csv"),
        (results_spectral_preservation, "E8_3_mincut_preservation_spectral.csv"),
        (results_dspar_preservation, "E8_4_mincut_preservation_dspar.csv"),
        (results_wcc_spectral, "E8_5_wcc_spectral.csv"),
        (results_wcc_dspar, "E8_6_wcc_dspar.csv")
    ]

    for results, filename in tables:
        if results:
            df = pd.DataFrame(results)
            df.to_csv(E8_RESULTS_DIR / filename, index=False)
            print(f"\nSaved: {filename}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Understanding Why Sparsification Works (E5-E8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python e5_e6_e7_e8_mechanism_analysis.py              # Run all experiments
  python e5_e6_e7_e8_mechanism_analysis.py -e E7        # Run only E7
  python e5_e6_e7_e8_mechanism_analysis.py -e E7 E8     # Run E7 and E8
  python e5_e6_e7_e8_mechanism_analysis.py -e E5,E6     # Run E5 and E6
        """
    )
    parser.add_argument(
        '-e', '--experiments',
        nargs='+',
        default=['E5', 'E6', 'E7', 'E8'],
        help='Experiments to run (E5, E6, E7, E8). Default: all'
    )

    args = parser.parse_args()

    # Parse experiments (handle both "E5 E6" and "E5,E6" formats)
    experiments = []
    for exp in args.experiments:
        experiments.extend([e.strip().upper() for e in exp.split(',')])

    valid_experiments = {'E5', 'E6', 'E7', 'E8'}
    experiments = [e for e in experiments if e in valid_experiments]

    if not experiments:
        print("No valid experiments specified. Use E5, E6, E7, or E8.")
        return

    print("=" * 70)
    print("PHASE 3: UNDERSTANDING WHY SPARSIFICATION WORKS")
    print(f"Running: {', '.join(experiments)}")
    print("=" * 70)

    # Create output directories
    for dir_path in [E5_RESULTS_DIR, E6_RESULTS_DIR, E7_RESULTS_DIR, E8_RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Run selected experiments
    if 'E5' in experiments:
        run_e5_experiment()
    if 'E6' in experiments:
        run_e6_experiment()
    if 'E7' in experiments:
        run_e7_experiment()
    if 'E8' in experiments:
        run_e8_experiment()

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print("\nResults saved to:")
    if 'E5' in experiments:
        print(f"  - {E5_RESULTS_DIR}")
    if 'E6' in experiments:
        print(f"  - {E6_RESULTS_DIR}")
    if 'E7' in experiments:
        print(f"  - {E7_RESULTS_DIR}")
    if 'E8' in experiments:
        print(f"  - {E8_RESULTS_DIR}")


if __name__ == "__main__":
    main()
