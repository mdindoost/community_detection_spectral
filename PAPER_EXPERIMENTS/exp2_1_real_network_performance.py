"""
Experiment 2.1: Real Network Performance (Main Results)

Purpose: Demonstrate DSpar effectiveness on diverse real networks
  - Multiple network types (citation, social, collaboration, communication)
  - Comprehensive baselines (random, spectral, degree-based)
  - Full metrics (Q, NMI, ARI, ratio, time)
  - Boundary cases (road networks - should fail)

This is the main experimental validation showing practical effectiveness.

Run: python exp2_1_real_network_performance.py
"""

import sys
import time
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("results/exp2_1_real_networks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RETENTION_VALUES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
N_RUNS = 10  # For stochastic methods


# Dataset configurations
DATASETS = {
    # Citation networks (should work)
    'cit-HepPh': {'path': 'datasets/cit-HepPh/cit-HepPh.txt', 'ground_truth': False},
    'cit-HepTh': {'path': 'datasets/cit-HepTh/cit-HepTh.txt', 'ground_truth': False},
    
    # Social networks (should work)
    'facebook': {'path': 'datasets/facebook/facebook_combined.txt', 'ground_truth': False},
    
    # Collaboration networks (should work)
    'ca-GrQc': {'path': 'datasets/ca-GrQc/ca-GrQc.txt', 'ground_truth': False},
    
    # Communication networks (should work)
    'email-Enron': {'path': 'datasets/email-Enron/email-Enron.txt', 'ground_truth': False},
    
    # Road networks (should FAIL - boundary condition)
    'road-CA': {'path': 'datasets/road-CA/roadNet-CA.txt', 'ground_truth': False},
}


def load_dataset(name):
    """Load dataset by name"""
    config = DATASETS[name]
    edge_file = Path(config['path'])
    
    if not edge_file.exists():
        raise FileNotFoundError(f"Dataset not found: {edge_file}")
    
    G = nx.Graph()
    with open(edge_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    G.add_edge(u, v)
    
    # Take largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    G = nx.convert_node_labels_to_integers(G)
    return G


def nx_to_igraph(G):
    """Convert NetworkX to igraph"""
    edges = list(G.edges())
    return ig.Graph(n=G.number_of_nodes(), edges=edges, directed=False)


def run_leiden(G):
    """Run Leiden clustering"""
    ig_graph = nx_to_igraph(G)
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    return partition.membership, partition.modularity


def dspar_sparsify(G, retention, seed):
    """DSpar sparsification"""
    np.random.seed(seed)
    degrees = dict(G.degree())
    
    edges = list(G.edges())
    scores = np.array([1.0/degrees[u] + 1.0/degrees[v] for u, v in edges])
    mean_score = scores.mean()
    
    probs = retention * scores / mean_score
    probs = np.minimum(probs, 1.0)
    
    keep = np.random.rand(len(edges)) < probs
    
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes())
    G_sparse.add_edges_from([edges[i] for i in range(len(edges)) if keep[i]])
    
    return G_sparse


def random_sparsify(G, retention, seed):
    """Baseline: Random edge sampling"""
    np.random.seed(seed)
    
    edges = list(G.edges())
    n_keep = int(len(edges) * retention)
    
    keep_indices = np.random.choice(len(edges), size=n_keep, replace=False)
    
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes())
    G_sparse.add_edges_from([edges[i] for i in keep_indices])
    
    return G_sparse


def degree_threshold_sparsify(G, retention):
    """
    Baseline: Remove edges incident to highest-degree nodes.
    Simple heuristic: remove hub edges deterministically.
    """
    degrees = dict(G.degree())
    
    # Assign score inversely proportional to max degree of endpoints
    edges = list(G.edges())
    scores = []
    for u, v in edges:
        max_deg = max(degrees[u], degrees[v])
        scores.append(1.0 / max_deg)  # Lower degree edges score higher
    
    # Keep top scoring edges
    scores = np.array(scores)
    n_keep = int(len(edges) * retention)
    keep_indices = np.argsort(-scores)[:n_keep]  # Descending order
    
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes())
    G_sparse.add_edges_from([edges[i] for i in keep_indices])
    
    return G_sparse


def compute_preservation_ratio(G, G_sparse, membership):
    """
    Calculate edge preservation ratio: inter_rate / intra_rate
    Ratio < 1 means preferential removal of inter-community edges
    """
    # Classify edges
    total_intra = sum(1 for u, v in G.edges() if membership[u] == membership[v])
    total_inter = G.number_of_edges() - total_intra
    
    # Count preserved
    sparse_edges = set((min(u,v), max(u,v)) for u,v in G_sparse.edges())
    preserved_intra = sum(1 for u,v in G.edges() 
                         if membership[u] == membership[v] and 
                         (min(u,v), max(u,v)) in sparse_edges)
    preserved_inter = sum(1 for u,v in G.edges() 
                         if membership[u] != membership[v] and 
                         (min(u,v), max(u,v)) in sparse_edges)
    
    intra_rate = preserved_intra / total_intra if total_intra > 0 else 1.0
    inter_rate = preserved_inter / total_inter if total_inter > 0 else 1.0
    ratio = inter_rate / intra_rate if intra_rate > 0 else float('inf')
    
    return {
        'ratio': ratio,
        'intra_rate': intra_rate,
        'inter_rate': inter_rate,
        'total_intra': total_intra,
        'total_inter': total_inter
    }


def compute_hub_bridge_correlation(G, membership):
    """Compute hub-bridge correlation"""
    degrees = dict(G.degree())
    
    inter_products = []
    intra_products = []
    
    for u, v in G.edges():
        product = degrees[u] * degrees[v]
        if membership[u] == membership[v]:
            intra_products.append(product)
        else:
            inter_products.append(product)
    
    if len(inter_products) == 0 or len(intra_products) == 0:
        return 0, 1.0
    
    mean_inter = np.mean(inter_products)
    mean_intra = np.mean(intra_products)
    
    return mean_inter - mean_intra, mean_inter / mean_intra


def compute_dspar_separation(G, membership):
    """Compute DSpar separation δ"""
    degrees = dict(G.degree())
    
    intra_scores = []
    inter_scores = []
    
    for u, v in G.edges():
        score = 1.0 / degrees[u] + 1.0 / degrees[v]
        if membership[u] == membership[v]:
            intra_scores.append(score)
        else:
            inter_scores.append(score)
    
    if len(inter_scores) == 0 or len(intra_scores) == 0:
        return 0, 0, 0
    
    mu_intra = np.mean(intra_scores)
    mu_inter = np.mean(inter_scores)
    
    return mu_intra - mu_inter, mu_intra, mu_inter


def run_single_experiment(method, G, membership_original, modularity_original, 
                          retention, seed):
    """
    Run single sparsification + community detection.
    
    Returns:
        Dict of results
    """
    start_sparse = time.time()
    
    # Sparsify
    if method == 'none':
        G_sparse = G.copy()
    elif method == 'dspar':
        G_sparse = dspar_sparsify(G, retention, seed)
    elif method == 'random':
        G_sparse = random_sparsify(G, retention, seed)
    elif method == 'degree_threshold':
        G_sparse = degree_threshold_sparsify(G, retention)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    time_sparse = time.time() - start_sparse
    
    # Community detection
    start_leiden = time.time()
    membership_sparse, modularity_sparse = run_leiden(G_sparse)
    time_leiden = time.time() - start_leiden
    
    # Metrics
    nmi = normalized_mutual_info_score(membership_original, membership_sparse)
    ari = adjusted_rand_score(membership_original, membership_sparse)
    
    # Preservation ratio
    ratio_stats = compute_preservation_ratio(G, G_sparse, membership_original)
    
    # Graph properties
    n_cc = nx.number_connected_components(G_sparse)
    
    return {
        'method': method,
        'retention': retention,
        'seed': seed,
        'n_edges_sparse': G_sparse.number_of_edges(),
        'edge_retention_actual': G_sparse.number_of_edges() / G.number_of_edges(),
        'n_cc': n_cc,
        'modularity': modularity_sparse,
        'modularity_change': modularity_sparse - modularity_original,
        'modularity_change_pct': 100 * (modularity_sparse - modularity_original) / modularity_original,
        'nmi': nmi,
        'ari': ari,
        'preservation_ratio': ratio_stats['ratio'],
        'intra_rate': ratio_stats['intra_rate'],
        'inter_rate': ratio_stats['inter_rate'],
        'time_sparse': time_sparse,
        'time_leiden': time_leiden,
        'time_total': time_sparse + time_leiden
    }


def main():
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else list(DATASETS.keys())[0]
    
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASETS.keys())}")
        return
    
    print("=" * 100)
    print(f"EXPERIMENT 2.1: REAL NETWORK PERFORMANCE - {dataset_name.upper()}")
    print("=" * 100)
    
    # Load dataset
    print(f"\nLoading {dataset_name}...")
    G = load_dataset(dataset_name)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    print(f"Graph: {n_nodes:,} nodes, {n_edges:,} edges")
    
    # Compute degree statistics
    degrees = [d for _, d in G.degree()]
    degree_gini = compute_gini(degrees)
    
    print(f"Degree statistics:")
    print(f"  Mean: {np.mean(degrees):.2f}")
    print(f"  Max: {max(degrees)}")
    print(f"  Gini coefficient: {degree_gini:.4f}")
    
    # Run Leiden on original
    print("\nRunning Leiden on original graph...")
    membership_original, modularity_original = run_leiden(G)
    n_communities = len(set(membership_original))
    
    print(f"Communities: {n_communities}, Modularity: {modularity_original:.4f}")
    
    # Compute structural properties
    print("\nComputing structural properties...")
    hub_bridge_corr, hub_bridge_ratio = compute_hub_bridge_correlation(G, membership_original)
    delta, mu_intra, mu_inter = compute_dspar_separation(G, membership_original)
    
    print(f"Hub-bridge correlation: {hub_bridge_corr:.1f} (ratio: {hub_bridge_ratio:.3f})")
    print(f"DSpar separation δ: {delta:.6f}")
    print(f"  μ_intra: {mu_intra:.6f}")
    print(f"  μ_inter: {mu_inter:.6f}")
    
    if delta > 0:
        print("✓ δ > 0: DSpar should improve modularity")
    else:
        print("✗ δ ≤ 0: DSpar may not improve (boundary condition)")
    
    # Run experiments
    print("\n" + "=" * 100)
    print("RUNNING EXPERIMENTS")
    print("=" * 100)
    
    methods = ['dspar', 'random', 'degree_threshold']
    results = []
    
    # Add baseline (no sparsification)
    baseline = {
        'method': 'none',
        'retention': 1.0,
        'seed': 0,
        'n_edges_sparse': n_edges,
        'edge_retention_actual': 1.0,
        'n_cc': 1,
        'modularity': modularity_original,
        'modularity_change': 0.0,
        'modularity_change_pct': 0.0,
        'nmi': 1.0,
        'ari': 1.0,
        'preservation_ratio': 1.0,
        'intra_rate': 1.0,
        'inter_rate': 1.0,
        'time_sparse': 0.0,
        'time_leiden': 0.0,
        'time_total': 0.0
    }
    results.append(baseline)
    
    total_exps = len(methods) * len([r for r in RETENTION_VALUES if r < 1.0]) * N_RUNS
    exp_count = 0
    
    for method in methods:
        for retention in RETENTION_VALUES:
            if retention == 1.0:
                continue  # Skip (already have baseline)
            
            for run in range(N_RUNS):
                seed = int(retention * 10000) + run
                
                print(f"\r{method}, r={retention:.2f}, run={run+1}/{N_RUNS} "
                      f"({exp_count+1}/{total_exps})", end='', flush=True)
                
                result = run_single_experiment(
                    method, G, membership_original, modularity_original,
                    retention, seed
                )
                results.append(result)
                exp_count += 1
    
    print(f"\n\nCompleted {len(results)} experiments")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save raw results
    output_file = OUTPUT_DIR / f"{dataset_name}_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved raw results to: {output_file}")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "=" * 100)
    print("RESULTS ANALYSIS")
    print("=" * 100)
    
    # Aggregate by method and retention
    agg = df.groupby(['method', 'retention']).agg({
        'modularity_change_pct': ['mean', 'std'],
        'nmi': ['mean', 'std'],
        'preservation_ratio': ['mean', 'std'],
        'time_total': 'mean'
    }).reset_index()
    
    # Display results table
    print("\nMODULARITY CHANGE (%):")
    print("-" * 100)
    print(f"{'Method':<20} {'Retention':<12} {'ΔQ % (mean)':<15} {'ΔQ % (std)':<15}")
    print("-" * 100)
    
    for _, row in agg.iterrows():
        method = row['method']
        retention = row['retention']
        mod_mean = row[('modularity_change_pct', 'mean')]
        mod_std = row[('modularity_change_pct', 'std')]
        
        indicator = '✓' if mod_mean > 0 else ' '
        print(f"{method:<20} {retention:<12.2f} {mod_mean:<+15.2f} {mod_std:<15.2f} {indicator}")
    
    print("-" * 100)
    
    # Find best configurations
    print("\nBEST CONFIGURATIONS:")
    
    # Best modularity improvement
    best_mod_idx = df['modularity_change_pct'].idxmax()
    best_mod = df.loc[best_mod_idx]
    print(f"\nHighest modularity improvement:")
    print(f"  Method: {best_mod['method']}, Retention: {best_mod['retention']:.2f}")
    print(f"  ΔQ: +{best_mod['modularity_change_pct']:.2f}%")
    print(f"  NMI: {best_mod['nmi']:.4f}, Ratio: {best_mod['preservation_ratio']:.3f}")
    
    # Best preservation ratio (lowest, meaning most preferential)
    best_ratio_idx = df[df['method'] == 'dspar']['preservation_ratio'].idxmin()
    best_ratio = df.loc[best_ratio_idx]
    print(f"\nLowest preservation ratio (most preferential removal):")
    print(f"  Method: {best_ratio['method']}, Retention: {best_ratio['retention']:.2f}")
    print(f"  Ratio: {best_ratio['preservation_ratio']:.3f}")
    print(f"  ΔQ: +{best_ratio['modularity_change_pct']:.2f}%")
    
    # Compare DSpar vs Random at retention=0.5
    print("\nCOMPARISON AT RETENTION = 0.5:")
    print("-" * 80)
    
    for method in ['dspar', 'random', 'degree_threshold']:
        subset = df[(df['method'] == method) & (df['retention'] == 0.5)]
        if len(subset) > 0:
            mod_mean = subset['modularity_change_pct'].mean()
            mod_std = subset['modularity_change_pct'].std()
            ratio_mean = subset['preservation_ratio'].mean()
            time_mean = subset['time_total'].mean()
            
            print(f"{method:<20}: ΔQ = {mod_mean:+.2f}% ± {mod_std:.2f}%, "
                  f"Ratio = {ratio_mean:.3f}, Time = {time_mean:.3f}s")
    
    print("-" * 80)
    
    # Speedup calculation
    baseline_leiden_time = baseline['time_leiden']  # Note: this is 0 in baseline, need to fix
    # Actually run Leiden once to get timing
    start = time.time()
    _, _ = run_leiden(G)
    baseline_leiden_time = time.time() - start
    
    # Estimate speedup for DSpar at 0.5 retention
    dspar_05 = df[(df['method'] == 'dspar') & (df['retention'] == 0.5)]
    if len(dspar_05) > 0:
        avg_total_time = dspar_05['time_total'].mean()
        speedup = baseline_leiden_time / avg_total_time if avg_total_time > 0 else 1.0
        
        print(f"\nSPEEDUP ESTIMATE (retention=0.5):")
        print(f"  Baseline (Leiden on full graph): {baseline_leiden_time:.3f}s")
        print(f"  DSpar + Leiden: {avg_total_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}×")
    
    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================
    summary = {
        'dataset': dataset_name,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'n_communities': n_communities,
        'degree_gini': degree_gini,
        'modularity_original': modularity_original,
        'hub_bridge_correlation': hub_bridge_corr,
        'hub_bridge_ratio': hub_bridge_ratio,
        'dspar_delta': delta,
        'mu_intra': mu_intra,
        'mu_inter': mu_inter,
        'best_method': best_mod['method'],
        'best_retention': best_mod['retention'],
        'best_mod_improvement': best_mod['modularity_change_pct']
    }
    
    summary_file = OUTPUT_DIR / f"{dataset_name}_summary.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary to: {summary_file}")
    
    print("\n" + "=" * 100)


def compute_gini(values):
    """Compute Gini coefficient for degree heterogeneity"""
    values = np.array(values)
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n


if __name__ == "__main__":
    main()
