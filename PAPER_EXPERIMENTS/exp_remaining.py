"""
EXPERIMENTS 2.2, 3.1, 4.1, 4.2: Remaining Critical Validation

This file contains:
  - Experiment 2.2: Weighting vs Sampling Comparison
  - Experiment 3.1: Scalability Analysis
  - Experiment 4.1: Property Correlation (Diagnostic Framework)
  - Experiment 4.2: Failure Cases (Boundary Conditions)

Run: python exp_remaining.py [experiment_number] [dataset]
Examples:
  python exp_remaining.py 2.2 cit-HepPh
  python exp_remaining.py 3.1
  python exp_remaining.py 4.1
  python exp_remaining.py 4.2
"""

import sys
import time
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# ============================================================================
# COMMON UTILITIES
# ============================================================================

def nx_to_igraph(G):
    """Convert NetworkX to igraph"""
    edges = list(G.edges())
    return ig.Graph(n=G.number_of_nodes(), edges=edges, directed=False)


def run_leiden(G, weighted=False, weight_attr=None):
    """
    Run Leiden clustering.
    
    Args:
        G: NetworkX graph
        weighted: If True, use edge weights
        weight_attr: Name of weight attribute
    """
    ig_graph = nx_to_igraph(G)
    
    if weighted and weight_attr:
        # Extract weights in same order as edges
        weights = [G[u][v].get(weight_attr, 1.0) for u, v in G.edges()]
        partition = leidenalg.find_partition(
            ig_graph, 
            leidenalg.ModularityVertexPartition,
            weights=weights
        )
    else:
        partition = leidenalg.find_partition(
            ig_graph, 
            leidenalg.ModularityVertexPartition
        )
    
    return partition.membership, partition.modularity


def compute_dspar_scores(G):
    """Compute DSpar scores for all edges"""
    degrees = dict(G.degree())
    scores = {}
    for u, v in G.edges():
        scores[(u, v)] = 1.0 / degrees[u] + 1.0 / degrees[v]
        scores[(v, u)] = scores[(u, v)]
    return scores, np.mean(list(scores.values()))


def load_dataset(name):
    """Load dataset"""
    datasets = {
        'cit-HepPh': 'datasets/cit-HepPh/cit-HepPh.txt',
        'cit-HepTh': 'datasets/cit-HepTh/cit-HepTh.txt',
        'facebook': 'datasets/facebook/facebook_combined.txt',
        'ca-GrQc': 'datasets/ca-GrQc/ca-GrQc.txt',
        'road-CA': 'datasets/road-CA/roadNet-CA.txt',
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")
    
    edge_file = Path(datasets[name])
    
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
    
    # Largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    G = nx.convert_node_labels_to_integers(G)
    return G


# ============================================================================
# EXPERIMENT 2.2: WEIGHTING VS SAMPLING
# ============================================================================

def exp_2_2_weighting_vs_sampling(dataset_name):
    """
    Compare DSpar weighting (keep all edges, re-weight) vs
    DSpar sampling (remove edges probabilistically).
    
    Theory (Theorem 3): Should converge asymptotically.
    """
    OUTPUT_DIR = Path("results/exp2_2_weighting_vs_sampling")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("EXPERIMENT 2.2: WEIGHTING VS SAMPLING")
    print("=" * 100)
    
    # Load dataset
    print(f"\nLoading {dataset_name}...")
    G = load_dataset(dataset_name)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    print(f"Graph: {n_nodes:,} nodes, {n_edges:,} edges")
    
    # Run baseline
    print("\nBaseline: Leiden on original graph...")
    membership_original, mod_original = run_leiden(G, weighted=False)
    print(f"Communities: {len(set(membership_original))}, Modularity: {mod_original:.4f}")
    
    # Compute DSpar scores
    print("\nComputing DSpar scores...")
    scores, mean_score = compute_dspar_scores(G)
    
    # Test different retention values
    RETENTION_VALUES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    N_RUNS = 10  # For sampling (stochastic)
    
    results = []
    
    print("\nRunning experiments...")
    
    for retention in RETENTION_VALUES:
        # ====================================================================
        # METHOD 1: WEIGHTING (deterministic)
        # ====================================================================
        print(f"\n  Weighting, retention={retention:.2f}")
        
        # Create weighted graph
        G_weighted = G.copy()
        for u, v in G_weighted.edges():
            weight = retention * scores[(u, v)] / mean_score
            weight = min(weight, 1.0)
            G_weighted[u][v]['weight'] = weight
        
        start = time.time()
        mem_weighted, mod_weighted = run_leiden(G_weighted, weighted=True, weight_attr='weight')
        time_weighted = time.time() - start
        
        nmi_weighted = normalized_mutual_info_score(membership_original, mem_weighted)
        
        results.append({
            'method': 'weighting',
            'retention': retention,
            'run': 0,  # Deterministic
            'modularity': mod_weighted,
            'modularity_change': mod_weighted - mod_original,
            'nmi': nmi_weighted,
            'time': time_weighted,
            'n_edges': n_edges  # All edges kept
        })
        
        # ====================================================================
        # METHOD 2: SAMPLING (stochastic)
        # ====================================================================
        print(f"  Sampling, retention={retention:.2f}, {N_RUNS} runs...")
        
        for run in range(N_RUNS):
            seed = int(retention * 10000) + run
            np.random.seed(seed)
            
            # Sample edges
            edges = list(G.edges())
            probs = np.array([retention * scores[e] / mean_score for e in edges])
            probs = np.minimum(probs, 1.0)
            
            keep = np.random.rand(len(edges)) < probs
            
            G_sampled = nx.Graph()
            G_sampled.add_nodes_from(G.nodes())
            G_sampled.add_edges_from([edges[i] for i in range(len(edges)) if keep[i]])
            
            start = time.time()
            mem_sampled, mod_sampled = run_leiden(G_sampled, weighted=False)
            time_sampled = time.time() - start
            
            nmi_sampled = normalized_mutual_info_score(membership_original, mem_sampled)
            
            results.append({
                'method': 'sampling',
                'retention': retention,
                'run': run,
                'modularity': mod_sampled,
                'modularity_change': mod_sampled - mod_original,
                'nmi': nmi_sampled,
                'time': time_sampled,
                'n_edges': G_sampled.number_of_edges()
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save
    output_file = OUTPUT_DIR / f"{dataset_name}_weighting_vs_sampling.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 100)
    print("ANALYSIS: WEIGHTING VS SAMPLING")
    print("=" * 100)
    
    # Aggregate sampling results
    sampling_agg = df[df['method'] == 'sampling'].groupby('retention').agg({
        'modularity': ['mean', 'std'],
        'time': 'mean',
        'n_edges': 'mean'
    }).reset_index()
    
    weighting_results = df[df['method'] == 'weighting']
    
    print("\nCOMPARISON:")
    print("-" * 100)
    print(f"{'Retention':<12} {'Weight Mod':<15} {'Sample Mod':<20} {'Weight Time':<15} {'Sample Time':<15}")
    print("-" * 100)
    
    for retention in RETENTION_VALUES:
        weight_row = weighting_results[weighting_results['retention'] == retention].iloc[0]
        sample_row = sampling_agg[sampling_agg['retention'] == retention].iloc[0]
        
        print(f"{retention:<12.2f} "
              f"{weight_row['modularity']:<15.4f} "
              f"{sample_row[('modularity', 'mean')]:.4f}±{sample_row[('modularity', 'std')]:.4f:<8} "
              f"{weight_row['time']:<15.3f} "
              f"{sample_row[('time', 'mean')]:<15.3f}")
    
    print("-" * 100)
    
    print("\nKEY FINDINGS:")
    print("1. Modularity: Weighting (deterministic) vs Sampling (mean across runs)")
    print("2. Time: Weighting uses all edges; Sampling uses fewer edges → faster Leiden")
    print("3. Theorem 3 prediction: Should converge as graph size increases")
    
    print("\n" + "=" * 100)


# ============================================================================
# EXPERIMENT 3.1: SCALABILITY ANALYSIS
# ============================================================================

def exp_3_1_scalability():
    """
    Test scalability: DSpar should be O(m), spectral is much slower.
    
    Strategy:
      - Generate scale-free networks of increasing size
      - Measure sparsification time
      - Measure community detection time
      - Compare DSpar vs spectral (if available)
    """
    OUTPUT_DIR = Path("results/exp3_1_scalability")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("EXPERIMENT 3.1: SCALABILITY ANALYSIS")
    print("=" * 100)
    
    # Network sizes to test
    SIZES = [
        (500, 2000),
        (1000, 5000),
        (2000, 10000),
        (5000, 25000),
        (10000, 50000),
        (20000, 100000),
    ]
    
    RETENTION = 0.5
    
    results = []
    
    print("\nGenerating and testing networks of increasing size...\n")
    
    for n_nodes, n_edges in SIZES:
        print(f"Testing n={n_nodes:,}, m={n_edges:,}...")
        
        # Generate scale-free network
        try:
            # Barabasi-Albert generates roughly m = (n-m0)*m0 edges
            m0 = int(n_edges / n_nodes)
            G = nx.barabasi_albert_graph(n_nodes, m0, seed=42)
            actual_edges = G.number_of_edges()
            
            print(f"  Generated: {actual_edges:,} edges")
            
            # Measure DSpar sparsification time
            start_dspar = time.time()
            
            degrees = dict(G.degree())
            edges = list(G.edges())
            scores = np.array([1.0/degrees[u] + 1.0/degrees[v] for u, v in edges])
            mean_score = scores.mean()
            
            probs = RETENTION * scores / mean_score
            probs = np.minimum(probs, 1.0)
            
            keep = np.random.rand(len(edges)) < probs
            
            G_sparse = nx.Graph()
            G_sparse.add_nodes_from(G.nodes())
            G_sparse.add_edges_from([edges[i] for i in range(len(edges)) if keep[i]])
            
            time_dspar = time.time() - start_dspar
            
            print(f"  DSpar time: {time_dspar:.3f}s")
            
            # Measure Leiden on original
            start_leiden_full = time.time()
            _, mod_full = run_leiden(G)
            time_leiden_full = time.time() - start_leiden_full
            
            print(f"  Leiden (full): {time_leiden_full:.3f}s, Q={mod_full:.4f}")
            
            # Measure Leiden on sparse
            start_leiden_sparse = time.time()
            _, mod_sparse = run_leiden(G_sparse)
            time_leiden_sparse = time.time() - start_leiden_sparse
            
            print(f"  Leiden (sparse): {time_leiden_sparse:.3f}s, Q={mod_sparse:.4f}")
            
            # Total pipeline time
            time_total_sparse = time_dspar + time_leiden_sparse
            speedup = time_leiden_full / time_total_sparse
            
            print(f"  Speedup: {speedup:.2f}×")
            
            results.append({
                'n_nodes': n_nodes,
                'n_edges': actual_edges,
                'time_dspar': time_dspar,
                'time_leiden_full': time_leiden_full,
                'time_leiden_sparse': time_leiden_sparse,
                'time_total_sparse': time_total_sparse,
                'speedup': speedup,
                'modularity_full': mod_full,
                'modularity_sparse': mod_sparse,
                'mod_change_pct': 100 * (mod_sparse - mod_full) / mod_full
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        
        print()
    
    # Save results
    df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "scalability_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 100)
    print("SCALABILITY ANALYSIS")
    print("=" * 100)
    
    print("\n" + "-" * 100)
    print(f"{'n':<10} {'m':<10} {'DSpar(s)':<12} {'Leiden(full)':<15} {'Leiden(sparse)':<15} {'Speedup':<10}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['n_nodes']:<10,} {row['n_edges']:<10,} "
              f"{row['time_dspar']:<12.3f} "
              f"{row['time_leiden_full']:<15.3f} "
              f"{row['time_leiden_sparse']:<15.3f} "
              f"{row['speedup']:<10.2f}×")
    
    print("-" * 100)
    
    # Check O(m) scaling
    print("\nSCALING VERIFICATION:")
    print("DSpar should scale as O(m)...")
    
    # Linear regression: log(time) vs log(m)
    from scipy.stats import linregress
    
    log_m = np.log10(df['n_edges'].values)
    log_time = np.log10(df['time_dspar'].values)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_m, log_time)
    
    print(f"  log(time) = {slope:.3f} * log(m) + {intercept:.3f}")
    print(f"  R² = {r_value**2:.4f}")
    print(f"  Expected slope ≈ 1.0 for O(m)")
    
    if 0.9 <= slope <= 1.1:
        print(f"  ✓ Empirical slope {slope:.3f} confirms O(m) scaling")
    else:
        print(f"  ~ Slope {slope:.3f} deviates from O(m)")
    
    print("\n" + "=" * 100)


# ============================================================================
# EXPERIMENT 4.1: PROPERTY CORRELATION (DIAGNOSTIC FRAMEWORK)
# ============================================================================

def exp_4_1_property_correlation():
    """
    Diagnostic framework: Predict when DSpar works based on network properties.
    
    Strategy:
      - Test on many diverse networks
      - Compute: δ, hub-bridge correlation, degree heterogeneity
      - Measure: modularity improvement
      - Correlation analysis: Which properties predict success?
    """
    OUTPUT_DIR = Path("results/exp4_1_property_correlation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("EXPERIMENT 4.1: PROPERTY CORRELATION (DIAGNOSTIC FRAMEWORK)")
    print("=" * 100)
    
    # Test datasets
    datasets = [
        'cit-HepPh',
        'cit-HepTh',
        'facebook',
        'ca-GrQc',
        'road-CA',  # Boundary case
    ]
    
    RETENTION = 0.5
    
    results = []
    
    print("\nTesting diverse networks...\n")
    
    for dataset_name in datasets:
        print(f"Processing {dataset_name}...")
        
        try:
            G = load_dataset(dataset_name)
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            
            # Compute network properties
            degrees = [d for _, d in G.degree()]
            degree_mean = np.mean(degrees)
            degree_std = np.std(degrees)
            degree_gini = compute_gini(degrees)
            
            # Run Leiden on original
            membership, mod_original = run_leiden(G)
            
            # Hub-bridge correlation
            hub_bridge_corr, hub_bridge_ratio = compute_hub_bridge_correlation(G, membership)
            
            # DSpar separation
            delta, mu_intra, mu_inter = compute_dspar_separation(G, membership)
            
            # Sparsify and re-run
            G_sparse = dspar_sparsify(G, RETENTION)
            membership_sparse, mod_sparse = run_leiden(G_sparse)
            
            mod_improvement = 100 * (mod_sparse - mod_original) / mod_original
            
            print(f"  δ={delta:.6f}, hub-bridge={hub_bridge_corr:.1f}, ΔQ={mod_improvement:+.2f}%")
            
            results.append({
                'dataset': dataset_name,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'degree_mean': degree_mean,
                'degree_std': degree_std,
                'degree_gini': degree_gini,
                'hub_bridge_correlation': hub_bridge_corr,
                'hub_bridge_ratio': hub_bridge_ratio,
                'dspar_delta': delta,
                'mu_intra': mu_intra,
                'mu_inter': mu_inter,
                'modularity_original': mod_original,
                'modularity_sparse': mod_sparse,
                'modularity_improvement_pct': mod_improvement
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save
    df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "property_correlation.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 100)
    print("CORRELATION ANALYSIS")
    print("=" * 100)
    
    print("\nNETWORK PROPERTIES VS IMPROVEMENT:")
    print("-" * 100)
    print(f"{'Dataset':<15} {'Gini':<10} {'δ':<12} {'Hub-bridge':<15} {'ΔQ %':<10}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        indicator = '✓' if row['modularity_improvement_pct'] > 0 else '✗'
        print(f"{row['dataset']:<15} "
              f"{row['degree_gini']:<10.4f} "
              f"{row['dspar_delta']:<12.6f} "
              f"{row['hub_bridge_correlation']:<15.1f} "
              f"{row['modularity_improvement_pct']:<+10.2f} {indicator}")
    
    print("-" * 100)
    
    # Correlation tests
    print("\nCORRELATION WITH MODULARITY IMPROVEMENT:")
    
    properties = ['degree_gini', 'dspar_delta', 'hub_bridge_correlation']
    
    for prop in properties:
        r, p = pearsonr(df[prop], df['modularity_improvement_pct'])
        print(f"  {prop:<30}: r = {r:+.4f}, p = {p:.4f}")
    
    # Most predictive property
    correlations = [(prop, abs(pearsonr(df[prop], df['modularity_improvement_pct'])[0])) 
                   for prop in properties]
    best_prop, best_r = max(correlations, key=lambda x: x[1])
    
    print(f"\nMost predictive property: {best_prop} (|r| = {best_r:.4f})")
    
    print("\n" + "=" * 100)


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


def dspar_sparsify(G, retention, seed=42):
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


def compute_gini(values):
    """Compute Gini coefficient"""
    values = np.array(values)
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n


# ============================================================================
# EXPERIMENT 4.2: FAILURE CASES (BOUNDARY CONDITIONS)
# ============================================================================

def exp_4_2_failure_cases():
    """
    Test explicit failure cases where DSpar should NOT work.
    
    Cases:
      - Regular lattices (homogeneous degree)
      - Road networks (spatial structure, no hub-bridging)
      - Random regular graphs (k-regular)
    """
    OUTPUT_DIR = Path("results/exp4_2_failure_cases")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("EXPERIMENT 4.2: FAILURE CASES (BOUNDARY CONDITIONS)")
    print("=" * 100)
    
    print("\nTesting networks where DSpar should FAIL (δ ≈ 0)...\n")
    
    RETENTION = 0.5
    results = []
    
    # ========================================================================
    # Case 1: 2D Grid Lattice
    # ========================================================================
    print("Case 1: 2D Grid Lattice (homogeneous degree)")
    
    try:
        G_grid = nx.grid_2d_graph(30, 30)
        G_grid = nx.convert_node_labels_to_integers(G_grid)
        
        membership, mod_original = run_leiden(G_grid)
        delta, _, _ = compute_dspar_separation(G_grid, membership)
        
        G_sparse = dspar_sparsify(G_grid, RETENTION)
        _, mod_sparse = run_leiden(G_sparse)
        
        improvement = 100 * (mod_sparse - mod_original) / mod_original
        
        print(f"  δ = {delta:.6f}, ΔQ = {improvement:+.2f}%")
        
        results.append({
            'case': '2D Grid',
            'n_nodes': G_grid.number_of_nodes(),
            'n_edges': G_grid.number_of_edges(),
            'delta': delta,
            'improvement_pct': improvement,
            'success': improvement > 0
        })
        
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # ========================================================================
    # Case 2: Random Regular Graph
    # ========================================================================
    print("\nCase 2: Random Regular Graph (k=5)")
    
    try:
        G_regular = nx.random_regular_graph(5, 1000, seed=42)
        
        membership, mod_original = run_leiden(G_regular)
        delta, _, _ = compute_dspar_separation(G_regular, membership)
        
        G_sparse = dspar_sparsify(G_regular, RETENTION)
        _, mod_sparse = run_leiden(G_sparse)
        
        improvement = 100 * (mod_sparse - mod_original) / mod_original
        
        print(f"  δ = {delta:.6f}, ΔQ = {improvement:+.2f}%")
        
        results.append({
            'case': 'Random Regular',
            'n_nodes': G_regular.number_of_nodes(),
            'n_edges': G_regular.number_of_edges(),
            'delta': delta,
            'improvement_pct': improvement,
            'success': improvement > 0
        })
        
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # ========================================================================
    # Case 3: Road Network (if available)
    # ========================================================================
    print("\nCase 3: Road Network (California)")
    
    try:
        G_road = load_dataset('road-CA')
        
        # Sample if too large
        if G_road.number_of_nodes() > 10000:
            nodes_sample = np.random.choice(G_road.nodes(), 10000, replace=False)
            G_road = G_road.subgraph(nodes_sample).copy()
            # Reconnect
            if not nx.is_connected(G_road):
                largest_cc = max(nx.connected_components(G_road), key=len)
                G_road = G_road.subgraph(largest_cc).copy()
        
        membership, mod_original = run_leiden(G_road)
        delta, _, _ = compute_dspar_separation(G_road, membership)
        
        G_sparse = dspar_sparsify(G_road, RETENTION)
        _, mod_sparse = run_leiden(G_sparse)
        
        improvement = 100 * (mod_sparse - mod_original) / mod_original
        
        print(f"  δ = {delta:.6f}, ΔQ = {improvement:+.2f}%")
        
        results.append({
            'case': 'Road Network',
            'n_nodes': G_road.number_of_nodes(),
            'n_edges': G_road.number_of_edges(),
            'delta': delta,
            'improvement_pct': improvement,
            'success': improvement > 0
        })
        
    except Exception as e:
        print(f"  Skipping road network: {e}")
    
    # Save
    df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "failure_cases.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 100)
    print("FAILURE CASE ANALYSIS")
    print("=" * 100)
    
    print("\n" + "-" * 80)
    print(f"{'Case':<20} {'δ':<15} {'ΔQ %':<15} {'Success?':<10}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        indicator = '✗ FAIL' if not row['success'] else '✓ PASS'
        print(f"{row['case']:<20} {row['delta']:<+15.6f} "
              f"{row['improvement_pct']:<+15.2f} {indicator:<10}")
    
    print("-" * 80)
    
    print("\nEXPECTED: All cases should show δ ≈ 0 and ΔQ ≈ 0 (no improvement)")
    print("This validates boundary conditions of the theory.")
    
    print("\n" + "=" * 100)


# ============================================================================
# MAIN DISPATCHER
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python exp_remaining.py [experiment] [dataset]")
        print("Experiments:")
        print("  2.2 - Weighting vs Sampling")
        print("  3.1 - Scalability Analysis")
        print("  4.1 - Property Correlation")
        print("  4.2 - Failure Cases")
        return
    
    experiment = sys.argv[1]
    
    if experiment == "2.2":
        dataset = sys.argv[2] if len(sys.argv) > 2 else "cit-HepPh"
        exp_2_2_weighting_vs_sampling(dataset)
    
    elif experiment == "3.1":
        exp_3_1_scalability()
    
    elif experiment == "4.1":
        exp_4_1_property_correlation()
    
    elif experiment == "4.2":
        exp_4_2_failure_cases()
    
    else:
        print(f"Unknown experiment: {experiment}")


if __name__ == "__main__":
    main()
