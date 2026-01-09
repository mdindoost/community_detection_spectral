"""
Main experiment runner: DSpar vs Spectral sparsification on citation networks.

Replicates functionality of experiments/cit_hepph_experiment.py using modular code.

Run: python src/main.py [dataset]
Datasets: cit-HepPh, cit-HepTh, citeseer, soc-LiveJournal1
"""
import sys
from pathlib import Path

# Setup path for absolute imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import igraph as ig

from src.config import (
    EPSILON_VALUES, RETENTION_VALUES, DSPAR_METHODS, CPM_RESOLUTIONS
)
from src.data import load_edges, load_graph
from src.clustering import run_leiden, run_leiden_timed
from src.clustering.leiden_cache import run_leiden_cached
from src.sparsifiers import dspar_sparsify, spectral_sparsify
from src.sparsifiers.dspar import dspar_sparsify_timed
from src.sparsifiers.spectral import spectral_sparsify_timed
from src.eval.metrics import compute_nmi_ari, calculate_edge_preservation_ratio, count_intra_inter_edges
from src.io import ResultsManager


def main():
    # Get dataset name from command line
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "cit-HepPh"
    
    print("=" * 90)
    print(f"{dataset_name.upper()} EXPERIMENT: DSpar vs Spectral Sparsification")
    print("=" * 90)
    
    # Initialize results manager
    results_mgr = ResultsManager()
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    edges = load_edges(dataset_name)
    G = ig.Graph(edges=edges.tolist(), directed=False)
    G.simplify()
    
    n_nodes = G.vcount()
    n_edges = G.ecount()
    n_cc = len(G.components())
    
    print(f"Original graph: {n_nodes:,} nodes, {n_edges:,} edges, {n_cc} connected components")
    
    # Store dataset info
    results_mgr.set_dataset_info(dataset_name, n_nodes, n_edges, n_cc)
    
    # Run Leiden on original (with caching)
    print("\nRunning Leiden on original graph...")
    mem_original, mod_original, n_comm_original, leiden_time_original, from_cache = run_leiden_cached(
        G, dataset_name, objective="modularity"
    )
    if from_cache:
        print(f"  Communities: {n_comm_original}, Modularity: {mod_original:.4f} (loaded from cache)")
    else:
        print(f"  Communities: {n_comm_original}, Modularity: {mod_original:.4f}, Time: {leiden_time_original:.2f}s")
    
    # Count intra/inter edges (vectorized)
    total_intra, total_inter = count_intra_inter_edges(edges, mem_original)
    
    # Store original results
    results_mgr.add_original_result(
        membership=mem_original,
        modularity=mod_original,
        n_communities=n_comm_original,
        leiden_time=leiden_time_original,
        total_intra=total_intra,
        total_inter=total_inter
    )
    
    # Results storage for summary
    results = []
    
    # Header
    print("\n" + "=" * 130)
    print(f"{'Method':<35} {'Param':<8} {'Edges':<12} {'%':<8} {'CC':<6} {'Comm':<6} {'Mod':<8} {'NMI':<8} {'ARI':<8} {'Intra%':<8} {'Inter%':<8} {'Ratio':<8} {'Spar(s)':<8} {'Leid(s)':<8}")
    print("-" * 130)
    print(f"{'Original':<35} {'-':<8} {n_edges:<12,} {'100%':<8} {n_cc:<6} {n_comm_original:<6} {mod_original:<8.4f} {'-':<8} {'-':<8} {'100.0':<8} {'100.0':<8} {'1.000':<8} {'-':<8} {leiden_time_original:<8.2f}")
    print(f"  (Intra: {total_intra:,} edges, Inter: {total_inter:,} edges)")
    print("-" * 130)
    
    # =========================================================================
    # SPECTRAL SPARSIFICATION
    # =========================================================================
    print("\n>>> SPECTRAL (Julia Laplacians.jl)")
    print("-" * 130)
    
    for epsilon in EPSILON_VALUES:
        try:
            G_sparse, spar_time = spectral_sparsify_timed(G, epsilon)
            
            n_edges_sparse = G_sparse.ecount()
            edge_pct = 100.0 * n_edges_sparse / n_edges
            n_cc_sparse = len(G_sparse.components())
            
            # Run Leiden on sparsified
            mem_sparse, mod_sparse, n_comm_sparse, leiden_time_sparse = run_leiden_timed(G_sparse)
            
            # Compute NMI/ARI
            nmi, ari = compute_nmi_ari(mem_original, mem_sparse)
            
            # Compute edge preservation ratio
            ratio_stats = calculate_edge_preservation_ratio(G, G_sparse, mem_original)
            intra_pct = ratio_stats['intra_rate'] * 100
            inter_pct = ratio_stats['inter_rate'] * 100
            ratio = ratio_stats['ratio']
            
            print(f"{'Spectral':<35} {'ε='+str(epsilon):<8} {n_edges_sparse:<12,} {edge_pct:<8.1f} {n_cc_sparse:<6} {n_comm_sparse:<6} {mod_sparse:<8.4f} {nmi:<8.4f} {ari:<8.4f} {intra_pct:<8.1f} {inter_pct:<8.1f} {ratio:<8.3f} {spar_time:<8.2f} {leiden_time_sparse:<8.2f}")
            
            # Store result
            results_mgr.add_experiment_result(
                method='Spectral',
                param=f'ε={epsilon}',
                n_edges_sparse=n_edges_sparse,
                edge_pct=edge_pct,
                n_components=n_cc_sparse,
                n_communities=n_comm_sparse,
                modularity=mod_sparse,
                nmi=nmi,
                ari=ari,
                intra_pct=intra_pct,
                inter_pct=inter_pct,
                ratio=ratio,
                sparsify_time=spar_time,
                leiden_time=leiden_time_sparse,
                membership=mem_sparse
            )
            
            results.append({
                'method': 'Spectral',
                'param': f'ε={epsilon}',
                'edges': n_edges_sparse,
                'edge_pct': edge_pct,
                'cc': n_cc_sparse,
                'communities': n_comm_sparse,
                'modularity': mod_sparse,
                'nmi': nmi,
                'ari': ari,
                'intra_pct': intra_pct,
                'inter_pct': inter_pct,
                'ratio': ratio,
                'spar_time': spar_time,
                'leiden_time': leiden_time_sparse
            })
            
        except Exception as e:
            print(f"{'Spectral':<35} {'ε='+str(epsilon):<8} ERROR: {e}")
    
    # =========================================================================
    # DSPAR SPARSIFICATION
    # =========================================================================
    for method in DSPAR_METHODS:
        print(f"\n>>> DSPAR ({method})")
        print("-" * 130)
        
        for retention in RETENTION_VALUES:
            try:
                G_sparse, spar_time = dspar_sparsify_timed(G, retention, method, seed=42)
                
                n_edges_sparse = G_sparse.ecount()
                edge_pct = 100.0 * n_edges_sparse / n_edges
                n_cc_sparse = len(G_sparse.components())
                
                # Run Leiden on sparsified
                mem_sparse, mod_sparse, n_comm_sparse, leiden_time_sparse = run_leiden_timed(G_sparse)
                
                # Compute NMI/ARI
                nmi, ari = compute_nmi_ari(mem_original, mem_sparse)
                
                # Compute edge preservation ratio
                ratio_stats = calculate_edge_preservation_ratio(G, G_sparse, mem_original)
                intra_pct = ratio_stats['intra_rate'] * 100
                inter_pct = ratio_stats['inter_rate'] * 100
                ratio = ratio_stats['ratio']
                
                method_name = f"DSpar ({method})"
                print(f"{method_name:<35} {'r='+str(retention):<8} {n_edges_sparse:<12,} {edge_pct:<8.1f} {n_cc_sparse:<6} {n_comm_sparse:<6} {mod_sparse:<8.4f} {nmi:<8.4f} {ari:<8.4f} {intra_pct:<8.1f} {inter_pct:<8.1f} {ratio:<8.3f} {spar_time:<8.2f} {leiden_time_sparse:<8.2f}")
                
                # Store result
                results_mgr.add_experiment_result(
                    method=method_name,
                    param=f'r={retention}',
                    n_edges_sparse=n_edges_sparse,
                    edge_pct=edge_pct,
                    n_components=n_cc_sparse,
                    n_communities=n_comm_sparse,
                    modularity=mod_sparse,
                    nmi=nmi,
                    ari=ari,
                    intra_pct=intra_pct,
                    inter_pct=inter_pct,
                    ratio=ratio,
                    sparsify_time=spar_time,
                    leiden_time=leiden_time_sparse,
                    membership=mem_sparse
                )
                
                results.append({
                    'method': method_name,
                    'param': f'r={retention}',
                    'edges': n_edges_sparse,
                    'edge_pct': edge_pct,
                    'cc': n_cc_sparse,
                    'communities': n_comm_sparse,
                    'modularity': mod_sparse,
                    'nmi': nmi,
                    'ari': ari,
                    'intra_pct': intra_pct,
                    'inter_pct': inter_pct,
                    'ratio': ratio,
                    'spar_time': spar_time,
                    'leiden_time': leiden_time_sparse
                })
                
            except Exception as e:
                print(f"{'DSpar ('+method+')':<35} {'r='+str(retention):<8} ERROR: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print(f"\nOriginal: {n_nodes:,} nodes, {n_edges:,} edges, {n_cc} CC, {n_comm_original} communities, Modularity={mod_original:.4f}")
    print(f"Leiden time on original: {leiden_time_original:.2f}s")
    
    if results:
        best_nmi = max(results, key=lambda x: x['nmi'])
        best_ari = max(results, key=lambda x: x['ari'])
        best_mod = max(results, key=lambda x: x['modularity'])
        best_ratio = min(results, key=lambda x: x['ratio'])  # Lower is better
        fastest_spar = min(results, key=lambda x: x['spar_time'])
        
        print(f"\nBest NMI: {best_nmi['method']} {best_nmi['param']} -> NMI={best_nmi['nmi']:.4f}")
        print(f"Best ARI: {best_ari['method']} {best_ari['param']} -> ARI={best_ari['ari']:.4f}")
        print(f"Best Modularity: {best_mod['method']} {best_mod['param']} -> Mod={best_mod['modularity']:.4f} (original={mod_original:.4f})")
        print(f"Best Ratio: {best_ratio['method']} {best_ratio['param']} -> Ratio={best_ratio['ratio']:.3f} (< 1 means inter removed faster)")
        print(f"Fastest sparsification: {fastest_spar['method']} {fastest_spar['param']} -> {fastest_spar['spar_time']:.2f}s")
    
    # =========================================================================
    # CPM RESOLUTION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 100)
    print("CPM RESOLUTION ANALYSIS (Constant Potts Model)")
    print("=" * 100)
    print("\nComparing community detection at different resolutions on original vs best sparsified graph")
    print("Lower resolution = larger communities, Higher resolution = smaller communities\n")
    
    cpm_results = []
    
    if results:
        best_result = max(results, key=lambda x: x['modularity'])
        best_method = best_result['method']
        best_param = best_result['param']
        
        # Re-run the best sparsification to get the graph
        if 'Spectral' in best_method:
            epsilon = float(best_param.split('=')[1])
            G_best_sparse, _ = spectral_sparsify_timed(G, epsilon)
        else:
            retention = float(best_param.split('=')[1])
            method = best_method.split('(')[1].split(')')[0]
            G_best_sparse, _ = dspar_sparsify_timed(G, retention, method)
        
        print(f"Best sparsification: {best_method} {best_param} (Mod={best_result['modularity']:.4f})")
        print(f"Edges: {G_best_sparse.ecount():,} ({100*G_best_sparse.ecount()/n_edges:.1f}%)\n")
        
        print(f"{'Resolution':<12} {'Original Comm':<15} {'Original Mod':<15} {'Sparse Comm':<15} {'Sparse Mod':<15}")
        print("-" * 75)
        
        for res in CPM_RESOLUTIONS:
            # Run on original (with caching)
            mem_orig_cpm, mod_orig_cpm, n_comm_orig_cpm, _, _ = run_leiden_cached(
                G, dataset_name, objective="CPM", resolution=res
            )
            
            # Run on best sparsified (no caching for sparsified graphs)
            mem_sparse_cpm, mod_sparse_cpm, n_comm_sparse_cpm = run_leiden(G_best_sparse, objective="CPM", resolution=res)
            
            print(f"{res:<12} {n_comm_orig_cpm:<15} {mod_orig_cpm:<15.4f} {n_comm_sparse_cpm:<15} {mod_sparse_cpm:<15.4f}")
            
            cpm_results.append({
                'resolution': res,
                'original_communities': n_comm_orig_cpm,
                'original_modularity': mod_orig_cpm,
                'sparse_communities': n_comm_sparse_cpm,
                'sparse_modularity': mod_sparse_cpm
            })
        
        results_mgr.add_cpm_results(cpm_results)
    
    print("\n" + "=" * 130)
    print("INTERPRETATION")
    print("=" * 130)
    print("""
METRICS:
- Mod (Modularity): Quality of clustering on that graph (higher = better separated communities)
- NMI/ARI: Similarity to original clustering (higher = more consistent with original)
- Intra%: Percentage of intra-community edges preserved
- Inter%: Percentage of inter-community edges preserved
- Ratio: Inter% / Intra% (< 1 means inter-community edges removed faster = DESIRED)
- CPM Resolution: Controls community granularity (lower = larger communities)

KEY INSIGHTS:
- Ratio < 1: Inter-community edges removed faster (good for denoising hypothesis)
- Ratio = 1: No preference between edge types (like random sparsification)
- Ratio > 1: Intra-community edges removed faster (bad - losing community structure)
- Modularity can INCREASE after sparsification if noise edges are removed
- High NMI/ARI + similar Modularity = sparsification preserves structure well
- Spectral preserves connectivity (keeps bridge edges)
- DSpar is faster but may disconnect graph (removes hub edges)
""")
    
    # Finalize results
    run_folder = results_mgr.finalize()
    print(f"\nResults saved to: {run_folder}")


if __name__ == "__main__":
    main()

