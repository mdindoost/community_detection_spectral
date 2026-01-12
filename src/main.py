#!/usr/bin/env python3
"""
Experiment 3: Scalability and Large-Graph Tradeoffs

Evaluates DSpar, baseline sparsifiers (uniform random, degree sampling), 
and optionally spectral sparsification on large graphs.

Clean modular implementation - all functionality in dedicated modules.
Uses parallel processing for speedup while maintaining ordered results.

Run: python -m src.main
     python -m src.main --datasets com-DBLP
     python -m src.main --datasets com-DBLP,com-Amazon --dry_run
     python -m src.main --max_edges 1000000
     python -m src.main --workers 8
"""
import sys
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

# Setup path for absolute imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Import from dedicated modules - each module does ONE thing
from src.config import (
    ALPHAS, METHODS, N_REPLICATES, SEED_BASE,
    DEFAULT_DATASETS, RESULTS_DIR,
    SPECTRAL_EPSILON_MAP, SPECTRAL_TIMEOUT
)
from src.data import load_large_dataset
from src.clustering import run_leiden_timed
from src.eval.metrics import compute_nmi, compute_modularity_fixed
from src.sparsifiers import sparsify_timed, SPECTRAL_AVAILABLE
from src.io.results_manager import generate_summary, print_key_results, get_next_run_folder


# =============================================================================
# PARALLEL TRIAL EXECUTION
# =============================================================================

@dataclass
class TrialConfig:
    """Configuration for a single trial - picklable for multiprocessing."""
    trial_idx: int  # For ordering results
    method: str
    alpha: float
    replicate: int
    seed: int
    dataset_name: str


def run_single_trial_worker(
    trial_config: TrialConfig,
    G_edges: np.ndarray,
    n_nodes: int,
    baseline_membership: List[int],
    Q0: float,
    T_leiden_orig: float,
    epsilon_map: dict
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Worker function for parallel trial execution.
    
    Returns (trial_idx, result_dict) to maintain ordering.
    """
    import igraph as ig
    from src.sparsifiers import sparsify_timed
    from src.clustering import run_leiden_timed
    from src.eval.metrics import compute_nmi, compute_modularity_fixed
    
    # Reconstruct graph from edges (can't pickle igraph objects)
    G = ig.Graph(n=n_nodes, edges=G_edges.tolist(), directed=False)
    
    m_edges = G.ecount()
    method = trial_config.method
    alpha = trial_config.alpha
    seed = trial_config.seed
    
    # Handle alpha = 1.0 (no sparsification)
    if alpha >= 1.0:
        G_sparse = G.copy()
        T_sparsify = 0.0
    else:
        G_sparse, T_sparsify = sparsify_timed(G, method, alpha, seed, epsilon_map=epsilon_map)
    
    # Handle failed sparsification
    if G_sparse is None:
        return (trial_config.trial_idx, None)
    
    m_sparse = G_sparse.ecount()
    retention_actual = m_sparse / m_edges if m_edges > 0 else 1.0
    
    # Compute Q_sparse_fixed
    Q_sparse_fixed = compute_modularity_fixed(G_sparse, baseline_membership)
    dQ_fixed = Q_sparse_fixed - Q0
    
    # Run Leiden on sparsified graph
    membership_sparse, Q_sparse_leiden, n_comm_sparse, T_leiden_sparse = run_leiden_timed(G_sparse)
    dQ_leiden = Q_sparse_leiden - Q0
    
    # Pipeline time and speedup
    T_pipeline = T_sparsify + T_leiden_sparse
    speedup = T_leiden_orig / T_pipeline if T_pipeline > 0 else 1.0
    
    # NMI between baseline and sparse partitions
    nmi_P0_Palpha = compute_nmi(baseline_membership, membership_sparse)
    
    result = {
        'dataset': trial_config.dataset_name,
        'method': method,
        'alpha': alpha,
        'replicate': trial_config.replicate,
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
        
        'n_communities_orig': len(set(baseline_membership)),
        'n_communities_sparse': n_comm_sparse,
        'nmi_P0_Palpha': nmi_P0_Palpha,
    }
    
    return (trial_config.trial_idx, result)


def run_dataset_experiments(
    G, 
    dataset_info: dict,
    methods: list, 
    alphas: list,
    n_replicates: int,
    n_workers: int = None,
    spectral_timeout_multiplier: float = 10.0
) -> list:
    """
    Run all experiments for a single dataset using parallel processing.
    Results are returned in deterministic order (method, alpha, replicate).
    """
    # Default to number of CPUs
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)
    
    print(f"\n  Running Leiden on original graph...")
    baseline_membership, Q0, n_communities_orig, T_leiden_orig = run_leiden_timed(G)
    
    print(f"    Baseline modularity Q0 = {Q0:.6f}")
    print(f"    Baseline communities = {n_communities_orig}")
    print(f"    Baseline Leiden time = {T_leiden_orig:.3f}s")
    
    # Prepare graph data for workers (can't pickle igraph objects)
    G_edges = np.array(G.get_edgelist())
    n_nodes = G.vcount()
    baseline_membership_list = list(baseline_membership)
    
    # Separate methods: run non-spectral first, then spectral
    non_spectral_methods = [m for m in methods if m != 'spectral']
    has_spectral = 'spectral' in methods
    
    results = []
    dspar_times = []
    
    # Phase 1: Run non-spectral methods in parallel
    print(f"\n  Phase 1: Running non-spectral methods (parallel, {n_workers} workers)...")
    
    # Create all trial configs with indices for ordering
    trial_configs = []
    trial_idx = 0
    for method in non_spectral_methods:
        for alpha in alphas:
            for rep in range(n_replicates):
                seed = SEED_BASE + hash((dataset_info['name'], method, alpha, rep)) % (2**20)
                trial_configs.append(TrialConfig(
                    trial_idx=trial_idx,
                    method=method,
                    alpha=alpha,
                    replicate=rep,
                    seed=seed,
                    dataset_name=dataset_info['name']
                ))
                trial_idx += 1
    
    total_trials = len(trial_configs)
    completed = 0
    indexed_results = {}
    
    # Run trials in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                run_single_trial_worker,
                config,
                G_edges,
                n_nodes,
                baseline_membership_list,
                Q0,
                T_leiden_orig,
                SPECTRAL_EPSILON_MAP
            ): config for config in trial_configs
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            config = futures[future]
            completed += 1
            
            try:
                idx, result = future.result()
                if result is not None:
                    indexed_results[idx] = result
                    # Track DSpar times for spectral timeout
                    if result['method'] == 'dspar' and result['alpha'] < 1.0:
                        dspar_times.append(result['T_sparsify_sec'])
                    status = "✓"
                else:
                    status = "SKIP"
            except Exception as e:
                status = f"ERR: {e}"
            
            print(f"\r    [{completed}/{total_trials}] {config.method}, α={config.alpha:.1f}, rep={config.replicate+1} {status}",
                  end='', flush=True)
    
    print()
    
    # Sort results by trial index to maintain order
    for idx in sorted(indexed_results.keys()):
        results.append(indexed_results[idx])
    
    # Phase 2: Run spectral with dynamic timeout (SEQUENTIAL to avoid memory issues)
    if has_spectral and SPECTRAL_AVAILABLE and len(dspar_times) > 0:
        max_dspar_time = max(dspar_times)
        dynamic_timeout = int(max_dspar_time * spectral_timeout_multiplier)
        dynamic_timeout = max(dynamic_timeout, 30)
        
        print(f"\n  Phase 2: Running spectral sparsification (SEQUENTIAL - memory intensive)...")
        print(f"    Max DSpar time: {max_dspar_time:.2f}s")
        print(f"    Spectral timeout: {dynamic_timeout}s ({spectral_timeout_multiplier}× DSpar)")
        
        # Create spectral trial configs
        spectral_configs = []
        trial_idx = 0
        for alpha in alphas:
            for rep in range(n_replicates):
                seed = SEED_BASE + hash((dataset_info['name'], 'spectral', alpha, rep)) % (2**20)
                spectral_configs.append(TrialConfig(
                    trial_idx=trial_idx,
                    method='spectral',
                    alpha=alpha,
                    replicate=rep,
                    seed=seed,
                    dataset_name=dataset_info['name']
                ))
                trial_idx += 1
        
        total_spectral = len(spectral_configs)
        
        # Run spectral trials SEQUENTIALLY (Julia is memory-intensive)
        for i, config in enumerate(spectral_configs):
            try:
                idx, result = run_single_trial_worker(
                    config,
                    G_edges,
                    n_nodes,
                    baseline_membership_list,
                    Q0,
                    T_leiden_orig,
                    SPECTRAL_EPSILON_MAP
                )
                if result is not None:
                    results.append(result)
                    status = "✓"
                else:
                    status = "SKIP"
            except Exception as e:
                status = f"ERR: {e}"
            
            print(f"    [{i+1}/{total_spectral}] spectral, α={config.alpha:.1f}, rep={config.replicate+1} {status}")
    
    elif has_spectral:
        print(f"\n  [WARNING] No DSpar times recorded, skipping spectral")
    
    return results


def run_all_experiments(
    datasets: list, 
    methods: list, 
    alphas: list,
    n_replicates: int, 
    max_edges: int = None,
    n_workers: int = None,
    spectral_timeout_multiplier: float = 10.0,
    output_file: Path = None
) -> pd.DataFrame:
    """Run experiments on all datasets with incremental saving."""
    all_results = []
    first_write = True
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        G, info = load_large_dataset(dataset_name, max_edges=max_edges)
        
        if G is None:
            print(f"  [SKIP] Could not load {dataset_name}")
            continue
        
        results = run_dataset_experiments(
            G, info, methods, alphas, n_replicates,
            n_workers=n_workers,
            spectral_timeout_multiplier=spectral_timeout_multiplier
        )
        all_results.extend(results)
        
        # Save incrementally
        if output_file and len(results) > 0:
            df_dataset = pd.DataFrame(results)
            if first_write:
                df_dataset.to_csv(output_file, index=False, mode='w')
                first_write = False
            else:
                df_dataset.to_csv(output_file, index=False, mode='a', header=False)
            print(f"\n  [SAVED] Results appended to {output_file}")
    
    return pd.DataFrame(all_results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Scalability and Large-Graph Tradeoffs"
    )
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated list of datasets (default: all DEFAULT_DATASETS)"
    )
    parser.add_argument(
        "--max_edges", type=int, default=None,
        help="Maximum edges to keep per dataset (for testing)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print configuration and exit without running"
    )
    parser.add_argument(
        "--replicates", type=int, default=N_REPLICATES,
        help=f"Number of replicates per configuration (default: {N_REPLICATES})"
    )
    parser.add_argument(
        "--no_spectral", action="store_true",
        help="Exclude spectral sparsification"
    )
    parser.add_argument(
        "--spectral_timeout", type=int, default=SPECTRAL_TIMEOUT,
        help=f"Initial timeout for spectral (default: {SPECTRAL_TIMEOUT})"
    )
    parser.add_argument(
        "--spectral_multiplier", type=float, default=10.0,
        help="Spectral timeout = multiplier × max(DSpar time). Default: 10.0"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    
    args = parser.parse_args()
    
    # Determine datasets
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else DEFAULT_DATASETS
    n_replicates = args.replicates
    
    # Determine methods
    methods = METHODS.copy()
    if not args.no_spectral:
        if SPECTRAL_AVAILABLE:
            methods.append('spectral')
        else:
            print("WARNING: Spectral not available, skipping")
    else:
        print("NOTE: Spectral excluded (--no_spectral flag)")
    
    print("=" * 100)
    print("EXPERIMENT 3: SCALABILITY AND LARGE-GRAPH TRADEOFFS")
    print("=" * 100)
    
    # Create numbered output directory (consecutive: 1, 2, 3, ...)
    OUTPUT_DIR = get_next_run_folder(RESULTS_DIR)
    
    print(f"\nConfiguration:")
    print(f"  Datasets: {datasets}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Methods: {methods}")
    print(f"  Replicates: {n_replicates}")
    print(f"  Max edges: {args.max_edges if args.max_edges else 'unlimited'}")
    print(f"  Workers: {args.workers if args.workers else 'auto (CPU-1)'}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    if 'spectral' in methods:
        print(f"\n  NOTE: Spectral sparsification enabled (runs SEQUENTIALLY)")
        print(f"        Dynamic timeout: {args.spectral_multiplier}× max(DSpar time)")
    
    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return
    
    # Run experiments
    raw_file = OUTPUT_DIR / "scalability_raw.csv"
    
    df = run_all_experiments(
        datasets=datasets,
        methods=methods,
        alphas=ALPHAS,
        n_replicates=n_replicates,
        max_edges=args.max_edges,
        n_workers=args.workers,
        spectral_timeout_multiplier=args.spectral_multiplier,
        output_file=raw_file
    )
    
    if len(df) == 0:
        print("\nERROR: No experiments completed successfully!")
        return
    
    print(f"\nRaw results saved to: {raw_file}")
    
    # Generate summary
    summary = generate_summary(df)
    summary_file = OUTPUT_DIR / "scalability_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")
    
    # Print key results
    print_key_results(df, alpha=0.8)
    
    # Final summary
    print(f"\n{'='*100}")
    print("EXPERIMENT 3 COMPLETE")
    print(f"{'='*100}")
    print(f"\nOutput files:")
    print(f"  Raw CSV: {raw_file}")
    print(f"  Summary CSV: {summary_file}")
    
    # Summary statistics
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}")
    
    for dataset in df['dataset'].unique():
        df_d = df[(df['dataset'] == dataset) & np.isclose(df['alpha'], 0.8)]
        print(f"\n{dataset}:")
        
        for method in df['method'].unique():
            df_m = df_d[df_d['method'] == method]
            if len(df_m) == 0:
                continue
            
            speedup = df_m['speedup'].mean()
            dQ = df_m['dQ_leiden'].mean()
            quality_note = "improved" if dQ > 0 else ("degraded" if dQ < -0.01 else "maintained")
            print(f"  {method}: {speedup:.2f}x speedup, quality {quality_note} (ΔQ={dQ:+.4f})")
    
    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()



