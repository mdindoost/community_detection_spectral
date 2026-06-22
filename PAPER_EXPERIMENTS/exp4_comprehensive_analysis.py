#!/usr/bin/env python3
"""
Experiment 4: Comprehensive DSpar-seeded Leiden analysis.

Computes 6 categories of properties per dataset:
  1. Graph structural properties
  2. Baseline Leiden stability (N runs)
  3. DSpar separation analysis (delta, Cohen's d, distribution overlap)
  4. Sparsification effects (per alpha)
  5. Seed quality analysis (per alpha, N runs)
  6. DSpar-seeded Leiden performance (per alpha, N runs)

Outputs:
  results/exp4_comprehensive/comprehensive_properties.csv
  results/exp4_comprehensive/comprehensive_alpha_results.csv
  results/exp4_comprehensive/figures/*.pdf
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import igraph as ig
from scipy import stats as sp_stats

from src.clustering.run_leiden import run_leiden
from src.clustering.leiden_progressive import leiden_dspar_seeded
from src.sparsifiers.dspar import dspar_sparsify
from src.data.load_dataset import load_large_dataset

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


DATASETS = [
    'facebook-combined',
    'ca-GrQc',
    'ca-HepTh',
    'wiki-Vote',
    'ca-HepPh',
    'ca-AstroPh',
    'ca-CondMat',
    'cit-HepTh',
    'email-Enron',
    'cit-HepPh',
    'soc-Epinions1',
    'com-Amazon',
    'com-DBLP',
    'wiki-Talk',
    'com-Youtube',
]

ALPHAS = [1.0, 0.90, 0.80]
N_RUNS = 10
SEED_BASE = 42

OUTPUT_DIR = Path(__file__).parent / "results" / "exp4_comprehensive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def compute_gini(values):
    """Gini coefficient of a distribution."""
    values = np.array(values, dtype=np.float64)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float(
        (2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals))
        / (n * np.sum(sorted_vals))
    )


def compute_distribution_overlap(a, b, n_bins=100):
    """Overlap coefficient between two distributions (histogram-based)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if lo == hi:
        return 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    return float(np.sum(np.minimum(hist_a, hist_b)) * bin_width)


def pairwise_nmi(partitions):
    """Mean pairwise NMI across all partition pairs."""
    n = len(partitions)
    if n < 2:
        return 1.0
    nmi_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            nmi_vals.append(
                ig.compare_communities(partitions[i], partitions[j], method='nmi')
            )
    return float(np.mean(nmi_vals))


# ============================================================
# Category 1: Graph Properties
# ============================================================

def compute_graph_properties(G):
    n = G.vcount()
    m = G.ecount()
    degrees = np.array(G.degree(), dtype=np.float64)
    return {
        'n_nodes': n,
        'n_edges': m,
        'density': 2 * m / (n * (n - 1)) if n > 1 else 0.0,
        'degree_mean': float(degrees.mean()),
        'degree_std': float(degrees.std()),
        'degree_max': int(degrees.max()),
        'degree_median': float(np.median(degrees)),
        'degree_skew': float(sp_stats.skew(degrees)),
        'degree_gini': compute_gini(degrees),
        'clustering_coef': G.transitivity_undirected(),
        'assortativity': G.assortativity_degree(directed=False),
    }


# ============================================================
# Category 2: Baseline Leiden Properties
# ============================================================

def compute_baseline_properties(G, n_runs=10):
    Q_vals = []
    n_comm_vals = []
    partitions = []

    for run in range(n_runs):
        mem, Q, n_comm = run_leiden(
            G, objective="modularity", resolution=1.0, n_iterations=2
        )
        Q_vals.append(Q)
        n_comm_vals.append(n_comm)
        partitions.append(mem)

    Q_vals = np.array(Q_vals)
    n_comm_vals = np.array(n_comm_vals, dtype=float)

    stability = pairwise_nmi(partitions)

    best_idx = int(np.argmax(Q_vals))
    mem_best = partitions[best_idx]

    # Bridge fraction
    edges = np.array(G.get_edgelist())
    mem_arr = np.array(mem_best)
    inter_mask = mem_arr[edges[:, 0]] != mem_arr[edges[:, 1]]
    bridge_fraction = float(inter_mask.mean())

    # Community size distribution
    comm_sizes = np.bincount(mem_arr)
    comm_sizes = comm_sizes[comm_sizes > 0]
    n_eff_communities = float(np.exp(-np.sum(
        (comm_sizes / comm_sizes.sum()) * np.log(comm_sizes / comm_sizes.sum() + 1e-12)
    )))

    props = {
        'Q_base_mean': float(Q_vals.mean()),
        'Q_base_std': float(Q_vals.std()),
        'Q_base_min': float(Q_vals.min()),
        'Q_base_max': float(Q_vals.max()),
        'Q_base_cv': float(Q_vals.std() / Q_vals.mean()) if Q_vals.mean() > 0 else 0.0,
        'Q_base_range': float(Q_vals.max() - Q_vals.min()),
        'n_comm_base_mean': float(n_comm_vals.mean()),
        'n_comm_base_std': float(n_comm_vals.std()),
        'n_comm_base_cv': float(n_comm_vals.std() / n_comm_vals.mean()) if n_comm_vals.mean() > 0 else 0.0,
        'partition_stability': stability,
        'bridge_fraction': bridge_fraction,
        'comm_size_gini': compute_gini(comm_sizes),
        'n_eff_communities': n_eff_communities,
    }

    return props, partitions[best_idx]


# ============================================================
# Category 3: DSpar Separation Analysis
# ============================================================

def compute_dspar_analysis(G, membership):
    degrees = np.array(G.degree(), dtype=np.float64)
    edges = np.array(G.get_edgelist())
    mem = np.array(membership)

    sources, targets = edges[:, 0], edges[:, 1]
    dspar_scores = 1.0 / degrees[sources] + 1.0 / degrees[targets]
    same_comm = mem[sources] == mem[targets]

    scores_intra = dspar_scores[same_comm]
    scores_inter = dspar_scores[~same_comm]

    mu_intra = float(scores_intra.mean()) if len(scores_intra) > 0 else 0.0
    mu_inter = float(scores_inter.mean()) if len(scores_inter) > 0 else 0.0
    delta = mu_intra - mu_inter

    # Hub-bridge ratio
    products = degrees[sources] * degrees[targets]
    mean_intra_prod = float(products[same_comm].mean()) if same_comm.any() else 1.0
    mean_inter_prod = float(products[~same_comm].mean()) if (~same_comm).any() else 0.0
    hb_ratio = mean_inter_prod / mean_intra_prod if mean_intra_prod > 0 else 1.0

    # Cohen's d
    var_intra = float(np.var(scores_intra)) if len(scores_intra) > 0 else 0.0
    var_inter = float(np.var(scores_inter)) if len(scores_inter) > 0 else 0.0
    pooled_var = (var_intra + var_inter) / 2
    cohens_d = delta / np.sqrt(pooled_var) if pooled_var > 0 else 0.0

    # Distribution overlap
    overlap = compute_distribution_overlap(scores_intra, scores_inter)

    return {
        'delta': delta,
        'mu_intra': mu_intra,
        'mu_inter': mu_inter,
        'hb_ratio': hb_ratio,
        'intra_dspar_std': float(np.std(scores_intra)) if len(scores_intra) > 1 else 0.0,
        'inter_dspar_std': float(np.std(scores_inter)) if len(scores_inter) > 1 else 0.0,
        'intra_dspar_median': float(np.median(scores_intra)) if len(scores_intra) > 0 else 0.0,
        'inter_dspar_median': float(np.median(scores_inter)) if len(scores_inter) > 0 else 0.0,
        'dspar_cohens_d': float(cohens_d),
        'dspar_overlap': overlap,
    }


# ============================================================
# Category 4: Sparsification Effects
# ============================================================

def compute_sparsification_effects(G, alpha, membership_baseline, seed=42):
    degrees = np.array(G.degree(), dtype=np.float64)
    m_orig = G.ecount()

    G_sparse = dspar_sparsify(G, retention=alpha, method="paper", seed=seed)
    if G_sparse.is_weighted():
        G_sparse = ig.Graph(
            n=G_sparse.vcount(), edges=G_sparse.get_edgelist(), directed=False
        )

    m_sparse = G_sparse.ecount()
    actual_retention = m_sparse / m_orig if m_orig > 0 else 0.0

    # Connectivity
    components = G_sparse.components()
    n_components = len(components)
    largest_cc = max(components.sizes())
    largest_cc_frac = largest_cc / G_sparse.vcount() if G_sparse.vcount() > 0 else 0.0

    # Edge removal pattern via degree products
    edges_sparse = np.array(G_sparse.get_edgelist())
    products_sparse = degrees[edges_sparse[:, 0]] * degrees[edges_sparse[:, 1]]
    mean_retained = float(products_sparse.mean()) if len(products_sparse) > 0 else 0.0

    edges_orig = np.array(G.get_edgelist())
    products_orig = degrees[edges_orig[:, 0]] * degrees[edges_orig[:, 1]]

    n_removed = m_orig - m_sparse
    if n_removed > 0:
        sum_removed = products_orig.sum() - products_sparse.sum()
        mean_removed = float(sum_removed / n_removed)
    else:
        mean_removed = mean_retained

    removal_bias = mean_removed / mean_retained if mean_retained > 0 else 1.0

    # DSpar separation on sparse graph (using baseline membership)
    mem = np.array(membership_baseline)
    src_s, tgt_s = edges_sparse[:, 0], edges_sparse[:, 1]
    dspar_s = 1.0 / degrees[src_s] + 1.0 / degrees[tgt_s]
    same_s = mem[src_s] == mem[tgt_s]
    mu_intra_s = float(dspar_s[same_s].mean()) if same_s.any() else 0.0
    mu_inter_s = float(dspar_s[~same_s].mean()) if (~same_s).any() else 0.0

    props = {
        'actual_retention': actual_retention,
        'n_components': n_components,
        'largest_component_frac': largest_cc_frac,
        'mean_degprod_retained': mean_retained,
        'mean_degprod_removed': mean_removed,
        'removal_bias': removal_bias,
        'delta_sparse': mu_intra_s - mu_inter_s,
    }
    return props, G_sparse


# ============================================================
# Category 5: Seed Quality Analysis
# ============================================================

def compute_seed_quality(G, G_sparse, membership_baseline, Q_baseline_mean, n_runs=10):
    Q_sparse_vals = []
    Q_on_orig_vals = []
    nmi_to_base_vals = []
    partitions_sparse = []

    for run in range(n_runs):
        mem_s, Q_s, _ = run_leiden(
            G_sparse, objective="modularity", resolution=1.0, n_iterations=2
        )
        Q_sparse_vals.append(Q_s)
        Q_on_orig_vals.append(G.modularity(mem_s, weights=None))
        nmi_to_base_vals.append(
            ig.compare_communities(mem_s, membership_baseline, method='nmi')
        )
        partitions_sparse.append(mem_s)

    Q_sparse_vals = np.array(Q_sparse_vals)
    Q_on_orig_vals = np.array(Q_on_orig_vals)
    nmi_to_base_vals = np.array(nmi_to_base_vals)
    stability_sparse = pairwise_nmi(partitions_sparse)

    return {
        'Q_sparse_mean': float(Q_sparse_vals.mean()),
        'Q_sparse_std': float(Q_sparse_vals.std()),
        'Q_on_orig_mean': float(Q_on_orig_vals.mean()),
        'Q_on_orig_std': float(Q_on_orig_vals.std()),
        'Q_transfer_loss': float(Q_baseline_mean - Q_on_orig_vals.mean()),
        'NMI_to_baseline_mean': float(nmi_to_base_vals.mean()),
        'NMI_to_baseline_std': float(nmi_to_base_vals.std()),
        'partition_stability_sparse': stability_sparse,
    }


# ============================================================
# Category 6: DSpar-Seeded Performance
# ============================================================

def compute_dspar_seeded_performance(G, alpha, Q_base_mean, Q_base_std, Q_base_max, n_runs=10):
    Q_final_vals = []
    n_comm_vals = []
    runtimes = []
    partitions_final = []

    for run in range(n_runs):
        seed = SEED_BASE + run * 100
        mem, info = leiden_dspar_seeded(
            G, retention=alpha, seed=seed, verbose=False,
        )
        Q_final_vals.append(info['modularity_final'])
        n_comm_vals.append(info['n_communities_final'])
        runtimes.append(info['total_runtime'])
        partitions_final.append(mem)

    Q_final_vals = np.array(Q_final_vals)
    n_comm_vals = np.array(n_comm_vals, dtype=float)
    std_reduction = Q_base_std / Q_final_vals.std() if Q_final_vals.std() > 0 else 1.0
    stability_final = pairwise_nmi(partitions_final)

    return {
        'Q_final_mean': float(Q_final_vals.mean()),
        'Q_final_std': float(Q_final_vals.std()),
        'Q_final_min': float(Q_final_vals.min()),
        'Q_final_max': float(Q_final_vals.max()),
        'n_comm_final_mean': float(n_comm_vals.mean()),
        'n_comm_final_std': float(n_comm_vals.std()),
        'delta_Q_mean': float(Q_final_vals.mean() - Q_base_mean),
        'delta_Q_std_reduction': float(std_reduction),
        'worst_beats_best_baseline': bool(Q_final_vals.min() > Q_base_max),
        'runtime_mean': float(np.mean(runtimes)),
        'partition_stability_final': stability_final,
    }


# ============================================================
# Main Analysis
# ============================================================

def run_analysis():
    dataset_rows = []
    alpha_rows = []

    for ds_name in DATASETS:
        print(f"\n{'='*80}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*80}")

        G, info = load_large_dataset(ds_name)
        if G is None:
            print(f"  SKIP: could not load {ds_name}")
            continue

        row = {'dataset': ds_name}

        # Category 1
        print("  [1/6] Graph properties...")
        t0 = time.perf_counter()
        row.update(compute_graph_properties(G))
        print(f"        n={row['n_nodes']:,}  m={row['n_edges']:,}  "
              f"clustering={row['clustering_coef']:.4f}  ({time.perf_counter()-t0:.1f}s)")

        # Category 2
        print(f"  [2/6] Baseline Leiden ({N_RUNS} runs)...")
        t0 = time.perf_counter()
        base_props, mem_baseline = compute_baseline_properties(G, n_runs=N_RUNS)
        row.update(base_props)
        print(f"        Q={base_props['Q_base_mean']:.6f}+/-{base_props['Q_base_std']:.6f}  "
              f"CV={base_props['Q_base_cv']:.5f}  "
              f"stability={base_props['partition_stability']:.4f}  ({time.perf_counter()-t0:.1f}s)")

        # Category 3
        print("  [3/6] DSpar separation analysis...")
        t0 = time.perf_counter()
        dspar_props = compute_dspar_analysis(G, mem_baseline)
        row.update(dspar_props)
        print(f"        delta={dspar_props['delta']:+.6f}  Cohen_d={dspar_props['dspar_cohens_d']:.4f}  "
              f"overlap={dspar_props['dspar_overlap']:.4f}  ({time.perf_counter()-t0:.1f}s)")

        dataset_rows.append(row)

        # Per-alpha analysis (Categories 4-6)
        for alpha in ALPHAS:
            print(f"\n  --- alpha={alpha} ---")
            alpha_row = {'dataset': ds_name, 'alpha': alpha}

            # Category 4
            print(f"  [4/6] Sparsification effects...")
            t0 = time.perf_counter()
            spar_props, G_sparse = compute_sparsification_effects(G, alpha, mem_baseline)
            alpha_row.update(spar_props)
            print(f"        ret={spar_props['actual_retention']:.1%}  "
                  f"comps={spar_props['n_components']}  "
                  f"bias={spar_props['removal_bias']:.3f}  ({time.perf_counter()-t0:.1f}s)")

            # Category 5
            print(f"  [5/6] Seed quality ({N_RUNS} runs)...")
            t0 = time.perf_counter()
            seed_props = compute_seed_quality(
                G, G_sparse, mem_baseline, base_props['Q_base_mean'], n_runs=N_RUNS
            )
            alpha_row.update(seed_props)
            print(f"        transfer_loss={seed_props['Q_transfer_loss']:.6f}  "
                  f"NMI={seed_props['NMI_to_baseline_mean']:.4f}  ({time.perf_counter()-t0:.1f}s)")

            # Category 6
            print(f"  [6/6] DSpar-seeded performance ({N_RUNS} runs)...")
            t0 = time.perf_counter()
            perf_props = compute_dspar_seeded_performance(
                G, alpha, base_props['Q_base_mean'], base_props['Q_base_std'],
                base_props['Q_base_max'], n_runs=N_RUNS
            )
            alpha_row.update(perf_props)
            print(f"        dQ={perf_props['delta_Q_mean']:+.6f}  "
                  f"sig_red={perf_props['delta_Q_std_reduction']:.2f}x  "
                  f"beats_best={perf_props['worst_beats_best_baseline']}  ({time.perf_counter()-t0:.1f}s)")

            alpha_rows.append(alpha_row)

        # Save incrementally
        pd.DataFrame(dataset_rows).to_csv(OUTPUT_DIR / "comprehensive_properties.csv", index=False)
        pd.DataFrame(alpha_rows).to_csv(OUTPUT_DIR / "comprehensive_alpha_results.csv", index=False)
        print(f"\n  Saved incrementally.")

    return pd.DataFrame(dataset_rows), pd.DataFrame(alpha_rows)


# ============================================================
# Figures
# ============================================================

def create_figures(df_props, df_alpha):
    if not HAS_MPL:
        print("matplotlib not available, skipping figures")
        return

    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    df_a1 = df_alpha[df_alpha['alpha'] == 1.0].copy().reset_index(drop=True)
    df_m = df_props.merge(df_a1, on='dataset', how='inner').reset_index(drop=True)

    if len(df_m) == 0:
        print("  No data for figures")
        return

    # --- Figure 1: delta vs dQ ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df_m['delta'], df_m['delta_Q_mean'], s=80, c='steelblue', edgecolors='black', zorder=3)
    for i, r in df_m.iterrows():
        ax.annotate(r['dataset'], (r['delta'], r['delta_Q_mean']),
                    fontsize=7, ha='left', va='bottom', xytext=(4, 4), textcoords='offset points')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$\delta$ (DSpar separation: $\mu_{intra} - \mu_{inter}$)', fontsize=12)
    ax.set_ylabel(r'$\Delta Q$ (DSpar-seeded $-$ Baseline)', fontsize=12)
    ax.set_title(r'DSpar Separation ($\delta$) vs Modularity Improvement ($\Delta Q$) at $\alpha=1.0$', fontsize=13)
    ax.grid(True, alpha=0.3)
    if len(df_m) > 2:
        r_val, p_val = sp_stats.pearsonr(df_m['delta'], df_m['delta_Q_mean'])
        ax.text(0.05, 0.95, f'r={r_val:.3f}, p={p_val:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(fig_dir / 'delta_vs_dQ.pdf', dpi=150)
    plt.savefig(fig_dir / 'delta_vs_dQ.png', dpi=150)
    plt.close()
    print(f"  Saved: delta_vs_dQ.pdf/png")

    # --- Figure 2: Feature correlations ---
    features = [
        ('delta', r'$\delta$'),
        ('dspar_cohens_d', "Cohen's d"),
        ('dspar_overlap', 'Distribution Overlap'),
        ('Q_base_cv', 'Q Coefficient of Variation'),
        ('partition_stability', 'Partition Stability (NMI)'),
        ('hb_ratio', 'Hub-Bridge Ratio'),
    ]
    valid_features = [(f, l) for f, l in features if f in df_m.columns and df_m[f].notna().sum() > 2]

    if len(valid_features) > 0:
        n_feat = len(valid_features)
        ncols = min(3, n_feat)
        nrows = (n_feat + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
        if n_feat == 1:
            axes = [axes]
        else:
            axes = axes.flat

        for idx, (feat, label) in enumerate(valid_features):
            ax = axes[idx]
            colors = ['green' if dq > 0 else 'red' for dq in df_m['delta_Q_mean']]
            ax.scatter(df_m[feat], df_m['delta_Q_mean'], c=colors, s=60, edgecolors='black', alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel(r'$\Delta Q$', fontsize=10)
            ax.grid(True, alpha=0.3)
            valid = df_m[[feat, 'delta_Q_mean']].dropna()
            if len(valid) > 2:
                r_val, p_val = sp_stats.pearsonr(valid[feat], valid['delta_Q_mean'])
                ax.set_title(f'r={r_val:.3f}, p={p_val:.3f}', fontsize=10)

        # Hide extra axes
        for idx in range(len(valid_features), len(list(axes))):
            axes[idx].set_visible(False)

        plt.suptitle(r'Feature Correlations with $\Delta Q$ ($\alpha=1.0$)', fontsize=14)
        plt.tight_layout()
        plt.savefig(fig_dir / 'feature_correlations.pdf', dpi=150)
        plt.savefig(fig_dir / 'feature_correlations.png', dpi=150)
        plt.close()
        print(f"  Saved: feature_correlations.pdf/png")

    # --- Figure 3: Multivariate predictor ---
    if HAS_SKLEARN and len(df_m) >= 5:
        feature_cols = ['delta', 'dspar_cohens_d', 'dspar_overlap', 'Q_base_cv',
                        'partition_stability', 'hb_ratio', 'clustering_coef',
                        'degree_gini', 'bridge_fraction', 'assortativity',
                        'comm_size_gini', 'degree_skew']
        valid_cols = [c for c in feature_cols if c in df_m.columns and df_m[c].notna().sum() > 0]

        X = df_m[valid_cols].fillna(0).values
        y = df_m['delta_Q_mean'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y)
        y_pred = ridge.predict(X_scaled)
        r2 = r2_score(y, y_pred)

        # LOO cross-validation
        loo_errors = []
        for i in range(len(X)):
            X_tr = np.delete(X_scaled, i, axis=0)
            y_tr = np.delete(y, i)
            m_loo = Ridge(alpha=1.0)
            m_loo.fit(X_tr, y_tr)
            loo_errors.append((y[i] - m_loo.predict(X_scaled[i:i+1])[0]) ** 2)
        loo_r2 = 1 - np.mean(loo_errors) / np.var(y) if np.var(y) > 0 else 0

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(y, y_pred, s=60, c='steelblue', edgecolors='black')
        for i in range(len(df_m)):
            ax.annotate(df_m.iloc[i]['dataset'], (y[i], y_pred[i]),
                        fontsize=6, ha='left', va='bottom', xytext=(3, 3), textcoords='offset points')
        lims = [min(y.min(), y_pred.min()) - 0.001, max(y.max(), y_pred.max()) + 0.001]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel(r'Actual $\Delta Q$', fontsize=11)
        ax.set_ylabel(r'Predicted $\Delta Q$', fontsize=11)
        ax.set_title(f'Ridge: R²={r2:.3f} (LOO R²={loo_r2:.3f})', fontsize=12)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        coef_order = np.argsort(np.abs(ridge.coef_))[::-1]
        ax.barh([valid_cols[i] for i in coef_order], ridge.coef_[coef_order],
                color='steelblue', edgecolor='black')
        ax.set_xlabel('Ridge Coefficient (standardized)', fontsize=11)
        ax.set_title('Feature Importance', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(fig_dir / 'multivariate_predictor.pdf', dpi=150)
        plt.savefig(fig_dir / 'multivariate_predictor.png', dpi=150)
        plt.close()
        print(f"  Saved: multivariate_predictor.pdf/png (R²={r2:.3f}, LOO R²={loo_r2:.3f})")

    # --- Figure 4: Alpha comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for ds in df_alpha['dataset'].unique():
        df_d = df_alpha[df_alpha['dataset'] == ds].sort_values('alpha')
        ax.plot(df_d['alpha'], df_d['delta_Q_mean'], 'o-', label=ds, markersize=4)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel(r'$\alpha$', fontsize=11)
    ax.set_ylabel(r'$\Delta Q$', fontsize=11)
    ax.set_title(r'$\Delta Q$ vs Retention Parameter $\alpha$', fontsize=12)
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for ds in df_alpha['dataset'].unique():
        df_d = df_alpha[df_alpha['dataset'] == ds].sort_values('alpha')
        ax.plot(df_d['alpha'], df_d['delta_Q_std_reduction'], 'o-', label=ds, markersize=4)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$\alpha$', fontsize=11)
    ax.set_ylabel(r'$\sigma_{base} / \sigma_{dspar}$', fontsize=11)
    ax.set_title(r'Variance Reduction Ratio vs $\alpha$', fontsize=12)
    ax.legend(fontsize=6, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'alpha_comparison.pdf', dpi=150)
    plt.savefig(fig_dir / 'alpha_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: alpha_comparison.pdf/png")

    # --- Figure 5: Failure taxonomy (PCA) ---
    if HAS_SKLEARN and len(df_m) >= 5:
        tax_cols = ['delta', 'dspar_cohens_d', 'Q_base_cv',
                    'partition_stability', 'hb_ratio', 'bridge_fraction']
        valid_tax = [c for c in tax_cols if c in df_m.columns and df_m[c].notna().sum() > 0]
        X_tax = df_m[valid_tax].fillna(0).values
        X_tax_scaled = StandardScaler().fit_transform(X_tax)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_tax_scaled)
        dQ = df_m['delta_Q_mean'].values

        fig, ax = plt.subplots(figsize=(10, 7))
        vabs = max(abs(dQ.min()), abs(dQ.max()))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=dQ, cmap='RdYlGn',
                             s=100, edgecolors='black', zorder=3, vmin=-vabs, vmax=vabs)
        plt.colorbar(scatter, ax=ax, label=r'$\Delta Q$')
        for i in range(len(df_m)):
            ax.annotate(df_m.iloc[i]['dataset'], (X_pca[i, 0], X_pca[i, 1]),
                        fontsize=7, ha='left', va='bottom', xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
        ax.set_title(r'Network Taxonomy: PCA of Graph Properties, colored by $\Delta Q$', fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / 'failure_taxonomy.pdf', dpi=150)
        plt.savefig(fig_dir / 'failure_taxonomy.png', dpi=150)
        plt.close()
        print(f"  Saved: failure_taxonomy.pdf/png")

    # --- Figure 6: Seed quality vs final quality ---
    if 'Q_transfer_loss' in df_alpha.columns and 'delta_Q_mean' in df_alpha.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        for alpha in ALPHAS:
            df_a = df_alpha[df_alpha['alpha'] == alpha]
            ax.scatter(df_a['Q_transfer_loss'], df_a['delta_Q_mean'],
                       s=60, label=f'α={alpha}', edgecolors='black', alpha=0.8)
            for _, r in df_a.iterrows():
                ax.annotate(r['dataset'], (r['Q_transfer_loss'], r['delta_Q_mean']),
                            fontsize=5, ha='left', va='bottom', xytext=(3, 3), textcoords='offset points')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Q Transfer Loss (Q_base - Q_sparse_on_orig)', fontsize=10)
        ax.set_ylabel(r'$\Delta Q$', fontsize=10)
        ax.set_title('Seed Transfer Loss vs Final Improvement', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for alpha in ALPHAS:
            df_a = df_alpha[df_alpha['alpha'] == alpha]
            ax.scatter(df_a['NMI_to_baseline_mean'], df_a['delta_Q_mean'],
                       s=60, label=f'α={alpha}', edgecolors='black', alpha=0.8)
            for _, r in df_a.iterrows():
                ax.annotate(r['dataset'], (r['NMI_to_baseline_mean'], r['delta_Q_mean']),
                            fontsize=5, ha='left', va='bottom', xytext=(3, 3), textcoords='offset points')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('NMI to Baseline Partition', fontsize=10)
        ax.set_ylabel(r'$\Delta Q$', fontsize=10)
        ax.set_title('Seed Similarity to Baseline vs Final Improvement', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / 'seed_quality.pdf', dpi=150)
        plt.savefig(fig_dir / 'seed_quality.png', dpi=150)
        plt.close()
        print(f"  Saved: seed_quality.pdf/png")


# ============================================================
# Summary Table
# ============================================================

def print_summary(df_props, df_alpha):
    print(f"\n{'='*160}")
    print("COMPREHENSIVE SUMMARY TABLE (alpha=1.0)")
    print(f"{'='*160}")

    df_a1 = df_alpha[df_alpha['alpha'] == 1.0]
    df_m = df_props.merge(
        df_a1[['dataset', 'delta_Q_mean', 'delta_Q_std_reduction',
               'worst_beats_best_baseline', 'Q_transfer_loss',
               'NMI_to_baseline_mean', 'actual_retention',
               'n_components', 'removal_bias', 'partition_stability_final']],
        on='dataset', how='left'
    )

    print(f"\n{'Dataset':<18} {'n':<10} {'m':<10} {'delta':<9} {'cohen_d':<9} "
          f"{'overlap':<9} {'Q_cv':<8} {'stab':<7} {'dQ':<11} {'sig_red':<8} "
          f"{'wins':<6} {'NMI_seed':<9} {'xfer_loss':<10}")
    print("-" * 134)

    for _, r in df_m.iterrows():
        wins = 'Y' if r.get('worst_beats_best_baseline', False) else 'N'
        print(f"{r['dataset']:<18} {int(r['n_nodes']):<10,} {int(r['n_edges']):<10,} "
              f"{r['delta']:<+9.4f} {r['dspar_cohens_d']:<9.4f} "
              f"{r['dspar_overlap']:<9.4f} {r['Q_base_cv']:<8.5f} "
              f"{r['partition_stability']:<7.4f} {r.get('delta_Q_mean', 0):<+11.6f} "
              f"{r.get('delta_Q_std_reduction', 1):<8.2f} {wins:<6} "
              f"{r.get('NMI_to_baseline_mean', 0):<9.4f} "
              f"{r.get('Q_transfer_loss', 0):<+10.6f}")

    # Correlation summary
    print(f"\n{'='*80}")
    print("PEARSON CORRELATIONS: feature vs delta_Q (alpha=1.0)")
    print(f"{'='*80}")
    corr_features = ['delta', 'dspar_cohens_d', 'dspar_overlap', 'Q_base_cv',
                     'partition_stability', 'hb_ratio', 'clustering_coef',
                     'degree_gini', 'bridge_fraction', 'assortativity',
                     'comm_size_gini', 'degree_skew']
    dQ = df_m.get('delta_Q_mean')
    if dQ is not None and len(dQ.dropna()) > 2:
        for feat in corr_features:
            if feat in df_m.columns:
                valid = df_m[[feat, 'delta_Q_mean']].dropna()
                if len(valid) > 2:
                    r_val, p_val = sp_stats.pearsonr(valid[feat], valid['delta_Q_mean'])
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                    print(f"  {feat:<25} r={r_val:+.4f}  p={p_val:.4f} {sig}")


if __name__ == "__main__":
    df_props, df_alpha = run_analysis()

    if len(df_props) > 0:
        print_summary(df_props, df_alpha)
        print("\nCreating figures...")
        create_figures(df_props, df_alpha)
        print(f"\nAll results saved to: {OUTPUT_DIR}")
