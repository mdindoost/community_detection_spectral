"""
Experiment 1.2: Theoretical Predictions Validation (FIXED)

This script separates:
(A) THEORY VALIDATION: all metrics computed w.r.t. a FIXED partition (membership from original graph)
(B) PIPELINE EFFECT: re-run Leiden on sparsified graph as a downstream metric

Fixes vs prior version:
- No meaningless correlation with constant predictions (use MAE/RMSE + CI)
- Compute modularity on sparsified graph using FIXED membership (theory-aligned)
- Correct weighted G-term normalization using total weight W, not m
- Increase N_RUNS for meaningful stats
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import networkx as nx
import igraph as ig
import pandas as pd
# scipy.stats.pearsonr removed - not meaningful for constant predictions
import matplotlib.pyplot as plt

from experiments.dspar import dspar_sparsify
from experiments.utils import load_snap_dataset

OUTPUT_DIR = Path(__file__).parent / "results" / "exp1_2_theoretical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RETENTION_VALUES = np.linspace(0.3, 1.0, 15)
N_RUNS = 10  # IMPORTANT: needs >1 for academic reporting


def load_dataset(name):
    # Map friendly names to SNAP dataset names
    name_map = {
        "facebook": "facebook-combined",
        "facebook-combined": "facebook-combined",
        "ego-Facebook": "ego-Facebook",
        "cit-HepPh": "cit-HepPh",
        "cit-HepTh": "cit-HepTh",
        "ca-HepPh": "ca-HepPh",
        "ca-HepTh": "ca-HepTh",
        "ca-AstroPh": "ca-AstroPh",
        "ca-CondMat": "ca-CondMat",
        "ca-GrQc": "ca-GrQc",
        "email-Enron": "email-Enron",
        "email-Eu-core": "email-Eu-core",
        "wiki-Vote": "wiki-Vote",
    }
    snap_name = name_map.get(name, name)
    edges, n_nodes, _ground_truth = load_snap_dataset(snap_name)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    G.add_edges_from(edge_set)
    return G


def nx_to_igraph(G: nx.Graph) -> ig.Graph:
    edges = list(G.edges())
    return ig.Graph(n=G.number_of_nodes(), edges=edges, directed=False)


def leiden_partition(G: nx.Graph):
    ig_graph = nx_to_igraph(G)
    part = ig_graph.community_leiden(objective_function="modularity")
    return part.membership, part.modularity


def modularity_fixed_membership(G: nx.Graph, membership):
    """Compute modularity for a FIXED membership on graph G."""
    ig_graph = nx_to_igraph(G)
    return ig_graph.modularity(membership)


def compute_dspar_scores(G: nx.Graph):
    deg = dict(G.degree())
    scores = {}
    for u, v in G.edges():
        s = 1.0 / deg[u] + 1.0 / deg[v]
        scores[(u, v)] = s
        scores[(v, u)] = s
    all_vals = list(scores.values())
    mean_score = float(np.mean(all_vals)) if all_vals else 1.0
    return scores, mean_score


def classify_edges(G: nx.Graph, membership):
    intra, inter = [], []
    for u, v in G.edges():
        if membership[u] == membership[v]:
            intra.append((u, v))
        else:
            inter.append((u, v))
    return intra, inter


def compute_G_term_unweighted(G: nx.Graph, membership):
    """
    G(G) = sum_c vol_c(G)^2 / (4 * m(G)^2)
    where vol_c is sum of degrees of nodes in community c.
    This is the modularity null-model penalty term for a FIXED partition.
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0

    degrees = dict(G.degree())
    communities = set(membership)

    vol = {c: 0.0 for c in communities}
    for node, deg in degrees.items():
        vol[membership[node]] += float(deg)

    G_val = sum(v * v for v in vol.values()) / (4.0 * m * m)
    return float(G_val)


def compute_theory(G: nx.Graph, membership):
    """
    Compute theoretical quantities w.r.t. FIXED membership.
    """
    scores, mean_score = compute_dspar_scores(G)
    intra_edges, inter_edges = classify_edges(G, membership)

    n1 = len(intra_edges)
    n2 = len(inter_edges)
    m = n1 + n2

    # Means
    mu_intra = float(np.mean([scores[e] for e in intra_edges])) if n1 > 0 else 0.0
    mu_inter = float(np.mean([scores[e] for e in inter_edges])) if n2 > 0 else 0.0
    delta = mu_intra - mu_inter

    # Unweighted F
    F_original = (n1 / m) if m > 0 else 0.0

    # Weighted F (exact, not via n1*mu)
    S_intra = float(np.sum([scores[e] for e in intra_edges])) if n1 > 0 else 0.0
    S_inter = float(np.sum([scores[e] for e in inter_edges])) if n2 > 0 else 0.0
    S_total = S_intra + S_inter
    F_weighted_exact = (S_intra / S_total) if S_total > 0 else 0.0

    # Theorem-2 predicted ΔF (your formula; keep it but guard)
    denom = (m * (n1 * mu_intra + n2 * mu_inter))
    if m > 0 and (n1 * mu_intra + n2 * mu_inter) > 0:
        dF_pred = (n1 * n2 * delta) / denom
    else:
        dF_pred = 0.0

    # Theorem-1 predicted ratio
    ratio_pred = (mu_inter / mu_intra) if mu_intra > 0 else 1.0

    # ---- G-term (fixed membership) ----
    # Unweighted: W = m
    W_unw = float(m) if m > 0 else 1.0

    deg = dict(G.degree())
    communities = set(membership)

    # volume per community (sum degrees)
    vol_unw = {}
    for c in communities:
        vol_unw[c] = sum(deg[i] for i in range(len(membership)) if membership[i] == c)

    # G = sum(vol_c^2) / (4 W^2), where W=m for unweighted
    G_unw = sum(v * v for v in vol_unw.values()) / (4.0 * W_unw * W_unw)

    # Weighted graph induced by edge weights w_e = score/mean_score
    # Total weight W_w = sum_e w_e (undirected, count each edge once)
    W_w = 0.0
    for u, v in G.edges():
        W_w += scores[(u, v)] / mean_score
    W_w = float(W_w) if W_w > 0 else 1.0

    # Weighted volume per community = sum weighted degrees in that community
    # weighted degree of node = sum_{nbr} w_(node,nbr)
    vol_w = {c: 0.0 for c in communities}
    for node in range(len(membership)):
        c = membership[node]
        wd = 0.0
        for nbr in G.neighbors(node):
            wd += scores[(node, nbr)] / mean_score
        vol_w[c] += wd

    G_w = sum(v * v for v in vol_w.values()) / (4.0 * W_w * W_w)
    dG = G_w - G_unw

    # “Lower bound” style prediction
    dQ_lb = dF_pred - dG

    return {
        "n1": n1, "n2": n2, "m": m,
        "mu_intra": mu_intra, "mu_inter": mu_inter, "delta": delta,
        "ratio_predicted": ratio_pred,
        "F_original": F_original,
        "F_weighted_exact": F_weighted_exact,
        "F_improvement_predicted": dF_pred,
        "W_unweighted": W_unw, "W_weighted": W_w,
        "G_original": G_unw, "G_weighted": G_w, "G_change": dG,
        "modularity_improvement_lower_bound": dQ_lb
    }


def compute_observed_fixed_membership(G: nx.Graph, G_sparse: nx.Graph, membership):
    """
    Observations on sparsified graph w.r.t. FIXED membership.
    """
    intra_edges, inter_edges = classify_edges(G, membership)
    n1 = len(intra_edges)
    n2 = len(inter_edges)

    sparse_edge_set = set((min(u, v), max(u, v)) for u, v in G_sparse.edges())

    preserved_intra = sum(1 for u, v in intra_edges if (min(u, v), max(u, v)) in sparse_edge_set)
    preserved_inter = sum(1 for u, v in inter_edges if (min(u, v), max(u, v)) in sparse_edge_set)

    intra_rate = preserved_intra / n1 if n1 > 0 else 1.0
    inter_rate = preserved_inter / n2 if n2 > 0 else 1.0
    ratio_obs = (inter_rate / intra_rate) if intra_rate > 0 else float("inf")

    m_sparse = G_sparse.number_of_edges()
    F_obs = (preserved_intra / m_sparse) if m_sparse > 0 else 0.0

    # modularity for fixed membership on sparse graph (THEORY-ALIGNED)
    Q_fixed_sparse = modularity_fixed_membership(G_sparse, membership)

    return {
        "preserved_intra": preserved_intra,
        "preserved_inter": preserved_inter,
        "intra_rate": intra_rate,
        "inter_rate": inter_rate,
        "ratio_observed": ratio_obs,
        "m_sparse": m_sparse,
        "F_observed": F_obs,
        "modularity_fixed_sparse": Q_fixed_sparse,
    }


def run_single_trial(G, membership_fixed, Q_fixed_original, retention, seed):
    theory = compute_theory(G, membership_fixed)

    if retention == 1.0:
        G_sparse = G.copy()
    else:
        G_sparse_weighted = dspar_sparsify(
            G, retention=retention, method="paper", seed=seed
        )
        # Convert to unweighted graph for Leiden (keep only topology)
        G_sparse = nx.Graph()
        G_sparse.add_nodes_from(G_sparse_weighted.nodes())
        G_sparse.add_edges_from(G_sparse_weighted.edges())

    obs_fixed = compute_observed_fixed_membership(G, G_sparse, membership_fixed)

    # Downstream: re-run Leiden on sparse graph (PIPELINE METRIC)
    # Note: Leiden runs on unweighted graph
    membership_sparse, Q_leiden_sparse = leiden_partition(G_sparse)

    out = {
        "retention": retention,
        "seed": seed,

        # fixed baseline modularity (membership from original graph)
        "modularity_fixed_original": Q_fixed_original,

        # theory
        **theory,

        # fixed-membership observations
        **obs_fixed,

        # downstream pipeline modularity
        "modularity_leiden_sparse": Q_leiden_sparse,
    }

    # Derived: theory-aligned deltas
    out["F_improvement_observed"] = out["F_observed"] - out["F_original"]
    out["ratio_prediction_error"] = abs(out["ratio_observed"] - out["ratio_predicted"])
    out["F_prediction_error"] = abs(out["F_improvement_observed"] - out["F_improvement_predicted"])

    out["modularity_fixed_change"] = out["modularity_fixed_sparse"] - Q_fixed_original
    out["modularity_leiden_change"] = out["modularity_leiden_sparse"] - Q_fixed_original

    # --- Observed ΔG on sparsified graph (fixed membership) ---
    G_orig = theory["G_original"]  # computed on original graph with fixed membership
    G_sparse_observed = compute_G_term_unweighted(G_sparse, membership_fixed)
    dG_observed = G_sparse_observed - G_orig

    out["G_sparse_observed"] = G_sparse_observed
    out["dG_observed"] = dG_observed

    # --- Reconstruct ΔQ via ΔF - ΔG (should match fixed-membership modularity change) ---
    out["dQ_reconstructed"] = out["F_improvement_observed"] - dG_observed

    return out


def summarize_by_retention(df: pd.DataFrame, cols):
    agg = df.groupby("retention")[cols].agg(["mean", "std"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg.reset_index()


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cit-HepPh"
    print("=" * 100)
    print(f"EXP 1.2 (FIXED): THEORETICAL VALIDATION — {dataset.upper()}")
    print("=" * 100)

    print("\nLoading dataset...")
    G = load_dataset(dataset)
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print("\nLeiden on ORIGINAL graph (defines fixed membership)...")
    membership_fixed, Q_leiden_original = leiden_partition(G)

    # Theory-aligned baseline modularity uses the SAME membership on the SAME graph
    Q_fixed_original = modularity_fixed_membership(G, membership_fixed)
    print(f"Communities: {len(set(membership_fixed)):,}")
    print(f"Q(original, fixed membership) = {Q_fixed_original:.6f}")
    print(f"Q(original, Leiden reported)  = {Q_leiden_original:.6f}")

    baseline_theory = compute_theory(G, membership_fixed)
    print("\nBaseline theory:")
    print(f"  n1={baseline_theory['n1']:,}  n2={baseline_theory['n2']:,}  m={baseline_theory['m']:,}")
    print(f"  mu_intra={baseline_theory['mu_intra']:.6e}  mu_inter={baseline_theory['mu_inter']:.6e}")
    print(f"  delta={baseline_theory['delta']:.6e}")
    print(f"  ratio_predicted={baseline_theory['ratio_predicted']:.6f}")
    print(f"  G_change={baseline_theory['G_change']:.6e}  (<=0 good)")
    print(f"  dQ_lower_bound={baseline_theory['modularity_improvement_lower_bound']:.6e}")

    results = []
    trial_total = len(RETENTION_VALUES) * N_RUNS
    t = 0

    print(f"\nRunning {trial_total} trials ({len(RETENTION_VALUES)} retentions × {N_RUNS} runs)...")
    for retention in RETENTION_VALUES:
        for run in range(N_RUNS):
            seed = int(retention * 100000) + run
            t += 1
            print(f"\rTrial {t}/{trial_total} retention={retention:.3f} run={run+1}/{N_RUNS}", end="", flush=True)

            results.append(run_single_trial(G, membership_fixed, Q_fixed_original, retention, seed))

    print("\n\nDone.")

    df = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / f"{dataset}_theoretical_validation_FIXED.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved raw results: {out_csv}")

    # --- THEOREM 1: ratio ---
    print("\n" + "=" * 100)
    print("THEOREM 1: Preservation Ratio (fixed membership)")
    print("=" * 100)
    ratio_mae = df["ratio_prediction_error"].mean()
    ratio_rmse = np.sqrt((df["ratio_prediction_error"] ** 2).mean())
    print(f"Predicted ratio (constant for dataset): {baseline_theory['ratio_predicted']:.6f}")
    print(f"MAE={ratio_mae:.6f}  RMSE={ratio_rmse:.6f}")

    ratio_ret = summarize_by_retention(df, ["ratio_observed"])
    print("\nBy retention (observed ratio mean±std):")
    print(ratio_ret[["retention", "ratio_observed_mean", "ratio_observed_std"]].to_string(index=False))

    # --- THEOREM 2: ΔF ---
    print("\n" + "=" * 100)
    print("THEOREM 2: ΔF Prediction (fixed membership)")
    print("=" * 100)

    # NOTE: F_improvement_predicted is constant per dataset, so Pearson correlation is undefined.
    # We validate via MAE/RMSE + sign consistency instead.
    mae_F = df["F_prediction_error"].mean()
    rmse_F = np.sqrt((df["F_prediction_error"] ** 2).mean())
    pred_dF = float(df["F_improvement_predicted"].iloc[0])

    sign_consistency = (np.sign(df["F_improvement_observed"]) == np.sign(pred_dF)).mean() * 100.0

    print(f"Predicted ΔF (constant): {pred_dF:.6e}")
    print(f"MAE(ΔF)  = {mae_F:.6e}")
    print(f"RMSE(ΔF) = {rmse_F:.6e}")
    print(f"Sign consistency vs predicted ΔF: {sign_consistency:.1f}%")

    # --- ΔQ bound vs observed (theory-aligned modularity on fixed membership) ---
    print("\n" + "=" * 100)
    print("ΔQ = ΔF − ΔG (theory-aligned, fixed membership modularity)")
    print("=" * 100)
    print(f"Predicted lower bound ΔQ: {baseline_theory['modularity_improvement_lower_bound']:.6e}")

    mod_ret = summarize_by_retention(df, ["modularity_fixed_change", "modularity_leiden_change", "dQ_reconstructed", "dG_observed"])
    print("\nBy retention:")
    print(mod_ret[[
        "retention",
        "modularity_fixed_change_mean", "modularity_fixed_change_std",
        "modularity_leiden_change_mean", "modularity_leiden_change_std",
    ]].to_string(index=False))

    # --- ΔQ RECONSTRUCTION CHECK ---
    # Compare reconstructed ΔQ (= ΔF - ΔG) to actual fixed-membership modularity change
    # These should match up to small numerical noise if F and G are computed consistently.
    print("\n" + "=" * 100)
    print("ΔQ RECONSTRUCTION CHECK: ΔQ_reconstructed = ΔF_obs - ΔG_obs vs modularity_fixed_change")
    print("=" * 100)

    df["dQ_reconstruction_error"] = (df["dQ_reconstructed"] - df["modularity_fixed_change"]).abs()

    print(f"Mean |ΔQ_rec - ΔQ_fixed| = {df['dQ_reconstruction_error'].mean():.6e}")
    print(f"Max  |ΔQ_rec - ΔQ_fixed| = {df['dQ_reconstruction_error'].max():.6e}")

    if df["dQ_reconstruction_error"].mean() < 1e-6:
        print("✓ Reconstruction matches perfectly (error < 1e-6)")
    elif df["dQ_reconstruction_error"].mean() < 1e-3:
        print("✓ Reconstruction matches well (error < 1e-3)")
    else:
        print("⚠ Large reconstruction error - check F/G definitions for consistency")

    # Show dG_observed variation by retention
    print("\nObserved ΔG by retention:")
    print(mod_ret[["retention", "dG_observed_mean", "dG_observed_std"]].to_string(index=False))

    # Save plot-ready summary
    summary_csv = OUTPUT_DIR / f"{dataset}_summary_FIXED.csv"
    mod_ret.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")

    # Minimal plots (optional)
    try:
        plt.figure()
        plt.errorbar(
            ratio_ret["retention"],
            ratio_ret["ratio_observed_mean"],
            yerr=ratio_ret["ratio_observed_std"],
            fmt="o-",
        )
        plt.axhline(baseline_theory["ratio_predicted"], linestyle="--")
        plt.xlabel("Retention α")
        plt.ylabel("Observed ratio (inter_rate / intra_rate)")
        plt.title(f"{dataset}: Theorem 1 ratio validation")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{dataset}_ratio_validation.png", dpi=200)
        plt.close()

        plt.figure()
        plt.errorbar(
            mod_ret["retention"],
            mod_ret["modularity_fixed_change_mean"],
            yerr=mod_ret["modularity_fixed_change_std"],
            fmt="o-",
            label="ΔQ fixed membership"
        )
        plt.errorbar(
            mod_ret["retention"],
            mod_ret["modularity_leiden_change_mean"],
            yerr=mod_ret["modularity_leiden_change_std"],
            fmt="o-",
            label="ΔQ Leiden re-optimized"
        )
        plt.axhline(baseline_theory["modularity_improvement_lower_bound"], linestyle="--", label="Pred. lower bound")
        plt.xlabel("Retention α")
        plt.ylabel("ΔQ vs original (baseline fixed membership)")
        plt.title(f"{dataset}: Modularity change")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{dataset}_modularity_change.png", dpi=200)
        plt.close()

        print("Saved plots: ratio_validation.png, modularity_change.png")

    except Exception as e:
        print(f"Plotting skipped due to error: {e}")


if __name__ == "__main__":
    main()
