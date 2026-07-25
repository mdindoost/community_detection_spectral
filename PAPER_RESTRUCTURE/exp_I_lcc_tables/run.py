#!/usr/bin/env python
"""
Experiment I: recompute paper Tables 1-2 (fixed-partition modularity decomposition
under with-replacement DSpar at nominal alpha = 0.8) on the LARGEST CONNECTED
COMPONENT of each simple undirected graph, so that Section 5.1 uses the same
preprocessing as every other experiment in the paper.

Published Tables 1-2 (labels tab:exp1_2_modularity / tab:exp1_2_decomposition in
Paper_materials/main-v2.tex) were produced by
PAPER_EXPERIMENTS/exp1_2_theoretical_predictions.py on the FULL graph as
downloaded.  This script keeps that pipeline bit-for-bit where it matters
(experiments/dspar.py, method="paper", i.e. WITH-replacement sampling with
q = ceil(0.8 * m) draws, weights dropped afterwards) and changes exactly one
thing: the input graph is g.connected_components().giant().

Quantities per (dataset, seed), all w.r.t. the FIXED partition P found by Leiden
on the LCC of the original graph:

  Q_orig                 = Q_G(P)
  Q_fixed_sparse         = Q_{G'}(P)                      -> dQ_fixed
  dF_obs                 = n1'/m' - n1/m                  (intra fraction)
  dG_obs                 = sum_c vol'_c^2/(4 m'^2) - sum_c vol_c^2/(4 m^2)
  identity               dQ_fixed = dF_obs - dG_obs       (checked to machine eps)
  dQ_Leiden^(sp)         = Q_{G'}(Leiden(G')) - Q_orig    (same definition as the
                                                           published table)
  realized retention     = m'/m
  ratio_predicted        = mu_inter / mu_intra            (Corollary 1)
  ratio_observed         = (inter preserved rate)/(intra preserved rate)

Outputs: results.csv, table1_lcc.tex, table2_lcc.tex, SUMMARY.md
"""

import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg as la
import networkx as nx

REPO = Path("/home/md724/community_detection_spectral")
sys.path.insert(0, str(REPO))
from experiments.dspar import dspar_sparsify  # noqa: E402

OUT = Path(__file__).resolve().parent

# Same 11 networks, same order, as the published Table 1.
DATASETS = [
    ("ca-AstroPh", REPO / "datasets/ca-AstroPh/ca-AstroPh.txt"),
    ("ca-CondMat", REPO / "datasets/ca-CondMat/ca-CondMat.txt"),
    ("ca-GrQc", REPO / "datasets/ca-GrQc/ca-GrQc.txt"),
    ("ca-HepPh", REPO / "datasets/ca-HepPh/ca-HepPh.txt"),
    ("ca-HepTh", REPO / "datasets/ca-HepTh/ca-HepTh.txt"),
    ("cit-HepPh", REPO / "datasets/cit-HepPh/cit-HepPh.txt"),
    ("cit-HepTh", REPO / "datasets/cit-HepTh/cit-HepTh.txt"),
    ("email-Enron", REPO / "datasets/email-Enron/email-Enron.txt"),
    ("facebook-combined", REPO / "datasets/facebook-combined/facebook-combined.txt"),
    ("wiki-Vote", REPO / "datasets/wiki-Vote/wiki-Vote.txt"),
    ("email-Eu-core", REPO / "datasets/email-Eu-core/email-Eu-core.txt"),
]

ALPHA = 0.8
N_SEEDS = 10
# The published run used seed = int(retention * 100000) + run with
# retention = np.linspace(0.3, 1.0, 15)[10] -> 80000 + run.  Kept identical.
SEED_BASE = 80000
LEIDEN_SEED = 42          # fixed seed for the reference partition
N_ITER = 2                # repo convention (run_leiden default)


# ---------------------------------------------------------------------------
# Loading: simple undirected graph -> largest connected component
# ---------------------------------------------------------------------------
def load_lcc(path):
    edges, nodes = [], set()
    with open(path) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)
    node_list = sorted(nodes)
    idx = {o: i for i, o in enumerate(node_list)}
    g_full = ig.Graph(n=len(node_list), edges=[(idx[u], idx[v]) for u, v in edges],
                      directed=False)
    g_full.simplify(multiple=True, loops=True)
    m_full = g_full.ecount()
    n_full = g_full.vcount()
    g = g_full.connected_components().giant()
    return g, n_full, m_full


def leiden(g, seed, weights=None):
    part = la.ModularityVertexPartition(g)
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    return list(part.membership), float(part.modularity)


def g_term(memb, deg, m, n_comm):
    """sum_c vol_c^2 / (4 m^2) for a FIXED membership."""
    if m == 0:
        return 0.0
    vol = np.bincount(memb, weights=deg, minlength=n_comm)
    return float(np.dot(vol, vol) / (4.0 * m * m))


# ---------------------------------------------------------------------------
def run_dataset(name, path, rows):
    t0 = time.perf_counter()
    g, n_full, m_full = load_lcc(path)
    n, m = g.vcount(), g.ecount()
    print(f"\n{'='*78}\n{name}: full n={n_full:,} m={m_full:,}   "
          f"LCC n={n:,} m={m:,}\n{'='*78}", flush=True)

    memb, Q_leiden_reported = leiden(g, LEIDEN_SEED)
    memb = np.asarray(memb, dtype=np.int64)
    n_comm = int(memb.max()) + 1
    Q_orig = float(g.modularity(memb.tolist()))
    print(f"  Leiden(LCC): k={n_comm:,}  Q_orig={Q_orig:.6f} "
          f"(part.modularity={Q_leiden_reported:.6f})", flush=True)

    # canonical edge array of the LCC
    E = np.asarray([(min(u, v), max(u, v)) for u, v in g.get_edgelist()],
                   dtype=np.int64)
    deg = np.asarray(g.degree(), dtype=np.float64)
    scores = 1.0 / deg[E[:, 0]] + 1.0 / deg[E[:, 1]]
    intra_mask = memb[E[:, 0]] == memb[E[:, 1]]
    n1 = int(intra_mask.sum())
    n2 = m - n1
    mu_intra = float(scores[intra_mask].mean()) if n1 else 0.0
    mu_inter = float(scores[~intra_mask].mean()) if n2 else 0.0
    delta = mu_intra - mu_inter
    ratio_predicted = (mu_inter / mu_intra) if mu_intra > 0 else float("nan")
    F_orig = n1 / m
    G_orig = g_term(memb, deg, m, n_comm)
    print(f"  n1={n1:,} n2={n2:,}  mu_intra={mu_intra:.6e} mu_inter={mu_inter:.6e} "
          f"delta={delta:.6e}  ratio_pred={ratio_predicted:.6f}", flush=True)
    print(f"  F_orig={F_orig:.6f}  G_orig={G_orig:.6f}  "
          f"(check Q_orig = F-G = {F_orig - G_orig:.6f})", flush=True)

    edge_index = {(int(a), int(b)): i for i, (a, b) in enumerate(E)}

    # networkx copy of the LCC, fed to the published sparsifier verbatim
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(n))
    G_nx.add_edges_from((int(a), int(b)) for a, b in E)
    assert G_nx.number_of_edges() == m

    for run in range(N_SEEDS):
        seed = SEED_BASE + run
        Gw = dspar_sparsify(G_nx, retention=ALPHA, method="paper", seed=seed)
        kept = np.zeros(m, dtype=bool)
        for u, v in Gw.edges():                       # weights dropped here
            kept[edge_index[(min(u, v), max(u, v))]] = True
        m_sp = int(kept.sum())
        Esp = E[kept]

        gs = ig.Graph(n=n, edges=[tuple(map(int, e)) for e in Esp], directed=False)
        Q_fixed_sparse = float(gs.modularity(memb.tolist()))
        deg_sp = np.asarray(gs.degree(), dtype=np.float64)
        G_sp = g_term(memb, deg_sp, m_sp, n_comm)

        pres_intra = int(kept[intra_mask].sum())
        pres_inter = int(kept[~intra_mask].sum())
        intra_rate = pres_intra / n1 if n1 else float("nan")
        inter_rate = pres_inter / n2 if n2 else float("nan")
        ratio_observed = inter_rate / intra_rate if intra_rate else float("nan")

        F_sp = pres_intra / m_sp
        dF = F_sp - F_orig
        dG = G_sp - G_orig
        dQ_fixed = Q_fixed_sparse - Q_orig
        recon_err = abs((dF - dG) - dQ_fixed)

        _, Q_leiden_sp = leiden(gs, seed)
        dQ_leiden_sp = Q_leiden_sp - Q_orig

        rows.append(dict(
            dataset=name, n_full=n_full, m_full=m_full, n_lcc=n, m_lcc=m,
            n_comm=n_comm, seed=seed, alpha=ALPHA,
            Q_orig=Q_orig, m_sparse=m_sp, realized_retention=m_sp / m,
            Q_fixed_sparse=Q_fixed_sparse, dQ_fixed=dQ_fixed,
            F_orig=F_orig, F_sparse=F_sp, dF_obs=dF,
            G_orig=G_orig, G_sparse=G_sp, dG_obs=dG,
            dQ_reconstructed=dF - dG, recon_abs_err=recon_err,
            Q_leiden_sparse=Q_leiden_sp, dQ_leiden_sparse=dQ_leiden_sp,
            n1=n1, n2=n2, preserved_intra=pres_intra, preserved_inter=pres_inter,
            intra_rate=intra_rate, inter_rate=inter_rate,
            ratio_observed=ratio_observed, ratio_predicted=ratio_predicted,
            mu_intra=mu_intra, mu_inter=mu_inter, delta=delta,
        ))
        print(f"  seed={seed} m'={m_sp:,} ret={m_sp/m:.4f} "
              f"dQ_fixed={dQ_fixed:+.6f} dF={dF:+.6f} -dG={-dG:+.6f} "
              f"|err|={recon_err:.2e} dQ_Leiden={dQ_leiden_sp:+.6f} "
              f"ratio_obs={ratio_observed:.4f}", flush=True)
    print(f"  [{time.perf_counter()-t0:.1f}s]", flush=True)


def main():
    rows = []
    names = sys.argv[1:] if len(sys.argv) > 1 else None
    for name, path in DATASETS:
        if names and name not in names:
            continue
        run_dataset(name, path, rows)
        with open(OUT / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("\nDone -> results.csv")


if __name__ == "__main__":
    main()
