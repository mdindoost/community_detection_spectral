#!/usr/bin/env python
"""
Task 1: cit-HepTh alpha-sweep on the LARGEST CONNECTED COMPONENT.

Pipeline is bit-for-bit PAPER_RESTRUCTURE/exp_I_lcc_tables/run.py (the script
that produced tables/tab_exp1_modularity_lcc.tex and
tables/tab_exp1_decomposition_lcc.tex), extended from the single alpha=0.8 cell
to the full alpha grid of PAPER_EXPERIMENTS/exp1_2_theoretical_predictions.py:

    simple undirected graph -> LCC
    fixed partition P = leidenalg.ModularityVertexPartition, rng seed 42, n_iterations 2
    sparsifier = experiments/dspar.py::dspar_sparsify(method="paper")   [WITH replacement]
    weights dropped; Q evaluated on the sparsified graph
    alpha grid  = np.linspace(0.3, 1.0, 15) EXCLUDING 1.0   -> 14 values
    10 sparsification seeds per alpha, seed = round(alpha*1e5) + run
    (at alpha = 0.8 this is 80000..80009, i.e. exactly the table's seeds)

Outputs (this directory):
    cit-HepTh_lcc_sweep_raw.csv   per-(alpha, seed) rows
    cit-HepTh_lcc_sweep.csv       per-alpha mean/std summary (figure input)
"""
import sys
import time
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg as la
import networkx as nx
import pandas as pd

REPO = Path("/home/md724/community_detection_spectral")
sys.path.insert(0, str(REPO))
from experiments.dspar import dspar_sparsify  # noqa: E402

OUT = Path(__file__).resolve().parent
DATASET = "cit-HepTh"
PATH = REPO / "datasets/cit-HepTh/cit-HepTh.txt"

ALPHAS = [a for a in np.linspace(0.3, 1.0, 15) if not np.isclose(a, 1.0)]
N_SEEDS = 10
LEIDEN_SEED = 42
N_ITER = 2


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
            nodes.add(u); nodes.add(v)
    node_list = sorted(nodes)
    idx = {o: i for i, o in enumerate(node_list)}
    g_full = ig.Graph(n=len(node_list), edges=[(idx[u], idx[v]) for u, v in edges],
                      directed=False)
    g_full.simplify(multiple=True, loops=True)
    return g_full.connected_components().giant(), g_full.vcount(), g_full.ecount()


def leiden(g, seed):
    part = la.ModularityVertexPartition(g)
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    return list(part.membership), float(part.modularity)


def g_term(memb, deg, m, n_comm):
    if m == 0:
        return 0.0
    vol = np.bincount(memb, weights=deg, minlength=n_comm)
    return float(np.dot(vol, vol) / (4.0 * m * m))


def main():
    g, n_full, m_full = load_lcc(PATH)
    n, m = g.vcount(), g.ecount()
    print(f"{DATASET}: full n={n_full:,} m={m_full:,} | LCC n={n:,} m={m:,}", flush=True)

    memb, _ = leiden(g, LEIDEN_SEED)
    memb = np.asarray(memb, dtype=np.int64)
    n_comm = int(memb.max()) + 1
    Q_orig = float(g.modularity(memb.tolist()))
    print(f"fixed partition: k={n_comm:,}  Q_orig={Q_orig:.6f}", flush=True)

    E = np.asarray([(min(u, v), max(u, v)) for u, v in g.get_edgelist()], dtype=np.int64)
    deg = np.asarray(g.degree(), dtype=np.float64)
    scores = 1.0 / deg[E[:, 0]] + 1.0 / deg[E[:, 1]]
    intra = memb[E[:, 0]] == memb[E[:, 1]]
    n1, n2 = int(intra.sum()), m - int(intra.sum())
    mu_intra, mu_inter = float(scores[intra].mean()), float(scores[~intra].mean())
    delta = mu_intra - mu_inter
    F_orig = n1 / m
    G_orig = g_term(memb, deg, m, n_comm)
    print(f"n1={n1:,} n2={n2:,} delta={delta:.6e}  F_orig={F_orig:.6f} G_orig={G_orig:.6f} "
          f"(F-G={F_orig-G_orig:.6f})", flush=True)

    edge_index = {(int(a), int(b)): i for i, (a, b) in enumerate(E)}
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(n))
    G_nx.add_edges_from((int(a), int(b)) for a, b in E)

    rows = []
    raw_path = OUT / f"{DATASET}_lcc_sweep_raw.csv"
    for alpha in ALPHAS:
        t0 = time.perf_counter()
        base = int(round(float(alpha) * 100000))
        for run in range(N_SEEDS):
            seed = base + run
            Gw = dspar_sparsify(G_nx, retention=float(alpha), method="paper", seed=seed)
            kept = np.zeros(m, dtype=bool)
            for u, v in Gw.edges():                       # weights dropped
                kept[edge_index[(min(u, v), max(u, v))]] = True
            m_sp = int(kept.sum())
            Esp = E[kept]
            gs = ig.Graph(n=n, edges=[tuple(map(int, e)) for e in Esp], directed=False)

            Q_fixed_sparse = float(gs.modularity(memb.tolist()))
            deg_sp = np.asarray(gs.degree(), dtype=np.float64)
            G_sp = g_term(memb, deg_sp, m_sp, n_comm)
            pres_intra = int(kept[intra].sum())
            pres_inter = int(kept[~intra].sum())
            F_sp = pres_intra / m_sp
            dF, dG = F_sp - F_orig, G_sp - G_orig
            dQ_fixed = Q_fixed_sparse - Q_orig
            _, Q_leiden_sp = leiden(gs, seed)

            rows.append(dict(
                dataset=DATASET, alpha=float(alpha), seed=seed,
                n_lcc=n, m_lcc=m, n_comm=n_comm, Q_orig=Q_orig,
                m_sparse=m_sp, realized_retention=m_sp / m,
                Q_fixed_sparse=Q_fixed_sparse, dQ_fixed=dQ_fixed,
                Q_leiden_sparse=Q_leiden_sp, dQ_leiden_sparse=Q_leiden_sp - Q_orig,
                F_orig=F_orig, F_sparse=F_sp, dF_obs=dF,
                G_orig=G_orig, G_sparse=G_sp, dG_obs=dG,
                dQ_reconstructed=dF - dG, recon_abs_err=abs((dF - dG) - dQ_fixed),
                intra_rate=pres_intra / n1, inter_rate=pres_inter / n2,
                ratio_observed=(pres_inter / n2) / (pres_intra / n1),
                ratio_predicted=mu_inter / mu_intra,
                mu_intra=mu_intra, mu_inter=mu_inter, delta=delta, n1=n1, n2=n2,
            ))
        pd.DataFrame(rows).to_csv(raw_path, index=False)
        last = rows[-1]
        print(f"  alpha={alpha:.2f} seeds {base}..{base+N_SEEDS-1}  "
              f"ret={np.mean([r['realized_retention'] for r in rows[-N_SEEDS:]]):.4f}  "
              f"dQ_fixed={np.mean([r['dQ_fixed'] for r in rows[-N_SEEDS:]]):+.6f}  "
              f"dQ_Leiden={np.mean([r['dQ_leiden_sparse'] for r in rows[-N_SEEDS:]]):+.6f}  "
              f"|recon|max={max(r['recon_abs_err'] for r in rows[-N_SEEDS:]):.2e}  "
              f"[{time.perf_counter()-t0:.1f}s]", flush=True)

    df = pd.DataFrame(rows)
    cols = ["dQ_fixed", "dQ_leiden_sparse", "dG_obs", "dF_obs",
            "realized_retention", "ratio_observed", "m_sparse"]
    agg = df.groupby("alpha")[cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    # column aliases kept for the plotting script / historical naming
    agg["retention"] = agg["alpha"]
    agg["modularity_fixed_change_mean"] = agg["dQ_fixed_mean"]
    agg["modularity_fixed_change_std"] = agg["dQ_fixed_std"]
    agg["modularity_leiden_change_mean"] = agg["dQ_leiden_sparse_mean"]
    agg["modularity_leiden_change_std"] = agg["dQ_leiden_sparse_std"]
    agg["dG_observed_mean"] = agg["dG_obs_mean"]
    agg["dG_observed_std"] = agg["dG_obs_std"]
    agg["Q_orig"] = Q_orig
    agg["n_lcc"] = n
    agg["m_lcc"] = m
    agg["n_seeds"] = N_SEEDS
    agg.to_csv(OUT / f"{DATASET}_lcc_sweep.csv", index=False)
    print("\nWrote", OUT / f"{DATASET}_lcc_sweep.csv")
    print(agg[["alpha", "realized_retention_mean", "modularity_fixed_change_mean",
               "modularity_leiden_change_mean", "dG_observed_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
