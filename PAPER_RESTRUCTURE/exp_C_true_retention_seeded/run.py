#!/usr/bin/env python
"""
Experiment C: mild TRUE-retention DSpar + seeded Leiden vs runtime-matched restarts.

Question
--------
All prior experiments in this repo used DSpar with-replacement sampling
("paper" method), where nominal alpha=0.8 yields only ~33-52% ACTUAL edge
retention. Scored honestly (partition found on the sparsified graph, evaluated
on the ORIGINAL graph) that regime LOSES 0.01-0.19 modularity.

Untested: MILD retention. Two questions:
  (i)  At which true retention does the raw-transfer loss shrink to ~0 / go positive?
  (ii) Does DSpar-seeded Leiden beat a RUNTIME-MATCHED vanilla-Leiden restarts
       baseline (same wall-clock budget spent on plain restarts)?

Sampler audit (done before running; see SUMMARY.md)
---------------------------------------------------
`experiments/dspar.py::dspar_sparsify(method="probabilistic_no_replace")` does:

    n_keep = ceil(alpha * m)
    p_e    = clip(score_e / sum(score) * n_keep, 0, 1)     # score = 1/d_u + 1/d_v
    keep e  iff  U_e < p_e                                  # independent Bernoulli

Because p_e is CLIPPED at 1, E[kept] = sum(clip(...)) < alpha*m whenever any raw
p_e exceeds 1 -- which happens for 13-35% of edges on these graphs. Measured
actual retention: 0.52-0.68 for alpha in {0.7..0.95}, and only ~0.80 even at
alpha=1.0. So this method CANNOT produce true 70-95% retention; it saturates.

Therefore this script runs TWO samplers:
  * "repo_noreplace"  -- bit-for-bit the repo formula above (reported as-is)
  * "calibrated"      -- same DSpar scores and same independent-Bernoulli scheme,
                         but the scale factor lambda is solved by bisection so that
                             sum_e min(1, lambda * score_e) = alpha * m
                         => E[retention] = alpha EXACTLY. This is the sampler the
                         paper's Definition actually describes (E[retention]=alpha,
                         no replacement) and is the only way to test mild TRUE
                         retention.

Both are numpy re-implementations of the repo's Bernoulli scheme, used so that
T_sparsify reflects the algorithm rather than networkx object-construction
overhead (a slow sparsifier would artificially inflate T_pipe and hand the
runtime-matched baseline extra restarts). Equivalence with
`experiments/dspar.py` is asserted by --verify-sampler.

Protocol
--------
For each dataset (LCC, undirected, simple):
  1. Baseline: leidenalg ModularityVertexPartition on original G, 5 seeds
     -> Q_base_mean, Q_base_best, T_leiden (median wall time of one run).
  2. For alpha in {0.7, 0.8, 0.9, 0.95} x 5 sparsification seeds:
     a. sparsify (T_sparsify), Leiden on G_sparse -> P_alpha (T_leiden_sparse).
        Q_orig(P_alpha) = G.modularity(P_alpha)                 [RAW TRANSFER]
     b. seeded: refine P_alpha ON THE ORIGINAL G via
        la.ModularityVertexPartition(G, initial_membership=P_alpha)
        + Optimiser.optimise_partition                          [SEEDED]
        -> Q_seeded, T_seed.  T_pipe = T_sparsify + T_leiden_sparse + T_seed
  3. Runtime-matched baseline: vanilla Leiden restarts on the ORIGINAL G with a
     budget of mean(T_pipe) for that (dataset, sampler, alpha); at least 1
     restart; keep best Q -> Q_matched_best, and report how many restarts fit.

Outputs: results.csv (per-run rows), results_summary.csv (aggregated), SUMMARY.md.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg as la

REPO = Path("/home/md724/community_detection_spectral")
OUT = Path(__file__).resolve().parent
DATASETS = {
    "email-Eu-core": REPO / "datasets/email-Eu-core/email-Eu-core.txt",
    "wiki-Vote": REPO / "datasets/wiki-Vote/wiki-Vote.txt",
    "ca-HepTh": REPO / "datasets/ca-HepTh/ca-HepTh.txt",
    "ca-CondMat": REPO / "datasets/ca-CondMat/ca-CondMat.txt",
    "email-Enron": REPO / "datasets/email-Enron/email-Enron.txt",
    "com-DBLP": REPO / "datasets/com-DBLP/com-DBLP.txt",
}
ALPHAS = [0.7, 0.8, 0.9, 0.95]
SAMPLERS = ["calibrated", "repo_noreplace"]
N_BASE_SEEDS = 5
N_SPARSE_SEEDS = 5
N_ITER = 2  # matches repo convention (run_leiden default)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_graph(path):
    """SNAP edge list -> undirected simple igraph on the largest connected component."""
    edges = []
    nodes = set()
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
    edges = [(idx[u], idx[v]) for u, v in edges]
    g = ig.Graph(n=len(node_list), edges=edges, directed=False)
    g.simplify(multiple=True, loops=True)          # undirected + simple
    g = g.connected_components().giant()           # LCC
    return g


# ----------------------------------------------------------------------------
# DSpar samplers (independent-Bernoulli, no replacement, unweighted output)
# ----------------------------------------------------------------------------
def dspar_scores(g):
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    return e, 1.0 / deg[e[:, 0]] + 1.0 / deg[e[:, 1]]


def _probs_repo(scores, alpha):
    """Exactly experiments/dspar.py::method='probabilistic_no_replace'."""
    m = len(scores)
    n_keep = int(np.ceil(alpha * m))
    return np.clip(scores / scores.sum() * n_keep, 0.0, 1.0)


def _probs_calibrated(scores, alpha):
    """Solve lambda s.t. sum(min(1, lambda*score)) = alpha*m  =>  E[retention]=alpha."""
    m = len(scores)
    target = alpha * m
    if alpha >= 1.0:
        return np.ones(m)
    lo, hi = 0.0, 1.0
    while np.minimum(1.0, hi * scores).sum() < target:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.minimum(1.0, mid * scores).sum() < target:
            lo = mid
        else:
            hi = mid
    return np.minimum(1.0, 0.5 * (lo + hi) * scores)


def sparsify(g, edge_arr, scores, alpha, sampler, seed):
    """Return (G_sparse, actual_retention). Unweighted, same vertex set."""
    probs = _probs_repo(scores, alpha) if sampler == "repo_noreplace" else _probs_calibrated(scores, alpha)
    rs = np.random.RandomState(seed)
    keep = rs.random_sample(len(scores)) < probs
    kept = edge_arr[keep]
    gs = ig.Graph(n=g.vcount(), edges=[tuple(x) for x in kept], directed=False)
    return gs, kept.shape[0] / edge_arr.shape[0]


# ----------------------------------------------------------------------------
# Leiden
# ----------------------------------------------------------------------------
def leiden(g, seed, initial_membership=None):
    """leidenalg ModularityVertexPartition. Returns (membership, Q, n_comm, seconds)."""
    t0 = time.perf_counter()
    if initial_membership is None:
        part = la.ModularityVertexPartition(g)
    else:
        _, memb = np.unique(np.asarray(initial_membership), return_inverse=True)
        part = la.ModularityVertexPartition(g, initial_membership=memb.tolist())
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    dt = time.perf_counter() - t0
    return part.membership, part.modularity, len(part), dt


# ----------------------------------------------------------------------------
# Sampler equivalence check vs experiments/dspar.py
# ----------------------------------------------------------------------------
def verify_sampler():
    sys.path.insert(0, str(REPO))
    import networkx as nx
    from experiments.dspar import dspar_sparsify

    print("Verifying 'repo_noreplace' against experiments/dspar.py "
          "(same edge ORDER => must be bit-identical)\n")
    for gname, G in [("karate", nx.karate_club_graph()),
                     ("email-Eu-core", None)]:
        if G is None:
            G = nx.Graph()
            with open(DATASETS["email-Eu-core"]) as f:
                for line in f:
                    if line[0] == "#":
                        continue
                    u, v = map(int, line.split()[:2])
                    if u != v:
                        G.add_edge(u, v)
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        m = G.number_of_edges()
        # replicate networkx's own edge ordering so the RNG stream lines up
        deg = dict(G.degree())
        elist = [(min(u, v), max(u, v)) for u, v in G.edges()]
        sc = np.array([1.0 / deg[u] + 1.0 / deg[v] for u, v in elist])
        for alpha in ALPHAS + [1.0]:
            for seed in (0, 1):
                ref = dspar_sparsify(G, retention=alpha,
                                     method="probabilistic_no_replace", seed=seed)
                ref_edges = {(min(u, v), max(u, v)) for u, v in ref.edges()}
                probs = _probs_repo(sc, alpha)
                rs = np.random.RandomState(seed)
                mine = {e for e, k in zip(elist, rs.random_sample(m) < probs) if k}
                assert mine == ref_edges, f"MISMATCH {gname} a={alpha} s={seed}"
            print(f"  {gname:14s} alpha={alpha:<5} identical  "
                  f"actual_ret(repo)={len(ref_edges)/m:.4f}  "
                  f"E[ret](calibrated)={_probs_calibrated(sc, alpha).sum()/m:.4f}")
    print("\nOK: 'repo_noreplace' reproduces experiments/dspar.py exactly.")


# ----------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------
def run_dataset(name, rows):
    g = load_graph(DATASETS[name])
    n, m = g.vcount(), g.ecount()
    print(f"\n{'='*78}\n{name}: n={n:,}  m={m:,}\n{'='*78}", flush=True)

    edge_arr, scores = dspar_scores(g)

    # --- 1. baseline -------------------------------------------------------
    base_Q, base_T = [], []
    for s in range(N_BASE_SEEDS):
        _, Q, nc, dt = leiden(g, seed=100 + s)
        base_Q.append(Q)
        base_T.append(dt)
        print(f"  baseline seed={100+s}: Q={Q:.6f} k={nc} t={dt:.3f}s", flush=True)
    Q_base_mean = float(np.mean(base_Q))
    Q_base_std = float(np.std(base_Q))
    Q_base_best = float(np.max(base_Q))
    T_leiden = float(statistics.median(base_T))
    print(f"  -> Q_base_mean={Q_base_mean:.6f} +- {Q_base_std:.6f}  "
          f"best={Q_base_best:.6f}  T_leiden={T_leiden:.3f}s", flush=True)

    summary = []
    for sampler in SAMPLERS:
        for alpha in ALPHAS:
            rets, Qraw, Qseed, Tpipe, Tspar, Tls, Tsd, ksp, kfin = ([] for _ in range(9))
            for s in range(N_SPARSE_SEEDS):
                t0 = time.perf_counter()
                gs, ret = sparsify(g, edge_arr, scores, alpha, sampler, seed=200 + s)
                t_spar = time.perf_counter() - t0
                memb_s, _, nc_s, t_ls = leiden(gs, seed=300 + s)
                q_raw = g.modularity(memb_s)                     # raw transfer
                memb_f, q_seed, nc_f, t_sd = leiden(g, seed=300 + s,
                                                    initial_membership=memb_s)
                rets.append(ret); Qraw.append(q_raw); Qseed.append(q_seed)
                Tspar.append(t_spar); Tls.append(t_ls); Tsd.append(t_sd)
                Tpipe.append(t_spar + t_ls + t_sd); ksp.append(nc_s); kfin.append(nc_f)
                rows.append(dict(dataset=name, n=n, m=m, sampler=sampler, alpha=alpha,
                                 spar_seed=200 + s, actual_retention=ret,
                                 m_sparse=gs.ecount(), Q_base_mean=Q_base_mean,
                                 Q_base_best=Q_base_best, Q_raw_transfer=q_raw,
                                 Q_seeded=q_seed, k_sparse=nc_s, k_seeded=nc_f,
                                 T_sparsify=t_spar, T_leiden_sparse=t_ls,
                                 T_seed=t_sd, T_pipe=t_spar + t_ls + t_sd,
                                 T_leiden_base=T_leiden))
            budget = float(np.mean(Tpipe))

            # --- 3. runtime-matched restarts on the ORIGINAL graph ----------
            best_matched, n_restarts, spent = -1.0, 0, 0.0
            while True:
                _, Qm, _, dt = leiden(g, seed=900 + n_restarts)
                n_restarts += 1
                spent += dt
                best_matched = max(best_matched, Qm)
                if spent >= budget:
                    break

            rec = dict(
                dataset=name, n=n, m=m, sampler=sampler, alpha=alpha,
                actual_ret_mean=float(np.mean(rets)), actual_ret_std=float(np.std(rets)),
                Q_base_mean=Q_base_mean, Q_base_std=Q_base_std, Q_base_best=Q_base_best,
                Q_raw_mean=float(np.mean(Qraw)), Q_raw_std=float(np.std(Qraw)),
                Q_raw_best=float(np.max(Qraw)),
                Q_seeded_mean=float(np.mean(Qseed)), Q_seeded_std=float(np.std(Qseed)),
                Q_seeded_best=float(np.max(Qseed)),
                Q_matched_best=float(best_matched), n_matched_restarts=n_restarts,
                T_leiden=T_leiden, T_sparsify_mean=float(np.mean(Tspar)),
                T_leiden_sparse_mean=float(np.mean(Tls)), T_seed_mean=float(np.mean(Tsd)),
                T_pipe_mean=budget, T_matched_spent=spent,
                k_sparse_mean=float(np.mean(ksp)), k_seeded_mean=float(np.mean(kfin)),
                raw_minus_base=float(np.mean(Qraw)) - Q_base_mean,
                seeded_minus_base=float(np.mean(Qseed)) - Q_base_mean,
                seeded_minus_matched=float(np.mean(Qseed)) - float(best_matched),
                seededbest_minus_matched=float(np.max(Qseed)) - float(best_matched),
            )
            summary.append(rec)
            print(f"  [{sampler:14s} a={alpha:<5}] ret={rec['actual_ret_mean']:.4f} "
                  f"Qraw={rec['Q_raw_mean']:.6f} ({rec['raw_minus_base']:+.6f}) "
                  f"Qseed={rec['Q_seeded_mean']:.6f} ({rec['seeded_minus_base']:+.6f}) "
                  f"Qmatch={best_matched:.6f} (x{n_restarts}) "
                  f"d_match={rec['seeded_minus_matched']:+.6f} "
                  f"T_pipe={budget:.2f}s", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-sampler", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    if args.verify_sampler:
        verify_sampler()
        return

    rows, summary = [], []
    for name in args.datasets:
        summary += run_dataset(name, rows)
        # incremental write so a long run is never lost
        with open(OUT / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        with open(OUT / "results_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader(); w.writerows(summary)
    print("\nDone. -> results.csv, results_summary.csv")


if __name__ == "__main__":
    main()
