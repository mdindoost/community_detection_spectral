#!/usr/bin/env python
"""
Experiment H: is the com-Youtube seeded-refinement gain real, or a statistical
artifact of a single unlucky runtime-matched baseline draw?

Background (PAPER_RESTRUCTURE/phase3_review/JUDGE_RULING.md, finding MC3)
------------------------------------------------------------------------
exp_E_delta_star measured, on com-Youtube at calibrated DSpar alpha=0.9 with 3
sparsification seeds:

    Q_base_mean   = 0.724506 +- 0.003437   (5 plain Leiden seeds 100..104)
    Q_seeded_mean = 0.729571
    Q_matched_best= 0.720932   (best of 2 runtime-matched plain restarts)
    y = Q_seeded_mean - Q_matched_best = +0.008639   (~2.5 sigma)

But Q_matched_best (0.7209) fell BELOW the base mean (0.7245) -- the matched
baseline was one unlucky realization of a best-of-2 draw, so the +0.0086 gain is
partly an artifact of which two restarts happened to be drawn. Only one
(sampler, alpha) configuration was ever tested on com-Youtube.

This script fixes both problems:
  1. Baseline distribution: 10 plain Leiden runs (seeds 100..109) on the
     ORIGINAL LCC -> Q_base mean/std/best.
  2. Matched baseline as a DISTRIBUTION, not a draw: the exact distribution of
     best-of-2 over all C(10,2)=45 unordered pairs of those runs. Its mean is
     the FAIR runtime-matched baseline (T_pipe ~ 186s vs T_leiden ~ 103s => a
     runtime match buys 2 restarts, which exp_E confirmed: n_matched_restarts=2).
     We also report the empirical P(best-of-2 >= q) for each seeded result.
  3. Seeded sweep: both samplers (calibrated lambda-bisection, and the repo's
     clipped no-replacement) x alpha in {0.7,0.8,0.9,0.95} x 5 sparsification
     seeds, recording actual retention and T_pipe.
  4. Verdict per configuration: (Q_seeded_mean - E[best-of-2]) / Q_base_std, and
     the same against best-of-5 and best-of-10.

All loading / sampling / seeded-refinement / Leiden code is reused verbatim from
exp_E_delta_star/run.py and exp_C_true_retention_seeded/run.py, with the same
seed conventions (base 100.., sparsify 200.., leiden-on-sparse 300..) so the
calibrated alpha=0.9 rows for sparsify seeds 200,201,202 are directly
comparable with exp_E's.

Usage
-----
  run.py plan                       -> print job list
  run.py worker <jobs.json> <out.json>
  run.py analyze                    -> results.csv, SUMMARY.md
"""

import itertools
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg as la

REPO = Path("/home/md724/community_detection_spectral")
HERE = Path(__file__).resolve().parent
NETWORK = "com-Youtube"
EDGE_FILE = REPO / "datasets" / NETWORK / f"{NETWORK}.txt"

ALPHAS = [0.7, 0.8, 0.9, 0.95]
SAMPLERS = ["calibrated", "repo_noreplace"]
N_BASE_SEEDS = 10          # seeds 100..109
N_SPARSE_SEEDS = 5         # sparsify seeds 200..204, leiden seeds 300..304
N_ITER = 2                 # repo convention (run_leiden default)


# ---------------------------------------------------------------------------
# Loading (verbatim from exp_E_delta_star/run.py, which took it from exp_B)
# ---------------------------------------------------------------------------

def _parse_edge_file(path):
    """Chunked whitespace edge-list parser (comments '#'). Returns (M,2) int64."""
    parts = []
    leftover = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 24)
            if not chunk:
                break
            buf = leftover + chunk
            cut = buf.rfind(b"\n")
            if cut == -1:
                leftover = buf
                continue
            block, leftover = buf[:cut], buf[cut + 1:]
            if b"#" in block:
                lines = [ln for ln in block.split(b"\n")
                         if ln and not ln.lstrip().startswith(b"#")]
                block = b"\n".join(lines)
            toks = block.split()
            if toks:
                parts.append(np.array(toks, dtype=np.int64))
    if leftover.strip() and not leftover.lstrip().startswith(b"#"):
        toks = leftover.split()
        if toks:
            parts.append(np.array(toks, dtype=np.int64))
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    if flat.size % 2:
        flat = flat[: (flat.size // 2) * 2]
    return flat.reshape(-1, 2)


def load_lcc_graph(path):
    """Undirected simple graph, largest connected component, as igraph.Graph."""
    raw = _parse_edge_file(path)
    u, v = raw[:, 0], raw[:, 1]
    del raw
    keep = u != v
    u, v = u[keep], v[keep]
    nodes = np.unique(np.concatenate([u, v]))
    n = nodes.size
    u = np.searchsorted(nodes, u)
    v = np.searchsorted(nodes, v)
    del nodes
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    del u, v
    key = lo.astype(np.int64) * n + hi
    key = np.unique(key)
    lo = (key // n).astype(np.int32)
    hi = (key % n).astype(np.int32)
    del key
    g = ig.Graph(n=int(n))
    g.add_edges(np.column_stack([lo, hi]))
    del lo, hi
    g = g.connected_components(mode="weak").giant()
    g.simplify(multiple=True, loops=True)
    return g


# ---------------------------------------------------------------------------
# DSpar samplers (verbatim from exp_C_true_retention_seeded/run.py)
# ---------------------------------------------------------------------------

def dspar_scores(g):
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    return e, 1.0 / deg[e[:, 0]] + 1.0 / deg[e[:, 1]]


def _probs_repo(scores, alpha):
    """Exactly experiments/dspar.py::method='probabilistic_no_replace' (clipped)."""
    m = len(scores)
    n_keep = int(np.ceil(alpha * m))
    return np.clip(scores / scores.sum() * n_keep, 0.0, 1.0)


def _probs_calibrated(scores, alpha):
    """lambda by bisection s.t. sum(min(1, lambda*s)) = alpha*m => E[retention]=alpha."""
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


def sparsify(g, edge_arr, probs, seed):
    """Return (G_sparse, actual_retention). Unweighted, same vertex set."""
    rs = np.random.RandomState(seed)
    keep = rs.random_sample(probs.size) < probs
    kept = edge_arr[keep]
    gs = ig.Graph(n=g.vcount())
    gs.add_edges(kept)
    return gs, kept.shape[0] / edge_arr.shape[0]


# ---------------------------------------------------------------------------
# Leiden (verbatim from exp_C / exp_E)
# ---------------------------------------------------------------------------

def leiden(g, seed, initial_membership=None):
    t0 = time.perf_counter()
    if initial_membership is None:
        part = la.ModularityVertexPartition(g)
    else:
        _, memb = np.unique(np.asarray(initial_membership), return_inverse=True)
        part = la.ModularityVertexPartition(g, initial_membership=memb.tolist())
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    return part.membership, part.modularity, len(part), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def all_jobs(alphas=None, n_sparse=None, samplers=None, n_base=None):
    alphas = ALPHAS if alphas is None else alphas
    n_sparse = N_SPARSE_SEEDS if n_sparse is None else n_sparse
    samplers = SAMPLERS if samplers is None else samplers
    n_base = N_BASE_SEEDS if n_base is None else n_base
    jobs = [dict(kind="base", seed=100 + s) for s in range(n_base)]
    for sampler in samplers:
        for alpha in alphas:
            for s in range(n_sparse):
                jobs.append(dict(kind="seeded", sampler=sampler, alpha=alpha,
                                 spar_seed=200 + s, leiden_seed=300 + s))
    return jobs


def run_worker(jobs_path, out_path):
    jobs = json.load(open(jobs_path))
    t0 = time.perf_counter()
    g = load_lcc_graph(EDGE_FILE)
    n, m = g.vcount(), g.ecount()
    print(f"[{os.getpid()}] loaded n={n:,} m={m:,} in {time.perf_counter()-t0:.1f}s",
          flush=True)

    need_scores = any(j["kind"] == "seeded" for j in jobs)
    edge_arr = scores = None
    if need_scores:
        edge_arr, scores = dspar_scores(g)
    prob_cache = {}

    results = []
    for j in jobs:
        if j["kind"] == "base":
            _, Q, nc, dt = leiden(g, seed=j["seed"])
            rec = dict(j, n=n, m=m, Q=Q, k=nc, T=dt)
            print(f"[{os.getpid()}] base seed={j['seed']} Q={Q:.6f} k={nc} "
                  f"t={dt:.1f}s", flush=True)
        else:
            key = (j["sampler"], j["alpha"])
            if key not in prob_cache:
                prob_cache.clear()
                prob_cache[key] = (_probs_repo(scores, j["alpha"])
                                   if j["sampler"] == "repo_noreplace"
                                   else _probs_calibrated(scores, j["alpha"]))
            probs = prob_cache[key]
            t0 = time.perf_counter()
            gs, ret = sparsify(g, edge_arr, probs, seed=j["spar_seed"])
            t_spar = time.perf_counter() - t0
            memb_s, Q_sparse_self, k_s, t_ls = leiden(gs, seed=j["leiden_seed"])
            q_raw = g.modularity(memb_s)
            del gs
            memb_f, q_seed, k_f, t_sd = leiden(g, seed=j["leiden_seed"],
                                               initial_membership=memb_s)
            del memb_s, memb_f
            rec = dict(j, n=n, m=m, retention=ret, Q_raw=q_raw,
                       Q_sparse_self=Q_sparse_self, k_sparse=k_s,
                       Q_seeded=q_seed, k_seeded=k_f,
                       T_sparsify=t_spar, T_leiden_sparse=t_ls, T_seed=t_sd,
                       T_pipe=t_spar + t_ls + t_sd)
            print(f"[{os.getpid()}] {j['sampler']} a={j['alpha']} "
                  f"seed={j['spar_seed']} ret={ret:.4f} Qraw={q_raw:.6f} "
                  f"Qseed={q_seed:.6f} k={k_f} T_pipe={rec['T_pipe']:.1f}s",
                  flush=True)
        rec["rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        results.append(rec)
        json.dump(results, open(out_path, "w"), indent=1)
    print(f"[{os.getpid()}] done, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB", flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def best_of_k_distribution(vals, k):
    """Exact distribution of max over all unordered k-subsets of vals."""
    return np.array([max(c) for c in itertools.combinations(vals, k)])


def analyze():
    import csv
    raw = []
    for p in sorted((HERE / "raw").glob("*.json")):
        raw.extend(json.load(open(p)))
    base = sorted([r for r in raw if r["kind"] == "base"], key=lambda r: r["seed"])
    seeded = [r for r in raw if r["kind"] == "seeded"]

    Qb = np.array([r["Q"] for r in base])
    Tb = np.array([r["T"] for r in base])
    mu, sd = float(Qb.mean()), float(Qb.std(ddof=1))
    sd_pop = float(Qb.std())

    bo2 = best_of_k_distribution(Qb, 2)
    bo5 = best_of_k_distribution(Qb, 5)
    bo10 = best_of_k_distribution(Qb, 10)

    # per-run rows
    with open(HERE / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "sampler", "alpha", "seed", "retention", "Q_raw",
                    "Q_seeded", "k", "T_pipe", "T_sparsify", "T_leiden_sparse",
                    "T_seed"])
        for r in base:
            w.writerow(["base", "", "", r["seed"], "", "", f"{r['Q']:.6f}",
                        r["k"], f"{r['T']:.2f}", "", "", ""])
        for r in sorted(seeded, key=lambda r: (r["sampler"], r["alpha"], r["spar_seed"])):
            w.writerow(["seeded", r["sampler"], r["alpha"], r["spar_seed"],
                        f"{r['retention']:.4f}", f"{r['Q_raw']:.6f}",
                        f"{r['Q_seeded']:.6f}", r["k_seeded"],
                        f"{r['T_pipe']:.2f}", f"{r['T_sparsify']:.2f}",
                        f"{r['T_leiden_sparse']:.2f}", f"{r['T_seed']:.2f}"])

    # per-configuration summary
    configs = []
    keys = sorted({(r["sampler"], r["alpha"]) for r in seeded})
    for sampler, alpha in keys:
        rr = [r for r in seeded if r["sampler"] == sampler and r["alpha"] == alpha]
        qs = np.array([r["Q_seeded"] for r in rr])
        qraw = np.array([r["Q_raw"] for r in rr])
        ret = np.array([r["retention"] for r in rr])
        tp = np.array([r["T_pipe"] for r in rr])
        c = dict(
            sampler=sampler, alpha=alpha, n_seeds=len(rr),
            retention=float(ret.mean()),
            Q_raw_mean=float(qraw.mean()),
            Q_seeded_mean=float(qs.mean()), Q_seeded_std=float(qs.std(ddof=1)),
            Q_seeded_best=float(qs.max()), Q_seeded_min=float(qs.min()),
            T_pipe_mean=float(tp.mean()),
            restarts_equiv=float(tp.mean() / np.median(Tb)),
            gain_vs_base_mean=float(qs.mean() - mu),
            gain_vs_bo2_mean=float(qs.mean() - bo2.mean()),
            gain_vs_bo5_mean=float(qs.mean() - bo5.mean()),
            gain_vs_bo10=float(qs.mean() - bo10.mean()),
            z_vs_base=float((qs.mean() - mu) / sd),
            z_vs_bo2=float((qs.mean() - bo2.mean()) / sd),
            z_vs_bo5=float((qs.mean() - bo5.mean()) / sd),
            z_vs_bo10=float((qs.mean() - bo10.mean()) / sd),
            p_bo2_ge=float((bo2 >= qs.mean()).mean()),
            p_bo5_ge=float((bo5 >= qs.mean()).mean()),
            p_bo10_ge=float((bo10 >= qs.mean()).mean()),
            p_base_ge=float((Qb >= qs.mean()).mean()),
            n_seeds_above_bo2mean=int((qs > bo2.mean()).sum()),
            n_seeds_above_base_best=int((qs > Qb.max()).sum()),
        )
        configs.append(c)

    with open(HERE / "config_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(configs[0]))
        w.writeheader()
        for c in configs:
            w.writerow(c)

    stats = dict(
        Q_base=list(map(float, Qb)), base_seeds=[r["seed"] for r in base],
        Q_base_mean=mu, Q_base_std=sd, Q_base_std_pop=sd_pop,
        Q_base_best=float(Qb.max()), Q_base_min=float(Qb.min()),
        T_leiden_median=float(np.median(Tb)), T_leiden_mean=float(Tb.mean()),
        bo2_mean=float(bo2.mean()), bo2_std=float(bo2.std(ddof=1)),
        bo2_min=float(bo2.min()), bo2_max=float(bo2.max()),
        bo5_mean=float(bo5.mean()), bo5_std=float(bo5.std(ddof=1)),
        bo10_mean=float(bo10.mean()),
    )
    json.dump(dict(stats=stats, configs=configs), open(HERE / "stats.json", "w"),
              indent=1)
    return stats, configs, Qb, bo2, bo5, bo10, base, seeded


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "worker":
        run_worker(sys.argv[2], sys.argv[3])
    elif cmd == "plan":
        print(json.dumps(all_jobs(), indent=1))
    elif cmd == "analyze":
        s, c, *_ = analyze()
        print(json.dumps(dict(stats=s, configs=c), indent=1))


if __name__ == "__main__":
    main()
