#!/usr/bin/env python
"""
Experiment P: algorithm generality.

Every verdict in this paper is Leiden/modularity-only.  Satuluri 2011's claim
covered several algorithms.  Exp P runs the SAME protocol (honest transfer
scoring, runtime matching, granularity-reported recovery, chance floor) under
three further detection algorithms from python-igraph:

  * Infomap                (community_infomap)
  * Louvain / multilevel   (community_multilevel)
  * Label propagation      (community_label_propagation)

Sparsifiers (machinery verbatim from exp_K / exp_L):
  * DSpar, calibrated Bernoulli sampler, target true retention 0.8 and 0.5,
    2 sparsifier seeds.  Detection runs on the UNWEIGHTED sparse graph (the
    paper's main protocol; exp_K established the HT weights are an
    evaluation-correctness device, not a performance device).
  * L-Spar (exact Jaccard, local top-ceil(d^e), union rule), target 0.5,
    deterministic -> no sparsifier seed.

Stages:
  run.py main     [nets]  -> results.csv (aggregated) + runs.csv (every run)
  run.py recovery [nets]  -> recovery.csv

Seeding: igraph's RNG is redirected to Python's `random` module once
(`ig.set_random_number_generator(random)`); every detection call does
`random.seed(seed)` immediately before invoking the algorithm.  Verified
deterministic (see `run.py selftest`).
"""

import csv
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import igraph as ig

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATASETS_DIR = REPO / "datasets"
DATA = REPO / "data"

ALGOS = ["infomap", "louvain", "labelprop"]
INFOMAP_TRIALS = 1              # DEVIATION from igraph default trials=10, see SUMMARY
ALGO_SEEDS = [300, 301, 302]    # 3 seeds per stochastic cell (all three are stochastic)
BASE_SEEDS = [100, 101, 102]    # baseline arm, same count
SPAR_SEEDS = [200, 201]         # 2 DSpar sampler seeds
MATCH_SEEDS = list(range(900, 990))
CHANCE_SEEDS = [11, 12, 13]
DSPAR_ALPHAS = [0.8, 0.5]
LSPAR_TARGETS = [0.5]
MIN_GT_SIZE = 3
K_TOL = 0.25                    # |dk|/k comparability rule from DESIGN.md

NETWORKS = ["email-Eu-core", "wiki-Vote", "ca-HepTh", "ca-CondMat",
            "email-Enron", "com-DBLP", "com-Amazon"]


# ---------------------------------------------------------------------------
# Loading (verbatim from exp_L_lspar/run.py)
# ---------------------------------------------------------------------------

def _parse_edge_file(path):
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


def load_lcc_graph(name, return_map=False):
    """Undirected simple graph, largest connected component, as igraph.Graph."""
    path = DATASETS_DIR / name / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = _parse_edge_file(path)
    u, v = raw[:, 0], raw[:, 1]
    del raw
    keep = u != v
    u, v = u[keep], v[keep]
    nodes = np.unique(np.concatenate([u, v]))
    n = nodes.size
    u = np.searchsorted(nodes, u)
    v = np.searchsorted(nodes, v)
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    del u, v
    key = np.unique(lo.astype(np.int64) * n + hi)
    lo = (key // n).astype(np.int64)
    hi = (key % n).astype(np.int64)
    del key
    g = ig.Graph(n=int(n))
    g.add_edges(np.column_stack([lo, hi]))
    del lo, hi
    comps = g.connected_components(mode="weak")
    memb = np.asarray(comps.membership)
    giant = int(np.argmax(comps.sizes()))
    keep_idx = np.where(memb == giant)[0]
    if keep_idx.size < g.vcount():
        g = g.induced_subgraph(keep_idx.tolist())
        nodes = nodes[keep_idx]
    g.simplify(multiple=True, loops=True)
    if return_map:
        return g, {int(o): i for i, o in enumerate(nodes)}
    return g


def load_email_labelled():
    """email-Eu-core with department labels; loader semantics of exp_F/exp_L."""
    import networkx as nx
    edge_path = DATA / "email-Eu-core.txt"
    lab_path = DATA / "email-Eu-core-department-labels.txt"
    if not edge_path.exists():                       # Fuji layout
        edge_path = DATASETS_DIR / "email-Eu-core" / "email-Eu-core.txt"
        lab_path = DATASETS_DIR / "email-Eu-core" / "email-Eu-core_labels.txt"
    G_raw = nx.Graph()
    with open(edge_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2 and p[0] != p[1]:
                G_raw.add_edge(int(p[0]), int(p[1]))
    gt = {}
    with open(lab_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                gt[int(p[0])] = int(p[1])
    G = nx.Graph()
    G.add_nodes_from(x for x in G_raw.nodes() if x in gt)
    for u, v in G_raw.edges():
        if u != v and u in gt and v in gt:
            G.add_edge(u, v)
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()
    order = sorted(G.nodes())
    remap = {o: i for i, o in enumerate(order)}
    G = nx.relabel_nodes(G, remap, copy=True)
    labels = [gt[o] for o in order]
    _, y = np.unique(np.asarray(labels, dtype=object).astype(str), return_inverse=True)
    g = ig.Graph(n=G.number_of_nodes(),
                 edges=[(u, v) for u, v in G.edges()], directed=False)
    g.simplify()
    return g, y


def load_ground_truth(path, id_map):
    comms = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            members = [id_map[int(t)] for t in line.split() if int(t) in id_map]
            if len(members) >= MIN_GT_SIZE:
                comms.append(np.unique(np.asarray(members, dtype=np.int64)))
    return comms


# ---------------------------------------------------------------------------
# Sparsifiers
# ---------------------------------------------------------------------------

def dspar_scores(g):
    """exp_K/exp_N verbatim: s_e = 1/d_u + 1/d_v."""
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    return e, 1.0 / deg[e[:, 0]] + 1.0 / deg[e[:, 1]]


def _probs_calibrated(scores, alpha):
    """exp_C calibration: p_e = min(1, lambda s_e) with sum p_e = alpha*m."""
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


def sparsify_calibrated(edge_arr, probs, seed):
    rs = np.random.RandomState(seed)
    keep = rs.random_sample(probs.size) < probs
    return edge_arr[keep], 1.0 / probs[keep]


def edge_jaccard(g, chunk=100_000):
    """Exact Jaccard of endpoint neighbourhoods for every edge (exp_L verbatim)."""
    n, m = g.vcount(), g.ecount()
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    u, v = E[:, 0].copy(), E[:, 1].copy()
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    A = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)),
                      shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    cn = np.empty(m, dtype=np.float64)
    for s in range(0, m, chunk):
        t = min(s + chunk, m)
        cn[s:t] = np.asarray(A[u[s:t]].multiply(A[v[s:t]]).sum(axis=1)).ravel()
    union = deg[u] + deg[v] - cn
    J = np.where(union > 0, cn / np.maximum(union, 1e-12), 0.0)
    return E, J, deg.astype(np.int64)


class LSpar:
    """exp_L verbatim: local top-ceil(d^e) by Jaccard; union rule."""

    def __init__(self, g, J=None, E=None, deg=None):
        self.n, self.m = g.vcount(), g.ecount()
        if J is None:
            E, J, deg = edge_jaccard(g)
        self.E, self.J, self.deg = E, J, deg
        node = np.concatenate([E[:, 0], E[:, 1]])
        eid = np.concatenate([np.arange(self.m), np.arange(self.m)])
        jj = np.concatenate([J, J])
        order = np.lexsort((-jj, node))
        self.eid_s = eid[order]
        d = np.bincount(node, minlength=self.n).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(d)[:-1]])
        self.rank = np.arange(node.size, dtype=np.int64) - np.repeat(starts, d)
        self.d_pos = np.repeat(d, d).astype(np.float64)
        self.d = d

    def select(self, e):
        k = np.ceil(np.power(np.maximum(self.d_pos, 1.0), e))
        sel = self.rank < k
        return np.unique(self.eid_s[sel])

    def retention(self, e):
        return self.select(e).size / self.m

    def bisect_e(self, target, tol=0.005, iters=40):
        lo_r, hi_r = self.retention(0.0), 1.0
        if target <= lo_r:
            return 0.0, lo_r, "floor"
        lo, hi = 0.0, 1.0
        best = (abs(lo_r - target), 0.0, lo_r)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            r = self.retention(mid)
            if abs(r - target) < best[0]:
                best = (abs(r - target), mid, r)
            if best[0] <= tol:
                break
            if r < target:
                lo = mid
            else:
                hi = mid
        return best[1], best[2], "ok"


def build_sparse(n, kept_edges):
    gs = ig.Graph(n=int(n))
    gs.add_edges(kept_edges)
    return gs


# ---------------------------------------------------------------------------
# Detection (the three igraph algorithms) + seeding
# ---------------------------------------------------------------------------

ig.set_random_number_generator(random)      # igraph draws from python's random


def detect(g, algo, seed, resolution=None):
    """Run `algo` on g with igraph RNG seeded by `seed`. Returns (memb, k, dt)."""
    random.seed(int(seed))
    t0 = time.perf_counter()
    if algo == "infomap":
        vc = g.community_infomap(trials=INFOMAP_TRIALS)
    elif algo == "louvain":
        if resolution is None:
            vc = g.community_multilevel()
        else:
            vc = g.community_multilevel(resolution=float(resolution))
    elif algo == "labelprop":
        vc = g.community_label_propagation()
    else:
        raise ValueError(algo)
    dt = time.perf_counter() - t0
    memb = np.asarray(vc.membership, dtype=np.int64)
    return memb, int(np.unique(memb).size), dt


def louvain_matched(g, seed, target_k, tol=0.05, max_iter=30):
    """Louvain on the ORIGINAL graph, resolution bisected so k ~= target_k."""
    def n_of(gamma):
        memb, k, _ = detect(g, "louvain", seed, resolution=gamma)
        return memb, k

    lo, hi = 1.0, 1.0
    memb, k = n_of(hi)
    best = (abs(k - target_k), memb, hi, k)
    it = 0
    if k > target_k:
        while k > target_k and lo > 1e-4 and it < max_iter:
            hi = lo
            lo /= 2.0
            memb, k = n_of(lo)
            it += 1
            if abs(k - target_k) < best[0]:
                best = (abs(k - target_k), memb, lo, k)
    else:
        while k < target_k and hi < 1e5 and it < max_iter:
            lo = hi
            hi *= 2.0
            memb, k = n_of(hi)
            it += 1
            if abs(k - target_k) < best[0]:
                best = (abs(k - target_k), memb, hi, k)
    for _ in range(max_iter - it):
        if best[0] <= max(1.0, tol * target_k):
            break
        mid = 0.5 * (lo + hi)
        memb, km = n_of(mid)
        if abs(km - target_k) < best[0]:
            best = (abs(km - target_k), memb, mid, km)
        if km < target_k:
            lo = mid
        else:
            hi = mid
    return best[1], best[2], best[3]


# ---------------------------------------------------------------------------
# CSV / logging
# ---------------------------------------------------------------------------

def append_row(path, row, fields=None):
    fields = fields or list(row)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def log(*a):
    print(*a, flush=True)


def done_pairs(path, keys):
    """Set of key-tuples already present in a csv (resume support)."""
    if not path.exists():
        return set()
    out = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out.add(tuple(r[k] for k in keys))
            except KeyError:
                pass
    return out


RUN_FIELDS = ["network", "algo", "arm", "target_ret", "spar_seed", "algo_seed",
              "realized_ret", "m_sparse", "Q_native", "Q_orig", "k", "t_detect"]


# ===========================================================================
# STAGE: main  (honest transfer quality + cost)
# ===========================================================================

def run_main(networks):
    out = HERE / "results.csv"
    runs = HERE / "runs.csv"
    done = done_pairs(out, ["network", "algo"])

    for name in networks:
        t0 = time.perf_counter()
        g = load_lcc_graph(name)
        n, m = g.vcount(), g.ecount()
        log(f"\n{'='*78}\n{name}: n={n:,} m={m:,}  (load {time.perf_counter()-t0:.1f}s)\n{'='*78}")

        # ---- sparse graphs, built once and shared by the three algorithms
        edge_arr, scores = dspar_scores(g)
        sparse_arms = {}          # (arm, target) -> [(spar_seed, gs, realized, T_sparsify)]
        for alpha in DSPAR_ALPHAS:
            probs = _probs_calibrated(scores, alpha)
            lst = []
            for ss in SPAR_SEEDS:
                ts = time.perf_counter()
                kept, _w = sparsify_calibrated(edge_arr, probs, ss)
                gs = build_sparse(n, kept)
                T_sp = time.perf_counter() - ts
                lst.append((ss, gs, kept.shape[0] / m, T_sp))
            sparse_arms[("dspar", alpha)] = lst
            log(f"  dspar alpha={alpha}: realized "
                f"{[round(x[2],4) for x in lst]}  T_sparsify~{np.mean([x[3] for x in lst]):.2f}s")

        tj = time.perf_counter()
        E, J, deg = edge_jaccard(g)
        T_jaccard = time.perf_counter() - tj
        tr = time.perf_counter()
        ls = LSpar(g, J=J, E=E, deg=deg)
        T_rank = time.perf_counter() - tr
        for target in LSPAR_TARGETS:
            e_used, realized, status = ls.bisect_e(target)
            tsel = time.perf_counter()
            eids = ls.select(e_used)
            T_select = time.perf_counter() - tsel
            gs = build_sparse(n, np.asarray(g.get_edgelist(), dtype=np.int64)[eids])
            sparse_arms[("lspar", target)] = [
                (-1, gs, realized, T_jaccard + T_rank + T_select)]
            log(f"  lspar target={target}: e={e_used:.4f} realized={realized:.4f} "
                f"({status}) m'={gs.ecount():,} T_sparsify={T_jaccard+T_rank+T_select:.2f}s")
        lspar_e = e_used
        del E, J, deg, ls, edge_arr, scores

        for algo in ALGOS:
            if (name, algo) in done:
                log(f"  [skip {algo}: already in results.csv]")
                continue
            # ---- baseline
            bQ, bK, bT = [], [], []
            for s in BASE_SEEDS:
                memb, k, dt = detect(g, algo, s)
                bQ.append(float(g.modularity(memb.tolist())))
                bK.append(k)
                bT.append(dt)
                append_row(runs, dict(network=name, algo=algo, arm="baseline",
                                      target_ret=1.0, spar_seed="", algo_seed=s,
                                      realized_ret=1.0, m_sparse=m,
                                      Q_native=bQ[-1], Q_orig=bQ[-1], k=k,
                                      t_detect=dt), RUN_FIELDS)
            Qb_mean, Qb_std, Qb_best = float(np.mean(bQ)), float(np.std(bQ)), float(max(bQ))
            T_orig = float(np.mean(bT))
            log(f"  [{algo}] baseline Q_orig={Qb_mean:.6f}+-{Qb_std:.6f} best={Qb_best:.6f} "
                f"k={np.mean(bK):.1f} T={T_orig:.2f}s")

            for (arm, target), lst in sparse_arms.items():
                Qn, Qo, Ks, Ts, rets, msp, Tsp = [], [], [], [], [], [], []
                for (ss, gs, realized, T_sp) in lst:
                    for s in ALGO_SEEDS:
                        memb, k, dt = detect(gs, algo, s)
                        q_native = float(gs.modularity(memb.tolist()))
                        q_orig = float(g.modularity(memb.tolist()))
                        Qn.append(q_native); Qo.append(q_orig); Ks.append(k); Ts.append(dt)
                        append_row(runs, dict(
                            network=name, algo=algo, arm=arm, target_ret=target,
                            spar_seed=("" if ss < 0 else ss), algo_seed=s,
                            realized_ret=round(realized, 6), m_sparse=gs.ecount(),
                            Q_native=q_native, Q_orig=q_orig, k=k, t_detect=dt), RUN_FIELDS)
                    rets.append(realized); msp.append(gs.ecount()); Tsp.append(T_sp)
                Qo_m, Qo_s, Qo_b = float(np.mean(Qo)), float(np.std(Qo)), float(max(Qo))
                T_sparse = float(np.mean(Ts))
                T_sparsify = float(np.mean(Tsp))

                # runtime-matched baseline: equal wall clock to the WHOLE pipeline
                budget = T_sparsify + T_sparse
                spent, best_match, nrest = 0.0, -1.0, 0
                for s in MATCH_SEEDS:
                    memb, _k, dt = detect(g, algo, s)
                    spent += dt
                    nrest += 1
                    best_match = max(best_match, float(g.modularity(memb.tolist())))
                    if spent >= budget:
                        break

                row = dict(
                    network=name, n=n, m=m, algo=algo, sparsifier=arm,
                    target_ret=target,
                    realized_ret=float(np.mean(rets)), m_sparse=float(np.mean(msp)),
                    e_used=(round(lspar_e, 6) if arm == "lspar" else ""),
                    n_spar_seeds=len(lst), n_algo_seeds=len(ALGO_SEEDS),
                    Q_base_mean=Qb_mean, Q_base_std=Qb_std, Q_base_best=Qb_best,
                    k_base=float(np.mean(bK)), T_algo_orig=T_orig,
                    Q_sparse_native=float(np.mean(Qn)),
                    Q_orig_mean=Qo_m, Q_orig_std=Qo_s, Q_orig_best=Qo_b,
                    k_sparse=float(np.mean(Ks)), T_algo_sparse=T_sparse,
                    T_sparsify=T_sparsify,
                    dQ_naive=float(np.mean(Qn)) - Qb_mean,
                    dQ_honest_vs_mean=Qo_m - Qb_mean,
                    dQ_honest_vs_best=Qo_m - Qb_best,   # exp_L semantics: mean vs best
                    dQ_bestbest=Qo_b - Qb_best,
                    dQ_vs_matched=Qo_m - best_match,
                    Q_matched_best=best_match, budget=budget, n_restarts=nrest,
                    pooled_sd=float(np.sqrt(0.5 * (Qb_std ** 2 + Qo_s ** 2))),
                    T_pipeline=T_sparsify + T_sparse,
                    speedup_vs_single=T_orig / max(T_sparsify + T_sparse, 1e-9),
                    speedup_algo_only=T_orig / max(T_sparse, 1e-9),
                )
                append_row(out, row)
                log(f"    {arm}@{target}: ret={row['realized_ret']:.4f} "
                    f"Q_native={row['Q_sparse_native']:.6f} Q_orig={Qo_m:.6f} "
                    f"dQ_vs_mean={row['dQ_honest_vs_mean']:+.6f} "
                    f"dQ_vs_best={row['dQ_honest_vs_best']:+.6f} "
                    f"dQ_vs_matched={row['dQ_vs_matched']:+.6f} (r={nrest}) "
                    f"k={row['k_sparse']:.0f} (base {row['k_base']:.0f}) "
                    f"spd={row['speedup_vs_single']:.2f}x")

        for lst in sparse_arms.values():
            for item in lst:
                del item
        del sparse_arms, g


# ===========================================================================
# STAGE: recovery
# ===========================================================================

def average_f1(membership, n2c, gt_sizes):
    """exp_L verbatim."""
    memb = np.asarray(membership, dtype=np.int64)
    cl_sizes = np.bincount(memb)
    n_clusters = int((cl_sizes > 0).sum())
    ge3 = cl_sizes >= 3
    n_ge3 = int(ge3.sum())
    overlap = defaultdict(int)
    for node, cids in n2c.items():
        cl = int(memb[node])
        for cm in cids:
            overlap[(cl, cm)] += 1
    best_cl = defaultdict(float)
    best_gt = np.zeros(len(gt_sizes))
    for (cl, cm), ov in overlap.items():
        f1 = 2.0 * ov / (cl_sizes[cl] + gt_sizes[cm])
        if f1 > best_cl[cl]:
            best_cl[cl] = f1
        if f1 > best_gt[cm]:
            best_gt[cm] = f1
    gt2det = float(best_gt.mean()) if len(gt_sizes) else 0.0
    sum_ge3 = sum(v for cl, v in best_cl.items() if ge3[cl])
    det2gt_ge3 = sum_ge3 / n_ge3 if n_ge3 else 0.0
    return dict(avgF1_ge3=0.5 * (gt2det + det2gt_ge3), gt2det=gt2det,
                det2gt_ge3=det2gt_ge3, n_clusters=n_clusters, n_clusters_ge3=n_ge3)


def size_matched_random(memb, seed):
    """Chance floor: same cluster-size multiset, membership permuted (exp_V)."""
    rs = np.random.RandomState(seed)
    return rs.permutation(np.asarray(memb))


REC_FIELDS = ["dataset", "algo", "condition", "spar_seed", "algo_seed", "k",
              "k_ge3", "AMI", "ARI", "NMI", "avgF1_ge3", "gt2det", "det2gt_ge3",
              "realized_ret", "e_used", "resolution", "notes"]


def arm_graphs(g, name):
    """{(arm, target): [(spar_seed, gs, realized)]} — same sparsifiers as main."""
    n, m = g.vcount(), g.ecount()
    edge_arr, scores = dspar_scores(g)
    arms = {}
    for alpha in DSPAR_ALPHAS:
        probs = _probs_calibrated(scores, alpha)
        lst = []
        for ss in SPAR_SEEDS:
            kept, _w = sparsify_calibrated(edge_arr, probs, ss)
            lst.append((ss, build_sparse(n, kept), kept.shape[0] / m))
        arms[("dspar", alpha)] = lst
    E, J, deg = edge_jaccard(g)
    ls = LSpar(g, J=J, E=E, deg=deg)
    e_used = None
    for target in LSPAR_TARGETS:
        e_used, realized, status = ls.bisect_e(target)
        eids = ls.select(e_used)
        gs = build_sparse(n, np.asarray(g.get_edgelist(), dtype=np.int64)[eids])
        arms[("lspar", target)] = [(-1, gs, realized)]
        log(f"  lspar target={target}: e={e_used:.4f} realized={realized:.4f} "
            f"({status}) m'={gs.ecount():,}")
    del E, J, deg, ls, edge_arr, scores
    return arms, e_used


def run_recovery(which):
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 normalized_mutual_info_score)
    out = HERE / "recovery.csv"
    done = done_pairs(out, ["dataset", "algo"])

    # ---------------- email-Eu-core (AMI/ARI/NMI) ----------------
    if "email-Eu-core" in which:
        g, y = load_email_labelled()
        log(f"\n=== RECOVERY email-Eu-core: n={g.vcount()} m={g.ecount()} "
            f"gt_classes={len(set(y.tolist()))}")
        arms, e_used = arm_graphs(g, "email-Eu-core")

        for algo in ALGOS:
            if ("email-Eu-core", algo) in done:
                log(f"  [skip {algo}]")
                continue

            def rec(cond, ss, s, memb, extra):
                memb = np.asarray(memb)
                r = dict(dataset="email-Eu-core", algo=algo, condition=cond,
                         spar_seed=ss, algo_seed=s,
                         k=int(np.unique(memb).size), k_ge3="",
                         AMI=float(adjusted_mutual_info_score(y, memb)),
                         ARI=float(adjusted_rand_score(y, memb)),
                         NMI=float(normalized_mutual_info_score(y, memb)),
                         avgF1_ge3="", gt2det="", det2gt_ge3="", **extra)
                append_row(out, r, REC_FIELDS)
                log(f"  [{algo}] {cond:20s} ss={ss} s={s} k={r['k']:5d} "
                    f"AMI={r['AMI']:.4f} ARI={r['ARI']:.4f} NMI={r['NMI']:.4f}")
                return r

            ks = {}
            base_parts = []
            kk = []
            for s in BASE_SEEDS:
                memb, k, _ = detect(g, algo, s)
                rec("baseline", "", s, memb,
                    dict(realized_ret=1.0, e_used="", resolution=1.0, notes=""))
                base_parts.append(memb)
                kk.append(k)
            ks["baseline"] = float(np.mean(kk))
            for cs in CHANCE_SEEDS:
                rec("chance_baseline", "", cs, size_matched_random(base_parts[0], cs),
                    dict(realized_ret="", e_used="", resolution="",
                         notes="size-matched random partition"))

            for (arm, target), lst in arms.items():
                cond = f"{arm}_{target}"
                kk, first = [], None
                for (ss, gs, realized) in lst:
                    for s in ALGO_SEEDS:
                        memb, k, _ = detect(gs, algo, s)
                        rec(cond, ("" if ss < 0 else ss), s, memb,
                            dict(realized_ret=round(realized, 4),
                                 e_used=(round(e_used, 4) if arm == "lspar" else ""),
                                 resolution=1.0, notes=""))
                        kk.append(k)
                        if first is None:
                            first = memb
                ks[cond] = float(np.mean(kk))
                for cs in CHANCE_SEEDS:
                    rec(f"chance_{cond}", "", cs, size_matched_random(first, cs),
                        dict(realized_ret="", e_used="", resolution="",
                             notes="size-matched random partition"))

            # granularity control: only Louvain has a resolution knob
            if algo == "louvain":
                base_k = ks["baseline"]
                for cond, kv in ks.items():
                    if cond == "baseline":
                        continue
                    if abs(kv - base_k) / base_k > K_TOL:
                        for s in BASE_SEEDS:
                            memb, gamma, kg = louvain_matched(g, s, int(round(kv)))
                            rec(f"resmatch_{cond}", "", s, memb,
                                dict(realized_ret=1.0, e_used="",
                                     resolution=round(gamma, 5),
                                     notes=f"granularity control for {cond} (k={kv:.1f})"))
                    else:
                        log(f"  [louvain] {cond}: k={kv:.1f} vs base {base_k:.1f} "
                            f"within {K_TOL:.0%} -> no resmatch needed")
        del g, arms

    # ---------------- com-DBLP / com-Amazon (avgF1_ge3) ----------------
    for name in ["com-DBLP", "com-Amazon"]:
        if name not in which:
            continue
        g, id_map = load_lcc_graph(name, return_map=True)
        comms = load_ground_truth(DATASETS_DIR / name / f"{name}_labels.txt", id_map)
        gt_sizes = np.array([c.size for c in comms], dtype=np.float64)
        n2c = defaultdict(list)
        for ci, mem in enumerate(comms):
            for node in mem:
                n2c[int(node)].append(ci)
        log(f"\n=== RECOVERY {name}: n={g.vcount():,} m={g.ecount():,} "
            f"gt_comms={len(comms)}")
        arms, e_used = arm_graphs(g, name)

        for algo in ALGOS:
            if (name, algo) in done:
                log(f"  [skip {algo}]")
                continue

            def rec2(cond, ss, s, memb, extra):
                f = average_f1(memb, n2c, gt_sizes)
                r = dict(dataset=name, algo=algo, condition=cond,
                         spar_seed=ss, algo_seed=s,
                         k=f["n_clusters"], k_ge3=f["n_clusters_ge3"],
                         AMI="", ARI="", NMI="",
                         avgF1_ge3=f["avgF1_ge3"], gt2det=f["gt2det"],
                         det2gt_ge3=f["det2gt_ge3"], **extra)
                append_row(out, r, REC_FIELDS)
                log(f"  [{algo}] {cond:20s} ss={ss} s={s} k={f['n_clusters']:7d} "
                    f"k>=3={f['n_clusters_ge3']:7d} avgF1_ge3={f['avgF1_ge3']:.4f} "
                    f"(gt2det={f['gt2det']:.4f} det2gt={f['det2gt_ge3']:.4f})")
                return f

            ks = {}
            kk, first = [], None
            for s in BASE_SEEDS:
                memb, k, _ = detect(g, algo, s)
                f = rec2("baseline", "", s, memb,
                         dict(realized_ret=1.0, e_used="", resolution=1.0, notes=""))
                kk.append(f["n_clusters_ge3"])
                if first is None:
                    first = memb
            ks["baseline"] = float(np.mean(kk))
            for cs in CHANCE_SEEDS:
                rec2("chance_baseline", "", cs, size_matched_random(first, cs),
                     dict(realized_ret="", e_used="", resolution="",
                          notes="size-matched random partition"))

            for (arm, target), lst in arms.items():
                cond = f"{arm}_{target}"
                kk, first = [], None
                for (ss, gs, realized) in lst:
                    for s in ALGO_SEEDS:
                        memb, k, _ = detect(gs, algo, s)
                        f = rec2(cond, ("" if ss < 0 else ss), s, memb,
                                 dict(realized_ret=round(realized, 4),
                                      e_used=(round(e_used, 4) if arm == "lspar" else ""),
                                      resolution=1.0, notes=""))
                        kk.append(f["n_clusters_ge3"])
                        if first is None:
                            first = memb
                ks[cond] = float(np.mean(kk))
                for cs in CHANCE_SEEDS:
                    rec2(f"chance_{cond}", "", cs, size_matched_random(first, cs),
                         dict(realized_ret="", e_used="", resolution="",
                              notes="size-matched random partition"))

            if algo == "louvain":
                base_k = ks["baseline"]
                for cond, kv in ks.items():
                    if cond == "baseline":
                        continue
                    if abs(kv - base_k) / base_k > K_TOL:
                        memb, gamma, kg = louvain_matched(g, BASE_SEEDS[0], int(round(kv)))
                        rec2(f"resmatch_{cond}", "", BASE_SEEDS[0], memb,
                             dict(realized_ret=1.0, e_used="", resolution=round(gamma, 5),
                                  notes=f"granularity control for {cond} (k>=3={kv:.0f})"))
                        # over-matched sweep for the com-Amazon adjudication (as exp_L)
                        if name == "com-Amazon" and cond.startswith("lspar"):
                            for mult in (1.1, 1.2, 1.4):
                                m2, g2, _ = louvain_matched(g, BASE_SEEDS[0],
                                                            int(round(kv * mult)))
                                rec2(f"resmatch_sweep_{cond}", "", BASE_SEEDS[0], m2,
                                     dict(realized_ret=1.0, e_used="",
                                          resolution=round(g2, 5),
                                          notes=f"over-matched x{mult} for {cond}"))
                    else:
                        log(f"  [louvain] {cond}: k>=3={kv:.0f} vs base {base_k:.0f} "
                            f"within {K_TOL:.0%} -> no resmatch needed")
        del g, arms, n2c, comms


# ===========================================================================
# selftest: RNG determinism
# ===========================================================================

def selftest():
    random.seed(7)
    g = ig.Graph.Barabasi(n=800, m=3)
    ok = True
    for algo in ALGOS:
        a, ka, _ = detect(g, algo, 300)
        b, kb, _ = detect(g, algo, 300)
        c, kc, _ = detect(g, algo, 301)
        same = bool(np.array_equal(a, b))
        diff = not bool(np.array_equal(a, c))
        log(f"  {algo:10s} seed300 twice identical={same} (k={ka},{kb})  "
            f"seed301 differs={diff} (k={kc})")
        ok = ok and same
    # interleaving check: a foreign RNG consumer between the two calls
    a, _, _ = detect(g, "infomap", 300)
    _ = [random.random() for _ in range(1000)]
    b, _, _ = detect(g, "infomap", 300)
    log(f"  infomap reseed-after-interleave identical={np.array_equal(a, b)}")
    log(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "main"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if stage == "main":
        run_main(arg.split(",") if arg else NETWORKS)
    elif stage == "recovery":
        run_recovery(arg.split(",") if arg else ["email-Eu-core"])
    elif stage == "selftest":
        raise SystemExit(0 if selftest() else 1)
    else:
        raise SystemExit(f"unknown stage {stage}")
