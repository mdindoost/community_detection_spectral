#!/usr/bin/env python
"""
Experiment L: L-Spar (Satuluri, Parthasarathy & Ruan, SIGMOD 2011) through our
full evaluation protocol.

The founding claim of sparsify-then-detect is theirs:
  "excellent speedups (often in the range 10-50), with little or no deterioration
   in the quality of the resulting clusters. In fact, for at least two of the four
   clustering algorithms, our sparsification consistently enables higher clustering
   accuracies."

Our paper has so far only tested degree-based (DSpar) and uniform sparsification.
This experiment runs the *similarity-based* original through the same controls:
  (a) fixed-objective / honest transfer scoring + runtime matching
  (b) mechanism (Jaccard separation, preferential retention) + configuration-model null
  (c) resolution-matched, chance-corrected ground-truth recovery
  (d) honest cost accounting (Jaccard computation is part of the price)

L-SPAR, implemented faithfully:
  * for every edge (u,v): J(u,v) = |N(u) cap N(v)| / |N(u) cup N(v)|
    (exact Jaccard; the 2011 paper approximates it with minhash -- exact is
     *more* favourable to L-Spar quality-wise and we charge it the exact cost)
  * every node i ranks its incident edges by J descending and selects the top
    ceil(d_i^e) of them, e in (0,1)  [paper's local top-k rule]
  * an edge survives if EITHER endpoint selects it (union)
  * e is bisected per network to hit the target realized retention
  * deterministic given e  -> no sparsification seeds needed (noted in SUMMARY)

Stages:
  run.py main       -> results.csv   (parts a, b, d)
  run.py null       -> null_arm.csv  (part b null control)
  run.py recovery   -> recovery.csv  (part c)
"""

import csv
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import igraph as ig
import leidenalg as la

REPO = Path("/home/md724/community_detection_spectral")
DATASETS_DIR = REPO / "datasets"
DATA = REPO / "data"
HERE = Path(__file__).resolve().parent

N_ITER = 2                      # repo convention
BASE_SEEDS = [100, 101, 102, 103, 104]
SPARSE_SEEDS = [300, 301, 302]
MATCH_SEEDS = list(range(900, 960))
TARGETS = [0.5, 0.2]
SWAPS_PER_EDGE = 10
REWIRE_SEED = 42
MIN_GT_SIZE = 3

NETWORKS = ["email-Eu-core", "wiki-Vote", "ca-HepTh", "ca-CondMat",
            "email-Enron", "com-DBLP", "com-Amazon"]
NULL_NETWORKS = ["email-Eu-core", "ca-CondMat", "email-Enron"]


# ---------------------------------------------------------------------------
# Loading (verbatim from exp_E_delta_star/run.py / exp_B_config_null/run.py)
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


# ---------------------------------------------------------------------------
# L-Spar
# ---------------------------------------------------------------------------

def edge_jaccard(g, chunk=100_000):
    """Exact Jaccard similarity of endpoint neighbourhoods, for every edge."""
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
    """Local top-ceil(d^e) ranking by Jaccard; edge kept if either endpoint picks it."""

    def __init__(self, g, J=None, E=None, deg=None):
        self.n, self.m = g.vcount(), g.ecount()
        if J is None:
            E, J, deg = edge_jaccard(g)
        self.E, self.J, self.deg = E, J, deg
        node = np.concatenate([E[:, 0], E[:, 1]])
        eid = np.concatenate([np.arange(self.m), np.arange(self.m)])
        jj = np.concatenate([J, J])
        order = np.lexsort((-jj, node))          # node asc, then J desc
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
        # NOTE: e must be exactly 0.0 for the floor: ceil(d**1e-9) == 2, not 1,
        # because d**1e-9 > 1 in floating point for every d > 1.
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


def subgraph_from_eids(g, eids):
    gs = ig.Graph(n=g.vcount())
    E = np.asarray(g.get_edgelist(), dtype=np.int64)[eids]
    gs.add_edges(E)
    return gs


# ---------------------------------------------------------------------------
# Leiden
# ---------------------------------------------------------------------------

def leiden(g, seed, resolution=None):
    t0 = time.perf_counter()
    if resolution is None:
        part = la.ModularityVertexPartition(g)
    else:
        part = la.RBConfigurationVertexPartition(g, resolution_parameter=resolution)
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    return (np.asarray(part.membership), float(part.modularity),
            len(set(part.membership)), time.perf_counter() - t0)


def leiden_matched(g, seed, target, tol=0.05, max_iter=30):
    """Leiden on the ORIGINAL graph, gamma bisected so #clusters ~= target."""
    def n_of(gamma):
        memb, _, nc, _ = leiden(g, seed, resolution=gamma)
        return memb, nc

    lo, hi = 1.0, 1.0
    memb, nc = n_of(hi)
    best = (abs(nc - target), memb, hi, nc)
    it = 0
    if nc > target:
        while nc > target and lo > 1e-4 and it < max_iter:
            hi = lo
            lo /= 2.0
            memb, nc = n_of(lo)
            it += 1
            if abs(nc - target) < best[0]:
                best = (abs(nc - target), memb, lo, nc)
    else:
        while nc < target and hi < 1e5 and it < max_iter:
            lo = hi
            hi *= 2.0
            memb, nc = n_of(hi)
            it += 1
            if abs(nc - target) < best[0]:
                best = (abs(nc - target), memb, hi, nc)
    for _ in range(max_iter - it):
        if best[0] <= max(1.0, tol * target):
            break
        mid = 0.5 * (lo + hi)
        memb, nm = n_of(mid)
        if abs(nm - target) < best[0]:
            best = (abs(nm - target), memb, mid, nm)
        if nm < target:
            lo = mid
        else:
            hi = mid
    return best[1], best[2], best[3]


# ---------------------------------------------------------------------------
# CSV helpers (incremental)
# ---------------------------------------------------------------------------

def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Mechanism block (shared by main + null)
# ---------------------------------------------------------------------------

def mechanism(g, memb, E, J, eids):
    """Jaccard separation + preferential retention under the fixed partition."""
    memb = np.asarray(memb)
    intra = memb[E[:, 0]] == memb[E[:, 1]]
    n_intra, n_inter = int(intra.sum()), int((~intra).sum())
    kept = np.zeros(E.shape[0], dtype=bool)
    kept[eids] = True
    return dict(
        n_intra=n_intra, n_inter=n_inter,
        meanJ_intra=float(J[intra].mean()) if n_intra else float("nan"),
        meanJ_inter=float(J[~intra].mean()) if n_inter else float("nan"),
        keep_intra_frac=float(kept[intra].mean()) if n_intra else float("nan"),
        keep_inter_frac=float(kept[~intra].mean()) if n_inter else float("nan"),
    )


# ===========================================================================
# STAGE: main
# ===========================================================================

def run_main(networks):
    out = HERE / "results.csv"
    for name in networks:
        t0 = time.perf_counter()
        g = load_lcc_graph(name)
        n, m = g.vcount(), g.ecount()
        log(f"\n{'='*78}\n{name}: n={n:,} m={m:,}  (load {time.perf_counter()-t0:.1f}s)\n{'='*78}")

        # ---- baseline Leiden, 5 seeds
        base_Q, base_T, base_nc, base_memb = [], [], [], None
        for s in BASE_SEEDS:
            memb, Q, nc, dt = leiden(g, s)
            base_Q.append(Q); base_T.append(dt); base_nc.append(nc)
            if base_memb is None:
                base_memb = memb
        Qb_mean, Qb_std, Qb_best = float(np.mean(base_Q)), float(np.std(base_Q)), float(max(base_Q))
        T_leiden_orig = float(np.mean(base_T))
        log(f"  baseline Q={Qb_mean:.6f}+-{Qb_std:.6f} best={Qb_best:.6f} "
            f"nc={np.mean(base_nc):.0f} T={T_leiden_orig:.2f}s")
        Q_fixed_orig = float(g.modularity(base_memb.tolist()))

        # ---- Jaccard (charged once; the honest price of L-Spar)
        tj = time.perf_counter()
        E, J, deg = edge_jaccard(g)
        T_jaccard = time.perf_counter() - tj
        tr = time.perf_counter()
        ls = LSpar(g, J=J, E=E, deg=deg)
        T_rank = time.perf_counter() - tr
        min_ret = ls.retention(0.0)
        ret_e05 = ls.retention(0.5)
        log(f"  jaccard {T_jaccard:.2f}s  rank {T_rank:.2f}s  "
            f"meanJ={J.mean():.4f}  min_ret(e->0)={min_ret:.4f}  ret(e=0.5)={ret_e05:.4f}")

        for target in TARGETS:
            e_used, realized, status = ls.bisect_e(target)
            ts = time.perf_counter()
            eids = ls.select(e_used)
            T_select = time.perf_counter() - ts
            T_sparsify = T_jaccard + T_rank + T_select
            gs = subgraph_from_eids(g, eids)
            log(f"  -- target={target} e={e_used:.4f} realized={realized:.4f} ({status}) "
                f"m'={gs.ecount():,}  T_sparsify={T_sparsify:.2f}s")

            # (a) honest transfer
            Qsp, Qor, ncs, Ts = [], [], [], []
            for s in SPARSE_SEEDS:
                memb, Q_s, nc, dt = leiden(gs, s)
                Qsp.append(Q_s)
                Qor.append(float(g.modularity(memb.tolist())))
                ncs.append(nc); Ts.append(dt)
            Qsp_m = float(np.mean(Qsp)); Qor_m = float(np.mean(Qor))
            T_leiden_sparse = float(np.mean(Ts))

            # runtime-matched baseline: equal wall clock to the WHOLE pipeline
            budget = T_sparsify + T_leiden_sparse
            spent, best_match, nrest = 0.0, -1.0, 0
            for s in MATCH_SEEDS:
                _, Q, _, dt = leiden(g, s)
                spent += dt; nrest += 1
                best_match = max(best_match, Q)
                if spent >= budget:
                    break
            # honest 'best-of-2 plain Leiden' reference too
            Q_best2 = float(max(base_Q[:2]))

            # (b) mechanism under the FIXED baseline partition
            Q_fixed_sparse = float(gs.modularity(base_memb.tolist()))
            mech = mechanism(g, base_memb, E, J, eids)
            pres = (mech["keep_intra_frac"] / mech["keep_inter_frac"]
                    if mech["keep_inter_frac"] else float("nan"))

            row = dict(
                network=name, n=n, m=m, avg_deg=2.0 * m / n,
                target_ret=target, e_used=round(e_used, 6),
                realized_ret=round(realized, 6), status=status,
                min_ret=round(min_ret, 6), ret_at_e0p5=round(ret_e05, 6),
                m_sparse=gs.ecount(),
                Q_base_mean=Qb_mean, Q_base_std=Qb_std, Q_base_best=Qb_best,
                Q_best_of_2=Q_best2, nc_base=float(np.mean(base_nc)),
                T_leiden_orig=T_leiden_orig,
                T_jaccard=T_jaccard, T_rank=T_rank, T_select=T_select,
                T_sparsify=T_sparsify,
                Q_sparse_PL=Qsp_m, Q_sparse_PL_std=float(np.std(Qsp)),
                Q_orig_PL=Qor_m, Q_orig_PL_std=float(np.std(Qor)),
                nc_sparse=float(np.mean(ncs)),
                T_leiden_sparse=T_leiden_sparse,
                dQ_naive=Qsp_m - Qb_mean,
                dQ_honest_vs_mean=Qor_m - Qb_mean,
                dQ_honest_vs_best=Qor_m - Qb_best,
                budget=budget, n_restarts=nrest, Q_matched_best=best_match,
                dQ_vs_matched=Qor_m - best_match,
                Q_fixed_orig=Q_fixed_orig, Q_fixed_sparse=Q_fixed_sparse,
                dQ_fixed=Q_fixed_sparse - Q_fixed_orig,
                meanJ_intra=mech["meanJ_intra"], meanJ_inter=mech["meanJ_inter"],
                deltaJ=mech["meanJ_intra"] - mech["meanJ_inter"],
                keep_intra_frac=mech["keep_intra_frac"],
                keep_inter_frac=mech["keep_inter_frac"],
                pres_ratio=pres,
                n_intra=mech["n_intra"], n_inter=mech["n_inter"],
                T_pipeline=T_sparsify + T_leiden_sparse,
                speedup_vs_single_leiden=T_leiden_orig / (T_sparsify + T_leiden_sparse),
                speedup_leiden_only=T_leiden_orig / T_leiden_sparse,
            )
            append_row(out, row)
            log(f"     Q_sparse(P_L)={Qsp_m:.6f}  Q_orig(P_L)={Qor_m:.6f}  "
                f"dQ_vs_mean={row['dQ_honest_vs_mean']:+.6f}  "
                f"dQ_vs_best={row['dQ_honest_vs_best']:+.6f}  "
                f"dQ_vs_matched={row['dQ_vs_matched']:+.6f} (r={nrest})")
            log(f"     dQ_fixed={row['dQ_fixed']:+.6f}  deltaJ={row['deltaJ']:+.5f}  "
                f"pres={pres:.3f}  speedup(pipeline)={row['speedup_vs_single_leiden']:.2f}x "
                f"speedup(leiden-only)={row['speedup_leiden_only']:.2f}x")
            del gs
        del g, ls, E, J


# ===========================================================================
# STAGE: null
# ===========================================================================

def one_arm(g, tag, name):
    rows = []
    memb, Qb, nc, T = leiden(g, BASE_SEEDS[0])
    E, J, deg = edge_jaccard(g)
    ls = LSpar(g, J=J, E=E, deg=deg)
    Q_fixed_orig = float(g.modularity(memb.tolist()))
    min_ret = ls.retention(0.0)
    for target in TARGETS:
        e_used, realized, status = ls.bisect_e(target)
        eids = ls.select(e_used)
        gs = subgraph_from_eids(g, eids)
        mech = mechanism(g, memb, E, J, eids)
        pres = (mech["keep_intra_frac"] / mech["keep_inter_frac"]
                if mech["keep_inter_frac"] else float("nan"))
        rows.append(dict(
            network=name, arm=tag, n=g.vcount(), m=g.ecount(),
            target_ret=target, e_used=round(e_used, 6),
            realized_ret=round(realized, 6), status=status, min_ret=round(min_ret, 6),
            nc_P0=nc, Q_fixed_orig=Q_fixed_orig,
            Q_fixed_sparse=float(gs.modularity(memb.tolist())),
            dQ_fixed=float(gs.modularity(memb.tolist())) - Q_fixed_orig,
            meanJ_all=float(J.mean()),
            meanJ_intra=mech["meanJ_intra"], meanJ_inter=mech["meanJ_inter"],
            deltaJ=mech["meanJ_intra"] - mech["meanJ_inter"],
            keep_intra_frac=mech["keep_intra_frac"],
            keep_inter_frac=mech["keep_inter_frac"], pres_ratio=pres,
            n_intra=mech["n_intra"], n_inter=mech["n_inter"],
        ))
        del gs
    return rows


def run_null(networks):
    out = HERE / "null_arm.csv"
    for name in networks:
        g = load_lcc_graph(name)
        log(f"\n=== NULL ARM {name}: n={g.vcount():,} m={g.ecount():,}")
        for r in one_arm(g, "real", name):
            append_row(out, r)
            log(f"  real  t={r['target_ret']} ret={r['realized_ret']:.4f} "
                f"dQ_fixed={r['dQ_fixed']:+.6f} deltaJ={r['deltaJ']:+.5f} "
                f"pres={r['pres_ratio']:.3f}")
        t0 = time.perf_counter()
        gr = g.copy()
        ig.set_random_number_generator(random.Random(REWIRE_SEED))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gr.rewire(n=SWAPS_PER_EDGE * g.ecount(), mode="simple")
        ig.set_random_number_generator(random)
        log(f"  rewired in {time.perf_counter()-t0:.1f}s")
        for r in one_arm(gr, "null", name):
            append_row(out, r)
            log(f"  null  t={r['target_ret']} ret={r['realized_ret']:.4f} "
                f"dQ_fixed={r['dQ_fixed']:+.6f} deltaJ={r['deltaJ']:+.5f} "
                f"pres={r['pres_ratio']:.3f}")
        del g, gr


# ===========================================================================
# STAGE: recovery
# ===========================================================================

def load_email_labelled():
    """email-Eu-core with department labels; loader semantics of exp_F."""
    import networkx as nx
    G_raw = nx.Graph()
    with open(DATA / "email-Eu-core.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2 and p[0] != p[1]:
                G_raw.add_edge(int(p[0]), int(p[1]))
    gt = {}
    with open(DATA / "email-Eu-core-department-labels.txt") as f:
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


def average_f1(membership, n2c, gt_sizes):
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


def run_recovery(which):
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 normalized_mutual_info_score)
    out = HERE / "recovery.csv"

    if "email-Eu-core" in which:
        g, y = load_email_labelled()
        log(f"\n=== RECOVERY email-Eu-core: n={g.vcount()} m={g.ecount()} "
            f"gt_classes={len(set(y.tolist()))}")
        E, J, deg = edge_jaccard(g)
        ls = LSpar(g, J=J, E=E, deg=deg)

        def rec(cond, seed, memb, extra):
            r = dict(dataset="email-Eu-core", condition=cond, seed=seed,
                     n_clusters=int(len(set(np.asarray(memb).tolist()))),
                     AMI=float(adjusted_mutual_info_score(y, memb)),
                     ARI=float(adjusted_rand_score(y, memb)),
                     NMI=float(normalized_mutual_info_score(y, memb)),
                     avgF1_ge3="", **extra)
            append_row(out, r)
            log(f"  {cond:22s} s={seed} nc={r['n_clusters']:4d} "
                f"AMI={r['AMI']:.4f} ARI={r['ARI']:.4f} NMI={r['NMI']:.4f}")
            return r

        base_nc = []
        for s in BASE_SEEDS:
            memb, _, nc, _ = leiden(g, s)
            rec("baseline", s, memb, dict(realized_ret=1.0, e_used="", resolution=1.0))
            base_nc.append(nc)
        base_nc = float(np.mean(base_nc))

        for target in TARGETS:
            e_used, realized, status = ls.bisect_e(target)
            gs = subgraph_from_eids(g, ls.select(e_used))
            ncs = []
            for s in SPARSE_SEEDS:
                memb, _, nc, _ = leiden(gs, s)
                rec(f"lspar_{target}", s, memb,
                    dict(realized_ret=round(realized, 4), e_used=round(e_used, 4),
                         resolution=1.0))
                ncs.append(nc)
            mean_nc = float(np.mean(ncs))
            log(f"  cluster counts: base={base_nc:.1f} lspar_{target}={mean_nc:.1f} "
                f"ratio={mean_nc/base_nc:.3f}")
            if abs(mean_nc - base_nc) / base_nc > 0.20:
                for s in BASE_SEEDS[:3]:
                    memb, gamma, nc = leiden_matched(g, s, int(round(mean_nc)))
                    rec(f"resmatch_{target}", s, memb,
                        dict(realized_ret=1.0, e_used="", resolution=round(gamma, 5)))
            else:
                log(f"  (cluster counts within 20% -> no resolution-matched control needed)")

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
        E, J, deg = edge_jaccard(g)
        ls = LSpar(g, J=J, E=E, deg=deg)

        def rec2(cond, seed, memb, extra):
            f = average_f1(memb, n2c, gt_sizes)
            r = dict(dataset=name, condition=cond, seed=seed,
                     n_clusters=f["n_clusters"], AMI="", ARI="", NMI="",
                     avgF1_ge3=f["avgF1_ge3"], **extra)
            append_row(out, r)
            log(f"  {cond:22s} s={seed} nc={f['n_clusters']:7d} "
                f"nc_ge3={f['n_clusters_ge3']:7d} avgF1_ge3={f['avgF1_ge3']:.4f} "
                f"(gt2det={f['gt2det']:.4f} det2gt={f['det2gt_ge3']:.4f})")
            return f

        base_nc = []
        for s in BASE_SEEDS[:3]:
            memb, _, nc, _ = leiden(g, s)
            f = rec2("baseline", s, memb,
                     dict(realized_ret=1.0, e_used="", resolution=1.0))
            base_nc.append(f["n_clusters_ge3"])
        base_nc = float(np.mean(base_nc))

        for target in TARGETS:
            e_used, realized, status = ls.bisect_e(target)
            gs = subgraph_from_eids(g, ls.select(e_used))
            ncs = []
            for s in SPARSE_SEEDS:
                memb, _, nc, _ = leiden(gs, s)
                f = rec2(f"lspar_{target}", s, memb,
                         dict(realized_ret=round(realized, 4),
                              e_used=round(e_used, 4), resolution=1.0))
                ncs.append(f["n_clusters_ge3"])
            mean_nc = float(np.mean(ncs))
            log(f"  clusters_ge3: base={base_nc:.0f} lspar={mean_nc:.0f} "
                f"ratio={mean_nc/base_nc:.3f}")
            if abs(mean_nc - base_nc) / base_nc > 0.20:
                for s in BASE_SEEDS[:1]:
                    memb, gamma, nc = leiden_matched(g, s, int(round(mean_nc)))
                    rec2(f"resmatch_{target}", s, memb,
                         dict(realized_ret=1.0, e_used="", resolution=round(gamma, 5)))
        del g, ls, E, J


# ===========================================================================

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "main"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if stage == "main":
        run_main(arg.split(",") if arg else NETWORKS)
    elif stage == "null":
        run_null(arg.split(",") if arg else NULL_NETWORKS)
    elif stage == "recovery":
        run_recovery(arg.split(",") if arg else ["email-Eu-core"])
    else:
        raise SystemExit(f"unknown stage {stage}")
