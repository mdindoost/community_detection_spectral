#!/usr/bin/env python
"""
Experiment V -- "spending the signal".

Exp L established that L-Spar's Jaccard signal genuinely sees community structure
(deltaJ collapses on configuration-model nulls) yet deploying it by DELETION loses
modularity in 14/14 cells.  Exp V asks the constructive question: is there ANY
deployment of that verified signal that survives honest transfer scoring, sharp
nulls, cost matching and resolution-matched recovery?

Arms (pre-registered in DESIGN.md, implemented verbatim):
  A  baseline        plain Leiden on G, seeds 100-104
  B  weighting       w(e) = 1 + J(e), weighted Leiden on the FULL graph, seeds 300-302,
                     partition scored on the UNWEIGHTED original graph
  Bs B-shuffle       identical to B with J permuted across edges (shuffle seeds 500,501)
  C  seeding         L-Spar@0.5 -> Leiden on sparse -> refine on full G (initial_membership)
  D  protected del.  L-Spar core at the largest e with retention <= 0.5, topped up with
                     uniformly random non-selected edges to exactly 0.5 (fill seeds 700,701)
  L  pure L-Spar@0.5 (Exp L bisection; reproduction / reference for D)
  U  uniform@0.5     (reference for D)
  config-null arm    B and C on degree-preserving rewirings (REWIRE_SEED=42, 10 swaps/edge)

Everything reusable (loader, L-Spar, e-bisection, runtime matching, resmatch,
recovery metrics, null conventions) is taken VERBATIM from exp_L_lspar/run.py so
that Exp V's numbers are directly comparable.  Seeded refinement follows
exp_N_enron_anatomy/run.py.

Stages:
  run.py main     [nets]  -> results.csv
  run.py null     [nets]  -> null_arm.csv
  run.py recovery [nets]  -> recovery.csv
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

N_ITER = 2                       # repo convention
BASE_SEEDS = [100, 101, 102, 103, 104]
SPARSE_SEEDS = [300, 301, 302]   # "3 seeds (300-302)" for every treated arm
SHUFFLE_SEEDS = [500, 501]
FILL_SEEDS = [700, 701]
MATCH_SEEDS = list(range(900, 960))
CHANCE_SEEDS = [11, 12, 13]
TARGET = 0.5                     # Exp V works at retention 0.5 only
SWAPS_PER_EDGE = 10
REWIRE_SEED = 42
MIN_GT_SIZE = 3

NETWORKS = ["email-Eu-core", "wiki-Vote", "ca-HepTh", "ca-CondMat",
            "email-Enron", "com-DBLP", "com-Amazon"]
NULL_NETWORKS = ["email-Eu-core", "ca-CondMat", "email-Enron"]


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


# ---------------------------------------------------------------------------
# L-Spar (verbatim from exp_L_lspar/run.py, plus bisect_e_le for arm D)
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

    def bisect_e_le(self, target, iters=40):
        """Largest e whose realized retention is <= target (arm D's protected core)."""
        r0 = self.retention(0.0)
        if r0 > target:
            return 0.0, r0, "floor_above_target"
        lo, r_lo = 0.0, r0
        r1 = self.retention(1.0)
        if r1 <= target:
            return 1.0, r1, "full"
        hi = 1.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            r = self.retention(mid)
            if r <= target:
                lo, r_lo = mid, r
            else:
                hi = mid
        return lo, r_lo, "ok"


def subgraph_from_eids(g, eids):
    gs = ig.Graph(n=g.vcount())
    E = np.asarray(g.get_edgelist(), dtype=np.int64)[eids]
    gs.add_edges(E)
    return gs


# ---------------------------------------------------------------------------
# Leiden (exp_L conventions + weights / initial_membership from exp_N)
# ---------------------------------------------------------------------------

def leiden(g, seed, resolution=None, weights=None, initial_membership=None):
    t0 = time.perf_counter()
    kw = {}
    if weights is not None:
        kw["weights"] = list(np.asarray(weights, dtype=np.float64))
    if initial_membership is not None:
        _, im = np.unique(np.asarray(initial_membership), return_inverse=True)
        kw["initial_membership"] = im.tolist()
    if resolution is None:
        part = la.ModularityVertexPartition(g, **kw)
    else:
        part = la.RBConfigurationVertexPartition(g, resolution_parameter=resolution, **kw)
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
# CSV helpers (incremental, fixed schema)
# ---------------------------------------------------------------------------

FIELDS = [
    "network", "graph", "arm", "n", "m", "m_arm", "n_runs",
    "target_ret", "e_used", "realized_ret", "status",
    "ret_core", "frac_core", "n_fill",
    "Q_base_mean", "Q_base_std", "Q_base_best", "nc_base", "T_leiden_orig",
    "Q_native_mean", "Q_native_std",
    "Q_orig_mean", "Q_orig_std", "Q_orig_best", "Q_pre_refine_mean",
    "nc_arm_mean", "nc_arm_std",
    "dQ_honest_vs_mean", "dQ_honest_vs_best",
    "T_jaccard", "T_prep", "T_leiden_arm", "T_arm_total",
    "budget", "n_restarts", "Q_matched_best", "dQ_vs_matched",
    "pooled_sd", "sig_2sd", "meanJ", "deltaJ", "notes",
]

REC_FIELDS = [
    "dataset", "graph", "condition", "seed", "n_clusters", "n_clusters_ge3",
    "AMI", "ARI", "NMI", "avgF1_ge3", "gt2det", "det2gt_ge3",
    "realized_ret", "e_used", "resolution", "notes",
]


def append_row(path, row, fields):
    new = not path.exists()
    full = {k: row.get(k, "") for k in fields}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(full)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Mechanism (verbatim from exp_L)
# ---------------------------------------------------------------------------

def mechanism(g, memb, E, J, eids):
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


# ---------------------------------------------------------------------------
# Arm bookkeeping: honest transfer + runtime-matched control (exp_L convention)
# ---------------------------------------------------------------------------

def finish_arm(g, out, name, graph_tag, arm, memb_list, Qnat, ncs, tls,
               T_jaccard, T_prep, base, extra=None):
    """memb_list: partitions living on G. Honest score = unweighted g.modularity."""
    Qor = [float(g.modularity(np.asarray(mm).tolist())) for mm in memb_list]
    T_leiden_arm = float(np.mean(tls))
    budget = T_jaccard + T_prep + T_leiden_arm

    spent, best_match, nrest = 0.0, -1.0, 0
    for s in MATCH_SEEDS:
        _, Q, _, dt = leiden(g, s)
        spent += dt
        nrest += 1
        best_match = max(best_match, Q)
        if spent >= budget:
            break

    sd_arm = float(np.std(Qor))
    pooled = float(np.sqrt(0.5 * (sd_arm ** 2 + base["Q_base_std"] ** 2)))
    dQm = float(np.mean(Qor)) - best_match
    row = dict(
        network=name, graph=graph_tag, arm=arm, n=g.vcount(), m=g.ecount(),
        n_runs=len(memb_list),
        Q_native_mean=float(np.mean(Qnat)), Q_native_std=float(np.std(Qnat)),
        Q_orig_mean=float(np.mean(Qor)), Q_orig_std=sd_arm,
        Q_orig_best=float(max(Qor)),
        nc_arm_mean=float(np.mean(ncs)), nc_arm_std=float(np.std(ncs)),
        dQ_honest_vs_mean=float(np.mean(Qor)) - base["Q_base_mean"],
        dQ_honest_vs_best=float(np.mean(Qor)) - base["Q_base_best"],
        T_jaccard=T_jaccard, T_prep=T_prep, T_leiden_arm=T_leiden_arm,
        T_arm_total=budget,
        budget=budget, n_restarts=nrest, Q_matched_best=best_match,
        dQ_vs_matched=dQm, pooled_sd=pooled,
        sig_2sd=int(dQm >= 0.002 and dQm > 2.0 * pooled),
        **base,
    )
    if extra:
        row.update(extra)
    append_row(out, row, FIELDS)
    log(f"  [{arm:14s}] Q_orig={row['Q_orig_mean']:.6f}+-{sd_arm:.6f} "
        f"k={row['nc_arm_mean']:.0f} dQ_vs_mean={row['dQ_honest_vs_mean']:+.6f} "
        f"dQ_vs_best={row['dQ_honest_vs_best']:+.6f} "
        f"dQ_vs_matched={dQm:+.6f} (r={nrest}, budget={budget:.2f}s) "
        f"sig={row['sig_2sd']}")
    return row


# ===========================================================================
# The five treated arms, run on an arbitrary graph (used by main AND null)
# ===========================================================================

def run_arms(g, name, graph_tag, out, arms=("B", "Bs", "C", "D", "L", "U")):
    n, m = g.vcount(), g.ecount()

    # ---- A. baseline, seeds 100-104
    base_Q, base_T, base_nc, base_memb = [], [], [], None
    for s in BASE_SEEDS:
        memb, Q, nc, dt = leiden(g, s)
        base_Q.append(Q); base_T.append(dt); base_nc.append(nc)
        if base_memb is None:
            base_memb = memb
    base = dict(Q_base_mean=float(np.mean(base_Q)), Q_base_std=float(np.std(base_Q)),
                Q_base_best=float(max(base_Q)), nc_base=float(np.mean(base_nc)),
                T_leiden_orig=float(np.mean(base_T)))
    log(f"  [A_baseline    ] Q={base['Q_base_mean']:.6f}+-{base['Q_base_std']:.6f} "
        f"best={base['Q_base_best']:.6f} k={base['nc_base']:.0f} "
        f"T={base['T_leiden_orig']:.3f}s")
    append_row(out, dict(network=name, graph=graph_tag, arm="A_baseline", n=n, m=m,
                         m_arm=m, n_runs=len(BASE_SEEDS), target_ret=1.0,
                         realized_ret=1.0, status="ok",
                         Q_orig_mean=base["Q_base_mean"], Q_orig_std=base["Q_base_std"],
                         Q_orig_best=base["Q_base_best"],
                         Q_native_mean=base["Q_base_mean"], Q_native_std=base["Q_base_std"],
                         nc_arm_mean=base["nc_base"], nc_arm_std=float(np.std(base_nc)),
                         dQ_honest_vs_mean=0.0,
                         dQ_honest_vs_best=base["Q_base_mean"] - base["Q_base_best"],
                         T_leiden_arm=base["T_leiden_orig"],
                         T_arm_total=base["T_leiden_orig"], **base), FIELDS)

    # ---- Jaccard (charged to B, Bs, C, D)
    tj = time.perf_counter()
    E, J, deg = edge_jaccard(g)
    T_jaccard = time.perf_counter() - tj
    mech = mechanism(g, base_memb, E, J, np.arange(m))
    jinfo = dict(meanJ=float(J.mean()),
                 deltaJ=mech["meanJ_intra"] - mech["meanJ_inter"])
    log(f"  jaccard {T_jaccard:.3f}s meanJ={jinfo['meanJ']:.5f} "
        f"deltaJ={jinfo['deltaJ']:+.5f}")

    # ---- B. weighting w = 1 + J
    if "B" in arms:
        tp = time.perf_counter()
        w = 1.0 + J
        T_prep = time.perf_counter() - tp
        mm, qn, kk, tt = [], [], [], []
        for s in SPARSE_SEEDS:
            memb, Qw, nc, dt = leiden(g, s, weights=w)
            mm.append(memb); qn.append(Qw); kk.append(nc); tt.append(dt)
        finish_arm(g, out, name, graph_tag, "B_weight", mm, qn, kk, tt,
                   T_jaccard, T_prep, base,
                   dict(m_arm=m, target_ret=1.0, realized_ret=1.0, status="ok",
                        notes="w=1+J, weighted Leiden, scored unweighted", **jinfo))

    # ---- B-shuffle. sharp null: J permuted across edges
    if "Bs" in arms:
        mm, qn, kk, tt = [], [], [], []
        tp_tot = 0.0
        for sh in SHUFFLE_SEEDS:
            tp = time.perf_counter()
            rs = np.random.RandomState(sh)
            w = 1.0 + rs.permutation(J)
            tp_tot += time.perf_counter() - tp
            for s in SPARSE_SEEDS:
                memb, Qw, nc, dt = leiden(g, s, weights=w)
                mm.append(memb); qn.append(Qw); kk.append(nc); tt.append(dt)
        finish_arm(g, out, name, graph_tag, "Bshuf_weight", mm, qn, kk, tt,
                   T_jaccard, tp_tot / len(SHUFFLE_SEEDS), base,
                   dict(m_arm=m, target_ret=1.0, realized_ret=1.0, status="ok",
                        notes="J permuted across edges (seeds 500,501) x 3 Leiden seeds",
                        **jinfo))

    ls = None
    if any(a in arms for a in ("C", "D", "L")):
        tr = time.perf_counter()
        ls = LSpar(g, J=J, E=E, deg=deg)
        T_rank = time.perf_counter() - tr
        log(f"  rank {T_rank:.3f}s min_ret={ls.retention(0.0):.4f}")

    # ---- C. seeding: L-Spar@0.5 -> Leiden(sparse) -> refine on G
    if "C" in arms:
        e_used, realized, status = ls.bisect_e(TARGET)
        tsel = time.perf_counter()
        eids = ls.select(e_used)
        gs = subgraph_from_eids(g, eids)
        T_sel = time.perf_counter() - tsel
        mm, qn, kk, tt, pre = [], [], [], [], []
        for s in SPARSE_SEEDS:
            memb_s, _, _, t1 = leiden(gs, s)
            pre.append(float(g.modularity(memb_s.tolist())))
            memb_f, Qf, nc, t2 = leiden(g, s, initial_membership=memb_s)
            mm.append(memb_f); qn.append(Qf); kk.append(nc); tt.append(t1 + t2)
        finish_arm(g, out, name, graph_tag, "C_seed", mm, qn, kk, tt,
                   T_jaccard, T_rank + T_sel, base,
                   dict(m_arm=gs.ecount(), target_ret=TARGET, e_used=round(e_used, 6),
                        realized_ret=round(realized, 6), status=status,
                        Q_pre_refine_mean=float(np.mean(pre)),
                        notes="Leiden on L-Spar@0.5 then optimise on G from that membership",
                        **jinfo))
        del gs

    # ---- L. pure L-Spar @ 0.5 (Exp L reproduction; reference for D)
    if "L" in arms:
        e_used, realized, status = ls.bisect_e(TARGET)
        tsel = time.perf_counter()
        eids = ls.select(e_used)
        gs = subgraph_from_eids(g, eids)
        T_sel = time.perf_counter() - tsel
        mm, qn, kk, tt = [], [], [], []
        for s in SPARSE_SEEDS:
            memb, Qs, nc, dt = leiden(gs, s)
            mm.append(memb); qn.append(Qs); kk.append(nc); tt.append(dt)
        finish_arm(g, out, name, graph_tag, "L_lspar05", mm, qn, kk, tt,
                   T_jaccard, T_rank + T_sel, base,
                   dict(m_arm=gs.ecount(), target_ret=TARGET, e_used=round(e_used, 6),
                        realized_ret=round(realized, 6), status=status,
                        notes="pure L-Spar deletion (Exp L reproduction)", **jinfo))
        del gs

    # ---- D. protected deletion at EXACTLY 0.5
    if "D" in arms:
        e_le, r_core, st_core = ls.bisect_e_le(TARGET)
        tsel = time.perf_counter()
        core = ls.select(e_le)
        n_keep = int(round(TARGET * m))
        rest = np.setdiff1d(np.arange(m, dtype=np.int64), core, assume_unique=False)
        n_fill = max(0, n_keep - core.size)
        T_sel = time.perf_counter() - tsel
        mm, qn, kk, tt = [], [], [], []
        tp_tot = 0.0
        for fs in FILL_SEEDS:
            tp = time.perf_counter()
            rs = np.random.RandomState(fs)
            fill = rs.choice(rest, size=min(n_fill, rest.size), replace=False)
            eids = np.concatenate([core, fill])
            gs = subgraph_from_eids(g, eids)
            tp_tot += time.perf_counter() - tp
            for s in SPARSE_SEEDS:
                memb, Qs, nc, dt = leiden(gs, s)
                mm.append(memb); qn.append(Qs); kk.append(nc); tt.append(dt)
            del gs
        finish_arm(g, out, name, graph_tag, "D_protected", mm, qn, kk, tt,
                   T_jaccard, T_rank + T_sel + tp_tot / len(FILL_SEEDS), base,
                   dict(m_arm=n_keep, target_ret=TARGET, e_used=round(e_le, 6),
                        realized_ret=round(n_keep / m, 6), status=st_core,
                        ret_core=round(r_core, 6), frac_core=round(core.size / n_keep, 6),
                        n_fill=n_fill,
                        notes="L-Spar core (largest e with ret<=0.5) + uniform random fill",
                        **jinfo))

    # ---- U. uniform random 50%
    if "U" in arms:
        n_keep = int(round(TARGET * m))
        mm, qn, kk, tt = [], [], [], []
        tp_tot = 0.0
        for fs in FILL_SEEDS:
            tp = time.perf_counter()
            rs = np.random.RandomState(fs)
            eids = np.sort(rs.choice(m, size=n_keep, replace=False))
            gs = subgraph_from_eids(g, eids)
            tp_tot += time.perf_counter() - tp
            for s in SPARSE_SEEDS:
                memb, Qs, nc, dt = leiden(gs, s)
                mm.append(memb); qn.append(Qs); kk.append(nc); tt.append(dt)
            del gs
        finish_arm(g, out, name, graph_tag, "U_uniform05", mm, qn, kk, tt,
                   0.0, tp_tot / len(FILL_SEEDS), base,
                   dict(m_arm=n_keep, target_ret=TARGET,
                        realized_ret=round(n_keep / m, 6), status="ok",
                        notes="uniform random 50% of edges (no J computed, no T_jaccard)"))

    del E, J, ls


# ===========================================================================
# STAGE: main
# ===========================================================================

def run_main(networks):
    out = HERE / "results.csv"
    for name in networks:
        t0 = time.perf_counter()
        g = load_lcc_graph(name)
        log(f"\n{'='*78}\n{name}: n={g.vcount():,} m={g.ecount():,} "
            f"(load {time.perf_counter()-t0:.1f}s)\n{'='*78}")
        run_arms(g, name, "real", out)
        del g


# ===========================================================================
# STAGE: null  (B and C on the degree-preserving rewiring; real arm for anchoring)
# ===========================================================================

def run_null(networks):
    out = HERE / "null_arm.csv"
    for name in networks:
        g = load_lcc_graph(name)
        log(f"\n{'='*78}\nNULL STAGE {name}: n={g.vcount():,} m={g.ecount():,}\n{'='*78}")
        log("-- real graph")
        run_arms(g, name, "real", out, arms=("B", "C"))
        t0 = time.perf_counter()
        gr = g.copy()
        ig.set_random_number_generator(random.Random(REWIRE_SEED))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gr.rewire(n=SWAPS_PER_EDGE * g.ecount(), mode="simple")
        ig.set_random_number_generator(random)
        log(f"-- rewired in {time.perf_counter()-t0:.1f}s")
        del g
        run_arms(gr, name, "null", out, arms=("B", "C"))
        del gr


# ===========================================================================
# STAGE: recovery
# ===========================================================================

def load_email_labelled():
    """email-Eu-core with department labels; loader semantics of exp_F/exp_L."""
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
    _, memb = np.unique(np.asarray(membership), return_inverse=True)
    memb = memb.astype(np.int64)
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
    """Chance floor: same cluster-size multiset, membership permuted across nodes."""
    rs = np.random.RandomState(seed)
    return rs.permutation(np.asarray(memb))


def arm_partitions(g, which):
    """Return {condition: [(seed, membership)]} for the recovery arms on graph g."""
    outp = defaultdict(list)
    if "baseline" in which:
        for s in BASE_SEEDS[:3]:
            memb, _, _, _ = leiden(g, s)
            outp["baseline"].append((s, memb))
    need_J = any(k in which for k in ("B_weight", "C_seed", "L_lspar05"))
    if not need_J:
        return outp, {}
    E, J, deg = edge_jaccard(g)
    info = {}
    if "B_weight" in which:
        w = 1.0 + J
        for s in SPARSE_SEEDS:
            memb, _, _, _ = leiden(g, s, weights=w)
            outp["B_weight"].append((s, memb))
    if "C_seed" in which or "L_lspar05" in which:
        ls = LSpar(g, J=J, E=E, deg=deg)
        e_used, realized, status = ls.bisect_e(TARGET)
        info = dict(e_used=round(e_used, 4), realized_ret=round(realized, 4))
        gs = subgraph_from_eids(g, ls.select(e_used))
        for s in SPARSE_SEEDS:
            memb_s, _, _, _ = leiden(gs, s)
            if "L_lspar05" in which:
                outp["L_lspar05"].append((s, memb_s))
            if "C_seed" in which:
                memb_f, _, _, _ = leiden(g, s, initial_membership=memb_s)
                outp["C_seed"].append((s, memb_f))
        del gs, ls
    del E, J
    return outp, info


def run_recovery(which):
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 normalized_mutual_info_score)
    out = HERE / "recovery.csv"
    ARMS = ["baseline", "B_weight", "C_seed", "L_lspar05"]

    if "email-Eu-core" in which:
        g, y = load_email_labelled()
        log(f"\n=== RECOVERY email-Eu-core: n={g.vcount()} m={g.ecount()} "
            f"gt_classes={len(set(y.tolist()))}")
        parts, info = arm_partitions(g, ARMS)

        def rec(cond, seed, memb, extra):
            memb = np.asarray(memb)
            r = dict(dataset="email-Eu-core", graph="real", condition=cond, seed=seed,
                     n_clusters=int(len(set(memb.tolist()))),
                     AMI=float(adjusted_mutual_info_score(y, memb)),
                     ARI=float(adjusted_rand_score(y, memb)),
                     NMI=float(normalized_mutual_info_score(y, memb)), **extra)
            append_row(out, r, REC_FIELDS)
            log(f"  {cond:22s} s={seed} k={r['n_clusters']:4d} AMI={r['AMI']:.4f} "
                f"ARI={r['ARI']:.4f} NMI={r['NMI']:.4f}")
            return r

        ks = {}
        for cond in ARMS:
            kk = []
            for s, memb in parts[cond]:
                ex = dict(realized_ret=1.0, resolution=1.0)
                if cond in ("C_seed", "L_lspar05"):
                    ex.update(info)
                    if cond == "C_seed":
                        ex["realized_ret"] = 1.0
                        ex["notes"] = f"seeded from L-Spar ret={info['realized_ret']}"
                r = rec(cond, s, memb, ex)
                kk.append(r["n_clusters"])
            ks[cond] = float(np.mean(kk))
            # chance floor (size-matched random partitions, 3 draws)
            for cs in CHANCE_SEEDS:
                rec(f"chance_{cond}", cs, size_matched_random(parts[cond][0][1], cs),
                    dict(realized_ret="", resolution="",
                         notes="size-matched random partition"))
        base_k = ks["baseline"]
        for cond in ARMS[1:]:
            if abs(ks[cond] - base_k) / base_k > 0.20:
                for s in BASE_SEEDS[:3]:
                    memb, gamma, nc = leiden_matched(g, s, int(round(ks[cond])))
                    rec(f"resmatch_{cond}", s, memb,
                        dict(realized_ret=1.0, resolution=round(gamma, 5),
                             notes=f"granularity control for {cond} (k={ks[cond]:.1f})"))
            else:
                log(f"  ({cond}: k={ks[cond]:.1f} vs base {base_k:.1f}, within 20% "
                    f"-> no resolution-matched control needed)")
        del g

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
        parts, info = arm_partitions(g, ARMS)

        def rec2(cond, seed, memb, extra):
            f = average_f1(memb, n2c, gt_sizes)
            r = dict(dataset=name, graph="real", condition=cond, seed=seed,
                     n_clusters=f["n_clusters"], n_clusters_ge3=f["n_clusters_ge3"],
                     avgF1_ge3=f["avgF1_ge3"], gt2det=f["gt2det"],
                     det2gt_ge3=f["det2gt_ge3"], **extra)
            append_row(out, r, REC_FIELDS)
            log(f"  {cond:22s} s={seed} k={f['n_clusters']:7d} k>=3={f['n_clusters_ge3']:7d} "
                f"avgF1_ge3={f['avgF1_ge3']:.4f} (gt2det={f['gt2det']:.4f} "
                f"det2gt={f['det2gt_ge3']:.4f})")
            return f

        ks = {}
        for cond in ARMS:
            kk = []
            for s, memb in parts[cond]:
                ex = dict(realized_ret=1.0, resolution=1.0)
                if cond in ("C_seed", "L_lspar05"):
                    ex.update(info)
                    if cond == "C_seed":
                        ex["realized_ret"] = 1.0
                        ex["notes"] = f"seeded from L-Spar ret={info['realized_ret']}"
                f = rec2(cond, s, memb, ex)
                kk.append(f["n_clusters_ge3"])
            ks[cond] = float(np.mean(kk))
            for cs in CHANCE_SEEDS:
                rec2(f"chance_{cond}", cs,
                     size_matched_random(parts[cond][0][1], cs),
                     dict(realized_ret="", resolution="",
                          notes="size-matched random partition"))
        base_k = ks["baseline"]
        for cond in ARMS[1:]:
            if abs(ks[cond] - base_k) / base_k > 0.20:
                for s in BASE_SEEDS[:1]:
                    memb, gamma, nc = leiden_matched(g, s, int(round(ks[cond])))
                    rec2(f"resmatch_{cond}", s, memb,
                         dict(realized_ret=1.0, resolution=round(gamma, 5),
                              notes=f"granularity control for {cond} (k>=3={ks[cond]:.0f})"))
                    # over-matched sweep (com-Amazon adjudication, as in Exp L)
                    if name == "com-Amazon" and cond == "L_lspar05":
                        for mult in (1.1, 1.2, 1.4):
                            memb2, g2, _ = leiden_matched(g, s,
                                                          int(round(ks[cond] * mult)))
                            rec2(f"resmatch_sweep_{cond}", s, memb2,
                                 dict(realized_ret=1.0, resolution=round(g2, 5),
                                      notes=f"over-matched x{mult}"))
            else:
                log(f"  ({cond}: k>=3={ks[cond]:.0f} vs base {base_k:.0f}, within 20% "
                    f"-> no resolution-matched control needed)")
        del g, parts


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
