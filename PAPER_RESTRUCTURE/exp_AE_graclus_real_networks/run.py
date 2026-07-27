#!/usr/bin/env python
"""
Experiment AB -- sparsifier coverage completion.

Closes the two coverage gaps the literature steelman exposed:
  (i)  Chen et al. (PVLDB 2024) name six top clustering-fidelity preservers; we had
       tested two.  Here we add K-Neighbor, Local Degree and Local Similarity.
  (ii) every sparsifier in the study can disconnect the graph.  A connectivity-
       preserving (MST-backbone) sparsifier structurally cannot fragment; that class
       had never been tested.  Arms 4/5 add it, with and without the Jaccard signal.

Arms (all at matched realized retention):
  kn           K-Neighbor  (Sadhanala 2016 / Chen et al. sec 2.3.2) -- OUR implementation
  ld           Local Degree (Hamann 2016)                           -- NetworKit reference
  lsim         Local Similarity (Satuluri 2011, rank-transformed)   -- NetworKit reference
  mst_jaccard  MST on (1 - Jaccard), kept unconditionally, filled by highest Jaccard
  mst_random   identical backbone, random fill  (isolates backbone from signal)
  lspar        L-Spar (exp_L machinery, verbatim)                   -- reference arm
  dspar        DSpar, calibrated sampler (exp_K/exp_N, verbatim)    -- reference arm

Everything is scored by HONEST TRANSFER on the ORIGINAL graph.  Per DESIGN + the exp_AA
lesson, dQ is reported against three baselines: the base mean, the best-of-5 restart, and
the runtime-matched best-of-N.  Fragment counts (nodes in sub-10-node clusters) are
reported for every arm, since P2 rests on them.

Usage:  run.py net <network>          # everything for one network (streaming driver)
        run.py net <network> --metis  # additionally run the fixed-k Metis detector
"""

import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import shutil
import subprocess
import tempfile
import numpy as np
import scipy.sparse as sp
import igraph as ig
import leidenalg as la

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATASETS_DIR = REPO / "datasets"

# ---------------------------------------------------------------------------
# constants (repo conventions)
# ---------------------------------------------------------------------------
N_ITER = 2
BASE_SEEDS = [100, 101, 102, 103, 104]     # best-of-5 restarts on the original graph
REP_SEEDS = [300, 301, 302]                # 3 replicates per arm
MATCH_SEEDS = list(range(900, 1000))       # runtime-matched restart pool
METIS_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # exp_AD: 10 seeds (exp_AA metis_noise)
TARGETS = [0.5, 0.2]
FRAG_MAX = 10                              # clusters with size < FRAG_MAX are fragments
RESMATCH_TRIGGER = 0.25                    # |dk|/k above this -> resolution-matched control
RESMATCH_MAX_CALLS = 22
RESMATCH_TIME_BUDGET = 900.0               # per resolution-match bisection
RESMATCH_NETWORK_BUDGET = 7200.0           # total per network; beyond this we skip+flag
MIN_GT_SIZE = 3
CHANCE_SEED = 7

ARMS = ["kn", "ld", "lsim", "mst_jaccard", "mst_random", "lspar", "dspar"]
STOCHASTIC = {"kn", "mst_random", "dspar"}
NEEDS_BACKBONE = {"mst_jaccard", "mst_random"}

LABELLED_HARD = {"email-Eu-core"}
LABELLED_OVERLAP = {"com-DBLP", "com-Amazon"}


METIS_ONLY = False
DETECTOR = "metis"            # exp_AE: "metis" or "graclus"
GRACLUS_BIN = os.environ.get("GRACLUS_BIN", str(REPO / "bin" / "graclus"))
# Graclus is deterministic, so a seed sweep over the partitioner is vacuous.
# For it the sweep runs over the SPARSIFIER instead (REP_SEEDS, three
# replicates) and the baseline is a single exact run, which is its own best.
GRACLUS_SEEDS = [0, 1, 2]


def log(*a):
    print(*a, flush=True)


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


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


def load_hard_labels(name, id_map, n):
    """Two-column <node> <label> file -> integer label vector aligned to LCC ids."""
    path = DATASETS_DIR / name / f"{name}_labels.txt"
    y = np.full(n, -1, dtype=np.int64)
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                o = int(p[0])
                if o in id_map:
                    y[id_map[o]] = int(p[1])
    return y


def load_overlap_gt(name, id_map):
    """SNAP overlapping ground-truth communities -> (gt_sizes, node->communities)."""
    comms = []
    with open(DATASETS_DIR / name / f"{name}_labels.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            members = [id_map[int(t)] for t in line.split() if int(t) in id_map]
            if len(members) >= MIN_GT_SIZE:
                comms.append(np.unique(np.asarray(members, dtype=np.int64)))
    gt_sizes = np.array([c.size for c in comms], dtype=np.float64)
    n2c = defaultdict(list)
    for ci, mem in enumerate(comms):
        for node in mem:
            n2c[int(node)].append(ci)
    return gt_sizes, n2c


def average_f1(membership, n2c, gt_sizes):
    """exp_L_lspar/run.py verbatim: symmetric best-match F1, detected side >=3 nodes."""
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


# ---------------------------------------------------------------------------
# Detectors
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


def _csr_adj(g):
    n, m = g.vcount(), g.ecount()
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    if m == 0:
        return np.zeros(n + 1, dtype=np.int32), np.zeros(0, dtype=np.int32)
    rows = np.concatenate([E[:, 0], E[:, 1]])
    cols = np.concatenate([E[:, 1], E[:, 0]])
    A = sp.csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    return A.indptr.astype(np.int32), A.indices.astype(np.int32)


def metis_partition(g, k, seed=0):
    """Real Metis via pymetis (exp_AA verbatim); k is fixed by construction."""
    import pymetis
    t0 = time.perf_counter()
    xadj, adjncy = _csr_adj(g)
    try:
        opts = pymetis.Options()
        opts.seed = int(seed)
        _, memb = pymetis.part_graph(k, xadj=xadj, adjncy=adjncy, options=opts)
    except Exception:
        _, memb = pymetis.part_graph(k, xadj=xadj, adjncy=adjncy)
    return np.asarray(memb, dtype=np.int64), time.perf_counter() - t0


def graclus_partition(g, k, seed=0):
    """Graclus 1.2 (Dhillon, Guan & Kulis 2007), fixed k, normalized cut.

    Unlike Metis it imposes NO balance constraint on part sizes; that is the
    reason this arm exists. Deterministic, so `seed` is accepted for signature
    compatibility with metis_partition and ignored.
    """
    n = g.vcount()
    t0 = time.perf_counter()
    tmpd = tempfile.mkdtemp(prefix="graclus_")
    try:
        gf = os.path.join(tmpd, "g.graph")
        # METIS adjacency format: header "n m" (m = undirected edge count), then
        # one line per vertex listing its 1-indexed neighbours.
        with open(gf, "w") as f:
            f.write("%d %d\n" % (n, g.ecount()))
            for nb in g.get_adjlist():
                f.write(" ".join(str(x + 1) for x in nb))
                f.write("\n")
        # graclus strips the directory from its argument and writes
        # "<basename>.part.<k>" into the CURRENT WORKING DIRECTORY, so it must be
        # run with cwd set to the scratch directory and given a bare filename.
        r = subprocess.run([GRACLUS_BIN, "g.graph", str(int(k))], cwd=tmpd,
                           capture_output=True, text=True, timeout=10800)
        pf = os.path.join(tmpd, "g.graph.part.%d" % int(k))
        if not os.path.exists(pf):
            raise RuntimeError("graclus wrote no partition (rc=%s)\nSTDOUT:%s\nSTDERR:%s"
                               % (r.returncode, r.stdout[-500:], r.stderr[-500:]))
        memb = np.loadtxt(pf, dtype=np.int64)
        memb = np.atleast_1d(memb)
        if memb.shape[0] != n:
            raise RuntimeError("graclus returned %d labels for %d vertices"
                               % (memb.shape[0], n))
        return memb, time.perf_counter() - t0
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def detect_fixed_k(g, k, seed=0):
    """Dispatch to the selected fixed-k detector."""
    if DETECTOR == "graclus":
        return graclus_partition(g, k, seed=seed)
    return metis_partition(g, k, seed=seed)


def fixed_k_seeds():
    return GRACLUS_SEEDS if DETECTOR == "graclus" else METIS_SEEDS


# ---------------------------------------------------------------------------
# L-Spar (exp_L_lspar/run.py verbatim)
# ---------------------------------------------------------------------------

def edge_jaccard(g, chunk=100_000):
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


# ---------------------------------------------------------------------------
# DSpar, calibrated sampler (exp_K / exp_N verbatim)
# ---------------------------------------------------------------------------

def dspar_scores(E, deg):
    return 1.0 / deg[E[:, 0]] + 1.0 / deg[E[:, 1]]


def _probs_calibrated(scores, alpha):
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


# ---------------------------------------------------------------------------
# K-Neighbor -- OUR implementation (Chen et al. sec 2.3.2; Sadhanala et al. 2016)
#   "selects k edges for each vertex, and if a vertex has less than k vertices, all of
#    its edges are included.  The edges are selected with probability proportional to
#    their weights (uniform for unweighted graphs)."
# The graphs here are unweighted, so selection is uniform among incident edges.
# k is integer by definition; Chen et al. state they "attempt to align" the coarse
# K-Neighbor prune-rate grid to the requested rate.  We do this with a fractional k:
# node i keeps floor(kf) + Bernoulli(kf - floor(kf)) incident edges (Bernoulli draw
# fixed per node across the bisection so retention is monotone in kf).
# ---------------------------------------------------------------------------

class KNeighbor:
    def __init__(self, E, n, seed):
        self.n, self.m = n, E.shape[0]
        rng = np.random.RandomState(seed)
        node = np.concatenate([E[:, 0], E[:, 1]])
        eid = np.concatenate([np.arange(self.m), np.arange(self.m)])
        key = rng.random_sample(node.size)
        order = np.lexsort((key, node))
        self.eid_s = eid[order]
        d = np.bincount(node, minlength=n).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(d)[:-1]])
        self.rank = np.arange(node.size, dtype=np.int64) - np.repeat(starts, d)
        self.d = d
        self.u_node = rng.random_sample(n)

    def select(self, kf):
        kfloor = int(np.floor(kf))
        frac = kf - kfloor
        kv = kfloor + (self.u_node < frac).astype(np.int64)
        sel = self.rank < np.repeat(kv, self.d)
        return np.unique(self.eid_s[sel])

    def retention(self, kf):
        return self.select(kf).size / self.m

    def bisect_k(self, target, tol=0.005, iters=40):
        r1 = self.retention(1.0)
        lo, hi = 0.0, 1.0
        while self.retention(hi) < target and hi < 4096:
            lo, hi = hi, hi * 2.0
        best = (abs(self.retention(hi) - target), hi, self.retention(hi))
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
        return best[1], best[2], r1


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------

def subgraph_from_eids(g, E, eids):
    gs = ig.Graph(n=g.vcount())
    gs.add_edges(E[np.asarray(eids, dtype=np.int64)])
    return gs


def graph_from_edges(n, edges):
    gs = ig.Graph(n=n)
    gs.add_edges(edges)
    return gs


def frag_stats(memb, n):
    sizes = np.bincount(np.asarray(memb, dtype=np.int64))
    live = sizes[sizes > 0]
    small = live[live < FRAG_MAX]
    return dict(n_clusters=int(live.size),
                n_frag_clusters=int(small.size),
                frag_nodes=int(small.sum()),
                frag_node_frac=float(small.sum()) / n,
                n_singletons=int((live == 1).sum()),
                max_cluster_frac=float(live.max()) / n)


# ---------------------------------------------------------------------------
# runtime-matched restart pool (computed once per network, replayed per arm)
# ---------------------------------------------------------------------------

class MatchPool:
    """Plain Leiden restarts on the ORIGINAL graph, in fixed seed order.

    best_for(budget) returns (best Q within cumulative wall clock <= budget, n_restarts).
    Identical in value to exp_L's per-row loop; computed once and replayed to save time.
    """

    def __init__(self, g):
        self.g = g
        self.Q = []
        self.T = []
        self._i = 0

    def _extend(self):
        if self._i >= len(MATCH_SEEDS):
            return False
        _, Q, _, dt = leiden(self.g, MATCH_SEEDS[self._i])
        self.Q.append(Q)
        self.T.append(dt)
        self._i += 1
        return True

    def best_for(self, budget):
        spent, best, n = 0.0, -1.0, 0
        i = 0
        while True:
            if i >= len(self.Q):
                if not self._extend():
                    break
            spent += self.T[i]
            best = max(best, self.Q[i])
            n += 1
            i += 1
            if spent >= budget:
                break
        return best, n


# ---------------------------------------------------------------------------
# resolution-matched control on the ORIGINAL graph (exp_L leiden_matched + budget)
# ---------------------------------------------------------------------------

def leiden_matched(g, seed, target, tol=0.05, max_iter=RESMATCH_MAX_CALLS,
                   time_budget=RESMATCH_TIME_BUDGET):
    t_start = time.perf_counter()
    calls = [0]

    def n_of(gamma):
        calls[0] += 1
        memb, _, nc, _ = leiden(g, seed, resolution=gamma)
        return memb, nc

    def out_of_budget():
        return (calls[0] >= max_iter or
                time.perf_counter() - t_start > time_budget)

    lo, hi = 1.0, 1.0
    memb, nc = n_of(hi)
    best = (abs(nc - target), memb, hi, nc)
    if nc > target:
        while nc > target and lo > 1e-4 and not out_of_budget():
            hi = lo
            lo /= 2.0
            memb, nc = n_of(lo)
            if abs(nc - target) < best[0]:
                best = (abs(nc - target), memb, lo, nc)
    else:
        while nc < target and hi < 1e5 and not out_of_budget():
            lo = hi
            hi *= 2.0
            memb, nc = n_of(hi)
            if abs(nc - target) < best[0]:
                best = (abs(nc - target), memb, hi, nc)
    while not out_of_budget():
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
    return best[1], best[2], best[3], calls[0]


# ---------------------------------------------------------------------------
# NetworKit bridge
# ---------------------------------------------------------------------------

def build_nk(n, E):
    """Undirected NetworKit graph with exactly one entry per igraph edge.

    NOTE: nk.GraphFromCoo with both (u,v) and (v,u) yields 2m edges even when
    directed=False, which silently doubles every retention ratio.  Verified on
    ca-HepTh (24,806 igraph edges -> 49,612 nk edges); we therefore add each edge
    once, and assert the count.
    """
    import networkit as nk
    G = nk.Graph(n, weighted=False, directed=False)
    for u, v in E:
        G.addEdge(int(u), int(v))
    G.removeSelfLoops()
    G.indexEdges()
    assert G.numberOfEdges() == E.shape[0], (G.numberOfEdges(), E.shape[0])
    return G


def nk_edges(G):
    return [(int(u), int(v)) for u, v in G.iterEdges()]


# ---------------------------------------------------------------------------
# per-network driver
# ---------------------------------------------------------------------------

def run_network(name, do_metis=False, k_mode="nc_base"):
    import networkit as nk

    res_csv = HERE / "results.csv"
    rec_csv = HERE / "recovery.csv"

    t0 = time.perf_counter()
    g, id_map = load_lcc_graph(name, return_map=True)
    n, m = g.vcount(), g.ecount()
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    log(f"\n{'='*78}\n{name}: n={n:,} m={m:,} d_avg={2*m/n:.2f} "
        f"(load {time.perf_counter()-t0:.1f}s)\n{'='*78}")

    # ---- ground truth -----------------------------------------------------
    mode = ("hard" if name in LABELLED_HARD else
            "overlap" if name in LABELLED_OVERLAP else None)
    y = gt_sizes = n2c = None
    gt_k = None
    if mode == "hard":
        y = load_hard_labels(name, id_map, n)
        gt_k = len(set(y[y >= 0].tolist()))
        log(f"  labels: {int((y >= 0).sum())}/{n} nodes, {gt_k} classes")
    elif mode == "overlap":
        gt_sizes, n2c = load_overlap_gt(name, id_map)
        gt_k = int(gt_sizes.size)
        log(f"  ground truth: {len(gt_sizes)} communities (>= {MIN_GT_SIZE} nodes)")

    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 normalized_mutual_info_score)
    chance_rng = np.random.RandomState(CHANCE_SEED)

    def rec_eval(memb):
        """Recovery metrics + a size-matched random-partition chance floor."""
        if mode is None:
            return {}
        memb = np.asarray(memb, dtype=np.int64)
        perm = memb.copy()
        chance_rng.shuffle(perm)          # same cluster-size profile, random assignment
        if mode == "hard":
            ok = y >= 0
            return dict(
                AMI=float(adjusted_mutual_info_score(y[ok], memb[ok])),
                ARI=float(adjusted_rand_score(y[ok], memb[ok])),
                NMI=float(normalized_mutual_info_score(y[ok], memb[ok])),
                AMI_chance=float(adjusted_mutual_info_score(y[ok], perm[ok])),
                avgF1_ge3="", avgF1_chance="")
        f = average_f1(memb, n2c, gt_sizes)
        fc = average_f1(perm, n2c, gt_sizes)
        return dict(AMI="", ARI="", NMI="", AMI_chance="",
                    avgF1_ge3=f["avgF1_ge3"], avgF1_chance=fc["avgF1_ge3"])

    def rec_row(arm, target, realized, seed, memb, extra):
        if mode is None:
            return
        fs = frag_stats(memb, n)
        r = dict(network=name, arm=arm, target_ret=target, realized_ret=realized,
                 seed=seed, n_clusters=fs["n_clusters"],
                 frag_nodes=fs["frag_nodes"], frag_node_frac=fs["frag_node_frac"],
                 **rec_eval(memb), **extra)
        append_row(rec_csv, r)
        return r

    # ---- baseline ---------------------------------------------------------
    base_Q, base_T, base_nc, base_memb = [], [], [], None
    for s in BASE_SEEDS:
        memb, Q, nc, dt = leiden(g, s)
        base_Q.append(Q); base_T.append(dt); base_nc.append(nc)
        if base_memb is None:
            base_memb = memb
        rec_row("baseline", 1.0, 1.0, s, memb, dict(resolution=1.0, detector="leiden"))
    Qb_mean, Qb_std, Qb_best = float(np.mean(base_Q)), float(np.std(base_Q)), float(max(base_Q))
    T_leiden_orig = float(np.mean(base_T))
    nc_base = float(np.mean(base_nc))
    fs_base = frag_stats(base_memb, n)
    log(f"  baseline Q={Qb_mean:.6f}+-{Qb_std:.6f} best5={Qb_best:.6f} nc={nc_base:.0f} "
        f"frag_nodes={fs_base['frag_nodes']} T={T_leiden_orig:.2f}s")

    pool = MatchPool(g)

    # ---- shared sparsifier machinery, each component timed separately -----
    tt = time.perf_counter(); _E, J, deg = edge_jaccard(g); T_jaccard = time.perf_counter() - tt
    tt = time.perf_counter(); ls = LSpar(g, J=J, E=E, deg=deg); T_lspar_rank = time.perf_counter() - tt
    tt = time.perf_counter()
    mst_eids = np.asarray(g.spanning_tree(weights=(1.0 - J).tolist(), return_tree=False),
                          dtype=np.int64)
    T_mst = time.perf_counter() - tt
    mst_floor = mst_eids.size / m
    non_mst = np.setdiff1d(np.arange(m, dtype=np.int64), mst_eids, assume_unique=False)
    tt = time.perf_counter(); ds_scores = dspar_scores(E, deg.astype(np.float64))
    T_dspar_score = time.perf_counter() - tt

    tt = time.perf_counter(); nkG = build_nk(n, E); T_nk_build = time.perf_counter() - tt
    ld_spars = nk.sparsification.LocalDegreeSparsifier()
    lsim_spars = nk.sparsification.LocalSimilaritySparsifier()
    tt = time.perf_counter(); ld_attr = ld_spars.scores(nkG); T_ld_score = time.perf_counter() - tt
    tt = time.perf_counter(); lsim_attr = lsim_spars.scores(nkG); T_lsim_score = time.perf_counter() - tt

    lspar_floor = ls.retention(0.0)
    kn_probe = KNeighbor(E, n, REP_SEEDS[0])
    kn_ret_k1 = kn_probe.retention(1.0)
    ld_floor = ld_spars.getSparsifiedGraph(nkG, 1.0, ld_attr).numberOfEdges() / m
    lsim_floor = lsim_spars.getSparsifiedGraph(nkG, 1.0, lsim_attr).numberOfEdges() / m
    log(f"  jaccard {T_jaccard:.2f}s  lspar_rank {T_lspar_rank:.2f}s  mst {T_mst:.2f}s  "
        f"nk_build {T_nk_build:.2f}s  ld_score {T_ld_score:.2f}s  lsim_score {T_lsim_score:.2f}s")
    log(f"  RETENTION FLOORS: mst_backbone=(n-1)/m={mst_floor:.4f}  "
        f"lspar(e->0)={lspar_floor:.4f}  ld(param=1)={ld_floor:.4f}  "
        f"lsim(param=1)={lsim_floor:.4f}  kn(integer k=1)={kn_ret_k1:.4f}")

    # ---- operating points -------------------------------------------------
    # {0.5, 0.2} as registered, plus the largest hard floor when it exceeds 0.2, so
    # that there is one aggressive point at which ALL SEVEN arms are matched.
    max_floor = max(mst_floor, lspar_floor, ld_floor, lsim_floor)
    ops = list(TARGETS)
    if max_floor > min(TARGETS) + 0.005:
        # round UP: rounding down puts the op just below the backbone's own floor and
        # the backbone arms get skipped at the very point that exists for them.
        ops.append(float(np.ceil(max_floor * 1e4 + 1.0) / 1e4))
    ops = sorted(set(ops), reverse=True)
    log(f"  operating points (target realized retention): {ops}")

    # ---- resolution-matched cache ----------------------------------------
    resmatch_cache = {}
    resmatch_spent = [0.0]
    per_call_budget = min(RESMATCH_TIME_BUDGET, max(120.0, 25.0 * T_leiden_orig))

    def resmatch(nc_target):
        key = int(round(np.log(max(nc_target, 1.0)) / np.log(1.05)))   # 5% log buckets
        if key in resmatch_cache:
            return resmatch_cache[key]
        if resmatch_spent[0] > RESMATCH_NETWORK_BUDGET:
            log(f"    [resmatch] SKIPPED for target_nc={nc_target:.0f} "
                f"(network budget {RESMATCH_NETWORK_BUDGET:.0f}s exhausted)")
            return None
        t = time.perf_counter()
        memb, gamma, nc_r, calls = leiden_matched(
            g, BASE_SEEDS[0], int(round(nc_target)), time_budget=per_call_budget)
        resmatch_spent[0] += time.perf_counter() - t
        Q_r = float(g.modularity(memb.tolist()))
        out = dict(memb=memb, gamma=gamma, nc=nc_r, Q=Q_r, calls=calls,
                   secs=time.perf_counter() - t)
        resmatch_cache[key] = out
        log(f"    [resmatch] target_nc={nc_target:.0f} -> nc={nc_r} gamma={gamma:.5g} "
            f"Q_orig={Q_r:.6f} ({calls} calls, {out['secs']:.1f}s)")
        rec_row("resmatch", 1.0, 1.0, BASE_SEEDS[0], memb,
                dict(resolution=round(gamma, 6), detector="leiden"))
        return out

    # ---- arm execution ----------------------------------------------------
    def sparsify(arm, target, seed):
        """-> (edge ids kept OR igraph, realized retention, T_sparsify, params, status)"""
        if arm == "kn":
            t = time.perf_counter()
            kn = KNeighbor(E, n, seed)
            kf, r, r1 = kn.bisect_k(target)
            eids = kn.select(kf)
            return eids, eids.size / m, time.perf_counter() - t, dict(param=kf, kn_ret_k1=r1), "ok"
        if arm == "ld":
            t = time.perf_counter()
            Gs = ld_spars.getSparsifiedGraphOfSize(nkG, target, ld_attr)
            ed = nk_edges(Gs)
            T = T_nk_build + T_ld_score + (time.perf_counter() - t)
            return ed, len(ed) / m, T, dict(param=""), "ok"
        if arm == "lsim":
            t = time.perf_counter()
            Gs = lsim_spars.getSparsifiedGraphOfSize(nkG, target, lsim_attr)
            ed = nk_edges(Gs)
            T = T_nk_build + T_lsim_score + (time.perf_counter() - t)
            return ed, len(ed) / m, T, dict(param=""), "ok"
        if arm in NEEDS_BACKBONE:
            if target < mst_floor - 1e-9:
                return None, mst_floor, 0.0, dict(param=""), "below_backbone_floor"
            t = time.perf_counter()
            want = int(round(target * m))
            extra = max(0, want - mst_eids.size)
            if arm == "mst_jaccard":
                order = non_mst[np.argsort(-J[non_mst], kind="stable")]
            else:
                rs = np.random.RandomState(seed)
                order = non_mst[rs.permutation(non_mst.size)]
            eids = np.concatenate([mst_eids, order[:extra]])
            T = T_jaccard + T_mst + (time.perf_counter() - t)
            return eids, eids.size / m, T, dict(param=extra), "ok"
        if arm == "lspar":
            t = time.perf_counter()
            e_used, realized, status = ls.bisect_e(target)
            eids = ls.select(e_used)
            T = T_jaccard + T_lspar_rank + (time.perf_counter() - t)
            return eids, eids.size / m, T, dict(param=e_used), status
        if arm == "dspar":
            t = time.perf_counter()
            probs = _probs_calibrated(ds_scores, target)
            rs = np.random.RandomState(seed)
            eids = np.where(rs.random_sample(m) < probs)[0]
            T = T_dspar_score + (time.perf_counter() - t)
            return eids, eids.size / m, T, dict(param=target), "ok"
        raise ValueError(arm)

    def make_graph(arm, obj):
        if arm in ("ld", "lsim"):
            return graph_from_edges(n, obj)
        return subgraph_from_eids(g, E, obj)

    for target in ([] if METIS_ONLY else ops):
        log(f"\n  ---- operating point target_ret={target} ----")
        for arm in ARMS:
            stoch = arm in STOCHASTIC
            Q_sp, Q_or, ncs, Ts, Tsp, rets, params, status = [], [], [], [], [], [], [], "ok"
            fragN, fragC, singles, maxc = [], [], [], []
            memb_first = None
            gs = None
            for i, s in enumerate(REP_SEEDS):
                if not stoch and i > 0:
                    obj, realized, T_s, prm, status = cached
                else:
                    obj, realized, T_s, prm, status = sparsify(arm, target, s)
                    cached = (obj, realized, T_s, prm, status)
                if status == "below_backbone_floor":
                    break
                if stoch or i == 0:
                    gs = make_graph(arm, obj)
                memb, Qs, nc, dt = leiden(gs, s)
                Qo = float(g.modularity(memb.tolist()))
                fs = frag_stats(memb, n)
                Q_sp.append(Qs); Q_or.append(Qo); ncs.append(nc); Ts.append(dt)
                Tsp.append(T_s); rets.append(realized); params.append(prm.get("param", ""))
                fragN.append(fs["frag_nodes"]); fragC.append(fs["n_frag_clusters"])
                singles.append(fs["n_singletons"]); maxc.append(fs["max_cluster_frac"])
                if memb_first is None:
                    memb_first = memb
                rec_row(arm, target, round(realized, 5), s, memb,
                        dict(resolution=1.0, detector="leiden"))

            if status == "below_backbone_floor":
                row = dict(network=name, n=n, m=m, avg_deg=2.0 * m / n, arm=arm,
                           detector="leiden", target_ret=target, realized_ret=mst_floor,
                           status=status)
                append_row(HERE / "skipped.csv", row)
                log(f"    {arm:12s} SKIPPED: target {target} < backbone floor {mst_floor:.4f}")
                continue

            mean_ret = float(np.mean(rets))
            if mean_ret > target + 0.02:
                status = "at_floor"          # sparsifier cannot prune this far
            elif mean_ret < target - 0.02:
                status = "coarse_grid"       # overshoot from a coarse parameter grid
            Qor_m = float(np.mean(Q_or))
            T_leiden_sparse = float(np.mean(Ts))
            T_sparsify = float(np.mean(Tsp))
            budget = T_sparsify + T_leiden_sparse
            Q_matched, n_restarts = pool.best_for(budget)
            nc_sparse = float(np.mean(ncs))

            rm = None
            if abs(nc_sparse - nc_base) / max(nc_base, 1.0) > RESMATCH_TRIGGER:
                rm = resmatch(nc_sparse)

            rec = rec_eval(memb_first) if mode else {}
            row = dict(
                network=name, n=n, m=m, avg_deg=2.0 * m / n,
                arm=arm, detector="leiden", stochastic=int(stoch),
                target_ret=target, realized_ret=float(np.mean(rets)),
                realized_ret_std=float(np.std(rets)), param=params[0], status=status,
                matched=int(abs(float(np.mean(rets)) - target) <= 0.02),
                m_sparse=int(round(np.mean(rets) * m)),
                mst_floor=mst_floor, lspar_floor=lspar_floor, ld_floor=ld_floor,
                lsim_floor=lsim_floor, kn_ret_k1=kn_ret_k1,
                # honest transfer
                Q_base_mean=Qb_mean, Q_base_std=Qb_std, Q_base_best=Qb_best,
                nc_base=nc_base,
                Q_sparse_on_sparse=float(np.mean(Q_sp)),
                Q_orig_mean=Qor_m, Q_orig_std=float(np.std(Q_or)),
                Q_orig_max=float(np.max(Q_or)),
                dQ_naive=float(np.mean(Q_sp)) - Qb_mean,
                dQ_vs_base_mean=Qor_m - Qb_mean,
                dQ_vs_base_best=Qor_m - Qb_best,
                dQ_vs_matched=Qor_m - Q_matched,
                Q_matched_best=Q_matched, n_restarts=n_restarts, budget=budget,
                # granularity + fragmentation
                nc_sparse=nc_sparse,
                frag_nodes=float(np.mean(fragN)), frag_node_frac=float(np.mean(fragN)) / n,
                n_frag_clusters=float(np.mean(fragC)),
                n_singletons=float(np.mean(singles)),
                max_cluster_frac=float(np.mean(maxc)),
                frag_nodes_base=fs_base["frag_nodes"],
                frag_node_frac_base=fs_base["frag_node_frac"],
                # resolution-matched control
                resmatch_done=int(rm is not None),
                nc_resmatch=(rm["nc"] if rm else ""),
                gamma_resmatch=(round(rm["gamma"], 6) if rm else ""),
                Q_resmatch=(rm["Q"] if rm else ""),
                dQ_vs_resmatch=(Qor_m - rm["Q"] if rm else ""),
                # recovery
                AMI=rec.get("AMI", ""), ARI=rec.get("ARI", ""), NMI=rec.get("NMI", ""),
                AMI_chance=rec.get("AMI_chance", ""),
                avgF1_ge3=rec.get("avgF1_ge3", ""),
                avgF1_chance=rec.get("avgF1_chance", ""),
                # cost
                T_leiden_orig=T_leiden_orig, T_sparsify=T_sparsify,
                T_leiden_sparse=T_leiden_sparse, T_pipeline=budget,
                speedup_pipeline=T_leiden_orig / budget,
                speedup_detect_only=T_leiden_orig / T_leiden_sparse,
            )
            append_row(res_csv, row)
            log(f"    {arm:12s} ret={row['realized_ret']:.4f}[{status}] nc={nc_sparse:8.0f} "
                f"frag={row['frag_node_frac']*100:6.2f}% "
                f"dQ_mean={row['dQ_vs_base_mean']:+.5f} dQ_best5={row['dQ_vs_base_best']:+.5f} "
                f"dQ_match={row['dQ_vs_matched']:+.5f}(r={n_restarts}) "
                f"{'dQ_res=%+.5f' % row['dQ_vs_resmatch'] if rm else ''} "
                f"T_sp={T_sparsify:.1f}s spd={row['speedup_pipeline']:.2f}x")
            del gs

    # ---- optional fixed-k Metis arm (exp_AA protocol: multi-seed, worst case) ----
    if do_metis:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if k_mode == "gt":
            if gt_k is None:
                log(f"    [skip] no ground-truth k for {name}")
                return
            k_fix = int(gt_k)
        else:
            k_fix = int(round(nc_base))
        log(f"\n  ==== {DETECTOR} (k_mode={k_mode}, fixed k={k_fix}) on {name} ====")
        Qm_b, Tm_b, AMIm_b, F1m_b, mb = [], [], [], [], None
        for s in fixed_k_seeds():
            p, dt = detect_fixed_k(g, k_fix, seed=s)
            Qm_b.append(float(g.modularity(p.tolist()))); Tm_b.append(dt)
            r = rec_eval(p) if mode else {}
            if r.get("AMI", "") != "":
                AMIm_b.append(r["AMI"])
            if r.get("avgF1_ge3", "") != "":
                F1m_b.append(r["avgF1_ge3"])
            rec_row("baseline", 1.0, 1.0, s, p, dict(resolution="", detector=DETECTOR))
            if mb is None:
                mb = p
        Q_mb_mean, Q_mb_best = float(np.mean(Qm_b)), float(np.max(Qm_b))
        T_mb = float(np.mean(Tm_b))
        fs_mb = frag_stats(mb, n)
        log(f"    {DETECTOR} base k={k_fix} Q={Q_mb_mean:.6f}+-{np.std(Qm_b):.6f} "
            f"best={Q_mb_best:.6f} "
            f"AMI={np.mean(AMIm_b) if AMIm_b else float('nan'):.4f} T={T_mb:.3f}s")
        for target in ops:
            for arm in ARMS:
                Qm_a, Tm_a, AMIm_a, F1m_a, rets_a, ms_first = [], [], [], [], [], None
                fsl = []
                gs = None
                for i, s in enumerate(fixed_k_seeds()):
                    if (arm in STOCHASTIC) or i == 0:
                        obj, realized, T_s, prm, status = sparsify(arm, target, REP_SEEDS[i % 3])
                        if status == "below_backbone_floor":
                            break
                        gs = make_graph(arm, obj)
                    ms, T_ms = detect_fixed_k(gs, k_fix, seed=s)
                    Qm_a.append(float(g.modularity(ms.tolist()))); Tm_a.append(T_ms)
                    rets_a.append(realized)
                    fsl.append(frag_stats(ms, n))
                    r = rec_eval(ms) if mode else {}
                    if r.get("AMI", "") != "":
                        AMIm_a.append(r["AMI"])
                    if r.get("avgF1_ge3", "") != "":
                        F1m_a.append(r["avgF1_ge3"])
                    rec_row(arm, target, round(realized, 5), s, ms,
                            dict(resolution="", detector=DETECTOR))
                    if ms_first is None:
                        ms_first = ms
                if status == "below_backbone_floor" or not Qm_a:
                    continue
                Qa_mean, Qa_min = float(np.mean(Qm_a)), float(np.min(Qm_a))
                realized = float(np.mean(rets_a))
                rec = rec_eval(ms_first) if mode else {}
                row = dict(
                    network=name, n=n, m=m, avg_deg=2.0 * m / n, arm=arm,
                    k_convention=k_mode,
                    detector=DETECTOR, stochastic=int(arm in STOCHASTIC), target_ret=target,
                    realized_ret=realized, realized_ret_std=float(np.std(rets_a)),
                    param=prm.get("param", ""), status=status,
                    matched=int(abs(realized - target) <= 0.02),
                    m_sparse=int(round(realized * m)),
                    mst_floor=mst_floor, lspar_floor=lspar_floor, ld_floor=ld_floor,
                    lsim_floor=lsim_floor, kn_ret_k1=kn_ret_k1,
                    Q_base_mean=Q_mb_mean, Q_base_std=float(np.std(Qm_b)),
                    Q_base_best=Q_mb_best, nc_base=k_fix,
                    Q_sparse_on_sparse=float(gs.modularity(ms_first.tolist())),
                    Q_orig_mean=Qa_mean, Q_orig_std=float(np.std(Qm_a)),
                    Q_orig_max=float(np.max(Qm_a)),
                    dQ_naive="", dQ_vs_base_mean=Qa_mean - Q_mb_mean,
                    dQ_vs_base_best=Qa_mean - Q_mb_best,
                    dQ_vs_matched=Qa_min - Q_mb_best,     # worst-case, exp_AA metis_noise
                    Q_matched_best=Q_mb_best, n_restarts=len(Qm_b),
                    budget=T_s + float(np.mean(Tm_a)),
                    nc_sparse=float(np.mean([f["n_clusters"] for f in fsl])),
                    frag_nodes=float(np.mean([f["frag_nodes"] for f in fsl])),
                    frag_node_frac=float(np.mean([f["frag_node_frac"] for f in fsl])),
                    n_frag_clusters=float(np.mean([f["n_frag_clusters"] for f in fsl])),
                    n_singletons=float(np.mean([f["n_singletons"] for f in fsl])),
                    max_cluster_frac=float(np.mean([f["max_cluster_frac"] for f in fsl])),
                    frag_nodes_base=fs_mb["frag_nodes"],
                    frag_node_frac_base=fs_mb["frag_node_frac"],
                    resmatch_done=0, nc_resmatch="", gamma_resmatch="", Q_resmatch="",
                    dQ_vs_resmatch="",
                    AMI=(float(np.mean(AMIm_a)) if AMIm_a else ""),
                    ARI=rec.get("ARI", ""), NMI=rec.get("NMI", ""),
                    AMI_chance=rec.get("AMI_chance", ""),
                    avgF1_ge3=(float(np.mean(F1m_a)) if F1m_a else ""),
                    avgF1_chance=rec.get("avgF1_chance", ""),
                    T_leiden_orig=T_mb, T_sparsify=T_s,
                    T_leiden_sparse=float(np.mean(Tm_a)),
                    T_pipeline=T_s + float(np.mean(Tm_a)),
                    speedup_pipeline=T_mb / max(T_s + float(np.mean(Tm_a)), 1e-9),
                    speedup_detect_only=T_mb / max(float(np.mean(Tm_a)), 1e-9),
                )
                append_row(res_csv, row)
                dami = ((float(np.mean(AMIm_a)) - float(np.mean(AMIm_b)))
                        if (AMIm_a and AMIm_b) else float("nan"))
                df1 = ((float(np.mean(F1m_a)) - float(np.mean(F1m_b)))
                       if (F1m_a and F1m_b) else float("nan"))
                log(f"    {DETECTOR} {arm:12s} t={target} ret={realized:.4f} "
                    f"dQ_mean={Qa_mean-Q_mb_mean:+.6f} dQ_worst={Qa_min-Q_mb_best:+.6f} "
                    f"dAMI={dami:+.4f} davgF1={df1:+.4f}")
                del gs

    log(f"\n  {name} DONE in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "net":
        raise SystemExit("usage: run.py net <network> [--metis|--graclus] [--metis-only|--graclus-only] [--k-mode nc_base|gt]")
    METIS_ONLY = ("--metis-only" in sys.argv) or ("--graclus-only" in sys.argv)
    if "--graclus" in sys.argv or "--graclus-only" in sys.argv:
        DETECTOR = "graclus"
        if not os.path.exists(GRACLUS_BIN):
            raise SystemExit("graclus binary not found at %s (set GRACLUS_BIN)" % GRACLUS_BIN)
    km = sys.argv[sys.argv.index("--k-mode") + 1] if "--k-mode" in sys.argv else "nc_base"
    run_network(sys.argv[2],
                do_metis=("--metis" in sys.argv or "--graclus" in sys.argv or METIS_ONLY),
                k_mode=km)
