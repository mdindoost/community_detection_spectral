#!/usr/bin/env python
"""
Experiment AA -- The Satuluri regime.

Pre-registered in DESIGN.md (2026-07-26). Three arms:

  A  average-degree sweep on LFR (n=1e4, tau1=2.1, tau2=1.5, mu in {.3,.5,.8},
     d_avg in {10,25,50,100,200}), Leiden detector, honest-transfer modularity
     on the ORIGINAL graph + chance-corrected recovery vs planted labels.
  B  the same graphs under FIXED-k partitioners (real Metis via pymetis, plus
     two explicitly-labelled proxies: igraph leading-eigenvector with clusters=k
     and sklearn SpectralClustering).
  C  noise injection: x% of m uniformly random NEW edges, then sparsify; the
     causal test of the denoising mechanism.

Sparsifiers: L-Spar (exact Jaccard, e bisected -- verbatim from exp_L_lspar),
DSpar (calibrated sampler -- verbatim from exp_N_enron_anatomy), uniform random.

All scoring is honest transfer: partitions found on the sparsified graph are
scored on the graph they claim to describe.

Usage:  run.py gen              # pre-generate + cache all LFR graphs
        run.py armA [cells]
        run.py armB [cells]
        run.py armC
"""

import csv
import itertools
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import igraph as ig
import leidenalg as la
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("AA_CACHE", "/tmp/aa_lfr_cache"))
CACHE.mkdir(parents=True, exist_ok=True)

N_ITER = 2
N_NODES = 10_000
TAU1, TAU2 = 2.1, 1.5
MIN_COMM, MAX_COMM = 20, 500
# networkx multiplies max_iters by 10*n internally before the community-assignment
# loop, so 300 == 3e6 assignment attempts for n=1e4 (300x the node count).
LFR_MAX_ITERS = 300

MUS = [0.3, 0.5, 0.8]
DEGS = [10, 25, 50, 100, 200]
LFR_SEEDS = [1, 2, 3]

BASE_SEEDS = [100, 101, 102, 103, 104]
SPARSE_SEEDS = [300, 301, 302]
LADDER_SEEDS = list(range(100, 140))
TARGETS = [0.5, 0.2, 0.15, 0.05]
SPARSIFIERS = ["lspar", "dspar", "random"]

ARMB_MUS = [0.5, 0.8]
ARMC_DEGS = [25, 50]
ARMC_NOISE = [0, 10, 25, 50, 100]
ARMC_TARGETS = [0.5, 0.2]


def log(*a):
    print(*a, flush=True)


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def done_keys(path, cols):
    """Set of already-completed row keys, for resume-safety."""
    if not path.exists():
        return set()
    out = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out.add(tuple(r[c] for c in cols))
            except KeyError:
                pass
    return out


# ---------------------------------------------------------------------------
# LFR generation (cached)
# ---------------------------------------------------------------------------

def _zipf_mean(gamma, lo, hi):
    x = np.arange(lo, hi + 1, dtype=np.float64)
    w = x ** (-gamma)
    return float((x * w).sum() / w.sum())


def solve_min_degree(gamma, target_avg, max_degree):
    """Smallest integer min_degree whose truncated-zipf mean ~= target_avg.

    networkx's own _generate_min_degree shares `max_iters` with the community
    assignment loop and never converges at its 1e-7 tolerance once max_iters is
    raised enough for n=10^4 nodes, so we solve it here and pass min_degree
    directly (this makes the generator's max_iters govern ONLY the community
    assignment, which is what it is needed for).
    """
    lo, hi = 1, max_degree - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _zipf_mean(gamma, mid, max_degree) < target_avg:
            lo = mid + 1
        else:
            hi = mid
    cands = [c for c in (lo - 1, lo, lo + 1) if 1 <= c < max_degree]
    return min(cands, key=lambda c: abs(_zipf_mean(gamma, c, max_degree) - target_avg))


GEN_MODES = ["registered", "maxdeg_cap100", "maxdeg_cap85", "maxdeg_cap70",
             "scaled_maxk5", "scaled_maxk3"]


def lfr_params(mu, d, mode):
    """Parameter box for one generation mode.

    networkx's LFR can only place every node when
        (1-mu)*min_degree < min_community   and   (1-mu)*max_degree < max_community,
    because community sizes sum to exactly n, so ANY community too small for the
    smallest node makes the assignment infeasible. With the registered box
    (min_community=20, max_community=500) that caps min_degree at 20/(1-mu) and
    therefore caps the achievable average degree well below the d_avg=100/200
    cells the design asks for. The ladder below stays on the registered box for
    as long as it is feasible and widens the community range only when forced.
    """
    if mode == "registered":
        return dict(max_degree=N_NODES - 1, min_community=MIN_COMM,
                    max_community=MAX_COMM)
    if mode.startswith("maxdeg_cap"):
        frac = int(mode[len("maxdeg_cap"):]) / 100.0
        return dict(max_degree=max(2, int((frac * MAX_COMM - 1) // (1.0 - mu))),
                    min_community=MIN_COMM, max_community=MAX_COMM)
    if mode.startswith("scaled_maxk"):
        K = int(mode[len("scaled_maxk"):])
        maxdeg = int(min(N_NODES - 1, max(2 * d, round(K * d))))
        mind = solve_min_degree(TAU1, d, maxdeg)
        # 0.85 margin: the LARGEST community actually drawn is typically well below
        # max_community, and a single node it cannot host makes placement infeasible.
        return dict(max_degree=maxdeg,
                    min_community=max(MIN_COMM, int(np.ceil((1 - mu) * mind)) + 2),
                    max_community=max(MAX_COMM,
                                      int(np.ceil((1 - mu) * maxdeg / 0.85)) + 2))
    raise ValueError(mode)


def lfr_feasible(mu, d, params):
    """Analytic precheck. Community sizes sum to exactly n, so every drawn
    community must be able to host the SMALLEST node; and the largest node must
    fit the largest drawn community (~0.85*max_community in practice). Modes
    that fail this burn 3e6 futile assignment iterations, so skip them."""
    mind = solve_min_degree(TAU1, d, params["max_degree"])
    return (np.ceil((1 - mu) * mind) < params["min_community"]
            and np.ceil((1 - mu) * params["max_degree"])
            < 0.85 * params["max_community"])


def _lfr_once(mu, d, seed, params):
    md = params["max_degree"]
    min_deg = solve_min_degree(TAU1, d, md)
    G = nx.LFR_benchmark_graph(
        n=N_NODES, tau1=TAU1, tau2=TAU2, mu=mu, min_degree=min_deg,
        max_degree=md, min_community=params["min_community"],
        max_community=params["max_community"],
        seed=seed, max_iters=LFR_MAX_ITERS)
    comms = {}
    for v in G.nodes():
        comms[v] = frozenset(G.nodes[v]["community"])
    uniq = {c: i for i, c in enumerate(sorted(set(comms.values()), key=lambda s: (len(s), min(s))))}
    y = np.array([uniq[comms[v]] for v in range(N_NODES)], dtype=np.int64)
    G.remove_edges_from(nx.selfloop_edges(G))
    E = np.array(list(G.edges()), dtype=np.int64)
    return E, y


def gen_lfr(mu, d, seed):
    """Return (E, y, meta). Registered params first; documented fallbacks after.

    gen_mode:
      registered   -- exactly DESIGN.md parameters
      maxdeg_cap   -- max_degree capped to (MAX_COMM-1)/(1-mu) so that the
                      registered community-size range can host every node
                      (networkx LFR requires community_size > (1-mu)*degree)
      *_recal      -- as above, plus the requested average_degree rescaled so
                      the REALIZED average degree matches the sweep target
    """
    f = CACHE / f"lfr_mu{mu}_d{d}_s{seed}.npz"
    if f.exists():
        z = np.load(f)
        return z["E"], z["y"], dict(
            gen_mode=str(z["gen_mode"]), gen_attempts=int(z["gen_attempts"]),
            gen_time=float(z["gen_time"]),
            gen_max_degree=int(z["gen_max_degree"]),
            gen_min_community=int(z["gen_min_community"]),
            gen_max_community=int(z["gen_max_community"]))
    t0 = time.perf_counter()
    attempts = 0
    best = None
    for name in GEN_MODES:
        params = lfr_params(mu, d, name)
        if not lfr_feasible(mu, d, params):
            log(f"    gen mode {name} skipped: provably infeasible "
                f"(max_degree={params['max_degree']}, "
                f"comm=[{params['min_community']},{params['max_community']}])")
            continue
        for s in ((seed,) if name == "registered"
                  else (seed, seed + 1000, seed + 2000)):
            attempts += 1
            try:
                E, y = _lfr_once(mu, d, s, params)
                best = (name, E, y, s, params)
                break
            except Exception as ex:
                log(f"    gen attempt {name} seed={s} failed: {type(ex).__name__}")
        if best is not None:
            break
    if best is None:
        raise RuntimeError(f"LFR generation failed for mu={mu} d={d} seed={seed}")

    name, E, y, used_seed, used_params = best
    # ---- recalibration pass: realized d_avg must track the sweep target
    # (parameter box frozen; only the min_degree solve target moves)
    real_d = 2.0 * E.shape[0] / N_NODES
    tries = 0
    ask = d
    while abs(real_d - d) / d > 0.10 and tries < 5:
        tries += 1
        ask = ask * (d / real_d)
        attempts += 1
        try:
            E2, y2 = _lfr_once(mu, ask, used_seed, used_params)
        except Exception:
            break
        rd2 = 2.0 * E2.shape[0] / N_NODES
        if abs(rd2 - d) < abs(real_d - d):
            E, y, real_d = E2, y2, rd2
            name = name + "_recal"
    meta = dict(gen_mode=name, gen_attempts=attempts,
                gen_time=time.perf_counter() - t0,
                gen_max_degree=used_params["max_degree"],
                gen_min_community=used_params["min_community"],
                gen_max_community=used_params["max_community"])
    np.savez_compressed(f, E=E, y=y, **{k: v for k, v in meta.items()})
    return E, y, meta


def graph_from_E(E, n=N_NODES):
    g = ig.Graph(n=n)
    g.add_edges([tuple(x) for x in E])
    g.simplify(multiple=True, loops=True)
    return g


def graph_meta(g, y):
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    intra = y[E[:, 0]] == y[E[:, 1]]
    sizes = np.bincount(y)
    sizes = sizes[sizes > 0]
    deg = np.asarray(g.degree(), dtype=float)
    return dict(n=g.vcount(), m=g.ecount(), d_avg_real=2.0 * g.ecount() / g.vcount(),
                n_comm_planted=int(sizes.size), mu_real=float((~intra).mean()),
                comm_size_min=int(sizes.min()), comm_size_max=int(sizes.max()),
                deg_min=int(deg.min()), deg_max=int(deg.max()),
                deg_cv=float(deg.std() / deg.mean()))


# ---------------------------------------------------------------------------
# Sparsifiers
# ---------------------------------------------------------------------------

def edge_jaccard(g, budget=20_000_000):
    n, m = g.vcount(), g.ecount()
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    u, v = E[:, 0].copy(), E[:, 1].copy()
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    A = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    chunk = max(2000, int(budget / max(1.0, deg.mean())))
    cn = np.empty(m, dtype=np.float64)
    for s in range(0, m, chunk):
        t = min(s + chunk, m)
        cn[s:t] = np.asarray(A[u[s:t]].multiply(A[v[s:t]]).sum(axis=1)).ravel()
    union = deg[u] + deg[v] - cn
    J = np.where(union > 0, cn / np.maximum(union, 1e-12), 0.0)
    return E, J, deg.astype(np.int64)


class LSpar:
    """exp_L_lspar/run.py, verbatim: local top-ceil(d^e) by Jaccard, union rule."""

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
        lo_r = self.retention(0.0)
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


def dspar_scores(g):
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    return e, 1.0 / np.maximum(deg[e[:, 0]], 1.0) + 1.0 / np.maximum(deg[e[:, 1]], 1.0)


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


def sparsify_edges(g, kind, target, spar_seed, lspar=None, dscores=None):
    """-> (eids, realized_retention, knob, status, T_sparsify)"""
    m = g.ecount()
    t0 = time.perf_counter()
    if kind == "lspar":
        e_used, realized, status = lspar.bisect_e(target)
        eids = lspar.select(e_used)
        return eids, eids.size / m, e_used, status, time.perf_counter() - t0
    if kind == "dspar":
        E, sc = dscores
        p = _probs_calibrated(sc, target)
        rs = np.random.RandomState(spar_seed)
        keep = np.where(rs.random_sample(m) < p)[0]
        return keep, keep.size / m, target, "ok", time.perf_counter() - t0
    if kind == "random":
        rs = np.random.RandomState(spar_seed)
        k = int(round(target * m))
        eids = np.sort(rs.choice(m, size=k, replace=False))
        return eids, eids.size / m, target, "ok", time.perf_counter() - t0
    raise ValueError(kind)


def subgraph_from_eids(g, eids):
    gs = ig.Graph(n=g.vcount())
    E = np.asarray(g.get_edgelist(), dtype=np.int64)[eids]
    gs.add_edges([tuple(x) for x in E])
    return gs


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
    return np.asarray(part.membership), time.perf_counter() - t0


def leiden_matched(g, seed, target, tol=0.05, max_iter=30):
    def n_of(gamma):
        memb, _ = leiden(g, seed, resolution=gamma)
        return memb, len(set(memb.tolist()))
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


def eigen_partition(g, k, seed=0):
    """PROXY (not Metis, not Graclus): igraph leading eigenvector, clusters=k."""
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vc = g.community_leading_eigenvector(clusters=int(k))
    return np.asarray(vc.membership, dtype=np.int64), time.perf_counter() - t0


def spectral_partition(g, k, seed=0):
    """PROXY: sklearn SpectralClustering (normalized cut) on the adjacency."""
    from sklearn.cluster import SpectralClustering
    t0 = time.perf_counter()
    n = g.vcount()
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    rows = np.concatenate([E[:, 0], E[:, 1]])
    cols = np.concatenate([E[:, 1], E[:, 0]])
    A = sp.csr_matrix((np.ones(rows.size, dtype=np.float64), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = SpectralClustering(n_clusters=int(k), affinity="precomputed",
                                eigen_solver="arpack", assign_labels="kmeans",
                                random_state=seed, n_init=3)
        memb = sc.fit_predict(A)
    return np.asarray(memb, dtype=np.int64), time.perf_counter() - t0


DETECTORS = {
    "leiden": None,
    "metis": metis_partition,
    "eigen_proxy": eigen_partition,
    "spectral_proxy": spectral_partition,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def size_cv(memb):
    s = np.bincount(np.asarray(memb))
    s = s[s > 0].astype(float)
    return float(s.std() / s.mean()) if s.size else float("nan")


def evaluate(g_orig, memb, y, rng=None):
    memb = np.asarray(memb)
    d = dict(nc=int(np.unique(memb).size),
             cv_size=size_cv(memb),
             Q_orig=float(g_orig.modularity(memb.tolist())),
             AMI=float(adjusted_mutual_info_score(y, memb)),
             ARI=float(adjusted_rand_score(y, memb)))
    if rng is not None:
        perm = rng.permutation(memb)
        d["AMI_chance"] = float(adjusted_mutual_info_score(y, perm))
        d["ARI_chance"] = float(adjusted_rand_score(y, perm))
    return d


class Ladder:
    """Leiden restarts on the ORIGINAL graph, extended lazily up to whatever
    runtime budget a cell asks for (best-Q-so-far inside the budget)."""

    def __init__(self, g, y, n0=5, nmax=len(LADDER_SEEDS)):
        self.g, self.y, self.nmax = g, y, nmax
        self.rows, self.cum = [], 0.0
        self._extend(n0)

    def _extend(self, upto):
        while len(self.rows) < min(upto, self.nmax):
            s = LADDER_SEEDS[len(self.rows)]
            memb, dt = leiden(self.g, s)
            self.cum += dt
            self.rows.append(dict(seed=s, dt=dt, cum=self.cum,
                                  **evaluate(self.g, memb, self.y)))

    def matched(self, budget):
        while self.cum < budget and len(self.rows) < self.nmax:
            self._extend(len(self.rows) + 1)
        n = 1
        for i, r in enumerate(self.rows):
            if r["cum"] <= budget:
                n = i + 1
        sub = self.rows[:n]
        bi = int(np.argmax([r["Q_orig"] for r in sub]))
        return sub[bi], n


RES_GAMMAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 60.0, 120.0]


class ResGrid:
    """Resolution-matched control: one gamma sweep of Leiden on the ORIGINAL
    graph per LFR graph, shared by every cell (cheaper and lower-variance than
    per-cell bisection; the achieved nc is reported so the match is auditable)."""

    def __init__(self, g, y, seed=BASE_SEEDS[0], budget=240.0):
        self.rows = []
        spent = 0.0
        for gam in RES_GAMMAS:          # ascending: cheap gammas first
            t0 = time.perf_counter()
            memb, _ = leiden(g, seed, resolution=gam)
            spent += time.perf_counter() - t0
            self.rows.append(dict(gamma=gam, **evaluate(g, memb, y)))
            if spent > budget:          # truncated on the largest graphs;
                break                   # nc_resmatch records the achieved match

    def nearest(self, target_nc):
        i = int(np.argmin([abs(r["nc"] - target_nc) for r in self.rows]))
        return self.rows[i]


# ---------------------------------------------------------------------------
# ARM A
# ---------------------------------------------------------------------------

ARMA_KEY = ["mu", "d_nom", "lfr_seed", "sparsifier", "target_ret"]


def run_armA(cells=None):
    out = HERE / "results_armA.csv"
    seen = done_keys(out, ARMA_KEY)
    rng = np.random.default_rng(7)
    combos = [(mu, d, s) for mu in MUS for d in DEGS for s in LFR_SEEDS]
    if cells:
        want = set(cells)
        combos = [c for c in combos if f"{c[0]}_{c[1]}" in want or str(c[1]) in want]

    for mu, d, ls in combos:
        tag = f"mu{mu}_d{d}_s{ls}"
        if all((str(mu), str(d), str(ls), sp_, str(t)) in seen
               for sp_ in SPARSIFIERS for t in TARGETS):
            log(f"[skip] {tag}")
            continue
        try:
            E, y, meta = gen_lfr(mu, d, ls)
        except Exception as ex:
            log(f"[GEN-FAIL] {tag}: {ex}")
            append_row(HERE / "gen_failures.csv",
                       dict(arm="A", mu=mu, d_nom=d, lfr_seed=ls, error=str(ex)))
            continue
        g = graph_from_E(E)
        gm = graph_meta(g, y)
        log(f"\n{'='*76}\n{tag}  {meta['gen_mode']}  n={gm['n']} m={gm['m']:,} "
            f"d_avg={gm['d_avg_real']:.1f} mu_real={gm['mu_real']:.3f} "
            f"k_planted={gm['n_comm_planted']}\n{'='*76}")

        ladder = Ladder(g, y, n0=len(BASE_SEEDS))
        base = ladder.rows[:len(BASE_SEEDS)]
        Qb = [r["Q_orig"] for r in base]
        Ab = [r["AMI"] for r in base]
        Rb = [r["ARI"] for r in base]
        Tb = float(np.mean([r["dt"] for r in base]))
        log(f"  baseline Leiden Q={np.mean(Qb):.5f}+-{np.std(Qb):.5f} "
            f"AMI={np.mean(Ab):.4f} ARI={np.mean(Rb):.4f} "
            f"nc={np.mean([r['nc'] for r in base]):.0f} T={Tb:.2f}s")

        tj = time.perf_counter()
        Ej, J, deg = edge_jaccard(g)
        T_jac = time.perf_counter() - tj
        lsp = LSpar(g, J=J, E=Ej, deg=deg)
        T_rank = time.perf_counter() - tj - T_jac
        ds = dspar_scores(g)
        min_ret = lsp.retention(0.0)
        log(f"  jaccard {T_jac:.2f}s rank {T_rank:.2f}s  lspar_floor={min_ret:.4f}")
        tg = time.perf_counter()
        resgrid = ResGrid(g, y)
        log(f"  resolution grid: nc range "
            f"{min(r['nc'] for r in resgrid.rows)}..{max(r['nc'] for r in resgrid.rows)} "
            f"({time.perf_counter()-tg:.1f}s)")

        for kind, target in itertools.product(SPARSIFIERS, TARGETS):
            key = (str(mu), str(d), str(ls), kind, str(target))
            if key in seen:
                continue
            eids, realized, knob, status, T_sel = sparsify_edges(
                g, kind, target, 500 + ls, lspar=lsp, dscores=ds)
            T_sparsify = T_sel + (T_jac + T_rank if kind == "lspar" else 0.0)
            gs = subgraph_from_eids(g, eids)

            evs, Ts = [], []
            for s in SPARSE_SEEDS:
                memb, dt = leiden(gs, s)
                Ts.append(dt)
                evs.append(evaluate(g, memb, y, rng=rng))
            T_det = float(np.mean(Ts))
            agg = {k: float(np.mean([e[k] for e in evs])) for k in evs[0]}
            agg_sd = {k: float(np.std([e[k] for e in evs])) for k in ("Q_orig", "AMI")}
            budget = T_sparsify + T_det
            mr, nrest = ladder.matched(budget)

            # resolution-matched control (Leiden has a resolution knob)
            rm = resgrid.nearest(agg["nc"])
            res = dict(Q_resmatch=rm["Q_orig"], AMI_resmatch=rm["AMI"],
                       ARI_resmatch=rm["ARI"], nc_resmatch=rm["nc"],
                       gamma_resmatch=rm["gamma"])

            row = dict(
                arm="A", detector="leiden", mu=mu, d_nom=d, lfr_seed=ls,
                gen_mode=meta["gen_mode"], **gm,
                sparsifier=kind, target_ret=target, realized_ret=round(realized, 5),
                knob=round(float(knob), 6), status=status, lspar_floor=round(min_ret, 5),
                m_sparse=gs.ecount(),
                Q_base_mean=float(np.mean(Qb)), Q_base_std=float(np.std(Qb)),
                Q_base_best=float(np.max(Qb)),
                AMI_base_mean=float(np.mean(Ab)), AMI_base_std=float(np.std(Ab)),
                ARI_base_mean=float(np.mean(Rb)),
                nc_base=float(np.mean([r["nc"] for r in base])),
                cv_base=float(np.mean([r["cv_size"] for r in base])),
                T_leiden_orig=Tb, T_sparsify=T_sparsify, T_detect_sparse=T_det,
                T_pipeline=budget,
                Q_orig_transfer=agg["Q_orig"], Q_orig_transfer_std=agg_sd["Q_orig"],
                AMI=agg["AMI"], AMI_std=agg_sd["AMI"], ARI=agg["ARI"],
                AMI_chance=agg["AMI_chance"], ARI_chance=agg["ARI_chance"],
                nc_sparse=agg["nc"], cv_sparse=agg["cv_size"],
                budget=budget, n_restarts=nrest,
                Q_matched_best=mr["Q_orig"], AMI_at_matched=mr["AMI"],
                ARI_at_matched=mr["ARI"], nc_matched=mr["nc"],
                dQ_vs_matched=agg["Q_orig"] - mr["Q_orig"],
                dQ_vs_base_best=agg["Q_orig"] - float(np.max(Qb)),
                dQ_vs_base_mean=agg["Q_orig"] - float(np.mean(Qb)),
                dAMI_vs_base_mean=agg["AMI"] - float(np.mean(Ab)),
                dAMI_vs_matched=agg["AMI"] - mr["AMI"],
                dARI_vs_base_mean=agg["ARI"] - float(np.mean(Rb)),
                speedup_pipeline=Tb / budget, speedup_detect_only=Tb / T_det,
                **res)
            append_row(out, row)
            log(f"  {kind:7s} t={target:<5} ret={realized:.4f} ({status}) "
                f"dQ_match={row['dQ_vs_matched']:+.5f} dQ_best={row['dQ_vs_base_best']:+.5f} "
                f"AMI={agg['AMI']:.4f} dAMI={row['dAMI_vs_base_mean']:+.4f} "
                f"nc={agg['nc']:.0f} cv={agg['cv_size']:.2f}")
            del gs
        del g, lsp, Ej, J


# ---------------------------------------------------------------------------
# ARM B
# ---------------------------------------------------------------------------

ARMB_KEY = ["mu", "d_nom", "lfr_seed", "detector", "sparsifier", "target_ret"]


def run_armB(cells=None, detectors=("metis", "eigen_proxy", "spectral_proxy")):
    out = HERE / "results_armB.csv"
    seen = done_keys(out, ARMB_KEY)
    rng = np.random.default_rng(11)
    combos = [(mu, d, s) for mu in ARMB_MUS for d in DEGS for s in LFR_SEEDS]
    if cells:
        want = set(cells)
        combos = [c for c in combos if f"{c[0]}_{c[1]}" in want or str(c[1]) in want]

    for mu, d, ls in combos:
        tag = f"mu{mu}_d{d}_s{ls}"
        try:
            E, y, meta = gen_lfr(mu, d, ls)
        except Exception as ex:
            log(f"[GEN-FAIL] {tag}: {ex}")
            continue
        g = graph_from_E(E)
        gm = graph_meta(g, y)
        k_fixed = gm["n_comm_planted"]          # k pinned per graph across ALL arms
        log(f"\n{'='*76}\n[B] {tag} {meta['gen_mode']} m={gm['m']:,} "
            f"d_avg={gm['d_avg_real']:.1f} k_fixed={k_fixed}\n{'='*76}")

        tj = time.perf_counter()
        Ej, J, deg = edge_jaccard(g)
        T_jac = time.perf_counter() - tj
        lsp = LSpar(g, J=J, E=Ej, deg=deg)
        T_rank = time.perf_counter() - tj - T_jac
        ds = dspar_scores(g)

        for det in detectors:
            fn = DETECTORS[det]
            if det == "spectral_proxy" and not (mu == 0.5 and ls == LFR_SEEDS[0]):
                continue
            # baseline on the ORIGINAL graph, same fixed k
            try:
                t0 = time.perf_counter()
                mb, T_base = fn(g, k_fixed, seed=0)
                eb = evaluate(g, mb, y, rng=rng)
            except Exception as ex:
                log(f"  [{det}] baseline FAILED: {type(ex).__name__}: {ex}")
                append_row(HERE / "detector_failures.csv",
                           dict(arm="B", mu=mu, d_nom=d, lfr_seed=ls, detector=det,
                                stage="baseline", error=f"{type(ex).__name__}: {ex}"))
                continue
            log(f"  [{det}] baseline Q={eb['Q_orig']:.5f} AMI={eb['AMI']:.4f} "
                f"ARI={eb['ARI']:.4f} nc={eb['nc']} cv={eb['cv_size']:.3f} T={T_base:.1f}s")

            for kind, target in itertools.product(SPARSIFIERS, TARGETS):
                # SpectralClustering costs ~100-400 s per call at n=1e4; it runs on a
                # REDUCED grid (mu=0.5, LFR seed 1, lspar/dspar, targets .5 and .05).
                if det == "spectral_proxy" and not (
                        mu == 0.5 and ls == LFR_SEEDS[0]
                        and kind in ("lspar", "dspar") and target in (0.5, 0.05)):
                    continue
                key = (str(mu), str(d), str(ls), det, kind, str(target))
                if key in seen:
                    continue
                eids, realized, knob, status, T_sel = sparsify_edges(
                    g, kind, target, 500 + ls, lspar=lsp, dscores=ds)
                T_sparsify = T_sel + (T_jac + T_rank if kind == "lspar" else 0.0)
                gs = subgraph_from_eids(g, eids)
                try:
                    memb, T_det = fn(gs, k_fixed, seed=0)
                    ev = evaluate(g, memb, y, rng=rng)
                except Exception as ex:
                    log(f"    [{det}] {kind}@{target} FAILED {type(ex).__name__}")
                    append_row(HERE / "detector_failures.csv",
                               dict(arm="B", mu=mu, d_nom=d, lfr_seed=ls, detector=det,
                                    stage=f"{kind}@{target}",
                                    error=f"{type(ex).__name__}: {ex}"))
                    del gs
                    continue
                row = dict(
                    arm="B", detector=det, is_proxy=int(det != "metis"),
                    mu=mu, d_nom=d, lfr_seed=ls, gen_mode=meta["gen_mode"], **gm,
                    k_fixed=k_fixed, sparsifier=kind, target_ret=target,
                    realized_ret=round(realized, 5), knob=round(float(knob), 6),
                    status=status, m_sparse=gs.ecount(),
                    Q_base=eb["Q_orig"], AMI_base=eb["AMI"], ARI_base=eb["ARI"],
                    nc_base=eb["nc"], cv_base=eb["cv_size"], T_base=T_base,
                    Q_orig_transfer=ev["Q_orig"], AMI=ev["AMI"], ARI=ev["ARI"],
                    nc_sparse=ev["nc"], cv_sparse=ev["cv_size"],
                    AMI_chance=ev["AMI_chance"], ARI_chance=ev["ARI_chance"],
                    T_sparsify=T_sparsify, T_detect_sparse=T_det,
                    T_pipeline=T_sparsify + T_det,
                    dQ=ev["Q_orig"] - eb["Q_orig"],
                    dAMI=ev["AMI"] - eb["AMI"], dARI=ev["ARI"] - eb["ARI"],
                    dcv=ev["cv_size"] - eb["cv_size"],
                    speedup_pipeline=T_base / max(1e-9, T_sparsify + T_det),
                    speedup_detect_only=T_base / max(1e-9, T_det))
                append_row(out, row)
                log(f"    {kind:7s} t={target:<5} ret={realized:.4f} "
                    f"dQ={row['dQ']:+.5f} dAMI={row['dAMI']:+.4f} dARI={row['dARI']:+.4f} "
                    f"cv {eb['cv_size']:.2f}->{ev['cv_size']:.2f}")
                del gs
        del g, lsp, Ej, J


# ---------------------------------------------------------------------------
# ARM C  -- noise injection
# ---------------------------------------------------------------------------

ARMC_KEY = ["mu", "d_nom", "lfr_seed", "noise_pct", "detector", "sparsifier", "target_ret"]


def inject_noise(g, pct, seed):
    """Add pct% of m uniformly random NEW edges (no rewiring, labels unchanged)."""
    n, m = g.vcount(), g.ecount()
    k = int(round(m * pct / 100.0))
    if k == 0:
        return g.copy(), 0
    rs = np.random.RandomState(seed)
    have = set(map(tuple, np.sort(np.asarray(g.get_edgelist()), axis=1).tolist()))
    new = []
    while len(new) < k:
        need = k - len(new)
        u = rs.randint(0, n, size=need * 2)
        v = rs.randint(0, n, size=need * 2)
        for a, b in zip(u.tolist(), v.tolist()):
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            if e in have:
                continue
            have.add(e)
            new.append(e)
            if len(new) >= k:
                break
    gn = g.copy()
    gn.add_edges(new)
    return gn, len(new)


def run_armC():
    out = HERE / "results_armC.csv"
    seen = done_keys(out, ARMC_KEY)
    rng = np.random.default_rng(13)
    mu = 0.5
    for d, ls in itertools.product(ARMC_DEGS, LFR_SEEDS):
        try:
            E, y, meta = gen_lfr(mu, d, ls)
        except Exception as ex:
            log(f"[GEN-FAIL] C mu{mu}_d{d}_s{ls}: {ex}")
            continue
        g0 = graph_from_E(E)
        gm0 = graph_meta(g0, y)
        k_fixed = gm0["n_comm_planted"]
        log(f"\n{'='*76}\n[C] mu{mu}_d{d}_s{ls}  m_clean={gm0['m']:,} "
            f"d_avg={gm0['d_avg_real']:.1f} k={k_fixed}\n{'='*76}")

        for x in ARMC_NOISE:
            gn, n_added = inject_noise(g0, x, seed=900 + ls)
            gn.simplify(multiple=True, loops=True)
            mn = gn.ecount()
            # baselines on the NOISY graph
            base = []
            for s in BASE_SEEDS[:3]:
                memb, dt = leiden(gn, s)
                e = evaluate(gn, memb, y)
                e["Q_clean"] = float(g0.modularity(memb.tolist()))
                e["dt"] = dt
                base.append(e)
            mb_metis, T_bm = metis_partition(gn, k_fixed)
            eb_metis = evaluate(gn, mb_metis, y)
            eb_metis["Q_clean"] = float(g0.modularity(mb_metis.tolist()))
            log(f"  x={x:3d}% m={mn:,} (+{n_added:,}) "
                f"leiden AMI={np.mean([e['AMI'] for e in base]):.4f} "
                f"metis AMI={eb_metis['AMI']:.4f}")

            tj = time.perf_counter()
            Ej, J, deg = edge_jaccard(gn)
            T_jac = time.perf_counter() - tj
            lsp = LSpar(gn, J=J, E=Ej, deg=deg)
            T_rank = time.perf_counter() - tj - T_jac
            ds = dspar_scores(gn)

            for kind, target in itertools.product(["lspar", "dspar"], ARMC_TARGETS):
                eids, realized, knob, status, T_sel = sparsify_edges(
                    gn, kind, target, 700 + ls, lspar=lsp, dscores=ds)
                gs = subgraph_from_eids(gn, eids)
                for det in ("leiden", "metis"):
                    key = (str(mu), str(d), str(ls), str(x), det, kind, str(target))
                    if key in seen:
                        continue
                    if det == "leiden":
                        evs = []
                        for s in SPARSE_SEEDS:
                            memb, _ = leiden(gs, s)
                            e = evaluate(gn, memb, y, rng=rng)
                            e["Q_clean"] = float(g0.modularity(memb.tolist()))
                            evs.append(e)
                        ev = {k: float(np.mean([e[k] for e in evs])) for k in evs[0]}
                        bAMI = float(np.mean([e["AMI"] for e in base]))
                        bARI = float(np.mean([e["ARI"] for e in base]))
                        bQ = float(np.mean([e["Q_orig"] for e in base]))
                        bQc = float(np.mean([e["Q_clean"] for e in base]))
                        bnc = float(np.mean([e["nc"] for e in base]))
                    else:
                        memb, _ = metis_partition(gs, k_fixed)
                        ev = evaluate(gn, memb, y, rng=rng)
                        ev["Q_clean"] = float(g0.modularity(memb.tolist()))
                        bAMI, bARI = eb_metis["AMI"], eb_metis["ARI"]
                        bQ, bQc, bnc = eb_metis["Q_orig"], eb_metis["Q_clean"], eb_metis["nc"]
                    row = dict(arm="C", mu=mu, d_nom=d, lfr_seed=ls, noise_pct=x,
                               detector=det, sparsifier=kind, target_ret=target,
                               realized_ret=round(realized, 5), status=status,
                               m_clean=gm0["m"], m_noisy=mn, n_added=n_added,
                               d_avg_clean=gm0["d_avg_real"],
                               d_avg_noisy=2.0 * mn / gn.vcount(),
                               k_planted=k_fixed, m_sparse=gs.ecount(),
                               AMI_base=bAMI, ARI_base=bARI,
                               Q_noisy_base=bQ, Q_clean_base=bQc, nc_base=bnc,
                               AMI=ev["AMI"], ARI=ev["ARI"],
                               Q_noisy_transfer=ev["Q_orig"], Q_clean_transfer=ev["Q_clean"],
                               nc_sparse=ev["nc"], cv_sparse=ev["cv_size"],
                               AMI_chance=ev.get("AMI_chance", ""),
                               dAMI=ev["AMI"] - bAMI, dARI=ev["ARI"] - bARI,
                               dQ_noisy=ev["Q_orig"] - bQ, dQ_clean=ev["Q_clean"] - bQc)
                    append_row(out, row)
                    log(f"    x={x:3d} {det:6s} {kind:6s} t={target} ret={realized:.3f} "
                        f"AMI={ev['AMI']:.4f} dAMI={row['dAMI']:+.4f} "
                        f"dQ_noisy={row['dQ_noisy']:+.5f}")
                del gs
            del gn, lsp, Ej, J
        del g0


def run_gen():
    ok = fail = 0
    for mu, d, s in itertools.product(MUS, DEGS, LFR_SEEDS):
        try:
            E, y, meta = gen_lfr(mu, d, s)
            g = graph_from_E(E)
            gm = graph_meta(g, y)
            log(f"mu={mu} d={d} s={s} {meta['gen_mode']:20s} m={gm['m']:>8,} "
                f"d_avg={gm['d_avg_real']:7.2f} mu_real={gm['mu_real']:.3f} "
                f"k={gm['n_comm_planted']:4d} cs=[{gm['comm_size_min']},{gm['comm_size_max']}] "
                f"t={meta['gen_time']:.1f}s")
            append_row(HERE / "lfr_generation.csv",
                       dict(mu=mu, d_nom=d, lfr_seed=s, **meta, **gm))
            ok += 1
        except Exception as ex:
            log(f"mu={mu} d={d} s={s} GEN-FAIL {type(ex).__name__}: {ex}")
            append_row(HERE / "gen_failures.csv",
                       dict(arm="gen", mu=mu, d_nom=d, lfr_seed=s,
                            error=f"{type(ex).__name__}: {ex}"))
            fail += 1
    log(f"\ngenerated {ok}, failed {fail}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "gen"
    rest = sys.argv[2:]
    if stage == "gen":
        run_gen()
    elif stage == "armA":
        run_armA(rest or None)
    elif stage == "armB":
        dets = [a[4:] for a in rest if a.startswith("det=")]
        cells = [a for a in rest if not a.startswith("det=")]
        if dets:
            run_armB(cells or None, detectors=tuple(dets[0].split("+")))
        else:
            run_armB(cells or None)
    elif stage == "armC":
        run_armC()
    else:
        raise SystemExit(f"unknown stage {stage}")
