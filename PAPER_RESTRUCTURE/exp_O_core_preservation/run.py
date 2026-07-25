#!/usr/bin/env python
"""
Experiment O: core-preservation / periphery-fragmentation.

Pre-registered design: exp_O_core_preservation/DESIGN.md (verbatim extract of the
EXPLORATION.md registration, commit 953aa10). This script implements exactly that
design; nothing here was chosen after seeing results.

Hypothesis: under edge sparsification the sparse-graph partition shatters into many
small clusters, but these are loose/peripheral pieces; the CORES of the main
communities are preserved.

Per node, on the ORIGINAL graph g and its partition P0 = Leiden(g, seed 100):
    embeddedness = (# neighbours in own P0 community) / degree
    k-core index, degree                      (cross-check axes, DESIGN constraint 3)

Three measurements, computed identically for every arm/condition:
  (1) agreement-by-embeddedness-decile between P0 and P' (community-matched:
      Hungarian = primary, plurality = secondary)
  (2) core cohesion: for each P0 community >=20 nodes, core = top-50%-embeddedness
      members; cohesion = largest fraction of the core landing in one P' cluster
      (periphery cohesion of the bottom 50% recorded alongside for contrast)
  (3) fragment composition: nodes in P' clusters of size <10, embeddedness-decile
      distribution vs the graph baseline (uniform 0.1 by construction: deciles are
      equal-sized rank deciles)

Conditions per arm:
  jitter    P' = Leiden(g, seed 101/102)          [seed-noise floor, extra control]
  dspar     calibrated DSpar, alpha 0.8 / 0.5, 2 spar seeds x 3 Leiden seeds
  lspar     L-Spar, realized-retention targets 0.5 / 0.2, 3 Leiden seeds
  resmatch  Leiden on the ORIGINAL graph, RBConfiguration gamma bisected so that
            k(resmatch) ~= k(P') of the matching sparsifier condition [control (b)]
  perm      P' labels randomly permuted over nodes              [control (c), chance]

Arms:
  real      g
  null      degree-preserving rewired g (10 swaps/edge, igraph simple rewire), with
            its OWN P0_null and its own embeddedness                [control (a)]

Outputs (all appended incrementally, row by row):
  results.csv     one row per (network, arm, condition, sparsifier, seeds)
  cohesion.csv    per-P0-community core/periphery cohesion (first Leiden seed only)
  node_attrs.csv  per node: degree, coreness, embeddedness, P0 community, deciles

Usage:
  run.py run  email-Eu-core,wiki-Vote,...        (default = small five)
"""

import csv
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import igraph as ig
import leidenalg as la
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score

REPO = Path("/home/md724/community_detection_spectral")
DATASETS_DIR = REPO / "datasets"
HERE = Path(__file__).resolve().parent

N_ITER = 2                       # repo convention
P0_SEED = 100                    # reference partition seed
JITTER_SEEDS = [101, 102]        # seed-noise floor
SPAR_SEEDS = [200, 201]          # DSpar sampling seeds (>=2, DESIGN)
LEIDEN_SEEDS = [300, 301, 302]   # 3 Leiden seeds per arm (DESIGN)
DSPAR_ALPHAS = [0.8, 0.5]        # calibrated -> true retention ~= alpha
LSPAR_TARGETS = [0.5, 0.2]
SWAPS_PER_EDGE = 10
REWIRE_SEED = 42
PERM_SEED = 777
DEC_SEEDS = (11, 12, 13)         # tie-breaking seeds for emb/core/deg deciles
MIN_COMM = 20                    # core cohesion: P0 communities >= 20 nodes
FRAG_MAX = 10                    # "fragment" = P' cluster with size < 10
DENSE_LIMIT = 4_000_000          # ka*kb above which Hungarian falls back to greedy

SMALL_FIVE = ["email-Eu-core", "wiki-Vote", "ca-HepTh", "ca-CondMat", "email-Enron"]
BIG_TWO = ["com-DBLP", "com-Amazon"]


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
# DSpar calibrated sampler (verbatim from exp_N_enron_anatomy/run.py, = exp_C/exp_K)
# ---------------------------------------------------------------------------

def dspar_scores(g):
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    return e, 1.0 / deg[e[:, 0]] + 1.0 / deg[e[:, 1]]


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


def sparsify(g, edge_arr, scores, alpha, seed):
    probs = _probs_calibrated(scores, alpha)
    rs = np.random.RandomState(seed)
    keep = rs.random_sample(len(scores)) < probs
    kept = edge_arr[keep]
    gs = ig.Graph(n=g.vcount(), edges=[tuple(x) for x in kept], directed=False)
    return gs, kept, kept.shape[0] / edge_arr.shape[0]


# ---------------------------------------------------------------------------
# L-Spar (verbatim from exp_L_lspar/run.py)
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


def subgraph_from_eids(g, eids):
    gs = ig.Graph(n=g.vcount())
    E = np.asarray(g.get_edgelist(), dtype=np.int64)[eids]
    gs.add_edges(E)
    return gs


# ---------------------------------------------------------------------------
# Leiden (repo convention)
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


def leiden_matched(g, seed, target, tol=0.05, max_iter=24):
    """Leiden on the ORIGINAL graph, gamma bisected so #clusters ~= target.
    Verbatim structure from exp_L_lspar/run.py:leiden_matched."""
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
# CSV helpers (incremental append; crash insurance)
# ---------------------------------------------------------------------------

def append_row(path, row):
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def append_rows(path, rows):
    if not rows:
        return
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Node statistics + deciles
# ---------------------------------------------------------------------------

def node_stats(g, memb):
    """embeddedness (fraction of neighbours in own community), degree, coreness."""
    n = g.vcount()
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    memb = np.asarray(memb)
    deg = np.asarray(g.degree(), dtype=np.int64)
    if E.size:
        same = memb[E[:, 0]] == memb[E[:, 1]]
        cnt = (np.bincount(E[same, 0], minlength=n)
               + np.bincount(E[same, 1], minlength=n))
    else:
        cnt = np.zeros(n, dtype=np.int64)
    emb = np.where(deg > 0, cnt / np.maximum(deg, 1), 0.0)
    core = np.asarray(g.coreness(), dtype=np.int64)
    return deg, emb, core


def rank_deciles(x, seed, n_bins=10):
    """Equal-sized rank deciles; ties broken at random (fixed seed) so that heavy
    ties (e.g. embeddedness == 1.0) spread evenly instead of by node id."""
    n = len(x)
    rs = np.random.RandomState(seed)
    order = np.lexsort((rs.random_sample(n), np.asarray(x, dtype=np.float64)))
    dec = np.empty(n, dtype=np.int64)
    dec[order] = (np.arange(n) * n_bins) // n
    return dec


def decile_mean(dec, vals, n_bins=10):
    cnt = np.bincount(dec, minlength=n_bins).astype(np.float64)
    s = np.bincount(dec, weights=vals.astype(np.float64), minlength=n_bins)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


# ---------------------------------------------------------------------------
# Community matching
# ---------------------------------------------------------------------------

def match_maps(M, ka, kb):
    """M = ka x kb contingency (csr). Returns hungarian map, plurality map, method.
    Maps send a P' cluster id -> P0 community id (or -1 if unmatched)."""
    coo = M.tocoo()
    # plurality: each P' cluster -> P0 community with the largest overlap
    plur = -np.ones(kb, dtype=np.int64)
    best = np.zeros(kb, dtype=np.float64)
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if v > best[j]:
            best[j] = v
            plur[j] = i
    # hungarian (exact when the dense matrix is affordable, else greedy)
    hung = -np.ones(kb, dtype=np.int64)
    greedy_delta = ""
    if ka * kb <= DENSE_LIMIT:
        D = M.toarray()
        r, c = linear_sum_assignment(-D)
        for i, j in zip(r, c):
            if D[i, j] > 0:
                hung[j] = i
        method = "hungarian"
        gh = _greedy_map(coo, ka, kb)
        greedy_delta = float(np.mean(gh != hung))
    else:
        hung = _greedy_map(coo, ka, kb)
        method = "greedy"
    return hung, plur, method, greedy_delta


def _greedy_map(coo, ka, kb):
    order = np.argsort(-coo.data, kind="stable")
    used_a = np.zeros(ka, dtype=bool)
    used_b = np.zeros(kb, dtype=bool)
    out = -np.ones(kb, dtype=np.int64)
    rows, cols = coo.row, coo.col
    for idx in order:
        i, j = rows[idx], cols[idx]
        if used_a[i] or used_b[j]:
            continue
        used_a[i] = True
        used_b[j] = True
        out[j] = i
    return out


# ---------------------------------------------------------------------------
# The three measurements
# ---------------------------------------------------------------------------

def build_ctx(g, name, arm):
    """P0, node stats, deciles, per-community cores. Everything measured on g."""
    t0 = time.perf_counter()
    memb0, Q0, k0, _ = leiden(g, P0_SEED)
    _, a = np.unique(memb0, return_inverse=True)
    ka = int(a.max()) + 1
    deg, emb, core = node_stats(g, a)
    dec_emb = rank_deciles(emb, DEC_SEEDS[0])
    dec_core = rank_deciles(core, DEC_SEEDS[1])
    dec_deg = rank_deciles(deg, DEC_SEEDS[2])

    order = np.argsort(a, kind="stable")
    counts = np.bincount(a, minlength=ka)
    offs = np.concatenate([[0], np.cumsum(counts)])
    cores, periphs, comm_ids, comm_sizes = [], [], [], []
    for c in range(ka):
        mem = order[offs[c]:offs[c + 1]]
        if mem.size < MIN_COMM:
            continue
        e = emb[mem]
        rs = np.random.RandomState(1000 + c)
        o = np.lexsort((rs.random_sample(mem.size), -e))   # emb desc, random ties
        h = int(np.ceil(mem.size / 2.0))
        cores.append(mem[o[:h]])
        periphs.append(mem[o[h:]])
        comm_ids.append(c)
        comm_sizes.append(int(mem.size))

    ctx = dict(name=name, arm=arm, n=g.vcount(), m=g.ecount(), memb0=a, ka=ka,
               Q0=Q0, k0=k0, deg=deg, emb=emb, core=core, dec_emb=dec_emb,
               dec_core=dec_core, dec_deg=dec_deg, cores=cores, periphs=periphs,
               comm_ids=comm_ids, comm_sizes=comm_sizes)
    log(f"  P0: Q={Q0:.6f} k0={k0} comms>={MIN_COMM}: {len(cores)} "
        f"emb_mean={emb.mean():.4f} frac(emb=1)={float((emb >= 1.0).mean()):.4f} "
        f"({time.perf_counter()-t0:.1f}s)")
    log(f"  corr(emb,coreness)={np.corrcoef(emb, core)[0,1]:+.4f} "
        f"corr(emb,deg)={np.corrcoef(emb, deg)[0,1]:+.4f} "
        f"corr(coreness,deg)={np.corrcoef(core, deg)[0,1]:+.4f}")
    return ctx


def measure(ctx, memb_p, fields, write_cohesion):
    """All three measurements of a partition P' against ctx's P0."""
    t0 = time.perf_counter()
    n, a, ka = ctx["n"], ctx["memb0"], ctx["ka"]
    _, b = np.unique(np.asarray(memb_p), return_inverse=True)
    kb = int(b.max()) + 1
    M = sp.coo_matrix((np.ones(n), (a, b)), shape=(ka, kb)).tocsr()
    hung, plur, method, gdelta = match_maps(M, ka, kb)

    ag_h = (hung[b] == a).astype(np.float64)
    ag_p = (plur[b] == a).astype(np.float64)

    # (1) agreement by embeddedness decile (+ coreness / degree cross-checks)
    ah = decile_mean(ctx["dec_emb"], ag_h)
    ap = decile_mean(ctx["dec_emb"], ag_p)
    ahc = decile_mean(ctx["dec_core"], ag_h)
    ahd = decile_mean(ctx["dec_deg"], ag_h)

    def gap3(v):
        return float(np.mean(v[7:10])), float(np.mean(v[0:3]))

    ah_top3, ah_bot3 = gap3(ah)
    ap_top3, ap_bot3 = gap3(ap)
    ahc_top3, ahc_bot3 = gap3(ahc)
    ahd_top3, ahd_bot3 = gap3(ahd)
    rho = spearmanr(np.arange(10), ah).correlation if np.ptp(ah) > 0 else 0.0
    n_incr = int(np.sum(np.diff(ah) > 0))

    # (2) core cohesion
    coh, pcoh, coh_rows = [], [], []
    for c, sz, cr, pr in zip(ctx["comm_ids"], ctx["comm_sizes"],
                             ctx["cores"], ctx["periphs"]):
        _, cc = np.unique(b[cr], return_counts=True)
        v = float(cc.max()) / cr.size
        coh.append(v)
        if pr.size:
            _, pc = np.unique(b[pr], return_counts=True)
            w = float(pc.max()) / pr.size
        else:
            w = np.nan
        pcoh.append(w)
        if write_cohesion:
            coh_rows.append(dict(fields, comm=c, comm_size=sz, core_size=int(cr.size),
                                 core_emb_mean=float(ctx["emb"][cr].mean()),
                                 periph_emb_mean=(float(ctx["emb"][pr].mean())
                                                  if pr.size else ""),
                                 core_cohesion=v, periph_cohesion=w))
    coh = np.asarray(coh, dtype=np.float64)
    pcoh = np.asarray(pcoh, dtype=np.float64)

    # (3) fragment composition
    csize = np.bincount(b, minlength=kb)
    is_frag = csize[b] < FRAG_MAX
    nf = int(is_frag.sum())
    dec = ctx["dec_emb"]
    dsz = np.bincount(dec, minlength=10).astype(np.float64)
    fcnt = np.bincount(dec[is_frag], minlength=10).astype(np.float64) if nf else np.zeros(10)
    fshare = fcnt / nf if nf else np.full(10, np.nan)
    frate = fcnt / np.maximum(dsz, 1)

    row = dict(fields)
    row.update(
        n=n, m=ctx["m"], k0=ctx["ka"], kp=kb, Q0=ctx["Q0"],
        ami=float(adjusted_mutual_info_score(a, b)),
        match_method=method, greedy_vs_exact_map_diff=gdelta,
        agree_all_hung=float(ag_h.mean()), agree_all_plur=float(ag_p.mean()),
        ah_bot3=ah_bot3, ah_top3=ah_top3, ah_gap=ah_top3 - ah_bot3,
        ap_bot3=ap_bot3, ap_top3=ap_top3, ap_gap=ap_top3 - ap_bot3,
        spearman_hung=float(rho), n_incr_steps_hung=n_incr,
        core_ah_bot3=ahc_bot3, core_ah_top3=ahc_top3, core_ah_gap=ahc_top3 - ahc_bot3,
        deg_ah_bot3=ahd_bot3, deg_ah_top3=ahd_top3, deg_ah_gap=ahd_top3 - ahd_bot3,
        coh_median=float(np.median(coh)) if coh.size else np.nan,
        coh_mean=float(np.mean(coh)) if coh.size else np.nan,
        coh_q1=float(np.percentile(coh, 25)) if coh.size else np.nan,
        coh_q3=float(np.percentile(coh, 75)) if coh.size else np.nan,
        coh_frac_ge_0p8=float(np.mean(coh >= 0.8)) if coh.size else np.nan,
        n_comms_ge20=int(coh.size),
        periph_coh_median=float(np.nanmedian(pcoh)) if pcoh.size else np.nan,
        periph_coh_mean=float(np.nanmean(pcoh)) if pcoh.size else np.nan,
        frac_frag_nodes=nf / n, n_frag_nodes=nf,
        frag_share_bot2=float(fshare[0] + fshare[1]) if nf else np.nan,
        frag_enrich_bot2=float((fshare[0] + fshare[1]) / 0.2) if nf else np.nan,
        frag_share_top2=float(fshare[8] + fshare[9]) if nf else np.nan,
        frag_enrich_top2=float((fshare[8] + fshare[9]) / 0.2) if nf else np.nan,
        frag_rate_bot2=float(np.mean(frate[0:2])), frag_rate_top2=float(np.mean(frate[8:10])),
        emb_mean_bot3=float(np.mean(decile_mean(dec, ctx["emb"])[0:3])),
        emb_mean_top3=float(np.mean(decile_mean(dec, ctx["emb"])[7:10])),
        frac_emb_eq1=float((ctx["emb"] >= 1.0).mean()),
        seconds=time.perf_counter() - t0,
    )
    for i in range(10):
        row[f"ah_d{i}"] = float(ah[i])
    for i in range(10):
        row[f"ap_d{i}"] = float(ap[i])
    for i in range(10):
        row[f"fs_d{i}"] = float(fshare[i])
    for i in range(10):
        row[f"fr_d{i}"] = float(frate[i])
    for i in range(10):
        row[f"ahc_d{i}"] = float(ahc[i])
    for i in range(10):
        row[f"ahd_d{i}"] = float(ahd[i])
    return row, coh_rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def base_fields(ctx, condition, sparsifier, target, realized, spar_seed,
                leiden_seed, gamma, m_sparse):
    return dict(network=ctx["name"], arm=ctx["arm"], condition=condition,
                sparsifier=sparsifier, target_ret=target, realized_ret=realized,
                spar_seed=spar_seed, leiden_seed=leiden_seed, gamma=gamma,
                m_sparse=m_sparse)


def run_arm(g, name, arm, res_csv, coh_csv, node_csv):
    ctx = build_ctx(g, name, arm)
    n = ctx["n"]

    # node attributes (cross-check axes; DESIGN constraint 3)
    append_rows(node_csv, [dict(network=name, arm=arm, node=int(i),
                                degree=int(ctx["deg"][i]), coreness=int(ctx["core"][i]),
                                embeddedness=float(ctx["emb"][i]),
                                comm=int(ctx["memb0"][i]),
                                dec_emb=int(ctx["dec_emb"][i]),
                                dec_core=int(ctx["dec_core"][i]),
                                dec_deg=int(ctx["dec_deg"][i]))
                           for i in range(n)])

    def emit(memb_p, fields, write_coh):
        row, crows = measure(ctx, memb_p, fields, write_coh)
        append_row(res_csv, row)
        append_rows(coh_csv, crows)
        log(f"    [{fields['condition']:8s} {fields['sparsifier']:6s} "
            f"t={fields['target_ret']} ss={fields['spar_seed']} ls={fields['leiden_seed']}] "
            f"k'={row['kp']:6d} AMI={row['ami']:.3f} agree={row['agree_all_hung']:.3f} "
            f"gap={row['ah_gap']:+.3f} coh_med={row['coh_median']:.3f} "
            f"frag={row['frac_frag_nodes']:.3f} enrich_b2={row['frag_enrich_bot2'] if isinstance(row['frag_enrich_bot2'], str) else round(row['frag_enrich_bot2'],2)}")
        return row

    # ---- seed-noise floor -------------------------------------------------
    for s in JITTER_SEEDS:
        memb, _, _, _ = leiden(g, s)
        emit(memb, base_fields(ctx, "jitter", "", 1.0, 1.0, "", s, 1.0, ctx["m"]),
             s == JITTER_SEEDS[0])

    kp_by_cfg = {}     # (sparsifier, target) -> mean k' (for the resolution control)
    perm_pending = []  # (fields, memb) for the chance baseline

    # ---- DSpar ------------------------------------------------------------
    edge_arr, scores = dspar_scores(g)
    for alpha in DSPAR_ALPHAS:
        kps = []
        for si, ss in enumerate(SPAR_SEEDS):
            gs, kept, realized = sparsify(g, edge_arr, scores, alpha, ss)
            for li, ls in enumerate(LEIDEN_SEEDS):
                memb, _, kp, _ = leiden(gs, ls)
                f = base_fields(ctx, "dspar", "dspar", alpha, round(realized, 6),
                                ss, ls, 1.0, gs.ecount())
                r = emit(memb, f, si == 0 and li == 0)
                kps.append(r["kp"])
                if si == 0 and li == 0:
                    perm_pending.append((dict(f, condition="perm"), memb))
            del gs
        kp_by_cfg[("dspar", alpha)] = float(np.mean(kps))
    del edge_arr, scores

    # ---- L-Spar -----------------------------------------------------------
    E, J, deg_j = edge_jaccard(g)
    ls_obj = LSpar(g, J=J, E=E, deg=deg_j)
    for target in LSPAR_TARGETS:
        e_used, realized, status = ls_obj.bisect_e(target)
        gs = subgraph_from_eids(g, ls_obj.select(e_used))
        log(f"  lspar target={target} e={e_used:.4f} realized={realized:.4f} ({status})")
        kps = []
        for li, lsd in enumerate(LEIDEN_SEEDS):
            memb, _, kp, _ = leiden(gs, lsd)
            f = base_fields(ctx, "lspar", "lspar", target, round(realized, 6),
                            "", lsd, 1.0, gs.ecount())
            r = emit(memb, f, li == 0)
            kps.append(r["kp"])
            if li == 0:
                perm_pending.append((dict(f, condition="perm"), memb))
        kp_by_cfg[("lspar", target)] = float(np.mean(kps))
        del gs
    del E, J, ls_obj

    # ---- resolution-matched control (b) -----------------------------------
    for (sparsifier, target), kp in kp_by_cfg.items():
        tgt = int(round(kp))
        t0 = time.perf_counter()
        memb, gamma, nc = leiden_matched(g, P0_SEED, tgt)
        log(f"  resmatch {sparsifier} t={target}: target_k={tgt} -> gamma={gamma:.5g} "
            f"k={nc} ({time.perf_counter()-t0:.1f}s)")
        f = base_fields(ctx, "resmatch", sparsifier, target, 1.0, "", P0_SEED,
                        round(gamma, 6), ctx["m"])
        emit(memb, f, True)
        for lsd in JITTER_SEEDS:
            memb2, _, _, _ = leiden(g, lsd, resolution=gamma)
            f2 = base_fields(ctx, "resmatch", sparsifier, target, 1.0, "", lsd,
                             round(gamma, 6), ctx["m"])
            emit(memb2, f2, False)

    # ---- permutation chance baseline (c) ----------------------------------
    rs = np.random.RandomState(PERM_SEED)
    for f, memb in perm_pending:
        emit(np.asarray(memb)[rs.permutation(n)], f, False)


def run(networks):
    res_csv = HERE / "results.csv"
    coh_csv = HERE / "cohesion.csv"
    node_csv = HERE / "node_attrs.csv"
    for name in networks:
        t0 = time.perf_counter()
        g = load_lcc_graph(name)
        log(f"\n{'='*78}\n{name}: n={g.vcount():,} m={g.ecount():,} "
            f"(load {time.perf_counter()-t0:.1f}s)\n{'='*78}")

        log(f"-- ARM real")
        run_arm(g, name, "real", res_csv, coh_csv, node_csv)

        t0 = time.perf_counter()
        gr = g.copy()
        ig.set_random_number_generator(random.Random(REWIRE_SEED))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gr.rewire(n=SWAPS_PER_EDGE * g.ecount(), mode="simple")
        ig.set_random_number_generator(random)
        log(f"-- ARM null (rewired {SWAPS_PER_EDGE} swaps/edge in "
            f"{time.perf_counter()-t0:.1f}s; degree seq identical: "
            f"{np.array_equal(np.sort(g.degree()), np.sort(gr.degree()))})")
        run_arm(gr, name, "null", res_csv, coh_csv, node_csv)
        del g, gr
        log(f"** {name} DONE")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if stage == "run":
        run(arg.split(",") if arg else SMALL_FIVE)
    else:
        raise SystemExit(f"unknown stage {stage}")
