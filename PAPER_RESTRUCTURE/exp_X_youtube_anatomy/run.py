#!/usr/bin/env python
"""
Experiment X -- com-Youtube mini-anatomy (exp_N-lite).

Pre-registered in DESIGN.md (2026-07-25).  Target: C9's SECOND claimed gain
(com-Youtube, calibrated DSpar, high alpha).  exp_H_youtube_sweep established
that the ONLY cell surviving Bonferroni over its 8 (sampler x alpha) cells is
  calibrated sampler, alpha = 0.95  ->  Q_seeded_mean 0.729484,
  gain vs plain mean +0.005150, gain vs E[best-of-2] +0.003056, z_mean +3.06.
That is the cell replicated and dissected here.

Arms (DESIGN):
  A. 20 plain Leiden restarts, seeds 100..119, n_iterations=2.
     (seeds 100..109 are exp_H's baseline seeds -> exact replication check.)
  B.  5 seeded pipelines, calibrated DSpar alpha=0.95, spar seeds 200..204 /
     leiden seeds 300..304: sparsify -> Leiden on G_sparse -> refine on G.
     (same seeds as exp_H -> exact replication check.)
  Arms are INTERLEAVED (4 plain, 1 pipeline, ... x5) so that wall-clock ratios
  stay valid under a concurrently loaded machine.

Measurements (DESIGN 1-4):
  1. statistical reality  -- gap, Mann-Whitney, EXACT permutation test over all
     C(25,5) label assignments, bootstrap CI, percentile of the seeded mean in
     the plain distribution, and the exp_H runtime-matched E[best-of-2] test.
  2. granularity          -- corr(k,Q) over the 20 plain restarts (+ leave-one-out
     jackknife), OLS residual of the seeded mean, k-matched plain subset (the
     registered control), shape stats (largest community share, communities >1%
     of n, top-5 share), and a SUPPLEMENTARY resolution(gamma)-matched plain arm
     used when the registered k-matched subset comes out empty.
  3. mechanism            -- cross-piece vs intra-piece DSpar removal rates for
     the baseline communities the seeded partition splits, and for the seeded
     communities that merge baseline communities (exp_N's 2.58x analogue), with
     endpoint degree-product ratios; fully vectorised over the edge array.
  4. compute              -- one pipeline vs one restart wall clock, and the
     matched-wall-clock bootstrap E[best-of-n].  ANY compute claim is phrased
     in expectation only (exp_N caveat C2).

Loader / samplers / Leiden calls are verbatim from exp_H_youtube_sweep/run.py
(which took them from exp_E / exp_C), so Q and k are directly comparable.

Outputs (lean -- Fuji disk is nearly full):
  runs.csv                 one row per Leiden run (incremental)
  mechanism_parents.csv    per-parent boundary-removal stats (capped)
  results.json             every number quoted in SUMMARY.md
  cache.npz                best plain + best seeded membership + keep mask
                           (scp'd back and DELETED from Fuji)
"""

import csv
import itertools
import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg as la
from scipy.sparse import coo_matrix
from scipy import stats as sstats
from sklearn.metrics import adjusted_mutual_info_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NETWORK = "com-Youtube"
EDGE_FILE = REPO / "datasets" / NETWORK / f"{NETWORK}.txt"

ALPHA = 0.95                      # exp_H's surviving cell
SAMPLER = "calibrated"
PLAIN_SEEDS = list(range(100, 120))          # arm A, 20 restarts
SPAR_SEEDS = [200, 201, 202, 203, 204]       # arm B
LEID_SEEDS = [300, 301, 302, 303, 304]
GAMMA_SEEDS = [500, 501, 502, 503, 504]      # supplementary resolution arm
N_ITER = 2
RNG = np.random.RandomState(20260725)

if "--smoke" in sys.argv:            # code-path validation on a small graph
    NETWORK = "email-Enron"
    EDGE_FILE = REPO / "datasets" / NETWORK / f"{NETWORK}.txt"
    HERE = Path(os.environ.get("SMOKE_OUT", "/tmp/expX_smoke"))
    HERE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# loading / sampling / leiden  (verbatim from exp_H_youtube_sweep/run.py)
# ---------------------------------------------------------------------------
def _parse_edge_file(path):
    parts, leftover = [], b""
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
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    del u, v
    key = np.unique(lo.astype(np.int64) * n + hi)
    lo = (key // n).astype(np.int32)
    hi = (key % n).astype(np.int32)
    del key
    g = ig.Graph(n=int(n))
    g.add_edges(np.column_stack([lo, hi]))
    del lo, hi
    g = g.connected_components(mode="weak").giant()
    g.simplify(multiple=True, loops=True)
    return g


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


def sparsify(g, edge_arr, probs, seed):
    rs = np.random.RandomState(seed)
    keep = rs.random_sample(probs.size) < probs
    kept = edge_arr[keep]
    gs = ig.Graph(n=g.vcount())
    gs.add_edges(kept)
    return gs, keep, kept.shape[0] / edge_arr.shape[0]


def leiden(g, seed, initial_membership=None, gamma=None):
    t0 = time.perf_counter()
    if gamma is None:
        if initial_membership is None:
            part = la.ModularityVertexPartition(g)
        else:
            _, memb = np.unique(np.asarray(initial_membership), return_inverse=True)
            part = la.ModularityVertexPartition(g, initial_membership=memb.tolist())
    else:
        part = la.RBConfigurationVertexPartition(g, resolution_parameter=float(gamma))
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    memb = np.asarray(part.membership, dtype=np.int32)
    dt = time.perf_counter() - t0
    Q = g.modularity(part.membership) if gamma is not None else part.modularity
    return memb, float(Q), len(part), dt


# ---------------------------------------------------------------------------
def shape_stats(memb, n):
    sizes = np.bincount(memb)
    sizes = np.sort(sizes[sizes > 0])[::-1]
    return dict(k=int(sizes.size),
                largest=int(sizes[0]),
                frac_largest=float(sizes[0] / n),
                n_comm_gt_1pct=int((sizes >= 0.01 * n).sum()),
                top5_share=float(sizes[:5].sum() / n),
                n_comm_ge20=int((sizes >= 20).sum()))


CSV_FIELDS = ["arm", "seed", "spar_seed", "alpha", "gamma", "retention",
              "Q", "Q_raw", "k", "k_sparse", "largest", "frac_largest",
              "n_comm_gt_1pct", "top5_share", "n_comm_ge20",
              "T", "T_sparsify", "T_leiden_sparse", "T_refine", "rss_gb"]


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ---------------------------------------------------------------------------
# stage 1 -- interleaved arms
# ---------------------------------------------------------------------------
def stage1(g, runs_csv):
    n, m = g.vcount(), g.ecount()
    edge_arr, scores = dspar_scores(g)
    probs = _probs_calibrated(scores, ALPHA)
    del scores

    plain, seeded = [], []
    memb_plain, memb_seeded, keep_seeded = {}, {}, {}

    plain_q = list(PLAIN_SEEDS)
    pipe_q = list(zip(SPAR_SEEDS, LEID_SEEDS))
    print(f"[stage1] interleaving {len(plain_q)} plain restarts with "
          f"{len(pipe_q)} pipelines", flush=True)

    while plain_q or pipe_q:
        for _ in range(4):                      # 4 plain
            if not plain_q:
                break
            s = plain_q.pop(0)
            memb, Q, k, dt = leiden(g, seed=s)
            sh = shape_stats(memb, n)
            row = dict(arm="plain", seed=s, Q=f"{Q:.8f}", T=f"{dt:.2f}",
                       rss_gb=f"{rss():.2f}", **sh)
            append_row(runs_csv, row)
            plain.append(dict(seed=s, Q=Q, T=dt, **sh))
            memb_plain[s] = memb
            print(f"  plain seed={s} Q={Q:.6f} k={k} largest={sh['largest']} "
                  f"t={dt:.1f}s", flush=True)
        if pipe_q:                              # 1 pipeline
            ss, ls = pipe_q.pop(0)
            t0 = time.perf_counter()
            gs, keep, ret = sparsify(g, edge_arr, probs, seed=ss)
            t_spar = time.perf_counter() - t0
            memb_s, _, k_s, t_ls = leiden(gs, seed=ls)
            q_raw = float(g.modularity(memb_s.tolist()))
            del gs
            memb_f, Q, k, t_rf = leiden(g, seed=ls, initial_membership=memb_s)
            del memb_s
            sh = shape_stats(memb_f, n)
            T = t_spar + t_ls + t_rf
            row = dict(arm="seeded", seed=ls, spar_seed=ss, alpha=ALPHA,
                       retention=f"{ret:.6f}", Q=f"{Q:.8f}", Q_raw=f"{q_raw:.8f}",
                       k_sparse=k_s, T=f"{T:.2f}", T_sparsify=f"{t_spar:.2f}",
                       T_leiden_sparse=f"{t_ls:.2f}", T_refine=f"{t_rf:.2f}",
                       rss_gb=f"{rss():.2f}", **sh)
            append_row(runs_csv, row)
            seeded.append(dict(seed=ls, spar_seed=ss, retention=ret, Q=Q,
                               Q_raw=q_raw, k_sparse=k_s, T=T, T_sparsify=t_spar,
                               T_leiden_sparse=t_ls, T_refine=t_rf, **sh))
            memb_seeded[ls] = memb_f
            keep_seeded[ss] = keep
            print(f"  seeded spar={ss} leiden={ls} ret={ret:.4f} "
                  f"Qraw={q_raw:.6f} Q={Q:.6f} k={k} largest={sh['largest']} "
                  f"T={T:.1f}s", flush=True)
    return plain, seeded, memb_plain, memb_seeded, keep_seeded, edge_arr


# ---------------------------------------------------------------------------
# stage 2 -- supplementary resolution(gamma)-matched plain arm
# ---------------------------------------------------------------------------
def stage2_gamma(g, runs_csv, target_k, plain_k_mean):
    """Plain Leiden with RB resolution gamma<1 tuned so k lands near the seeded
    k, scored with STANDARD modularity.  Supplementary control, used because the
    registered k-matched subset can be empty (seeded k below the plain k range).
    """
    n = g.vcount()
    probes = []

    def probe(gamma, seed):
        memb, Q, k, dt = leiden(g, seed=seed, gamma=gamma)
        sh = shape_stats(memb, n)
        probes.append(dict(gamma=float(gamma), k=int(k), Q=float(Q)))
        append_row(runs_csv, dict(arm="gamma_probe", seed=seed,
                                  gamma=f"{gamma:.4f}", Q=f"{Q:.8f}",
                                  T=f"{dt:.2f}", rss_gb=f"{rss():.2f}", **sh))
        print(f"  gamma probe g={gamma:.4f} k={k} Q={Q:.6f} t={dt:.1f}s",
              flush=True)
        return k

    for i, gm in enumerate((0.90, 0.80, 0.70)):
        probe(gm, 490 + i)
    # log-log least squares through the probes plus the gamma=1 anchor
    xs = np.log([p["gamma"] for p in probes] + [1.0])
    ys = np.log([p["k"] for p in probes] + [plain_k_mean])
    a, b = np.polyfit(xs, ys, 1)             # log k = a*log gamma + b
    gstar = float(np.clip(np.exp((np.log(target_k) - b) / a), 0.3, 0.999)) \
        if a != 0 else 0.9
    # if a probe already lands closer than the fit's expected error, take it
    best = min(probes, key=lambda p: abs(p["k"] - target_k))
    if abs(best["k"] - target_k) <= 0.02 * target_k:
        gstar = best["gamma"]
    print(f"  log-log fit slope={a:.3f}; gamma* = {gstar:.4f} "
          f"(target k={target_k:.0f}, closest probe k={best['k']})", flush=True)
    rows = []
    for s in GAMMA_SEEDS:
        memb, Q, k, dt = leiden(g, seed=s, gamma=gstar)
        sh = shape_stats(memb, n)
        append_row(runs_csv, dict(arm="gamma_matched", seed=s,
                                  gamma=f"{gstar:.4f}", Q=f"{Q:.8f}",
                                  T=f"{dt:.2f}", rss_gb=f"{rss():.2f}", **sh))
        rows.append(dict(seed=s, gamma=gstar, Q=Q, T=dt, **sh))
        print(f"  gamma_matched seed={s} g={gstar:.4f} Q={Q:.6f} k={k} "
              f"t={dt:.1f}s", flush=True)
    return gstar, probes, rows


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def exact_perm_pvalue(a, b):
    """P(mean(perm seeded group) >= observed) over ALL C(na+nb, nb) splits."""
    pooled = np.concatenate([a, b])
    N, nb = pooled.size, b.size
    obs = b.mean() - a.mean()
    tot = pooled.sum()
    cnt = 0
    tot_splits = 0
    for idx in itertools.combinations(range(N), nb):
        sb = pooled[list(idx)].sum()
        d = sb / nb - (tot - sb) / (N - nb)
        cnt += (d >= obs - 1e-15)
        tot_splits += 1
    return float(cnt / tot_splits), int(tot_splits)


def boot_ci_gap(a, b, B=20000, rs=None):
    rs = rs or RNG
    d = np.empty(B)
    for i in range(B):
        d[i] = (rs.choice(b, b.size, replace=True).mean()
                - rs.choice(a, a.size, replace=True).mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(d.mean())


def best_of_k_exact(vals, k):
    return np.array([max(c) for c in itertools.combinations(vals, k)])


def boot_best_of_n(vals, n, B=20000, rs=None):
    rs = rs or RNG
    return rs.choice(vals, size=(B, n), replace=True).max(axis=1)


def jackknife_corr(x, y):
    r_full = float(np.corrcoef(x, y)[0, 1])
    rs = []
    for i in range(len(x)):
        m = np.ones(len(x), bool)
        m[i] = False
        rs.append(float(np.corrcoef(x[m], y[m])[0, 1]))
    rs = np.array(rs)
    return dict(r=r_full, jk_min=float(rs.min()), jk_max=float(rs.max()),
                jk_mean=float(rs.mean()))


# ---------------------------------------------------------------------------
# mechanism (measurement 3), fully vectorised
# ---------------------------------------------------------------------------
def relabel(m):
    _, out = np.unique(np.asarray(m), return_inverse=True)
    return out.astype(np.int32)


def mechanism(g, edge_arr, deg, P, C, keep, direction, min_parent=20,
              piece_frac=0.10, max_rows=200):
    """Parents = communities of P broken into >=2 pieces of C (each >=piece_frac
    of the parent).  Compare DSpar removal of cross-piece vs intra-piece edges."""
    P, C = relabel(P), relabel(C)
    kP = int(P.max()) + 1
    sizeP = np.bincount(P, minlength=kP)
    M = coo_matrix((np.ones(P.size, np.int32), (P, C)),
                   shape=(kP, int(C.max()) + 1)).tocsr()

    sel = np.zeros(kP, bool)
    npieces = np.zeros(kP, np.int32)
    indptr, data = M.indptr, M.data
    for i in range(kP):
        if sizeP[i] < min_parent:
            continue
        row = data[indptr[i]:indptr[i + 1]]
        nb = int((row >= piece_frac * sizeP[i]).sum())
        npieces[i] = nb
        if nb >= 2:
            sel[i] = True

    e0, e1 = edge_arr[:, 0], edge_arr[:, 1]
    pu, pv = P[e0], P[e1]
    intra_parent = pu == pv
    same_child = C[e0] == C[e1]
    inpar = intra_parent & sel[pu]
    cross = inpar & ~same_child
    intra = inpar & same_child
    dprod = deg[e0] * deg[e1]

    glob_removal = float(1.0 - keep.mean())
    rem_cross = float(1.0 - keep[cross].mean()) if cross.any() else float("nan")
    rem_intra = float(1.0 - keep[intra].mean()) if intra.any() else float("nan")

    # per-parent breakdown
    pidx = pu.copy()
    n_cross = np.bincount(pidx[cross], minlength=kP)
    n_intra = np.bincount(pidx[intra], minlength=kP)
    rm_cross = np.bincount(pidx[cross], weights=(~keep[cross]).astype(float),
                           minlength=kP)
    rm_intra = np.bincount(pidx[intra], weights=(~keep[intra]).astype(float),
                           minlength=kP)
    dp_cross = np.bincount(pidx[cross], weights=dprod[cross], minlength=kP)
    dp_intra = np.bincount(pidx[intra], weights=dprod[intra], minlength=kP)

    ids = np.nonzero(sel)[0]
    rows, dpratios, biasratios = [], [], []
    for i in ids:
        if n_cross[i] == 0 or n_intra[i] == 0:
            dpr = float("nan"); bias = float("nan")
        else:
            dpr = float((dp_cross[i] / n_cross[i]) / (dp_intra[i] / n_intra[i]))
            rc = rm_cross[i] / n_cross[i]
            ri = rm_intra[i] / n_intra[i]
            bias = float(rc / ri) if ri > 0 else float("nan")
            dpratios.append(dpr)
            biasratios.append(bias)
        rows.append(dict(direction=direction, parent_comm=int(i),
                         size_parent=int(sizeP[i]), n_pieces=int(npieces[i]),
                         cross_edges=int(n_cross[i]), intra_edges=int(n_intra[i]),
                         cross_removed=int(rm_cross[i]),
                         intra_removed=int(rm_intra[i]),
                         cross_removal_rate=float(rm_cross[i] / n_cross[i])
                         if n_cross[i] else float("nan"),
                         intra_removal_rate=float(rm_intra[i] / n_intra[i])
                         if n_intra[i] else float("nan"),
                         removal_bias=bias, dprod_ratio=dpr))
    rows.sort(key=lambda r: -r["size_parent"])
    dpratios = np.array(dpratios, float)
    biasratios = np.array(biasratios, float)

    stats = dict(
        direction=direction, n_parents=int(sel.sum()),
        parent_nodes=int(sizeP[sel].sum()),
        parent_node_share=float(sizeP[sel].sum() / P.size),
        cross_edges=int(cross.sum()), intra_edges=int(intra.sum()),
        cross_removed=int((~keep[cross]).sum()), intra_removed=int((~keep[intra]).sum()),
        cross_removal_rate=rem_cross, intra_removal_rate=rem_intra,
        global_removal_rate=glob_removal,
        removal_bias_pooled=float(rem_cross / rem_intra) if rem_intra else float("nan"),
        removal_bias_vs_global=float(rem_cross / glob_removal) if glob_removal else float("nan"),
        dprod_cross_mean=float(dprod[cross].mean()) if cross.any() else float("nan"),
        dprod_intra_mean=float(dprod[intra].mean()) if intra.any() else float("nan"),
        dprod_ratio_pooled=float(dprod[cross].mean() / dprod[intra].mean())
        if cross.any() and intra.any() else float("nan"),
        dprod_ratio_median_per_parent=float(np.nanmedian(dpratios)) if dpratios.size else float("nan"),
        n_parents_dprod_ratio_gt1=int(np.nansum(dpratios > 1)),
        n_parents_dprod_evaluated=int(np.isfinite(dpratios).sum()),
        removal_bias_median_per_parent=float(np.nanmedian(biasratios)) if biasratios.size else float("nan"),
        n_parents_bias_gt1=int(np.nansum(biasratios > 1)),
        n_parents_no_cross_edge_removed=int(sum(
            1 for r in rows if r["cross_edges"] > 0 and r["cross_removed"] == 0)),
    )
    return stats, rows[:max_rows]


# ---------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    runs_csv = HERE / "runs.csv"
    print(f"[load] {EDGE_FILE}", flush=True)
    t0 = time.perf_counter()
    g = load_lcc_graph(EDGE_FILE)
    n, m = g.vcount(), g.ecount()
    deg = np.asarray(g.degree(), dtype=np.float64)
    print(f"  n={n:,} m={m:,} loaded in {time.perf_counter()-t0:.1f}s "
          f"rss={rss():.2f}GB", flush=True)

    plain, seeded, memb_plain, memb_seeded, keep_seeded, edge_arr = \
        stage1(g, runs_csv)

    Qp = np.array([r["Q"] for r in plain])
    kp = np.array([r["k"] for r in plain], float)
    Tp = np.array([r["T"] for r in plain])
    Qs = np.array([r["Q"] for r in seeded])
    ks = np.array([r["k"] for r in seeded], float)
    Ts = np.array([r["T"] for r in seeded])
    Qraw = np.array([r["Q_raw"] for r in seeded])

    # cache what the mechanism stage needs, so a crash here costs no compute
    best_p = int(np.argmax(Qp)); best_s = int(np.argmax(Qs))
    np.savez_compressed(HERE / "cache.npz",
                        best_plain=memb_plain[plain[best_p]["seed"]],
                        best_seeded=memb_seeded[seeded[best_s]["seed"]],
                        keep=keep_seeded[seeded[best_s]["spar_seed"]],
                        Qp=Qp, kp=kp, Tp=Tp, Qs=Qs, ks=ks, Ts=Ts)
    print(f"[cache] written, rss={rss():.2f}GB", flush=True)

    # -------- supplementary gamma-matched arm ------------------------------
    print("\n[stage2] resolution-matched plain arm", flush=True)
    gstar, gprobes, grows = stage2_gamma(g, runs_csv, float(ks.mean()),
                                         float(kp.mean()))
    Qg = np.array([r["Q"] for r in grows])
    kg = np.array([r["k"] for r in grows], float)

    # -------- measurement 1: statistical reality ---------------------------
    print("\n[m1] statistical reality", flush=True)
    gap = float(Qs.mean() - Qp.mean())
    mw = sstats.mannwhitneyu(Qs, Qp, alternative="greater")
    perm_p, perm_n = exact_perm_pvalue(Qp, Qs)
    ci_lo, ci_hi, ci_mean = boot_ci_gap(Qp, Qs)
    bo2 = best_of_k_exact(list(Qp), 2)
    ratio = float(Ts.mean() / Tp.mean())
    n_matched = max(1, int(round(ratio)))
    bo_matched = boot_best_of_n(Qp, n_matched)
    # bootstrap SE of E[best-of-2] (exp_H's sharper test)
    samp = Qp[RNG.randint(0, Qp.size, size=(2000, Qp.size))]
    ebo2 = np.array([best_of_k_exact(list(row), 2).mean() for row in samp])
    se_bo2 = float(ebo2.std(ddof=1))
    se_seeded = float(Qs.std(ddof=1) / np.sqrt(Qs.size))
    m1 = dict(
        n_plain=int(Qp.size), n_seeded=int(Qs.size),
        Q_plain_mean=float(Qp.mean()), Q_plain_std=float(Qp.std(ddof=1)),
        Q_plain_min=float(Qp.min()), Q_plain_max=float(Qp.max()),
        Q_seeded_mean=float(Qs.mean()), Q_seeded_std=float(Qs.std(ddof=1)),
        Q_seeded_min=float(Qs.min()), Q_seeded_max=float(Qs.max()),
        Q_sparse_transfer_mean=float(Qraw.mean()),
        gap_vs_plain_mean=gap,
        z_vs_plain_single_run_sd=float(gap / Qp.std(ddof=1)),
        mannwhitney_U=float(mw.statistic), mannwhitney_p=float(mw.pvalue),
        exact_perm_p=perm_p, exact_perm_n_splits=perm_n,
        boot_ci95_gap=[ci_lo, ci_hi], boot_gap_mean=ci_mean,
        seeded_mean_percentile_in_plain=float((Qp < Qs.mean()).mean() * 100),
        n_plain_beating_seeded_mean=int((Qp > Qs.mean()).sum()),
        n_plain_beating_seeded_best=int((Qp > Qs.max()).sum()),
        n_plain_beating_seeded_worst=int((Qp > Qs.min()).sum()),
        head_to_head_win_rate_vs_bo2=float(np.mean(
            [(q > bo2).mean() for q in Qs])),
        E_best_of_2=float(bo2.mean()), best_of_2_std=float(bo2.std(ddof=1)),
        gain_vs_E_best_of_2=float(Qs.mean() - bo2.mean()),
        p_bo2_ge_seeded_mean=float((bo2 >= Qs.mean()).mean()),
        se_E_best_of_2=se_bo2, se_seeded_mean=se_seeded,
        z_mean_vs_bo2=float((Qs.mean() - bo2.mean())
                            / np.sqrt(se_bo2 ** 2 + se_seeded ** 2)),
    )
    print(json.dumps(m1, indent=1), flush=True)

    # -------- measurement 2: granularity -----------------------------------
    print("\n[m2] granularity", flush=True)
    jk = jackknife_corr(kp, Qp)
    slope, intercept = np.polyfit(kp, Qp, 1)
    pred_at_seeded_k = float(slope * ks.mean() + intercept)
    lo_k, hi_k = float(ks.min()), float(ks.max())
    in_range = (kp >= lo_k) & (kp <= hi_k)
    # widened fallbacks
    near5 = np.argsort(np.abs(kp - ks.mean()))[:5]
    m2 = dict(
        corr_k_Q_plain=jk, ols_slope=float(slope), ols_intercept=float(intercept),
        mean_k_plain=float(kp.mean()), std_k_plain=float(kp.std(ddof=1)),
        min_k_plain=float(kp.min()), max_k_plain=float(kp.max()),
        mean_k_seeded=float(ks.mean()), std_k_seeded=float(ks.std(ddof=1)),
        min_k_seeded=lo_k, max_k_seeded=hi_k,
        seeded_k_below_all_plain=bool(hi_k < kp.min()),
        ols_pred_Q_at_seeded_k=pred_at_seeded_k,
        ols_residual_of_seeded_mean=float(Qs.mean() - pred_at_seeded_k),
        # registered k-matched control
        n_plain_in_seeded_k_range=int(in_range.sum()),
        Q_plain_k_matched_mean=float(Qp[in_range].mean()) if in_range.any() else None,
        k_matched_gap=float(Qs.mean() - Qp[in_range].mean()) if in_range.any() else None,
        # fallback: 5 nearest-k plain restarts
        nearest5_k=[float(x) for x in kp[near5]],
        nearest5_Q_mean=float(Qp[near5].mean()),
        nearest5_gap=float(Qs.mean() - Qp[near5].mean()),
        # supplementary resolution-matched arm
        gamma_star=float(gstar), gamma_probes=gprobes,
        Q_gamma_mean=float(Qg.mean()), Q_gamma_std=float(Qg.std(ddof=1)),
        Q_gamma_best=float(Qg.max()), mean_k_gamma=float(kg.mean()),
        gamma_matched_gap=float(Qs.mean() - Qg.mean()),
        gamma_matched_gap_vs_best=float(Qs.mean() - Qg.max()),
        # shape (exp_N third check)
        shape_plain=dict(
            frac_largest_mean=float(np.mean([r["frac_largest"] for r in plain])),
            largest_mean=float(np.mean([r["largest"] for r in plain])),
            n_comm_gt_1pct_mean=float(np.mean([r["n_comm_gt_1pct"] for r in plain])),
            top5_share_mean=float(np.mean([r["top5_share"] for r in plain])),
            n_comm_ge20_mean=float(np.mean([r["n_comm_ge20"] for r in plain]))),
        shape_seeded=dict(
            frac_largest_mean=float(np.mean([r["frac_largest"] for r in seeded])),
            largest_mean=float(np.mean([r["largest"] for r in seeded])),
            n_comm_gt_1pct_mean=float(np.mean([r["n_comm_gt_1pct"] for r in seeded])),
            top5_share_mean=float(np.mean([r["top5_share"] for r in seeded])),
            n_comm_ge20_mean=float(np.mean([r["n_comm_ge20"] for r in seeded]))),
        shape_best_plain=dict((k2, plain[best_p][k2]) for k2 in
                              ("k", "largest", "frac_largest", "n_comm_gt_1pct",
                               "top5_share", "n_comm_ge20")),
        shape_best_seeded=dict((k2, seeded[best_s][k2]) for k2 in
                               ("k", "largest", "frac_largest", "n_comm_gt_1pct",
                                "top5_share", "n_comm_ge20")),
    )
    print(json.dumps(m2, indent=1, default=float), flush=True)

    # -------- measurement 3: mechanism -------------------------------------
    print("\n[m3] mechanism", flush=True)
    B = memb_plain[plain[best_p]["seed"]]
    S = memb_seeded[seeded[best_s]["seed"]]
    keep = keep_seeded[seeded[best_s]["spar_seed"]]
    st_split, rows_split = mechanism(g, edge_arr, deg, B, S, keep,
                                     "base_split_by_seeded")
    st_merge, rows_merge = mechanism(g, edge_arr, deg, S, B, keep,
                                     "seeded_merges_base")
    allrows = rows_split + rows_merge
    if allrows:
        with open(HERE / "mechanism_parents.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0]))
            w.writeheader()
            w.writerows(allrows)
    m3 = dict(base_split_by_seeded=st_split, seeded_merges_base=st_merge,
              best_plain_seed=plain[best_p]["seed"],
              best_seeded_seed=seeded[best_s]["seed"],
              best_spar_seed=seeded[best_s]["spar_seed"],
              Q_best_plain=float(Qp.max()), Q_best_seeded=float(Qs.max()),
              dQ_best_vs_best=float(Qs.max() - Qp.max()))
    print(json.dumps(m3, indent=1, default=float), flush=True)

    # -------- measurement 4: compute ---------------------------------------
    print("\n[m4] compute", flush=True)
    exact_bo = {}
    for kk in (2, 3, 4, 5):
        if kk <= Qp.size:
            v = best_of_k_exact(list(Qp), kk)
            exact_bo[kk] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)),
                                p_ge_seeded_mean=float((v >= Qs.mean()).mean()))
    m4 = dict(
        T_plain_mean=float(Tp.mean()), T_plain_median=float(np.median(Tp)),
        T_pipe_mean=float(Ts.mean()),
        T_pipe_sparsify=float(np.mean([r["T_sparsify"] for r in seeded])),
        T_pipe_leiden_sparse=float(np.mean([r["T_leiden_sparse"] for r in seeded])),
        T_pipe_refine=float(np.mean([r["T_refine"] for r in seeded])),
        pipe_over_restart=ratio, n_matched_restarts=n_matched,
        E_best_of_matched=float(bo_matched.mean()),
        gain_vs_E_best_of_matched=float(Qs.mean() - bo_matched.mean()),
        exact_best_of_k=exact_bo,
        boot_E_best_of_20=float(boot_best_of_n(Qp, 20).mean()),
        cost_20_restarts_over_1_pipe=float(20 * Tp.mean() / Ts.mean()),
        note="wall clock is LOAD-CONTAMINATED (shared machine); arms were "
             "interleaved so ratios remain valid, absolute seconds do not. "
             "All compute statements are IN EXPECTATION (exp_N caveat C2).",
    )
    print(json.dumps(m4, indent=1, default=float), flush=True)

    # -------- landscape probe (AMI, budgeted -- supplementary, not registered)
    print("\n[m5] landscape probe (AMI, 900s budget)", flush=True)
    t_ami = time.perf_counter()
    BUDGET = 900.0

    def budgeted(pairlist):
        out = []
        for a, b in pairlist:
            if time.perf_counter() - t_ami > BUDGET:
                break
            out.append(float(adjusted_mutual_info_score(a, b)))
        return out

    ami_ps = budgeted([(memb_plain[r["seed"]], S) for r in plain])
    ami_pp = budgeted([(memb_plain[plain[i]["seed"]], memb_plain[plain[j]["seed"]])
                       for i, j in itertools.combinations(range(len(plain)), 2)][:10])
    ami_ss = budgeted([(memb_seeded[a["seed"]], memb_seeded[b["seed"]])
                       for a, b in itertools.combinations(seeded, 2)])

    def d(v):
        return (dict(n=len(v), mean=float(np.mean(v)), max=float(np.max(v)),
                     min=float(np.min(v))) if v else dict(n=0))

    m5 = dict(ami_plain_to_best_seeded=d(ami_ps), ami_plain_plain=d(ami_pp),
              ami_seeded_seeded=d(ami_ss),
              closest_plain_Q=float(Qp[int(np.argmax(ami_ps))]) if ami_ps else None,
              gap_best_seeded_minus_best_plain=float(Qs.max() - Qp.max()),
              ami_seconds=float(time.perf_counter() - t_ami))
    print(json.dumps(m5, indent=1, default=float), flush=True)

    # -------- verdicts ------------------------------------------------------
    kmg = m2["k_matched_gap"]
    verdict = dict(
        P1_reproduces=bool(0.002 <= gap <= 0.005 and m1["exact_perm_p"] < 0.05),
        P1_gap=gap, P1_p=m1["exact_perm_p"],
        P2_corr_ok=bool(abs(jk["r"]) < 0.3),
        P2_kmatched_positive=(bool(kmg > 0) if kmg is not None else None),
        P2_corr_r=jk["r"],
        P3_boundary_bias=st_split["removal_bias_pooled"],
        P3_ok=bool(st_split["removal_bias_pooled"] > 1.5)
        if np.isfinite(st_split["removal_bias_pooled"]) else False,
        KILL_kmatched_le_0=(bool(kmg <= 0) if kmg is not None else "no_k_matched_plain_restarts"),
        KILL_granularity=bool(jk["r"] < -0.5 and ks.mean() < kp.mean()),
        KILL_nonreproducible=bool(m1["exact_perm_p"] > 0.10),
    )
    print("\n[verdict]", json.dumps(verdict, indent=1, default=float), flush=True)

    res = dict(experiment="exp_X_youtube_anatomy",
               graph=dict(network=NETWORK, n=n, m=m),
               config=dict(sampler=SAMPLER, alpha=ALPHA, n_iterations=N_ITER,
                           plain_seeds=PLAIN_SEEDS, spar_seeds=SPAR_SEEDS,
                           leiden_seeds=LEID_SEEDS,
                           exp_H_cell="calibrated alpha=0.95 (only Bonferroni "
                                      "survivor of exp_H's 8 cells)"),
               m1_statistical=m1, m2_granularity=m2, m3_mechanism=m3,
               m4_compute=m4, m5_landscape=m5, verdict=verdict,
               peak_rss_gb=rss(), elapsed_sec=time.perf_counter() - t_start)
    with open(HERE / "results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nDone in {res['elapsed_sec']:.1f}s, peak RSS {rss():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
