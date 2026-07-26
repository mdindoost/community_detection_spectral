#!/usr/bin/env python
"""
Experiment AC -- Does the Exp AA transition depend on degree heterogeneity?

Pre-registered in DESIGN.md (2026-07-26).  This file MIRRORS
exp_AA_satuluri_regime/run.py Arm A and Arm B with the generator swapped: the
sparsifiers, retention targets, detectors, metrics, controls and seed counts are
literally IMPORTED from exp_AA's run.py so the two experiments are comparable
cell for cell.  Only the graph generator differs.

  Generator 1 "sbm"    -- nx.stochastic_block_model, n=1e4, 100 equal blocks of
                          100 nodes (matches exp_AA's mu=0.5 community-count
                          regime: ~100 planted communities, mean size 100).
                          Poisson degrees => degree CV ~ 1/sqrt(d).
  Generator 2 "dcsbm"  -- same block structure, per-node power-law degree
                          propensity theta (exponent 2.1, matching exp_AA's
                          LFR tau1), normalised to mean 1 WITHIN each block, so
                          P(i~j) = min(1, theta_i*theta_j*p_block).  Degree
                          heterogeneity restored, block model retained.

Both generators are calibrated to hit exp_AA's REALIZED average degree and
REALIZED mixing at mu_nominal=0.5, per degree cell, so the sweep axes line up:

    d_nom      10      25      50     100     200
    d_avg    11.18   24.63   49.42  101.93  220.93   (exp_AA mu=0.5 realized)
    mu       0.5785  0.5976  0.6129  0.6415  0.6719  (exp_AA mu=0.5 realized)

Realized average degree, realized mixing and realized degree CV are MEASURED and
written into every row; nothing is assumed.

Arms
  A  Leiden (free granularity), honest-transfer modularity on the ORIGINAL
     graph, best-of-5 restart baseline + runtime-matched baseline, AMI/ARI vs
     planted labels with a size-matched chance floor, resolution-matched control.
  B  real Metis (pymetis) with k pinned to the planted block count.
  M  Metis option-seed robustness (mirrors exp_AA/metis_noise.py, 10 seeds).

Usage:  run.py gen
        run.py armA <sbm|dcsbm> [d_nom ...]
        run.py armB <sbm|dcsbm> [d_nom ...]
        run.py metisseeds <sbm|dcsbm>
"""

import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
AA = HERE.parent / "exp_AA_satuluri_regime"

# ---- verbatim reuse of exp_AA machinery (sparsifiers, detectors, metrics,
# ---- controls, seed lists, retention targets).  Loaded by path because both
# ---- files are called run.py.
import importlib.util                                # noqa: E402
_spec = importlib.util.spec_from_file_location("aa_run", AA / "run.py")
aa = importlib.util.module_from_spec(_spec)
sys.modules["aa_run"] = aa
_spec.loader.exec_module(aa)

N_NODES = aa.N_NODES
BASE_SEEDS, SPARSE_SEEDS = aa.BASE_SEEDS, aa.SPARSE_SEEDS
TARGETS, SPARSIFIERS = aa.TARGETS, aa.SPARSIFIERS
edge_jaccard, LSpar, dspar_scores = aa.edge_jaccard, aa.LSpar, aa.dspar_scores
sparsify_edges, subgraph_from_eids = aa.sparsify_edges, aa.subgraph_from_eids
leiden, metis_partition, evaluate = aa.leiden, aa.metis_partition, aa.evaluate
Ladder, ResGrid = aa.Ladder, aa.ResGrid
graph_from_E, graph_meta = aa.graph_from_E, aa.graph_meta
append_row, done_keys, log = aa.append_row, aa.done_keys, aa.log

CACHE = Path(os.environ.get("AC_CACHE", "/tmp/ac_sbm_cache"))
CACHE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

K_BLOCKS = 100                       # planted block count (equal-block arms)
BLOCK = N_NODES // K_BLOCKS          # 100 nodes per block
DEGS = [10, 25, 50, 100, 200]
GRAPH_SEEDS = [1, 2, 3]

# The four generators form a 2x2 factorial in (degree heterogeneity) x
# (community-size heterogeneity).  The first two are the ones pre-registered in
# DESIGN.md; the "_ps" pair is a DISCLOSED ADDITION (see SUMMARY caveats): the
# equal-block SBM hands Metis a perfectly correct balance prior, so an equal-
# block negative result cannot by itself separate "degree heterogeneity" from
# "community-size heterogeneity".
#   sbm       homogeneous degree, equal blocks          (DESIGN generator 1)
#   dcsbm     power-law degree,   equal blocks          (DESIGN generator 2)
#   sbm_ps    homogeneous degree, power-law block sizes (added control)
#   dcsbm_ps  power-law degree,   power-law block sizes (added control; the
#             cell closest to LFR on both heterogeneity axes)
GENERATORS = ["sbm", "dcsbm", "sbm_ps", "dcsbm_ps"]
HETDEG = {"sbm": 0, "dcsbm": 1, "sbm_ps": 0, "dcsbm_ps": 1}
HETSIZE = {"sbm": 0, "dcsbm": 0, "sbm_ps": 1, "dcsbm_ps": 1}

# exp_AA realized values at mu_nominal = 0.5 (mean over its 3 LFR seeds),
# from exp_AA_satuluri_regime/lfr_generation.csv
D_TARGET = {10: 11.183, 25: 24.635, 50: 49.423, 100: 101.930, 200: 220.927}
MU_TARGET = {10: 0.57853, 25: 0.59760, 50: 0.61286, 100: 0.64153, 200: 0.67193}
LFR_DEG_CV = {10: 2.327, 25: 1.765, 50: 1.367, 100: 0.973, 200: 0.734}

DC_ALPHA = 2.1        # power-law exponent of the degree propensity (= LFR tau1)
DC_TRUNC = 100.0      # theta support [1, 100] before per-block normalisation
SIZE_TAU = 1.5        # power-law exponent of block sizes in the *_ps arms
                      # (= LFR tau2 in exp_AA)
# Community-size box per degree cell, mirroring exp_AA's REALIZED LFR community
# sizes at mu=0.5 (lfr_generation.csv: [20,443..492] up to d_nom=100,
# [40..42, 565..583] at d_nom=200 where the generator was forced to widen it).
SIZE_BOX = {10: (20, 500), 25: (20, 500), 50: (20, 500),
            100: (20, 500), 200: (40, 580)}

CAL_ITERS = 8
CAL_TOL = 0.02        # relative tolerance on intra- and inter-edge counts


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _edge_split(E, y):
    intra = y[E[:, 0]] == y[E[:, 1]]
    return int(intra.sum()), int((~intra).sum())


def block_sizes(gen, d_nom, seed):
    """Equal blocks, or LFR-style power-law block sizes summing to exactly n."""
    if not HETSIZE[gen]:
        return np.full(K_BLOCKS, BLOCK, dtype=np.int64)
    lo, hi = SIZE_BOX[d_nom]
    rs = np.random.RandomState(int(seed) + 31_000)
    a = 1.0 - SIZE_TAU
    sizes, tot = [], 0
    while tot < N_NODES:
        u = rs.random_sample()
        x = (lo ** a + u * (hi ** a - lo ** a)) ** (1.0 / a)
        s = int(round(x))
        if tot + s > N_NODES:
            s = N_NODES - tot
            if s < lo:                       # remainder too small for a block
                sizes[rs.randint(len(sizes))] += s
                tot += s
                break
        sizes.append(s)
        tot += s
    return np.asarray(sizes, dtype=np.int64)


def labels_from_sizes(sizes):
    return np.repeat(np.arange(len(sizes)), sizes).astype(np.int64)


def draw_theta(gen, sizes, seed):
    """Power-law propensity, exponent DC_ALPHA on [1, DC_TRUNC], normalised to
    mean 1 WITHIN each block.  Homogeneous arms get theta == 1."""
    if not HETDEG[gen]:
        return np.ones(N_NODES)
    rs = np.random.RandomState(int(seed) + 7_000)
    u = rs.random_sample(N_NODES)
    a = 1.0 - DC_ALPHA
    th = (1.0 + u * (DC_TRUNC ** a - 1.0)) ** (1.0 / a)
    out = np.empty(N_NODES)
    off = 0
    for nb in sizes:
        blk = th[off:off + nb]
        out[off:off + nb] = blk * (nb / blk.sum())
        off += nb
    return out


def base_probs(sizes, d, mu):
    """p_in per block and a single p_out such that EVERY node has expected
    intra-degree (1-mu)*d and expected inter-degree mu*d, independent of its
    block size (this is how LFR defines mu -- per node, not per block)."""
    p_in = (1.0 - mu) * d / np.maximum(sizes - 1, 1)
    p_out = mu * d / max(1.0, N_NODES - float(sizes.mean()))
    return p_in, float(p_out)


def sample_nx_sbm(sizes, p_in, p_out, seed):
    """DESIGN generator 1 path: networkx stochastic_block_model."""
    K = len(sizes)
    P = np.full((K, K), min(p_out, 1.0))
    np.fill_diagonal(P, np.minimum(p_in, 1.0))
    G = nx.stochastic_block_model([int(s) for s in sizes], P.tolist(),
                                  seed=int(seed), sparse=True, selfloops=False)
    return np.asarray(list(G.edges()), dtype=np.int64)


def sample_dcsbm(y, theta, p_in, p_out, seed, chunk=500):
    """Degree-corrected SBM: P(i~j) = min(1, theta_i theta_j p_block)."""
    rs = np.random.RandomState(int(seed))
    pin_node = p_in[y]
    rows, cols = [], []
    idx = np.arange(N_NODES)
    for s in range(0, N_NODES, chunk):
        t = min(s + chunk, N_NODES)
        O = np.outer(theta[s:t], theta)
        P = O * p_out
        same = y[s:t][:, None] == y[None, :]
        P[same] = (O * pin_node[s:t][:, None])[same]
        np.minimum(P, 1.0, out=P)
        upper = idx[None, :] > np.arange(s, t)[:, None]
        hit = (rs.random_sample(P.shape) < P) & upper
        r, c = np.nonzero(hit)
        rows.append(r + s)
        cols.append(c)
        del O, P, same, upper, hit
    return np.column_stack([np.concatenate(rows),
                            np.concatenate(cols)]).astype(np.int64)


def _calibrate(gen, d_nom, seed):
    """Hit the target intra/inter edge counts by empirical rescaling of p_in and
    p_out.  Every realized quantity is MEASURED, never assumed; the number of
    calibration passes and the residual relative error are recorded."""
    d, mu = D_TARGET[d_nom], MU_TARGET[d_nom]
    m_target = d * N_NODES / 2.0
    tgt_in, tgt_out = (1.0 - mu) * m_target, mu * m_target

    sizes = block_sizes(gen, d_nom, seed)
    y = labels_from_sizes(sizes)
    theta = draw_theta(gen, sizes, seed)
    p_in, p_out = base_probs(sizes, d, mu)

    best = None
    for it in range(CAL_ITERS):
        if HETDEG[gen]:
            E = sample_dcsbm(y, theta, p_in, p_out, seed + 100 * it)
        else:
            E = sample_nx_sbm(sizes, p_in, p_out, seed + 100 * it)
        m_in, m_out = _edge_split(E, y)
        err = max(abs(m_in - tgt_in) / tgt_in, abs(m_out - tgt_out) / tgt_out)
        if best is None or err < best[0]:
            best = (err, E, it + 1, p_in.copy(), p_out)
        if err <= CAL_TOL:
            break
        p_in = p_in * float(np.clip(tgt_in / max(m_in, 1), 0.25, 8.0))
        p_out = p_out * float(np.clip(tgt_out / max(m_out, 1), 0.25, 8.0))
        if p_in.max() > 1e4 or p_out > 1e4:
            break
    err, E, n_pass, p_in_used, p_out_used = best
    meta = dict(gen=gen, d_nom=d_nom, graph_seed=seed,
                het_deg=HETDEG[gen], het_size=HETSIZE[gen],
                d_target=d, mu_target=mu,
                p_in_mean=float(np.mean(p_in_used)),
                p_in_max=float(np.max(p_in_used)), p_out=p_out_used,
                n_blocks=int(len(sizes)),
                cal_passes=n_pass, cal_relerr=round(err, 5),
                cal_status="ok" if err <= CAL_TOL else "off_target")
    return E, y, meta


def gen_graph(gen, d_nom, seed):
    """Return (E, y, meta), cached."""
    f = CACHE / f"{gen}_d{d_nom}_s{seed}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        meta = {k: z[k].item() for k in z.files if k not in ("E", "y")}
        return z["E"], z["y"], meta
    t0 = time.perf_counter()
    E, y, meta = _calibrate(gen, d_nom, seed)
    meta["gen_time"] = time.perf_counter() - t0
    np.savez_compressed(f, E=E.astype(np.int32), y=y.astype(np.int32), **meta)
    return E, y, meta


def run_gen():
    for gen, d, s in itertools.product(GENERATORS, DEGS, GRAPH_SEEDS):
        E, y, meta = gen_graph(gen, d, s)
        g = graph_from_E(E)
        gm = graph_meta(g, y)
        log(f"{gen:9s} d_nom={d:<4} s={s} m={gm['m']:>9,} "
            f"d_avg={gm['d_avg_real']:7.2f} (target {meta['d_target']:.2f})  "
            f"mu={gm['mu_real']:.4f} (target {meta['mu_target']:.4f})  "
            f"deg_cv={gm['deg_cv']:.3f} (LFR {LFR_DEG_CV[d]:.3f})  "
            f"deg=[{gm['deg_min']},{gm['deg_max']}] k={gm['n_comm_planted']} "
            f"cs=[{gm['comm_size_min']},{gm['comm_size_max']}] "
            f"cal={meta['cal_passes']}p/{meta['cal_status']} "
            f"t={meta['gen_time']:.1f}s")
        append_row(HERE / "sbm_generation.csv",
                   dict(**meta, lfr_deg_cv=LFR_DEG_CV[d], **gm))
        del g


# ---------------------------------------------------------------------------
# ARM A -- Leiden, mirrors exp_AA.run_armA
# ---------------------------------------------------------------------------

ARMA_KEY = ["gen", "d_nom", "graph_seed", "sparsifier", "target_ret"]


def run_armA(gen, cells=None):
    out = HERE / f"results_armA_{gen}.csv"
    seen = done_keys(out, ARMA_KEY)
    rng = np.random.default_rng(7)
    combos = [(d, s) for d in DEGS for s in GRAPH_SEEDS]
    if cells:
        combos = [c for c in combos if str(c[0]) in set(cells)]

    for d, gs_seed in combos:
        tag = f"{gen}_d{d}_s{gs_seed}"
        if all((gen, str(d), str(gs_seed), sp_, str(t)) in seen
               for sp_ in SPARSIFIERS for t in TARGETS):
            log(f"[skip] {tag}")
            continue
        E, y, meta = gen_graph(gen, d, gs_seed)
        g = graph_from_E(E)
        gm = graph_meta(g, y)
        log(f"\n{'='*76}\n[A] {tag}  n={gm['n']} m={gm['m']:,} "
            f"d_avg={gm['d_avg_real']:.2f} mu_real={gm['mu_real']:.4f} "
            f"deg_cv={gm['deg_cv']:.3f} k_planted={gm['n_comm_planted']}\n{'='*76}")

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
        log(f"  jaccard {T_jac:.2f}s rank {T_rank:.2f}s lspar_floor={min_ret:.4f}")
        tg = time.perf_counter()
        resgrid = ResGrid(g, y)
        log(f"  resolution grid: nc range "
            f"{min(r['nc'] for r in resgrid.rows)}..{max(r['nc'] for r in resgrid.rows)} "
            f"({time.perf_counter()-tg:.1f}s)")

        for kind, target in itertools.product(SPARSIFIERS, TARGETS):
            key = (gen, str(d), str(gs_seed), kind, str(target))
            if key in seen:
                continue
            eids, realized, knob, status, T_sel = sparsify_edges(
                g, kind, target, 500 + gs_seed, lspar=lsp, dscores=ds)
            T_sparsify = T_sel + (T_jac + T_rank if kind == "lspar" else 0.0)
            gsp = subgraph_from_eids(g, eids)

            evs, Ts = [], []
            for s in SPARSE_SEEDS:
                memb, dt = leiden(gsp, s)
                Ts.append(dt)
                evs.append(evaluate(g, memb, y, rng=rng))
            T_det = float(np.mean(Ts))
            agg = {k: float(np.mean([e[k] for e in evs])) for k in evs[0]}
            agg_sd = {k: float(np.std([e[k] for e in evs])) for k in ("Q_orig", "AMI")}
            budget = T_sparsify + T_det
            mr, nrest = ladder.matched(budget)
            rm = resgrid.nearest(agg["nc"])

            row = dict(
                arm="A", gen=gen, detector="leiden", d_nom=d, graph_seed=gs_seed,
                het_deg=meta["het_deg"], het_size=meta["het_size"],
                d_target=meta["d_target"], mu_target=meta["mu_target"],
                p_in_mean=meta["p_in_mean"], p_out=meta["p_out"],
                cal_passes=meta["cal_passes"], cal_status=meta["cal_status"],
                lfr_deg_cv=LFR_DEG_CV[d], **gm,
                sparsifier=kind, target_ret=target, realized_ret=round(realized, 5),
                knob=round(float(knob), 6), status=status,
                lspar_floor=round(min_ret, 5), m_sparse=gsp.ecount(),
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
                Q_resmatch=rm["Q_orig"], AMI_resmatch=rm["AMI"],
                ARI_resmatch=rm["ARI"], nc_resmatch=rm["nc"],
                gamma_resmatch=rm["gamma"])
            append_row(out, row)
            log(f"  {kind:7s} t={target:<5} ret={realized:.4f} ({status}) "
                f"dQ_match={row['dQ_vs_matched']:+.5f} "
                f"dQ_best={row['dQ_vs_base_best']:+.5f} "
                f"AMI={agg['AMI']:.4f} dAMI={row['dAMI_vs_base_mean']:+.4f} "
                f"dAMI_res={row['dAMI_vs_matched']:+.4f} nc={agg['nc']:.0f}")
            del gsp
        del g, lsp, Ej, J


# ---------------------------------------------------------------------------
# ARM B -- real Metis, k pinned to the planted block count
# ---------------------------------------------------------------------------

ARMB_KEY = ["gen", "d_nom", "graph_seed", "detector", "sparsifier", "target_ret"]


def run_armB(gen, cells=None, detectors=("metis",)):
    out = HERE / f"results_armB_{gen}.csv"
    seen = done_keys(out, ARMB_KEY)
    rng = np.random.default_rng(11)
    combos = [(d, s) for d in DEGS for s in GRAPH_SEEDS]
    if cells:
        combos = [c for c in combos if str(c[0]) in set(cells)]

    for d, gs_seed in combos:
        tag = f"{gen}_d{d}_s{gs_seed}"
        E, y, meta = gen_graph(gen, d, gs_seed)
        g = graph_from_E(E)
        gm = graph_meta(g, y)
        k_fixed = gm["n_comm_planted"]
        log(f"\n{'='*76}\n[B] {tag} m={gm['m']:,} d_avg={gm['d_avg_real']:.2f} "
            f"mu_real={gm['mu_real']:.4f} deg_cv={gm['deg_cv']:.3f} "
            f"k_fixed={k_fixed}\n{'='*76}")

        tj = time.perf_counter()
        Ej, J, deg = edge_jaccard(g)
        T_jac = time.perf_counter() - tj
        lsp = LSpar(g, J=J, E=Ej, deg=deg)
        T_rank = time.perf_counter() - tj - T_jac
        ds = dspar_scores(g)

        for det in detectors:
            fn = {"metis": metis_partition}[det]
            t0 = time.perf_counter()
            mb, T_base = fn(g, k_fixed, seed=0)
            eb = evaluate(g, mb, y, rng=rng)
            log(f"  [{det}] baseline Q={eb['Q_orig']:.5f} AMI={eb['AMI']:.4f} "
                f"ARI={eb['ARI']:.4f} nc={eb['nc']} cv={eb['cv_size']:.3f} "
                f"T={T_base:.1f}s")

            for kind, target in itertools.product(SPARSIFIERS, TARGETS):
                key = (gen, str(d), str(gs_seed), det, kind, str(target))
                if key in seen:
                    continue
                eids, realized, knob, status, T_sel = sparsify_edges(
                    g, kind, target, 500 + gs_seed, lspar=lsp, dscores=ds)
                T_sparsify = T_sel + (T_jac + T_rank if kind == "lspar" else 0.0)
                gsp = subgraph_from_eids(g, eids)
                memb, T_det = fn(gsp, k_fixed, seed=0)
                ev = evaluate(g, memb, y, rng=rng)
                row = dict(
                    arm="B", gen=gen, detector=det, is_proxy=0,
                    d_nom=d, graph_seed=gs_seed,
                    het_deg=meta["het_deg"], het_size=meta["het_size"],
                    d_target=meta["d_target"], mu_target=meta["mu_target"],
                    cal_status=meta["cal_status"], lfr_deg_cv=LFR_DEG_CV[d], **gm,
                    k_fixed=k_fixed, sparsifier=kind, target_ret=target,
                    realized_ret=round(realized, 5), knob=round(float(knob), 6),
                    status=status, m_sparse=gsp.ecount(),
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
                    f"dQ={row['dQ']:+.5f} dAMI={row['dAMI']:+.4f} "
                    f"dARI={row['dARI']:+.4f} "
                    f"cv {eb['cv_size']:.2f}->{ev['cv_size']:.2f}")
                del gsp
        del g, lsp, Ej, J


# ---------------------------------------------------------------------------
# Metis option-seed robustness (mirrors exp_AA/metis_noise.py)
# ---------------------------------------------------------------------------

MSEEDS = list(range(10))
MTARGETS = [0.5, 0.2, 0.15]
MKEY = ["gen", "d_nom", "graph_seed", "sparsifier", "target_ret"]


def run_metisseeds(gen):
    out = HERE / "metis_seeds.csv"
    seen = done_keys(out, MKEY)
    for d, gs_seed in itertools.product(DEGS, GRAPH_SEEDS):
        E, y, meta = gen_graph(gen, d, gs_seed)
        g = graph_from_E(E)
        gm = graph_meta(g, y)
        k = gm["n_comm_planted"]
        log(f"\n=== [M] {gen} d{d} s{gs_seed} m={gm['m']:,} "
            f"d_avg={gm['d_avg_real']:.2f} deg_cv={gm['deg_cv']:.3f} k={k}")
        base = [evaluate(g, metis_partition(g, k, seed=s)[0], y) for s in MSEEDS]
        Qb = np.array([e["Q_orig"] for e in base])
        Ab = np.array([e["AMI"] for e in base])
        log(f"  ORIGINAL Q={Qb.mean():.5f}+-{Qb.std():.5f} "
            f"AMI={Ab.mean():.4f}+-{Ab.std():.4f}")
        Ej, J, deg = edge_jaccard(g)
        lsp = LSpar(g, J=J, E=Ej, deg=deg)
        ds = dspar_scores(g)
        for kind, t in itertools.product(SPARSIFIERS, MTARGETS):
            if (gen, str(d), str(gs_seed), kind, str(t)) in seen:
                continue
            eids, realized, knob, status, _ = sparsify_edges(
                g, kind, t, 500 + gs_seed, lspar=lsp, dscores=ds)
            gsp = subgraph_from_eids(g, eids)
            sp_ = [evaluate(g, metis_partition(gsp, k, seed=s)[0], y) for s in MSEEDS]
            Qs = np.array([e["Q_orig"] for e in sp_])
            As = np.array([e["AMI"] for e in sp_])
            row = dict(gen=gen, het_deg=HETDEG[gen], het_size=HETSIZE[gen],
                       d_nom=d, graph_seed=gs_seed,
                       d_avg=gm["d_avg_real"], mu_real=gm["mu_real"],
                       deg_cv=gm["deg_cv"], m=gm["m"], k=k,
                       sparsifier=kind, target_ret=t,
                       realized_ret=round(realized, 5), status=status,
                       n_metis_seeds=len(MSEEDS),
                       Q_base_mean=Qb.mean(), Q_base_std=Qb.std(),
                       Q_base_min=Qb.min(), Q_base_max=Qb.max(),
                       AMI_base_mean=Ab.mean(), AMI_base_std=Ab.std(),
                       AMI_base_min=Ab.min(), AMI_base_max=Ab.max(),
                       Q_sparse_mean=Qs.mean(), Q_sparse_std=Qs.std(),
                       Q_sparse_min=Qs.min(), Q_sparse_max=Qs.max(),
                       AMI_sparse_mean=As.mean(), AMI_sparse_std=As.std(),
                       AMI_sparse_min=As.min(), AMI_sparse_max=As.max(),
                       dQ=Qs.mean() - Qb.mean(), dAMI=As.mean() - Ab.mean(),
                       worstcase_dQ=Qs.min() - Qb.max(),
                       worstcase_dAMI=As.min() - Ab.max())
            append_row(out, row)
            log(f"  {kind:7s} t={t:<5} ret={realized:.4f} "
                f"Q {Qb.mean():.5f}->{Qs.mean():.5f} (dQ={row['dQ']:+.5f}, "
                f"worst {row['worstcase_dQ']:+.5f})  "
                f"AMI {Ab.mean():.4f}->{As.mean():.4f} (dAMI={row['dAMI']:+.4f}, "
                f"worst {row['worstcase_dAMI']:+.4f})")
            del gsp
        del g, lsp, Ej, J


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if stage == "gen":
        run_gen()
    elif stage == "armA":
        run_armA(sys.argv[2], sys.argv[3:] or None)
    elif stage == "armB":
        run_armB(sys.argv[2], sys.argv[3:] or None)
    elif stage == "metisseeds":
        run_metisseeds(sys.argv[2])
    else:
        raise SystemExit(f"unknown stage {stage}")
