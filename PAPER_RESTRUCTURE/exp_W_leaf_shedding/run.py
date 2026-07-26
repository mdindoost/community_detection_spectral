#!/usr/bin/env python
"""
Experiment W: leaf-shedding (pre-registered follow-up of exp_O's H-O1).

Pre-registered design: exp_W_leaf_shedding/DESIGN.md. This script implements exactly
that design.

H-O1: exp_O measured fragment composition on the EMBEDDEDNESS axis and found the
direction reversed (fragments over-represented among HIGH-embeddedness nodes).  Since
embeddedness anti-correlates with degree (r -0.25..-0.55) and k-core (r -0.22..-0.57),
the real axis may be DEGREE / K-CORE: fragments = the graph's low-degree, low-coreness
leaves.  exp_O stored fragment composition only per EMBEDDEDNESS decile (fs_d*/fr_d*),
and did not store partitions, so the composition on the degree/k-core axes is not
recoverable from its CSVs.  This script re-runs ONLY the cells DESIGN names and
recomputes composition on all three axes.

Everything is reused verbatim from exp_O_core_preservation/run.py (imported as a
module): graph loading, DSpar, L-Spar, Leiden, node stats, rank deciles, matching,
seeds.  The re-run is bit-for-bit deterministic, so each row also re-derives exp_O's
stored quantities (kp, n_frag_nodes, agree_all_hung, fs_d*) as a reproduction check
(columns chk_*).

Cells (DESIGN): the moderate-fragmentation cells dspar 0.5, lspar 0.5, lspar 0.2, plus
their resolution-matched controls (gamma read from exp_O results.csv -- the bisection
result is reused, so no re-bisection), plus the config-null arm, on the exp_O five.
ADDITION (labelled, does not replace a registered arm): the plain-Leiden seed-jitter
condition, as a "no sparsification, no resolution change" floor.

Outputs (appended incrementally):
  frag_composition.csv   one row per (network, arm, condition, sparsifier, seeds)

Usage:  run.py [network1,network2,...]
"""

import importlib.util
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import igraph as ig

REPO = Path("/home/md724/community_detection_spectral")
EXPO = REPO / "PAPER_RESTRUCTURE" / "exp_O_core_preservation"
HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("runO", EXPO / "run.py")
O = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(O)

# ---- registered cells ------------------------------------------------------
DSPAR_ALPHAS = [0.5]              # DESIGN: dspar 0.5
LSPAR_TARGETS = [0.5, 0.2]        # DESIGN: lspar 0.5, lspar 0.2
RESMATCH_CFGS = [("dspar", 0.5), ("lspar", 0.5), ("lspar", 0.2)]
RESMATCH_SEEDS = [100, 101, 102]  # exp_O used P0_SEED then JITTER_SEEDS

OUT = HERE / "frag_composition.csv"


def load_gammas():
    """(network, arm, sparsifier, target, leiden_seed) -> gamma, from exp_O results."""
    import csv as _csv
    out = {}
    with open(EXPO / "results.csv") as fh:
        for r in _csv.DictReader(fh):
            if r["condition"] != "resmatch":
                continue
            out[(r["network"], r["arm"], r["sparsifier"], float(r["target_ret"]),
                 int(r["leiden_seed"]))] = float(r["gamma"])
    return out


def load_expO_rows():
    """(network, arm, condition, sparsifier, target, spar_seed, leiden_seed) -> row."""
    import csv as _csv
    out = {}
    with open(EXPO / "results.csv") as fh:
        for r in _csv.DictReader(fh):
            key = (r["network"], r["arm"], r["condition"], r["sparsifier"],
                   r["target_ret"], r["spar_seed"], r["leiden_seed"])
            out[key] = r
    return out


def load_node_attrs(name):
    """exp_O node_attrs rows for one network: {arm: dict of arrays}."""
    import csv as _csv
    per = {}
    with open(EXPO / "node_attrs.csv") as fh:
        for r in _csv.DictReader(fh):
            if r["network"] != name:
                continue
            per.setdefault(r["arm"], []).append(
                (int(r["node"]), int(r["degree"]), int(r["coreness"]),
                 float(r["embeddedness"]), int(r["dec_emb"]), int(r["dec_core"]),
                 int(r["dec_deg"])))
    out = {}
    for arm, rows in per.items():
        rows.sort()
        a = np.asarray(rows, dtype=np.float64)
        out[arm] = dict(degree=a[:, 1].astype(np.int64),
                        coreness=a[:, 2].astype(np.int64),
                        embeddedness=a[:, 3],
                        dec_emb=a[:, 4].astype(np.int64),
                        dec_core=a[:, 5].astype(np.int64),
                        dec_deg=a[:, 6].astype(np.int64))
    return out


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def composition(dec, is_frag, nf):
    """(share per decile, rate per decile) for a 10-bin decile vector."""
    dsz = np.bincount(dec, minlength=10).astype(np.float64)
    fcnt = (np.bincount(dec[is_frag], minlength=10).astype(np.float64)
            if nf else np.zeros(10))
    share = fcnt / nf if nf else np.full(10, np.nan)
    rate = fcnt / np.maximum(dsz, 1)
    return share, rate


def measure_W(ctx, memb_p, fields):
    t0 = time.perf_counter()
    n, a, ka = ctx["n"], ctx["memb0"], ctx["ka"]
    _, b = np.unique(np.asarray(memb_p), return_inverse=True)
    kb = int(b.max()) + 1

    import scipy.sparse as sp
    M = sp.coo_matrix((np.ones(n), (a, b)), shape=(ka, kb)).tocsr()
    hung, plur, method, _ = O.match_maps(M, ka, kb)
    ag_h = (hung[b] == a).astype(np.float64)
    ag_p = (plur[b] == a).astype(np.float64)

    # measurement (2): agreement gradient by k-core / degree / embeddedness decile
    ahc = O.decile_mean(ctx["dec_core"], ag_h)
    ahd = O.decile_mean(ctx["dec_deg"], ag_h)
    ahe = O.decile_mean(ctx["dec_emb"], ag_h)

    def gap3(v):
        return float(np.mean(v[7:10])) - float(np.mean(v[0:3]))

    # measurement (1)/(3): fragment composition on all three axes
    csize = np.bincount(b, minlength=kb)
    is_frag = csize[b] < O.FRAG_MAX
    nf = int(is_frag.sum())

    fs_c, fr_c = composition(ctx["dec_core"], is_frag, nf)   # k-core axis
    fs_d, fr_d = composition(ctx["dec_deg"], is_frag, nf)    # degree axis
    fs_e, fr_e = composition(ctx["dec_emb"], is_frag, nf)    # embeddedness (exp_O check)

    deg, core = ctx["deg"], ctx["core"]
    row = dict(fields)
    row.update(
        n=n, k0=ka, kp=kb, match_method=method,
        agree_all_hung=float(ag_h.mean()), agree_all_plur=float(ag_p.mean()),
        n_frag_nodes=nf, frac_frag_nodes=nf / n,
        # --- registered P1 statistics ---
        core_enrich_bot2=float(fs_c[0] + fs_c[1]) / 0.2 if nf else np.nan,
        core_enrich_top2=float(fs_c[8] + fs_c[9]) / 0.2 if nf else np.nan,
        deg_enrich_bot2=float(fs_d[0] + fs_d[1]) / 0.2 if nf else np.nan,
        deg_enrich_top2=float(fs_d[8] + fs_d[9]) / 0.2 if nf else np.nan,
        emb_enrich_bot2=float(fs_e[0] + fs_e[1]) / 0.2 if nf else np.nan,
        emb_enrich_top2=float(fs_e[8] + fs_e[9]) / 0.2 if nf else np.nan,
        # --- tie-robust complements (deciles of degree/coreness are heavily tied) ---
        frag_deg_median=float(np.median(deg[is_frag])) if nf else np.nan,
        all_deg_median=float(np.median(deg)),
        frag_deg_mean=float(np.mean(deg[is_frag])) if nf else np.nan,
        all_deg_mean=float(np.mean(deg)),
        frag_core_median=float(np.median(core[is_frag])) if nf else np.nan,
        all_core_median=float(np.median(core)),
        frag_core_mean=float(np.mean(core[is_frag])) if nf else np.nan,
        all_core_mean=float(np.mean(core)),
        frag_frac_deg_le2=float(np.mean(deg[is_frag] <= 2)) if nf else np.nan,
        all_frac_deg_le2=float(np.mean(deg <= 2)),
        frag_frac_core_le1=float(np.mean(core[is_frag] <= 1)) if nf else np.nan,
        all_frac_core_le1=float(np.mean(core <= 1)),
        lift_deg_le2=(float(np.mean(deg[is_frag] <= 2)) / float(np.mean(deg <= 2))
                      if nf and np.mean(deg <= 2) > 0 else np.nan),
        lift_core_le1=(float(np.mean(core[is_frag] <= 1)) / float(np.mean(core <= 1))
                       if nf and np.mean(core <= 1) > 0 else np.nan),
        # --- measurement (2) ---
        core_ah_gap=gap3(ahc), deg_ah_gap=gap3(ahd), emb_ah_gap=gap3(ahe),
        seconds=time.perf_counter() - t0,
    )
    for i in range(10):
        row[f"fsc_d{i}"] = float(fs_c[i])
    for i in range(10):
        row[f"frc_d{i}"] = float(fr_c[i])
    for i in range(10):
        row[f"fsd_d{i}"] = float(fs_d[i])
    for i in range(10):
        row[f"frd_d{i}"] = float(fr_d[i])
    for i in range(10):
        row[f"fse_d{i}"] = float(fs_e[i])
    for i in range(10):
        row[f"ahc_d{i}"] = float(ahc[i])
    for i in range(10):
        row[f"ahd_d{i}"] = float(ahd[i])
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_arm(g, name, arm, gammas, expO, nattrs):
    ctx = O.build_ctx(g, name, arm)
    n = ctx["n"]

    # ---- reproduction check against exp_O node_attrs.csv --------------------
    na = nattrs.get(arm)
    if na is not None and na["degree"].size == n:
        ok = (np.array_equal(na["degree"], ctx["deg"])
              and np.array_equal(na["coreness"], ctx["core"])
              and np.allclose(na["embeddedness"], ctx["emb"])
              and np.array_equal(na["dec_core"], ctx["dec_core"])
              and np.array_equal(na["dec_deg"], ctx["dec_deg"])
              and np.array_equal(na["dec_emb"], ctx["dec_emb"]))
        O.log(f"  REPRO node_attrs match: {ok}")
    else:
        O.log(f"  REPRO node_attrs match: NO ROWS (n={n})")

    def emit(memb_p, fields):
        row = measure_W(ctx, memb_p, fields)
        key = (name, arm, fields["condition"], fields["sparsifier"],
               str(fields["target_ret"]), str(fields["spar_seed"]),
               str(fields["leiden_seed"]))
        ref = expO.get(key)
        if ref is None and fields["condition"] == "jitter":
            ref = expO.get((name, arm, "jitter", "", "1.0", "", str(fields["leiden_seed"])))
        if ref is not None:
            row["chk_kp_match"] = int(int(ref["kp"]) == row["kp"])
            row["chk_nfrag_match"] = int(int(ref["n_frag_nodes"]) == row["n_frag_nodes"])
            row["chk_agree_diff"] = abs(float(ref["agree_all_hung"]) - row["agree_all_hung"])
            fse = np.array([row[f"fse_d{i}"] for i in range(10)])
            try:
                ref_fs = np.array([float(ref[f"fs_d{i}"]) for i in range(10)])
                row["chk_fs_maxdiff"] = float(np.nanmax(np.abs(ref_fs - fse))) \
                    if np.isfinite(ref_fs).any() else 0.0
            except ValueError:
                row["chk_fs_maxdiff"] = 0.0 if row["n_frag_nodes"] == 0 else np.nan
        else:
            row["chk_kp_match"] = ""
            row["chk_nfrag_match"] = ""
            row["chk_agree_diff"] = ""
            row["chk_fs_maxdiff"] = ""
        O.append_row(OUT, row)
        O.log(f"    [{fields['condition']:8s} {fields['sparsifier']:6s} "
              f"t={fields['target_ret']} ss={fields['spar_seed']} "
              f"ls={fields['leiden_seed']}] k'={row['kp']:6d} "
              f"frag={row['frac_frag_nodes']:.3f} "
              f"enrB2 deg={_f(row['deg_enrich_bot2'])} core={_f(row['core_enrich_bot2'])} "
              f"emb={_f(row['emb_enrich_bot2'])} | "
              f"lift(deg<=2)={_f(row['lift_deg_le2'])} "
              f"repro k/f/a={row['chk_kp_match']}/{row['chk_nfrag_match']}/"
              f"{row['chk_agree_diff'] if row['chk_agree_diff']=='' else round(row['chk_agree_diff'],6)}")
        return row

    def bf(cond, sparsifier, target, realized, spar_seed, lseed, gamma, msp):
        return dict(network=name, arm=arm, condition=cond, sparsifier=sparsifier,
                    target_ret=target, realized_ret=realized, spar_seed=spar_seed,
                    leiden_seed=lseed, gamma=gamma, m_sparse=msp)

    # ---- jitter floor (labelled ADDITION) ---------------------------------
    for s in O.JITTER_SEEDS:
        memb, _, _, _ = O.leiden(g, s)
        emit(memb, bf("jitter", "", 1.0, 1.0, "", s, 1.0, ctx["m"]))

    # ---- DSpar 0.5 ---------------------------------------------------------
    edge_arr, scores = O.dspar_scores(g)
    for alpha in DSPAR_ALPHAS:
        for ss in O.SPAR_SEEDS:
            gs, kept, realized = O.sparsify(g, edge_arr, scores, alpha, ss)
            for ls in O.LEIDEN_SEEDS:
                memb, _, _, _ = O.leiden(gs, ls)
                emit(memb, bf("dspar", "dspar", alpha, round(realized, 6), ss, ls,
                              1.0, gs.ecount()))
            del gs
    del edge_arr, scores

    # ---- L-Spar 0.5 / 0.2 --------------------------------------------------
    E, J, deg_j = O.edge_jaccard(g)
    ls_obj = O.LSpar(g, J=J, E=E, deg=deg_j)
    for target in LSPAR_TARGETS:
        e_used, realized, status = ls_obj.bisect_e(target)
        gs = O.subgraph_from_eids(g, ls_obj.select(e_used))
        O.log(f"  lspar target={target} e={e_used:.4f} realized={realized:.4f} ({status})")
        for lsd in O.LEIDEN_SEEDS:
            memb, _, _, _ = O.leiden(gs, lsd)
            emit(memb, bf("lspar", "lspar", target, round(realized, 6), "", lsd,
                          1.0, gs.ecount()))
        del gs
    del E, J, ls_obj

    # ---- resolution-matched control (gamma reused from exp_O) --------------
    for sparsifier, target in RESMATCH_CFGS:
        for lsd in RESMATCH_SEEDS:
            gamma = gammas.get((name, arm, sparsifier, target, lsd))
            if gamma is None:
                O.log(f"  !! no stored gamma for {name}/{arm}/{sparsifier}/{target}/{lsd}")
                continue
            memb, _, _, _ = O.leiden(g, lsd, resolution=gamma)
            emit(memb, bf("resmatch", sparsifier, target, 1.0, "", lsd,
                          round(gamma, 6), ctx["m"]))


def _f(x):
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "  nan"


def run(networks):
    gammas = load_gammas()
    expO = load_expO_rows()
    for name in networks:
        t0 = time.perf_counter()
        g = O.load_lcc_graph(name)
        nattrs = load_node_attrs(name)
        O.log(f"\n{'='*78}\n{name}: n={g.vcount():,} m={g.ecount():,} "
              f"(load {time.perf_counter()-t0:.1f}s)\n{'='*78}")

        O.log("-- ARM real")
        run_arm(g, name, "real", gammas, expO, nattrs)

        t0 = time.perf_counter()
        gr = g.copy()
        ig.set_random_number_generator(random.Random(O.REWIRE_SEED))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gr.rewire(n=O.SWAPS_PER_EDGE * g.ecount(), mode="simple")
        ig.set_random_number_generator(random)
        O.log(f"-- ARM null (rewired in {time.perf_counter()-t0:.1f}s; deg seq identical: "
              f"{np.array_equal(np.sort(g.degree()), np.sort(gr.degree()))})")
        run_arm(gr, name, "null", gammas, expO, nattrs)
        del g, gr
        O.log(f"** {name} DONE")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    run(arg.split(",") if arg else O.SMALL_FIVE)
