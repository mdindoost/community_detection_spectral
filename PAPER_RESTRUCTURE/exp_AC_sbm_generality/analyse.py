#!/usr/bin/env python
"""Exp AC analysis: P1-P4 and both kill directions, with the exp_AA LFR
reference recomputed from exp_AA's own CSVs so the comparison is like for like."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AA = HERE.parent / "exp_AA_satuluri_regime"
GENS = ["sbm", "dcsbm", "sbm_ps", "dcsbm_ps"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 600)


def sec(t):
    print("\n" + "=" * 92 + f"\n{t}\n" + "=" * 92)


def load(pat):
    fr = []
    for g in GENS:
        f = HERE / pat.format(g=g)
        if f.exists():
            fr.append(pd.read_csv(f))
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


# --------------------------------------------------------------------------
# LFR reference (exp_AA, mu = 0.5, real Metis)
# --------------------------------------------------------------------------

def lfr_ref():
    f = AA / "results_armB.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    d = d[(d.detector == "metis") & (d.mu == 0.5)].copy()
    d["gen"] = "LFR(exp_AA)"
    return d


def lfr_refA():
    f = AA / "results_armA.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    d = d[d.mu == 0.5].copy()
    d["gen"] = "LFR(exp_AA)"
    return d


# --------------------------------------------------------------------------

def inventory():
    d = load("results_armA_{g}.csv")
    if d.empty:
        d = load("results_armB_{g}.csv")
    if d.empty:
        print("no results yet")
        return
    sec("GRAPH INVENTORY -- realized values (never assumed).  The deg_cv column "
        "is the heterogeneity axis.")
    inv = (d.groupby(["gen", "d_nom"])
             .agg(het_deg=("het_deg", "first"), het_size=("het_size", "first"),
                  m=("m", "mean"), d_avg_real=("d_avg_real", "mean"),
                  d_target=("d_target", "first"),
                  mu_real=("mu_real", "mean"), mu_target=("mu_target", "first"),
                  deg_cv=("deg_cv", "mean"), lfr_deg_cv=("lfr_deg_cv", "first"),
                  deg_min=("deg_min", "mean"), deg_max=("deg_max", "mean"),
                  k=("n_comm_planted", "mean"),
                  cs_min=("comm_size_min", "mean"), cs_max=("comm_size_max", "mean"),
                  nseeds=("graph_seed", "nunique")).round(3))
    print(inv)
    r = lfr_refA()
    if not r.empty:
        print("\n-- exp_AA LFR (mu=0.5) reference --")
        print(r.groupby("d_nom").agg(m=("m", "mean"), d_avg_real=("d_avg_real", "mean"),
                                     mu_real=("mu_real", "mean"),
                                     deg_cv=("deg_cv", "mean"),
                                     k=("n_comm_planted", "mean"),
                                     cs_min=("comm_size_min", "mean"),
                                     cs_max=("comm_size_max", "mean")).round(3))


def armA():
    d = load("results_armA_{g}.csv")
    if d.empty:
        print("no armA data")
        return
    sec("ARM A / P4 -- Leiden honest-transfer modularity.  P4 predicts NO gain "
        "on any SBM variant at any degree.")
    g = d.groupby(["gen", "d_nom", "sparsifier", "target_ret"]).agg(
        d_avg=("d_avg_real", "mean"), deg_cv=("deg_cv", "mean"),
        ret=("realized_ret", "mean"),
        dQ_match=("dQ_vs_matched", "mean"), dQ_best=("dQ_vs_base_best", "mean"),
        seed_sd=("Q_base_std", "mean"), n=("dQ_vs_matched", "size")).reset_index()
    print(g.round(5).to_string(index=False))

    print("\n--- P4 KILL CHECK: cells with dQ_vs_matched > 0 AND > 2*baseline seed sd ---")
    pos = g[(g.dQ_match > 0) & (g.dQ_match > 2 * g.seed_sd)]
    if len(pos):
        print(pos.round(5).to_string(index=False))
        print("\n  ...same cells against the STRICTER best-of-5-restart baseline:")
        print(pos[["gen", "d_nom", "sparsifier", "target_ret", "dQ_best",
                   "seed_sd"]].round(5).to_string(index=False))
        strict = pos[(pos.dQ_best > 0) & (pos.dQ_best > 2 * pos.seed_sd)]
        print(f"\n  cells surviving BOTH baselines: {len(strict)} / {len(g)}")
        if len(strict):
            print(strict.round(5).to_string(index=False))
    else:
        print("  NONE -- P4 holds (no honest Leiden modularity gain on any generator)")

    sec("ARM A -- recovery (AMI) and the resolution-matched control")
    r = d.assign(dAMI_res=d.AMI - d.AMI_resmatch)
    rr = r.groupby(["gen", "d_nom", "sparsifier", "target_ret"]).agg(
        d_avg=("d_avg_real", "mean"), AMI=("AMI", "mean"),
        AMI_base=("AMI_base_mean", "mean"), dAMI=("dAMI_vs_base_mean", "mean"),
        base_sd=("AMI_base_std", "mean"), AMI_res=("AMI_resmatch", "mean"),
        dAMI_res=("dAMI_res", "mean"), nc_s=("nc_sparse", "mean"),
        nc_res=("nc_resmatch", "mean"), n=("AMI", "size")).reset_index()
    print(rr.round(4).to_string(index=False))
    win = rr[(rr.dAMI > 0) & (rr.dAMI > 2 * rr.base_sd)]
    print(f"\n  AMI-gain cells beyond 2x baseline seed sd: {len(win)} / {len(rr)}")
    if len(win):
        print(win.round(4).to_string(index=False))
        surv = win[win.dAMI_res > 0]
        print(f"  ...of which ALSO beat the resolution-matched control: {len(surv)}")
        if len(surv):
            print(surv.round(4).to_string(index=False))

    sec("ARM A -- chance floor / granularity / cost sanity")
    print(d.groupby(["gen", "d_nom"]).agg(
        AMI=("AMI", "mean"), AMI_chance=("AMI_chance", "mean"),
        ARI_chance=("ARI_chance", "mean"),
        nc_base=("nc_base", "mean"), nc_sparse=("nc_sparse", "mean"),
        n_restarts=("n_restarts", "mean"),
        speedup_pipe=("speedup_pipeline", "mean"),
        speedup_det=("speedup_detect_only", "mean")).round(4))


def _metis_table(d, label):
    """exp_AA SUMMARY section-2 table, per generator and degree."""
    rows = []
    for (gen, dn), s in d.groupby(["gen", "d_nom"]):
        rec = dict(gen=gen, d_nom=dn,
                   d_avg=round(s.d_avg_real.mean(), 1),
                   deg_cv=round(s.deg_cv.mean(), 3),
                   AMI_base=round(s.AMI_base.mean(), 3))
        for arm in ("lspar", "dspar", "random"):
            a = s[s.sparsifier == arm]
            if a.empty:
                continue
            rec[f"{arm}_dQ+"] = f"{int((a.dQ > 0).sum())}/{len(a)}"
            rec[f"{arm}_dAMI+"] = f"{int((a.dAMI > 0).sum())}/{len(a)}"
            rec[f"{arm}_dQ"] = round(a.dQ.mean(), 4)
            rec[f"{arm}_dAMI"] = round(a.dAMI.mean(), 4)
            rec[f"{arm}_dQmax"] = round(
                a.groupby("target_ret").dQ.mean().max(), 4)
            rec[f"{arm}_dAMImax"] = round(
                a.groupby("target_ret").dAMI.mean().max(), 4)
        rows.append(rec)
    t = pd.DataFrame(rows).sort_values(["gen", "d_nom"])
    print(f"\n[{label}]")
    print(t.to_string(index=False))
    return t


def transition(t):
    """Transition = smallest realized d_avg at which the best L-Spar retention
    point has a positive mean honest-transfer gain (dQ, and separately dAMI)."""
    out = []
    for gen, s in t.groupby("gen"):
        s = s.sort_values("d_avg")
        tq = s[s.get("lspar_dQmax", pd.Series(dtype=float)) > 0]
        ta = s[s.get("lspar_dAMImax", pd.Series(dtype=float)) > 0]
        out.append(dict(
            gen=gen,
            deg_cv_at_d50=float(s[s.d_nom == 50].deg_cv.mean()) if (s.d_nom == 50).any() else np.nan,
            transition_dQ=float(tq.d_avg.iloc[0]) if len(tq) else np.inf,
            transition_dAMI=float(ta.d_avg.iloc[0]) if len(ta) else np.inf,
            max_dQ=float(s.lspar_dQmax.max()) if "lspar_dQmax" in s else np.nan,
            max_dAMI=float(s.lspar_dAMImax.max()) if "lspar_dAMImax" in s else np.nan))
    return pd.DataFrame(out)


def armB():
    d = load("results_armB_{g}.csv")
    ref = lfr_ref()
    if d.empty and ref.empty:
        print("no armB data")
        return
    sec("ARM B -- real Metis, k pinned to the planted block count.  "
        "P1: the sign flip should occur on both SBM variants.")
    parts = []
    if not ref.empty:
        parts.append(_metis_table(ref, "exp_AA LFR, mu=0.5 (reference)"))
    if not d.empty:
        parts.append(_metis_table(d, "exp_AC SBM family"))
    t = pd.concat(parts, ignore_index=True)

    sec("TRANSITION LOCATION -- smallest realized d_avg with a positive mean "
        "L-Spar+Metis gain at the best retention point")
    print(transition(t).round(4).to_string(index=False))

    if not d.empty:
        sec("ARM B -- full cell detail")
        print(d.groupby(["gen", "d_nom", "sparsifier", "target_ret"]).agg(
            d_avg=("d_avg_real", "mean"), deg_cv=("deg_cv", "mean"),
            ret=("realized_ret", "mean"), Q_base=("Q_base", "mean"),
            dQ=("dQ", "mean"), AMI_base=("AMI_base", "mean"),
            AMI=("AMI", "mean"), dAMI=("dAMI", "mean"), dARI=("dARI", "mean"),
            cv_b=("cv_base", "mean"), cv_s=("cv_sparse", "mean"),
            AMI_chance=("AMI_chance", "mean"), n=("dQ", "size")).round(4))

        sec("HETEROGENEITY AXIS -- the 2x2 factorial at each degree "
            "(best L-Spar+Metis dQ / dAMI)")
        f = []
        for (hd, hs, dn), s in d.groupby(["het_deg", "het_size", "d_nom"]):
            a = s[s.sparsifier == "lspar"]
            f.append(dict(het_deg=hd, het_size=hs, d_nom=dn,
                          d_avg=round(s.d_avg_real.mean(), 1),
                          deg_cv=round(s.deg_cv.mean(), 3),
                          AMI_base=round(s.AMI_base.mean(), 3),
                          best_dQ=round(a.groupby("target_ret").dQ.mean().max(), 4),
                          best_dAMI=round(a.groupby("target_ret").dAMI.mean().max(), 4)))
        print(pd.DataFrame(f).sort_values(["d_nom", "het_deg", "het_size"]).to_string(index=False))


def mseeds():
    f = HERE / "metis_seeds.csv"
    if not f.exists():
        print("\nno metis_seeds.csv")
        return
    d = pd.read_csv(f)
    sec("METIS OPTION-SEED ROBUSTNESS (10 seeds; worst case = min sparse - max base)")
    print(d.groupby(["gen", "d_nom", "sparsifier", "target_ret"]).agg(
        d_avg=("d_avg", "mean"), deg_cv=("deg_cv", "mean"),
        ret=("realized_ret", "mean"),
        Qb=("Q_base_mean", "mean"), Qb_sd=("Q_base_std", "mean"),
        dQ=("dQ", "mean"), worst_dQ=("worstcase_dQ", "mean"),
        Ab=("AMI_base_mean", "mean"), Ab_sd=("AMI_base_std", "mean"),
        dAMI=("dAMI", "mean"), worst_dAMI=("worstcase_dAMI", "mean"),
        n=("dQ", "size")).round(5))
    pos = d[(d.dQ > 0) & (d.worstcase_dQ > 0)]
    print(f"\n  cells with positive dQ in the WORST case over 10 Metis seeds: "
          f"{len(pos)} / {len(d)}")
    if len(pos):
        print(pos.groupby(["gen", "d_nom", "sparsifier"]).agg(
            dQ=("dQ", "mean"), worst=("worstcase_dQ", "mean"),
            dAMI=("dAMI", "mean"), n=("dQ", "size")).round(5))


def heterogeneity_corr():
    """Is the size of the fixed-k gain associated with degree CV?  Reported with
    a leave-one-out jackknife interval (EXPLORATION.md controls checklist)."""
    d = load("results_armB_{g}.csv")
    ref = lfr_ref()
    if d.empty:
        return
    frames = [d]
    if not ref.empty:
        frames.append(ref)
    a = pd.concat(frames, ignore_index=True)
    a = a[a.sparsifier == "lspar"]
    per = (a.groupby(["gen", "d_nom", "graph_seed"] if "graph_seed" in a
                     else ["gen", "d_nom"])
            .agg(deg_cv=("deg_cv", "mean"), d_avg=("d_avg_real", "mean"),
                 dQ=("dQ", "max"), dAMI=("dAMI", "max")).reset_index())
    sec("HETEROGENEITY vs GAIN -- Spearman with leave-one-out jackknife")
    from scipy.stats import spearmanr
    for ycol in ("dQ", "dAMI"):
        x, y = per.deg_cv.values, per[ycol].values
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size < 4:
            continue
        rho = spearmanr(x, y).statistic
        jk = [spearmanr(np.delete(x, i), np.delete(y, i)).statistic
              for i in range(x.size)]
        print(f"  spearman(deg_cv, {ycol}) = {rho:+.3f}   "
              f"jackknife [{min(jk):+.3f}, {max(jk):+.3f}]   n={x.size}")
        xd = per.d_avg.values[ok]
        rho2 = spearmanr(xd, y).statistic
        jk2 = [spearmanr(np.delete(xd, i), np.delete(y, i)).statistic
               for i in range(xd.size)]
        print(f"  spearman(d_avg , {ycol}) = {rho2:+.3f}   "
              f"jackknife [{min(jk2):+.3f}, {max(jk2):+.3f}]")


if __name__ == "__main__":
    what = sys.argv[1:] or ["inv", "A", "B", "M", "H"]
    if "inv" in what:
        inventory()
    if "A" in what:
        armA()
    if "B" in what:
        armB()
    if "M" in what:
        mseeds()
    if "H" in what:
        heterogeneity_corr()
