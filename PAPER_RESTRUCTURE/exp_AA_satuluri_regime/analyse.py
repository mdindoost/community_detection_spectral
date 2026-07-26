#!/usr/bin/env python
"""Exp AA analysis: evaluate pre-registered P1-P4 and both kill directions."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def armA():
    f = HERE / "results_armA.csv"
    if not f.exists():
        print("no armA csv")
        return
    d = pd.read_csv(f)
    sec("ARM A -- graph inventory (realized)")
    inv = (d.groupby(["mu", "d_nom"])
             .agg(gen_mode=("gen_mode", "first"), m=("m", "mean"),
                  d_avg=("d_avg_real", "mean"), mu_real=("mu_real", "mean"),
                  k=("n_comm_planted", "mean"), deg_max=("deg_max", "mean"),
                  deg_cv=("deg_cv", "mean"), floor=("lspar_floor", "mean"),
                  nseeds=("lfr_seed", "nunique")).round(3))
    print(inv)

    sec("ARM A / P1 -- honest-transfer modularity gain vs RUNTIME-MATCHED Leiden")
    p = (d.groupby(["mu", "d_nom", "sparsifier", "target_ret"])
           .agg(ret=("realized_ret", "mean"), dQ_match=("dQ_vs_matched", "mean"),
                dQ_match_sd=("dQ_vs_matched", "std"),
                dQ_best=("dQ_vs_base_best", "mean"),
                seed_sd=("Q_base_std", "mean"), n=("dQ_vs_matched", "size")).round(5))
    print(p)

    print("\n--- P1 KILL CHECK: cells with dQ_vs_matched > 0 AND > 2*seed sd ---")
    g = d.groupby(["mu", "d_nom", "sparsifier", "target_ret"]).agg(
        dQ=("dQ_vs_matched", "mean"), sd=("Q_base_std", "mean"),
        dQb=("dQ_vs_base_best", "mean"), ret=("realized_ret", "mean")).reset_index()
    pos = g[(g.dQ > 0) & (g.dQ > 2 * g.sd)]
    print(pos.round(5) if len(pos) else "  NONE -- P1 holds (no positive honest modularity gain)")
    print(f"  positive cells at high degree (d_nom>=50): "
          f"{len(pos[pos.d_nom >= 50])} / {len(g[g.d_nom >= 50])}")

    sec("ARM A / P2 -- recovery (AMI) gain vs baseline Leiden, by degree")
    r = (d.groupby(["mu", "d_nom", "sparsifier", "target_ret"])
           .agg(ret=("realized_ret", "mean"), AMI=("AMI", "mean"),
                AMI_base=("AMI_base_mean", "mean"),
                dAMI=("dAMI_vs_base_mean", "mean"), dAMI_sd=("dAMI_vs_base_mean", "std"),
                dAMI_match=("dAMI_vs_matched", "mean"),
                base_sd=("AMI_base_std", "mean"), n=("AMI", "size")).round(4))
    print(r)
    gg = d.groupby(["mu", "d_nom", "sparsifier", "target_ret"]).agg(
        dAMI=("dAMI_vs_base_mean", "mean"), sd=("AMI_base_std", "mean")).reset_index()
    win = gg[(gg.dAMI > 0) & (gg.dAMI > 2 * gg.sd)]
    print("\n--- P2: AMI-gain cells beyond 2x baseline seed sd ---")
    print(win.round(4) if len(win) else "  NONE")
    print(f"  of which d_nom>=50: {len(win[win.d_nom>=50])};  d_nom<50: {len(win[win.d_nom<50])}")

    sec("ARM A -- chance floor + granularity sanity")
    print(d.groupby(["d_nom"]).agg(AMI=("AMI", "mean"), AMI_chance=("AMI_chance", "mean"),
                                   nc_sparse=("nc_sparse", "mean"), nc_base=("nc_base", "mean"),
                                   cv_sparse=("cv_sparse", "mean"), cv_base=("cv_base", "mean"),
                                   speedup=("speedup_pipeline", "mean")).round(4))
    if "AMI_resmatch" in d and d.AMI_resmatch.notna().any():
        sec("ARM A -- resolution-matched control (where nc drifted >20%)")
        s = d[d.AMI_resmatch.notna()]
        s = s.assign(dAMI_vs_resmatch=s.AMI - s.AMI_resmatch,
                     dQ_vs_resmatch=s.Q_orig_transfer - s.Q_resmatch)
        print(s.groupby(["mu", "d_nom", "sparsifier", "target_ret"])
               .agg(dAMI_res=("dAMI_vs_resmatch", "mean"),
                    dQ_res=("dQ_vs_resmatch", "mean"), n=("AMI", "size")).round(4))


def armB():
    f = HERE / "results_armB.csv"
    if not f.exists():
        print("no armB csv")
        return
    d = pd.read_csv(f)
    sec("ARM B -- fixed-k partitioners (metis = REAL Metis; *_proxy = proxies)")
    print(d.groupby(["detector", "mu", "d_nom", "sparsifier", "target_ret"])
           .agg(ret=("realized_ret", "mean"), dQ=("dQ", "mean"), dAMI=("dAMI", "mean"),
                dARI=("dARI", "mean"), AMI=("AMI", "mean"), AMI_base=("AMI_base", "mean"),
                cv_b=("cv_base", "mean"), cv_s=("cv_sparse", "mean"),
                n=("dQ", "size")).round(4))
    sec("ARM B / P3 -- per (detector, degree) : does ANY sparsifier x retention gain AMI?")
    for det, sub in d.groupby("detector"):
        rows = []
        for (mu, dn), s2 in sub.groupby(["mu", "d_nom"]):
            g = s2.groupby(["sparsifier", "target_ret"]).dAMI.mean()
            sd = s2.groupby(["sparsifier", "target_ret"]).dAMI.std().fillna(0)
            best = g.idxmax()
            rows.append(dict(mu=mu, d_nom=dn, best_arm=str(best),
                             best_dAMI=round(g.max(), 4),
                             sd=round(float(sd.get(best, 0)), 4),
                             n_pos=int((g > 0).sum()), n_arms=int(g.size),
                             dQ_best=round(s2.groupby(["sparsifier", "target_ret"]).dQ.mean().max(), 4),
                             dcv=round(s2.dcv.mean(), 3)))
        print(f"\n[{det}]")
        print(pd.DataFrame(rows).to_string(index=False))


def armC():
    f = HERE / "results_armC.csv"
    if not f.exists():
        print("no armC csv")
        return
    d = pd.read_csv(f)
    sec("ARM C -- noise injection: dAMI vs injected noise x")
    print(d.groupby(["detector", "sparsifier", "target_ret", "d_nom", "noise_pct"])
           .agg(ret=("realized_ret", "mean"), AMI=("AMI", "mean"),
                AMI_base=("AMI_base", "mean"), dAMI=("dAMI", "mean"),
                dQ_noisy=("dQ_noisy", "mean"), dQ_clean=("dQ_clean", "mean"),
                n=("AMI", "size")).round(4))
    sec("ARM C / P4 -- Spearman(x, dAMI) per (detector, sparsifier, retention, degree) cell")
    rows = []
    for key, s in d.groupby(["detector", "sparsifier", "target_ret", "d_nom"]):
        agg = s.groupby("noise_pct").dAMI.mean()
        rho, pv = spearmanr(agg.index.values, agg.values)
        rows.append(dict(detector=key[0], sparsifier=key[1], target_ret=key[2],
                         d_nom=key[3], rho=round(float(rho), 3), p=round(float(pv), 4),
                         dAMI_at_0=round(float(agg.get(0, np.nan)), 4),
                         dAMI_at_100=round(float(agg.get(100, np.nan)), 4),
                         monotone=bool(np.all(np.diff(agg.values) >= -1e-9))))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    ok = t[(t.rho > 0.8) & (t.dAMI_at_0 <= 0)]
    print(f"\nP4 satisfied cells (rho>0.8 AND dAMI<=0 at x=0): {len(ok)} / {len(t)}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["A", "B", "C"]
    if "A" in what:
        armA()
    if "B" in what:
        armB()
    if "C" in what:
        armC()
