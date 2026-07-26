#!/usr/bin/env python
"""Exp AB analysis: evaluates P1-P4 and both directions of the kill criterion."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)

r = pd.read_csv(HERE / "results.csv")
rec = pd.read_csv(HERE / "recovery.csv") if (HERE / "recovery.csv").exists() else None
L = r[r.detector == "leiden"].copy()
M = r[r.detector == "metis"].copy() if (r.detector == "metis").any() else None

print("=" * 100)
print("COVERAGE:", L.network.nunique(), "networks,", len(L), "leiden rows",
      "| metis rows:", 0 if M is None else len(M))
print(L.groupby("network").size().to_string())

print("\n" + "=" * 100)
print("RETENTION FLOORS PER NETWORK  (an arm cannot be run below its own floor)")
fl = (L.groupby("network")[["n", "m", "avg_deg", "mst_floor", "lspar_floor",
                            "ld_floor", "lsim_floor", "kn_ret_k1"]].first()
      .sort_values("m"))
print(fl.round(4).to_string())

print("\n" + "=" * 100)
print("P1 / KILL DIRECTION 1 -- honest-transfer dQ on the ORIGINAL graph")
print("matched rows only (|realized - target| <= 0.02)")
mm = L[L.matched == 1]
piv = mm.pivot_table(index=["network", "target_ret"], columns="arm",
                     values="dQ_vs_base_best")
print(piv.round(4).to_string())
print("\nrows with dQ_vs_base_best > 0 (best-of-5 restart baseline):")
pos = mm[mm.dQ_vs_base_best > 0]
cols = ["network", "arm", "target_ret", "realized_ret", "dQ_vs_base_mean",
        "dQ_vs_base_best", "dQ_vs_matched", "Q_base_std", "nc_base", "nc_sparse"]
print(pos[cols].round(5).to_string(index=False) if len(pos) else "  NONE")
print("\nrows exceeding 2x baseline seed sd:")
strong = mm[mm.dQ_vs_base_best > 2 * mm.Q_base_std]
print(strong[cols].round(5).to_string(index=False) if len(strong) else "  NONE")
print("\nnetworks with a positive dQ_vs_base_best, per arm:")
print(pos.groupby("arm").network.nunique().to_string() if len(pos) else "  none")
print("\nbest dQ_vs_base_best per arm (over all matched rows):")
print(mm.groupby("arm").dQ_vs_base_best.max().round(5).to_string())

print("\n" + "=" * 100)
print("P2 -- FRAGMENTATION (nodes in clusters of size < 10, as % of n)")
fr = mm.pivot_table(index=["network", "target_ret"], columns="arm",
                    values="frag_node_frac") * 100
print(fr.round(2).to_string())
print("\nbaseline (unsparsified) fragment %:")
print((L.groupby("network").frag_node_frac_base.first() * 100).round(2).to_string())
bb = mm[mm.arm.isin(["mst_jaccard", "mst_random"])]
oo = mm[mm.arm.isin(["lspar", "dspar", "kn", "ld", "lsim"])]
print(f"\nbackbone arms: max frag% = {bb.frag_node_frac.max()*100:.3f}, "
      f"mean = {bb.frag_node_frac.mean()*100:.3f}, "
      f"rows under 1% = {(bb.frag_node_frac < 0.01).sum()}/{len(bb)}")
print(f"non-backbone arms: mean frag% = {oo.frag_node_frac.mean()*100:.2f}, "
      f"rows under 1% = {(oo.frag_node_frac < 0.01).sum()}/{len(oo)}")

print("\n" + "=" * 100)
print("P3 / KILL DIRECTION 2 -- resolution-matched control and recovery")
haveres = mm[mm.resmatch_done == 1]
if len(haveres):
    print(haveres.pivot_table(index=["network", "target_ret"], columns="arm",
                              values="dQ_vs_resmatch").round(4).to_string())
if rec is not None and len(rec):
    for net, sub in rec.groupby("network"):
        for det, subd in sub.groupby("detector"):
            metric = "AMI" if subd.AMI.notna().any() else "avgF1_ge3"
            if not subd[metric].notna().any():
                continue
            print(f"\n--- recovery {net} [{det}] ({metric}) ---")
            t = (subd.groupby(["arm", "target_ret"])
                 .agg(val=(metric, "mean"), nc=("n_clusters", "mean"),
                      frag=("frag_node_frac", "mean"),
                      chance=(("AMI_chance" if metric == "AMI" else "avgF1_chance"), "mean"))
                 .round(4))
            print(t.to_string())
            base = subd[subd.arm == "baseline"][metric].mean()
            res = subd[subd.arm == "resmatch"][metric]
            print(f"  baseline {metric} = {base:.4f}"
                  + (f" | resolution-matched best = {res.max():.4f}" if len(res) else ""))
            beat = t[(t.val > base) & (~t.index.get_level_values(0)
                                       .isin(["baseline", "resmatch"]))]
            if len(res):
                beat = beat[beat.val > res.max()]
            print("  arms beating BOTH baseline and resolution-matched control:")
            print(beat.to_string() if len(beat) else "    NONE")

print("\n" + "=" * 100)
print("KILL DIRECTION 2 -- nc-MATCHED recovery: every arm cell against the")
print("resolution-matched original-graph partition with the CLOSEST cluster count.")
print("(excess = metric - its own size-matched chance floor)")
if rec is not None and len(rec):
    out = []
    for (net, det), sub in rec.groupby(["network", "detector"]):
        metric = "AMI" if sub.AMI.notna().any() else "avgF1_ge3"
        cfield = "AMI_chance" if metric == "AMI" else "avgF1_chance"
        if not sub[metric].notna().any():
            continue
        res = sub[sub.arm == "resmatch"]
        base = sub[sub.arm == "baseline"]
        base_v = base[metric].mean()
        base_c = base[cfield].mean()
        pool = pd.concat([res, base])
        for (arm, tr), a in sub[~sub.arm.isin(["resmatch", "baseline"])].groupby(
                ["arm", "target_ret"]):
            nc_a = a.n_clusters.mean()
            v_a = a[metric].mean()
            c_a = a[cfield].mean()
            if len(pool):
                j = (pool.n_clusters - nc_a).abs().idxmin()
                nc_r, v_r, c_r = (pool.loc[j, "n_clusters"], pool.loc[j, metric],
                                  pool.loc[j, cfield])
            else:
                nc_r = v_r = c_r = np.nan
            out.append(dict(network=net, detector=det, arm=arm, target=tr,
                            metric=metric, nc_arm=nc_a, val_arm=round(v_a, 4),
                            exc_arm=round(v_a - c_a, 4), nc_ctrl=nc_r,
                            val_ctrl=round(v_r, 4), exc_ctrl=round(v_r - c_r, 4),
                            d_excess=round((v_a - c_a) - (v_r - c_r), 4),
                            vs_base=round((v_a - c_a) - (base_v - base_c), 4)))
    O = pd.DataFrame(out)
    print(O.to_string(index=False))
    print("\ncells where the arm beats BOTH the nc-matched control and the baseline "
          "(on chance-excess):")
    win = O[(O.d_excess > 0) & (O.vs_base > 0)]
    print(win.to_string(index=False) if len(win) else "  NONE")
    if len(win):
        print("\nnetworks with such a win, per arm:")
        print(win.groupby("arm").network.nunique().to_string())

if M is not None:
    print("\n" + "=" * 100)
    print("SECONDARY DETECTOR -- real Metis, k fixed = nc_base (5 option seeds)")
    c = ["network", "arm", "target_ret", "realized_ret", "dQ_vs_base_mean",
         "dQ_vs_base_best", "dQ_vs_matched", "Q_base_std", "AMI", "avgF1_ge3"]
    print(M[c].round(5).to_string(index=False))

print("\n" + "=" * 100)
print("P4 -- Chen's top fidelity preservers (kn, ld, lsim): honest quality")
p4 = mm[mm.arm.isin(["kn", "ld", "lsim"])]
print(p4.groupby("arm")[["dQ_vs_base_mean", "dQ_vs_base_best", "dQ_vs_matched"]]
      .agg(["mean", "max"]).round(5).to_string())

print("\n" + "=" * 100)
print("COST -- end-to-end speedup including the sparsifier's own time")
print(mm.pivot_table(index="network", columns="arm",
                     values="speedup_pipeline").round(2).to_string())
print("\ndetection-only speedup:")
print(mm.pivot_table(index="network", columns="arm",
                     values="speedup_detect_only").round(2).to_string())

sk = HERE / "skipped.csv"
if sk.exists():
    print("\n" + "=" * 100)
    print("SKIPPED (target below the backbone's own retention floor)")
    print(pd.read_csv(sk).round(4).to_string(index=False))
