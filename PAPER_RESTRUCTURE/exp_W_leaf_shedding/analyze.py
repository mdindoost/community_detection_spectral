#!/usr/bin/env python
"""
Exp W analysis: evaluate P1, P2, P3 and the kill criterion exactly as registered in
exp_W_leaf_shedding/DESIGN.md.  Reads:
  frag_composition.csv                     (this experiment; degree/k-core/emb axes)
  ../exp_O_core_preservation/results.csv   (measurement 2: ahc_d*/ahd_d* gradients)
  ../exp_O_core_preservation/node_attrs.csv (decile tie ceilings)
Writes VERDICT.txt (full machine-generated tables) to stdout.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPO = HERE.parent / "exp_O_core_preservation"

NETS = ["email-Eu-core", "wiki-Vote", "ca-HepTh", "ca-CondMat", "email-Enron"]
CFGS = [("dspar", 0.5), ("lspar", 0.5), ("lspar", 0.2)]
MIN_FRAG = 20          # a cell is "testable" if it has >= 20 fragment nodes
RES_MIN_FRAG = 1       # resmatch control needs at least some fragments to compare


def rd(p, **kw):
    return pd.read_csv(p, keep_default_na=False, na_values=[""], **kw)


def cell_mean(df, net, arm, cond, sp, tgt, col):
    s = df[(df.network == net) & (df.arm == arm) & (df.condition == cond)
           & (df.sparsifier == sp) & (df.target_ret == tgt)][col]
    s = pd.to_numeric(s, errors="coerce")
    return float(np.nanmean(s)) if len(s) else np.nan


def cell_vals(df, net, arm, cond, sp, tgt, col):
    s = df[(df.network == net) & (df.arm == arm) & (df.condition == cond)
           & (df.sparsifier == sp) & (df.target_ret == tgt)][col]
    return pd.to_numeric(s, errors="coerce").to_numpy()


def main():
    W = rd(HERE / "frag_composition.csv")
    O = rd(EXPO / "results.csv")
    out = []
    P = out.append

    P("=" * 78)
    P("EXP W -- leaf-shedding (H-O1 follow-up).  Machine-generated verdict tables.")
    P("=" * 78)

    # ---------------------------------------------------------------- REPRO --
    P("\nREPRODUCTION CHECK vs exp_O (re-run is bit-for-bit deterministic)")
    for c in ["chk_kp_match", "chk_nfrag_match"]:
        v = pd.to_numeric(W[c], errors="coerce").dropna()
        P(f"  {c}: {int(v.sum())}/{len(v)} rows match")
    for c in ["chk_agree_diff", "chk_fs_maxdiff"]:
        v = pd.to_numeric(W[c], errors="coerce").dropna()
        P(f"  {c}: max = {v.max():.3e} over {len(v)} rows")
    P(f"  rows in frag_composition.csv: {len(W)}")

    # ------------------------------------------------- decile tie ceilings ---
    P("\nTABLE 0 -- decile tie ceilings (max attainable bottom-2 enrichment).")
    P("  Rank deciles break ties at random, so a fragment set drawn entirely from the")
    P("  minimum-value tied block of mass F >= 0.2 can only reach enrichment 1/F.")
    P(f"  {'network':16s} {'arm':5s} {'n':>7s} {'F_deg':>7s} {'ceil_deg':>9s} "
      f"{'F_core':>7s} {'ceil_core':>10s} {'F_emb':>7s} {'ceil_emb':>9s}")
    NA = rd(EXPO / "node_attrs.csv")
    ceilings = {}
    for net in NETS:
        for arm in ["real", "null"]:
            sub = NA[(NA.network == net) & (NA.arm == arm)]
            if not len(sub):
                continue
            row = [net, arm, len(sub)]
            for col, dcol in [("degree", "dec_deg"), ("coreness", "dec_core"),
                              ("embeddedness", "dec_emb")]:
                x = sub[col].to_numpy(dtype=float)
                thr = sub.loc[sub[dcol] <= 1, col].max()   # value at the bot-2 boundary
                F = float(np.mean(x <= thr))
                row += [F, 1.0 / F]
                ceilings[(net, arm, col)] = 1.0 / F
            P(f"  {row[0]:16s} {row[1]:5s} {row[2]:7d} {row[3]:7.3f} {row[4]:9.2f} "
              f"{row[5]:7.3f} {row[6]:10.2f} {row[7]:7.3f} {row[8]:9.2f}")

    # ------------------------------------------------------------------ P1 ---
    P("\n" + "=" * 78)
    P("P1 -- fragment nodes >=2x over-represented in the bottom-2 DEGREE deciles AND")
    P("      bottom-2 K-CORE deciles, in >= 2/3 of testable fragmentation cells")
    P("      (testable = real-arm sparsifier cell with >= %d fragment nodes)" % MIN_FRAG)
    P("=" * 78)
    P(f"  {'network':16s} {'cell':14s} {'nfrag':>7s} {'fracfrag':>9s} "
      f"{'enr_deg':>8s} {'enr_core':>9s} {'enr_emb':>8s} {'testable':>9s} {'pass':>6s}")
    p1_rows = []
    for net in NETS:
        for sp, tgt in CFGS:
            nf = cell_mean(W, net, "real", sp, sp, tgt, "n_frag_nodes")
            ff = cell_mean(W, net, "real", sp, sp, tgt, "frac_frag_nodes")
            ed = cell_mean(W, net, "real", sp, sp, tgt, "deg_enrich_bot2")
            ec = cell_mean(W, net, "real", sp, sp, tgt, "core_enrich_bot2")
            ee = cell_mean(W, net, "real", sp, sp, tgt, "emb_enrich_bot2")
            testable = (not np.isnan(nf)) and nf >= MIN_FRAG
            ok = testable and ed >= 2.0 and ec >= 2.0
            p1_rows.append(dict(network=net, cell=f"{sp} {tgt}", nfrag=nf, fracfrag=ff,
                                enr_deg=ed, enr_core=ec, enr_emb=ee,
                                testable=testable, passes=ok))
            P(f"  {net:16s} {sp+' '+str(tgt):14s} {nf:7.0f} {ff:9.4f} "
              f"{ed:8.2f} {ec:9.2f} {ee:8.2f} {str(testable):>9s} {str(ok):>6s}")
    p1 = pd.DataFrame(p1_rows)
    t = p1[p1.testable]
    P(f"\n  testable cells: {len(t)} / {len(p1)}")
    P(f"  cells passing (deg>=2 AND core>=2): {int(t.passes.sum())} / {len(t)}")
    P(f"  cells passing on DEGREE only:  {int((t.enr_deg >= 2).sum())} / {len(t)}")
    P(f"  cells passing on K-CORE only:   {int((t.enr_core >= 2).sum())} / {len(t)}")
    P("  per network (pass / testable, and whether >= 2/3):")
    net_pass = {}
    for net in NETS:
        tn = t[t.network == net]
        if not len(tn):
            P(f"    {net:16s} no testable cells")
            net_pass[net] = None
            continue
        frac = tn.passes.mean()
        net_pass[net] = frac >= 2 / 3
        P(f"    {net:16s} {int(tn.passes.sum())}/{len(tn)} = {frac:.2f}  "
          f"{'PASS' if frac >= 2/3 else 'FAIL'}")
    nn = [v for v in net_pass.values() if v is not None]
    P(f"  networks with >=2/3 of testable cells passing: {sum(nn)} / {len(nn)}")
    P("  sensitivity: with testable = ANY fragments (>=1 node):")
    for net in NETS:
        for sp, tgt in CFGS:
            nf = cell_mean(W, net, "real", sp, sp, tgt, "n_frag_nodes")
            if np.isnan(nf) or nf < 1 or nf >= MIN_FRAG:
                continue
            ed = cell_mean(W, net, "real", sp, sp, tgt, "deg_enrich_bot2")
            ec = cell_mean(W, net, "real", sp, sp, tgt, "core_enrich_bot2")
            P(f"    extra cell {net:16s} {sp} {tgt}: nfrag={nf:.0f} "
              f"enr_deg={ed:.2f} enr_core={ec:.2f}")

    # ------------------------------------------- tie-robust complement -------
    P("\nTABLE 1 -- tie-robust complement (no decile binning): what fragments ARE.")
    P(f"  {'network':16s} {'cell':14s} {'degmed_f':>9s} {'degmed_g':>9s} "
      f"{'coremed_f':>10s} {'coremed_g':>10s} {'p(deg<=2|f)':>12s} {'p(deg<=2)':>10s} "
      f"{'lift':>6s} {'p(k<=1|f)':>10s} {'p(k<=1)':>8s} {'lift':>6s}")
    for net in NETS:
        for sp, tgt in CFGS:
            nf = cell_mean(W, net, "real", sp, sp, tgt, "n_frag_nodes")
            if np.isnan(nf) or nf < 1:
                continue
            g = lambda c: cell_mean(W, net, "real", sp, sp, tgt, c)
            P(f"  {net:16s} {sp+' '+str(tgt):14s} {g('frag_deg_median'):9.1f} "
              f"{g('all_deg_median'):9.1f} {g('frag_core_median'):10.1f} "
              f"{g('all_core_median'):10.1f} {g('frag_frac_deg_le2'):12.3f} "
              f"{g('all_frac_deg_le2'):10.3f} {g('lift_deg_le2'):6.2f} "
              f"{g('frag_frac_core_le1'):10.3f} {g('all_frac_core_le1'):8.3f} "
              f"{g('lift_core_le1'):6.2f}")

    # ------------------------------------------------------------------ P2 ---
    P("\n" + "=" * 78)
    P("P2 -- registered PREDICTION: the resolution-matched control shows the SAME")
    P("      direction, and sparsification's enrichment exceeds the control's by")
    P("      >=1.5x in FEWER THAN HALF the cells (i.e. NOT sparsification-specific).")
    P("=" * 78)
    P(f"  {'network':16s} {'cell':14s} {'spar_deg':>9s} {'res_deg':>8s} {'ratio':>6s} "
      f"{'spar_core':>10s} {'res_core':>9s} {'ratio':>6s} {'res_nfrag':>10s} "
      f"{'>=1.5x?':>8s}")
    p2_rows = []
    for net in NETS:
        for sp, tgt in CFGS:
            nf = cell_mean(W, net, "real", sp, sp, tgt, "n_frag_nodes")
            if np.isnan(nf) or nf < MIN_FRAG:
                continue
            sd = cell_mean(W, net, "real", sp, sp, tgt, "deg_enrich_bot2")
            sc = cell_mean(W, net, "real", sp, sp, tgt, "core_enrich_bot2")
            rd_ = cell_mean(W, net, "real", "resmatch", sp, tgt, "deg_enrich_bot2")
            rc = cell_mean(W, net, "real", "resmatch", sp, tgt, "core_enrich_bot2")
            rn = cell_mean(W, net, "real", "resmatch", sp, tgt, "n_frag_nodes")
            r1 = sd / rd_ if rd_ and rd_ > 0 else np.nan
            r2 = sc / rc if rc and rc > 0 else np.nan
            exceeds = (not np.isnan(r1)) and (not np.isnan(r2)) and r1 >= 1.5 and r2 >= 1.5
            p2_rows.append(dict(network=net, cell=f"{sp} {tgt}", spar_deg=sd, res_deg=rd_,
                                ratio_deg=r1, spar_core=sc, res_core=rc, ratio_core=r2,
                                res_nfrag=rn, exceeds=exceeds))
            P(f"  {net:16s} {sp+' '+str(tgt):14s} {sd:9.2f} {rd_:8.2f} {r1:6.2f} "
              f"{sc:10.2f} {rc:9.2f} {r2:6.2f} {rn:10.0f} {str(exceeds):>8s}")
    p2 = pd.DataFrame(p2_rows)
    if len(p2):
        P(f"\n  cells where sparsifier exceeds resmatch by >=1.5x on BOTH axes: "
          f"{int(p2.exceeds.sum())} / {len(p2)}")
        P(f"  cells where sparsifier exceeds resmatch by >=1.5x on DEGREE: "
          f"{int((p2.ratio_deg >= 1.5).sum())} / {len(p2)}")
        P(f"  cells where sparsifier exceeds resmatch by >=1.5x on K-CORE: "
          f"{int((p2.ratio_core >= 1.5).sum())} / {len(p2)}")
        P(f"  cells where resmatch MATCHES OR EXCEEDS the sparsifier (ratio <= 1.0), "
          f"degree axis: {int((p2.ratio_deg <= 1.0).sum())} / {len(p2)}")
        P(f"  cells where resmatch MATCHES OR EXCEEDS the sparsifier (ratio <= 1.0), "
          f"k-core axis: {int((p2.ratio_core <= 1.0).sum())} / {len(p2)}")
        P(f"  mean ratio deg = {p2.ratio_deg.mean():.3f}  median = {p2.ratio_deg.median():.3f}")
        P(f"  mean ratio core = {p2.ratio_core.mean():.3f}  median = {p2.ratio_core.median():.3f}")
        P("  leave-one-network-out jackknife on mean ratio (degree axis):")
        for net in NETS:
            s = p2[p2.network != net]
            if len(s):
                P(f"    drop {net:16s} -> mean ratio {s.ratio_deg.mean():.3f} "
                  f"(n={len(s)} cells)")
        P("  jitter floor (plain Leiden seeds 101/102, no sparsification, no gamma change):")
        for net in NETS:
            nfj = cell_mean(W, net, "real", "jitter", "", 1.0, "n_frag_nodes")
            if np.isnan(nfj) or nfj < 1:
                P(f"    {net:16s} jitter nfrag={0 if np.isnan(nfj) else nfj:.0f} (no test)")
                continue
            P(f"    {net:16s} jitter nfrag={nfj:.0f} "
              f"enr_deg={cell_mean(W, net, 'real', 'jitter', '', 1.0, 'deg_enrich_bot2'):.2f} "
              f"enr_core={cell_mean(W, net, 'real', 'jitter', '', 1.0, 'core_enrich_bot2'):.2f}")

    # ------------------------------------------------------------------ P3 ---
    P("\n" + "=" * 78)
    P("P3 -- config-null arm shows WEAKER degree-selectivity than the real graph.")
    P("=" * 78)
    P(f"  {'network':16s} {'cell':14s} {'real_deg':>9s} {'null_deg':>9s} {'real_core':>10s} "
      f"{'null_core':>10s} {'real_nf':>8s} {'null_nf':>8s} {'weaker?':>8s}")
    p3_rows = []
    for net in NETS:
        for sp, tgt in CFGS:
            rnf = cell_mean(W, net, "real", sp, sp, tgt, "n_frag_nodes")
            nnf = cell_mean(W, net, "null", sp, sp, tgt, "n_frag_nodes")
            rdg = cell_mean(W, net, "real", sp, sp, tgt, "deg_enrich_bot2")
            ndg = cell_mean(W, net, "null", sp, sp, tgt, "deg_enrich_bot2")
            rcr = cell_mean(W, net, "real", sp, sp, tgt, "core_enrich_bot2")
            ncr = cell_mean(W, net, "null", sp, sp, tgt, "core_enrich_bot2")
            testable = (rnf >= MIN_FRAG) and (nnf >= MIN_FRAG)
            weaker = testable and (ndg < rdg)
            p3_rows.append(dict(network=net, cell=f"{sp} {tgt}", real_deg=rdg,
                                null_deg=ndg, real_core=rcr, null_core=ncr,
                                real_nf=rnf, null_nf=nnf, testable=testable,
                                weaker=weaker))
            P(f"  {net:16s} {sp+' '+str(tgt):14s} {rdg:9.2f} {ndg:9.2f} {rcr:10.2f} "
              f"{ncr:10.2f} {rnf:8.0f} {nnf:8.0f} "
              f"{(str(weaker) if testable else 'n/t'):>8s}")
    p3 = pd.DataFrame(p3_rows)
    tt = p3[p3.testable]
    P(f"\n  cells testable in BOTH arms (>= {MIN_FRAG} fragments each): {len(tt)} / {len(p3)}")
    if len(tt):
        P(f"  null weaker on degree axis: {int(tt.weaker.sum())} / {len(tt)}")
        P(f"  null weaker on k-core axis: {int((tt.null_core < tt.real_core).sum())} / {len(tt)}")

    # -------------------------------------------------- measurement (2) ------
    P("\n" + "=" * 78)
    P("MEASUREMENT 2 -- agreement gradient by K-CORE / DEGREE decile")
    P("  (analysed from exp_O results.csv columns core_ah_gap / deg_ah_gap / ahc_d* /")
    P("   ahd_d*; positive = high-coreness/high-degree nodes agree MORE with P0)")
    P("=" * 78)
    P(f"  {'network':16s} {'arm':5s} {'cell':16s} {'emb_gap':>8s} {'core_gap':>9s} "
      f"{'deg_gap':>8s}")
    for net in NETS:
        for arm in ["real", "null"]:
            for cond, sp, tgt in ([("jitter", "", 1.0)]
                                  + [(s, s, t) for s, t in CFGS]
                                  + [("resmatch", s, t) for s, t in CFGS]):
                sel = O[(O.network == net) & (O.arm == arm) & (O.condition == cond)
                        & (O.target_ret == tgt)]
                if cond == "resmatch":
                    sel = sel[sel.sparsifier == sp]
                elif cond != "jitter":
                    sel = sel[sel.sparsifier == sp]
                if not len(sel):
                    continue
                lbl = cond if cond == "jitter" else f"{cond} {sp} {tgt}"
                P(f"  {net:16s} {arm:5s} {lbl:16s} "
                  f"{sel.ah_gap.mean():8.3f} {sel.core_ah_gap.mean():9.3f} "
                  f"{sel.deg_ah_gap.mean():8.3f}")

    P("\n  Per-decile Hungarian agreement by K-CORE decile (real arm, mean over seeds):")
    for net in NETS:
        for sp, tgt in CFGS:
            sel = O[(O.network == net) & (O.arm == "real") & (O.condition == sp)
                    & (O.target_ret == tgt)]
            if not len(sel):
                continue
            v = [sel[f"ahc_d{i}"].mean() for i in range(10)]
            P(f"    {net:16s} {sp} {tgt}: " + " ".join(f"{x:.3f}" for x in v))
    P("\n  Per-decile Hungarian agreement by DEGREE decile (real arm, mean over seeds):")
    for net in NETS:
        for sp, tgt in CFGS:
            sel = O[(O.network == net) & (O.arm == "real") & (O.condition == sp)
                    & (O.target_ret == tgt)]
            if not len(sel):
                continue
            v = [sel[f"ahd_d{i}"].mean() for i in range(10)]
            P(f"    {net:16s} {sp} {tgt}: " + " ".join(f"{x:.3f}" for x in v))

    # ---------------------------------------------------------------- KILL ---
    P("\n" + "=" * 78)
    P("KILL CRITERION (registered):")
    P("  (a) if P1 fails (no >=2x bottom-2 enrichment in a MAJORITY of testable cells),")
    P("      H-O1 is dead outright.")
    P("  (b) if P1 holds but the resmatch control matches or exceeds the enrichment in")
    P("      >= half the cells, the finding is a GRANULARITY fact, not a sparsification")
    P("      fact.")
    P("=" * 78)
    maj = int(t.passes.sum()) > len(t) / 2 if len(t) else False
    P(f"  (a) majority of testable cells with >=2x on BOTH axes: "
      f"{int(t.passes.sum())}/{len(t)} -> P1 {'HOLDS' if maj else 'FAILS'}")
    if len(p2):
        ge_half_deg = int((p2.ratio_deg <= 1.0).sum()) >= len(p2) / 2
        ge_half_core = int((p2.ratio_core <= 1.0).sum()) >= len(p2) / 2
        P(f"  (b) resmatch matches-or-exceeds on degree axis in "
          f"{int((p2.ratio_deg <= 1.0).sum())}/{len(p2)} cells -> "
          f"{'YES' if ge_half_deg else 'NO'}")
        P(f"  (b) resmatch matches-or-exceeds on k-core axis in "
          f"{int((p2.ratio_core <= 1.0).sum())}/{len(p2)} cells -> "
          f"{'YES' if ge_half_core else 'NO'}")

    txt = "\n".join(out)
    (HERE / "VERDICT.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
