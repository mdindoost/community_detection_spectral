#!/usr/bin/env python
"""Exp O verdict machine: evaluates P1-P5 and the kill criterion EXACTLY as
pre-registered in DESIGN.md. Reads results.csv, prints a text report.
Nothing here re-slices the data; every threshold is the registered one."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
NETS = ["email-Eu-core", "wiki-Vote", "ca-HepTh", "ca-CondMat", "email-Enron",
        "com-DBLP", "com-Amazon"]
RET05 = {"dspar": 0.5, "lspar": 0.5}     # "retention 0.5" cells


def sel(d, **kw):
    m = np.ones(len(d), dtype=bool)
    for k, v in kw.items():
        m &= (d[k] == v)
    return d[m]


def main():
    d = pd.read_csv(HERE / "results.csv")
    d["arm"] = d["arm"].fillna("null")   # pandas reads the string "null" as NaN
    nets = [x for x in NETS if x in set(d.network)]
    real = d[d.arm == "real"]
    null = d[d.arm == "null"]
    print(f"networks: {nets}\nrows: {len(d)} (real {len(real)}, null {len(null)})")

    # ------------------------------------------------------- structural preamble
    npath = HERE / "node_attrs.csv"
    if npath.exists():
        na = pd.read_csv(npath, low_memory=False)
        na["arm"] = na["arm"].fillna("null")
        print("\n" + "=" * 100)
        print("TABLE 0 -- how the pre-registered 'core' proxy (embeddedness) relates to "
              "k-core index and degree")
        print("=" * 100)
        print(f"{'network':14s} {'arm':5s} {'n':>8s} {'k0':>5s} {'embmean':>8s} "
              f"{'frac(emb=1)':>11s} {'r(emb,core)':>11s} {'r(emb,deg)':>11s} "
              f"{'r(core,deg)':>11s} {'medcore_top3dec':>15s} {'medcore_bot3dec':>15s} "
              f"{'meddeg_top3dec':>14s} {'meddeg_bot3dec':>14s}")
        for net in nets:
            for arm in ("real", "null"):
                s = na[(na.network == net) & (na.arm == arm)]
                if not len(s):
                    continue
                e, c, dg = s.embeddedness.values, s.coreness.values, s.degree.values
                de = s.dec_emb.values
                hi, lo = c[de >= 7], c[de <= 2]
                hd, ld = dg[de >= 7], dg[de <= 2]
                print(f"{net:14s} {arm:5s} {len(s):8d} {s.comm.nunique():5d} "
                      f"{e.mean():8.4f} {float((e >= 1).mean()):11.4f} "
                      f"{np.corrcoef(e, c)[0,1]:+11.4f} {np.corrcoef(e, dg)[0,1]:+11.4f} "
                      f"{np.corrcoef(c, dg)[0,1]:+11.4f} "
                      f"{np.median(hi):15.1f} {np.median(lo):15.1f} "
                      f"{np.median(hd):14.1f} {np.median(ld):14.1f}")

    # ------------------------------------------------------------------ table
    print("\n" + "=" * 100)
    print("TABLE 1 -- real arm, per condition (mean over seeds; +- sd)")
    print("=" * 100)
    hdr = (f"{'network':14s} {'cond':9s} {'spars':6s} {'t':>5s} {'k0':>5s} {'k+':>7s} "
           f"{'AMI':>6s} {'agree':>6s} {'bot3':>6s} {'top3':>6s} {'gap':>7s} {'rho':>6s} "
           f"{'incr':>4s} {'cohmed':>6s} {'pcoh':>6s} {'frag':>6s} {'enrB2':>6s} {'enrT2':>6s}")
    print(hdr)
    for net in nets:
        for arm, tag in (("real", ""), ("null", "*")):
            sub = d[(d.network == net) & (d.arm == arm)]
            for (cond, spars, t), grp in sub.groupby(
                    ["condition", "sparsifier", "target_ret"], dropna=False):
                if cond == "resmatch" and (grp.gamma == 1.0).all():
                    pass
                print(f"{net + tag:14s} {cond:9s} {str(spars):6s} {t:5.2f} "
                      f"{grp.k0.mean():5.0f} {grp.kp.mean():7.0f} "
                      f"{grp.ami.mean():6.3f} {grp.agree_all_hung.mean():6.3f} "
                      f"{grp.ah_bot3.mean():6.3f} {grp.ah_top3.mean():6.3f} "
                      f"{grp.ah_gap.mean():+7.3f} {grp.spearman_hung.mean():6.2f} "
                      f"{grp.n_incr_steps_hung.mean():4.1f} {grp.coh_median.mean():6.3f} "
                      f"{grp.periph_coh_median.mean():6.3f} "
                      f"{grp.frac_frag_nodes.mean():6.3f} "
                      f"{grp.frag_enrich_bot2.mean():6.2f} {grp.frag_enrich_top2.mean():6.2f}")
        print("-" * 100)

    # ------------------------------------------------------------------ P1
    print("\n" + "=" * 100)
    print("P1: agreement rises monotonically with embeddedness decile; "
          "top-3-decile agreement > 0.85 at retention 0.5 for DSpar")
    print("=" * 100)
    p1 = []
    for net in nets:
        g = sel(real, network=net, condition="dspar", target_ret=0.5)
        if not len(g):
            continue
        top3, rho, incr = g.ah_top3.mean(), g.spearman_hung.mean(), g.n_incr_steps_hung.mean()
        mono = incr >= 8.0
        ok = (top3 > 0.85) and mono
        p1.append(ok)
        print(f"  {net:14s} top3={top3:.3f} (sd {g.ah_top3.std():.3f})  rho={rho:+.2f}  "
              f"incr_steps={incr:.1f}/9  monotone={mono}  -> {'PASS' if ok else 'FAIL'}")
    print(f"  VERDICT P1: {sum(p1)}/{len(p1)} networks")

    # ------------------------------------------------------------------ P2
    print("\n" + "=" * 100)
    print("P2: sub-10-node P' fragments >= 2x over-represented in the bottom-2 "
          "embeddedness deciles (enrich_bot2 >= 2.0)")
    print("=" * 100)
    p2 = []
    for net in nets:
        for spars in ("dspar", "lspar"):
            for t in sorted(set(sel(real, condition=spars).target_ret)):
                g = sel(real, network=net, condition=spars, target_ret=t)
                if not len(g):
                    continue
                nf = g.n_frag_nodes.mean()
                e = g.frag_enrich_bot2.mean()
                if nf < 20:
                    print(f"  {net:14s} {spars:6s} t={t:.2f}  n_frag={nf:8.0f}  "
                          f"(too few fragments -- untestable)")
                    continue
                ok = e >= 2.0
                p2.append(ok)
                print(f"  {net:14s} {spars:6s} t={t:.2f}  n_frag={nf:8.0f} "
                      f"({g.frac_frag_nodes.mean()*100:5.1f}% of n)  enrich_bot2={e:5.2f}  "
                      f"enrich_top2={g.frag_enrich_top2.mean():5.2f}  "
                      f"-> {'PASS' if ok else 'FAIL'}")
    print(f"  VERDICT P2: {sum(p2)}/{len(p2)} testable cells")

    # ------------------------------------------------------------------ P3
    print("\n" + "=" * 100)
    print("P3: median core cohesion >= 0.8 at retention 0.5, both sparsifiers")
    print("=" * 100)
    p3 = []
    for net in nets:
        for spars, t in RET05.items():
            g = sel(real, network=net, condition=spars, target_ret=t)
            if not len(g):
                continue
            c = g.coh_median.mean()
            ok = c >= 0.8
            p3.append(ok)
            print(f"  {net:14s} {spars:6s} t={t:.2f}  core_coh_med={c:.3f} "
                  f"(sd {g.coh_median.std():.3f})  periph_coh_med={g.periph_coh_median.mean():.3f}  "
                  f"core-periph={c - g.periph_coh_median.mean():+.3f}  "
                  f"-> {'PASS' if ok else 'FAIL'}")
    print(f"  VERDICT P3: {sum(p3)}/{len(p3)} cells")

    # ------------------------------------------------------------------ P4
    print("\n" + "=" * 100)
    print("P4: null arm gradient at most HALF the real gradient "
          "(top-minus-bottom decile gap)")
    print("=" * 100)
    p4 = []
    for net in nets:
        for spars in ("dspar", "lspar"):
            for t in sorted(set(sel(real, condition=spars).target_ret)):
                gr = sel(real, network=net, condition=spars, target_ret=t)
                gn = sel(null, network=net, condition=spars, target_ret=t)
                if not len(gr) or not len(gn):
                    continue
                r, nl = gr.ah_gap.mean(), gn.ah_gap.mean()
                se = np.sqrt(gr.ah_gap.std(ddof=1) ** 2 / len(gr)
                             + gn.ah_gap.std(ddof=1) ** 2 / len(gn))
                ok = nl <= 0.5 * r
                p4.append(ok)
                print(f"  {net:14s} {spars:6s} t={t:.2f}  gap_real={r:+.3f}  "
                      f"gap_null={nl:+.3f}  ratio={nl/r if r else float('nan'):+.2f}  "
                      f"SE(diff)={se:.3f}  -> {'PASS' if ok else 'FAIL'}")
    print(f"  VERDICT P4: {sum(p4)}/{len(p4)} cells")

    # ------------------------------------------------------------------ P5
    print("\n" + "=" * 100)
    print("P5: resolution-matched control does NOT reproduce the embeddedness "
          "selectivity (its fragment composition is closer to uniform)")
    print("=" * 100)
    p5 = []
    for net in nets:
        for spars in ("dspar", "lspar"):
            for t in sorted(set(sel(real, condition=spars).target_ret)):
                gs = sel(real, network=net, condition=spars, target_ret=t)
                gm = real[(real.network == net) & (real.condition == "resmatch")
                          & (real.sparsifier == spars) & (real.target_ret == t)]
                if not len(gs) or not len(gm):
                    continue
                if gs.n_frag_nodes.mean() < 20 and gm.n_frag_nodes.mean() < 20:
                    continue
                es, em = gs.frag_enrich_bot2.mean(), gm.frag_enrich_bot2.mean()
                # "closer to uniform" = |enrich - 1| smaller for the resmatch control
                ok = abs(em - 1.0) < abs(es - 1.0)
                p5.append(ok)
                print(f"  {net:14s} {spars:6s} t={t:.2f}  enrichB2 spars={es:5.2f} "
                      f"(n_frag {gs.n_frag_nodes.mean():7.0f})  resmatch={em:5.2f} "
                      f"(n_frag {gm.n_frag_nodes.mean():7.0f})  "
                      f"|dev| {abs(es-1):.2f} vs {abs(em-1):.2f} -> "
                      f"{'PASS' if ok else 'FAIL'}")
                print(f"                 gap_spars={gs.ah_gap.mean():+.3f} "
                      f"gap_resmatch={gm.ah_gap.mean():+.3f}  "
                      f"coh_spars={gs.coh_median.mean():.3f} "
                      f"coh_resmatch={gm.coh_median.mean():.3f}")
    print(f"  VERDICT P5: {sum(p5)}/{len(p5)} cells")

    # ------------------------------------------------------------------ KILL
    print("\n" + "=" * 100)
    print("KILL CRITERION: (A) gradient flat (top-3 minus bottom-3 gap < 0.1) OR "
          "(B) null reproduces the gradient within noise on >= half the networks")
    print("=" * 100)
    flatA, nullB = [], []
    for net in nets:
        gaps = []
        for spars in ("dspar", "lspar"):
            for t in sorted(set(sel(real, condition=spars).target_ret)):
                gr = sel(real, network=net, condition=spars, target_ret=t)
                gn = sel(null, network=net, condition=spars, target_ret=t)
                if not len(gr):
                    continue
                gaps.append((spars, t, gr.ah_gap.mean(),
                             gn.ah_gap.mean() if len(gn) else np.nan,
                             np.sqrt(gr.ah_gap.std(ddof=1) ** 2 / len(gr)
                                     + (gn.ah_gap.std(ddof=1) ** 2 / len(gn) if len(gn) else 0))))
        if not gaps:
            continue
        mean_gap = float(np.mean([x[2] for x in gaps]))
        flat = mean_gap < 0.1
        flatA.append(flat)
        # (B) "within noise": null gap >= real gap - 2 SE, on the majority of cells
        repro = [x[3] >= x[2] - 2 * x[4] for x in gaps if not np.isnan(x[3])]
        rep = (sum(repro) > len(repro) / 2) if repro else False
        nullB.append(rep)
        jg = sel(real, network=net, condition="jitter").ah_gap.mean()
        print(f"  {net:14s} mean_gap_real={mean_gap:+.3f} (flat<0.1: {flat})  "
              f"null_reproduces={sum(repro)}/{len(repro)} cells -> {rep}   "
              f"[seed-jitter floor gap={jg:+.3f}]")
    print(f"  (A) flat on {sum(flatA)}/{len(flatA)} networks   "
          f"(B) null reproduces on {sum(nullB)}/{len(nullB)} networks "
          f"(kill if >= {len(nullB)/2:.1f})")
    killed = (sum(flatA) == len(flatA)) or (sum(nullB) >= len(nullB) / 2)
    print(f"  ==> KILL CRITERION FIRES: {killed}")

    # ------------------------------------------------------------------ extras
    print("\n" + "=" * 100)
    print("CROSS-CHECKS (coreness / degree gradients vs embeddedness gradient; "
          "seed-jitter floor; chance baseline)")
    print("=" * 100)
    for net in nets:
        for cond, spars, t in (("jitter", np.nan, 1.0), ("dspar", "dspar", 0.5),
                               ("lspar", "lspar", 0.5), ("lspar", "lspar", 0.2),
                               ("perm", "dspar", 0.5)):
            g = real[(real.network == net) & (real.condition == cond)
                     & (real.target_ret == t)]
            if spars == spars:  # not nan
                g = g[g.sparsifier == spars]
            if not len(g):
                continue
            print(f"  {net:14s} {cond:8s} t={t:.2f}  emb_gap={g.ah_gap.mean():+.3f}  "
                  f"core_gap={g.core_ah_gap.mean():+.3f}  deg_gap={g.deg_ah_gap.mean():+.3f}  "
                  f"agree={g.agree_all_hung.mean():.3f} "
                  f"plur_agree={g.agree_all_plur.mean():.3f} "
                  f"plur_gap={g.ap_gap.mean():+.3f}")
        print("-" * 60)

    print("\nDecile profiles (real arm, mean over seeds), Hungarian agreement:")
    for net in nets:
        for cond, t in (("jitter", 1.0), ("dspar", 0.8), ("dspar", 0.5),
                        ("lspar", 0.5), ("lspar", 0.2), ("perm", 0.5)):
            g = real[(real.network == net) & (real.condition == cond)
                     & (real.target_ret == t)]
            if not len(g):
                continue
            v = [g[f"ah_d{i}"].mean() for i in range(10)]
            print(f"  {net:14s} {cond:8s} t={t:.2f} " + " ".join(f"{x:.3f}" for x in v))
        for cond, t in (("dspar", 0.5), ("lspar", 0.2)):
            g = real[(real.network == net) & (real.condition == cond)
                     & (real.target_ret == t)]
            if len(g) and g.n_frag_nodes.mean() >= 20:
                v = [g[f"fs_d{i}"].mean() for i in range(10)]
                print(f"  {net:14s} FRAGSHARE {cond} t={t:.2f} "
                      + " ".join(f"{x:.3f}" for x in v))
        print("-" * 60)


if __name__ == "__main__":
    main()
