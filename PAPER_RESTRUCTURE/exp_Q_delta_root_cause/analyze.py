#!/usr/bin/env python3
"""Exp Q analysis: sanity gates, P1-P4, kill criterion. Reads trajectories.csv
and checkpoints.csv; writes verdicts.csv and prints the full report."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
EXPM = HERE.parent / "exp_M_suppression_probe"
NETS = ["ca-GrQc", "email-Eu-core", "ca-HepTh", "ca-CondMat", "email-Enron"]
EPS = 1e-9


def expM_table():
    dfs = []
    for i, f in enumerate(["probe_results_batch2.csv",
                           "probe_results_batch2_small.csv",
                           "probe_results.csv"]):
        d = pd.read_csv(EXPM / f, keep_default_na=False, na_values=[""])
        d["_pri"] = i
        dfs.append(d)
    d = pd.concat(dfs, ignore_index=True)
    d["rewire_seed"] = d["rewire_seed"].fillna("")
    d = (d.sort_values("_pri")
           .drop_duplicates(["network", "arm", "rewire_seed"], keep="first"))
    out = {}
    for net in NETS:
        s = d[d.network == net]
        out[net] = {
            "delta_real": float(s[s.arm == "real"]["delta"].iloc[0]),
            "r_real": float(s[s.arm == "real"]["r_pb"].iloc[0]),
            "p_real": float(s[s.arm == "real"]["p_intra"].iloc[0]),
            "hubLift_real": float(s[s.arm == "real"]["hub_inter_lift"].iloc[0]),
            "delta_null": float(s[s.arm == "null"]["delta"].mean()),
            "r_null": float(s[s.arm == "null"]["r_pb"].mean()),
            "p_null": float(s[s.arm == "null"]["p_intra"].mean()),
            "k_null": float(s[s.arm == "null"]["n_comms"].mean()),
        }
    return out


def interp(traj, net, chain, col, spe):
    """value of `col` on the neutral chains at matched swap count, per seed."""
    vals = []
    for sd in sorted(traj[(traj.network == net) &
                          (traj.chain == chain)].chain_seed.unique()):
        s = traj[(traj.network == net) & (traj.chain == chain) &
                 (traj.chain_seed == sd)].sort_values("spe")
        if len(s) == 0:
            continue
        vals.append(float(np.interp(spe, s["spe"], s[col])))
    return np.array(vals)


def main():
    traj = pd.read_csv(HERE / "trajectories.csv")
    ckpt = pd.read_csv(HERE / "checkpoints.csv")
    traj = traj.drop_duplicates(["network", "chain", "chain_seed", "accepted"],
                                keep="last")
    ckpt = ckpt.drop_duplicates(["network", "chain", "chain_seed", "accepted"],
                                keep="last")
    traj["tri_ratio"] = np.log((traj.tri_intra + EPS) / (traj.tri_inter + EPS))
    ref = expM_table()
    rows = []

    for variant, (u, d, nname) in {
        "literal": ("up", "down", "neutral"),
        "pfix": ("up_pfix", "down_pfix", "neutral_pfix"),
    }.items():
        print("=" * 100)
        print(f"VARIANT: {variant}   arms {u} / {d} / {nname}")
        print("=" * 100)
        for net in NETS:
            R = ref[net]
            t0 = traj[(traj.network == net) & (traj.spe == 0)]
            if len(t0) == 0:
                print(f"[{net}] NO DATA")
                continue
            d0 = float(t0["delta"].iloc[0])
            print(f"\n### {net}")
            print(f"  GATE2 delta(P0)@swap0 = {d0:.6f}   exp_M delta_real = "
                  f"{R['delta_real']:.6f}   diff = {d0 - R['delta_real']:+.2e}"
                  f"   [{'PASS' if abs(d0 - R['delta_real']) < 1e-9 else 'FAIL'}]")
            print(f"  exp_M delta_null(fresh Leiden on rewired) = "
                  f"{R['delta_null']:.6f}   p_real={R['p_real']:.4f} "
                  f"p_null={R['p_null']:.4f}")

            # neutral endpoint (fluctuation control) + fresh-Leiden gate 1
            nt = traj[(traj.network == net) & (traj.chain == nname) &
                      (traj.final == 1)]
            if len(nt):
                print(f"  NEUTRAL({nname}) end: spe={nt.spe.mean():.3f} "
                      f"delta(P0)={nt.delta.mean():+.5f} "
                      f"p_intra={nt.p_intra.mean():.4f} "
                      f"auc_s={nt.auc_s.mean():.4f}  "
                      f"[frozen-P0 quantity, NOT delta_null]")
            nc = ckpt[(ckpt.network == net) & (ckpt.chain == nname) &
                      (ckpt.final == 1)]
            if len(nc) and variant == "literal":
                print(f"  GATE1 neutral end + FRESH Leiden: "
                      f"delta={nc.fresh_delta.mean():+.5f} "
                      f"(exp_M delta_null {R['delta_null']:+.5f}, "
                      f"rel {(nc.fresh_delta.mean()/R['delta_null']-1)*100:+.1f}%)"
                      f"  k={nc.fresh_n_comms.mean():.0f} "
                      f"(exp_M {R['k_null']:.0f}) "
                      f"p={nc.fresh_p_intra.mean():.4f} "
                      f"(exp_M {R['p_null']:.4f})")

            rec = {"network": net, "variant": variant,
                   "delta_real": R["delta_real"], "delta_null": R["delta_null"],
                   "delta0_check": d0}
            for lab, ch in (("up", u), ("down", d)):
                # endpoint = last checkpoint per seed at which delta(P0) is still
                # DEFINED.  The literal UP chain can drive p_intra to 0 (every
                # edge inter), which makes delta undefined -- that degeneracy is
                # reported separately, it is not an endpoint value.
                cc0 = traj[(traj.network == net) & (traj.chain == ch)]
                if len(cc0) == 0:
                    continue
                fin = cc0[np.isfinite(cc0.delta)]
                e = (fin.sort_values("accepted").groupby("chain_seed").tail(1)
                     if len(fin) else cc0.iloc[0:0])
                degen = int((cc0.final == 1).any() and
                            not np.isfinite(cc0[cc0.final == 1].delta).all())
                rec[f"{lab}_degenerate_p0"] = degen
                if len(e) == 0:
                    continue
                spe_end = float(e.spe.mean())
                de = float(e.delta.mean())
                band = interp(traj, net, nname, "delta", spe_end)
                bmean = float(band.mean()) if band.size else np.nan
                bsd = float(band.std(ddof=1)) if band.size > 1 else np.nan
                sep = (abs(de - bmean) > 2 * bsd) if np.isfinite(bsd) else np.nan
                stalled = int(cc0.stalled.max())
                rec[f"{lab}_spe_stop"] = float(cc0.spe.max())
                rec[f"{lab}_acc_end"] = list(e.accepted)
                print(f"  {lab.upper():4s}({ch}) last-defined point: "
                      f"spe={spe_end:.3f} (chain stopped at "
                      f"spe={cc0.spe.max():.3f}, degenerate_p0={degen}) "
                      f"stalled={stalled} "
                      f"acc_rate_cum={e.acc_rate_cum.mean():.5f} "
                      f"delta(P0)={de:+.5f} r_pb={e.r_pb.mean():+.4f} "
                      f"auc_s={e.auc_s.mean():.4f} p={e.p_intra.mean():.4f} "
                      f"hubLift={e.hub_inter_lift.mean():.4f} "
                      f"Minter/Mtot={ (e.M_inter/e.M_total).mean():.4f}")
                print(f"        neutral band @spe={spe_end:.3f}: "
                      f"{bmean:+.5f} +/- 2sd({2*bsd:.5f}) -> "
                      f"[{bmean-2*bsd:+.5f},{bmean+2*bsd:+.5f}]  "
                      f"separated={sep}")
                rec[f"{lab}_spe_end"] = spe_end
                rec[f"{lab}_stalled"] = stalled
                rec[f"{lab}_delta_end"] = de
                rec[f"{lab}_r_end"] = float(e.r_pb.mean())
                rec[f"{lab}_auc_end"] = float(e.auc_s.mean())
                rec[f"{lab}_p_end"] = float(e.p_intra.mean())
                rec[f"{lab}_hublift_end"] = float(e.hub_inter_lift.mean())
                t0r = traj[(traj.network == net) & (traj.chain == ch) &
                           (traj.spe == 0)]
                ti0 = float(t0r.tri_intra.iloc[0])
                sd0 = float(t0r.sd_s.iloc[0])
                r0 = float(t0r.r_pb.iloc[0])
                q0 = float(t0r.Q.iloc[0])
                rec[f"{lab}_tri_intra_0"] = ti0
                rec[f"{lab}_tri_intra_end"] = float(e.tri_intra.mean())
                rec[f"{lab}_tri_inter_0"] = float(t0r.tri_inter.iloc[0])
                rec[f"{lab}_tri_inter_end"] = float(e.tri_inter.mean())
                rec[f"{lab}_sd_s_drift"] = float(e.sd_s.mean()) / sd0 - 1.0
                rec[f"{lab}_Q_drift"] = float(np.abs(cc0.Q - q0).max())
                rec[f"{lab}_term_scale"] = float(np.log(e.sd_s.mean() / sd0))
                rec[f"{lab}_term_sorting"] = (
                    float(np.log(e.r_pb.mean() / r0))
                    if e.r_pb.mean() * r0 > 0 else np.nan)
                print(f"        tri_intra {ti0:.3f} -> "
                      f"{e.tri_intra.mean():.3f} "
                      f"({(e.tri_intra.mean()/ti0-1)*100:+.1f}%);  "
                      f"tri_inter {rec[f'{lab}_tri_inter_0']:.3f} -> "
                      f"{rec[f'{lab}_tri_inter_end']:.3f};  "
                      f"sd_s drift {rec[f'{lab}_sd_s_drift']*100:+.2f}%;  "
                      f"max |Q-Q0| over chain {rec[f'{lab}_Q_drift']:.2e}")
                rec[f"{lab}_neutral_at_end"] = bmean
                rec[f"{lab}_neutral_2sd"] = 2 * bsd
                rec[f"{lab}_separated"] = sep
                # max excursion over the chain (direction-consistent)
                cc = cc0[np.isfinite(cc0.delta)]
                rec[f"{lab}_delta_max"] = float(cc.delta.max())
                rec[f"{lab}_delta_min"] = float(cc.delta.min())
                # P2: within-chain corr(hub_inter_lift, delta)
                pr = []
                for sd in sorted(cc.chain_seed.unique()):
                    s = cc[cc.chain_seed == sd]
                    if len(s) > 3 and s.hub_inter_lift.std() > 0:
                        pr.append(pearsonr(s.hub_inter_lift, s.delta)[0])
                rec[f"{lab}_P2_corr_hublift_delta"] = (float(np.mean(pr))
                                                       if pr else np.nan)
                print(f"        P2 within-chain corr(hubLift, delta) = "
                      f"{np.mean(pr) if pr else float('nan'):+.4f}  "
                      f"(per seed: {[round(x,3) for x in pr]})")

            # P1 magnitude target: 25% of the way from delta_real to delta_null
            tgt = R["delta_real"] + 0.25 * (R["delta_null"] - R["delta_real"])
            rec["P1_target_delta"] = tgt
            if "up_delta_end" in rec:
                frac = ((rec["up_delta_end"] - R["delta_real"]) /
                        (R["delta_null"] - R["delta_real"]))
                fracmax = ((rec["up_delta_max"] - R["delta_real"]) /
                           (R["delta_null"] - R["delta_real"]))
                rec["P1_frac_of_gap_end"] = frac
                rec["P1_frac_of_gap_max"] = fracmax
                print(f"  P1 magnitude: UP-end delta {rec['up_delta_end']:+.5f} "
                      f"= {frac*100:.1f}% of the delta_real->delta_null gap "
                      f"(target 25%); best point on chain "
                      f"{rec['up_delta_max']:+.5f} = {fracmax*100:.1f}%")
                rec["P1_dir_ok"] = bool(rec["up_delta_end"] > d0 and
                                        rec.get("down_delta_end", np.inf) < d0)
                print(f"  P1 direction (UP>base and DOWN<base): "
                      f"{rec['P1_dir_ok']}")

            # P3: triangle ratio vs delta, pooling up+down, conditioning on direction
            pool = traj[(traj.network == net) & (traj.chain.isin([u, d]))].copy()
            pool = pool[np.isfinite(pool.tri_ratio)]
            if len(pool) > 5:
                dir_ind = (pool.chain == u).astype(float).values
                x = pool.tri_ratio.values
                y = pool.delta.values
                def resid(v):
                    A = np.column_stack([np.ones_like(dir_ind), dir_ind])
                    b = np.linalg.lstsq(A, v, rcond=None)[0]
                    return v - A @ b
                rx, ry = resid(x), resid(y)
                pc = (pearsonr(rx, ry)[0] if rx.std() > 0 and ry.std() > 0
                      else np.nan)
                rec["P3_partial_corr_triratio_delta"] = pc
                print(f"  P3 partial corr(log tri_ratio, delta | direction) = "
                      f"{pc:+.4f}  (|.|<0.4 required)   n={len(pool)}")

            # P4: fresh-Leiden delta at UP-chain ends
            fu = ckpt[(ckpt.network == net) & (ckpt.chain == u) &
                      (ckpt.accepted.isin(rec.get("up_acc_end", [])))]
            if len(fu):
                rec["P4_fresh_delta_up_end"] = float(fu.fresh_delta.mean())
                rec["P4_ami_up_end"] = float(fu.ami_P0.mean())
                # best fresh delta anywhere on the up chain
                fall = ckpt[(ckpt.network == net) & (ckpt.chain == u)]
                rec["P4_fresh_delta_up_max"] = float(fall.fresh_delta.max())
                print(f"  P4 fresh-Leiden delta at UP end = "
                      f"{rec['P4_fresh_delta_up_end']:+.5f} vs delta_real "
                      f"{R['delta_real']:+.5f}  "
                      f"[{'PASS' if rec['P4_fresh_delta_up_end'] > R['delta_real'] else 'fail'}]"
                      f"  AMI(P0)={rec['P4_ami_up_end']:.4f}; "
                      f"max over chain {rec['P4_fresh_delta_up_max']:+.5f}")
            fd = ckpt[(ckpt.network == net) & (ckpt.chain == d) &
                      (ckpt.accepted.isin(rec.get("down_acc_end", [])))]
            if len(fd):
                rec["P4_fresh_delta_down_end"] = float(fd.fresh_delta.mean())
                rec["P4_ami_down_end"] = float(fd.ami_P0.mean())
                print(f"     fresh-Leiden delta at DOWN end = "
                      f"{rec['P4_fresh_delta_down_end']:+.5f}  "
                      f"AMI(P0)={rec['P4_ami_down_end']:.4f}")

            # registered untestable rule
            unt = []
            for lab in ("up", "down"):
                if rec.get(f"{lab}_stalled") == 1 and rec.get(f"{lab}_spe_stop", 9) < 2.0:
                    unt.append(lab)
            rec["untestable_letter"] = ",".join(unt)
            if unt:
                print(f"  REGISTERED UNTESTABLE RULE fires (stall < 2 swaps/edge): "
                      f"{unt}")
            rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.drop(columns=[c for c in out.columns if c.endswith("_acc_end")])
    out.to_csv(HERE / "verdicts.csv", index=False)
    print("\nwrote verdicts.csv")

    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PREDICTION TALLIES")
    print("=" * 100)
    for variant in ("literal", "pfix"):
        s = out[out.variant == variant]
        if len(s) == 0:
            continue
        n = len(s)
        print(f"\n[{variant}]  networks with data: {n}")
        print(f"  P1 direction (UP>base, DOWN<base): "
              f"{int(s.P1_dir_ok.sum())}/{n}")
        okmag = (s.P1_frac_of_gap_end >= 0.25).sum()
        okmagmax = (s.P1_frac_of_gap_max >= 0.25).sum()
        print(f"  P1 magnitude >=25% of gap at chain end: {okmag}/{n}; "
              f"at best point on chain: {okmagmax}/{n}")
        print(f"  P1 (letter, excluding networks flagged UNTESTABLE): "
              f"testable = {int((s.untestable_letter=='').sum())}/{n}")
        for lab in ("up", "down"):
            print(f"  P2 mean within-chain corr(hubLift,delta) {lab}: "
                  f"{s[f'{lab}_P2_corr_hublift_delta'].mean():+.4f}; "
                  f">0.8 on {(s[f'{lab}_P2_corr_hublift_delta']>0.8).sum()}/{n}")
        print(f"  P3 |partial corr| < 0.4: "
              f"{(s.P3_partial_corr_triratio_delta.abs()<0.4).sum()}/{n} "
              f"(values {[round(v,3) for v in s.P3_partial_corr_triratio_delta]})")
        print(f"  P4 fresh delta(UP end) > delta_real: "
              f"{(s.P4_fresh_delta_up_end > s.delta_real).sum()}/{n}")
        print(f"  KILL: UP and DOWN both separated from neutral +/-2sd: "
              f"{int(((s.up_separated==True)&(s.down_separated==True)).sum())}/{n}")


if __name__ == "__main__":
    sys.exit(main())
