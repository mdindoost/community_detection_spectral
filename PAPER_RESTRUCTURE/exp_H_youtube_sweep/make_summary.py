#!/usr/bin/env python
"""Build SUMMARY.md from raw/*.json (see run.py for the protocol)."""

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from run import analyze, best_of_k_distribution  # noqa: E402

EXP_E = dict(Q_base_mean=0.7245063774934801, Q_base_std=0.003436901864147399,
             Q_seeded_mean=0.7295705203479402, Q_matched_best=0.7209317665597437,
             y=0.008638753788196563)


def fmt(x, p=6):
    return f"{x:.{p}f}"


def main():
    stats, configs, Qb, bo2, bo5, bo10, base, seeded = analyze()
    sd = stats["Q_base_std"]
    L = []
    A = L.append

    A("# Experiment H — com-Youtube: is the seeded-refinement gain real?")
    A("")
    A("Referee finding MC3 (`PAPER_RESTRUCTURE/phase3_review/JUDGE_RULING.md`) flagged the")
    A("com-Youtube result from `exp_E_delta_star` as fragile: **one** configuration")
    A("(calibrated DSpar, alpha=0.9, 3 sparsification seeds), and a runtime-matched")
    A("baseline that was a **single realization** of a best-of-2 draw which landed")
    A(f"*below* the plain-Leiden mean (Q_matched = {fmt(EXP_E['Q_matched_best'])} vs")
    A(f"Q_base_mean = {fmt(EXP_E['Q_base_mean'])}). This experiment replaces that single")
    A("draw with the **exact distribution** of best-of-2 and sweeps both samplers x 4")
    A("alphas x 5 sparsification seeds.")
    A("")
    A("Protocol, seeds, loader, samplers, seeded refinement and Leiden calls are reused")
    A("verbatim from `exp_E_delta_star/run.py` and `exp_C_true_retention_seeded/run.py`")
    A("(base seeds 100.., sparsify seeds 200.., Leiden-on-sparse seeds 300..,")
    A("`la.ModularityVertexPartition` + `Optimiser.optimise_partition`, n_iterations=2).")
    A(f"Graph: com-Youtube LCC, n={base[0]['n']:,}, m={base[0]['m']:,}.")
    A("")

    # ---- 1. baseline distribution -----------------------------------------
    A("## 1. Baseline distribution (10 plain Leiden runs, seeds 100-109)")
    A("")
    A("| seed | Q_base | k | T (s) |")
    A("|---|---|---|---|")
    for r in base:
        A(f"| {r['seed']} | {fmt(r['Q'])} | {r['k']} | {r['T']:.1f} |")
    A("")
    A(f"- mean   = **{fmt(stats['Q_base_mean'])}**")
    A(f"- std    = **{fmt(sd)}** (sample, ddof=1); {fmt(stats['Q_base_std_pop'])} (population)")
    A(f"- best   = **{fmt(stats['Q_base_best'])}**   min = {fmt(stats['Q_base_min'])}")
    A(f"- range  = {fmt(stats['Q_base_best'] - stats['Q_base_min'])}")
    A(f"- T_leiden median = {stats['T_leiden_median']:.1f} s "
      f"(4 workers ran concurrently, so wall times are inflated ~1.3-2x vs a solo run;")
    A("  only the *ratio* T_pipe / T_leiden is used, and both are inflated equally)")
    A("")
    A(f"exp_E's 5-seed estimate was mean {fmt(EXP_E['Q_base_mean'])} +- "
      f"{fmt(EXP_E['Q_base_std'])} (population std); the 10-seed estimate here is")
    A(f"mean {fmt(stats['Q_base_mean'])} +- {fmt(stats['Q_base_std_pop'])} (population std).")
    A("")

    # ---- 2. matched-baseline distribution ---------------------------------
    A("## 2. Runtime-matched baseline as a distribution, not a draw")
    A("")
    A("T_pipe / T_leiden is 1.5-1.7x for every configuration below, and exp_E measured")
    A("`n_matched_restarts = 2` on com-Youtube, so the honest runtime-matched baseline")
    A("is **best-of-2 plain Leiden** (which slightly over-credits the baseline, since a")
    A("strict match buys only ~1.5-1.7 restarts). Its exact distribution over all C(10,2)=45")
    A("unordered pairs of the 10 baseline runs:")
    A("")
    A(f"| statistic | best-of-2 (45 pairs) | best-of-5 (252 subsets) | best-of-10 |")
    A("|---|---|---|---|")
    A(f"| mean | **{fmt(stats['bo2_mean'])}** | {fmt(stats['bo5_mean'])} | "
      f"{fmt(stats['bo10_mean'])} |")
    A(f"| std | {fmt(stats['bo2_std'])} | {fmt(stats['bo5_std'])} | 0 |")
    A(f"| min | {fmt(stats['bo2_min'])} | {fmt(bo5.min())} | — |")
    A(f"| max | {fmt(stats['bo2_max'])} | {fmt(bo5.max())} | — |")
    A("")
    A(f"exp_E's single matched draw was {fmt(EXP_E['Q_matched_best'])}. Its percentile in")
    A(f"this best-of-2 distribution: **{(bo2 <= EXP_E['Q_matched_best']).mean()*100:.1f}%**")
    A(f"(i.e. {(bo2 > EXP_E['Q_matched_best']).mean()*100:.1f}% of runtime-matched baselines")
    A(f"beat it), and it sits {(EXP_E['Q_matched_best'] - stats['bo2_mean'])/sd:+.2f} sigma")
    A("from the best-of-2 mean. MC3's diagnosis is confirmed: the exp_E baseline was a")
    A("low draw, and the +0.0086 headline gain is inflated by that draw.")
    A("")

    # ---- 3+4. sweep --------------------------------------------------------
    A("## 3-4. Seeded sweep and per-configuration verdict")
    A("")
    A("`gain` columns are `Q_seeded_mean - <baseline>`; `z` columns divide by")
    A(f"Q_base_std = {fmt(sd)}. `P(bo2>=)` is the empirical fraction of the 45")
    A("runtime-matched best-of-2 draws that meet or beat the seeded mean.")
    A("")
    A("| sampler | alpha | true ret. | Q_raw | Q_seeded mean | +-std | best | T_pipe (s) | restarts | gain vs base mean | z | gain vs E[bo2] | **z vs bo2** | P(bo2>=) | gain vs E[bo5] | z vs bo5 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in configs:
        A("| {s} | {a} | {ret:.4f} | {qr} | {qs} | {qsd} | {qb} | {tp:.0f} | {rs:.1f}x | "
          "{g0:+.6f} | {z0:+.2f} | {g2:+.6f} | **{z2:+.2f}** | {p2:.2f} | {g5:+.6f} | {z5:+.2f} |".format(
              s=c["sampler"], a=c["alpha"], ret=c["retention"],
              qr=fmt(c["Q_raw_mean"]), qs=fmt(c["Q_seeded_mean"]),
              qsd=fmt(c["Q_seeded_std"]), qb=fmt(c["Q_seeded_best"]),
              tp=c["T_pipe_mean"], rs=c["restarts_equiv"],
              g0=c["gain_vs_base_mean"], z0=c["z_vs_base"],
              g2=c["gain_vs_bo2_mean"], z2=c["z_vs_bo2"], p2=c["p_bo2_ge"],
              g5=c["gain_vs_bo5_mean"], z5=c["z_vs_bo5"]))
    A("")
    n2 = sum(1 for c in configs if c["z_vs_bo2"] > 2)
    n0 = sum(1 for c in configs if c["z_vs_bo2"] > 0)
    n5 = sum(1 for c in configs if c["z_vs_bo5"] > 0)
    n5_2 = sum(1 for c in configs if c["z_vs_bo5"] > 2)
    nbest = sum(1 for c in configs if c["Q_seeded_mean"] > stats["Q_base_best"])
    A(f"- configurations with gain > 2 sigma vs the FAIR (expected) best-of-2 baseline: "
      f"**{n2} / {len(configs)}**")
    A(f"- configurations with any positive gain vs E[best-of-2]: {n0} / {len(configs)}")
    A(f"- configurations with gain > 2 sigma vs E[best-of-5]: {n5_2} / {len(configs)}; "
      f"positive: {n5} / {len(configs)}")
    A(f"- configurations whose seeded mean exceeds the best of all 10 plain runs: "
      f"{nbest} / {len(configs)}")
    A("")

    # ---- 4b. sharper test on the MEANS -------------------------------------
    A("### 4b. A sharper test (the z columns above are deliberately conservative)")
    A("")
    A("Dividing by Q_base_std compares a *mean of 5 seeded runs* against the spread of a")
    A("*single* run, so it under-states significance. Two extra tests:")
    A("")
    A("* **Head-to-head win rate**: over all 5 x 45 = 225 (seeded run, best-of-2 draw)")
    A("  pairs, the fraction where the seeded run wins.")
    A("* **Bootstrap on the means**: resample the 10 baseline runs with replacement")
    A("  (B=20000) to get SE(E[best-of-2]); combine with SE of the seeded mean")
    A("  (std/sqrt(5)) to get z_mean = gain / sqrt(SE_bo2^2 + SE_seeded^2), and a")
    A("  bootstrap p-value P(E[best-of-2]* >= Q_seeded_mean).")
    A("")
    rng = np.random.default_rng(0)
    B = 20000
    boot = np.array([best_of_k_distribution(rng.choice(Qb, size=10, replace=True), 2).mean()
                     for _ in range(B)])
    se_bo2 = float(boot.std(ddof=1))
    A(f"SE(E[best-of-2]) from bootstrap = {se_bo2:.6f}")
    A("")
    A("| sampler | alpha | gain vs E[bo2] | head-to-head win rate | SE_seeded | z_mean | bootstrap p |")
    A("|---|---|---|---|---|---|---|")
    for c in configs:
        rr = [r for r in seeded if r["sampler"] == c["sampler"] and r["alpha"] == c["alpha"]]
        qs = np.array([r["Q_seeded"] for r in rr])
        wr = float((qs[:, None] > bo2[None, :]).mean())
        se_s = float(qs.std(ddof=1) / np.sqrt(len(qs)))
        zm = c["gain_vs_bo2_mean"] / np.sqrt(se_bo2 ** 2 + se_s ** 2)
        pb = float((boot >= c["Q_seeded_mean"]).mean())
        c["headtohead_winrate"] = wr
        c["z_mean_bootstrap"] = float(zm)
        c["p_bootstrap"] = pb
        A(f"| {c['sampler']} | {c['alpha']} | {c['gain_vs_bo2_mean']:+.6f} | {wr:.3f} | "
          f"{se_s:.6f} | **{zm:+.2f}** | {pb:.4f} |")
    A("")
    nz2 = sum(1 for c in configs if c["z_mean_bootstrap"] > 2)
    A(f"- configurations with z_mean > 2 on this sharper test: **{nz2} / {len(configs)}** "
      f"(before any multiplicity correction; 8 configurations were tested, so a "
      f"Bonferroni-corrected two-sided 5% threshold is |z| > 2.73)")
    nz273 = sum(1 for c in configs if c["z_mean_bootstrap"] > 2.73)
    A(f"- configurations surviving Bonferroni (z_mean > 2.73): **{nz273} / {len(configs)}**")
    A("")
    A("Budget note: T_pipe / T_leiden is 1.5-1.7x, so a strictly runtime-matched baseline")
    A("buys only ~1.5-1.7 restarts. Charging the baseline a full best-of-2 is *generous to")
    A("the baseline* (conservative for the seeded method); exp_E did the same.")
    A("")

    # per-run detail
    A("### 4c. Per-run seeded results")
    A("")
    A("| sampler | alpha | spar seed | retention | Q_raw | Q_seeded | k | T_pipe (s) |")
    A("|---|---|---|---|---|---|---|---|")
    for r in sorted(seeded, key=lambda r: (r["sampler"], r["alpha"], r["spar_seed"])):
        A(f"| {r['sampler']} | {r['alpha']} | {r['spar_seed']} | {r['retention']:.4f} | "
          f"{fmt(r['Q_raw'])} | {fmt(r['Q_seeded'])} | {r['k_seeded']} | {r['T_pipe']:.0f} |")
    A("")

    # ---- 5. verdict --------------------------------------------------------
    best = max(configs, key=lambda c: c["z_mean_bootstrap"])
    A("## 5. Verdict")
    A("")
    A(f"**com-Youtube is a weak, narrow, alpha-dependent gain -- not Enron-class, but not")
    A("pure noise either.**")
    A("")
    A(f"1. MC3 is confirmed. exp_E's matched baseline ({fmt(EXP_E['Q_matched_best'])}) is beaten")
    A(f"   by {(bo2 > EXP_E['Q_matched_best']).mean()*100:.0f}% of the 45 possible best-of-2")
    A(f"   draws. Against the FAIR baseline E[best-of-2] = {fmt(stats['bo2_mean'])}, the")
    A("   same configuration (calibrated, alpha=0.9) gives")
    A(f"   {[c for c in configs if c['sampler']=='calibrated' and c['alpha']==0.9][0]['gain_vs_bo2_mean']:+.6f},")
    A(f"   not +{EXP_E['y']:.4f}. **The headline gain shrinks by ~3.8x.**")
    A(f"2. On the requested metric (gain / Q_base_std), **{n2} of {len(configs)}** configurations")
    A("   clear 2 sigma. On the sharper mean-vs-mean bootstrap, 2 of 8 clear z=2 and only")
    A(f"   1 of 8 (calibrated alpha=0.95, z={best['z_mean_bootstrap']:+.2f}) survives Bonferroni")
    A("   correction for the 8 configurations tested.")
    A("3. The effect is **entirely confined to the calibrated sampler at high alpha**.")
    A("   All four repo_noreplace configurations (true retention 0.48-0.53) are")
    A("   *negative* vs E[best-of-2], and calibrated alpha=0.7 is strongly negative")
    A("   (-3.85 sigma). Only alpha in {0.9, 0.95} is positive, and the gain is monotone")
    A("   increasing in alpha -- i.e. it grows as the sparsifier does less. Extrapolating,")
    A("   alpha -> 1 (no sparsification at all) would be best, which is the signature of a")
    A("   *seeding/refinement* artifact rather than a sparsification benefit.")
    A("4. Effect size: the largest honest gain is")
    A(f"   {best['gain_vs_bo2_mean']:+.6f} at calibrated alpha={best['alpha']}, about")
    A(f"   {best['gain_vs_bo2_mean']/sd:.2f} single-run sigma, for a {best['restarts_equiv']:.1f}x")
    A("   runtime cost. Contrast email-Enron (exp_C), where **all 8 of 8** configurations,")
    A("   both samplers, alpha 0.7-0.95, gave seeded-minus-base +0.0057 to +0.0130 on a")
    A("   Q_base_std of 0.00187 (z = 3-7).")
    A("")
    A("So: com-Youtube is **not** an Enron-class robust gain -- Enron is robust across the")
    A("whole sampler x alpha grid, com-Youtube is positive in 2 of 8 cells and negative in 6.")
    A("It is best described as a **weak-but-real gain in a narrow high-alpha corner**: at")
    A("calibrated alpha=0.95 the seeded mean beats every one of the 45 runtime-matched")
    A("best-of-2 draws and every one of the 10 individual plain runs (head-to-head win rate")
    A("1.000, bootstrap p < 1e-4), which is not something noise produces. But it is roughly")
    A("a third of the size exp_E reported, it does not survive as a general claim about the")
    A("sparsifier, and the +0.0086 / 2.5 sigma figure should not be used.")
    A("")

    json.dump(dict(n_gt2sigma_bo2=n2, n_pos_bo2=n0), open(HERE / "verdict.json", "w"))
    (HERE / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
