#!/usr/bin/env python
"""Build SUMMARY.md from raw/*.json (see run.py for the protocol)."""

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from run import analyze  # noqa: E402

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
    A("T_pipe / T_leiden is ~1.8-2.0 for every configuration below, and exp_E measured")
    A("`n_matched_restarts = 2` on com-Youtube, so the honest runtime-matched baseline")
    A("is **best-of-2 plain Leiden**. Its exact distribution over all C(10,2)=45")
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

    # per-run detail
    A("### Per-run seeded results")
    A("")
    A("| sampler | alpha | spar seed | retention | Q_raw | Q_seeded | k | T_pipe (s) |")
    A("|---|---|---|---|---|---|---|---|")
    for r in sorted(seeded, key=lambda r: (r["sampler"], r["alpha"], r["spar_seed"])):
        A(f"| {r['sampler']} | {r['alpha']} | {r['spar_seed']} | {r['retention']:.4f} | "
          f"{fmt(r['Q_raw'])} | {fmt(r['Q_seeded'])} | {r['k_seeded']} | {r['T_pipe']:.0f} |")
    A("")

    json.dump(dict(n_gt2sigma_bo2=n2, n_pos_bo2=n0), open(HERE / "verdict.json", "w"))
    (HERE / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
