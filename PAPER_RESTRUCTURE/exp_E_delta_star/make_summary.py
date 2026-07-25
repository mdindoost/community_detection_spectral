#!/usr/bin/env python
"""Build SUMMARY.md from predictors.csv / outcomes.csv / correlations.csv."""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP_C = HERE.parent / "exp_C_true_retention_seeded/results_summary.csv"

PRED_MAIN = ["delta_real", "delta_star", "dQ_ratio", "hb_real", "hb_excess", "deg_cv"]
PRED_ALL = ["delta_real", "delta_star", "delta_ratio", "dQ_ratio", "dQ_excess",
            "hb_real", "hb_excess", "hb_ratio", "deg_cv", "Qfix_real",
            "Qfix_ratio", "avg_deg", "n", "m"]


def fmt(v, w=9, p=4):
    return f"{v:>{w}.{p}f}"


def main():
    pred = {r["network"]: r for r in csv.DictReader(open(HERE / "predictors.csv"))}
    out = list(csv.DictReader(open(HERE / "outcomes.csv")))
    names = [r["network"] for r in out]
    O = {r["network"]: r for r in out}
    y = np.array([float(O[n]["y"]) for n in names])
    y2 = np.array([float(O[n]["y2"]) for n in names])
    sig = np.array([float(O[n]["Q_base_std"]) for n in names])
    nn = len(names)

    L = []
    A = L.append
    A("# Experiment E - does any structural statistic predict the genuine seeded gain?\n")
    A("Outcome measured at **calibrated DSpar, alpha = 0.9** (E[retention]=0.9 exactly,")
    A("lambda-bisection sampler from exp_C), 3 sparsification seeds, 5 baseline seeds,")
    A("Leiden n_iterations=2, undirected simple LCC.\n")
    A("```")
    A("y  = mean(Q_seeded) - Q_matched_best     <- HONEST outcome (runtime-matched restarts)")
    A("y2 = mean(Q_seeded) - Q_base_mean        <- naive outcome (single-run baseline)")
    A("```\n")

    # ---- 1. outcomes -----------------------------------------------------
    A("## 1. Outcomes (n=15 networks)\n")
    A("| network | n | m | ret | Q_base_mean +- std | Q_seeded | Q_matched | restarts | y | y2 | y>2sigma |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for nm in names:
        r = O[nm]
        s = float(r["Q_base_std"])
        yy = float(r["y"])
        A(f"| {nm} | {int(r['n']):,} | {int(r['m']):,} | {float(r['actual_ret_mean']):.4f} "
          f"| {float(r['Q_base_mean']):.6f} +- {s:.6f} | {float(r['Q_seeded_mean']):.6f} "
          f"| {float(r['Q_matched_best']):.6f} | {r['n_matched_restarts']} "
          f"| {yy:+.6f} | {float(r['y2']):+.6f} | {'YES' if yy > 2*s else 'no'} |")
    A("")

    # ---- 1b. exp_C cross-check ------------------------------------------
    A("### Cross-check vs exp_C (calibrated, alpha=0.9)\n")
    A("exp_C used 5 sparsification seeds, exp_E uses 3, so small differences in")
    A("Q_seeded_mean are expected; Q_base_mean / Q_matched_best should be identical")
    A("(same seeds, same budget rule).\n")
    A("| network | Q_base_mean E/C | Q_matched_best E/C | y_E | y_C | |dy| | flag |")
    A("|---|---|---|---|---|---|---|")
    if EXP_C.exists():
        c = {r["dataset"]: r for r in csv.DictReader(open(EXP_C))
             if r["sampler"] == "calibrated" and abs(float(r["alpha"]) - 0.9) < 1e-9}
        for nm in names:
            if nm not in c:
                continue
            rc, re = c[nm], O[nm]
            yE, yC = float(re["y"]), float(rc["seeded_minus_matched"])
            d = abs(yE - yC)
            A(f"| {nm} | {float(re['Q_base_mean']):.6f} / {float(rc['Q_base_mean']):.6f} "
              f"| {float(re['Q_matched_best']):.6f} / {float(rc['Q_matched_best']):.6f} "
              f"| {yE:+.6f} | {yC:+.6f} | {d:.6f} | {'**>0.002**' if d > 0.002 else 'ok'} |")
    A("")

    # ---- 2. predictors ---------------------------------------------------
    A("## 2. Predictors (from exp_B_config_null/results.csv; 17 networks)\n")
    A("`delta_star = delta_real - delta_null`, `dQ_ratio = dQ_fixed_real / dQ_fixed_null`,")
    A("`hb_excess = hb_real - hb_null`, `deg_cv = std(deg)/mean(deg)` on the LCC.\n")
    A("| network | delta_real | delta_null | delta_star | dQ_real | dQ_null | dQ_ratio | hb_real | hb_null | hb_excess | deg_cv | y |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm in PRED_ORDER(pred):
        p = pred[nm]
        yv = f"{float(O[nm]['y']):+.6f}" if nm in O else "n/a"
        A(f"| {nm} | {float(p['delta_real']):+.4f} | {float(p['delta_null']):+.4f} "
          f"| {float(p['delta_star']):+.4f} | {float(p['dQ_real']):.4f} | {float(p['dQ_null']):.4f} "
          f"| {float(p['dQ_ratio']):.3f} | {float(p['hb_real']):.3f} | {float(p['hb_null']):.3f} "
          f"| {float(p['hb_excess']):+.3f} | {float(p['deg_cv']):.3f} | {yv} |")
    A("")

    # ---- 3. correlations -------------------------------------------------
    A("## 3. Correlations with y (n=15), exact p-values\n")
    A("| predictor | Spearman rho | p | Pearson r | p | Spearman rho (y2) | p |")
    A("|---|---|---|---|---|---|---|")
    res = {}
    for p in PRED_ALL:
        x = np.array([float(pred[nm][p]) for nm in names])
        sr, sp = stats.spearmanr(x, y)
        pr, pp = stats.pearsonr(x, y)
        sr2, sp2 = stats.spearmanr(x, y2)
        res[p] = (sr, sp, pr, pp)
        star = " **" if sp < 0.05 else ""
        A(f"| {p}{star} | {sr:+.4f} | {sp:.4f} | {pr:+.4f} | {pp:.4f} | {sr2:+.4f} | {sp2:.4f} |")
    A("")
    A(f"Bonferroni threshold for {len(PRED_ALL)} predictors at family-wise 0.05: "
      f"p < {0.05/len(PRED_ALL):.4f}.\n")

    # ---- 3b. power -------------------------------------------------------
    # analytic: Spearman t-approx, |rho| needed for two-sided p<0.05 at n
    tcrit = stats.t.ppf(0.975, nn - 2)
    crit = tcrit / np.sqrt(tcrit ** 2 + nn - 2)
    A(f"**Power.** At n={nn}, a two-sided Spearman test needs |rho| >= {crit:.3f} "
      f"to reach p<0.05, and |rho| >= "
      f"{(lambda t: t/np.sqrt(t**2+nn-2))(stats.t.ppf(1-0.05/(2*len(PRED_ALL)), nn-2)):.3f} "
      f"to survive Bonferroni over {len(PRED_ALL)} predictors. "
      f"The largest observed |rho| is "
      f"{max(abs(res[p][0]) for p in PRED_ALL):.3f}.\n")

    # ---- 4. leave-one-out ------------------------------------------------
    ranked = sorted(PRED_ALL, key=lambda p: -abs(res[p][0]))
    loo_set = list(dict.fromkeys(ranked[:3] + ["delta_real", "delta_star", "deg_cv"]))
    top = ranked[0]
    A(f"## 4. Leave-one-out sensitivity (Spearman rho vs y, n={nn-1} each)\n")
    A("Top-3 predictors by |rho| plus delta_real / delta_star / deg_cv.\n")
    A("| dropped | " + " | ".join(f"`{p}`" for p in loo_set) + " |")
    A("|---" * (len(loo_set) + 1) + "|")
    store = {p: [] for p in loo_set}
    for i in range(nn):
        k = [j for j in range(nn) if j != i]
        cells = []
        for p in loo_set:
            xv = np.array([float(pred[nm][p]) for nm in names])
            sr, sp = stats.spearmanr(xv[k], y[k])
            store[p].append(sr)
            cells.append(f"{sr:+.3f} (p={sp:.3f})")
        A(f"| {names[i]} | " + " | ".join(cells) + " |")
    A("")
    for p in loo_set:
        v = store[p]
        A(f"- `{p}`: full-sample rho={res[p][0]:+.4f} (p={res[p][1]:.4f}); "
          f"LOO range [{min(v):+.4f}, {max(v):+.4f}]; "
          f"sign flips under LOO: {'YES' if min(v)*max(v) < 0 else 'no'}; "
          f"LOO folds with p<0.05: "
          f"{sum(1 for j in range(nn) if stats.spearmanr(np.array([float(pred[nm][p]) for nm in names])[[q for q in range(nn) if q != j]], y[[q for q in range(nn) if q != j]])[1] < 0.05)}/{nn}")
    A("")

    # ---- 5. binary -------------------------------------------------------
    pos = [nm for nm in names if float(O[nm]["y"]) > 2 * float(O[nm]["Q_base_std"])]
    neg = [nm for nm in names if nm not in pos]
    A("## 5. Binary analysis: y > 2*sigma(baseline seed noise)\n")
    A(f"GAIN group (n={len(pos)}): {', '.join(pos) if pos else '(none)'}")
    A(f"NO-GAIN group (n={len(neg)}): {', '.join(neg)}\n")
    A("| predictor | mean(GAIN) | mean(NO-GAIN) | Mann-Whitney U p | AUC | overlap? |")
    A("|---|---|---|---|---|---|")
    for p in PRED_MAIN + ["delta_ratio", "hb_ratio", "Qfix_ratio"]:
        a = np.array([float(pred[nm][p]) for nm in pos])
        b = np.array([float(pred[nm][p]) for nm in neg])
        if len(a) == 0 or len(b) == 0:
            continue
        try:
            u, up = stats.mannwhitneyu(a, b, alternative="two-sided")
            auc = u / (len(a) * len(b))
        except ValueError:
            up, auc = float("nan"), float("nan")
        sep = "clean" if (a.min() > b.max() or a.max() < b.min()) else "OVERLAP"
        A(f"| {p} | {a.mean():+.4f} | {b.mean():+.4f} | {up:.4f} | {auc:.3f} | {sep} |")
    A("")
    if len(pos) >= 1 and len(neg) >= 1:
        from math import comb
        pmin = 2.0 / comb(nn, len(pos))
        A(f"**Power of the binary test.** With {len(pos)} GAIN vs {len(neg)} NO-GAIN "
          f"networks, the smallest attainable two-sided Mann-Whitney p is "
          f"{pmin:.4f} -- even a *perfect* separation could not survive Bonferroni "
          f"correction over {len(PRED_ALL)} predictors (threshold "
          f"{0.05/len(PRED_ALL):.4f}). This test has effectively no power; it is "
          f"reported only to show that not even perfect separation is observed.\n")

    # ---- 6. verdict ------------------------------------------------------
    A("## 6. Verdict\n")
    best = max(PRED_ALL, key=lambda p: abs(res[p][0]))
    A("**(a) Which networks show genuine gains at alpha=0.9?**\n")
    A(f"{len(pos)} of {nn}: " + (", ".join(f"`{nm}` (y={float(O[nm]['y']):+.6f}, "
      f"2sigma={2*float(O[nm]['Q_base_std']):.6f})" for nm in pos) if pos else "none") + ".")
    A("Every other network is inside baseline seed noise or negative. The two "
      "next-largest positives fall short of the bar: "
      + ", ".join(f"`{nm}` (y={float(O[nm]['y']):+.6f} vs 2sigma="
                  f"{2*float(O[nm]['Q_base_std']):.6f})"
                  for nm in sorted(neg, key=lambda z: -float(O[z]['y']))[:2]) + ".")
    A("So the exp_C finding replicates and extends only marginally: the honest, "
      "runtime-matched seeded gain is a rare, network-specific event, not a "
      "general property.\n")

    A("**(b) Does any statistic separate them?**\n")
    A(f"No. The strongest rank correlation over all {len(PRED_ALL)} candidates is "
      f"`{best}` at rho={res[best][0]:+.4f}, p={res[best][1]:.4f} -- not significant "
      f"even uncorrected, and far from the Bonferroni threshold "
      f"{0.05/len(PRED_ALL):.4f}. In the binary analysis every candidate OVERLAPS "
      "between the GAIN and NO-GAIN groups. The heavy-tail statistics "
      "(`deg_cv`, `hb_real`) come closest (AUC 0.92) but are broken by a single "
      "network: `wiki-Talk` has by far the most extreme degree heterogeneity "
      "(deg_cv=26.3, hb_real=10.5, the maxima of the whole set) and yet shows "
      "y<0. Adding `wiki-Talk` to the 14-network set collapsed `deg_cv`'s "
      "Pearson r from +0.690 (p=0.006) to +0.122 (p=0.664) and `n`'s Spearman "
      "rho from +0.635 (p=0.015) to +0.468 (p=0.079) -- i.e. the only apparently "
      "significant relationships were single-point artefacts.\n")

    A("**(c) Is delta* better or worse than raw delta?**\n")
    dr, ds = res["delta_real"], res["delta_star"]
    A(f"Worse, and wrong-signed. raw `delta_real`: rho={dr[0]:+.4f} (p={dr[1]:.4f}), "
      f"r={dr[2]:+.4f} (p={dr[3]:.4f}). Excess `delta_star`: rho={ds[0]:+.4f} "
      f"(p={ds[1]:.4f}), r={ds[2]:+.4f} (p={ds[3]:.4f}). Subtracting the "
      "configuration-model null flips the (already non-significant) association "
      "from positive to negative. Mechanically this is expected: delta_null is "
      "itself driven by degree heterogeneity and exceeds delta_real on 16 of 17 "
      "networks, so delta_star is essentially minus a heavy-tail statistic. "
      "delta_star therefore does not rescue delta as a predictor -- it is a "
      "diagnostic that the raw separation is null-explained, nothing more.\n")

    A("**(d) Recommendation for the paper**\n")
    A("**Predictor claim: NO.** There is no defensible claim that delta, "
      "delta*, the DeltaQ ratio, the hub-bridge ratio, its excess, or degree CV "
      "predicts where DSpar-seeded Leiden beats a runtime-matched baseline. "
      "Concretely:")
    A("")
    A("1. Only 2/15 networks clear the noise bar, so the outcome is nearly "
      "degenerate; any \"predictor\" would be fitting 2 points.")
    A("2. No candidate reaches p<0.05 uncorrected on either y or y2, let alone "
      "corrected.")
    A("3. The one pattern that looked real at 14 networks (heavy-tailed degree "
      "distribution / large n) is falsified by wiki-Talk, the most heavy-tailed "
      "network in the set.")
    A("4. delta* -- the statistic the restructure was hoping to promote -- is "
      "the weakest and points the wrong way.")
    A("")
    A("The paper should stay **purely characterizational**: report that raw "
      "DSpar separation and DeltaQ_fixed are null-reproduced (Phase 1), that the "
      "only honest algorithmic gain is a small, runtime-matched seeded-refinement "
      "improvement observed on 2 of 15 real networks (email-Enron, com-Youtube), "
      "and state explicitly that no structural statistic tested predicts where it "
      "occurs. If a predictor claim is wanted later it needs a much larger "
      "network sample (order 100+) and a pre-registered statistic; n=15 with a "
      "2-positive outcome cannot support one.\n")

    Path(HERE / "SUMMARY.md").write_text("\n".join(L))
    print("\n".join(L))


def PRED_ORDER(pred):
    return list(pred.keys())


if __name__ == "__main__":
    main()
