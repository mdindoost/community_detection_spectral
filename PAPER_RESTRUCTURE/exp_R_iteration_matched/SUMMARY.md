# Exp R — Iteration-matched plain-Leiden control on email-Enron (closes exp_N caveat C4)

Design pre-registered in DESIGN.md before any code existed. Files: run.py, run_supp_n5.py,
results.csv (134 runs: 25 x arms A/B_n3/B_n4/C main + 9 calibration + 25 supplementary),
results.json, results_supp_n5.json, run.log, run_supp_n5.log (logs gitignored, local only).
Graph: email-Enron LCC n=33,696 m=180,811, loader and Leiden conventions copied verbatim from
exp_N/run.py. The seeded arm was NOT re-run: exp_N/partition_quality.csv rows kind==seeded
(5 runs, mean 0.614634 +- 0.002518, best 0.617876, mean k 156.2, mean 2.289 s) are the reference.

## Verdict (three parts)

**V1 — Kill criterion did NOT fire. Deeper plain Leiden does not close the Enron gap at matched
cost.** At the calibrated budget the plain arm means are 0.609802 (n_iter=3), 0.610141 (n_iter=4)
and 0.610357 (n_iter=5, supplementary) against the seeded 0.614634: gaps **-0.00483, -0.00449,
-0.00428**, Mann-Whitney p = 0.0131 / 0.0189 / 0.0225, permutation p = 0.0120 / 0.0186 / (n.a.).
Kill criterion 1 required mean_B >= 0.613634 (never approached, closest 0.610357); criterion 2
required p>0.10 AND |gap|<0.002 (both fail on both counts). The "reliability and price" framing of
C9/Exp N survives the iteration dimension.
[results.csv arm=B_n3|B_n4 phase=main, arm=B_n5_supp phase=supplementary; results.json `tests`,
`kill_criterion`; results_supp_n5.json]

**V2 — Iteration-deepening buys almost nothing; even the asymptote falls short.** Mean Q by
iteration count: 0.609219 (2) -> 0.609802 (3) -> 0.610141 (4) -> 0.610357 (5) -> 0.612222
(convergence, median 23 iterations). That is **+0.00058, +0.00034, +0.00022** per extra iteration,
a curve flattening ~0.0024 BELOW the seeded mean. Arm C costs a median 10.053 s = **4.39x** one
pipeline and still averages -0.00241 relative to seeded; the direction agrees with P2 but the
difference is not statistically distinguishable at n=25 vs 5 (MWU p=0.152, permutation p=0.192).
[results.csv arm=C; results.json `arm_A`,`arm_B`,`arm_C`,`tests.C`]

**V3 — Seeding's variance reduction is not an iteration effect.** Seeded sd 0.002518; every plain
arm sits at sd 0.0037-0.0038 regardless of iteration count, i.e. variance ratios **2.28x (A),
2.22x (n=3), 2.21x (n=4), 2.19x (n=5), 2.17x (converged)** (population; 1.80-1.90x sample).
P3 holds with room to spare, and it holds for the convergence arm too, which was not required.
[results.json `tests.*.var_ratio_pop`]

## Pre-registered predictions

- **P1 (arm B mean < seeded mean): HOLDS**, both bracketing arms and the supplementary one.
- **P2 (arm C mean < seeded mean): HOLDS in point estimate** (0.612222 < 0.614634), but flagged as
  directional: p=0.152. Exp R cannot claim a significant convergence-vs-seeded gap.
- **P3 (arm B variance >= 1.5x seeded): HOLDS** (2.21-2.22x pop, 1.84-1.85x sample).
- **Kill criterion: NOT FIRED.**

## Arm B calibration (reported in full, per the design)

Target = one seeded pipeline, 2.289 s (exp_N finding 1). Calibration on 3 throwaway seeds
(500-502): n_iter=2 -> 1.268 s (0.55x), n_iter=3 -> **1.765 s (0.77x)**, n_iter=4 -> **2.518 s
(1.10x)**. No n_iterations landed within 5% of the target, so per DESIGN.md the closest value
below AND above were both run at 25 seeds. Over the 25 main seeds the medians came out lower than
on the calibration seeds (n_iter=3: 1.702 s = 0.74x; n_iter=4: 2.057 s = 0.90x), so a
**supplementary, explicitly NOT pre-registered** arm n_iter=5 was added (median 2.474 s = **1.08x**
the budget) to guarantee a strictly above-budget bracket. That arm gives plain Leiden more compute
than the pipeline and is therefore adversarial to Exp R's own conclusion; it does not change it
(mean 0.610357, gap -0.00428, p=0.0225).
[results.json `calibration`; results.csv phase=calibration and phase=supplementary]

## Numbered findings

1. **Arm A reproduces exp_N exactly.** Seeds 100-104 give Q identical to exp_N's baseline rows to
   all printed digits (max |dQ| = 0.000e+00), confirming loader, simplification, LCC extraction and
   RNG seeding are byte-identical between the experiments. Arm A's 25-seed distribution (seeds
   100-124): mean 0.609219 +- 0.003802, best 0.615257, worst 0.602538, mean k 183.1 — statistically
   the same population as exp_N's own 25 plain restarts (mean 0.608432 +- 0.004446), which used a
   different seed set (100-104 + 900-919).
   [results.json `reproduction_check`, `arm_A`]

2. **Machine speed is unchanged, so the wall-clock budget transfers.** Arm A median 1.231 s vs
   exp_N's 1.236 s per plain restart = **0.996x**. A ratio-matched budget would have been 2.279 s,
   indistinguishable from the absolute 2.289 s target.
   [results.json `reproduction_check.machine_speed_ratio_vs_expN`, `calibration.ratio_matched_target_seconds`]

3. **At equal cost the single pipeline still beats most deepened plain runs.** The seeded mean sits
   at the **92nd** percentile of arm A, the **84th** of n_iter=3, the **76th** of n_iter=4 and of
   n_iter=5, and the **68th** of the converged arm. exp_N's "88th percentile" statement was measured
   against its own 25 plain restarts at n_iter=2; the equal-cost, iteration-deepened analogue is
   76th. Fraction of plain runs beating the seeded BEST: 0/25 in every arm except convergence.
   [results.json `tests.*.percentile_of_seeded_mean_in_arm`, `frac_runs_above_seeded_best`]

4. **One converged plain run beats the best seeded partition.** Arm C seed 109 reaches Q=0.619731,
   **+0.001856** above the best seeded partition (0.617876), at 9.9 s. This is the same phenomenon
   as exp_N V4 / caveat C2 (plain restart 917 came within 0.000186) and reinforces it: the seeded
   solution is reachable by plain Leiden, just not reliably or cheaply. Any "plain Leiden cannot
   reach it" phrasing stays forbidden.
   [results.csv arm=C seed=109; results.json `tests.C.best_minus_seeded_best`]

5. **Deeper iteration moves plain Leiden AWAY from the seeded partition's granularity.** Mean
   community count rises monotonically with iterations: 183.1 (n=2) -> 186.9 (3) -> 189.0 (4) ->
   189.3 (5) -> 192.1 (converged), while the seeded partitions sit at 156.2. Whatever the pipeline
   does, it is not "more of the same optimisation": more Leiden effort fragments further, seeding
   consolidates. (Descriptive; no partitions were stored in Exp R, so no AMI check — see caveats.)
   [results.csv column k; results.json `arm_*.mean_k`]

6. **Convergence cost and depth.** Plain Leiden needs a median of **23 iterations** (min 12, max 47)
   and a median of **10.053 s** (mean 10.263 s, max 19.98 s) to converge on email-Enron: 4.39x one
   pipeline, and 8.2x one plain restart. The manual single-iteration loop used to count iterations
   was verified equivalent to leidenalg's native n_iterations=-1 on seeds 500 and 501 (Q identical
   to 1e-16, diff exactly 0.0).
   [results.json `arm_C`, `convergence_equivalence`; results.csv column n_iters_run]

## Controls checklist

- Runtime/cost matching: **the entire point of the experiment** — three plain arms at 0.74x, 0.90x
  and 1.08x of one pipeline, plus the 4.39x asymptote.
- Granularity: reported (finding 5). Exp N already ruled out the granularity artifact for the
  seeded gain (corr(k,Q) = -0.001 over 25 plain restarts, k-matched gap +0.0060); Exp R adds that
  iteration count raises k, i.e. moves away from the seeded k.
- Chance-corrected metrics: not applicable (modularity comparison on one fixed graph; no AMI /
  recovery claim made here).
- Config-model null: not applicable (this is a compute-budget control, not a structure claim).
- Jackknife on correlations: no correlation is claimed.

## Caveats

- **C1.** The seeded reference is n=5 (exp_N seeds 300-304). Every gap in this experiment is
  25-vs-5; the Mann-Whitney tests are correspondingly weak, and the arm C comparison (p=0.15) is
  not significant. Exp R establishes the cost-matched B arms, not the asymptote.
- **C2.** P2 is supported in direction only. The honest sentence is "run to convergence at 4.4x the
  pipeline cost, plain Leiden still averages below the seeded mean (-0.0024), though not
  significantly so at these sample sizes", NOT "convergence cannot reach it".
- **C3.** The n_iter=5 arm is post-hoc (added after the main run, when the 25-seed medians came in
  below the calibration medians). It is reported as supplementary; the pre-registered arms are
  n_iter=3 and n_iter=4. It was added in the direction that could only hurt our conclusion.
- **C4.** Wall clock was measured on a loaded desktop; per-arm medians vary by up to ~20% between
  the calibration seeds and the main seeds (n_iter=4: 2.518 s vs 2.057 s). The three-arm bracket
  (0.74x / 0.90x / 1.08x) plus the 4.39x asymptote makes the conclusion robust to that jitter.
- **C5.** Single network (email-Enron), single alpha (0.90), single sampler (calibrated), one
  implementation (leidenalg 0.12.0, single-threaded). Nothing here generalises to other networks;
  it closes exactly the one hole exp_N flagged as C4.
- **C6.** No partitions were saved, so there is no AMI / structural comparison between the
  iteration-deepened plain partitions and the seeded ones. Finding 5 is community-count only.

## Effect on Exp N and on claim C9

Exp N caveat **C4 is closed** and should be rewritten from "there is no n_iterations-matched plain
control" to a pointer to Exp R. **No sentence of the Exp N discussion paragraph needs to be
weakened or removed** — the kill did not fire, and the two claims that could have been damaged
("88th percentile at 1/10 the compute of a matching restart search" and "halves the seed variance")
both survive: iteration-deepened plain Leiden at the same budget still ranks the seeded mean at the
76th percentile and keeps 2.2x the seeded variance.

The one recommended edit is an **addition** of a single sentence, so that "price" covers both ways
of spending compute. Current text:

  "What seeding buys is reliability and price. It places a single 2.3-second pipeline at the 88th
  percentile of the plain-restart distribution and halves the seed variance, whereas reaching the
  same expected quality by brute force takes roughly ten times the compute."

Suggested continuation (numbers from Exp R):

  "Nor is the budget better spent on a deeper search: plain Leiden given the same 2.3 seconds as
  extra iterations rather than extra restarts gains only +0.0002 to +0.0006 modularity per
  iteration and still averages 0.0043 to 0.0048 below the seeded runs (p < 0.03), and run all the
  way to convergence — a median of 23 iterations at 4.4 times the pipeline cost — it averages
  0.6122, still below the seeded 0.6146. Deeper iteration also moves the partition away from the
  seeded one, raising the community count from 183 to 192 against the seeded 156."

STORY.md C9 needs no change beyond noting that the compute control is now two-dimensional
(restarts and iterations). This paragraph is a proposal; per EXPLORATION.md rule 8 it requires
Mohammad's sign-off before it enters STORY.md or any .tex file.
