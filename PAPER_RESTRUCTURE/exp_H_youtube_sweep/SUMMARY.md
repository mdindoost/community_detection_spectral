# Experiment H — com-Youtube: is the seeded-refinement gain real?

Referee finding MC3 (`PAPER_RESTRUCTURE/phase3_review/JUDGE_RULING.md`) flagged the
com-Youtube result from `exp_E_delta_star` as fragile: **one** configuration
(calibrated DSpar, alpha=0.9, 3 sparsification seeds), and a runtime-matched
baseline that was a **single realization** of a best-of-2 draw which landed
*below* the plain-Leiden mean (Q_matched = 0.720932 vs
Q_base_mean = 0.724506). This experiment replaces that single
draw with the **exact distribution** of best-of-2 and sweeps both samplers x 4
alphas x 5 sparsification seeds.

Protocol, seeds, loader, samplers, seeded refinement and Leiden calls are reused
verbatim from `exp_E_delta_star/run.py` and `exp_C_true_retention_seeded/run.py`
(base seeds 100.., sparsify seeds 200.., Leiden-on-sparse seeds 300..,
`la.ModularityVertexPartition` + `Optimiser.optimise_partition`, n_iterations=2).
Graph: com-Youtube LCC, n=1,134,890, m=2,987,624.

## 1. Baseline distribution (10 plain Leiden runs, seeds 100-109)

| seed | Q_base | k | T (s) |
|---|---|---|---|
| 100 | 0.726999 | 5533 | 55.3 |
| 101 | 0.728465 | 5661 | 67.5 |
| 102 | 0.723870 | 6978 | 67.3 |
| 103 | 0.718447 | 7211 | 68.9 |
| 104 | 0.724751 | 6754 | 67.1 |
| 105 | 0.719548 | 7160 | 67.6 |
| 106 | 0.723145 | 7028 | 69.6 |
| 107 | 0.728759 | 5562 | 69.1 |
| 108 | 0.722669 | 6694 | 71.2 |
| 109 | 0.726690 | 5761 | 70.2 |

- mean   = **0.724334**
- std    = **0.003519** (sample, ddof=1); 0.003338 (population)
- best   = **0.728759**   min = 0.718447
- range  = 0.010312
- T_leiden median = 68.3 s (4 workers ran concurrently, so wall times are inflated ~1.3-2x vs a solo run;
  only the *ratio* T_pipe / T_leiden is used, and both are inflated equally)

exp_E's 5-seed estimate was mean 0.724506 +- 0.003437 (population std); the 10-seed estimate here is
mean 0.724334 +- 0.003338 (population std).

## 2. Runtime-matched baseline as a distribution, not a draw

T_pipe / T_leiden is 1.5-1.7x for every configuration below, and exp_E measured
`n_matched_restarts = 2` on com-Youtube, so the honest runtime-matched baseline
is **best-of-2 plain Leiden** (which slightly over-credits the baseline, since a
strict match buys only ~1.5-1.7 restarts). Its exact distribution over all C(10,2)=45
unordered pairs of the 10 baseline runs:

| statistic | best-of-2 (45 pairs) | best-of-5 (252 subsets) | best-of-10 |
|---|---|---|---|
| mean | **0.726428** | 0.728211 | 0.728759 |
| std | 0.002301 | 0.000909 | 0 |
| min | 0.719548 | 0.723870 | — |
| max | 0.728759 | 0.728759 | — |

exp_E's single matched draw was 0.720932. Its percentile in
this best-of-2 distribution: **2.2%**
(i.e. 97.8% of runtime-matched baselines
beat it), and it sits -1.56 sigma
from the best-of-2 mean. MC3's diagnosis is confirmed: the exp_E baseline was a
low draw, and the +0.0086 headline gain is inflated by that draw.

## 3-4. Seeded sweep and per-configuration verdict

`gain` columns are `Q_seeded_mean - <baseline>`; `z` columns divide by
Q_base_std = 0.003519. `P(bo2>=)` is the empirical fraction of the 45
runtime-matched best-of-2 draws that meet or beat the seeded mean.

| sampler | alpha | true ret. | Q_raw | Q_seeded mean | +-std | best | T_pipe (s) | restarts | gain vs base mean | z | gain vs E[bo2] | **z vs bo2** | P(bo2>=) | gain vs E[bo5] | z vs bo5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibrated | 0.7 | 0.7000 | 0.703343 | 0.719605 | 0.003300 | 0.725282 | 118 | 1.7x | -0.004729 | -1.34 | -0.006822 | **-1.94** | 0.98 | -0.008606 | -2.45 |
| calibrated | 0.8 | 0.8000 | 0.720814 | 0.725128 | 0.004002 | 0.729187 | 115 | 1.7x | +0.000793 | +0.23 | -0.001300 | **-0.37** | 0.67 | -0.003083 | -0.88 |
| calibrated | 0.9 | 0.9000 | 0.726584 | 0.728681 | 0.001230 | 0.729679 | 111 | 1.6x | +0.004347 | +1.24 | +0.002254 | **+0.64** | 0.20 | +0.000470 | +0.13 |
| calibrated | 0.95 | 0.9500 | 0.727856 | 0.729484 | 0.000431 | 0.729969 | 113 | 1.7x | +0.005150 | +1.46 | +0.003056 | **+0.87** | 0.00 | +0.001273 | +0.36 |
| repo_noreplace | 0.7 | 0.4756 | 0.596848 | 0.725488 | 0.000461 | 0.726104 | 105 | 1.5x | +0.001154 | +0.33 | -0.000940 | **-0.27** | 0.67 | -0.002723 | -0.77 |
| repo_noreplace | 0.8 | 0.5022 | 0.623392 | 0.724607 | 0.000838 | 0.725332 | 104 | 1.5x | +0.000273 | +0.08 | -0.001820 | **-0.52** | 0.78 | -0.003603 | -1.02 |
| repo_noreplace | 0.9 | 0.5242 | 0.641599 | 0.724556 | 0.001084 | 0.725479 | 105 | 1.5x | +0.000222 | +0.06 | -0.001871 | **-0.53** | 0.78 | -0.003655 | -1.04 |
| repo_noreplace | 0.95 | 0.5347 | 0.649449 | 0.724502 | 0.001102 | 0.725672 | 103 | 1.5x | +0.000168 | +0.05 | -0.001925 | **-0.55** | 0.78 | -0.003709 | -1.05 |

- configurations with gain > 2 sigma vs the FAIR (expected) best-of-2 baseline: **0 / 8**
- configurations with any positive gain vs E[best-of-2]: 2 / 8
- configurations with gain > 2 sigma vs E[best-of-5]: 0 / 8; positive: 2 / 8
- configurations whose seeded mean exceeds the best of all 10 plain runs: 1 / 8

### 4b. A sharper test (the z columns above are deliberately conservative)

Dividing by Q_base_std compares a *mean of 5 seeded runs* against the spread of a
*single* run, so it under-states significance. Two extra tests:

* **Head-to-head win rate**: over all 5 x 45 = 225 (seeded run, best-of-2 draw)
  pairs, the fraction where the seeded run wins.
* **Bootstrap on the means**: resample the 10 baseline runs with replacement
  (B=20000) to get SE(E[best-of-2]); combine with SE of the seeded mean
  (std/sqrt(5)) to get z_mean = gain / sqrt(SE_bo2^2 + SE_seeded^2), and a
  bootstrap p-value P(E[best-of-2]* >= Q_seeded_mean).

SE(E[best-of-2]) from bootstrap = 0.000979

| sampler | alpha | gain vs E[bo2] | head-to-head win rate | SE_seeded | z_mean | bootstrap p |
|---|---|---|---|---|---|---|
| calibrated | 0.7 | -0.006822 | 0.071 | 0.001476 | **-3.85** | 1.0000 |
| calibrated | 0.8 | -0.001300 | 0.476 | 0.001790 | **-0.64** | 0.8636 |
| calibrated | 0.9 | +0.002254 | 0.849 | 0.000550 | **+2.01** | 0.0000 |
| calibrated | 0.95 | +0.003056 | 1.000 | 0.000193 | **+3.06** | 0.0000 |
| repo_noreplace | 0.7 | -0.000940 | 0.333 | 0.000206 | **-0.94** | 0.7839 |
| repo_noreplace | 0.8 | -0.001820 | 0.271 | 0.000375 | **-1.74** | 0.9367 |
| repo_noreplace | 0.9 | -0.001871 | 0.253 | 0.000485 | **-1.71** | 0.9421 |
| repo_noreplace | 0.95 | -0.001925 | 0.253 | 0.000493 | **-1.76** | 0.9474 |

- configurations with z_mean > 2 on this sharper test: **2 / 8** (before any multiplicity correction; 8 configurations were tested, so a Bonferroni-corrected two-sided 5% threshold is |z| > 2.73)
- configurations surviving Bonferroni (z_mean > 2.73): **1 / 8**

Budget note: T_pipe / T_leiden is 1.5-1.7x, so a strictly runtime-matched baseline
buys only ~1.5-1.7 restarts. Charging the baseline a full best-of-2 is *generous to
the baseline* (conservative for the seeded method); exp_E did the same.

### 4c. Per-run seeded results

| sampler | alpha | spar seed | retention | Q_raw | Q_seeded | k | T_pipe (s) |
|---|---|---|---|---|---|---|---|
| calibrated | 0.7 | 200 | 0.7000 | 0.700643 | 0.717504 | 3176 | 119 |
| calibrated | 0.7 | 201 | 0.6999 | 0.699875 | 0.718198 | 4595 | 121 |
| calibrated | 0.7 | 202 | 0.7000 | 0.711961 | 0.725282 | 4397 | 120 |
| calibrated | 0.7 | 203 | 0.7000 | 0.698707 | 0.717385 | 2928 | 117 |
| calibrated | 0.7 | 204 | 0.7002 | 0.705529 | 0.719657 | 4119 | 114 |
| calibrated | 0.8 | 200 | 0.8001 | 0.717036 | 0.721285 | 4373 | 117 |
| calibrated | 0.8 | 201 | 0.7998 | 0.724773 | 0.729187 | 3991 | 116 |
| calibrated | 0.8 | 202 | 0.8001 | 0.717155 | 0.720886 | 4473 | 116 |
| calibrated | 0.8 | 203 | 0.8000 | 0.724603 | 0.728973 | 4336 | 113 |
| calibrated | 0.8 | 204 | 0.8000 | 0.720504 | 0.725306 | 4570 | 113 |
| calibrated | 0.9 | 200 | 0.8999 | 0.727552 | 0.729661 | 4781 | 101 |
| calibrated | 0.9 | 201 | 0.9001 | 0.727684 | 0.729679 | 4871 | 114 |
| calibrated | 0.9 | 202 | 0.9000 | 0.727293 | 0.729372 | 4779 | 115 |
| calibrated | 0.9 | 203 | 0.9000 | 0.725423 | 0.727521 | 4952 | 114 |
| calibrated | 0.9 | 204 | 0.9000 | 0.724965 | 0.727174 | 4505 | 114 |
| calibrated | 0.95 | 200 | 0.9500 | 0.728483 | 0.729969 | 5171 | 114 |
| calibrated | 0.95 | 201 | 0.9500 | 0.727992 | 0.729521 | 5369 | 112 |
| calibrated | 0.95 | 202 | 0.9500 | 0.727796 | 0.729633 | 5375 | 114 |
| calibrated | 0.95 | 203 | 0.9500 | 0.727960 | 0.729509 | 5399 | 112 |
| calibrated | 0.95 | 204 | 0.9501 | 0.727047 | 0.728788 | 4981 | 112 |
| repo_noreplace | 0.7 | 200 | 0.4755 | 0.601767 | 0.726104 | 3907 | 105 |
| repo_noreplace | 0.7 | 201 | 0.4756 | 0.595353 | 0.724974 | 3678 | 105 |
| repo_noreplace | 0.7 | 202 | 0.4756 | 0.594087 | 0.725107 | 3820 | 105 |
| repo_noreplace | 0.7 | 203 | 0.4756 | 0.595847 | 0.725526 | 4502 | 106 |
| repo_noreplace | 0.7 | 204 | 0.4755 | 0.597183 | 0.725728 | 4112 | 104 |
| repo_noreplace | 0.8 | 200 | 0.5021 | 0.623666 | 0.724880 | 3927 | 104 |
| repo_noreplace | 0.8 | 201 | 0.5022 | 0.624597 | 0.725332 | 4363 | 105 |
| repo_noreplace | 0.8 | 202 | 0.5021 | 0.624448 | 0.723189 | 3761 | 104 |
| repo_noreplace | 0.8 | 203 | 0.5022 | 0.622655 | 0.724588 | 3920 | 105 |
| repo_noreplace | 0.8 | 204 | 0.5022 | 0.621593 | 0.725049 | 3642 | 104 |
| repo_noreplace | 0.9 | 200 | 0.5241 | 0.638650 | 0.725448 | 3848 | 105 |
| repo_noreplace | 0.9 | 201 | 0.5243 | 0.640943 | 0.725094 | 3952 | 104 |
| repo_noreplace | 0.9 | 202 | 0.5240 | 0.641211 | 0.723384 | 3887 | 105 |
| repo_noreplace | 0.9 | 203 | 0.5242 | 0.645125 | 0.723376 | 4403 | 104 |
| repo_noreplace | 0.9 | 204 | 0.5242 | 0.642068 | 0.725479 | 3767 | 104 |
| repo_noreplace | 0.95 | 200 | 0.5347 | 0.647716 | 0.725154 | 3964 | 105 |
| repo_noreplace | 0.95 | 201 | 0.5348 | 0.650345 | 0.725038 | 3777 | 104 |
| repo_noreplace | 0.95 | 202 | 0.5346 | 0.649498 | 0.723343 | 3913 | 102 |
| repo_noreplace | 0.95 | 203 | 0.5348 | 0.649327 | 0.723305 | 4341 | 102 |
| repo_noreplace | 0.95 | 204 | 0.5348 | 0.650357 | 0.725672 | 3933 | 101 |

## 5. Verdict

**com-Youtube is a weak, narrow, alpha-dependent gain -- not Enron-class, but not
pure noise either.**

1. MC3 is confirmed. exp_E's matched baseline (0.720932) is beaten
   by 98% of the 45 possible best-of-2
   draws. Against the FAIR baseline E[best-of-2] = 0.726428, the
   same configuration (calibrated, alpha=0.9) gives
   +0.002254,
   not +0.0086. **The headline gain shrinks by ~3.8x.**
2. On the requested metric (gain / Q_base_std), **0 of 8** configurations
   clear 2 sigma. On the sharper mean-vs-mean bootstrap, 2 of 8 clear z=2 and only
   1 of 8 (calibrated alpha=0.95, z=+3.06) survives Bonferroni
   correction for the 8 configurations tested.
3. The effect is **entirely confined to the calibrated sampler at high alpha**.
   All four repo_noreplace configurations (true retention 0.48-0.53) are
   *negative* vs E[best-of-2], and calibrated alpha=0.7 is strongly negative
   (-3.85 sigma). Only alpha in {0.9, 0.95} is positive, and the gain is monotone
   increasing in alpha -- i.e. it grows as the sparsifier does less. Extrapolating,
   alpha -> 1 (no sparsification at all) would be best, which is the signature of a
   *seeding/refinement* artifact rather than a sparsification benefit.
4. Effect size: the largest honest gain is
   +0.003056 at calibrated alpha=0.95, about
   0.87 single-run sigma, for a 1.7x
   runtime cost. Contrast email-Enron (exp_C), where **all 8 of 8** configurations,
   both samplers, alpha 0.7-0.95, gave seeded-minus-base +0.0057 to +0.0130 on a
   Q_base_std of 0.00187 (z = 3-7).

So: com-Youtube is **not** an Enron-class robust gain -- Enron is robust across the
whole sampler x alpha grid, com-Youtube is positive in 2 of 8 cells and negative in 6.
It is best described as a **weak-but-real gain in a narrow high-alpha corner**: at
calibrated alpha=0.95 the seeded mean beats every one of the 45 runtime-matched
best-of-2 draws and every one of the 10 individual plain runs (head-to-head win rate
1.000, bootstrap p < 1e-4), which is not something noise produces. But it is roughly
a third of the size exp_E reported, it does not survive as a general claim about the
sparsifier, and the +0.0086 / 2.5 sigma figure should not be used.

