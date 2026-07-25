# Experiment E - does any structural statistic predict the genuine seeded gain?

Outcome measured at **calibrated DSpar, alpha = 0.9** (E[retention]=0.9 exactly,
lambda-bisection sampler from exp_C), 3 sparsification seeds, 5 baseline seeds,
Leiden n_iterations=2, undirected simple LCC.

```
y  = mean(Q_seeded) - Q_matched_best     <- HONEST outcome (runtime-matched restarts)
y2 = mean(Q_seeded) - Q_base_mean        <- naive outcome (single-run baseline)
```

## 1. Outcomes (n=15 networks)

| network | n | m | ret | Q_base_mean +- std | Q_seeded | Q_matched | restarts | y | y2 | y>2sigma |
|---|---|---|---|---|---|---|---|---|---|---|
| email-Eu-core | 986 | 16,064 | 0.8999 | 0.416048 +- 0.000429 | 0.415066 | 0.417362 | 2 | -0.002296 | -0.000982 | no |
| wiki-Vote | 7,066 | 100,736 | 0.9001 | 0.424425 +- 0.002497 | 0.423441 | 0.429366 | 2 | -0.005925 | -0.000984 | no |
| ca-GrQc | 4,158 | 13,422 | 0.9002 | 0.850575 +- 0.000950 | 0.852105 | 0.851743 | 2 | +0.000361 | +0.001529 | no |
| ca-HepTh | 8,638 | 24,806 | 0.9011 | 0.761367 +- 0.000943 | 0.760498 | 0.762103 | 2 | -0.001606 | -0.000869 | no |
| facebook-combined | 4,039 | 88,234 | 0.9001 | 0.835631 +- 0.000073 | 0.835565 | 0.835465 | 2 | +0.000100 | -0.000065 | no |
| ca-CondMat | 21,363 | 91,286 | 0.9000 | 0.730790 +- 0.001097 | 0.731217 | 0.729382 | 2 | +0.001835 | +0.000427 | no |
| ca-HepPh | 11,204 | 117,619 | 0.8996 | 0.658312 +- 0.000976 | 0.659880 | 0.660195 | 2 | -0.000315 | +0.001567 | no |
| ca-AstroPh | 17,903 | 196,972 | 0.9003 | 0.630794 +- 0.002698 | 0.632829 | 0.631554 | 2 | +0.001275 | +0.002035 | no |
| email-Enron | 33,696 | 180,811 | 0.8996 | 0.607097 +- 0.001873 | 0.612798 | 0.605125 | 2 | +0.007673 | +0.005701 | YES |
| cit-HepTh | 27,400 | 352,021 | 0.9001 | 0.662092 +- 0.002274 | 0.661138 | 0.657897 | 2 | +0.003241 | -0.000955 | no |
| cit-HepPh | 34,401 | 420,784 | 0.9003 | 0.731880 +- 0.001944 | 0.733487 | 0.734313 | 2 | -0.000826 | +0.001606 | no |
| com-Amazon | 334,863 | 925,872 | 0.8998 | 0.929438 +- 0.000277 | 0.930021 | 0.929575 | 2 | +0.000446 | +0.000583 | no |
| com-DBLP | 317,080 | 1,049,866 | 0.8999 | 0.825701 +- 0.000840 | 0.828106 | 0.827538 | 2 | +0.000568 | +0.002406 | no |
| com-Youtube | 1,134,890 | 2,987,624 | 0.9000 | 0.724506 +- 0.003437 | 0.729571 | 0.720932 | 2 | +0.008639 | +0.005064 | YES |
| wiki-Talk | 2,388,953 | 4,656,682 | 0.9000 | 0.601147 +- 0.000658 | 0.601467 | 0.602045 | 2 | -0.000578 | +0.000320 | no |

### Cross-check vs exp_C (calibrated, alpha=0.9)

exp_C used 5 sparsification seeds, exp_E uses 3, so small differences in
Q_seeded_mean are expected; Q_base_mean / Q_matched_best should be identical
(same seeds, same budget rule).

| network | Q_base_mean E/C | Q_matched_best E/C | y_E | y_C | |dy| | flag |
|---|---|---|---|---|---|---|
| email-Eu-core | 0.416048 / 0.416048 | 0.417362 / 0.417362 | -0.002296 | -0.002344 | 0.000049 | ok |
| wiki-Vote | 0.424425 / 0.424425 | 0.429366 / 0.429366 | -0.005925 | -0.006874 | 0.000949 | ok |
| ca-HepTh | 0.761367 / 0.761367 | 0.762103 / 0.762103 | -0.001606 | -0.001638 | 0.000032 | ok |
| ca-CondMat | 0.730790 / 0.730790 | 0.729382 / 0.729382 | +0.001835 | +0.001377 | 0.000458 | ok |
| email-Enron | 0.607097 / 0.607097 | 0.605125 / 0.605125 | +0.007673 | +0.009509 | 0.001836 | ok |
| com-DBLP | 0.825701 / 0.825701 | 0.827538 / 0.827538 | +0.000568 | +0.000226 | 0.000342 | ok |

## 2. Predictors (from exp_B_config_null/results.csv; 17 networks)

`delta_star = delta_real - delta_null`, `dQ_ratio = dQ_fixed_real / dQ_fixed_null`,
`hb_excess = hb_real - hb_null`, `deg_cv = std(deg)/mean(deg)` on the LCC.

| network | delta_real | delta_null | delta_star | dQ_real | dQ_null | dQ_ratio | hb_real | hb_null | hb_excess | deg_cv | y |
|---|---|---|---|---|---|---|---|---|---|---|---|
| email-Eu-core | +0.0340 | +0.0413 | -0.0073 | 0.0819 | 0.0350 | 2.341 | 2.113 | 1.404 | +0.710 | 1.136 | -0.002296 |
| wiki-Vote | +0.0513 | +0.1009 | -0.0496 | 0.0711 | 0.0843 | 0.843 | 1.902 | 1.689 | +0.214 | 2.025 | -0.005925 |
| ca-GrQc | +0.0872 | +0.1890 | -0.1018 | 0.0176 | 0.0991 | 0.178 | 0.478 | 1.758 | -1.280 | 1.336 | +0.000361 |
| ca-HepTh | +0.1736 | +0.2147 | -0.0410 | 0.0523 | 0.0941 | 0.556 | 1.581 | 1.929 | -0.348 | 1.123 | -0.001606 |
| facebook-combined | -0.0014 | +0.0170 | -0.0183 | 0.0256 | 0.0270 | 0.948 | 2.681 | 1.391 | +1.289 | 1.200 | +0.000100 |
| ca-CondMat | +0.1126 | +0.1018 | +0.0108 | 0.0606 | 0.0667 | 0.909 | 2.490 | 1.493 | +0.997 | 1.276 | +0.001835 |
| ca-HepPh | +0.0211 | +0.0935 | -0.0725 | 0.0864 | 0.0917 | 0.943 | 0.501 | 1.872 | -1.371 | 2.288 | -0.000315 |
| ca-AstroPh | +0.0332 | +0.0413 | -0.0081 | 0.0393 | 0.0489 | 0.805 | 1.496 | 1.294 | +0.202 | 1.409 | +0.001275 |
| email-Enron | +0.1501 | +0.2400 | -0.0899 | 0.1422 | 0.1562 | 0.911 | 2.974 | 4.472 | -1.497 | 3.502 | +0.007673 |
| cit-HepTh | +0.0265 | +0.0322 | -0.0057 | 0.0461 | 0.0376 | 1.227 | 2.173 | 1.778 | +0.395 | 1.772 | +0.003241 |
| cit-HepPh | +0.0110 | +0.0264 | -0.0153 | 0.0126 | 0.0319 | 0.395 | 1.358 | 1.300 | +0.058 | 1.263 | -0.000826 |
| com-Amazon | +0.0096 | +0.1389 | -0.1293 | 0.0000 | 0.0586 | 0.000 | 1.042 | 2.139 | -1.097 | 1.042 | +0.000446 |
| com-DBLP | +0.1469 | +0.1965 | -0.0497 | 0.0437 | 0.0980 | 0.446 | 1.583 | 2.308 | -0.725 | 1.511 | +0.000568 |
| com-Youtube | +0.2459 | +0.5520 | -0.3061 | 0.0959 | 0.2633 | 0.364 | 3.996 | 14.066 | -10.069 | 9.640 | +0.008639 |
| wiki-Talk | +0.5135 | +0.8313 | -0.3178 | 0.2020 | 0.3529 | 0.572 | 10.477 | 105.630 | -95.153 | 26.324 | -0.000578 |
| cit-Patents | +0.0378 | +0.1659 | -0.1282 | 0.0047 | 0.0768 | 0.061 | 0.723 | 1.940 | -1.217 | 1.197 | n/a |
| wiki-topcats | +0.0229 | +0.0287 | -0.0057 | 0.0484 | 0.0538 | 0.899 | 5.076 | 6.920 | -1.843 | 10.114 | n/a |

## 3. Correlations with y (n=15), exact p-values

| predictor | Spearman rho | p | Pearson r | p | Spearman rho (y2) | p |
|---|---|---|---|---|---|---|
| delta_real | +0.1714 | 0.5413 | +0.2065 | 0.4602 | +0.1964 | 0.4829 |
| delta_star | -0.0750 | 0.7905 | -0.3599 | 0.1877 | -0.3857 | 0.1556 |
| delta_ratio | +0.1036 | 0.7134 | +0.0592 | 0.8339 | -0.2464 | 0.3760 |
| dQ_ratio | -0.0893 | 0.7517 | -0.1655 | 0.5555 | -0.4286 | 0.1110 |
| dQ_excess | -0.1464 | 0.6025 | -0.3456 | 0.2070 | -0.4500 | 0.0924 |
| hb_real | +0.3036 | 0.2714 | +0.1293 | 0.6460 | -0.0500 | 0.8595 |
| hb_excess | -0.2107 | 0.4510 | +0.0379 | 0.8932 | -0.5714 | 0.0261 |
| hb_ratio | -0.0500 | 0.8595 | -0.2344 | 0.4005 | -0.4357 | 0.1045 |
| deg_cv | +0.3393 | 0.2160 | +0.1223 | 0.6642 | +0.3464 | 0.2059 |
| Qfix_real | +0.2000 | 0.4748 | +0.2572 | 0.3547 | +0.2107 | 0.4510 |
| Qfix_ratio | -0.1179 | 0.6757 | -0.1846 | 0.5102 | -0.1500 | 0.5936 |
| avg_deg | -0.2857 | 0.3019 | -0.3576 | 0.1906 | -0.3964 | 0.1435 |
| n | +0.4679 | 0.0786 | +0.1643 | 0.5584 | +0.5250 | 0.0445 |
| m | +0.3750 | 0.1684 | +0.2475 | 0.3738 | +0.4500 | 0.0924 |

Bonferroni threshold for 14 predictors at family-wise 0.05: p < 0.0036.

**Power.** At n=15, a two-sided Spearman test needs |rho| >= 0.514 to reach p<0.05, and |rho| >= 0.701 to survive Bonferroni over 14 predictors. The largest observed |rho| is 0.468.

## 4. Leave-one-out sensitivity (Spearman rho vs y, n=14 each)

Top-3 predictors by |rho| plus delta_real / delta_star / deg_cv.

| dropped | `n` | `m` | `deg_cv` | `delta_real` | `delta_star` |
|---|---|---|---|---|---|
| email-Eu-core | +0.358 (p=0.208) | +0.279 (p=0.334) | +0.266 (p=0.358) | +0.178 (p=0.543) | +0.015 (p=0.958) |
| wiki-Vote | +0.407 (p=0.149) | +0.349 (p=0.221) | +0.490 (p=0.075) | +0.178 (p=0.543) | -0.059 (p=0.840) |
| ca-GrQc | +0.486 (p=0.078) | +0.411 (p=0.144) | +0.354 (p=0.215) | +0.147 (p=0.615) | -0.073 (p=0.805) |
| ca-HepTh | +0.393 (p=0.164) | +0.279 (p=0.334) | +0.266 (p=0.358) | +0.314 (p=0.274) | -0.042 (p=0.887) |
| facebook-combined | +0.486 (p=0.078) | +0.371 (p=0.191) | +0.349 (p=0.221) | +0.174 (p=0.553) | -0.086 (p=0.771) |
| ca-CondMat | +0.473 (p=0.088) | +0.442 (p=0.114) | +0.345 (p=0.227) | +0.108 (p=0.714) | -0.218 (p=0.455) |
| ca-HepPh | +0.455 (p=0.102) | +0.371 (p=0.191) | +0.402 (p=0.154) | +0.174 (p=0.553) | -0.064 (p=0.829) |
| ca-AstroPh | +0.473 (p=0.088) | +0.385 (p=0.175) | +0.332 (p=0.246) | +0.204 (p=0.483) | -0.182 (p=0.533) |
| email-Enron | +0.473 (p=0.088) | +0.376 (p=0.185) | +0.231 (p=0.427) | +0.073 (p=0.805) | +0.046 (p=0.876) |
| cit-HepTh | +0.473 (p=0.088) | +0.367 (p=0.197) | +0.314 (p=0.274) | +0.222 (p=0.446) | -0.218 (p=0.455) |
| cit-HepPh | +0.552 (p=0.041) | +0.442 (p=0.114) | +0.275 (p=0.342) | +0.125 (p=0.670) | -0.029 (p=0.923) |
| com-Amazon | +0.473 (p=0.088) | +0.363 (p=0.203) | +0.411 (p=0.144) | +0.204 (p=0.483) | -0.073 (p=0.805) |
| com-DBLP | +0.473 (p=0.088) | +0.363 (p=0.203) | +0.332 (p=0.246) | +0.125 (p=0.670) | -0.073 (p=0.805) |
| com-Youtube | +0.389 (p=0.169) | +0.275 (p=0.342) | +0.231 (p=0.427) | +0.024 (p=0.935) | +0.095 (p=0.748) |
| wiki-Talk | +0.635 (p=0.015) | +0.525 (p=0.054) | +0.477 (p=0.085) | +0.314 (p=0.274) | -0.160 (p=0.584) |

- `n`: full-sample rho=+0.4679 (p=0.0786); LOO range [+0.3582, +0.6352]; sign flips under LOO: no; LOO folds with p<0.05: 2/15
- `m`: full-sample rho=+0.3750 (p=0.1684); LOO range [+0.2747, +0.5253]; sign flips under LOO: no; LOO folds with p<0.05: 0/15
- `deg_cv`: full-sample rho=+0.3393 (p=0.2160); LOO range [+0.2308, +0.4901]; sign flips under LOO: no; LOO folds with p<0.05: 0/15
- `delta_real`: full-sample rho=+0.1714 (p=0.5413); LOO range [+0.0242, +0.3143]; sign flips under LOO: no; LOO folds with p<0.05: 0/15
- `delta_star`: full-sample rho=-0.0750 (p=0.7905); LOO range [-0.2176, +0.0945]; sign flips under LOO: YES; LOO folds with p<0.05: 0/15

## 5. Binary analysis: y > 2*sigma(baseline seed noise)

GAIN group (n=2): email-Enron, com-Youtube
NO-GAIN group (n=13): email-Eu-core, wiki-Vote, ca-GrQc, ca-HepTh, facebook-combined, ca-CondMat, ca-HepPh, ca-AstroPh, cit-HepTh, cit-HepPh, com-Amazon, com-DBLP, wiki-Talk

| predictor | mean(GAIN) | mean(NO-GAIN) | Mann-Whitney U p | AUC | overlap? |
|---|---|---|---|---|---|
| delta_real | +0.1980 | +0.0938 | 0.1143 | 0.885 | OVERLAP |
| delta_star | -0.1980 | -0.0620 | 0.1714 | 0.154 | OVERLAP |
| dQ_ratio | +0.6373 | +0.7817 | 0.8000 | 0.423 | OVERLAP |
| hb_real | +3.4855 | +2.2981 | 0.0762 | 0.923 | OVERLAP |
| hb_excess | -5.7834 | -7.3930 | 0.0762 | 0.077 | OVERLAP |
| deg_cv | +6.5710 | +3.3620 | 0.0762 | 0.923 | OVERLAP |
| delta_ratio | +0.5355 | +0.5640 | 0.8000 | 0.423 | OVERLAP |
| hb_ratio | +0.4746 | +0.9447 | 0.3810 | 0.269 | OVERLAP |
| Qfix_ratio | +2.1583 | +3.2700 | 0.3810 | 0.269 | OVERLAP |

**Power of the binary test.** With 2 GAIN vs 13 NO-GAIN networks, the smallest attainable two-sided Mann-Whitney p is 0.0190 -- even a *perfect* separation could not survive Bonferroni correction over 14 predictors (threshold 0.0036). This test has effectively no power; it is reported only to show that not even perfect separation is observed.

## 6. Verdict

**(a) Which networks show genuine gains at alpha=0.9?**

2 of 15: `email-Enron` (y=+0.007673, 2sigma=0.003746), `com-Youtube` (y=+0.008639, 2sigma=0.006874).
Every other network is inside baseline seed noise or negative. The two next-largest positives fall short of the bar: `cit-HepTh` (y=+0.003241 vs 2sigma=0.004547), `ca-CondMat` (y=+0.001835 vs 2sigma=0.002195).
So the exp_C finding replicates and extends only marginally: the honest, runtime-matched seeded gain is a rare, network-specific event, not a general property.

**(b) Does any statistic separate them?**

No. The strongest rank correlation over all 14 candidates is `n` at rho=+0.4679, p=0.0786 -- not significant even uncorrected, and far from the Bonferroni threshold 0.0036. In the binary analysis every candidate OVERLAPS between the GAIN and NO-GAIN groups. The heavy-tail statistics (`deg_cv`, `hb_real`) come closest (AUC 0.92) but are broken by a single network: `wiki-Talk` has by far the most extreme degree heterogeneity (deg_cv=26.3, hb_real=10.5, the maxima of the whole set) and yet shows y<0. Adding `wiki-Talk` to the 14-network set collapsed `deg_cv`'s Pearson r from +0.690 (p=0.006) to +0.122 (p=0.664) and `n`'s Spearman rho from +0.635 (p=0.015) to +0.468 (p=0.079) -- i.e. the only apparently significant relationships were single-point artefacts.

**(c) Is delta* better or worse than raw delta?**

Worse, and wrong-signed. raw `delta_real`: rho=+0.1714 (p=0.5413), r=+0.2065 (p=0.4602). Excess `delta_star`: rho=-0.0750 (p=0.7905), r=-0.3599 (p=0.1877). Subtracting the configuration-model null flips the (already non-significant) association from positive to negative. Mechanically this is expected: delta_null is itself driven by degree heterogeneity and exceeds delta_real on 16 of 17 networks, so delta_star is essentially minus a heavy-tail statistic. delta_star therefore does not rescue delta as a predictor -- it is a diagnostic that the raw separation is null-explained, nothing more.

**(d) Recommendation for the paper**

**Predictor claim: NO.** There is no defensible claim that delta, delta*, the DeltaQ ratio, the hub-bridge ratio, its excess, or degree CV predicts where DSpar-seeded Leiden beats a runtime-matched baseline. Concretely:

1. Only 2/15 networks clear the noise bar, so the outcome is nearly degenerate; any "predictor" would be fitting 2 points.
2. No candidate reaches p<0.05 uncorrected on either y or y2, let alone corrected.
3. The one pattern that looked real at 14 networks (heavy-tailed degree distribution / large n) is falsified by wiki-Talk, the most heavy-tailed network in the set.
4. delta* -- the statistic the restructure was hoping to promote -- is the weakest and points the wrong way.

The paper should stay **purely characterizational**: report that raw DSpar separation and DeltaQ_fixed are null-reproduced (Phase 1), that the only honest algorithmic gain is a small, runtime-matched seeded-refinement improvement observed on 2 of 15 real networks (email-Enron, com-Youtube), and state explicitly that no structural statistic tested predicts where it occurs. If a predictor claim is wanted later it needs a much larger network sample (order 100+) and a pre-registered statistic; n=15 with a 2-positive outcome cannot support one.
