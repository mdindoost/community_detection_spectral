# Experiment F — does the draft's Exp-3 ground-truth result survive chance correction and a granularity control?

`baseline` = Leiden (ModularityVertexPartition) on the original graph. `dspar_paper_08` = the draft's pipeline: DSpar method="paper" (with replacement + reweight), nominal retention 0.8, weights dropped. `resmatch_X` = **no sparsification**, Leiden on the ORIGINAL graph with RBConfiguration resolution tuned so the cluster count matches X (granularity control). `dspar_cal_09` = calibrated sampler with E[true retention] = 0.90 exactly.

10 seeds per cell. AMI = adjusted mutual information (chance-corrected, the metric that is *not* inflated by returning more clusters); NMI is the draft's metric.


## Graphs

| dataset | n (LCC) | m | ground-truth communities |
|---|---|---|---|
| Karate | 34 | 78 | 2 |
| Dolphins | 62 | 159 | 2 |
| Football | 115 | 613 | 12 |
| Polbooks | 105 | 441 | 3 |
| email-Eu-core | 986 | 16064 | 42 |

## Results (mean ± std over 10 seeds; Δ vs baseline)

| dataset | condition | true ret. | γ | n_clusters | AMI | ΔAMI | ARI | ΔARI | NMI | ΔNMI |
|---|---|---|---|---|---|---|---|---|---|---|
| Karate | baseline | 1.000 | 1.000 | 4.0±0.0 | 0.5573±0.0281 | — | 0.4573±0.0219 | — | 0.5788±0.0270 | — |
| Karate | dspar_paper_08 | 0.541 | 1.000 | 7.3±1.7 | 0.3661±0.0588 | -0.1912 | 0.2943±0.0730 | -0.1630 | 0.4292±0.0502 | -0.1497 |
| Karate | resmatch_dspar_paper_08 | 1.000 | 2.000 | 7.3±0.5 | 0.3921±0.0053 | -0.1652 | 0.2389±0.0094 | -0.2183 | 0.4454±0.0028 | -0.1334 |
| Karate | dspar_cal_09 | 0.887 | 1.000 | 4.0±0.0 | 0.4925±0.0377 | -0.0648 | 0.4159±0.0306 | -0.0414 | 0.5171±0.0360 | -0.0618 |
| Dolphins | baseline | 1.000 | 1.000 | 5.0±0.4 | 0.2702±0.0287 | — | 0.2460±0.0327 | — | 0.2930±0.0269 | — |
| Dolphins | dspar_paper_08 | 0.520 | 1.000 | 9.7±1.2 | 0.2158±0.0452 | -0.0545 | 0.1324±0.0287 | -0.1136 | 0.2645±0.0387 | -0.0285 |
| Dolphins | resmatch_dspar_paper_08 | 1.000 | 2.300 | 10.3±0.9 | 0.2231±0.0179 | -0.0471 | 0.1251±0.0096 | -0.1208 | 0.2704±0.0137 | -0.0226 |
| Dolphins | dspar_cal_09 | 0.892 | 1.000 | 4.9±0.5 | 0.2614±0.0211 | -0.0089 | 0.1952±0.0474 | -0.0507 | 0.2836±0.0192 | -0.0094 |
| Football | baseline | 1.000 | 1.000 | 9.7±0.5 | 0.8486±0.0184 | — | 0.7789±0.0472 | — | 0.8804±0.0160 | — |
| Football | dspar_paper_08 | 0.549 | 1.000 | 9.2±0.9 | 0.8119±0.0315 | -0.0368 | 0.7367±0.0590 | -0.0422 | 0.8494±0.0278 | -0.0311 |
| Football | resmatch_dspar_paper_08 | 1.000 | 1.000 | 9.7±0.5 | 0.8486±0.0184 | +0.0000 | 0.7789±0.0472 | +0.0000 | 0.8804±0.0160 | +0.0000 |
| Football | dspar_cal_09 | 0.899 | 1.000 | 8.9±0.3 | 0.8165±0.0125 | -0.0322 | 0.7022±0.0266 | -0.0767 | 0.8524±0.0113 | -0.0280 |
| Polbooks | baseline | 1.000 | 1.000 | 4.7±0.5 | 0.5351±0.0257 | — | 0.6248±0.0614 | — | 0.5510±0.0241 | — |
| Polbooks | dspar_paper_08 | 0.527 | 1.000 | 6.8±0.7 | 0.4129±0.0425 | -0.1223 | 0.3406±0.0690 | -0.2842 | 0.4388±0.0405 | -0.1122 |
| Polbooks | resmatch_dspar_paper_08 | 1.000 | 1.500 | 6.8±0.6 | 0.4307±0.0122 | -0.1044 | 0.3334±0.0211 | -0.2914 | 0.4557±0.0103 | -0.0953 |
| Polbooks | dspar_cal_09 | 0.900 | 1.000 | 5.2±0.7 | 0.4754±0.0282 | -0.0597 | 0.4898±0.0880 | -0.1350 | 0.4942±0.0252 | -0.0568 |
| email-Eu-core | baseline | 1.000 | 1.000 | 7.7±0.5 | 0.5577±0.0104 | — | 0.3227±0.0212 | — | 0.5826±0.0108 | — |
| email-Eu-core | dspar_paper_08 | 0.443 | 1.000 | 8.6±0.5 | 0.5651±0.0135 | +0.0074 | 0.3600±0.0264 | +0.0373 | 0.5921±0.0134 | +0.0094 |
| email-Eu-core | resmatch_dspar_paper_08 | 1.000 | 1.075 | 8.6±0.9 | 0.5715±0.0171 | +0.0138 | 0.3614±0.0430 | +0.0387 | 0.5980±0.0180 | +0.0154 |
| email-Eu-core | dspar_cal_09 | 0.900 | 1.000 | 7.3±0.6 | 0.5490±0.0238 | -0.0087 | 0.2978±0.0367 | -0.0249 | 0.5731±0.0245 | -0.0095 |

## Sparsification vs its granularity-matched control

If `resmatch` matches or beats the sparsified run, the sparsifier contributed nothing beyond changing partition granularity.

| dataset | condition | AMI (sparse) | AMI (resmatch) | diff | ARI (sparse) | ARI (resmatch) | diff | NMI (sparse) | NMI (resmatch) | diff |
|---|---|---|---|---|---|---|---|---|---|---|
| Karate | dspar_paper_08 | 0.3661 | 0.3921 | -0.0260 | 0.2943 | 0.2389 | +0.0553 | 0.4292 | 0.4454 | -0.0163 |
| Dolphins | dspar_paper_08 | 0.2158 | 0.2231 | -0.0074 | 0.1324 | 0.1251 | +0.0073 | 0.2645 | 0.2704 | -0.0059 |
| Football | dspar_paper_08 | 0.8119 | 0.8486 | -0.0368 | 0.7367 | 0.7789 | -0.0422 | 0.8494 | 0.8804 | -0.0311 |
| Polbooks | dspar_paper_08 | 0.4129 | 0.4307 | -0.0178 | 0.3406 | 0.3334 | +0.0072 | 0.4388 | 0.4557 | -0.0169 |
| email-Eu-core | dspar_paper_08 | 0.5651 | 0.5715 | -0.0064 | 0.3600 | 0.3614 | -0.0014 | 0.5921 | 0.5980 | -0.0059 |

## Seed-paired deltas (10 matched seeds, paired t)

`vs baseline` answers "is the sparsified run better than plain Leiden?"; `vs resmatch` answers "is it better than plain Leiden made equally fine-grained?"

| dataset | comparison | metric | mean Δ | std | t | wins |
|---|---|---|---|---|---|---|
| Karate | dspar_paper_08 vs baseline | AMI | -0.1912 | 0.0745 | -8.12 | 0/10 |
| Karate | dspar_paper_08 vs baseline | ARI | -0.1630 | 0.0837 | -6.16 | 0/10 |
| Karate | dspar_paper_08 vs baseline | NMI | -0.1497 | 0.0641 | -7.38 | 0/10 |
| Karate | dspar_cal_09 vs baseline | AMI | -0.0648 | 0.0471 | -4.35 | 1/10 |
| Karate | dspar_cal_09 vs baseline | ARI | -0.0414 | 0.0392 | -3.34 | 1/10 |
| Karate | dspar_cal_09 vs baseline | NMI | -0.0618 | 0.0451 | -4.33 | 1/10 |
| Karate | dspar_paper_08 vs resmatch | AMI | -0.0260 | 0.0616 | -1.33 | 3/10 |
| Karate | dspar_paper_08 vs resmatch | ARI | +0.0553 | 0.0771 | +2.27 | 6/10 |
| Karate | dspar_paper_08 vs resmatch | NMI | -0.0163 | 0.0521 | -0.99 | 3/10 |
| Karate | resmatch vs baseline | AMI | -0.1652 | 0.0279 | -18.72 | 0/10 |
| Karate | resmatch vs baseline | ARI | -0.2183 | 0.0246 | -28.02 | 0/10 |
| Karate | resmatch vs baseline | NMI | -0.1334 | 0.0286 | -14.73 | 0/10 |
| Dolphins | dspar_paper_08 vs baseline | AMI | -0.0545 | 0.0503 | -3.42 | 2/10 |
| Dolphins | dspar_paper_08 vs baseline | ARI | -0.1136 | 0.0503 | -7.14 | 0/10 |
| Dolphins | dspar_paper_08 vs baseline | NMI | -0.0285 | 0.0434 | -2.08 | 2/10 |
| Dolphins | dspar_cal_09 vs baseline | AMI | -0.0089 | 0.0396 | -0.71 | 2/10 |
| Dolphins | dspar_cal_09 vs baseline | ARI | -0.0507 | 0.0549 | -2.92 | 1/10 |
| Dolphins | dspar_cal_09 vs baseline | NMI | -0.0094 | 0.0371 | -0.80 | 2/10 |
| Dolphins | dspar_paper_08 vs resmatch | AMI | -0.0074 | 0.0474 | -0.49 | 3/10 |
| Dolphins | dspar_paper_08 vs resmatch | ARI | +0.0073 | 0.0285 | +0.81 | 4/10 |
| Dolphins | dspar_paper_08 vs resmatch | NMI | -0.0059 | 0.0409 | -0.46 | 3/10 |
| Dolphins | resmatch vs baseline | AMI | -0.0471 | 0.0324 | -4.59 | 0/10 |
| Dolphins | resmatch vs baseline | ARI | -0.1208 | 0.0329 | -11.62 | 0/10 |
| Dolphins | resmatch vs baseline | NMI | -0.0226 | 0.0287 | -2.49 | 3/10 |
| Football | dspar_paper_08 vs baseline | AMI | -0.0368 | 0.0331 | -3.51 | 2/10 |
| Football | dspar_paper_08 vs baseline | ARI | -0.0422 | 0.0563 | -2.37 | 3/10 |
| Football | dspar_paper_08 vs baseline | NMI | -0.0311 | 0.0289 | -3.40 | 2/10 |
| Football | dspar_cal_09 vs baseline | AMI | -0.0322 | 0.0253 | -4.02 | 2/10 |
| Football | dspar_cal_09 vs baseline | ARI | -0.0767 | 0.0621 | -3.91 | 2/10 |
| Football | dspar_cal_09 vs baseline | NMI | -0.0280 | 0.0223 | -3.98 | 2/10 |
| Football | dspar_paper_08 vs resmatch | AMI | -0.0368 | 0.0331 | -3.51 | 2/10 |
| Football | dspar_paper_08 vs resmatch | ARI | -0.0422 | 0.0563 | -2.37 | 3/10 |
| Football | dspar_paper_08 vs resmatch | NMI | -0.0311 | 0.0289 | -3.40 | 2/10 |
| Football | resmatch vs baseline | AMI | +0.0000 | 0.0000 | n/a | 0/10 |
| Football | resmatch vs baseline | ARI | +0.0000 | 0.0000 | n/a | 0/10 |
| Football | resmatch vs baseline | NMI | +0.0000 | 0.0000 | n/a | 0/10 |
| Polbooks | dspar_paper_08 vs baseline | AMI | -0.1223 | 0.0532 | -7.27 | 0/10 |
| Polbooks | dspar_paper_08 vs baseline | ARI | -0.2842 | 0.1049 | -8.57 | 0/10 |
| Polbooks | dspar_paper_08 vs baseline | NMI | -0.1122 | 0.0507 | -6.99 | 0/10 |
| Polbooks | dspar_cal_09 vs baseline | AMI | -0.0597 | 0.0388 | -4.87 | 0/10 |
| Polbooks | dspar_cal_09 vs baseline | ARI | -0.1350 | 0.1028 | -4.15 | 0/10 |
| Polbooks | dspar_cal_09 vs baseline | NMI | -0.0568 | 0.0354 | -5.07 | 0/10 |
| Polbooks | dspar_paper_08 vs resmatch | AMI | -0.0178 | 0.0361 | -1.56 | 2/10 |
| Polbooks | dspar_paper_08 vs resmatch | ARI | +0.0072 | 0.0718 | +0.32 | 5/10 |
| Polbooks | dspar_paper_08 vs resmatch | NMI | -0.0169 | 0.0359 | -1.49 | 2/10 |
| Polbooks | resmatch vs baseline | AMI | -0.1044 | 0.0338 | -9.77 | 0/10 |
| Polbooks | resmatch vs baseline | ARI | -0.2914 | 0.0702 | -13.13 | 0/10 |
| Polbooks | resmatch vs baseline | NMI | -0.0953 | 0.0302 | -9.99 | 0/10 |
| email-Eu-core | dspar_paper_08 vs baseline | AMI | +0.0074 | 0.0186 | +1.26 | 7/10 |
| email-Eu-core | dspar_paper_08 vs baseline | ARI | +0.0373 | 0.0448 | +2.63 | 8/10 |
| email-Eu-core | dspar_paper_08 vs baseline | NMI | +0.0094 | 0.0194 | +1.54 | 7/10 |
| email-Eu-core | dspar_cal_09 vs baseline | AMI | -0.0087 | 0.0248 | -1.11 | 3/10 |
| email-Eu-core | dspar_cal_09 vs baseline | ARI | -0.0249 | 0.0402 | -1.96 | 3/10 |
| email-Eu-core | dspar_cal_09 vs baseline | NMI | -0.0095 | 0.0257 | -1.17 | 3/10 |
| email-Eu-core | dspar_paper_08 vs resmatch | AMI | -0.0064 | 0.0227 | -0.88 | 5/10 |
| email-Eu-core | dspar_paper_08 vs resmatch | ARI | -0.0014 | 0.0458 | -0.10 | 6/10 |
| email-Eu-core | dspar_paper_08 vs resmatch | NMI | -0.0059 | 0.0227 | -0.83 | 5/10 |
| email-Eu-core | resmatch vs baseline | AMI | +0.0138 | 0.0239 | +1.82 | 3/10 |
| email-Eu-core | resmatch vs baseline | ARI | +0.0387 | 0.0638 | +1.92 | 3/10 |
| email-Eu-core | resmatch vs baseline | NMI | +0.0154 | 0.0261 | +1.86 | 3/10 |

## Verdict


**(a) Does email-Eu-core's gain survive chance-adjusted metrics?**
Only in the weakest sense. Under the draft's own metric the effect shrinks by half at the
LCC-restricted, 10-seed replication (draft: ΔNMI=+0.025, ΔARI=+0.071; here ΔNMI=+0.009,
ΔARI=+0.037). Under the chance-corrected AMI it is ΔAMI=+0.007 ± 0.019 (paired t=+1.26,
7/10 seeds), i.e. statistically indistinguishable from zero. ΔARI=+0.037 ± 0.045
(t=+2.63) is the only nominally significant piece, and it is not robust to (b).

**(b) Does it survive the resolution-matched control?**
No. Leiden on the UNSPARSIFIED graph, with the resolution tuned to the same cluster
count DSpar produces (7.7 -> 8.6 clusters), reproduces the entire gain and slightly
exceeds it: ΔAMI=+0.014, ΔARI=+0.039, ΔNMI=+0.015 vs baseline — larger than DSpar's
+0.007/+0.037/+0.009. Sparsified minus resolution-matched is NEGATIVE on all three
metrics (ΔAMI=-0.006, ΔARI=-0.001, ΔNMI=-0.006; |t| <= 0.9). The email-Eu-core
"improvement" is a granularity effect available for free by turning one knob, at 100%
of the edges; DSpar throws away 56% of the distinct edges to buy nothing.

**(c) Do the four negative datasets stay negative?**
Yes, all four, on every metric, and more strongly under AMI than under NMI.
ΔAMI: Karate -0.191, Polbooks -0.122, Dolphins -0.055, Football -0.037
(paired t = -8.1, -7.3, -3.4, -3.5; 0-2 winning seeds out of 10). The calibrated
true-retention-0.90 sampler — which does not shred the graph — is also negative on all
four (ΔAMI -0.065, -0.060, -0.009, -0.032) and negative on email-Eu-core too
(ΔAMI=-0.009), so the single positive result depends on the with-replacement sampler's
~44% true retention, not on DSpar's degree-based scores.

**(d) Implication for the Exp-3 rewrite.**
Score is 0/5, not 1/5: once recovery is measured with a chance-corrected metric and
compared against an equally fine-grained unsparsified partition, DSpar does not improve
ground-truth recovery on any of the draft's five datasets — the email-Eu-core cell
should be reported as a partition-granularity artifact of with-replacement sampling
(nominal alpha=0.8 -> 0.44 true retention), with the resolution-matched control as the
correct baseline, which retires the delta>0 "favorable regime" claim built on it.

