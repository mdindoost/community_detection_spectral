# Exp L — L-Spar (Satuluri, Parthasarathy & Ruan, SIGMOD 2011) under the full protocol

Files: run.py, results.csv (14 rows), null_arm.csv (12 rows), recovery.csv (44 rows).
Logs (gitignored, local only): main.log, null.log, recovery_email.log, recovery_gt.log.
Coverage: main arm 7/7 networks x 2 retention targets; config-null arm 3/3 networks;
recovery 3/3 labelled networks x 2 targets, every cell granularity-controlled.

## Verdict (four parts)

**V1 — Under a modularity objective and a resolution-free optimizer, L-Spar loses in 14/14 cells.**
Scored on the ORIGINAL graph, the L-Spar partition is worse than plain Leiden in
every network at every retention: `dQ_vs_matched` runs -0.0142 to -0.4035, `dQ_honest_vs_best`
-0.0117 to -0.4066 (results.csv). Reading modularity off the sparsified graph flips the sign in
all 14 cells (`dQ_naive` +0.044..+0.389; `dQ_fixed` +0.022..+0.322) — a clean demonstration of
Artifact I under the founding sparsifier.

**SCOPE CORRECTION (2026-07-26, after reading the primary source — see
references/NOTES_satuluri2011.md).** This is NOT a refutation of Satuluri et al. 2011, for three
reasons established by reading their paper in full:
(a) **They do not commit Artifact I.** §4.3: "we cannot simply measure the conductance using the
very same sparsified graph, since that would tell us nothing about how well the sparsified graph
retained the cluster structure in the original graph." Table 2 caption: "phi_avg is always
calculated w.r.t. the original graph." Their other metric is external ground truth, where
Artifact I is structurally impossible. Artifact I is a pitfall of the literature that FOLLOWED
them. They also match cluster counts by construction (k is an input to Metis/Graclus/Metis+MQI)
and charge sparsification time to their speedups.
(b) **They never claim a modularity improvement.** Their accuracy claim is agreement with
external ground truth, for two named algorithms (Metis 4/4, Graclus 3/3). Their own
graph-internal objective results (conductance, honestly transferred) are 5W/5L/1T — a coin flip
they print in full.
(c) **We tested outside their regime.** Their LFR sweep (§4.5) states L-Spar "actually outperforms
the original clustering starting from degree 50." All seven of our networks have average degree
5.53-32.58. On their own two low-degree datasets (DIP 6.4, Human 10.8) their eight F-score deltas
are +1.54/+1.09/-0.47/-0.12/-0.16/+0.11/+0.68/+0.37 — a wash, matching ours. Where our regimes
overlap, we agree with them.
The honest statement of V1 is therefore: *L-Spar does not improve modularity for a resolution-free
optimizer on low-degree graphs* — a regime the original authors did not claim.

**V2 — Mechanism IS structure-aware, and this is the first sparsifier in the study whose
selection signal the configuration null does NOT reproduce.** L-Spar's Jaccard separation
deltaJ = meanJ_intra - meanJ_inter is +0.0772..+0.1778 on real graphs and collapses to
-0.0089 / -0.0056 / +0.00012 on degree-preserving rewirings (null_arm.csv). Preferential
retention of intra-community edges drops from 2.4-3.0x (real) to 1.16-1.46x (null) at
retention 0.5. The signal is a property of real community structure, not of the degree
sequence. CRITICAL COROLLARY: `dQ_fixed` is *fully* reproduced by the null (email-Enron
target 0.2: null +0.31487 exceeds real +0.25908; ca-CondMat: null +0.12641 vs real +0.18721),
so a positive `dQ_fixed` is no evidence of structure-awareness whatsoever — only deltaJ and
the retention ratio are.

**V3 — Ground-truth recovery: the granularity control kills 5 of 6 cells. The exception is
com-Amazon at moderate retention, where L-Spar's gain SURVIVES an over-matched control.**
Every apparent recovery gain on email-Eu-core and com-DBLP is Artifact II (granularity):
matching the baseline's resolution to L-Spar's cluster count reproduces or beats it. But on
com-Amazon at realized retention 0.542, L-Spar scores avgF1_ge3 = 0.4019 +- 0.0003 against a
control that was given MORE clusters and still reached only 0.3425-0.3555. This is the only
granularity-controlled positive in the study, and it coexists with a modularity LOSS in the
same cell (dQ_vs_matched = -0.0789): the two criteria point in opposite directions.

**V4 — In our regime there is no speedup, and exact Jaccard is part of why.**
End-to-end `speedup_vs_single_leiden` = 0.87-1.35x across all 14 cells, and it is BELOW 1.0
on the two largest graphs (com-DBLP 0.906, com-Amazon 0.880 at target 0.2). Sparsification is not
free: T_jaccard is 2.8%-46.8% of pipeline wall clock. Even Leiden-in-isolation is only 0.92-2.59x.
**The paper's 10-50x is not reachable here, and the reason is REGIME, NOT ACCOUNTING** — they
charge sparsification time exactly as we do (Table 2 caption: speedups "take into account both the
sparsification as well as the clustering times"), and they disclose their own slowdowns
(Metis+MQI Wiki 0.46x, Orkut 0.7x). The gap decomposes into: (i) algorithm class — their baselines
are 2-10 hour runs of Metis/Metis+MQI/MLR-MCL on 53-117M-edge graphs (Metis+MQI on Wiki = 35,511 s)
vs our 12.4 s near-linear Leiden on <=1.05M edges; (ii) retention — they ran 0.04-0.17, we ran
0.30-0.54 realized, whose cost ceiling is 1.9-3.3x and our observed 0.92-2.59x sits inside it;
(iii) a ~12x "clearer graph converges faster" effect they document (Metis 80 s on L-Spar-Wiki vs
940 s on RandomEdge-Wiki AT EQUAL EDGE COUNT) **which reverses sign for Leiden** at aggressive
retention, because the shattered graph yields 72,736/57,788 communities and the optimizer does more
work. That sign reversal is a new observation worth stating: the structure-clarity runtime effect
is positive for cut-based partitioners and negative for modularity optimizers pushed past their
natural resolution.

## Findings with pointers

### 1. Honest-transfer quality, all 7 networks x 2 targets (results.csv)

| network | target | realized | status | e | Q_base_mean | Q_orig(P_L) | dQ_honest_vs_mean | dQ_honest_vs_best | dQ_vs_matched | restarts |
|---|---|---|---|---|---|---|---|---|---|---|
| email-Eu-core | 0.5 | 0.5021 | ok | 0.7266 | 0.41605 | 0.38793 | -0.0281 | -0.0285 | **-0.0267** | 1 |
| email-Eu-core | 0.2 | 0.2014 | ok | 0.4219 | 0.41605 | 0.35880 | -0.0573 | -0.0576 | **-0.0558** | 1 |
| wiki-Vote | 0.5 | 0.5018 | ok | 0.7188 | 0.42443 | 0.41512 | -0.0093 | -0.0117 | **-0.0142** | 1 |
| wiki-Vote | 0.2 | 0.1970 | ok | 0.4062 | 0.42443 | 0.37973 | -0.0447 | -0.0471 | **-0.0496** | 1 |
| ca-HepTh | 0.5 | 0.5006 | ok | 0.2656 | 0.76137 | 0.68906 | -0.0723 | -0.0733 | **-0.0730** | 1 |
| ca-HepTh | 0.2 | 0.2760 | floor | 0.0 | 0.76137 | 0.45290 | -0.3085 | -0.3095 | **-0.3092** | 1 |
| ca-CondMat | 0.5 | 0.4972 | ok | 0.5000 | 0.73079 | 0.66347 | -0.0673 | -0.0690 | **-0.0659** | 1 |
| ca-CondMat | 0.2 | 0.1862 | (floor) | 0.0 | 0.73079 | 0.32591 | -0.4049 | -0.4066 | **-0.4035** | 1 |
| email-Enron | 0.5 | 0.5054 | ok | 0.6309 | 0.60710 | 0.58518 | -0.0219 | -0.0242 | **-0.0199** | 1 |
| email-Enron | 0.2 | 0.1586 | (floor) | 0.0 | 0.60710 | 0.21785 | -0.3892 | -0.3915 | **-0.3873** | 1 |
| com-DBLP | 0.5 | 0.5030 | ok | 0.3750 | 0.82570 | 0.72648 | -0.0992 | -0.1001 | **-0.1011** | 1 |
| com-DBLP | 0.2 | 0.2441 | floor | 0.0 | 0.82570 | 0.42826 | -0.3974 | -0.3983 | **-0.3993** | 2 |
| com-Amazon | 0.5 | 0.5418 | ok | 0.0625 | 0.92944 | 0.85071 | -0.0787 | -0.0791 | **-0.0789** | 1 |
| com-Amazon | 0.2 | 0.3006 | floor | 0.0 | 0.92944 | 0.53118 | -0.3983 | -0.3986 | **-0.3984** | 2 |

Mildest cell wiki-Vote@0.5 (-0.0142); worst ca-CondMat@0.2 (-0.4035). Not one positive.
Columns: `dQ_honest_vs_mean`, `dQ_honest_vs_best`, `dQ_vs_matched`, `n_restarts` in results.csv.

### 2. The e-bisection hits a hard retention floor on 5 of 7 networks
`min_ret` = retention at e=0 (each node keeps only its single highest-Jaccard incident edge;
an edge survives if either endpoint picks it): email-Eu-core 0.0533, wiki-Vote 0.0673,
email-Enron 0.1586, ca-CondMat 0.1862, com-DBLP 0.2441, ca-HepTh 0.2760, com-Amazon 0.3006
(results.csv `min_ret`). Only email-Eu-core and wiki-Vote can actually reach a 20% target;
the other five bottom out at e=0.0 (`e_used`=0.0 in results.csv). CAVEAT ON THE FLAG:
`status="floor"` is set only when target <= min_ret, so ca-CondMat@0.2 (0.1862) and
email-Enron@0.2 (0.1586) read "ok" while sitting at the floor — read `e_used`==0.0, not
`status`. Second bisection caveat: com-Amazon@0.5 realized 0.5418, 8% above target, because
ceil(d^e) makes retention a step function of e (retention at e=0.5 is 0.6746 there). It is the
only "ok" row materially off target; L-Spar removes 46%, not 50%, of com-Amazon's edges.

### 3. Configuration-model null: deltaJ is real, dQ_fixed is not (null_arm.csv)

| network | target | arm | meanJ_all | deltaJ | pres_ratio | dQ_fixed |
|---|---|---|---|---|---|---|
| email-Eu-core | 0.5 | real | 0.16454 | **+0.09161** | 2.719 | +0.21710 |
| email-Eu-core | 0.5 | null | 0.06479 | **-0.00889** | 1.160 | +0.02796 |
| email-Eu-core | 0.2 | real | 0.16454 | +0.09161 | 5.911 | +0.32169 |
| email-Eu-core | 0.2 | null | 0.06479 | -0.00889 | 1.492 | +0.08101 |
| ca-CondMat | 0.5 | real | 0.22651 | **+0.17776** | 3.038 | +0.14794 |
| ca-CondMat | 0.5 | null | 0.00074 | **+0.00012** | 1.224 | +0.06500 |
| ca-CondMat | 0.2 | real | 0.22651 | +0.17776 | 5.352 | +0.18721 |
| ca-CondMat | 0.2 | null | 0.00074 | +0.00012 | 1.518 | +0.12641 |
| email-Enron | 0.5 | real | 0.11024 | **+0.07723** | 2.431 | +0.16079 |
| email-Enron | 0.5 | null | 0.00845 | **-0.00562** | 1.462 | +0.09487 |
| email-Enron | 0.2 | real | 0.11024 | +0.07723 | 6.632 | +0.25908 |
| email-Enron | 0.2 | null | 0.00845 | -0.00562 | 3.505 | **+0.31487** |

Three separate collapses under rewiring: mean Jaccard itself (0.2265 -> 0.00074 on ca-CondMat,
306x), the intra/inter separation deltaJ (sign flips on two of three networks), and preferential
retention (3.04x -> 1.22x). Contrast the bolded final cell: the null's dQ_fixed (+0.31487)
EXCEEDS the real graph's (+0.25908). Rewired graphs also gain fixed-partition modularity under
L-Spar, so dQ_fixed measures the sparsification-induced density change, not community awareness.
Config null is n=3 networks with a single rewiring realisation (REWIRE_SEED=42, 10 swaps/edge).

### 4. Recovery under the granularity control (recovery.csv)

email-Eu-core, department labels, chance-corrected:

| condition | seeds | mean k | AMI | ARI | NMI | gamma |
|---|---|---|---|---|---|---|
| baseline | 5 | 8.0 | 0.5626 +- 0.0077 | 0.3351 +- 0.0118 | 0.5882 | 1.0 |
| lspar_0.5 | 3 | 11.7 | 0.6123 +- 0.0060 | 0.4220 +- 0.0090 | 0.6431 | 1.0 |
| **resmatch_0.5** | 3 | 12.0 | **0.6189 +- 0.0161** | **0.4697 +- 0.0371** | **0.6502** | 1.25-1.5 |
| lspar_0.2 | 3 | 17.7 | 0.6442 +- 0.0097 | 0.4713 +- 0.0193 | 0.6833 | 1.0 |
| **resmatch_0.2** | 3 | 18.7 | **0.6478 +- 0.0035** | **0.5210 +- 0.0060** | **0.6866** | 1.75 |

The headline "L-Spar lifts AMI 0.563 -> 0.612/0.644" is true only against the unmatched
baseline. Simply turning Leiden's resolution knob to the same k gets there and further, on all
three metrics at both targets. ARI is the decisive column (+0.048 and +0.050 for the control).
This is the exact Artifact II pattern already established for DSpar.

com-DBLP, 5000 top SNAP communities, avgF1_ge3 (MIN_GT_SIZE=3):

| condition | seeds | k (total) | k>=3 | avgF1_ge3 | gt2det | det2gt | gamma |
|---|---|---|---|---|---|---|---|
| baseline | 3 | 227.0 | 227 | 0.1149 +- 0.0043 | 0.018 | 0.212 | 1.0 |
| lspar_0.5 | 3 | 10947.7 | 10948 | 0.1930 +- 0.0001 | 0.267 | 0.119 | 1.0 |
| **resmatch_0.5** | 1 | 10812 | 10621 | **0.2971** | 0.422 | 0.172 | 576 |
| lspar_0.2 | 3 | 72736 | 47239 | 0.3760 +- 0.0000 | 0.656 | 0.096 | 1.0 |
| **resmatch_0.2** | 1 | 46081 | 40833 | **0.3967** | 0.692 | 0.102 | 7168 |

THE CRITICAL RESULT. avgF1_ge3 rises monotonically with fragmentation — 0.1149 at k=227,
0.1930 at k=10,948, 0.3760 at k=72,736 — and NONE of it is attributable to L-Spar. At matched
granularity plain Leiden scores 0.2971 (vs L-Spar's 0.1930, **-0.1041** for L-Spar) and 0.3967
(vs 0.3760, **-0.0207**), the latter while holding 14% FEWER >=3-node clusters (40,833 vs
47,239), i.e. the control is handicapped and still wins. Best-match average-F1 against many
small reference communities mechanically rewards shattering; without the resolution-matched
arm this cell would have read as a large L-Spar win.

com-Amazon, 5000 top SNAP communities, avgF1_ge3 — THE ONE EXCEPTION:

| condition | seeds | k (total) | k>=3 | avgF1_ge3 | gt2det | det2gt | gamma |
|---|---|---|---|---|---|---|---|
| baseline | 3 | 306.3 | 306 | 0.1827 +- 0.0090 | 0.097 | 0.269 | 1.0 |
| **lspar_0.5** | 3 | 8657.7 | 8658 | **0.4019 +- 0.0003** | 0.703 | 0.101 | 1.0 |
| resmatch_0.5 | 3 | 8404.3 | 8404 | 0.3352 +- 0.0037 | 0.586 | 0.084 | 320 |
| resmatch_0.5_sweep | 1 | 8824 | 8824 | 0.3425 | 0.602 | 0.083 | 352 |
| resmatch_0.5_sweep | 1 | 9222 | 9222 | 0.3424 | 0.605 | 0.080 | 384 |
| resmatch_0.5_sweep | 1 | 9989 | 9989 | 0.3555 | 0.634 | 0.077 | 448 |
| lspar_0.2 | 3 | 57788 | 45511 | 0.4130 +- 0.0000 | 0.796 | 0.030 | 1.0 |
| **resmatch_0.2** | 1 | 43974 | 41541 | **0.4419** | 0.853 | 0.030 | 7168 |

At target 0.5 the control LOSES, and it loses after being over-matched. The gamma sweep gives
the baseline 8,824 / 9,222 / 9,989 clusters against L-Spar's 8,658 — up to 15% MORE — and it
never exceeds 0.3555. Linear interpolation of the control curve to L-Spar's exact k=8,658
gives ~0.340, so L-Spar is **+0.062** at matched granularity, ~17 baseline seed-sigmas
(baseline sd 0.0090; control sd at fixed gamma 0.0037). The granularity slope over this range
is d(avgF1)/d(ln k) ~ 0.07, so the residual 2.7% k mismatch of the primary control accounts
for only ~0.002 of the 0.067 raw gap. At target 0.2 (retention floor, 0.3006) the control wins
again by +0.0289 despite holding 9% fewer >=3-node clusters.

Net: **1 of 6 granularity-controlled recovery cells favours L-Spar**, com-Amazon at realized
retention 0.542. In that same cell modularity is worse than the runtime-matched baseline
(dQ_vs_matched = -0.0789, results.csv) — a clean dissociation between the modularity objective
and reference-community recovery, and a caution against reading either as the other.

### 5. Cost accounting: Jaccard is the price (results.csv)

| network | T_leiden_orig | T_jaccard | T_rank | T_select | T_sparsify | T_leiden_sparse | T_pipeline | jaccard % | speedup_vs_single_leiden | speedup_leiden_only |
|---|---|---|---|---|---|---|---|---|---|---|
| email-Eu-core 0.5 | 0.059 | 0.019 | 0.003 | 0.0011 | 0.023 | 0.033 | 0.056 | 33.6 | 1.051 | 1.792 |
| email-Eu-core 0.2 | 0.059 | 0.019 | 0.003 | 0.0006 | 0.023 | 0.023 | 0.046 | 41.4 | 1.296 | 2.588 |
| wiki-Vote 0.5 | 0.437 | 0.201 | 0.024 | 0.0082 | 0.234 | 0.269 | 0.502 | 40.0 | **0.869** | 1.625 |
| wiki-Vote 0.2 | 0.437 | 0.201 | 0.024 | 0.0044 | 0.230 | 0.199 | 0.429 | 46.8 | 1.018 | 2.191 |
| ca-HepTh 0.5 | 0.300 | 0.016 | 0.005 | 0.0021 | 0.023 | 0.215 | 0.238 | 6.8 | 1.260 | 1.395 |
| ca-HepTh 0.2 | 0.300 | 0.016 | 0.005 | 0.0011 | 0.022 | 0.246 | 0.268 | 6.1 | 1.120 | 1.221 |
| ca-CondMat 0.5 | 0.868 | 0.052 | 0.021 | 0.0067 | 0.080 | 0.563 | 0.643 | 8.2 | **1.350** | 1.543 |
| ca-CondMat 0.2 | 0.868 | 0.052 | 0.021 | 0.0024 | 0.076 | 0.648 | 0.724 | 7.3 | 1.200 | 1.341 |
| email-Enron 0.5 | 1.235 | 0.278 | 0.040 | 0.0157 | 0.334 | 0.863 | 1.197 | 23.2 | 1.032 | 1.431 |
| email-Enron 0.2 | 1.235 | 0.278 | 0.040 | 0.0039 | 0.322 | 0.860 | 1.182 | 23.5 | 1.044 | 1.436 |
| com-DBLP 0.5 | 12.429 | 0.478 | 0.257 | 0.114 | 0.849 | 9.702 | 10.552 | 4.5 | 1.178 | 1.281 |
| com-DBLP 0.2 | 12.429 | 0.478 | 0.257 | 0.048 | 0.784 | 12.932 | 13.717 | 3.5 | **0.906** | 0.961 |
| com-Amazon 0.5 | 11.002 | 0.351 | 0.211 | 0.106 | 0.668 | 9.675 | 10.343 | 3.4 | 1.064 | 1.137 |
| com-Amazon 0.2 | 11.002 | 0.351 | 0.211 | 0.050 | 0.611 | 11.897 | 12.508 | 2.8 | **0.880** | 0.925 |

Seconds, means over seeds. Three things. (i) Nothing resembling 10-50x: 0.87-1.35x end to end.
(ii) The most aggressive retention is often SLOWER: on com-DBLP and com-Amazon at target 0.2,
Leiden on the 24%/30%-retained graph takes LONGER than on the original (0.96x and 0.92x
leiden-only), because the shattered graph produces 72,736 / 57,788 communities and the
optimiser does more work. Edge count is not the cost driver Leiden's runtime tracks.
(iii) T_jaccard alone (exact, not the paper's minhash approximation) is 2.8-46.8% of the
pipeline; we charge the exact version, which is more favourable to L-Spar on quality and
strictly more expensive on time, and note that a minhash implementation would reduce (iii)
but cannot rescue (i) since T_leiden_sparse alone already exceeds T_leiden_orig in two cells.
Cost-matching caveat: the budget buys the baseline only 1-2 restarts (`n_restarts` = 1 in 12
cells, 2 in the two com-DBLP/com-Amazon target-0.2 cells), so "matched best" here is
best-of-1-or-2 — a weak control that L-Spar nonetheless loses in all 14 cells.

### 6. L-Spar is deterministic; at the retention floor Leiden becomes deterministic too
L-Spar itself has no randomness (exact Jaccard, deterministic local top-ceil(d^e), union rule),
so no sparsification seeds exist and SPARSE_SEEDS vary only Leiden. Empirically Leiden's seed
variance vanishes at the retention floor: `Q_sparse_PL_std` = 0.000e+00 or 1.1e-16 for
ca-HepTh@0.2, ca-CondMat@0.2, com-DBLP@0.2, com-Amazon@0.2, and the recovery rows are
byte-identical (com-DBLP lspar_0.2 = 72736 / 0.3759817713988281 three times; com-Amazon
lspar_0.2 = 57788 / 0.412993174369652 three times). At target 0.5 variance is genuine on the
same graphs (com-DBLP k = 10950/10948/10945, avgF1 0.19307/0.19286/0.19293; com-Amazon
k = 8653/8658/8662, avgF1 0.40141/0.40223/0.40196). READ THE FLOOR ROWS AS n=1, NOT n=3:
the e=0 graph is so fragmented that Leiden's greedy phase has essentially one fixed point.
This does not affect any verdict — the floor cells are the ones L-Spar loses most heavily —
but no error bar from those rows may be quoted as a 3-seed interval.

## Caveats

C1. **No DESIGN.md exists for this experiment** (EXPLORATION.md rule 1). The protocol was
inherited wholesale from exps E/F/K rather than pre-registered for L-Spar, so nothing here was
predicted in advance. Treat V1/V2/V4 as confirmatory (they replicate the established DSpar
pattern under a new sparsifier) and V3's com-Amazon exception as what it is: an unpredicted
positive from a pre-planned control, i.e. a HYPOTHESIS in the sense of rule 2, not a result.
It needs its own pre-registered follow-up before it may be claimed.
C2. The com-Amazon exception rests on one network, one retention (0.542), one metric family
(best-match average F1 against SNAP top-5000 communities), and n=3 L-Spar seeds that are
near-degenerate (sd 0.0003). Its control is 3 seeds at fixed gamma plus a 3-point single-seed
gamma sweep. AMI/ARI were not computed for com-Amazon (the GT is overlapping; the code path
scores avgF1 only), so the exception has NOT been checked with a chance-corrected metric.
C3. `leiden_matched` bisects on TOTAL cluster count while the com-DBLP/com-Amazon target is a
>=3-node cluster count, so the controls there land 2-14% below L-Spar's k>=3. The bias is
conservative in every cell (the control is handicapped), and the com-Amazon gamma sweep
removes it explicitly for the one cell where it would have mattered.
C4. resmatch on com-DBLP is n=1 seed at each target (BASE_SEEDS[:1], per run.py), and
resmatch_0.2 on com-Amazon is likewise n=1. Both won, so under-powering does not threaten
those verdicts.
C5. Single-threaded leidenalg, N_ITER=2, ModularityVertexPartition. Timings are one machine,
one process, no repetition of the timing measurement itself.
C6. Exact Jaccard, not the paper's minhash. This deviates from Satuluri 2011 in L-Spar's favour
on quality and against it on time; both directions are stated in finding 5.
C7. Config null is 3/7 networks, one rewiring realisation.

## Effect on claims

- The "sparsify-then-detect improves quality" claim now fails for the *original, similarity-
  based* method, not only for degree-based DSpar and uniform sampling. The paper can stop
  saying "we test degree-based sparsification" and say "we test the founding method too".
- New, and the first positive result in the study for any sparsifier under full controls:
  L-Spar's Jaccard signal is genuinely structure-aware (V2) where DSpar's degree signal was
  reproduced by the configuration null. The mechanism is real even though the payoff is not.
  This sharpens the paper's thesis from "sparsification cannot see communities" to the more
  defensible and more interesting "seeing communities is necessary but not sufficient: L-Spar
  demonstrably sees them and still loses on modularity in 14/14 cells".
- `dQ_fixed` must be retired as evidence of anything (finding 3, bolded null cell).
- com-Amazon@0.542 is a live lead for a follow-up, not a claim. Suggested pre-registration:
  chance-corrected metrics on com-Amazon's overlapping GT, a retention sweep to locate the
  effect's range, and the same cell under Infomap/Louvain (folds naturally into Exp P).
