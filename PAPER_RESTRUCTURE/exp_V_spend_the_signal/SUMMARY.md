# Exp V — "spending the signal": can L-Spar's verified structure-awareness be
# converted into an honest gain?

Files: run.py, results.csv (49 rows = 7 networks x 7 arms), null_arm.csv (18 rows =
3 networks x {real,null} x 3 arms), recovery.csv (88 rows).
Logs (local): main_email-Eu-core.log, main_small.log, main_dblp.log, main_amazon.log,
null.log, recovery_email.log, recovery_dblp.log, recovery_amazon.log, forced_resmatch.log.
Coverage: main arm 7/7 networks x 7 arms at retention 0.5; config-null arm 3/3 networks
(arms B and C, real + rewired); recovery 3/3 labelled networks, every arm granularity-
controlled and chance-floored.
Pre-registered in DESIGN.md BEFORE any code existed. Loader, L-Spar, e-bisection,
runtime matching, resmatch and null conventions taken verbatim from exp_L_lspar/run.py;
seeded refinement from exp_N_enron_anatomy/run.py.

## VERDICT (five parts)

**V1 — The constructive claim (C17) is NOT dead, but it survives only on the recovery
branch of its own kill criterion, on one network.** The registered kill criterion had two
escape routes. The modularity route is CLOSED: no arm reaches dQ_vs_matched >= +0.002
beyond 2 pooled sd on >= 2/7 networks — the maximum for any arm is 1/7 (B_weight,
Bshuf_weight and C_seed, all on email-Enron; results.csv `sig_2sd`). The recovery route is
OPEN: arm C (seeding) on com-Amazon scores avgF1_ge3 = 0.2210 +- 0.0085 at k=320 against a
forced granularity control of 0.1848 +- 0.0098 at k=309 and a plain baseline of
0.1827 +- 0.0110 at k=306, with size-matched random-partition chance floors of 0.0149 vs
0.0123 — a chance-corrected +0.0356 at 4.0 pooled sd (recovery.csv, dataset=com-Amazon,
conditions `C_seed` / `resmatch_forced_C_seed` / `baseline` / `chance_*`). This is the
study's first arm to beat both a granularity control and a chance floor on a labelled
network. It is one network, one metric family, three seeds.

**V2 — The one modularity gain in the whole experiment is NOT the signal: the sharp null
reproduces it, and so does a configuration-model null.** On email-Enron, weighting w=1+J
reaches Q_orig = 0.615340 +- 0.003380 vs baseline 0.607097 and matched-best 0.605125
(dQ_vs_matched = +0.010215, sig). B-shuffle — the same weights permuted across edges, which
destroys every alignment between weight and structure — reaches 0.616819 +- 0.002006,
dQ_vs_matched = +0.011693, i.e. +0.00972 over baseline at 5.0 pooled sd (results.csv,
network=email-Enron, arms `B_weight` / `Bshuf_weight`). Pre-registered prediction P1c said
any B gain would COLLAPSE in B-shuffle; it does not — it grows. Independently, on the
ca-CondMat degree-preserving rewiring, where deltaJ = +0.000125 (i.e. no signal at all),
B_weight still achieves dQ_vs_matched = +0.002275 and clears the significance test
(null_arm.csv, network=ca-CondMat, graph=null, arm=B_weight). Two independent nulls both
manufacture the "gain". Similarity weighting's modularity benefit is weight heterogeneity
perturbing Leiden's search, not community information.

**V3 — Seeding is the only deployment whose advantage is structure-dependent, and it is
also the only one with any positive result.** Arm C's dQ_vs_matched on the three null-arm
networks: real -0.00054 / -0.00738 / +0.01028 (email-Eu-core / ca-CondMat / email-Enron)
against null -0.00128 / -0.01516 / -0.01203 (null_arm.csv, arm=C_seed). The null is worse
in all three, unlike B. C is also the arm carrying the com-Amazon recovery win (V1). But it
loses on modularity in 6/7 networks and costs 1.6-2.0x a single plain Leiden (finding 5).

**V4 — Protected deletion is a near-degenerate construction, and it wins exactly where it
is not degenerate.** As registered, arm D keeps L-Spar's selection at the largest e whose
retention <= 0.5 and tops up with uniformly random edges. On 6/7 networks L-Spar's step
function already lands within 0.6% of 0.5, so `frac_core` >= 0.964 and D is arithmetically
almost identical to pure L-Spar (results.csv `frac_core`, `ret_core`). Only com-Amazon,
where L-Spar's realized retention overshoots to 0.5418 and its floor sits at 0.3006, gives a
real 60/40 core/fill mixture — and there D is +0.0188 better than pure L-Spar in honest
transfer (0.869547 vs 0.850707) while holding 8% FEWER edges (462,936 vs 501,609). P3's
">= 5/7" clause therefore FAILS (3/7: email-Eu-core +0.00031, ca-CondMat +0.00140,
com-Amazon +0.01884), its "negative vs baseline on >= 6/7" clause holds at 7/7, and the
honest reading is that the prediction was untestable on 6 of 7 networks.

**V5 — com-Amazon's Exp L anomaly is PROMOTED from hypothesis to result.** L-Spar at
realized retention 0.5418 scores avgF1_ge3 = 0.40187 +- 0.00042 at k=8,658 (an exact
reproduction of Exp L's 0.4019). Resolution-matched and deliberately over-matched controls
reach only 0.3302 (k=8,424), 0.3424 (k=9,222), 0.3669 (k=10,695) and 0.3724 (k=11,863) —
the last with 37% MORE clusters than L-Spar (recovery.csv, conditions `resmatch_L_lspar05`
and `resmatch_sweep_L_lspar05`). The pre-registered size-matched random-partition chance
floor at L-Spar's own k is 0.05675 +- 0.00145 (`chance_L_lspar05`), and since the floor
rises with k (0.0123 at k=300 -> 0.0568 at k=8,653), subtracting it WIDENS L-Spar's margin
over the k=11,863 control from +0.0295 to about +0.034. Granularity, cluster-size
distribution and chance are all excluded. The effect is real; it coexists with a modularity
loss of -0.0789 in the same cell.

## Findings with pointers

### 1. Honest-transfer modularity, all 7 networks, retention 0.5 (results.csv)

dQ_vs_matched (arm mean Q on the ORIGINAL unweighted graph minus best-of-k plain restarts
fitting the arm's wall clock; k = `n_restarts`):

| network | B_weight | Bshuf_weight | C_seed | D_protected | L_lspar05 | U_uniform05 |
|---|---|---|---|---|---|---|
| email-Eu-core | -0.00127 | -0.00224 | -0.00054 | -0.02913 | -0.02943 | -0.01606 |
| wiki-Vote | -0.00749 | -0.00454 | -0.00217 | -0.01517 | -0.01425 | -0.02413 |
| ca-HepTh | -0.00015 | -0.00308 | -0.00557 | -0.07457 | -0.07304 | -0.14134 |
| ca-CondMat | -0.00024 | -0.00081 | -0.00738 | -0.06451 | -0.06591 | -0.08516 |
| email-Enron | **+0.01022** | **+0.01169** | **+0.01028** | -0.02567 | -0.01995 | -0.06634 |
| com-DBLP | -0.00243 | -0.00289 | -0.01028 | -0.10117 | -0.10106 | -0.11725 |
| com-Amazon | -0.00028 | -0.00088 | -0.00671 | -0.06003 | -0.07887 | -0.11961 |

Three bolded cells, one network, and the sharp null takes all three (V2). `sig_2sd`
(dQ_vs_matched >= +0.002 AND > 2 pooled sd) is 1 only in those three cells.
Cross-check on comparability with Exp L: `L_lspar05` reproduces exp_L_lspar/results.csv's
target-0.5 rows to 5 decimal places on every network (0.387930 / 0.415116 / 0.689063 /
0.663474 / 0.585180 / 0.726479 / 0.850707 vs Exp L's 0.38793 / 0.41512 / 0.68906 / 0.66347 /
0.58518 / 0.72648 / 0.85071), and `A_baseline` reproduces Q_base_mean (email-Eu-core
0.416048 vs 0.41605; com-DBLP 0.825701 vs 0.82570; com-Amazon 0.929438 vs 0.92944).

### 2. Selection vs deletion: the D / L / U ordering (results.csv `dQ_honest_vs_mean`)

All three arms remove ~50% of edges; U removes them uniformly at random, L by L-Spar's
Jaccard rule, D by L-Spar's rule topped up with random fill to exactly 0.5.

| network | D_protected | L_lspar05 | U_uniform05 | frac_core | ordering |
|---|---|---|---|---|---|
| email-Eu-core | -0.02781 | -0.02812 | **-0.01752** | 0.9998 | U > D > L |
| wiki-Vote | -0.01023 | -0.00931 | -0.01919 | 0.9999 | L > D > U |
| ca-HepTh | -0.07383 | -0.07230 | -0.14060 | 0.9963 | L > D > U |
| ca-CondMat | -0.06592 | -0.06732 | -0.08656 | 0.9943 | D > L > U |
| email-Enron | -0.02764 | -0.02192 | -0.06832 | 0.9644 | L > D > U |
| com-DBLP | -0.09933 | -0.09922 | -0.11541 | 0.9993 | L ~ D > U |
| com-Amazon | -0.05989 | -0.07873 | -0.11947 | 0.6011 | **D > L > U** |

Two readings. (i) Selection beats uniform deletion on 6/7 networks, often by a lot
(ca-HepTh -0.072 vs -0.141), so L-Spar's signal genuinely buys something *relative to
random removal* — consistent with Exp L's V2. email-Eu-core is the exception, where uniform
50% removal is the least harmful of the three. (ii) But selection never buys enough to reach
the baseline: every cell in this table is negative, and the runtime-matched comparison
(finding 1) is negative in all 21. Cluster counts explain part of the harm ordering: U
shatters the graph far more (email-Enron k = 6,946 for U vs 367/588 for L/D; com-DBLP
53,103 vs ~10,950; results.csv `nc_arm_mean`).

### 3. Sharp null and configuration null (results.csv, null_arm.csv)

B-shuffle vs baseline, in pooled-sd units (`Bshuf_dQ_vs_base / pooled_sd`): -0.78
(email-Eu-core), +0.14 (wiki-Vote), -2.08 (ca-HepTh), -2.37 (ca-CondMat), **+5.01**
(email-Enron), -1.14 (com-DBLP), -3.33 (com-Amazon). The single network where weighting
helps is the single network where shuffled weighting helps more.

Configuration-model null (degree-preserving rewiring, REWIRE_SEED=42, 10 swaps/edge):

| network | graph | meanJ | deltaJ | B dQ_vs_matched | C dQ_vs_matched |
|---|---|---|---|---|---|
| email-Eu-core | real | 0.16454 | +0.09161 | -0.00127 | -0.00054 |
| email-Eu-core | null | 0.06479 | -0.00889 | -0.00295 | -0.00128 |
| ca-CondMat | real | 0.22651 | +0.17776 | -0.00024 | -0.00738 |
| ca-CondMat | null | 0.00074 | **+0.00012** | **+0.00228 (sig)** | -0.01516 |
| email-Enron | real | 0.11024 | +0.07723 | **+0.01022 (sig)** | **+0.01028 (sig)** |
| email-Enron | null | 0.00845 | -0.00562 | -0.00323 | -0.01203 |

deltaJ collapses under rewiring exactly as Exp L found (this file re-derives it
independently). The decisive cell is ca-CondMat/null: a graph with no community structure
and no Jaccard separation still yields a "significant" weighting gain. Arm C never does
this — its null is worse than its real graph in all three networks.

### 4. Ground-truth recovery, all arms granularity-controlled and chance-floored (recovery.csv)

email-Eu-core, department labels, chance-corrected metrics, 3 seeds each:

| condition | k | AMI | ARI | NMI | gamma |
|---|---|---|---|---|---|
| baseline | 8.0 | 0.5664 +- 0.0096 | 0.3402 | 0.5917 | 1.0 |
| B_weight | 7.3 | 0.5509 +- 0.0165 | 0.2899 | 0.5752 | 1.0 |
| C_seed | 8.0 | 0.5684 +- 0.0061 | 0.3363 | 0.5936 | 1.0 |
| L_lspar05 | 11.7 | 0.6123 +- 0.0073 | 0.4220 | 0.6431 | 1.0 |
| **resmatch_L_lspar05** | 12.0 | **0.6189 +- 0.0197** | **0.4697** | **0.6502** | 1.375 |
| chance_baseline | 8.0 | 0.0031 | -0.0000 | 0.0613 | — |
| chance_L_lspar05 | 11.0 | 0.0004 | -0.0002 | 0.0761 | — |

B loses to the baseline outright; C is +0.0020 AMI, a fifth of a baseline sd; L-Spar's
apparent lift is again beaten by the resolution knob. Chance floors confirm all arms are far
above chance (AMI ~0.003) — the problem is never chance here, it is granularity.

com-DBLP, SNAP top-5000 communities, avgF1_ge3, 3 seeds each:

| condition | k (>=3) | avgF1_ge3 | gt2det | det2gt | gamma |
|---|---|---|---|---|---|
| baseline | 227 | 0.1149 +- 0.0053 | 0.018 | 0.212 | 1.0 |
| B_weight | 249 | 0.1283 +- 0.0193 | 0.022 | 0.234 | 1.0 |
| resmatch_forced_B_weight | 251 | 0.0891 +- 0.0096 | — | — | 1.375-1.5 |
| C_seed | 179 | 0.1298 +- 0.0083 | 0.016 | 0.243 | 1.0 |
| **resmatch_C_seed** | 176 | **0.3185** | 0.025 | 0.612 | 0.0269 |
| L_lspar05 | 10,948 | 0.1930 +- 0.0001 | 0.267 | 0.119 | 1.0 |
| **resmatch_L_lspar05** | 10,621 | **0.2971** | 0.422 | 0.172 | 576 |
| chance_baseline | 225 | 0.0235 | — | — | — |
| chance_L_lspar05 | 10,950 | 0.0902 | — | — | — |

No arm survives here. C loses to its granularity control by -0.189. B beats its RB-based
control but only by +0.039 while being just +0.0133 over the plain baseline — 0.94 pooled sd,
not significant (see HYPOTHESIS H1 on why the RB control is the wrong yardstick in this
regime). The chance floor rising 0.0235 -> 0.0902 as k goes 225 -> 10,950 quantifies
Artifact III directly: four-fifths of L-Spar's raw "improvement" over the baseline
(0.1930 - 0.1149 = +0.078) is matched by the chance floor's own rise (+0.067).

com-Amazon, SNAP top-5000 communities, avgF1_ge3 — THE ADJUDICATION CELL:

| condition | n | k (>=3) | avgF1_ge3 | gt2det | det2gt | gamma |
|---|---|---|---|---|---|---|
| baseline | 3 | 306 | 0.1827 +- 0.0110 | 0.097 | 0.269 | 1.0 |
| chance_baseline | 3 | 300 | 0.0123 +- 0.0005 | — | — | — |
| B_weight | 3 | 370 | 0.1918 +- 0.0142 | 0.114 | 0.270 | 1.0 |
| resmatch_B_weight | 1 | 376 | 0.1537 | 0.089 | 0.219 | 1.5 |
| chance_B_weight | 3 | 385 | 0.0146 +- 0.0004 | — | — | — |
| **C_seed** | 3 | 320 | **0.2210 +- 0.0085** | 0.113 | 0.329 | 1.0 |
| **resmatch_forced_C_seed** | 3 | 309 | **0.1848 +- 0.0098** | — | — | 1.0-1.125 |
| chance_C_seed | 3 | 326 | 0.0149 +- 0.0006 | — | — | — |
| **L_lspar05** | 3 | 8,658 | **0.4019 +- 0.0004** | 0.703 | 0.101 | 1.0 |
| resmatch_L_lspar05 | 1 | 8,424 | 0.3302 | 0.577 | 0.084 | 320 |
| resmatch_sweep_L_lspar05 | 1 | 9,222 | 0.3424 | 0.605 | 0.080 | 352 |
| resmatch_sweep_L_lspar05 | 1 | 10,695 | 0.3669 | 0.659 | 0.075 | 384 |
| resmatch_sweep_L_lspar05 | 1 | 11,863 | 0.3724 | 0.675 | 0.070 | 448 |
| chance_L_lspar05 | 3 | 8,653 | 0.0568 +- 0.0015 | — | — | — |

Two granularity-controlled, chance-floored positives, both on this network:
(i) **L_lspar05 (PROMOTES Exp L's V3 hypothesis to a result)**: beats a control handed 37%
more clusters, +0.0295 raw, ~+0.034 after chance correction, ~4.7 baseline seed-sigmas.
(ii) **C_seed (NEW, and the reason C17 is not dead)**: +0.0362 over a forced control at
matched k (4.0 pooled sd), +0.0382 over the plain baseline at k 320 vs 306 (3.9 pooled sd),
+0.0356 after subtracting each arm's own size-matched chance floor. Both gt2det (0.113 vs
0.097) and det2gt (0.329 vs 0.269) rise, so it is not a one-sided best-match artifact.
The +4.6% k difference cannot explain it: in this k range avgF1 DECREASES with k on this
network (0.1827 at k=306, 0.1537 at k=376).
Both positives coexist with modularity LOSSES in the same cells (L -0.0789, C -0.0067,
results.csv `dQ_vs_matched`) — the dissociation Exp L found, now reproduced on two arms.

### 5. Cost: nothing here is cheap, and the only positive arm is the most expensive
(results.csv, arm wall clock / one plain Leiden on the original graph)

| network | B_weight | Bshuf | C_seed | D_protected | L_lspar05 | U_uniform05 |
|---|---|---|---|---|---|---|
| email-Eu-core | 1.34 | 1.38 | 1.82 | 1.13 | 1.08 | 0.78 |
| wiki-Vote | 1.55 | 1.50 | 2.01 | 1.23 | 1.19 | 0.83 |
| ca-HepTh | 1.02 | 1.02 | 1.66 | 0.88 | 0.85 | 1.09 |
| ca-CondMat | 1.03 | 1.01 | 1.62 | 0.85 | 0.82 | 0.98 |
| email-Enron | 1.23 | 1.23 | 1.80 | 1.06 | 1.02 | 1.12 |
| com-DBLP | 1.01 | 1.01 | 1.70 | 0.87 | 0.85 | 1.21 |
| com-Amazon | 1.00 | 1.00 | 1.68 | 1.04 | 1.10 | 1.17 |

Arm C — the only arm with any surviving positive — costs 1.62-2.01x a single plain Leiden,
because it pays for Jaccard, for Leiden on the sparse graph, AND for a full refinement pass
on the original graph (com-DBLP: 0.493s + 0.687s + 20.19s vs 12.59s for plain Leiden;
`T_jaccard`, `T_prep`, `T_leiden_arm`, `T_leiden_orig`). Weighting is essentially free on
the large graphs (1.00-1.01x) because T_jaccard is 0.36-0.49s against a 11-13s Leiden.
The runtime-matched control's budget buys the baseline only 1-2 restarts everywhere
(`n_restarts` in {1,2} in all 42 treated cells) — the same weak-control caveat as Exp L,
and the treated arms still lose in 39 of 42.

## Post-hoc HYPOTHESES (rule 2 — not results; each needs its own pre-registration)

H1. **The RBConfiguration resolution-matched control is not a pure granularity knob in the
low-k regime.** At essentially the same k it scores WORSE than plain modularity Leiden:
com-DBLP 0.0891 at k=251 vs baseline 0.1149 at k=227; com-Amazon 0.1537 at k=376 vs 0.1827
at k=306 (recovery.csv `resmatch_forced_B_weight`, `resmatch_B_weight`). Changing gamma
changes which partition you land in, not only how many clusters. Consequence for reading
this and any prior experiment: "arm beats resmatch" is weak evidence on its own; the arm
must also beat the plain baseline (arm C on com-Amazon does, which is why V1 stands).

H2. **Best-match F1 can be gamed from the low-k side too.** com-DBLP's control at k=176
scores 0.3185 with det2gt = 0.612 — nearly triple the baseline's 0.212 at k=227
(recovery.csv `resmatch_C_seed`). A partition of a few giant clusters plus many minimum-size
ones scores extremely well on the detected->GT direction. Artifact III has a mirror image.

H3. **Weight heterogeneity per se is a cheap perturbation that sometimes helps Leiden.**
B-shuffle beats B on email-Enron (+0.01169 vs +0.01022) and wiki-Vote (-0.00454 vs -0.00749).
If true in general, "random-weight restarts" would be a control every weighted-similarity
paper needs and none run.

H4. **com-Amazon is the exception at the network level, not the method level.** It is the
only network where L-Spar's recovery survives (V5), the only one where C's recovery survives
(V1), and the only one where D beats L (V4). A product co-purchase graph with near-clique
ground-truth communities and the study's highest baseline modularity (0.9294) may simply be
where edge selection has room to act. Exp P (other algorithms) and a network-property sweep
would test this.

## Caveats and deviations from DESIGN.md

C1. **Deviation (addition): the recovery stage also runs `L_lspar05`,** which DESIGN.md
lists only for arms B and C. The com-Amazon chance floor is meaningless unless it is applied
to the anomalous partition itself, and adjudicating Exp L's V3 was the registered purpose of
the cell. It also reproduces Exp L's numbers exactly, which is the comparability check the
task required.
C2. **Deviation (addition): chance floors were computed on all three labelled networks,**
not just com-Amazon as registered, and forced granularity controls were run for the two
cells the inherited 20%-rule skipped (com-Amazon C_seed, com-DBLP B_weight;
`resmatch_forced_*` in recovery.csv). Both are extra controls; no finding rests on their
absence, and the com-Amazon one is what makes V1 defensible.
C3. **Arm D is degenerate on 6/7 networks by construction** (`frac_core` >= 0.964), so P3
was effectively testable only on com-Amazon. This is a property of the registered definition
("largest e with retention <= 0.5, then random fill"), reported rather than fixed — no
rescue tweaks, per rule 10.
C4. **com-Amazon's L-Spar arm sits at realized retention 0.5418, not 0.5** (the ceil(d^e)
step function; same issue Exp L documented). D at exactly 0.500 therefore holds 8% fewer
edges than the L-Spar arm it beats — conservative for D, and V4's margin is a lower bound.
C5. **Seed counts**: baseline 5 seeds (100-104), treated arms 3 Leiden seeds (300-302),
B-shuffle 2 shuffle seeds x 3 = 6 runs, D and U 2 fill seeds x 3 = 6 runs. Recovery arms 3
seeds; single-seed controls are marked n=1 in the tables. Everything at N_ITER=2,
ModularityVertexPartition, single-threaded leidenalg, one machine, one timing measurement.
C6. **Runtime matching buys only 1-2 restarts** (`n_restarts`), inherited verbatim from
Exp L so the numbers stay comparable. It is a weak control; it happens to be weak in the
treated arms' FAVOUR, and they still lose 39/42.
C7. **Config null is 3/7 networks, one rewiring realisation, arms B and C only** — exactly
as registered, and exactly Exp L's coverage.
C8. **Exp V runs retention target 0.5 only.** DESIGN.md registers 0.5 for arms C and D and
the full graph for B/B-shuffle; no 0.2 arm was ever part of Exp V. Exp L's 0.2 cells are not
re-derived here.
C9. **No jackknife intervals appear** because Exp V makes no correlational claim; the
controls checklist items that apply (config null, granularity matching, chance correction,
runtime matching) are all run.
C10. The com-Amazon C_seed positive rests on 3 seeds, one retention, one metric family
(best-match avgF1 against SNAP top-5000 overlapping communities) on one network, and has
NOT been checked with a chance-corrected information metric (the overlapping GT does not
admit AMI/ARI in this code path). Same limitation as Exp L's C2.

## Effect on claims

- **C17 (constructive arm) is registered as SURVIVING, narrowly.** The honest phrasing is
  not "the signal can be spent" but: *of four deployments of a verified structure-aware
  signal — weighting, sharp-null-controlled weighting, seeded refinement, and protected
  deletion — none yields an honest modularity gain on more than 1 of 7 networks, and that
  one gain is fully reproduced by a sharp null and by a configuration-model null; the single
  effect that survives every control is a recovery gain from seeded refinement on one
  network, at a cost of 1.7x a plain Leiden run and with a modularity loss in the same cell.*
- **C5/C9 are strengthened, not overturned.** The paper can now say the negative result is
  not an artifact of the deletion deployment: weighting the same signal instead of deleting
  by it moves modularity by less than 0.002 on 6/7 networks, and where it moves more, the
  null moves it further.
- **Exp L's com-Amazon anomaly (V3) is PROMOTED** from hypothesis to result under its own
  pre-registered follow-up: it survives an over-matched control at +37% k and a size-matched
  chance floor. The paper may now state it as a finding, with the dissociation between
  modularity and reference-community recovery attached.
- **New for the discussion of WHY the field's intuition fails**: the edge-weighting
  literature's reported gains are consistent with V2 — a permuted-weight arm reproduces them.
  Any future weighting paper should be asked for a shuffled-weight control; this experiment
  supplies the template and the numbers.
- **`dQ_fixed` was already retired by Exp L; V2 adds that "weighted-graph modularity improved"
  and even "honest-transfer modularity improved on one network" are equally unsafe without a
  shuffled-weight arm.**
