# Exp X — com-Youtube mini-anatomy (n=1,134,890, m=2,987,624, LCC)

**VERDICT: KILL CRITERION FIRES. The com-Youtube gain is a granularity artifact.
C9 is DEMOTED to "one genuine gain (email-Enron)". exp_H's calibrated alpha=0.95
cell must be annotated as granularity-confounded.**

The gain is statistically bulletproof and simultaneously fully explained by community
count. Both of those statements are true, and the pre-registered criterion says the
second one decides.

Registered in DESIGN.md before any Exp X code ran. Replicates the ONLY cell of
exp_H_youtube_sweep that survived Bonferroni over its 8 (sampler x alpha) cells:
**calibrated DSpar sampler, alpha = 0.95**, true retention 0.9500 (exp_H
`config_summary.csv`, row sampler=calibrated alpha=0.95: Q_seeded_mean 0.7294840,
gain_vs_bo2_mean +0.0030564, z_mean +3.06). Arms: A = 20 plain Leiden restarts
(seeds 100-119, n_iterations=2); B = 5 seeded pipelines (spar seeds 200-204 /
leiden seeds 300-304), DSpar -> Leiden on G_sparse -> refine on G. Arms were
INTERLEAVED (4 plain, 1 pipeline, x5).

Files: `run.py`, `runs.csv` (30 rows: 20 plain, 5 seeded, 3 gamma probes, 5
gamma-matched), `results.json` (every number below), `mechanism_parents.csv`,
`run.log`, `cache.npz` (best plain + best seeded membership + keep mask),
`kmatch_probe.py` / `.csv` / `.json` / `kmatch.log` (post-hoc supplementary probe).

## Replication of exp_H: EXACT

Seeds 100-109 and spar seeds 200-204 are shared with exp_H. All 15 reproduce
bit-for-bit (Q to 6 dp and k exactly): **10/10 plain, 5/5 pipelines**. Graph
dimensions match (n=1,134,890, m=2,987,624). Nothing in exp_H's measurement was
wrong; what follows is about what that measurement *means*.
[`runs.csv` cols `arm`,`seed`,`spar_seed`,`Q`,`k`; exp_H `results.csv`]

## Verdict (four parts)

**V1 — Statistical reality: HOLDS, decisively. P1 passes on substance.**
Seeded 0.729484 +- 0.000431 (n=5) vs plain 0.723718 +- 0.004680 (n=20):
**+0.005766**. Exact permutation test over ALL C(25,5)=53,130 label assignments:
**p = 1.88e-5**, which is 1/53,130 — the smallest value the test can return.
Mann-Whitney U=100 (the maximum), p=1.88e-5. Bootstrap 95% CI on the gap
**[+0.00397, +0.00801]**. The seeded mean sits at the **100th percentile** of the
plain distribution; **0 of 20** plain restarts beat the seeded mean, the seeded
best, or even the seeded *worst* (0.728788). Head-to-head win rate against all 190
runtime-matched best-of-2 draws: **1.000**. Against E[best-of-2]=0.726063 the gain
is **+0.003421**, z_mean **+5.24**, P(bo2 >= seeded mean) = 0.
Seed variance is cut ~11x (sd 0.00043 vs 0.00468).
*P1 caveat:* the registered band was +0.002..+0.005 vs the plain mean; the observed
+0.005766 overshoots it by 0.0008. Against the runtime-matched baseline (+0.003421)
it is inside the band. P1's substantive claim — "reproduces exp_H with p<0.05" — is
confirmed at p=1.9e-5.
[`results.json` `m1_statistical`; `runs.csv`]

**V2 — Granularity: NOT ruled out. It is the whole effect. P2 FAILS; KILL FIRES.**
This is the opposite of exp_N's V1, and it is the verdict.
- corr(k,Q) across the 20 plain restarts = **-0.799**, leave-one-out jackknife
  **[-0.824, -0.784]** (no single point drives it). P2 predicted |r| < 0.3.
- Seeded k = 5259 +- 180 vs plain 6428 +- 743. The seeded k is **below ALL 20**
  plain restarts (plain min 5488, `seeded_k_below_all_plain: true`).
- The **registered k-matched control is EMPTY**: 0 of 20 plain restarts fall in the
  seeded k range [4981, 5399]. The registered clause "k-matched gap <= 0" is
  therefore not evaluable; clause 2 of the kill criterion is.
- **The decisive number.** OLS of Q on k over the 20 plain restarts: slope
  -5.03e-6 per community, intercept 0.756067. Predicted Q at k=5259 is
  **0.729600**; the seeded mean is **0.729484**; residual **-0.000116**. The seeded
  partitions land *on* the plain k-Q line. There is no gain left over once community
  count is accounted for.
- Nearest-k plain restarts (the 5 closest, k 5488-5608) average 0.727161, gap
  +0.002323 — positive, but exactly what the regression predicts for a 229-349
  community offset.
- Shape check (exp_N's third route): the seeded partitions ARE mildly more balanced
  — largest community 164,041 (14.45% of n) vs 172,048 (15.16%); communities >1% of
  n 19.2 vs 18.95; top-5 share 0.4827 vs 0.4960. But the effect is ~7x weaker than
  Enron's (where the giant fell 14.8% -> 10.2%), and corr(frac_largest, Q) over the
  plain restarts is only **-0.124**. Shape does not carry the gain here; k does.

Contrast with exp_N (email-Enron): corr(k,Q) = **-0.0009**, k-matched gap
**+0.005991**, giant 14.8% -> 10.2%. Enron's gain survived all three routes.
com-Youtube's survives none.
[`results.json` `m2_granularity`; `runs.csv` cols `k`,`Q`,`largest`,`frac_largest`,
`n_comm_gt_1pct`,`top5_share`]

**V3 — Mechanism: the Enron signature is present but causally negligible.
P3 passes the registered ratio test and fails the sanity test.**
Best plain = seed 107 (Q 0.728759, k 5562); best seeded = leiden 300 / spar 200
(Q 0.729969, k 5171); dQ best-vs-best only **+0.001210**.
Over the 35 plain communities the seeded partition splits (116,658 nodes, 10.3% of
the graph), DSpar removed **0.458%** of cross-piece edges against **0.060%** of
intra-piece edges — a **7.60x** bias, clearing P3's >1.5x threshold. The targeting
is real and matches Enron: cross-piece endpoint degree product is **3.75x** the
intra-piece value pooled (median 2.58 per parent), and exceeds 1 in **32/35**
parents. The merge direction agrees (47 parents, 3.45x bias, dprod 3.60x, >1 in
43/47).
**But the absolute magnitudes make it causally inert.** At alpha=0.95 the global
removal rate is 5.00%, and these boundary edges were removed at **0.09x the global
rate** — only **28 of 6,114** cross-piece edges were deleted across all 35 split
parents (84 of 139,345 intra-piece). **28 of 35 split parents lost NO cross-piece
edge at all**; in the merge direction, 40 of 47. Twenty-eight edge deletions on a
2,987,624-edge graph cannot manufacture +0.0058 of modularity. exp_N flagged this
same failure mode on a minority of parents (13/29); on com-Youtube it is the
overwhelming majority. The splits are restart variation, not sparsification.
[`results.json` `m3_mechanism`; `mechanism_parents.csv`]

**V4 — Landscape: no separate basin (agrees with exp_N V4).**
AMI from the 20 plain partitions to the best seeded partition: mean 0.691
(max 0.740, min 0.593). AMI among plain partitions themselves: mean 0.704
(10 pairs). The seeded partition is **not** further from the plain partitions than
they are from each other. Seeded-seeded AMI is the tightest group (0.747), i.e.
variance reduction, not a distinct optimum. Best seeded exceeds best plain by only
+0.001210.
[`results.json` `m5_landscape`]

## Compute (in expectation only — exp_N caveat C2 discipline)

One pipeline 156.53 s vs one plain restart 88.17 s = **1.775x** (breakdown:
sparsify 0.93 s, Leiden on G_sparse 88.03 s, refine on G 67.56 s). Twenty restarts
cost **11.27x** one pipeline. Exact best-of-k over the 20 plain runs: E[bo2]
0.726063, E[bo3] 0.726851, E[bo4] 0.727300, E[bo5] 0.727596 — all below the seeded
mean, P(>= seeded mean) = 0 for each. Bootstrap E[best-of-20] = 0.728496, still
below the seeded mean 0.729484.
**Correct phrasing:** *in expectation*, plain restarts do not reach the seeded mean
even at 11x the compute. This is an expectation statement about a quantity we have
just shown is confounded with community count; it is NOT evidence that the pipeline
finds better community structure.
**Absolute wall-clock is LOAD-CONTAMINATED** — another job ran on Fuji throughout.
Arms were interleaved specifically so the 1.775x ratio survives; the seconds do not.
[`results.json` `m4_compute`; `runs.csv` col `T`]

## Where the gain is created

The partition found on G_sparse, scored on the ORIGINAL graph with no refinement,
already averages **0.727856** — above the plain mean by +0.004138, i.e. **71.8%** of
the total advantage is inherited from the sparse graph and refinement supplies only
+0.001628 (28.2%). (exp_N/Enron: 56% inherited, 44% refinement.)
[`runs.csv` col `Q_raw`; `results.json` `m1_statistical.Q_sparse_transfer_mean`]

## Two controls that could not be built (reported, not rescued)

**(a) Resolution-matched plain arm — DEGENERATE.** Registered as a way to get plain
partitions at the seeded granularity. On com-Youtube, k(gamma) is **decreasing in
gamma** over [0.7, 1.0] — gamma=0.7 -> k=7634, 0.8 -> 6631, 0.9 -> 6458 — the
opposite of the textbook direction, because low gamma leaves more unmerged periphery
even as the giant grows (giant 22.6% at gamma=0.7 vs 15.2% at gamma=1). The log-log
fit therefore clipped at gamma*=0.999 and the arm collapsed to plain Leiden (mean
k 6487, Q 0.723883). No k-match achieved.
[`runs.csv` arms `gamma_probe`,`gamma_matched`; `results.json` `m2_granularity.gamma_probes`]

**(b) Post-hoc gamma>1 probe — FAILED to match.** HYPOTHESIS-GRADE, NOT REGISTERED
(EXPLORATION.md rule 2); run because control (a) degenerated and the registered
k-matched subset was empty. Pushing gamma above 1 (the direction that lowers k here)
reached k=5619 at gamma=1.5 and k=4999 at gamma=1.83 on single probes, but the 5
production seeds at gamma*=1.8315 scattered over **k 6149-6604 (mean 6394)** against
a target of 5259, with a structurally different partition (giant **8.0%** of n vs
14.4% seeded) and Q 0.722984. **Community count on com-Youtube is dominated by seed
noise in periphery absorption, not by the resolution parameter.** A k-matched plain
control is not constructible on this graph by resolution tuning in either direction.
This probe changes nothing about the registered verdict.
[`kmatch_probe.json`, `kmatch_probe.csv`]

## HYPOTHESIS (post-hoc, needs its own pre-registered follow-up)

The k-Q confound decomposes entirely into **sub-20-node fragments**, not macro
structure. Communities of >=20 nodes: plain **322.2** vs seeded **317.2** — the same
macro partition, to 1.6%. Sub-20-node fragments: plain **6105** vs seeded **4942**.
Correlations over the 20 plain restarts: corr(fragment count, Q) = **-0.799**;
corr(#communities >= 20, Q) = **+0.162**. Regressing Q on fragment count leaves the
seeded mean with residual **-0.000053** (fully explained); regressing on macro
community count leaves **+0.005959** (unexplained).
So what the pipeline reliably buys on com-Youtube is *fewer orphaned periphery
fragments*, at unchanged macro granularity and a slightly smaller giant. Whether
that is a real optimization win or a bookkeeping artifact of modularity is an open
question — and it is exactly the kind of post-hoc observation this project has been
burned by three times (r=0.92; deg-CV 0.69->0.12; tri_intra +0.74->+0.06). It is
recorded as a HYPOTHESIS and does NOT modify the verdict.
[`runs.csv` cols `k`,`n_comm_ge20`,`Q`]

## Kill criterion, evaluated exactly as registered

| clause | registered text | outcome |
|---|---|---|
| 1 | k-matched gap <= 0 | **NOT EVALUABLE** — 0/20 plain restarts in the seeded k range |
| 2 | corr(k,Q) < -0.5 with seeded k systematically lower | **FIRES** — r=-0.799; seeded k below all 20 plain restarts |
| 3 | P1 fails to reproduce (p > 0.10) | does not fire — p = 1.9e-5 |

Clause 2 fires. **The com-Youtube gain is recorded as a granularity artifact.
C9 is DEMOTED to "one genuine gain (email-Enron)". exp_H's calibrated alpha=0.95
cell is annotated: the +0.003 vs E[best-of-2] is real as a measurement and
confounded with community count as an interpretation.** No rescue tweaks.

## Caveats

- **C1.** Single alpha (0.95), single sampler (calibrated), n_iterations=2, 5 seeded
  runs. Exp X dissects exp_H's surviving cell; it does not re-sweep the grid.
- **C2.** Every compute statement is *in expectation*. Best-of-20 plain falls below
  the seeded mean on average; individual restarts get close (best plain 0.728759 vs
  seeded worst 0.728788 — a 0.00003 gap).
- **C3.** Absolute wall-clock is load-contaminated (concurrent job on Fuji).
  The 1.775x pipeline/restart ratio is protected by interleaving; seconds are not.
- **C4.** The mechanism analysis uses the single sparsification seed that produced
  the best seeded partition (spar seed 200), as in exp_N caveat C6.
- **C5.** corr(k,Q)=-0.799 is a correlation across restarts. It does not by itself
  establish that lowering k *causes* higher Q; the attempt to establish that
  causally (controls (a) and (b) above) failed because k is not steerable on this
  graph. The registered criterion is a correlational test and was applied as written.
- **C6.** The 20-pair AMI budget capped the landscape probe at 40 comparisons
  (10 plain-plain pairs, not all 190).
