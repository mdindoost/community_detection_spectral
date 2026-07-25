# Exp O — core-preservation / periphery-fragmentation (small five, real + config-null arms)

Design: DESIGN.md (pre-registered, EXPLORATION.md commit 953aa10). Networks email-Eu-core,
wiki-Vote, ca-HepTh, ca-CondMat, email-Enron (LCC). Sparsifiers DSpar (calibrated sampler,
alpha 0.8/0.5, 2 sparsification seeds) and L-Spar (realized-retention targets 0.5/0.2),
3 Leiden seeds per arm, n_iterations=2. P0 = Leiden(g, seed 100).

Files: run.py, analyze.py, VERDICT.txt (full machine-generated tables), results.csv (360 rows
x 114 cols), cohesion.csv (2,232 per-community rows), node_attrs.csv (143,498 node rows:
degree, k-core index, embeddedness, P0 community, deciles), run_email-Eu-core.log, run_rest.log.

## VERDICT

**The hypothesis fails on four of its five pre-registered predictions. The registered kill
criterion does NOT fire, but it does not save the hypothesis: the criterion asked whether an
agreement/embeddedness gradient exists, and a weak, structure-specific one does; the hypothesis
asked whether sparsification preserves community CORES and shreds the loose PERIPHERY, and it
does neither.** Three independent results kill it:

- **V1 — the "core" proxy is upside down.** Embeddedness (fraction of neighbours inside one's own
  P0 community) is *negatively* correlated with the k-core index (r = -0.22 to -0.57) and with
  degree (r = -0.25 to -0.55) on all five graphs. The top-3 embeddedness deciles have median
  k-core 1-4 and median degree 1-4; the bottom-3 deciles have median k-core 4-27 and median
  degree 7-42. "Top-50% embeddedness" selects low-degree leaves, not the structural core.
  [VERDICT.txt TABLE 0; node_attrs.csv `embeddedness`,`coreness`,`degree`,`dec_emb`]
- **V2 — where the partition actually shatters, it shatters cores as hard as peripheries.** On
  ca-HepTh and ca-CondMat at retention 0.5 the median core cohesion is 0.25-0.35 and only
  0.0-6.8% of P0 communities (>=20 nodes) keep >=80% of their core in one P' cluster; the
  core-minus-periphery cohesion contrast is +0.02 to +0.05 (once negative). P3 passes only on the
  two graphs whose sparse partition does not fragment at all.
  [results.csv `coh_median`,`coh_frac_ge_0p8`,`periph_coh_median`; cohesion.csv]
- **V3 — nothing here is specific to sparsification.** At matched cluster count the
  resolution-matched control on the ORIGINAL graph preserves cores BETTER than the sparsified
  partition in 16/20 cells (mean d = -0.053), reproduces the same (inverted) fragment
  composition, and its clusters are purer in 6/9 heavy-fragmentation cells. And the
  seed-jitter floor -- two plain Leiden seeds on the same graph -- already produces
  embeddedness gradients of +0.050 to +0.124 against the sparsifiers' +0.063 to +0.171.
  [results.csv condition=`resmatch` vs `dspar`/`lspar`; condition=`jitter`]

## Pre-registered predictions, evaluated exactly as written

**P1 — FAIL, 0/5 networks.** Top-3-decile Hungarian agreement at DSpar retention 0.5:
email-Eu-core 0.834, wiki-Vote 0.751, ca-HepTh 0.297, ca-CondMat 0.306, email-Enron 0.566
(threshold >0.85). Under the plurality matching variant (DESIGN allows either) 0.893 / 0.768 /
0.499 / 0.447 / 0.798, i.e. 1/5. Monotonicity fails everywhere: 4.7-5.7 of 9 decile steps
increase, Spearman rho over the ten deciles +0.41 to +0.62. The decile profiles show why --
agreement rises across deciles 0-3 and is flat thereafter, e.g. email-Enron DSpar 0.5:
0.321 0.436 0.516 0.605 0.570 0.567 0.565 0.563 0.560 0.575.
[results.csv `ah_top3`,`ap_top3`,`spearman_hung`,`n_incr_steps_hung`,`ah_d0..ah_d9`]

**P2 — FAIL, 0/11 testable cells, and the direction is REVERSED.** Required: sub-10-node P'
fragments >=2x over-represented in the bottom-2 embeddedness deciles. Observed enrichment
0.08-0.46 in the eight moderate-fragmentation cells (ca-HepTh dspar 0.5: 0.46; ca-CondMat lspar
0.5: 0.08; email-Enron dspar 0.5: 0.17) with top-2-decile enrichment 1.1-1.6; the three extreme
cells where >50% of nodes are fragments are trivially uniform (0.96-1.15). Fragments are drawn
preferentially from HIGH-embeddedness nodes, which by V1 are the low-degree leaves.
Untestable on email-Eu-core (0 fragments) and wiki-Vote (5-9 fragment nodes).
[results.csv `frag_enrich_bot2`,`frag_enrich_top2`,`n_frag_nodes`,`fs_d0..fs_d9`]

**P3 — 5/10 cells; 2/5 networks with both sparsifiers.** Median core cohesion at retention 0.5:
email-Eu-core 0.971/0.928, email-Enron 0.892/0.819 (PASS, dspar/lspar), wiki-Vote 0.753/0.835,
ca-HepTh 0.335/0.338, ca-CondMat 0.346/0.247. The passes are the cells where the sparse partition
does not shatter: on email-Enron DSpar *coarsens* (k' = 108 vs k0 = 195). At L-Spar 0.2 cohesion
collapses to 0.088 / 0.047 / 0.226 (k' = 1,875 / 5,157 / 5,860 against k0 = 49 / 61 / 195).
[results.csv `coh_median`,`kp`,`k0`; cohesion.csv `core_cohesion`,`periph_cohesion`]

**P4 — PASS, 16/20 cells.** Mean agreement gap real vs config-null: email-Eu-core +0.171/+0.014,
wiki-Vote +0.135/+0.037, ca-HepTh +0.079/+0.034, ca-CondMat +0.063/+0.010, email-Enron
+0.141/+0.064. The four failures are the two flattest real cells (ca-HepTh lspar 0.2/0.5,
ca-CondMat lspar 0.2) plus email-Enron dspar 0.8 (ratio 0.57). See Caveat C3: the null arm is
also at an agreement floor (0.066-0.182 overall vs 0.279-0.786 real, Q0_null 0.12-0.41), so a
compressed null gradient is partly mechanical.
[results.csv arm=`null` vs `real`, `ah_gap`,`Q0`,`agree_all_hung`]

**P5 — FAIL, 3/11 cells.** The resolution-matched control reproduces the fragment composition
almost exactly: ca-HepTh dspar 0.5, enrich_bot2 0.46 (sparsifier) vs 0.21 (resmatch);
ca-CondMat lspar 0.5, 0.08 vs 0.07; ca-HepTh lspar 0.2, 0.98 vs 1.07; email-Enron lspar 0.2,
1.15 vs **1.48** -- there the resolution control is *more* low-embeddedness-selective than the
sparsifier. Core cohesion at matched k: sparsifier minus resmatch = -0.053 on average,
sparsifier better in only 4/20 cells.
[VERDICT.txt P5 block; results.csv condition=`resmatch`]

## Kill criterion

Registered: dead if (A) the agreement gradient is flat (top-3 minus bottom-3 gap <0.1) OR (B) the
null reproduces the gradient within noise on >=half the networks. DESIGN did not say how to
aggregate (A) across networks, so both readings are reported: per-network mean gap over the four
sparsifier cells = +0.166 / +0.143 / +0.070 / +0.050 / +0.146 -> flat on **2/5** networks
(ca-HepTh, ca-CondMat); per cell, 7/20 are flat. (B): with "within noise" = gap_null >=
gap_real - 2 SE over the seed replicates, the null reproduces on **0/5** networks.
**The kill criterion does not fire.** It was mis-targeted: it tested the existence of a gradient,
not core preservation or periphery selectivity, and a weak structure-specific gradient does exist
(V3 shows it is not sparsification-specific). No rescue tweak was applied and no post-hoc
re-slicing was used to reach the verdicts above; every threshold is the registered one.
[VERDICT.txt KILL CRITERION block]

## Numbered findings

1. **Embeddedness is an anti-core statistic on real networks.** r(emb, k-core) = -0.549
   (email-Eu-core), -0.572 (wiki-Vote), -0.224 (ca-HepTh), -0.250 (ca-CondMat), -0.460
   (email-Enron); r(emb, degree) = -0.551 / -0.437 / -0.379 / -0.332 / -0.253; r(k-core, degree)
   = +0.66 to +0.79. A node with 2 neighbours both inside its community scores 1.0; a hub with
   40% of its edges crossing a boundary scores 0.6. [VERDICT.txt TABLE 0; node_attrs.csv]

2. **Embeddedness is massively tied at 1.0**: 20.4% (email-Eu-core), 45.3% (wiki-Vote), 63.6%
   (ca-HepTh), 54.7% (ca-CondMat), 66.4% (email-Enron) of nodes. Decile means reach exactly 1.000
   by decile 5 on three of the five graphs, so deciles 5-9 are one undifferentiated block split
   by random tie-breaking. All of the measured gradient lives in deciles 0-3.
   [VERDICT.txt TABLE 0 `frac(emb=1)`; results.csv `emb_mean_bot3`,`emb_mean_top3`,`frac_emb_eq1`]

3. **The fragmentation premise itself is cell-specific.** DSpar at retention 0.5-0.8 produces
   almost no sub-10-node clusters (0.0-1.0% of nodes on 4/5 networks) and on email-Enron it
   *coarsens* (k' = 108 vs k0 = 195, consistent with Exp N's k 182->156). The nc explosion is an
   L-Spar-at-0.2 and coauthorship phenomenon: fragment mass 77.9% (ca-HepTh), 86.3%
   (ca-CondMat), 54.2% (email-Enron). [results.csv `frac_frag_nodes`,`kp`]

4. **Fragments are pure subdivisions, but that is a granularity fact, not a sparsification fact.**
   At L-Spar 0.2 plurality agreement is 0.940 / 0.944 / 0.932 (ca-HepTh / ca-CondMat /
   email-Enron) while Hungarian agreement is 0.078 / 0.034 / 0.139: the shards do not mix
   communities, they subdivide them. The resolution-matched control at the same k gives
   0.946 / 0.944 / 0.931. At intermediate granularity the sparsifier is markedly LESS pure than
   the resolution control (ca-HepTh dspar 0.5: 0.460 vs 0.801; ca-CondMat lspar 0.5: 0.501 vs
   0.820), i.e. sparsification does move nodes across community boundaries.
   [results.csv `agree_all_plur` vs `agree_all_hung`, condition `resmatch`]

5. **Seed-jitter floor (added control).** Two plain Leiden seeds on the same untouched graph give
   embeddedness gradients of +0.069 / +0.114 / +0.050 / +0.066 / +0.124. The sparsifiers' mean
   gradients exceed that floor by only +0.102 / +0.021 / +0.029 / -0.003 / +0.017. On 4/5
   networks the "core-preservation gradient" is within 0.03 of what a different random seed
   produces. [results.csv condition=`jitter`]

6. **Chance baseline is clean.** Permuting P' labels over nodes gives overall agreement
   0.029-0.210 and gradients -0.040 to +0.024, core cohesion 0.021-0.273 -- i.e. the measured
   agreements and cohesions are far above chance even where they fail the predictions.
   [results.csv condition=`perm`]

7. **Matching method is not a confound.** All 360 rows used exact Hungarian assignment
   (k0 x k' <= 4e6 everywhere); the greedy-descending-overlap fallback was computed alongside as
   a cross-check and differs on 0% of clusters in the real-arm sparsifier cells.
   [results.csv `match_method`,`greedy_vs_exact_map_diff`]

8. **Retention calibration verified.** DSpar realized retention 0.4992-0.5013 and 0.7991-0.8017
   against nominal 0.5/0.8. L-Spar hits 0.497-0.505 for target 0.5; for target 0.2 the local
   top-k rule floors at 0.276 (ca-HepTh) and lands at 0.159-0.201 elsewhere -- same behaviour as
   Exp L. [results.csv `realized_ret`; run_rest.log "lspar target=... (floor)"]

## HYPOTHESES (post-hoc, not results -- each needs its own pre-registration)

- **H-O1 (leaf-shedding).** The real phenomenon may be degree/k-core-selective rather than
  embeddedness-selective: fragments concentrate in low-degree, low-coreness nodes. A direct test
  would stratify fragment composition by k-core index and degree with a granularity-matched
  control. The coreness- and degree-decile agreement gradients recorded here
  (`core_ah_gap`, `deg_ah_gap`, `ahc_d*`, `ahd_d*`) are the raw material: they are strongly
  NEGATIVE on email-Enron under L-Spar (-0.128 at 0.5, -0.318 at 0.2), i.e. the graph's k-core
  is where the assignment moves.
- **H-O2 (boundary-only gradient).** Agreement appears depressed only for nodes with a nonzero
  fraction of out-of-community neighbours; among fully embedded nodes (20-66% of the graph) it is
  flat. A pre-registered version would bin by *number* of boundary edges rather than by decile.
- **H-O3 (mild-fragmentation core preservation on email-Enron).** The single cell where a weak
  version of the hypothesis beats its granularity control by a visible margin: L-Spar 0.5
  (k' = 367 vs k0 = 195) gives core cohesion 0.819 vs the resolution-matched 0.737. 4/20 cells
  point this way; 16/20 point the other way.

## Caveats and deviations from DESIGN.md

- **C1 (measurement, not fixable post hoc).** The design's operationalization of "core" as
  top-50% embeddedness does not measure what the hypothesis means by "core" (finding 1). This
  invalidates the *interpretation* of P2/P3 as a test of hub/core preservation, but it does not
  rescue the hypothesis: V2 shows that the communities' most-embedded halves are not held
  together either, and V3 shows nothing is sparsification-specific.
- **C2.** Deciles are equal-sized RANK deciles with random tie-breaking (fixed seeds 11/12/13),
  which is what makes the fragment-composition baseline exactly 0.1 per decile. With 20-66% of
  nodes tied at embeddedness 1.0 the top deciles are an arbitrary split of one tied block
  (finding 2). Any decile-level statement about the top half should be read as a statement about
  "fully embedded nodes", not about a gradient.
- **C3.** P4's pass is weaker than it looks: the config null has Q0 = 0.12-0.41 and overall
  agreement 0.066-0.182, so its gradient is compressed by a floor. The sharper specificity
  controls are the resolution-matched arm (P5, fails) and the seed-jitter floor (finding 5).
- **C4 (deviation).** DESIGN says "Hungarian/plurality mapping". Both were computed; Hungarian is
  primary for P1/P4/kill and plurality is reported alongside (P1 fails under both).
- **C5 (addition).** Two arms not in DESIGN were added, both as controls, neither replacing a
  registered one: the seed-jitter arm (P' = plain Leiden seeds 101/102) and periphery cohesion
  (bottom-50%-embeddedness members) next to core cohesion. Fragment *rate* per decile
  (`fr_d0..fr_d9`) is recorded next to fragment *share* (`fs_d0..fs_d9`).
- **C6 (deviation).** P0 is a single Leiden seed (100) per graph; DESIGN's "3 Leiden seeds per
  arm" was spent on the sparse side. P0-seed sensitivity is bounded by the jitter arm.
- **C7 (deviation).** DESIGN did not specify how to aggregate kill-criterion clause (A) across
  networks; both the per-network-mean and per-cell readings are reported, and clause (B)'s
  ">=half the networks" rule was applied to both.
- **C8 (scope).** com-DBLP and com-Amazon were not run. DESIGN gates them on the small five
  passing sanity; the small five ran clean but four of five predictions failed, and
  EXPLORATION.md rule 10 ("scale up only if the cheap version says the idea is alive") says stop.
  Cost if wanted for generality: ~1-2 h, dominated by the L-Spar-0.2 resolution matching at
  k ~ 70,000.
- **C9.** Single P0 objective (ModularityVertexPartition, n_iterations=2, Leiden only). The
  resolution-matched arm necessarily uses RBConfigurationVertexPartition; at gamma = 1 it is
  equivalent to the modularity objective, and where the bisection returned gamma = 1.0 with seed
  100 the row is P0 itself (agreement 1.0 by construction; email-Eu-core dspar rows). Those
  degenerate rows are kept in results.csv and are visible as `gamma`=1.0 with
  `agree_all_hung`=1.0; they do not enter any P1-P5 count except through the resmatch seeds
  101/102 alongside them.

## Effect on the claims table

No new claim. C16 is NOT created. The proposed upgrade of C5/C6 into "sparsification preserves
community cores and reorganizes hub-mediated boundaries and loose periphery" is **not supported**:
the core-preservation half is false wherever the partition actually fragments, and the periphery
half is false in direction. Exp N's mechanism (V5: 2.58x biased removal of hub-mediated boundary
edges) is untouched by this -- it is a statement about which EDGES are removed, and Exp O shows it
does not translate into a node-level core/periphery split of the resulting partition. The honest
descriptive sentence available for the paper is:

  Under sparsification the sparse-graph partition subdivides the original communities rather than
  mixing them -- at heavy retention loss 93-94% of nodes still sit in a cluster whose plurality
  identity is their original community -- but the subdivision is not selective for loosely
  attached nodes: it cuts the most-embedded half of a community as readily as the least-embedded
  half (median core cohesion 0.25-0.35 on ca-HepTh/ca-CondMat at retention 0.5), and a partition
  of the ORIGINAL graph at matched resolution reproduces the same profile, preserving cores
  slightly better in 16 of 20 configurations. The fragmentation reported in C5/C6 is therefore a
  granularity effect, not benign shedding of the periphery.
