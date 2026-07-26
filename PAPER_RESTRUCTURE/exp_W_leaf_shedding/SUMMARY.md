# Exp W — leaf-shedding (H-O1 follow-up): fragment composition on the degree / k-core axes

Design: DESIGN.md (pre-registered before any Exp W code ran; promotes exp_O's post-hoc
H-O1). Networks: the exp_O five (email-Eu-core, wiki-Vote, ca-HepTh, ca-CondMat,
email-Enron, LCC). Cells: dspar 0.5, lspar 0.5, lspar 0.2 + their resolution-matched
controls + the config-null arm, both arms, exp_O's seeds throughout.

Files: run.py (imports exp_O_core_preservation/run.py as a module — every sampler,
Leiden call, decile rule and seed is exp_O's verbatim), analyze.py, VERDICT.txt (full
machine-generated tables), frag_composition.csv (230 rows x 84 cols),
run_email-Eu-core.log, run_rest.log.

## VERDICT

**H-O1 is dead. The registered kill criterion fires on clause (a): P1 fails in 0 of 9
testable cells.** exp_O found fragment nodes over-represented among HIGH-embeddedness
nodes and hypothesised that the true axis was degree / k-core — that sparsification's
sub-10-node clusters are the graph's leaves. On the degree and k-core axes the
predicted concentration is simply absent, and on email-Enron it is reversed: the
fragments are the graph's mid-degree, mid-coreness nodes, and their degrees are *above*
the graph median. There is no leaf-shedding phenomenon to attribute to sparsification,
so clause (b)'s "granularity fact, not sparsification fact" fallback never engages
either — there is no fact to downgrade.

- **V1 — no bottom-2 enrichment on either registered axis.** Largest bottom-2 degree
  enrichment over all 9 testable cells is 1.53 and largest k-core enrichment is 1.66
  (both ca-CondMat dspar 0.5), against the registered threshold of 2.0; six of nine
  cells are *below* 1.0 on both axes. [frag_composition.csv `deg_enrich_bot2`,
  `core_enrich_bot2`, rows arm=real condition in {dspar,lspar}; VERDICT.txt P1 table]
- **V2 — the failure is not a decile-tie ceiling.** Rank deciles with random
  tie-breaking cap the attainable bottom-2 enrichment at 1/F where F is the mass of the
  boundary tied block. F_deg = 0.206-0.370 (ceiling 2.71-4.86) and F_core = 0.229-0.430
  (ceiling 2.32-4.52) on the five real graphs, so >=2x was reachable everywhere.
  [VERDICT.txt TABLE 0; node_attrs.csv `degree`,`coreness`,`dec_deg`,`dec_core`]
- **V3 — fragmentation is a MID-degree phenomenon.** The fragment *rate* per degree
  decile peaks in deciles 2-6 in 8 of 9 cells and collapses at both ends, e.g.
  email-Enron lspar 0.5: 0.004 0.005 0.005 0.025 0.073 0.116 0.107 0.047 0.009 0.000,
  and ca-HepTh lspar 0.5: 0.082 0.091 0.131 0.146 0.183 0.170 0.103 0.097 0.034 0.005.
  The resolution-matched control on the ORIGINAL graph produces the same hump.
  [frag_composition.csv `frd_d0..frd_d9`, `frc_d0..frc_d9`; VERDICT.txt]

## Pre-registered predictions, evaluated exactly as written

**P1 — FAIL, 0/9 testable cells, 0/3 networks that have any.** Registered: fragment
nodes >=2x over-represented in the bottom-2 DEGREE deciles *and* bottom-2 K-CORE
deciles, in >= 2/3 of testable fragmentation cells per network. "Testable" = real-arm
sparsifier cell with >= 20 fragment nodes (email-Eu-core has 0 fragments in all three
cells; wiki-Vote has 5-9, which exp_O also called untestable — reported below as a
sensitivity). Mean over seeds, bottom-2 enrichment (degree / k-core):
ca-HepTh dspar 0.5 = 1.31 / 1.55, lspar 0.5 = 0.83 / 0.88, lspar 0.2 = 0.91 / 0.92;
ca-CondMat dspar 0.5 = 1.53 / 1.66, lspar 0.5 = 0.80 / 0.98, lspar 0.2 = 0.95 / 0.96;
email-Enron dspar 0.5 = 0.02 / 0.00, lspar 0.5 = 0.11 / 0.10, lspar 0.2 = 0.08 / 0.09.
0/9 pass on degree alone, 0/9 on k-core alone, 0/9 on both. Seed spread is negligible
(deg_enrich_bot2 range within a cell <= 0.14).
Sensitivity, the three wiki-Vote cells (5-9 fragment nodes, i.e. below any usable
precision): enrichment 1.16 / 0.56 / 0.42 on degree and 3.04 / 1.53 / 1.39 on k-core —
the only place the predicted direction appears, on 5-9 nodes out of 7,066.
[frag_composition.csv `n_frag_nodes`,`deg_enrich_bot2`,`core_enrich_bot2`;
VERDICT.txt P1 block]

**P2 — CONFIRMED (the prediction was that the effect is NOT sparsification-specific).**
Registered: sparsification's enrichment exceeds the resolution-matched control's by
>=1.5x in fewer than half the cells. Observed: 1/9 cells on both axes jointly, 1/9 on
degree, 2/9 on k-core; the resolution control matches or exceeds the sparsifier
(ratio <= 1.0) in 5/9 cells on each axis. Mean ratio = 1.012 (degree), 1.072 (k-core);
medians 0.989 and 0.944. Leave-one-network-out on the mean degree ratio: 0.761 (drop
ca-HepTh) to 1.327 (drop email-Enron). The single cell that exceeds 1.5x on both axes
is ca-HepTh dspar 0.5 (2.46 / 2.54, against a control with 103 fragment nodes).
[frag_composition.csv condition=`resmatch` vs `dspar`/`lspar`; VERDICT.txt P2 block]

**P3 — FAIL, 0/8 cells, direction reversed.** Registered: the config-null arm shows
weaker degree-selectivity than the real graph. In all 8 cells testable in both arms
(>= 20 fragments each) the null's bottom-2 degree enrichment is *higher* than the real
graph's: ca-HepTh dspar 0.5 1.99 vs 1.31, lspar 0.5 4.44 vs 0.83, lspar 0.2 1.08 vs
0.91; ca-CondMat dspar 0.5 1.87 vs 1.53, lspar 0.2 1.05 vs 0.95; email-Enron dspar 0.5
3.40 vs 0.02, lspar 0.5 3.36 vs 0.11, lspar 0.2 0.86 vs 0.08. Same on the k-core axis
(0/8). Interpretation: in a degree-preserving rewired graph there is no community
structure to hold low-degree nodes in place, so they shard off — the real graph's
low-degree nodes do NOT, which is the opposite of leaf-shedding.
[frag_composition.csv arm=`null` vs `real`; VERDICT.txt P3 block]

## Kill criterion

Registered: (a) if P1 fails (no >=2x bottom-2 enrichment in a majority of testable
cells), H-O1 is dead outright; (b) if P1 holds but the resmatch control matches or
exceeds the enrichment in >= half the cells, the finding is recorded as a GRANULARITY
fact, not a sparsification fact.

**Clause (a) FIRES: 0/9 testable cells, which is not a majority under any reading.**
Clause (b) is moot but was computed anyway and would also have fired (resmatch matches
or exceeds in 5/9 cells on both axes). No rescue tweak was applied; no post-hoc
re-slicing was used; every threshold is the registered one. H-O1 does not enter the
paper in any form.

## Numbered findings

1. **The re-run is bit-for-bit identical to exp_O.** exp_O did not save partitions and
   stored fragment composition only per embeddedness decile, so the degree/k-core
   composition was not recoverable from its CSVs; run.py re-derives the same partitions
   by importing exp_O's run.py and reusing its seeds, and reads the resolution-matched
   gammas straight out of exp_O's results.csv instead of re-bisecting. Checks on all
   230 rows: k' identical 230/230, fragment count identical 230/230, |d agreement| max
   0.0, |d embeddedness-decile fragment share| max 0.0; per-node degree, coreness,
   embeddedness and all three decile vectors identical to node_attrs.csv on all 10
   (network, arm) pairs. [frag_composition.csv `chk_kp_match`,`chk_nfrag_match`,
   `chk_agree_diff`,`chk_fs_maxdiff`; run_*.log "REPRO node_attrs match: True"]

2. **Fragments on email-Enron are above-median-degree nodes.** Median degree of
   fragment nodes vs graph: 4 vs 3 (dspar 0.5), 4 vs 3 (lspar 0.5), 5 vs 3 (lspar 0.2);
   median coreness 3.6 / 4 / 4 vs 3. P(degree <= 2 | fragment) = 0.077 / 0.067 / 0.127
   against 0.384 in the graph (lift 0.20 / 0.17 / 0.33); P(k-core <= 1 | fragment) =
   0.005 / 0.032 / 0.026 against 0.283 (lift 0.02 / 0.11 / 0.09). This is the sharpest
   refutation in the experiment and it is tie-free — it uses no decile binning.
   [frag_composition.csv `frag_deg_median`,`all_deg_median`,`frag_frac_deg_le2`,
   `lift_deg_le2`,`frag_frac_core_le1`,`lift_core_le1`; VERDICT.txt TABLE 1]

3. **On the coauthorship graphs fragments are close to a uniform draw.** ca-HepTh and
   ca-CondMat lift(degree <= 2) = 1.11 / 1.01 / 0.99 and 1.50 / 0.81 / 0.96 across
   dspar 0.5 / lspar 0.5 / lspar 0.2; lift(k-core <= 1) = 1.56 / 0.82 / 0.90 and
   1.98 / 0.69 / 0.86. The mild positive lift lives only in the dspar 0.5 cells, which
   are also the cells with the fewest fragments (399 and 114 nodes).
   [frag_composition.csv `lift_deg_le2`,`lift_core_le1`; VERDICT.txt TABLE 1]

4. **Fragment rate peaks in the middle of the degree distribution and vanishes at both
   ends.** argmax decile of the per-decile fragment rate: 4, 5, 4 (ca-HepTh dspar 0.5 /
   lspar 0.2 / lspar 0.5), 0, 4, 2 (ca-CondMat), 5, 6, 5 (email-Enron). Top-decile
   fragment rate is at or near zero in the moderate cells (0.009, 0.005, 0.001, 0.000,
   0.000): hubs are never shed. The resolution-matched control reproduces the shape,
   e.g. email-Enron resmatch lspar 0.5: 0.001 0.001 0.001 0.005 0.016 0.017 0.018
   0.016 0.001 0.000 next to the sparsifier's 0.004 0.005 0.005 0.025 0.073 0.116
   0.107 0.047 0.009 0.000. Top-2-decile enrichment is 0.04-0.35 in the moderate cells,
   confirming the same thing from the other end.
   [frag_composition.csv `frd_d0..frd_d9`, `deg_enrich_top2`, `core_enrich_top2`]

5. **Seed-jitter floor: zero fragments.** Plain Leiden with seeds 101/102 on the
   untouched graph produces 0 sub-10-node-cluster nodes on all five real graphs. Unlike
   exp_O's agreement gradients — which the jitter arm largely reproduced — the
   fragmentation itself is not a seed artifact; it is produced by sparsification and by
   resolution increase alike. [frag_composition.csv condition=`jitter`]

6. **Measurement 2 (agreement gradient by k-core / degree decile), from exp_O's stored
   ahc_d*/ahd_d*.** Registered as a measurement, not attached to a prediction. Top-3
   minus bottom-3 gaps in the real arm are small and positive on four graphs
   (email-Eu-core +0.03..+0.12, wiki-Vote +0.00..+0.06, ca-HepTh +0.04..+0.07,
   ca-CondMat +0.03..+0.13 on the k-core axis) and comparable to the jitter floor
   (+0.06, +0.01, +0.07, +0.13). email-Enron is the exception and is NEGATIVE:
   core_ah_gap = +0.005 (dspar 0.5), -0.128 (lspar 0.5), -0.318 (lspar 0.2), i.e. the
   high-coreness nodes are the ones that move. Its granularity control accounts for
   more than half of that: resmatch core_ah_gap = -0.021 / -0.159 / -0.159, so the
   control is *stronger* than the sparsifier at lspar 0.5 and about half of it at
   lspar 0.2. The per-decile profile at email-Enron lspar 0.2 is monotone downward:
   0.358 0.368 0.309 0.081 0.082 0.065 0.044 0.035 0.033 0.014.
   [exp_O_core_preservation/results.csv `core_ah_gap`,`deg_ah_gap`,`ahc_d*`,`ahd_d*`;
   VERDICT.txt MEASUREMENT 2 block]

## Effect on the claims table

No claim changes; no new claim. H-O1 is closed as dead and does not enter the paper.
exp_O's descriptive sentence stands unmodified — the fragmentation is a granularity
effect, and Exp W now adds that it is not even leaf-selective: the nodes that shard off
are mid-degree, mid-coreness members, hubs are essentially never shed, and the lowest-
degree nodes are among the least likely to be shed. If one sentence is wanted for the
discussion, it is:

  The sub-10-node clusters that appear under sparsification are not the graph's
  loose leaves: their bottom-2 degree- and k-core-decile enrichment never reaches the
  2x we pre-registered (max 1.53 / 1.66 over nine cells, against attainable ceilings of
  2.7-4.9), on email-Enron their median degree exceeds the graph's (4-5 vs 3) and only
  0.5-3% of them sit in the 1-core against 28% of the graph, and the per-decile shedding
  rate peaks in the middle of the degree distribution while the top decile is never shed.
  A degree-preserving configuration null is MORE low-degree-selective than the real graph
  in all eight comparable cells.

## Caveats and deviations from DESIGN.md

- **C1 (scope of the test).** Only 9 of 15 real-arm sparsifier cells are testable at the
  >= 20-fragment threshold: email-Eu-core has 0 fragments in all three cells and
  wiki-Vote 5-9. So P1 rests on three networks (ca-HepTh, ca-CondMat, email-Enron).
  DESIGN anticipated this ("lspar 0.2 where fragments exist"); the wiki-Vote numbers are
  reported as a sensitivity and are the only ones pointing the predicted way, on 5-9
  nodes.
- **C2 (decile ties).** Inherited from exp_O C2. Degree and coreness are heavily tied at
  their minima (F = 0.21-0.43 at the bottom-2 boundary), so the bottom-2 deciles are a
  random subset of the minimum-degree block and enrichment is capped at 1/F. TABLE 0
  reports the per-network ceiling; all ceilings exceed 2.0, so P1's failure is not a
  binning artifact. The tie-free complements (median degree/coreness of fragment nodes,
  P(deg <= 2 | fragment), P(k-core <= 1 | fragment) and their lifts) were added for
  exactly this reason and agree with the decile verdict in every cell.
- **C3 (the extreme cells are degenerate).** In the lspar 0.2 cells 54-86% of all nodes
  are fragments (ca-HepTh 0.78, ca-CondMat 0.86, email-Enron 0.54), so any composition
  statistic is mechanically near 1.0. They are counted as testable because DESIGN named
  them, but they carry almost no information; the informative cells are dspar 0.5 and
  lspar 0.5.
- **C4 (thin control in one cell).** The resolution-matched control for ca-CondMat
  dspar 0.5 has only 12 fragment nodes, so its ratio (1.90 on k-core) is noisy. Dropping
  it takes P2's k-core exceedance count from 2/9 to 1/8 and does not change any verdict;
  the both-axes exceedance cell (ca-HepTh dspar 0.5) has a control with 103 fragments.
- **C5 (deviation, cells).** DESIGN names dspar 0.5, lspar 0.5, lspar 0.2. dspar 0.8 was
  NOT re-run: exp_O showed it produces essentially no sub-10-node clusters on 4/5
  networks, so it would add only untestable cells.
- **C6 (addition, labelled).** The plain-Leiden seed-jitter condition (seeds 101/102)
  was added as a "no sparsification, no resolution change" floor. It replaces no
  registered arm and enters no P1-P3 count; it is reported as finding 5.
- **C7 (deviation, testability threshold).** DESIGN says "cells where fragments exist"
  but sets no minimum. A cell is called testable at >= 20 fragment nodes; the threshold
  was fixed before the composition numbers were read, on the basis of exp_O's own
  statement that 5-9 fragment nodes is untestable, and the >= 1-node reading is reported
  alongside. Neither reading changes the verdict (0/9 and 0/12 respectively).
- **C8 (efficiency deviation, not a measurement change).** The resolution-matched arm
  reuses the gamma values exp_O's bisection produced (read from
  exp_O_core_preservation/results.csv) rather than re-bisecting. Since Leiden is seeded
  and deterministic this reproduces exp_O's resmatch partitions exactly — verified by
  the k' and agreement checks in finding 1.
- **C9 (inherited).** exp_O C1, C6 and C9 carry over unchanged: P0 is a single Leiden
  seed (100) per graph, the objective is ModularityVertexPartition with n_iterations=2,
  and the resolution-matched arm necessarily uses RBConfigurationVertexPartition.
  com-DBLP and com-Amazon were not run (exp_O C8; EXPLORATION.md rule 10).
- **C10 (post-hoc, labelled).** Finding 4's "fragmentation is a mid-degree phenomenon"
  is a direct read of the pre-registered measurement (fragment composition by degree
  decile), but the *interpretation* — that shedding requires a node to have enough edges
  to be cut loose but few enough to be cut loose from — is post-hoc and would need its
  own pre-registration before it is claimed as a mechanism.
