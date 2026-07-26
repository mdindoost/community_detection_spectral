# Exp V — "spending the signal": can L-Spar's verified structure-awareness be converted
# into an honest, controlled gain? (pre-registered design)

Registered 2026-07-25, BEFORE any Exp V code or run existed. Greenlit by Mohammad.

**Context.** Exp L established that L-Spar's Jaccard signal genuinely sees community structure
(deltaJ collapses on config nulls — the only sparsifier in the study with that property) yet
deploying it by DELETION loses modularity in 14/14 cells. The edge-weighting literature
(Khadivi 2011; Berry 2011; Sun/adaptive-modularity 2017; triangle weighting 2014; iterative
reweighting 2024) claims quality gains from similarity weighting but evaluates with
weighted-graph modularity or uncontrolled NMI — the same artifact shapes this paper audits.
Exp V asks the constructive question: is there ANY deployment of the verified signal that
survives honest transfer scoring, sharp nulls, cost matching, and resolution-matched recovery?
Either outcome is a contribution: a controlled positive (first constructive result), or a
controlled negative upgrading the thesis to "even a verified structure-aware signal cannot be
converted into honest gains by weighting, seeding, or protected deletion."

**Claim it could change:** adds a constructive arm to the paper (new claim C17 if positive);
if negative, strengthens C5/C9 and the discussion's account of WHY the field's intuition fails.
Also adjudicates the com-Amazon live lead from Exp L (V3 exception).

## Design

Networks: the Exp L seven — email-Eu-core, wiki-Vote, ca-HepTh, ca-CondMat, email-Enron,
com-DBLP, com-Amazon. Loaders/conventions verbatim from exp_L_lspar/run.py. N_ITER=2,
ModularityVertexPartition (weighted variant where applicable). Jaccard J(u,v) exact, as Exp L.

Arms (per network):
- **A. Baseline**: plain Leiden on G, 5 seeds (100-104). Also the runtime-matching pool
  (seeds 900-…): for each treated arm, matched control = best of as many plain restarts as fit
  in that arm's measured wall clock (charging T_jaccard to every arm that computes J).
- **B. Weighting**: full graph, w(e) = 1 + J(e), weighted Leiden, 3 seeds (300-302).
  Partition scored on the UNWEIGHTED original graph (honest transfer). No edges removed.
- **B-shuffle (sharp null)**: identical to B but J values PERMUTED across edges
  (shuffle seeds 500, 501). Preserves the weight multiset, destroys alignment. Any gain in B
  that persists in B-shuffle is weight-heterogeneity mechanics, not the signal.
- **C. Seeding**: L-Spar at target retention 0.5 (e bisected as Exp L) -> Leiden on the sparse
  graph (3 seeds) -> refinement on the full original graph from that initial membership
  (leidenalg initial_membership + optimise; pipeline shape of Exp N). Honest scoring is
  automatic (final partition lives on G).
- **D. Protected deletion**: at total retention 0.5 — keep L-Spar's selection at the largest e
  whose retention <= 0.5, fill the remainder with uniformly random non-selected edges to reach
  0.5 exactly (2 fill seeds x 3 Leiden seeds). Compare against pure L-Spar 0.5 and uniform 0.5
  (uniform arm: random 50% of edges, same seeds). Isolates deletion-selection from deletion.
- **Config-null arm**: B and C repeated on the degree-preserving rewiring (REWIRE_SEED=42,
  10 swaps/edge) of email-Eu-core, ca-CondMat, email-Enron (structure-specificity check,
  matching Exp L's null coverage).

Recovery (labelled sets, all with resolution/granularity-matched controls as in Exp L):
email-Eu-core (AMI/ARI/NMI), com-DBLP and com-Amazon (avgF1_ge3) for arms B and C.
com-Amazon additionally gets a chance floor: avgF1_ge3 of size-matched random partitions
(3 draws) so the anomaly can be read against chance; this is the pre-registered follow-up of
Exp L's V3 exception.

Metrics recorded per cell: Q_orig(P_arm) mean/sd/best, dQ_honest_vs_mean/best, dQ_vs_matched
(runtime-matched), wall-clock decomposition, k (cluster count), and for recovery the matched
controls at the arm's k.

## Pre-registered predictions

- P1 (weighting): B improves on pure L-Spar deletion in every network (less harm than Exp L's
  same-network cells) but achieves dQ_vs_matched > 0 on at most 2/7; and any B gain COLLAPSES
  in B-shuffle (shuffle arm within noise of baseline).
- P2 (seeding): C achieves dQ_vs_matched > 0 on at most 2/7 networks; email-Enron is the most
  likely positive (mechanism continuity with Exp N's DSpar seeding).
- P3 (protected deletion): D strictly reduces the honest-transfer loss vs pure L-Spar at
  matched retention on >= 5/7 networks, but stays negative vs baseline on >= 6/7.
- P4 (recovery): no arm's recovery gain survives the granularity control on email-Eu-core or
  com-DBLP; com-Amazon is the open cell — if an arm beats the over-matched control there with
  the chance floor subtracted, the Exp L anomaly is PROMOTED from hypothesis to result.

## Kill criterion

The constructive claim (C17) is DEAD unless at least one arm achieves dQ_vs_matched >= +0.002
beyond seed noise (>2 pooled sd) on >= 2/7 networks, OR a recovery win survives the matched
control AND the chance floor on a labelled network. If dead: report as the strengthened
negative ("the signal cannot be spent"), no rescue tweaks, no new arms bolted on post hoc.
The com-Amazon adjudication is reported either way (promote / demote the Exp L hypothesis).

## Output
PAPER_RESTRUCTURE/exp_V_spend_the_signal/{DESIGN.md,run.py,results.csv,recovery.csv,
null_arm.csv,SUMMARY.md}. Cost estimate: Exp-L-sized; local machine, MemoryMax caps,
one job at a time; com-DBLP/com-Amazon weighted-Leiden cells are the slow tail.
