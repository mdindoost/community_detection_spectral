# Exp AC — Does the transition depend on degree heterogeneity? An SBM arm.

Pre-registered 2026-07-26 BEFORE any Exp AC code ran. Requested by Mohammad after he asked
whether the synthetic evidence covers SBM as well as LFR. It does not: a repository-wide
search confirms **every synthetic graph in this study comes from `nx.LFR_benchmark_graph`**
(exp_A, exp_AA, exp_J). No stochastic block model appears anywhere.

**Why this matters.** LFR and SBM differ in exactly the dimension the mechanism runs through.
LFR has a power-law degree distribution; the plain SBM is degree-homogeneous. Our own Exp A
found standard LFR to be nearly degree-homogeneous relative to real graphs (degree CV ~0.4
against 1 to 26 in our suite) and to lack degree-community coupling. Since a degree-based
sparsifier's entire effect is governed by a degree-derived score, and since Exp AA's threshold
already moved from ~50 on LFR to <=29 on real graphs, the location of the transition is
plausibly a function of degree heterogeneity rather than of density alone.

**Additional reason:** the two most relevant pieces of adjacent literature both work in SBM.
Laeuchli's analysis (references/NOTES_chen_and_fastcd.md, Paper B) is entirely SBM-based, and
the 2026 effective-resistance paper (references/NOTES_effres_2026.md) evaluates on SBM. A
referee from that side of the literature will ask.

**Claims at risk:** C19 and C20 (the boundary and its location), and the scope sentence in the
introduction. If the transition sits at a very different density under SBM, then density is
confirmed as a proxy and degree heterogeneity is the operative variable, which sharpens the
paper. If it sits in the same place, density stands on its own and the claim generalizes
across two generators.

## Design

Mirror Exp AA's Arm A and Arm B exactly, substituting the generator. Everything else
(sparsifiers, retentions, detectors, metrics, controls, seed counts) is held identical so the
two experiments are directly comparable cell for cell.

- **Generator 1, plain SBM:** `nx.stochastic_block_model`, n = 10^4, equal-sized blocks,
  block count chosen to match the LFR community-count regime (target mean community size
  100 to 250). Within-block and between-block probabilities set to hit the target average
  degree and a mixing parameter comparable to the LFR runs (realized mixing reported, not
  assumed). Degree distribution is Poisson, so CV ~ 1/sqrt(d).
- **Generator 2, degree-corrected SBM:** same block structure with a power-law degree
  propensity (exponent 2.1, matching the LFR runs), so that degree heterogeneity is restored
  while the block model is retained. This is the arm that separates "generator family" from
  "degree heterogeneity".
- **Average degree sweep:** 10, 25, 50, 100, 200, as in Exp AA. 3 graphs per cell.
- **Sparsifiers:** L-Spar, DSpar, uniform random at matched realized retention.
- **Detectors:** Leiden (free granularity) and real Metis via pymetis with k pinned to the
  planted block count, exactly as in Exp AA Arm B.
- **Metrics and controls:** honest transfer scoring on the original graph; best-of-5 restart
  baseline alongside the runtime-matched one (the Exp AA lesson); AMI/ARI against planted
  labels with a size-matched chance floor; resolution-matched control for Leiden; realized
  retention and realized mixing reported per row; per-graph degree CV reported so the
  heterogeneity axis is auditable.

## Pre-registered predictions

- **P1:** under Metis, the sign flip occurs on both SBM variants, since the fixed-$k$
  mechanism (removal of misleading between-block edges) does not require heterogeneous
  degrees.
- **P2:** the transition occurs at a **higher** average degree under plain SBM than under
  LFR, because a homogeneous degree sequence gives a degree-based or similarity-based
  selector less signal to exploit at a given density.
- **P3:** the degree-corrected SBM transition falls closer to the LFR value than the plain
  SBM one does, identifying degree heterogeneity rather than generator family as the
  operative variable.
- **P4:** under Leiden, no honest modularity gain on either SBM variant at any degree,
  reproducing Exp AA and Exp AB.

## Kill criteria (bidirectional)

- If **P1 fails** and Metis shows no gain on SBM at any density, then the Exp AA result is
  LFR-specific and the boundary claim must be restated as a property of power-law benchmark
  graphs and real networks, not of dense graphs generally. Report loudly.
- If **P4 fails** and Leiden gains honestly on SBM, the negative half of the boundary needs a
  generator-scoped exception.
- If **P2 and P3 both hold**, degree heterogeneity is promoted from a suspected confound to a
  named axis of the boundary, and the introduction's density condition should be restated in
  terms of removable redundancy with heterogeneity as a modifier.

## Execution

Fuji (empty datasets directory, ~5.5 GB free; synthetic graphs are generated in place and are
small). One detached capped job at a time, incremental CSV appends, commit per generator.
Estimated a few hours; the n = 10^4 at average degree 200 cells are the slow tail, as in
Exp AA.

## Output
PAPER_RESTRUCTURE/exp_AC_sbm_generality/{DESIGN.md,run.py,results_sbm.csv,
results_dcsbm.csv,SUMMARY.md}.
