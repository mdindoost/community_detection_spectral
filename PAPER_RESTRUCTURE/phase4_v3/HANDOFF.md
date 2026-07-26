# Handoff, end of session 2026-07-26

## Where we are
Exploration is finished (24 experiment folders, 22 with verdicts). We are writing v3.

`Paper_materials/v3/` is clean, self-contained and Overleaf-ready. Nothing is inherited from
v1 (`main-tnse.tex`) or v2 (`main-v2.tex`); only the CSVs and the verified claims table feed it.

## Read first
1. `Paper_materials/v3/README_WORKFLOW.md` file map and the two-editor sync protocol
2. `Paper_materials/v3/STYLE.md` hard rules (no em-dashes, no AI phrasing) with grep checks
3. `PAPER_RESTRUCTURE/phase4_v3/PAPER_STRUCTURE.md` framing, section weights, the one-sentence claim
4. `PAPER_RESTRUCTURE/phase4_v3/INTRO_PLAN.md` the seven beats and the settled decisions D1-D5
5. `PAPER_RESTRUCTURE/TERMINOLOGY.md` fixed vocabulary; "cross-graph" is retired

## Written
- **§1 Introduction**, ~2070 words. Fully audited against the evidence: eight factual errors
  found and fixed, including two that contradicted our own newest data. Reduced over four
  passes of Mohammad's remove-and-check-flow loop.
- **§5 The Boundary**, from exp_AA.

## Not written
Background, protocol, negative territory, mechanism, related work, discussion.
**Abstract and conclusion are written LAST.**

## In flight
**Exp AC** (`exp_AC_sbm_generality/`), running detached on Fuji, writing `results_armA_sbm.csv`.
Tests whether the transition location depends on degree heterogeneity rather than density, using
plain SBM and degree-corrected SBM against exp_AA's LFR numbers. Pre-registered predictions and
bidirectional kill criteria are in its DESIGN.md. Collect and write its SUMMARY next session.

## Open decisions for Mohammad
- Two introduction elements flagged as removable but not cut, both judgment calls: the
  SimClusters sentence in the prior-work paragraph, and the mechanism sentence at the end of B6.
- Contributions bullet 1 is now heavier than the other three. Defensible, but if it reads as
  overloaded the natural split is to promote the connectivity-preserving result to its own bullet.
- Venue: TNSE was chosen when this was a different paper. Worth revisiting with the abstract.
- Exp S (com-Orkut feasibility) still blocked on Fuji disk space.

## Standing facts that bit us this session, do not relearn them
- The quality sweeps ran on **seven** networks; the mechanism work (delta decomposition,
  configuration nulls) ran on **seventeen**. Do not attribute seventeen to a quality claim.
- exp_AA's degree threshold of ~50 is an **LFR** statement. On real graphs exp_AB found the
  fixed-k gain already at average degree 28.5. The paper states a condition, not a threshold.
- "A resolution-tuned optimizer beats every sparsified pipeline" is **false on one real
  network**: Local Similarity on com-Amazon reaches avgF1 0.4704 against the sweep's best 0.4398.
- Satuluri et al. did NOT commit Artifact I. They forbade it in print in 2011, matched cluster
  counts, and charged sparsification time. Three of our original charges were withdrawn.
