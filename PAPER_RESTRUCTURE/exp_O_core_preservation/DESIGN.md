# Exp O — core-preservation / periphery-fragmentation (pre-registered design)

Registered 2026-07-25 in EXPLORATION.md (commit 953aa10) BEFORE any Exp O code or run existed.
This file is a verbatim extract; EXPLORATION.md is authoritative.

**Hypothesis (Mohammad's, stated before any dedicated experiment):** under edge sparsification
the sparse-graph partition shatters into many small clusters, but these are loose/peripheral
connections; the CORES of the main communities are preserved. Fragments of support already in
hand: nc explosion (exp_L results.csv: ca-CondMat 55→503→5157; com-DBLP 227→10,950→72,736);
Exp N finding 4 (movers are periphery-tilted, hubs stay); Chen PVLDB 2024
"fragmentation-with-pruning".

**Claim it could change:** upgrades C5/C6's "slightly harmed / not improved" into a structural
statement — "sparsification preserves community cores and reorganizes hub-mediated boundaries
and loose periphery" — unifying with Exp N's mechanism. New claim number C16 if it survives.

**Design.** Networks: email-Eu-core, wiki-Vote, ca-HepTh, ca-CondMat, email-Enron (small five
first; com-DBLP, com-Amazon only after the small five pass sanity). Sparsifiers: DSpar
(calibrated sampler, retentions ~0.5 and ~0.8) and L-Spar (targets 0.5, 0.2; reuse
exp_L_lspar/run.py machinery). Seeds: 3 Leiden seeds per arm minimum.
Per node, computed on the ORIGINAL graph + its partition P0: embeddedness = fraction of
neighbors in own community; k-core index; degree. Then:
(1) **Agreement-by-embeddedness**: node-level agreement between P0 and sparse-graph partition
    P' (community-matched by Hungarian/plurality mapping), stratified by embeddedness decile.
(2) **Core cohesion**: for each P0 community ≥20 nodes, core = top-50%-embeddedness members;
    cohesion = largest fraction of the core landing in one P' community.
(3) **Fragment composition**: nodes in P' clusters of size <10 — their embeddedness decile
    distribution vs the graph's.
**Controls:** (a) config-null arm — same pipeline on degree-preserving rewired graph (its own
P0_null); structure-specificity check. (b) Resolution-matched control — partition the ORIGINAL
graph with resolution tuned to match P''s community count; compute (1)-(3) identically; the
hypothesis requires sparsification's fragmentation to be MORE embeddedness-selective than mere
resolution. (c) Chance correction: permutation baseline for agreement.

**Pre-registered predictions:**
- P1: agreement rises monotonically with embeddedness decile; top-3-decile agreement >0.85 at
  retention 0.5 for DSpar.
- P2: sub-10-node P' fragments are ≥2x over-represented in the bottom-2 embeddedness deciles.
- P3: median core cohesion ≥0.8 at retention 0.5, both sparsifiers.
- P4: null arm shows a materially flatter agreement-embeddedness gradient (top-minus-bottom
  decile gap at most half the real graph's).
- P5: resolution-matched control does NOT reproduce the embeddedness selectivity of the
  fragmentation (its fragment composition is closer to uniform).

**Kill criterion:** if the agreement gradient is flat (top-3 minus bottom-3 decile gap <0.1) OR
the null arm reproduces the gradient within noise on ≥half the networks, the hypothesis is dead
and the observation stays a descriptive footnote. No rescue tweaks.

**Output:** PAPER_RESTRUCTURE/exp_O_core_preservation/{DESIGN.md,run.py,results CSVs,SUMMARY.md}.
