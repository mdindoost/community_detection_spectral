# Handoff, end of session 2026-07-26 (night)

Supersedes the evening handoff. Exploration is finished: 25 experiment folders, 24 with
verdicts (exp_J is figures). We are writing v3, and the paper was restructured tonight.

`Paper_materials/v3/` is clean, self-contained and Overleaf-ready.

## NEXT SESSION'S TASK, set by Mohammad before it started

A three-stage review of everything written, **in this order**, done by reading the files and the
CSVs rather than by trusting any summary:

1. **Read §1.** The story changed tonight, so check that the introduction still tells it and that
   nothing major was lost. Do not rely on recollection of what it used to say.
2. **Examine §3.** Validate that we actually do what the protocol claims we do, and ask of each
   control whether it is necessary.
3. **Re-check §4.** Every number and every claim, with the whole picture in mind.

Mohammad chose to start a fresh session for this deliberately: the reviewer should not be the
author, and every error caught this session was caught by opening a CSV rather than by
remembering.

## Read first
1. `Paper_materials/v3/README_WORKFLOW.md` file map and the two-editor sync protocol
2. `Paper_materials/v3/STYLE.md` hard rules (no em-dashes, no AI phrasing) with grep checks
3. `PAPER_RESTRUCTURE/phase4_v3/PAPER_STRUCTURE.md` framing and the one-sentence claim.
   **Its section-weights table is now STALE**: it describes the nine-section paper that was
   replaced tonight. The framing and claim sections above it are current.
4. `PAPER_RESTRUCTURE/TERMINOLOGY.md` fixed vocabulary; "cross-graph" is retired
5. `PAPER_RESTRUCTURE/phase4_v3/INTRO_PLAN.md` the seven beats and decisions D1-D5

## The claim the paper defends (amended twice on 2026-07-26)

> Sparsification's effect on community detection is decided by the detector's degrees of freedom
> and by the graph's removable redundancy rather than by the sparsifier; outside those two
> conditions there is no gain on the objective, no end-to-end speed benefit, and no gain in
> recovery that survives comparison with an unsparsified run of a better detector, and the
> accounting in common use reports one anyway.

**This is a measurement and scoping paper, not a survey and not a benchmark.** The contribution
is the conditional answer, both halves as one finding. The protocol is the instrument that makes
it credible and is priced (45 of 63), not asserted. Do not reframe the gap as "papers use
different protocols so comparison is hard": that is a standardization pitch, weaker than our
evidence, and it puts us against Chen et al.\ on breadth where we lose. Our evidence says the
common accounting **inverts** the answer.

## Structure, rebuilt tonight to Mohammad's design

Six sections. The old nine-section shape (separate protocol, below-the-boundary, boundary,
mechanism, related work) is gone.

| § | State | File |
|---|---|---|
| 1 Introduction | written, 2058 w, **revise last** | `01_introduction.tex` |
| 2 Background and Related Work | **not written** | — |
| 3 Evaluation Protocol | written, 1485 w, **argument only** | `03_protocol.tex` |
| 4 Experiments | complete, 6156 w | five files, below |
| 5 Discussion | **not written** | — |
| 6 Conclusion | **write last** | — |

Section 4 is `04_experiments_setup.tex` (opening + setup), `04_accounting.tex` (4.2 what the
accounting costs), `04_negative.tex` (4.3 where it does not help, 4.4 four ways the loss could be
an artifact, 4.5 the exception), `04_boundary.tex` (4.6), `04_mechanism.tex` (4.7).

**The design rule for §3, and the reason the restructure happened:** the protocol section states
each control and the reason for it, argued from a definition, with **no measured consequences, no
cell counts and no networks named**. All evidence lives in §4. A protocol section that carries no
measurements cannot duplicate the experiments, which is what both external reviewers attacked.

`sections/_mined/` holds the pre-restructure versions: `03_protocol_v1_full.tex`,
`04_below_v1_full.tex`, `05_boundary_retired.tex`. None is input by `main.tex`. They are the
source record; do not delete them and do not reinstate them without checking their numbers,
since several were wrong.

## Everything corrected on 2026-07-26, and why

Each of these was written confidently and was wrong. The pattern is one thing: a figure taken
from a SUMMARY and attached to a sentence, without opening the CSV.

1. **§3.1 mechanism of self-scoring inflation.** Said the null term is normalized by $m$ so
   deleting edges shrinks it. False: under uniform deletion the intra fraction and the null term
   shrink together and exp_G's fixed-partition arm measures the change at $-0.001$ to $+0.0002$.
   The inflation is re-optimization on a graph that admits higher modularity (Guimera 2004).
   Found by the codex review, confirmed against exp_G.
2. **§4 exclusion paragraph.** Claimed Metis had no implementation available. It has one, we had
   used it in three experiments, and exp_AB had already run it on two networks. Led to exp_AD.
3. **"Four of the arms"** that the backbone beats: three.
4. **exp_P SUMMARY's gamma column** contains values from two other runs. The com-DBLP Louvain
   control is $\gamma = 640$, not 576 (that is Leiden's), and the DSpar control is 24, not 40
   (that is com-Amazon's).
5. **exp_V's "5.0 pooled sd"** belongs to the shuffled arm against the baseline, not the weighted
   arm against the matched control, which is 3.7.
6. **"1 to 2 restarts"** on the email-Enron label-prop cells: they drew four and five.
7. **Clip fraction "13 to 35 per cent"**: measured range is 11.96 to 35.70.
8. **"0.662 retention at alpha=1.0"**: no such row exists anywhere. Recomputed from the sampler's
   own formula, validated against `clip_fractions.csv` at alpha=0.95, the value is 0.665.
9. **With-replacement sampler "0.33 to 0.55"**: exp_B's realized range is 0.332 to 0.515.
10. **The retired §5's "factor of between two and sixty"** for seed robustness. Recomputed per
    cell: $-1.32$ to $5.79$ on modularity and $-1.88$ to $14.36$ on AMI, with several cells
    negative at retention 0.5.
11. **The retired §5's "harmful on both measures below degree 25"**, contradicted by the table
    printed directly above it (6/12 positive on AMI at 24.6).
12. **The retired §5's speed figures** (1.49x / 1.01x / 0.58x). Arm B medians are 0.75x at
    $d=10$ falling to 0.26x at 200.
13. **"Degree-based and uniform do not reproduce the effect at any degree."** False on LFR at the
    two highest degrees (DSpar 2/12 and 3/12) and false on real data (exp_AD).
14. **exp_AB SUMMARY's "0.00% fragment nodes"** for the backbone: 0.00 on most cells, up to 1.8%
    for random fill and 13.7% for Jaccard fill. What is exactly zero is singletons.
15. **STORY.md C8's "AMI 0.82-0.94"** is the UP arm only; the DOWN ends re-detect at 0.41-0.54.
    Now scoped in STORY.md.

**One flag that was my error, not the data's:** exp_M's decomposition terms are defined relative
to the real arm, so real rows are $\log 1 = 0$ by construction. Computed on the null rows, where
the decomposition is meaningful, the sorting term is positive on 16 of 16 networks with median
$+0.5743$ and the identity holds to $1.7\times10^{-14}$, exactly as the notes said.

## New experiment run tonight: exp_AD

`exp_AD_metis_real_networks/`, complete, with DESIGN, SUMMARY, 170 rows, both $k$ conventions.
Run because §4 claimed Metis was unavailable. Verdicts: P2 holds decisively (90 cells on the five
networks below average degree 11, zero positive); P1 half fails (wiki-Vote reproduces,
email-Eu-core does not survive ten seeds where exp_AB used three); P3 fails (DSpar produces the
gain on wiki-Vote, so the similarity attribution is an LFR statement); P4 holds on the objective
and fails on recovery. Read its SUMMARY before writing anything about the fixed-$k$ arm.

## Open decisions for Mohammad
- **Venue.** codex says TNSE; the Opus reviewer argues PVLDB Experiments and Analyses, because
  TNSE reviewers ask what the new method is and our answer is that there is not one. TNSE was
  chosen when this was a different paper.
- **Where the self-scoring arithmetic goes** (a published modularity of 0.87 on a graph whose
  maximum over all partitions is 0.6046, and 0.63 on karate whose maximum is 0.4198). The Opus
  reviewer wants it in the accounting subsection; my view is a pointer there and the arithmetic in
  background and related work.
- **§2 must define removable redundancy** or the claim's second clause floats.
- **A $k$ sweep on wiki-Vote** is the obvious next experiment: the only surviving real-network
  positive rests on one unlabelled network at $k=6$, which could not be cross-checked.
- **MLR-MCL**, time-boxed. It is the algorithm the founding claim is largest about and we have not
  run it. If it will not build, the paper should say so.

## Standing facts. Do not relearn them.
1. **SUMMARY files are not citable. Go to the CSVs.** See the fifteen corrections above.
2. The 45 of 63 spans **three** algorithms (Infomap, Louvain, label propagation). Leiden's
   evidence for the same artifact is separate. TERMINOLOGY.md's "four" is loose.
3. **"Every apparent gain dissolves under the three controls" is FALSE.** L-Spar's recovery gain
   on com-Amazon survives granularity matching, an over-matched control and the chance floor. What
   defeats it is an unsparsified run of a better algorithm.
4. Quality sweeps ran on **seven** networks; mechanism work on **seventeen**.
5. The ~50 threshold is an **LFR** statement. Real graphs: absent below 10.7, present at 28.5 on
   one of the two networks above it. SBM: no later than LFR.
6. Satuluri et al. did NOT commit Artifact I. They forbade it in print in 2011, matched cluster
   counts, and charged sparsification time. Three original charges were withdrawn.
7. exp_AC excluded degree heterogeneity as the operative variable (+0.65 density against -0.26
   heterogeneity), and its speedups (1.4x to 2.3x) contradict exp_AA and the real networks, so no
   speed claim may cite them.
8. **Local python has no numpy, pandas, igraph or pymetis.** Run analysis on Fuji
   (`~/md724/venv/bin/python`), and copy any referenced experiment's CSVs there first.
9. Fuji datasets are disposable; "no disk" is solvable, not a blocker.

## Working rules Mohammad set
- **Do not start writing sections unprompted.** "Continue" from a handoff is not authorization.
- **Edit exactly what he points at.** If the same flaw appears elsewhere, say so and leave it.
- **Do not optimize prose for length.** Academic merit decides; shortening is a separate
  conversation about named passages.
- **The introduction names conditions; the body proves them.**
- **No sentence may state a truism as a principle**, and no sentence may explain the paper's own
  editorial reasoning inside the paper.
- Commits and pushes to `refactor_v2` never need asking.
