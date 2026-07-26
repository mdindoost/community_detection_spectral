# Handoff, end of session 2026-07-26 (evening)

Supersedes the afternoon handoff. Exploration is finished: 24 experiment folders, 23 with
verdicts (exp_J is figures). We are writing v3.

`Paper_materials/v3/` is clean, self-contained and Overleaf-ready. Nothing is inherited from
v1 (`main-tnse.tex`) or v2 (`main-v2.tex`); only the CSVs and the verified claims table feed it.

## Read first
1. `Paper_materials/v3/README_WORKFLOW.md` file map and the two-editor sync protocol
2. `Paper_materials/v3/STYLE.md` hard rules (no em-dashes, no AI phrasing) with grep checks
3. `PAPER_RESTRUCTURE/phase4_v3/PAPER_STRUCTURE.md` framing, weights, and the one-sentence
   claim, **rewritten this session** with the superseded version and what falsified it
4. `PAPER_RESTRUCTURE/phase4_v3/INTRO_PLAN.md` the seven beats and decisions D1-D5
5. `PAPER_RESTRUCTURE/TERMINOLOGY.md` fixed vocabulary; "cross-graph" is retired

## The claim the paper defends (rewritten 2026-07-26 evening)

> Sparsification's effect on community detection is decided by the detector's degrees of
> freedom and by the graph's removable redundancy rather than by the sparsifier; outside those
> two conditions there is no benefit of any kind, and the accounting in common use reports one
> anyway.

Settled with Mohammad the same evening: **this is a measurement and scoping paper, not a
survey and not a benchmark.** The contribution is the conditional answer, both halves as one
finding. The protocol is the instrument that makes it credible and is priced (45 of 63), not
asserted. Do not frame the gap as "papers use different protocols so comparison is hard": that
is a standardization framing, it is weaker than our evidence, and it puts us against Chen et
al.\ on breadth, where we lose. Our evidence says the common accounting **inverts** the answer.

## Written
- **§1 Introduction**, 1969 words. Revised four times this session: the two conditions now
  answer the question at the end of B1; B1's paragraphs 2 and 3 merged so that what a spectral
  guarantee preserves precedes DSpar; B3 and B6 rewritten to name conditions rather than prove
  them; B6 absorbed Exp AC.
- **§3 Evaluation Protocol**, 2260 words, three tables. Title is Mohammad's (over "What Honest
  Evaluation Requires": "honest" implies the alternative is dishonesty, which is not our
  position). Eight controls, each priced. Every number checked against source CSVs.
- **§5 The Boundary** exists as `sections/05_boundary.tex` but is **commented out of main.tex**
  at Mohammad's instruction. It was written before §3 existed and before the terminology was
  fixed; it returns once the body is rewritten and can be reconciled. Do not delete the file.

## Not written
§2 Background, §4 Below the Boundary, §6 Mechanism, §7 Related Work, §8 Discussion.
**Abstract and conclusion are written LAST.**

Section order in main.tex was settled this session and follows B6: background, protocol, below
the boundary, the boundary, mechanism, related work, discussion. §2 is deliberately written
after §4, because its content is determined by what §3 and §4 actually use.

## Next: §4, three decisions already made
1. **Tables.** One summary table of counts over the whole negative territory (shape of
   Table 2), plus one small table for the backbone test, since "the negative is not
   fragmentation" is a distinct claim. Per-cell detail to supplementary material.
2. **The com-Amazon exception gets its own subsection** at the end, named as an exception, and
   hands to the Discussion where the dissociation is contribution four.
3. **§4 opens with the seven-sparsifier rationale**: the two families that claim gains, Chen's
   three top clustering-fidelity preservers (so nobody can say we picked weak ones), a
   connectivity-preserving class that removes the fragmentation explanation by construction,
   and uniform random as the no-information control. Exclusions named in the same paragraph:
   no MCL/Metis/Graclus on real networks (python-igraph has none, so Satuluri's strongest
   algorithm is untested), no node-removing sparsifiers, no learned sparsifiers.

## Parked, not blocking
- **§6 is misnamed.** PAPER_STRUCTURE calls it "Mechanism: why the boundary is there". The
  delta work explains the *degree-based channel*; what explains the boundary is the
  noise-injection experiment in §5. Rename when §6 is drafted.
- **§2 must define removable redundancy** or the claim's second clause floats.
- **The claim's "no benefit of any kind" needs its exception clause** before the abstract is
  written against it (see standing fact 3).
- Venue: TNSE was chosen when this was a different paper. Revisit with the abstract.
- Exp S (com-Orkut) not started. Fuji disk is not a blocker: datasets there are disposable and
  get deleted when idle.

## Standing facts that cost us time. Do not relearn them.

1. **SUMMARY files are not citable. Go to the CSVs.** Four wrong numbers reached §3 from
   summaries this session and were caught only by recomputation: exp_P finding 4's gamma column
   holds values from two other runs (Leiden's 576 and com-Amazon's 40 appear in com-DBLP
   Louvain rows; the truth is 640 and 24); exp_V's "5.0 pooled sd" belongs to the shuffled arm
   against the baseline, not the weighted arm against the matched control (3.7); the "1 to 2
   restarts" figure is modal across a sweep, while the cells quoted drew four and five; the
   clip-fraction range is 12 to 36 per cent, not 13 to 35; the with-replacement sampler realizes
   0.33 to 0.52, not 0.55; and "0.662 retention at alpha=1.0" had no row anywhere (recomputed:
   0.665).
2. **The 45 of 63 spans three algorithms**, Infomap, Louvain and label propagation. Leiden's
   evidence for the same artifact is separate. TERMINOLOGY.md's "four algorithms" is loose.
3. **"Every apparent gain dissolves under the three controls" is FALSE.** L-Spar's recovery gain
   on com-Amazon survives granularity matching, an over-matched control and the chance floor
   (exp_V V5); exp_AB found a second instance from an unrelated family. What defeats it is an
   unsparsified run of a better algorithm (exp_P V4). This overreach was in the introduction all
   day and is now corrected in B6.
4. The quality sweeps ran on **seven** networks; the mechanism work on **seventeen**.
5. exp_AA's degree threshold of ~50 is an **LFR** statement. exp_AB found the fixed-k gain at
   average degree 28.5 on real graphs. The paper states a condition, not a threshold.
6. Satuluri et al. did NOT commit Artifact I. They forbade it in print in 2011, matched cluster
   counts, and charged sparsification time. Three original charges were withdrawn.
7. **Exp AC (new, complete):** degree heterogeneity is EXCLUDED as the variable setting the
   transition. P1 and P4 hold, P2 and P3 falsified. Spearman(density, gain) +0.65 against
   Spearman(degree CV, gain) -0.26. See its SUMMARY for five results recorded against us,
   including that **Exp AC's speedups (1.4x to 2.3x) contradict exp_AA and the real networks,
   so no speed claim may cite them.**
8. **Local python has no pandas.** Run analysis on Fuji (`~/md724/venv/bin/python`), and copy
   any referenced experiment's CSVs there first: exp_AC's analyse.py needed exp_AA's.

## Working rules Mohammad set this session
- **Do not start writing sections unprompted.** "Continue" from a handoff is not authorization.
- **Edit exactly what he points at.** If the same flaw appears elsewhere, say so and leave it.
- **Do not optimize prose for length.** Academic merit decides; shortening is a separate
  conversation about named passages.
- **The introduction names conditions; the body proves them.** Numbers that a later section owns
  do not belong in §1.
- **No sentence may state a truism as a principle** ("Retention is measured, never assumed") and
  no sentence may explain the paper's own editorial reasoning inside the paper.
