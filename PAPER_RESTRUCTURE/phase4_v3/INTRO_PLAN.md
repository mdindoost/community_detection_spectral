# The Introduction — architecture before prose

**Title (agreed 2026-07-26): "The Boundary of Graph Sparsification for Community Detection."**
**Rule: v3 is written clean. No prose, framing, or ordering is inherited from v1 or v2.**
**Order: introduction first (it carries the story); abstract and conclusion last.**

This file is for THINKING. No paragraphs are written until the beats are agreed.

---

## What the introduction has to accomplish

An introduction earns its length by doing four jobs. Ours must:

1. Make the reader want the answer (why the question is worth a paper).
2. Explain why the answer is not already known — this is the unusual one, and it is where
   our paper's identity lives. The reason is a *measurement* failure, not a lack of effort.
3. State the answer concretely enough that the reader could disagree with it.
4. Make the reader trust that we did not simply find what we set out to find.

Job 4 is unusual to state explicitly, but it matters here: the paper reports a negative over most
of its territory and a positive at the edge, and the reader's natural suspicion is that we picked
a framing to suit our results. The counter is the pre-registration record — including hypotheses
of ours that died — but the introduction should *earn* that trust structurally, not by protesting.

---

## Proposed beats (the argument, one idea per beat)

**B1 — The practical question.**
Communities must be found on graphs whose edge counts, not node counts, are the cost driver.
Nodes cannot be discarded (every node needs a label); edges can. So: can we delete most edges,
keep the communities, and go faster? This is a natural, well-posed, fifteen-year-old question.
*Ends on:* the question deserves a definite answer and does not have one.

**B2 — The answer the field believes, and where it came from.**
Satuluri, Parthasarathy and Ruan (SIGMOD 2011) answered yes and answered it carefully:
large speedups, and higher accuracy for two of four clustering algorithms. Their claim was
scoped — fixed-k balanced partitioners, dense noisy graphs, external ground truth, quality always
scored back on the original graph, sparsification time charged to the speedup. Methods in this
line now ship in NetworKit, CRAN, Neo4j GDS, and a production system (SimClusters).
*Ends on:* what the field repeats today is not what that paper claimed.

**B3 — The scope drift, stated without accusation.**
Over fifteen years, "faster clustering for these algorithms on dense noisy graphs" became
"sparsify before you detect." The degree threshold the original authors themselves published
(their own LFR sweep: the method begins to beat the original graph at average degree ~50) is
absent from the descendants. And one control was lost.
*Ends on:* naming the control.

**B4 — The lost control, and what its absence costs.** *(the hinge of the paper)*
If you sparsify a graph and then measure your partition's quality *on the sparsified graph*, you
are grading against evidence you have already edited: the sparsifier preferentially removes
inter-community edges, so the sparse graph flatters every partition scored on it. The correct
comparison scores both partitions on the original graph. This is not our invention — it is stated
in the 2011 paper's §4.3 — but it is absent from much of what followed, present as a default in
the reference benchmark's own released code, and absent from two 2026 papers, one of which reports
a modularity above the maximum attainable on the graph it describes.
*The number that makes this concrete:* restoring the control flips the sign of the conclusion in
45 of 63 controlled cells across four detection algorithms.
*Ends on:* if the yardstick was wrong, the boundary was invisible. Hence this paper.

**B5 — What we did.**
Rebuilt the evaluation (honest transfer scoring, granularity matching, chance floors,
configuration-model nulls, cost-matched baselines), then re-asked the question across both
sparsifier families that ever claimed gains — degree-based and similarity-based, including the
2011 method itself — four detection algorithms, and a controlled sweep of the one variable the
original authors said mattered: average degree.
*Ends on:* the design is a boundary-finding design, not a refutation design. Say it plainly.

**B6 — The boundary (the answer, concretely).**
- Below the threshold, for modularity-family free-granularity detectors: no gain, on any metric,
  for either sparsifier family, at any retention that preserves quality — and every gain reported
  in that regime dissolves into granularity or chance under control.
- Above average degree ~50, for fixed-k balanced partitioners: a genuine gain on both the
  objective and chance-corrected recovery, at almost exactly the threshold the 2011 authors
  published, and not reproduced by degree-based or uniform sparsification at matched retention.
- The variable that decides is not the sparsifier. It is the algorithm family, the graph's
  density, and the yardstick.
*Ends on:* the boundary is real, narrow, and was published in outline in 2011.

**B7 — Contributions, as a short list.**
(i) The boundary, with the threshold and the algorithm family identified.
(ii) The protocol, with each control motivated by a specific failure it catches, and the measured
cost of omitting it.
(iii) The mechanism: why the boundary sits where it does.
(iv) An open phenomenon: the objective and ground-truth recovery come apart — the partitions that
best recover reference communities are not the ones that score best on modularity.

---

## Decisions — SETTLED by Mohammad, 2026-07-26

**D1 — SETTLED: no names in the introduction.** The contemporary instances of the missing control
(including a benchmark's released code line and a reported modularity above its graph's maximum)
live in related work, with the arithmetic. The introduction stays about the science. Wherever they
are named, fairness requires noting that the 2011 paper did it right.

**D2 — SETTLED: our own earlier draft is NOT mentioned. It was never published.**
Mohammad's ruling, and it is correct: an unpublished manuscript is not part of the literature and
there is nothing to disclose. Confessing to a paper no one has seen would be performance, not
candour. The internal record (PAPER_RESTRUCTURE/) keeps the history; the paper does not.
*Consequence:* B4 and B5 carry no self-referential clause.

**D3 — SETTLED: preview the dissociation as an open phenomenon**, one sentence in the contribution
list, never stated as a result.

**D4 — SETTLED: no length constraint while drafting.** Write it complete; cut only what proves
unnecessary once the body exists. Do not pre-compress.

**D5 — SETTLED: mechanism gets one line in B6 and one contribution bullet**; full treatment in its
own section.

---

## Beats explicitly NOT in this introduction (guarding against v1/v2 habits)

- No "we audit prior claims" framing. The paper finds a boundary; it does not prosecute.
- No artifact catalogue enumerated in the introduction. One control, named and quantified (B4),
  is more forceful than three listed.
- No spectral-guarantee material up front. It belongs in background; leading with it makes the
  paper look like a theory paper it is not.
- No hedging on the positive result. Exp AA's finding is stated as flatly as the negative.
