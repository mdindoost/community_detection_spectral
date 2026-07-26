# v3 — Framing, spine, and section weights

**Settled 2026-07-26 with Mohammad.**

---

## The framing decision: CONTRIBUTION PAPER, not a review

**Ruled: contribution.** The literature is motivation and related work. It is never the spine.

*Why this was a real decision.* Our story is organized around other people's papers — scope drift,
an artifact catalogue, four papers audited — which is review-flavoured packaging on
contribution-flavoured content. Left unchecked, the paper would read as a survey, put us
head-to-head with Chen et al. (PVLDB 2024) on breadth (12 sparsifiers x 16 metrics x 14 graphs)
where we lose, and invite the "what is new here?" review.

*The test that settled it.* Strip out every summary of other people's work. What remains:

1. **The boundary** — a degree threshold (~50) and an algorithm-family split that nobody has
   published. Satuluri et al. state the threshold in one sentence about an LFR sweep and never
   return to it; we establish it with real Metis, k pinned by construction, seed-robust, with
   matched-retention controls showing it is the similarity signal and not the edge budget.
2. **The causal mechanism result** — steering hub-edge placement at fixed degree sequence, frozen
   partition, exactly fixed intra-edge fraction and modularity; delta moves 20-450 sd in either
   direction and survives re-detection (AMI 0.82-0.94).
3. **The shuffled-weight null** — a new instrument that retroactively undermines the
   edge-weighting literature's evaluations. Nobody runs it.
4. **The measured cost of the missing control** — 45 of 63 cells flip sign across four detection
   algorithms. An original measurement; Chen never computes an objective at all.
5. **The objective/recovery dissociation** — a phenomenon, stated for the first time.

Five original contributions. A review has zero.

## The one sentence the paper exists to defend

> **Sparsification's effect on community detection is determined by average degree and algorithm
> family, not by the sparsifier — and the threshold sits near average degree 50.**

Every section either supports that sentence or explains why nobody had found it. Anything that
does neither is cut.

*Corollary settled at the same time:* **the boundary leads; the yardstick is the instrument that
found it.** A contribution paper's headline is its finding, not its method — even when the method
is itself a contribution.

---

## Section weights (page targets are relative, not hard limits — D4: no length constraint while drafting)

| § | Section | Weight | Job | Evidence |
|---|---|---|---|---|
| 1 | Introduction | **heavy** | Carries the story; B1-B7 (INTRO_PLAN.md) | — |
| 2 | Background and setting | light | Only what is needed: the two sparsifier families, modularity, the two algorithm families. No spectral theory up front. | DSpar/L-Spar defs |
| 3 | Honest evaluation (the instrument) | **medium-heavy** | The controls, each motivated by the specific failure it catches; the 45/63 measurement lives here | exp_P V2, exp_G, TERMINOLOGY.md |
| 4 | Below the boundary: where it does not help | medium-heavy | BOTH sparsifier families, four algorithms, both metrics; every apparent gain dissolved | exp_L, K, P, V, X + A-F |
| 5 | **The boundary** | **heavy — equal to §4** | The degree sweep, real Metis, the threshold, matched-retention controls, the noise-injection mechanism | exp_AA |
| 6 | Mechanism: why the boundary is there | medium | delta, its decomposition, the causal steering | exp_B, M, Q, N |
| 7 | Related work | light-medium | Where other people's papers live. Satuluri credited precisely; the contemporary instances of the missing control, with arithmetic | references/NOTES_*.md |
| 8 | Discussion | medium | The dissociation as an open phenomenon; what the boundary means for practice | exp_V, L, P |
| 9 | Conclusion | light | **Written last** | — |
| — | Abstract | — | **Written last** | — |

## The hard consequence of these weights

**Effort is not page count.** Twenty-two experiments of negative territory (A-K, L, M, N, O, P, R,
V, W, X) compress into ONE section plus a table. One experiment (AA) gets equal weight, because it
carries the novel finding. This will feel wrong while writing — months of work reduced to
paragraphs — and it is correct. A contribution paper spends its space on what is new, not on what
was laborious.

Specifically, most of the exploration phase becomes:
- a table of what was tested and what survived;
- three or four sentences per killed hypothesis (ours included — exp_O core preservation,
  exp_W leaf shedding, exp_M triangles, exp_X Youtube);
- the pre-registration record as a credibility asset, mentioned once, not narrated.

## Weight given to other people's papers

- **Satuluri et al. 2011** — the most weight of any external work, and it appears in three places:
  the introduction (as the scoped answer the field generalized), §5 (their threshold reproduces),
  and related work (precisely credited; the three charges we withdrew are simply never made).
  We restore their control; we do not claim it.
- **Chen et al. 2024** — related work; cited for fragmentation-at-scale, monotone fidelity
  degradation, and "match sparsifier to task." Our answer to the overlap: they measure *fidelity to
  the full-graph partition*; the field's claims and ours are about *quality* and *speed*, which
  they do not measure.
- **Laeuchli 2020** — related work; the boundary of our scope claim (dense planted-partition +
  superlinear spectral solver = the regime where sparsification genuinely pays), plus his
  independent statements of ER degeneracy and cost cancellation.
- **The two 2026 papers** — related work only, with arithmetic, never in the introduction (D1).
- **Hamann 2016, Blagus 2015, Gottesbüren 2025** — one-line citations in related work as prior art
  for pieces of the artifact framing.
