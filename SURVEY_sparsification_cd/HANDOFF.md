# HANDOFF — the survey, clue-driven. Written 2026-07-28.

**Next session starts here. Mohammad will say "continue on survey".**

## The direction changed on 2026-07-28. Read this before anything else.

The paper is **not** being written right now. Mohammad's decision, and his reasoning, verbatim in
substance: *reproduction has no meaning without an idea or a gap. We did it before and we are
here with nothing in hand.* Reproduction verifies; it does not generate. Three rounds of auditing
narrowed the claim each time and never produced a new question.

So the order is now: **survey first, reproduction second.**

And the survey is **clue-driven, not comprehensive**. He rejected breadth-for-its-own-sake
explicitly: *"I will go clue by clue, not widening without meaning."* His model of a clue is the
Sotiropoulos and Tsourakakis framing line, quoted in the survey, which reveals that by 2021 the
field treated the pipeline as an established success in need of explanation rather than as a
proposition in need of testing.

**Do not restart a broad literature sweep.** Search saturated at ~15 on-subject works across
three sessions. Widening happens only when a clue demands it.

## Method, agreed and in use

Per clue, four steps:

1. Locate the source claim, verbatim, with a page number.
2. Citation-chase both directions. Who did the paper cite for it; who has cited it since.
3. Verdict: answered / partly answered / untouched. If answered, record who answered it.
4. If untouched: one paragraph stating the gap, and the experiment that would close it.

Stopping rule per clue: two rounds of chasing in both directions with nothing new.
Stopping rule overall: the survey is done when the clue list is worked through, gap or no gap.

**Verification marks are mandatory and are the point of the document.**
`[FULL]` read end to end · `[META]` fields off a fetched record · `[2ND]` described, not read,
not safe to characterize. Nothing from recall, ever. A `[2ND]` line must never be promoted
without checking: that mistake was made once in this project and is documented in the survey.

## Deliverable

`SURVEY_sparsification_cd/` — `survey.tex`, `refs.bib`, `survey.pdf`. Everything in one folder,
at Mohammad's request. Currently 8 pages, 34 references, 11 papers read in full.

The survey will be **restructured around clues rather than chronology**. Each clue becomes a
subsection ending in one of two sentences: "this was resolved by X" or "this remains open, and
the experiment is Y."

## THE CLUE LIST

Ten clues, extracted from the eleven papers read in full. Ranked as agreed.

### Clue 1 — CHASED, verdict UNTOUCHED. Ready to write up.

Satuluri et al.\ 2011, p.\ 729, verbatim:

> "on the Wiki dataset, Metis requires 80 seconds to cluster the L-Spar sparsified graph and 8
> seconds to cluster the G-Spar one, compared to 940 seconds for RandomEdge and 1040 seconds for
> ForestFire. This difference in clustering times, despite the fact that all the sparsified graphs
> contain almost the same number of edges, is because clustering algorithms generally tend to
> execute faster on graphs with clearer cluster structures."

A **130x runtime spread at constant edge count**, explained in one uncited sentence, never
revisited. Every speed claim in this literature attributes gains to reducing $|E|$.

*Chase result.* 100 citing papers examined by title (API limit; true count 169-206). Nobody picks
it up. Three near-misses, none connecting: Gottesbueren et al.\ ESA 2025 use sparsification inside
the multilevel scheme and report coarsening dynamics depending on which edges survive, but frame
it as engineering for linear time. PASCO (Lasalle et al., Machine Learning 2024/25) is the closest
framing, an overlay to speed up clustering by structure-preserving coarsening, but attributes the
speedup to size and never measures runtime at fixed size. von Luxburg's tutorial gives the
mechanism, eigensolver convergence scaling with the eigengap, but only for spectral methods, and
Satuluri's example is Metis, which is not spectral.

*Our own data, already computed, do not re-run.* At matched realized retention the effect exists
but is modest at $n = 10^4$: Metis 1.27x to 1.73x in 8 of 8 cells, Leiden 1.2x to 2.0x in 15 of 15,
**Graclus absent or inverted (5 of 8 cells below 1.0)**. So it is real, directional, and
detector-specific.

*Why this matters more after the chase.* It is the third measurement pointing the same way. Metis
gains quality from L-Spar, shows the density transition, and runs faster on L-Spar's graph.
Graclus does none of the three. The speed effect, the quality effect, and the threshold may all be
facts about **one algorithm**, which the field has read as facts about sparsification.

*Open sub-question, deliberately not run on Mohammad's instruction:* does the ratio grow with
graph size? Satuluri's 130x was on Wiki; ours is 1.3-1.7x on $10^4$ vertices where detection takes
0.1-0.7s, possibly too fast for the effect to show.

### Clue 2 — NOT STARTED. Next.

He, Drineas & Khanna 2025 (arXiv:2510.12669), \S5, verbatim:

> "uniform sampling actually performs slightly better than effective resistance sampling. We
> hypothesize that this is due to uniform sampling being biased towards undersampling cross
> cluster edges. We leave further investigation of this phenomena to future work."

An explicitly abandoned mechanistic thread, three months old. Why would a rule that treats every
edge identically preferentially spare intra-cluster edges? If the hypothesis is right there is an
elementary explanation nobody has written down.

### Clue 3 — NOT STARTED

Dreveton et al.\ 2024 versus the whole similarity-based line, a flat contradiction. The metric
backbone **keeps** inter-community bridges (their Figure 2, red edges) and preserves community
structure. L-Spar **removes** bridges and claims the same outcome. Two opposite operations, same
claimed result. At least one mechanism is misattributed.

### Clue 4 — NOT STARTED

Hamann et al.\ 2016, verbatim: *"the preserved community structure is not necessarily the same as
the one the Louvain algorithm finds."* Sparsification leads the algorithm to a **different**
structure, not a clearer one. Written in 2016, never followed up.

### Clue 5 — NOT STARTED

Satuluri's own scope limit: their \S4.5 says the method beats the unsparsified graph only from
average degree $\approx 50$, and they never return to it. The field dropped the condition its
founding paper stated.

### Clue 6 — NOT STARTED

Bravo-Hermsdorff & Gunderson 2019: deletion and contraction are the reciprocal limits of edge
weight 0 and $\infty$; sparsification and coarsening are one operation. Never applied to
community detection.

### Clue 7 — NOT STARTED

Blagus et al.\ 2015: sampling can **manufacture** community structure. Never used as a control by
anyone downstream.

### Clue 8 — NOT STARTED

Chen et al.\ 2024: the most rigorous benchmark in the field measures fidelity to the full graph's
partition, never quality, never runtime.

### Clue 9 — NOT STARTED

Hashemi et al.\ 2024 (IJCAI survey): **zero** occurrences of "community". The abandonment,
documented in the field's own survey.

### Clue 10 — NOT STARTED

Satuluri's minhash: approximate similarity at a 240x cost advantage over exact (4 hours vs 1
minute). Every modern reimplementation, including ours, uses exact Jaccard. The speed question may
have been settled against a strawman.

## State of everything else, so it is not relearned

**The claim and title were settled 2026-07-27** and are recorded in
`PAPER_RESTRUCTURE/phase4_v3/PAPER_STRUCTURE.md`, with both retired versions kept and what killed
each. Title: *Graph Sparsification for Community Detection: Repair, Not Improvement.* The boundary
was retired when Graclus showed the degree-50 transition is a property of Metis.

**New experiments this session, all committed, none written into the tex:**
`exp_AE_graclus_real_networks/` (Graclus on seven real networks, 170 cells),
`exp_AF_graclus_lfr_VERDICT.md` (Graclus on the LFR sweep: no transition),
`exp_AG_uniform_sampling_VERDICT.md` (1,080 new cells at low mixing; He et al.\ tested against
detection quality; the repair pattern confirmed out of sample). Note: exp_AG's verdict file
**oversells** result 1 and needs correcting. At strong clustering both L-Spar and uniform are
negative for Graclus, so "uniform nearly suffices" partly means "when nothing works, all the
nothings are similar." Flagged and not yet fixed.

**Tex is untouched since fix 6.** Section 1 still argues the retired claim. Do not write prose
into the paper without an explicit instruction.

**PR #2 was closed as stale** on 2026-07-28, not merged. Branch `refactor_v2` intact, 242 commits.
`main` is far behind and has no unique content.

**Fuji** works: `ssh mayooran@100.88.245.65`. Tailscale SSH prints a `login.tailscale.com` URL
that Mohammad must click; hold the connection open in the background while he does. Datasets are
all present (35 MB total). Disk is at 100% with ~4 GB free, which has not been a constraint.
Graclus and MLR-MCL both build there; binary at `bin/graclus`.

## Working relationship, and this matters

Mohammad's trust in my judgment is low, and with cause. Over this session I was confidently wrong
at least eight times, each time in the same register as when I was right. He said so directly and
he was correct.

Consequences for how to work:

- **Distinguish verified from reasoned, every time.** "I checked X in file Y" versus "I think".
  Uniform confidence is the failure mode, not the errors themselves.
- **Do not oversell a next step.** I told him one more experiment would make a claim unbreakable
  right after two experiments had each destroyed a headline. Do not do that again.
- **Point him at checks that do not require trusting me.** He verified the uniform-deletion
  control himself; that was worth more than any assurance.
- Bringing in a second model has twice caught things I missed and once caught an error I had just
  introduced. Do it for load-bearing decisions. He asked for Fable by name.
- He reads carefully and catches real problems. When he says something feels wrong, it usually is.
