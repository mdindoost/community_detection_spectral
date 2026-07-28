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

**Updated 2026-07-27, after exp_AE (Graclus on seven real networks) and exp_AF (Graclus on the
LFR sweep). Both previous versions are kept below, because what falsified each is instructive
and because this is now the second time a headline has been retired by adding one arm.**

**TITLE, settled 2026-07-27:**
**"Graph Sparsification for Community Detection: Repair, Not Improvement."**
Head-noun first, because that is how this venue's readers scan, and the colon form survives
abbreviation in reference lists. `INTRO_PLAN.md` and `main.tex` still carry the old title and
should be changed when §1 is rewritten, not before, or the compiled PDF will advertise a claim
its introduction does not argue.

> **Sparsification moves a detector toward the unsparsified frontier and never across it: its
> real gains appear only where a detector falls short of the best result any detector we test
> achieves on that graph and measure without sparsification, and every such gain is exceeded by
> simply running that better detector. What density governs is the damage term, not the gain:
> the cost of deletion falls as a graph's benign redundancy rises, so for a detector at the
> frontier sparsification approaches a no-op on dense graphs, and the shrinking damage is what
> lets a weaker detector's repair show. The remaining reported gains are the accounting artifact
> of scoring on the sparsified graph and the granularity artifact of comparing partitions of
> different resolution, and there is no end-to-end speed benefit in the implementations we
> measure.**

*Three sentences, not one. The one-sentence rule existed to enforce one CLAIM that every
section must support, and the version it replaced was already three claims joined by
semicolons. The rule is kept in function: sentence 1 alone is the quotable headline, each
sentence must name the tables that defend it, and three is a ceiling.*

**Scoping that must not be lost when this is paraphrased.**
- "The best result any detector we test achieves" means across the **whole pool**, not within
  the fixed-$k$ family. Within the family it has counterexamples in both directions: sparsified
  Metis at LFR $d=50$ beats untouched Graclus on $Q$ by up to $+0.038$, and sparsified Graclus
  on email-Eu-core at $k=42$ edges untouched Metis by $+0.002$. Against the full pool it is
  clean everywhere checked.
- It is also **not** "never exceeds the same detector's clean baseline". That is false:
  sparsified noisy-graph Metis at $d=50$ with 100\% injected noise reaches AMI $0.624$ against
  its own clean-graph baseline of $0.596$. Only the cross-detector frontier claim holds.
- "Benign redundancy" presumes the §2 split between benign redundancy, which makes deletion
  cheap, and spurious edges, whose presence damages the baseline so that removing them is
  repair. Without that split the paper's own arm C reads as a counterexample to the density
  sentence.

**What each clause rests on.** Frontier and repair: exp_AE and exp_AF, plus the baseline tables
in both. Domination: wiki-Vote, best sparsified Metis over all arms and seeds $0.4006$ against
untouched Graclus $0.4164$; LFR $d=49$, sparsified Metis AMI $0.75$ against untouched Graclus
$0.8238$; com-Amazon, sparsified $0.402$ against unsparsified Infomap $0.465$ and label
propagation $0.480$. Damage falling with density: uniform random is the clean probe
(Metis $-0.172 \to -0.027$, Graclus $-0.104 \to -0.035$ on $Q$; on recovery the absolute
sparsified level rises monotonically for all four repair-free pairs and damage as a fraction of
baseline falls monotonically, which is the form to print, because the raw $\Delta$AMI is
non-monotone only through the baseline rising). Accounting: 45 of 63. Granularity: com-DBLP,
apparent $+0.091$ against a matched control of $0.300$. Speed: fastest quality-preserving
configuration $0.66\times$.

*Amended 2026-07-26 (late): "no benefit of any kind" was a universal with two disclosed
exceptions, which is a rhetorical liability for a two-word saving. Both reviewers flagged it. The
three-part form is what the data supports: 0/102 matched Leiden cells and 90/90 fixed-k cells on
sparse real graphs give no objective gain; the fastest quality-preserving configuration is 0.66x;
and the recovery gains that do survive granularity and chance (com-Amazon under Leiden, exp_AB's
Local Degree) are beaten by an unsparsified run of a different detector.*

Every section either supports that sentence or explains why nobody had found it. Anything that
does neither is cut.

*Superseded version 2 (2026-07-26 evening to 2026-07-27), and why:*

> ~~Sparsification's effect on community detection is decided by the detector's degrees of
> freedom and by the graph's removable redundancy rather than by the sparsifier; outside those
> two conditions there is no gain on the objective, no end-to-end speed benefit, and no gain in
> recovery that survives comparison with an unsparsified run of a better detector, and the
> accounting in common use reports one anyway.~~

**Killed by adding one detector.** Graclus takes $k$ as an input and, unlike Metis, imposes no
balance constraint, so it isolates the condition from the confound. On the LFR sweep that
produced Table VIII it shows **no transition**: $\Delta Q > 0$ in 3/12, 8/12, 6/12, 7/12, 6/12
by degree against Metis's 0/12, 0/12, 8/12, 10/12, 12/12, and it is positive in the mean by
$d=24.6$. On recovery it never gains at any degree. On the seven real networks the two
detectors disagree about which network gains, and each gains where it is the weaker baseline:
Graclus on the five sparse networks where it trails Metis by 0.016 to 0.066, Metis on wiki-Vote
where it trails Graclus by 0.034, and neither on email-Eu-core at $k=8$ where they sit within
0.011. A protocol check confirmed this is not the seed convention: recomputing Metis against
the mean of its seeds rather than the best leaves it at 0 of 128, since its seed spread buys
only 0.0004 to 0.0044 against a 0.05 effect.

So the degree-50 threshold was the density at which **Metis** falls decisively behind what a
fixed-$k$ detector can reach on these graphs. The Metis/Graclus AMI gap opens from 0.008 at
$d=11$ to 0.228 at $d=49$ and stays there, which is exactly where the "boundary" sat. Density
did not stop being a variable; it stopped governing the gain and now governs the damage.

*The second condition was inferred from a single detector, and the first thing that tested it
falsified it. That is the same failure mode as superseded version 1 below: a pattern true of
one arm, promoted to a condition.*

*Superseded version 1, and why:*

> ~~Sparsification's effect on community detection is determined by average degree and algorithm
> family, not by the sparsifier, and the threshold sits near average degree 50.~~

Three of our own results undercut it. **"The threshold sits near average degree 50" is an LFR
statement**: exp_AB found the fixed-k gain on real graphs at d_avg 28.5, and exp_AC found the SBM
family crossing no later than LFR, so the paper now states a condition and not a threshold.
**"Determined by average degree"** overstates what we can support: exp_AA's noise injection shows
the gain growing at fixed density when spurious edges are added, so the operative quantity is how
much redundancy can be removed, of which average degree is the strongest correlate we can measure
(Spearman +0.65 in exp_AC) and degree heterogeneity is not a correlate at all (-0.26, tested and
excluded). **The old sentence also carried only the positive half.** The measured cost of the
missing control (45 of 63 sign flips) and the delimited negative territory are what change what
other people do, and a headline that omits them sells the weaker half of the finding first.

*Corollary settled at the same time:* **the boundary leads; the yardstick is the instrument that
found it.** A contribution paper's headline is its finding, not its method — even when the method
is itself a contribution.

---

## Section weights

**STALE as of 2026-07-26 (night). The table below describes the nine-section paper that was
replaced. The current shape is six sections: introduction, background and related work,
evaluation protocol (argument only), experiments (setup, accounting, negative, rival
explanations, exception, boundary, mechanism), discussion, conclusion. See
phase4_v3/HANDOFF.md for the current state. The framing and the one-sentence claim above are
current; this table is kept only as a record of what the weights were when they were set.**

## Section weights, as set on 2026-07-26 morning (superseded)

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
