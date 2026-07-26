# The Complete Experiment Report
### Sparsification and Community Detection — every experiment, what it asked, what it found
*Compiled 2026-07-26, at the close of the exploration phase. Every number traces to an
exp_*/SUMMARY.md with CSV pointers. This document is the reading companion for the
synthesis phase.*

---

## How to read this

The project has one question: **should you sparsify a graph before detecting communities?**
The February draft said yes. The audit said the evidence was broken. Everything below is the
rebuilding: first the audit experiments (A–K) that tested the old claims, then the
exploration experiments (L–X) that tested every idea we could think of — including our own.

A recurring cast of characters:
- **DSpar** — degree-based sparsifier (keeps edges scoring high on 1/d_u + 1/d_v).
- **L-Spar** — similarity-based sparsifier (each node keeps its highest-Jaccard edges);
  the method behind the field's founding "sparsification helps" claim (Satuluri 2011).
- **Honest transfer scoring** — always score a partition on the ORIGINAL graph, never the
  sparse one.
- **The three artifacts** — I: scoring on the sparse graph; II: granularity (more clusters
  mechanically help some metrics); III: uncorrected/chance-inflated recovery metrics.
- **δ (delta)** — how much the sparsifier's score separates intra- from inter-community
  edges; the mechanism quantity.

---

## Phase 1 — The audit (Experiments A–K)

**Exp A — LFR benchmark forensics.** Asked whether the draft's synthetic-benchmark results
meant what they claimed. Found LFR graphs are degree-homogeneous and structurally uncoupled
in the way that matters for DSpar; hub-rewiring restores the coupling. The draft's synthetic
evidence had a provenance confound. *(Feeds C13.)*

**Exp B — Configuration-model nulls.** The pivotal control. Asked: does DSpar's
fixed-partition modularity gain require community structure at all? Answer: **no** —
degree-preserving randomized graphs reproduce the gain on 17/17 networks, usually *larger*
(median ratio 1.24). Bonus discovery: real structure *suppresses* δ relative to null on
16/17 — an open question that later spawned Exps M and Q. *(C7, C8.)*

**Exp C — Runtime-matched seeded detection.** Asked whether sparsify-then-seed-then-refine
ever beats plain Leiden at equal compute. Found exactly one robust case: **email-Enron**,
+0.008 modularity across 8/8 configurations. The project's one genuine positive. *(C9.)*

**Exp D — Ground-truth recovery at scale.** Recovery against real labels on the three large
labeled networks: sparsification improved recovery **0/3** times.

**Exp E — The predictor hunt.** Swept 15 networks for genuine runtime-matched gains (found
2/15: Enron, Youtube) and asked whether any structural statistic predicts them. **None
does.** Also caught a live example of correlation fragility: a promising r=0.69 collapsed
to 0.12 when one network was added — the draft's r=0.92 claim died the same way. *(C10.)*

**Exp F — Ground-truth recheck (small networks).** Re-tested the draft's recovery wins with
chance-corrected AMI and resolution-matched controls: **0/5 survive.** Also found the
draft's Dolphins "ground truth" was actually an algorithmic surrogate, not real labels.
*(C6.)*

**Exp G — Other sparsifiers + spectral forensics.** Artifact I (sparse-graph scoring) is
**sparsifier-agnostic** — uniform random deletion produces it 6/6 too. Also mapped the
draft's spectral parameter conversion (~100x off) and showed the two controls (null +
resolution matching) catch different failure modes. *(C14.)*

**Exp H — Youtube sweep.** The screening figure that looked like a big Youtube effect was
retired; a careful 8-cell sweep left one Bonferroni-surviving cell: +0.003 at calibrated
α=0.95. *(Later demoted by Exp X.)*

**Exp I — LCC-corrected tables.** Recomputed the paper's main tables on largest connected
components; verified the F/G mechanism decomposition quantitatively (identity to 1e-13,
direction 9/11). *(C4.)*

**Exp J — Figures.** Vector figures regenerated on corrected data; an LFR dataset mislabel
fixed; claims made exact.

**Exp K — The weighted regime.** DSpar's theory says keep importance weights. Does that
help detection? Preservation of the fixed objective: verified 24/24 cells — but it is an
**evaluation-correctness fact, not a performance win**: weighted pipelines transfer worse,
recovery is worse in every configuration, and there is no speed payoff (0.97–1.06x).
*(C2.)*

**Phase-1 bottom line:** every headline positive was an artifact; the mechanism theory
holds; one genuine gain (Enron) survives; weights are bookkeeping, not benefit.

---

## Phase 2 — The exploration (Experiments L–X)

**Exp L — The founding claim on trial (L-Spar).** The field's original "sparsification
improves clustering" claim (Satuluri 2011) had never been re-tested under controls — and
our experiments hadn't covered similarity-based methods either. Ran L-Spar through the full
protocol on 7 networks. Results: **the quality claim dies in 14/14 cells** (honest ΔQ
−0.014 to −0.40; naive sparse-graph scoring flips the sign in every single cell), and the
promised 10–50x speedup is actually 0.87–1.35x. **But**: L-Spar is the study's first
genuinely *structure-aware* sparsifier — its Jaccard signal collapses on randomized nulls,
where DSpar's degree signal never did. And one anomaly: on com-Amazon, L-Spar's recovery
gain (+0.062) survived a granularity control given MORE clusters than it had. *(C16.)*

**Exp M — Why does real structure suppress δ? (17/17 networks).** The Exp B open question.
An exact algebraic decomposition splits the suppression into sorting vs granularity terms:
**it is a sorting (partition–score misalignment) effect on 16/16 networks** — real
partitions simply don't align with the degree-driven score, while null partitions must.
The intuitive triangle/clustering explanation **died**: its correlation collapsed from
+0.74 to +0.19 when the six biggest networks were added (exactly as the jackknife had
predicted). Most suppressed networks (com-Amazon, cit-Patents) are the ones whose real
partitions are completely score-blind (AUC ≈ chance). One non-circular candidate survived:
hub_inter_lift → handed to Exp Q. *(C8.)*

**Exp N — Anatomy of the Enron gain.** Dissected the one genuine positive. It **survives an
adversarial granularity check three ways** (corr(k,Q) = −0.001; k-matched gap +0.006; the
seeded partition is *finer* where it matters). Mechanism identified: DSpar removes
community-boundary edges at **2.58x** the rate of internal ones — precisely the
hub-mediated shortcuts it is built to drop. The gain is diffuse (no few merges carry it),
there is **no separate basin** (a plain restart came within 0.0002), and compute claims are
honest only *in expectation*. What seeding buys: reliability (88th percentile in one shot,
half the variance) and price (~1/10 the compute of a matching restart search). *(C9.)*

**Exp O — Mohammad's core-preservation hypothesis.** Pre-registered test of "sparsification
shatters loose periphery but preserves community cores." **Failed 4 of 5 predictions.**
Where partitions genuinely shatter, cores shatter too; fragments are drawn from
*fully-embedded* nodes (direction reversed); and a resolution-matched partition of the
original graph reproduces the whole pattern, preserving cores *better* in 16/20 cells.
One real insight survived: the shards subdivide communities without mixing them (93–94%
plurality purity) — but that is a granularity fact, not periphery-shedding. Also exposed
that embeddedness is an anti-core statistic (leaves score 1.0). *(C6 footnote.)*

**Exp P — Is any of this Leiden-specific? (Algorithm generality).** The biggest referee
hole: everything was Leiden/modularity. Ran DSpar + L-Spar under Infomap, Louvain, and
label propagation — 63 cells on Fuji. **The negatives are universal: 60/63 cells negative**
against a fair restart baseline (Infomap and Louvain 21/21 each). Artifact I is
algorithm-general (sign flips in 45/63 cells). Label propagation's apparent "gains" are a
known pathology being relieved. The com-Amazon anomaly got scoped: it reproduces under
Louvain but **Infomap and label-prop beat it without any sparsification** — it is
granularity repair of the modularity family's coarse default, not a capability
sparsification adds. New live lead (unpredicted, unclaimed): Infomap recovery gain at mild
retention. *(C18.)*

**Exp Q — The causal test of δ.** The most elegant result. With degrees fixed, the
partition frozen, and the intra-edge fraction AND modularity held *exactly* constant,
steering hub-edge mass onto/off community boundaries **causally controls δ on 5/5
networks** — movable 20–450 standard deviations in either direction, past the
configuration null itself, and the effect survives re-detecting the partition from scratch
(AMI 0.82–0.94). Triangles were refuted causally too (δ driven from +0.09 to −0.87 while
triangle richness moved −2.8%). Honest limit: the lever is proven, but whether it is what
actually separates real from null partitions is NOT established — the attribution test
failed. C8: half-closed. *(C8.)*

**Exp R — Iteration-matched Enron control.** Closed exp_N's last soft spot: maybe the
seeded pipeline just beats *under-iterated* plain Leiden? No. Given the pipeline's exact
budget as deeper iterations, plain Leiden stays 0.0043–0.0048 below the seeded mean
(p<0.03); even run to full convergence (23 iterations, 4.4x the cost) it averages below.
Variance stays ~2.2x the seeded at every depth. Iteration-deepening also moves partitions
*away* from the seeded granularity — seeding is a different search direction, not more of
the same. *(C9.)*

**Exp V — Spending the signal.** If L-Spar genuinely sees communities (Exp L), can that
signal be deployed for an honest gain — by weighting edges instead of deleting, by seeding,
or by protected deletion? Result: **the modularity route is closed** — the one gain
(Enron, ~+0.010) is reproduced and *exceeded* by a shuffled-weight null (+0.0117), meaning
similarity-weighting's benefit is weight-heterogeneity perturbing the optimizer, not
community information. This retroactively indicts the edge-weighting literature's
evaluations; the shuffled-weight control is a new mandatory check that nobody runs.
Seeding is the only structure-dependent deployment, and it carries the study's sole
all-controls recovery survivor: com-Amazon, chance-corrected +0.036 at 4.0σ — with a
modularity LOSS in the same cell. The Exp L Amazon anomaly was formally promoted to a
result here (then scoped by Exp P). *(C17.)*

**Exp W — Leaf-shedding (the rescue attempt for Exp O).** Maybe the fragments are the
graph's low-degree leaves? Pre-registered, tested, **dead: 0/9 cells.** Fragments are
*mid-degree* nodes; hubs are never shed; on Enron fragment degrees exceed the graph median.
The beautiful inversion: the randomized null is MORE leaf-selective than the real graph in
8/8 cells — **real community structure holds low-degree nodes in place**; only when
structure is destroyed do the leaves shard off.

**Exp X — The Youtube gain on trial.** C9's second gain (+0.003, Bonferroni-surviving in
exp_H) got the full Enron treatment. It replicated bit-exactly and is statistically
bulletproof (p = 1.9e-5, zero of 20 restarts beat even the seeded worst) — **and it is
entirely a granularity artifact**: corr(k,Q) = −0.80, the seeded runs have fewer
communities than every plain restart, and the seeded mean lands exactly ON the plain k–Q
regression line (residual −0.0001). The Enron mechanism signature is present but causally
inert (28 edge deletions on 3M edges). The pre-registered kill criterion fired.
**C9 is now: one genuine gain, email-Enron, alone.** The Enron/Youtube contrast — same
statistics, opposite verdicts under the granularity check — is the paper's cleanest
demonstration of why the check matters.

---

## Where this leaves the paper

**The answer to the title question:** sparsification does not help community detection —
not for quality (60/63 cells, 4 algorithms, both families including the founding method),
not for speed (≤0.66x at quality-preserving retention), not via weights (Exp K), not via
weighting-instead-of-deleting (Exp V), and the belief that it does traces to three
measurable evaluation artifacts, now proven algorithm-general. The exceptions are exactly
two, both narrow and both instructive: Enron (a genuine, mechanism-backed search-reliability
win for seeding) and com-Amazon (a genuine recovery win that *costs* modularity and exists
only for the modularity family).

**The recurring theme nobody planned:** the objective and the truth dissociate. Three
independent times (Amazon/L-Spar, Amazon/seeding, Infomap/eu-core), the partitions that
recover reference communities best are NOT the ones that score best on modularity. That
observation was never a goal of this project; it emerged from the controls, and it may be
the most interesting question the exploration leaves open.

**Hypotheses that died honestly along the way** (all pre-registered, all killed by their
own criteria): the triangle explanation of δ-suppression (M, Q), core-preservation (O),
leaf-shedding (W), the Youtube gain (X), and my own prediction that the shuffled-weight
null would collapse the weighting gain (V — it did the opposite, which was the discovery).

**Still open:** Exp S (Orkut scale demo — awaiting Fuji disk space), the Exp T decision
(predictor at 100+ networks), and the synthesis conversation.
