# Adversarial reading notes: Chen et al. (PVLDB 2024) and Laeuchli (PAKDD 2020)

*Compiled 2026-07-26 for the novelty / prior-art check on "graph sparsification does not
help community detection under honest evaluation." Both papers read in full from primary
sources; Chen's released evaluation code also read. Every quotation below is verbatim from
the source indicated.*

---

# PAPER A — Chen et al., "Demystifying Graph Sparsification Algorithms in Graph Properties Preservation"

## A.1 Bibliography

- Yuhan Chen, Haojie Ye, Sanketh Vedula, Alex Bronstein, Ronald Dreslinski, Trevor Mudge,
  Nishil Talati. *Demystifying Graph Sparsification Algorithms in Graph Properties
  Preservation.* PVLDB 17(3): 427-440. doi:10.14778/3632093.3632106.
  (The paper's own reference block says "2023"; the volume is the 2024 VLDB cycle.)
- Michigan + Technion. Funded under IARPA AGILE.
- Artifact: https://github.com/yuhanchan/sparsification (branch `master`, not `main`).
- Local copy: references/chen_pvldb2024_demystifying.pdf
- Code read: `src/metrics_nk.py` (451 lines) from the artifact repo.

## A.2 What is actually in the paper

**Scale.** "16 widely-used graph metrics, 12 representative graph sparsification algorithms,
and 14 real-world input graphs"; "over 30,000 data points."

**Sparsifiers (12).** Random (RN); K-Neighbor (KN, Sadhanala 2016); Rank Degree (RD,
Voudigari 2016); Local Degree (LD, Hamann 2016); Spanning Forest (SF); t-Spanner (SP-3/5/7);
Forest Fire (FF, Leskovec 2006); L-Spar (LS, Satuluri 2011); G-Spar (GS, global Jaccard);
Local Similarity (LSim); SCAN (Xu 2007); Effective Resistance (ER, Spielman-Srivastava),
run in two variants **ER-weighted** and **ER-unweighted**.
**DSpar is not among them.** The nearest degree-family members are Local Degree and Rank
Degree; DSpar's 1/d_u + 1/d_v score does not appear.

**Metrics (16), grouped.**
- Basic: degree distribution (Bhattacharyya distance, 100 bins), Laplacian quadratic form
  (100 random x, mean ratio), plus connectivity diagnostics (pair-unreachable ratio, vertex-
  isolated ratio).
- Distance: SPSP stretch (100,000 sampled pairs), diameter (approx., 10 seeds), eccentricity.
- Centrality: betweenness (approx., 500 samples), closeness, eigenvector, Katz — all scored
  as **top-100 precision** vs the original graph's ranking.
- Clustering: **#communities**, **MCC**, **GCC**, **clustering F1 similarity**.
- Application: PageRank (top-100 precision), min-cut/max-flow (100,000 pairs, mean stretch),
  GNN (GraphSAGE on ogbn-proteins, ClusterGCN on Reddit).

**Datasets (14).** ego-Facebook (4,039 / 88k), ego-Twitter (81k / 1.77M), human_gene2
(14,340 / 9.04M), com-DBLP (317k / 1.05M), com-Amazon (335k / 926k), email-Enron (36,692 /
184k), ca-AstroPh (18,772 / 198k), ca-HepPh (12,008 / 118k), web-BerkStan (685k / 7.6M),
web-Google (876k / 5.1M), web-NotreDame (326k / 1.5M), web-Stanford (282k / 2.3M),
Reddit (233k / 57.3M), ogbn-proteins (133k / 39.6M).
Scale is comparable to ours (their max 57M edges vs our 25M); overlap with our networks
includes com-DBLP, com-Amazon, email-Enron.

**Protocol.** "We sweep the prune rate from 0.1 to 0.9, with a step of 0.1" (i.e. retention
0.9 down to 0.1). Non-deterministic sparsifiers: "we generate 10 graphs at each prune rate,
measure graph metrics using the mean value, and indicate their standard deviation." Hardware:
Xeon Platinum 8380, 1 TB RAM; A40 for GNN.

**Clustering protocol specifically.**
- #communities: "We employ the widely recognized Louvain method [12] for community detection,
  assuming the number of communities is unknown, and use the number detected in the original
  graph as the ground truth."
- Clustering F1: paper §2.2.4 prints
  `Precision = Sum_i max_j{a_ij} / Sum_i Sum_j a_ij`, `Recall = Sum_i max_j{a_ij} / n`, `F1 = 2PR/(P+R)`.
- The F1 figure includes a baseline line: "The green dashed line represents the clustering F1
  similarity when applying clustering algorithms twice on the original graph; it is not 1 due
  to the inherent randomness in the clustering algorithm."

## A.3 STEELMAN (with numbers)

1. **It is the broadest, most careful sparsification benchmark that exists.** 12 x 16 x 14,
   30,000+ data points, an open extensible framework, explicit anti-cherry-picking rules
   ("we always include Random as it serves as a naive sparsifier for comparison"), approximate
   algorithms validated against exact ones on small graphs, and 10-run averaging with reported
   standard deviations for stochastic sparsifiers. Nothing in our study matches its breadth.

2. **It independently establishes fragmentation-with-pruning at benchmark scale.** "As the
   prune rate increases, the graph becomes increasingly disconnected, and the number of
   communities consistently rises." Figure 8 (com-DBLP) spans 10^2 to 10^5 communities across
   the prune sweep. This is the *premise* of our Artifact II, published before us, on a
   network we also use.

3. **It independently establishes that clustering structure is not preserved by anything.**
   - MCC/GCC: "None of the sparsifiers demonstrate outstanding performance in preserving MCC
     and GCC, as they all degrade linearly with respect to the prune rate." / "Overall, no
     sparsifier proves effective in preserving clustering coefficients."
   - F1: "For all sparsifiers, F1 similarity decreases as the prune rate increases."
   Twelve sparsifiers, fourteen graphs, no exceptions.

4. **It has a genuine ordering result for clustering fidelity.** "K-Neighbor exhibits the best
   overall performance, while Local Similarity, Local Degree, and L-Spar also demonstrate
   strong results. These sparsifiers share a focus on local edges, and locally similar
   vertices more likely to belong to the same community." And: "Across graphs, K-Neighbor,
   Local Degree, Local Similarity, L-Spar, ER-unweighted, and ER-weighted consistently rank as
   top performers, while G-Spar and SCAN persistently underperform."

5. **It contains a self-undercutting result for the "structure-aware" school that we should
   quote.** The two sparsifiers explicitly built to keep community edges are the worst at
   almost everything, *including clustering fidelity*: "G-Spar and SCAN always under-perform
   because they both tend to keep intra-community edges. This leads to a more disconnected
   graph and a high unreachable/isolation ratio." And "In contrast, G-Spar and SCAN display
   poor performance in preserving clustering similarity."

6. **It is honest about sparsifier cost in at least one place.** "The computation of effective
   resistance takes 990 seconds for ogbn-proteins" and ER's "execution time ... is
   approximately an order of magnitude higher than that of other sparsifiers," with the
   caveat that ER can still pay off "if ... the sparsification overhead is less than the time
   that can be saved in performing the downstream task."

7. **Its conclusion is deliberately non-triumphalist.** "no single sparsifier excels in
   preserving all graph properties, and it is important to select appropriate sparsification
   algorithms based on the downstream task."

## A.4 AUDIT against our artifact list

| Control | Chen's status | Evidence |
|---|---|---|
| **Artifact I** — scoring on the sparse graph | **Not applicable in the paper; COMMITTED in the released code** | The paper never computes any graph-scored objective: "modularity" appears **0 times**, "NMI"/"mutual information" **0 times**. The F1 and #communities metrics compare *partitions*, with the reference taken on the **original** graph — the honest direction. **However**, the artifact's `src/metrics_nk.py:412-414` contains `C = nk.community.detectCommunities(Graph)` followed by `modularity = nk.community.Modularity().getQuality(C, Graph)` where `Graph` is the **sparsified** graph. Detected on sparse, scored on sparse. The function logs it. It is not reported in the paper, but any user of the framework who asks "does sparsification preserve modularity?" gets Artifact I by default. |
| **Artifact II** — granularity | **SILENT — documented, never controlled, and the metric definition is internally inconsistent** | They *observe* fragmentation (Fig. 8, 10^2->10^5) and never ask what it does to their similarity metric. Worse: the **printed formula** in §2.2.4 has `Sum_i Sum_j a_ij = n`, so Precision = Recall = F1 = `(1/n)*Sum_i max_j a_ij` — i.e. **purity of the sparse partition w.r.t. the reference**, which is monotone *non-decreasing* under refinement and -> 1 as the graph shatters. That is Artifact II in its purest published form. But the reported curves *decrease*, so the printed formula cannot be what was run. The code (`metrics_nk.py:356-364`) instead computes `nk.community.CoverF1Similarity(Graph, C, reference_C).run().getUnweightedAverage()` — NetworKit's per-cluster best-match F1, unweighted-averaged over clusters, which *penalises* fragmentation (a tiny fragment scores poorly against a large reference community). So the paper's clustering-similarity metric is **granularity-biased in an undisclosed direction, opposite to the direction the printed definition implies**, and there is **no resolution-matched control on the original graph** anywhere. |
| **Artifact III** — chance correction | **COMMITTED (uncorrected)** | No NMI, AMI, ARI or any chance-corrected index appears in the paper (grep: 0 hits). All clustering comparison is raw best-match F1. No chance floor is reported or discussed, despite community counts spanning three orders of magnitude — precisely the range over which our own floor moves 0.02 -> 0.09. |
| **Real ground truth vs algorithmic surrogate** | **COMMITTED (surrogate)** | The reference partition is Louvain-on-the-original ("use the number detected in the original graph as the ground truth"), and for F1 it is an **LFM overlapping cover** of the original. com-DBLP and com-Amazon ship SNAP ground-truth communities, and the artifact even contains an unused `ClusteringF1SimilarityWithGroundTruth_nk()` function (`metrics_nk.py:371`). No ground-truth recovery result is reported in the paper. This is the same class of error our Exp F caught in our own February draft (Dolphins). |
| **End-to-end cost / speedup** | **SILENT — asserted, never measured** | The motivation is explicit: sparsification "can be applied to greatly reduce the run time of graph algorithms," and the conclusion claims it can "optimize computational efficiency without significantly compromising output quality." **No downstream runtime is measured anywhere in the paper.** §4.6 "Sparsification Time" times *only the sparsifier*. Grep: "speedup"/"speed-up" 0 hits, "runtime" 0 hits, "end-to-end" 0 hits. The central efficiency premise of the field is restated but not tested. |
| **Fair stochastic baseline (best-of-N at matched wall clock)** | **SILENT (partial credit)** | They do plot the Louvain-run-twice-on-the-original line as an F1 ceiling and note "it is not 1 due to the inherent randomness in the clustering algorithm" — an acknowledgement of run-to-run variance that most papers omit. But there is no best-of-N, no restart budget, no matched-compute comparison. |
| **Null models (configuration model)** | **NOT PRESENT** | No randomised-graph control of any kind. Nothing tests whether an observed preservation effect requires community structure. |
| **Algorithm generality** | **PARTIAL / undisclosed mixture** | #communities uses NetworKit `detectCommunities` (PLM, a Louvain variant). Clustering F1 uses **LFM with LFMLocal seeds — an overlapping local-fitness method, not Louvain and not stated in the paper**. No Leiden, no Infomap, no label propagation. The two clustering metrics therefore use two different, partly undocumented algorithms. |
| **Statistical testing** | **NOT PRESENT** | Ten-run means with standard deviations shown for stochastic sparsifiers; no hypothesis tests, no multiple-comparison correction, no confidence intervals on differences between sparsifiers. |
| **Weighted vs unweighted transfer** | **CONTROLLED (credit)** | They correctly identify that ER is "the only one that modifies edge weights" and run **both** ER-weighted and ER-unweighted throughout. This is the same distinction our Exp K makes, and they make it cleanly. |

**Cannot determine:** (i) whether Fig. 10's F1 curves would change sign under a chance-corrected
index — the raw per-cell data is in the artifact but was not re-run here; (ii) the exact reason the
printed §2.2.4 formula differs from the implemented metric (transcription error vs. deliberate
simplification); (iii) whether LFM-vs-Louvain was a considered choice or an accident, since the
paper never names the algorithm used for F1.

## A.5 VERDICT on Chen

**(b) + (c), with no (a).**

There is **no valid result here that survives our controls as a rival positive claim**, for the
simple reason that Chen never makes a positive claim about community detection. Every
clustering result they report is a *degradation* result, and every one of them points the same
direction as ours. On the narrow question "does the sparse graph's partition resemble the full
graph's partition?", they answer *no, monotonically, for all 12 sparsifiers* — and they answer
it more broadly than we do.

**(b) Regimes we never tested — real and referee-relevant.** Chen's clustering-fidelity ranking
puts **K-Neighbor** first, with **Local Degree** and **Local Similarity** also "strong." We have
tested neither K-Neighbor nor Local Similarity nor Local Degree nor Rank Degree nor Forest Fire
nor Spanning Forest nor t-Spanner nor G-Spar nor SCAN. Our four-sparsifier scope (DSpar, L-Spar,
uniform random, light ER) covers **two** of Chen's six named top clustering preservers.

**(c) Methods deserving a fair test.** K-Neighbor and Local Degree are the two worth running
through the honest protocol, because they are (i) cheap — Chen shows KN has among the lowest
sparsification overhead, which is the only way the end-to-end cost arithmetic could ever work,
and (ii) Chen's own best fidelity performers. Both are in NetworKit; Chen's framework is open;
this is a cheap defensive experiment. Bounded risk: our Exp G showed Artifact I is
sparsifier-agnostic (uniform random reproduces it 6/6), and Exp P showed the negatives hold
across 4 algorithms — so the prior is strongly against a surprise. But "you tested 2 of the 6
best" is a referee sentence we do not want to receive without an answer.

**Gift to us.** Their released code computing modularity on the sparsified graph
(`metrics_nk.py:414`) is the cleanest possible evidence that Artifact I is not a straw man: it
is present in the tooling of the field's most rigorous benchmark. Cite it precisely, by file and
line, without mockery.

## A.6 The novelty question: how much of us does "match sparsifier to task" already cover?

Being unflattering to us first — **Chen owns, and we must cede:**
1. The fragmentation-with-pruning fact at benchmark scale (Fig. 8). Our Artifact II's premise
   is prior art. We contribute the *consequence*, not the phenomenon.
2. "No universal winner; choose by downstream task." We cannot claim any novelty for "it
   depends on the sparsifier."
3. Comprehensive coverage of the sparsifier space, and a public framework. Our breadth is
   narrower and we should say so.
4. The observation that no sparsifier preserves clustering coefficients.
5. Partial cost honesty for ER (990 s on ogbn-proteins).

**Chen does not touch, and we own:**
1. **Quality.** They never compute a graph-scored objective for community detection. The
   question "is the partition obtained from the sparse graph better, equal, or worse *when
   scored on the original graph*" — the question the field's positive claims are actually
   about — is never asked. Our Artifact I (sign flip in 45/63 cells) has no counterpart.
2. **Speed.** They never time the downstream task. The field's central efficiency premise is
   untested in the field's most rigorous benchmark. Our end-to-end 0.87-1.35x, and <=0.66x at
   quality-preserving retention, has no counterpart.
3. **Granularity control.** They document fragmentation and then use an uncontrolled
   best-match metric whose printed definition and implemented behaviour disagree about which
   direction fragmentation pushes it. Resolution-matched controls on the original graph are
   entirely ours.
4. **Chance correction.** Zero chance-corrected indices in the paper. Our floor analysis
   (0.02 -> 0.09 as k goes 225 -> 10,950) is entirely ours.
5. **Null models.** No randomised control of any kind. Our configuration-model result — that
   degree-preserving nulls reproduce the DSpar modularity gain on 17/17 networks, median ratio
   1.24 — has no counterpart and no precedent here.
6. **Ground-truth recovery.** Not reported, despite labelled datasets in hand and a written
   function for it. The reference is an algorithmic surrogate throughout.
7. **Baseline fairness.** One run vs best-of-N at matched wall clock. They gesture at
   run-to-run variance; they do not budget against it.
8. **Algorithm generality as a tested claim.** One-and-a-half algorithms (PLM for counts, LFM
   for F1). Our 63-cell, four-algorithm generality result is ours.
9. **DSpar.** Absent from their sparsifier set.
10. **Mechanism.** delta, its decomposition, and the causal steering experiment have no analogue.

**The precise relationship, stated for the paper.** Chen answers *"which sparsifier preserves
which graph property, and how fast does it degrade?"* We answer *"granting that some sparsifier
preserves enough, does the resulting pipeline actually buy you anything — under honest transfer
scoring, resolution-matched controls, chance-corrected recovery, fair stochastic baselines, and
end-to-end wall-clock accounting?"* Their conclusion, "match sparsifier to task," presupposes
that the fidelity metrics used to do the matching are trustworthy. Our three artifacts are the
demonstration that, for the community-detection task specifically, they are not — and Chen's own
metric definition/implementation mismatch on clustering F1 is an unintended illustration of the
point. **"Match sparsifier to task" is not a rival to our claim; it is the recommendation our
controls show is under-determined for this particular task.**

---

# PAPER B — Laeuchli, "Fast Community Detection with Graph Sparsification"

## B.1 Bibliography and access

- Jesse Laeuchli (Cyber Security Research and Innovation Centre, Deakin University, Geelong).
  *Fast Community Detection with Graph Sparsification.* In PAKDD 2020 — Advances in Knowledge
  Discovery and Data Mining, LNCS 12084, pp. 291-304. doi:10.1007/978-3-030-47426-3_23.
  PMCID PMC7206315. Single author.
- ResearchGate blocked; **open full text obtained** from PMC (open-access subset). Full prose
  recovered. **Limitation on this reading:** the numbered equations and both halves of Table 1
  are rendered as images in the PMC HTML and could not be extracted; the publisher PDF returned
  403/HTML from four endpoints. Quantities that live only in the equations or in Table 1 are
  reported below as unrecoverable, not guessed.

## B.2 What is actually in the paper

**Scope, stated by the author.** Two communities only: "in this paper we restrict our attention
to the case where the number of communities is fixed at two," justified by (i) more available
theory and (ii) recursive bisection being standard in HPC partitioning. Detection is spectral:
find the Fiedler vector of the (regularised, scaled) Laplacian and split by sign.

**Two error sources studied.** "(i) dropping edges using different sparsification strategies;
and (ii) inaccurately computing the eigenvectors."

**Contribution 1 — how much can you drop.** Using the Mossel-Neeman-Sly estimator for the SBM
parameters (a, b) from non-backtracking walk counts in O(n) time, predict the post-sparsification
(a', b') and hence whether the graph stays above the detectability threshold. Stated as: "our
contribution is to determine the level of sparsification that can take place while still
recovering communities." Expected recovery fraction is predicted by an erf(.) expression from
Nadakuditi-Newman.

**Contribution 2 — the effective-resistance negative result.** Citing von Luxburg, Radl and
Hein: for SBMs "the effective resistance of a given edge (i, j) in the graph tends toward
[2/d]. Since the degrees of the nodes in this model are O(n), the variation between effective
resistances will be small, and will in any case not reflect the community structure of the
graph. At this point our spectral sparsifier will be selecting edges essentially at random."
And explicitly: "While in some sense this is a drawback, since this result is telling us we may
as well sample randomly, our algorithm can still function, and we can save the cost of computing
the effective resistances." He notes von Luxburg's empirical finding that this degeneracy
"arises even for small communities of 1,000 vertices."

**Contribution 3 — scaled effective resistance.** Multiply effective resistance by the sum of
degrees: "the variance around two may be meaningful. Using these 'scaled' effective resistances
captures the community structure of a SBM." He derives the *average* scaled resistance inside
vs. between communities analytically, defines their ratio, and uses it to correct the predicted
(a', b'). **Crucially, his sampler inverts the standard rule:** "we modify the probability
density function to sample the edges that have a low effective resistance over those that have
a high resistance, since these are the edges that make up our community. This approach is
slightly different from the standard algorithm of Spielman and Srivastava, which seeks to sample
the highest resistance edges."

**Contribution 4 — early-stopping criterion.** A Chebyshev/semicircle-law argument for when the
Fiedler signal dominates the residual eigenvector contamination enough for correct labelling,
allowing power iteration to stop long before numerical convergence.

**Regularisation.** Saade et al. regularised scaled Laplacian, with a Sherman-Morrison trick so
the rank-one regulariser does not destroy sparsity.

**Experiments.**
- Synthetic: SBM(10000, 0.5, 0.3) and SBM(10000, 0.5, 0.2).
- Real: one graph — the political-blogs network of Saade et al., **1,222 nodes**, chosen because
  "the communities are difficult to recover ... and because the graph structure is not exactly
  captured by the SBM model."
- Recovery result: "using the Regularized Laplacian we can quickly recover almost all the nodes
  correctly, at around the sparsification level, predicted by [Eqs. 18 and 21]," with scaled ER
  "converging faster, and following the prediction of Eq. (21) more closely." Retention is
  described as "very small, of the order of [expression unrecoverable] of the original graph for
  the Scaled Effective Resistance method"; the design target is Theta(n log n) edges.
- Blogs result: "we are still able to recover the communities even after a significant amount of
  sparsification is applied, at the point that our criteria indicate we should be successful."
- Speed: "When using the off-the-shelf solver available in Matlab to find the desired
  eigenvector, with our best method we achieve essentially an order of magnitude speed-up."
  Table 1 numbers unrecoverable (image). He is explicit that the theoretically best case is not
  benchmarked: the Spielman-Teng near-linear solvers "are not available for use in production
  code, so we do not benchmark them here."
- Early stopping: "In all four cases all the community nodes were recovered, even though the
  sparsification was of the order of [unrecoverable]."

**The author's own cost caveat, in full.** "obtaining the scaled effective resistances using the
method of Spielman and Srivastava, requires us to solve a number of linear systems. If a nearly
linear time solver is available, this will take O(m) time, where m is the number of edges
**before** our dropping strategy. This will dominate the cost of the computation, and we will
not get significant speed-up from using power iteration... In this case it makes sense not to use
the scaled effective resistance."

**The author's own scope caveat.** "the model has certain intrinsic limits which prevent it from
modeling certain real-world networks well. We would like to provide a similar analysis for more
complex community models, in particular models which have a non-constant average degree. We
could then apply our model to a larger variety of real-world graphs."

## B.3 STEELMAN

1. **It is a theory paper that does the thing we keep asking for: it predicts, in advance, how
   much sparsification is survivable, from measurable parameters.** The pipeline
   (estimate a,b in O(n) -> predict a',b' post-sampling -> check against the detectability
   threshold -> predict recovery fraction via erf) is a genuine, non-circular, pre-registered-by-
   construction criterion. Our Exp E hunted for exactly such a predictor across 15 real networks
   and found none. Laeuchli has one — because he restricted to a model where it is derivable.

2. **He derives an analytic delta.** The ratio of mean scaled effective resistance inside vs.
   between communities, and its propagation into the post-sparsification (a', b'), is
   structurally the same quantity as our delta (score separation of intra- vs inter-community
   edges), obtained in closed form for one sparsifier on one model. This is the closest prior
   analogue to our mechanism quantity and we should acknowledge it as such.

3. **The evaluation is genuinely artifact-resistant.** k is fixed at 2 by construction; labels
   are planted and external; the metric is percent-of-nodes-correctly-classified with a constant
   50% chance floor; the comparison target is an analytic prediction, not a tuned baseline.
   Fragmentation cannot inflate anything. Sparse-graph self-scoring is impossible. This is a
   cleaner evaluation than most of the empirical literature, ours included in some respects.

4. **He publishes a negative result against his own tool family.** Standard effective-resistance
   sparsification carries no community signal on SBMs — "we may as well sample randomly" — and
   he reports it rather than burying it. And he shows random dropping *still works* in this
   regime, "even when we are dropping edges randomly."

5. **He publishes the cost-cancellation argument against his own headline.** Quoted in full
   above. A 2020 PAKDD author states, unprompted, that when the downstream solver is fast the
   sparsifier's own cost dominates and the speedup evaporates. This is our Artifact-adjacent
   cost finding, arrived at analytically, six years early.

6. **He does not overclaim.** Nowhere does he claim sparsification *improves* recovery. The
   claim throughout is *retention at lower cost*.

## B.4 AUDIT against our artifact list

| Control | Status | Evidence |
|---|---|---|
| **Artifact I** — scoring on the sparse graph | **NOT COMMITTED — structurally immune** | The evaluation target is planted SBM membership, external to both graphs. No graph-scored objective (no modularity, no conductance) is used at any point. Recovery is "percentage of the nodes we will recover" against ground truth. |
| **Artifact II** — granularity | **NOT COMMITTED — structurally immune** | k = 2, fixed, by construction: "we restrict our attention to the case where the number of communities is fixed at two." Detection is a sign split of the Fiedler vector. Fragmentation is not an available degree of freedom. |
| **Artifact III** — chance floor | **PARTIALLY CONTROLLED** | The floor is a constant 50% and does not move, so the drift that corrupts our recovery comparisons cannot occur. He does not state the floor explicitly, but the baseline he plots against is the *analytic* predicted recovery from the post-sparsification parameters, which is a stronger reference than a chance floor. Not chance-*corrected* in the AMI sense; does not need to be. |
| **Cost accounting** | **SILENT in the experiments, CONFESSED in the prose** | Table 1a times the eigensolver on sparsified vs unsparsified input. That excludes the resistance computation. He then states in §"A Comment on Complexity" that with a fast solver that computation "will dominate the cost ... and we will not get significant speed-up." No end-to-end number combining sparsifier + solver is reported. This is exactly our end-to-end gap — identified but not measured. |
| **Fair baseline / best-of-N** | **N/A but incomplete** | The pipeline is deterministic given the sample, so restart variance is not the issue. The missing comparison is the same early-stopping criterion applied to the **unsparsified** graph — i.e. how much of the "order of magnitude speed-up" comes from sparsification vs from stopping early. The two accelerations are never separated. |
| **Null model** | **NOT RUN, but substantively pre-empted** | He proves the standard-ER signal degenerates to random on SBMs. That is a stronger statement than a null experiment for that sparsifier. For scaled ER, no null is run — but its signal is derived analytically from (a, b), so it is community-dependent by construction. |
| **Real-network coverage** | **COMMITTED (severe)** | One real graph, n = 1,222. All quantitative results are on SBM(10000, 0.5, .). |
| **Density regime** | **COMMITTED (severe, and decisive)** | *Our derivation, not his:* SBM(10000, 0.5, 0.3) has ~12.5M intra + 7.5M inter ~ **20M edges, average degree ~4,000, density ~0.4**. His retention target of Theta(n log n) ~ 92k edges is **~0.5% retention**. A real social/collaboration network at n = 10,000 has average degree ~10 and ~50k edges — *fewer edges than his sparsified graph*. The redundancy headroom his entire result depends on does not exist in the graphs our paper is about. |
| **Algorithm generality** | **NOT ADDRESSED** | Spectral bisection only. |
| **Number of communities** | **NOT ADDRESSED** | Two, always. He flags recursive bisection as the practical extension but does not test it, and does not address the resolution/stopping question that recursion immediately raises. |

**Cannot determine:** the exact retention percentages and the exact speedup factors in Table 1
(images, publisher PDF inaccessible); whether the political-blogs recovery exceeded, matched, or
fell below unsparsified spectral clustering on the same graph (the figure caption describes only
that recovery succeeds "at the point that our criteria indicate").

## B.5 VERDICT on Laeuchli

**(b) — a regime we never tested, in which the result is valid and largely immune to our
artifacts, and which we must explicitly scope out rather than dismiss.**

It is **not (a)**: nothing here is a positive claim about our regime, and the paper never claims
an improvement in detection quality over the full graph. It is **not (d)**: the result is sound
and two of its passages are directly usable by us.

The regime is: **dense, near-regular, balanced, assortative, planted two-block graphs, detected
by an iterative eigensolver.** Three properties make sparsification pay there and not here:
1. **Redundancy headroom.** ~0.5% retention is available because average degree is ~4,000.
2. **Known, fixed, tiny k.** No granularity channel, and no resolution question.
3. **A superlinear, conditioning-sensitive downstream algorithm.** Eigensolver cost scales with
   nnz *and* with spectral conditioning; halving edges genuinely halves work and can improve
   conditioning.

**This third point is the thing to say loudly.** Every detector in our study — Leiden, Louvain,
Infomap, label propagation — is already near-linear and highly optimised. Sparsification can only
ever shave a near-linear constant, which is precisely why our end-to-end numbers land at
0.87-1.35x and why the sparsifier's own cost cancels the gain. Laeuchli's ~10x is real *because
his solver is not near-linear*. A referee who knows this literature will raise it, and the raise
is legitimate: **"graph sparsification does not help community detection" is a claim about
near-linear detectors on sparse real-world networks, and we should say so in the abstract rather
than be told so in review.** It costs us almost nothing — the near-linear modularity/flow family
is the family the field's practical claims (Satuluri, DSpar) are actually about — and it makes
the claim defensible instead of overreaching.

**Two passages to cite in our favour:**
- The ER degeneracy: an independent, published, theoretically-grounded statement that the
  canonical "principled" spectral sparsifier carries **no community signal** on planted-partition
  graphs — "we may as well sample randomly." This is prior-art support for our Exp G finding that
  Artifact I is sparsifier-agnostic and for our config-null results generally.
- The complexity comment: an independent, published statement of **cost cancellation** — the
  sparsifier's own O(m)-on-the-original cost dominating the savings. Our contribution becomes
  "we measured what Laeuchli predicted," which is a stronger position than claiming the insight.

**No new experiment is required.** Optional and cheap if a referee pushes: run our honest
protocol on a dense synthetic planted-partition graph (avg degree in the hundreds) to show the
crossover point at which sparsification starts paying — that would convert a scope concession
into a positive result about *when* it works.

---

# NOVELTY DELTA vs OUR WORK — consolidated

## What we must now cede as prior art
1. **Fragmentation under pruning, at benchmark scale, across 12 sparsifiers** — Chen, Fig. 8
   (com-DBLP, 10^2 -> 10^5 communities). Cite as the premise of Artifact II. We contribute the
   consequence for measurement, not the phenomenon.
2. **"No sparsifier preserves clustering coefficients; clustering fidelity degrades
   monotonically for all of them"** — Chen §4.4.
3. **"No universal winner; match the sparsifier to the downstream task"** — Chen, conclusion.
   We cannot claim novelty for any "it depends" framing.
4. **"Effective resistance carries no community signal on planted-partition graphs; one may as
   well sample randomly"** — Laeuchli, via von Luxburg et al.
5. **The cost-cancellation argument in principle** — Laeuchli, §"A Comment on Complexity."
6. **An analytic intra-vs-inter score separation (a closed-form delta) for one sparsifier on one
   model** — Laeuchli.

## What remains entirely ours
1. **Honest transfer scoring and its consequences.** Neither paper scores a partition against
   the original graph's objective. Chen computes no objective at all in the paper — and computes
   it **on the sparse graph** in its released code (`metrics_nk.py:414`), which is the artifact
   in the wild. Our 45/63 sign flips have no precedent.
2. **Resolution-matched controls.** Chen documents fragmentation and never controls for it, with
   a similarity metric whose printed definition (purity — rewards fragmentation) contradicts its
   implementation (NetworKit per-cluster best-match F1 — punishes it). No paper in this space
   asks what a resolution-matched partition of the *original* graph would achieve.
3. **Chance correction.** Zero chance-corrected indices in Chen. Our chance-floor analysis
   (0.02 -> 0.09 as k moves 225 -> 10,950) is ours.
4. **Configuration-model nulls.** Absent from both. Our 17/17 result — that degree-preserving
   randomisation reproduces the DSpar modularity gain, median ratio 1.24 — is unprecedented here.
5. **Measured end-to-end cost for community detection.** Chen asserts speedup and never times the
   downstream task; Laeuchli times only the eigensolver and confesses the omission. Our
   0.87-1.35x, and <=0.66x at quality-preserving retention across 4 algorithms, is the only
   measurement of the field's central premise that we have found.
6. **Fair stochastic baselines.** One run vs best-of-N at matched wall clock. Absent from both.
7. **Algorithm generality as a tested result.** Chen: PLM for counts, LFM for F1, undocumented.
   Laeuchli: spectral only. Our 63 cells x 4 algorithms is ours.
8. **Ground-truth recovery on real labelled networks.** Chen uses an algorithmic surrogate and
   leaves its own ground-truth function unused; Laeuchli uses planted labels on synthetic graphs
   plus one 1,222-node real graph.
9. **DSpar.** In neither paper.
10. **The mechanism programme** — delta decomposition, causal steering, the sorting-vs-granularity
    split, the shuffled-weight null. No analogue in either.

## Threats to us that these two papers surface
1. **[HIGH — act on it] Detector-complexity scope.** Our negative is about near-linear detectors.
   Laeuchli's positive is about a superlinear eigensolver on a dense graph. Scope the headline
   claim explicitly in the abstract: *sparse real-world networks, near-linear
   modularity/flow-family detectors.* Do not let a referee supply this qualification.
2. **[MEDIUM — cheap to close] Sparsifier coverage.** We tested 2 of Chen's 6 named top
   clustering preservers. K-Neighbor and Local Degree are the two to add — both cheap (KN has
   among the lowest sparsification overhead in Chen's Fig. 14, which is the only regime where the
   cost arithmetic could plausibly work), both in NetworKit, both with Chen's framework open.
   Prior is strongly against a surprise (Exp G: Artifact I is sparsifier-agnostic; Exp P:
   negatives across 4 algorithms), but the sentence "you tested 2 of the 6 best" should be
   pre-answered.
3. **[LOW] Rival-framing.** Someone may read Chen as already having concluded "sparsification
   degrades clustering; pick the least-bad one." The answer is one sentence: Chen measures
   *fidelity to the full-graph partition*; the field's claims — and ours — are about *quality*
   and *speed*, and Chen measures neither.

## Concrete actions
- **Cite Chen** for: fragmentation at scale; monotone clustering-fidelity degradation across 12
  sparsifiers; the "match sparsifier to task" framing (as the recommendation our controls show
  is under-determined for this task); G-Spar/SCAN — the explicitly community-preserving
  sparsifiers — underperforming even at preserving clustering.
- **Cite Chen's artifact, by file and line**, as evidence that Artifact I is live in the field's
  most rigorous benchmark's own tooling: `src/metrics_nk.py:412-414`,
  `nk.community.Modularity().getQuality(C, Graph)` with `Graph` the sparsified graph. Report it
  factually; it is a strong point and does not need editorialising.
- **Optionally note** Chen's printed-formula/implementation mismatch on clustering F1 as an
  illustration that granularity-sensitive similarity metrics are hard to get right even with
  care — the printed definition reduces to purity, which fragmentation drives toward 1.
- **Cite Laeuchli** for: the ER-degeneracy result ("we may as well sample randomly"); the
  cost-cancellation argument; and as the named boundary of our scope claim (dense planted-
  partition + spectral solver = the regime where sparsification genuinely pays).
- **Edit the abstract** to scope the claim to sparse real-world networks and near-linear
  detectors, with the spectral/dense regime named as the exception.
- **Consider** one dense-synthetic crossover experiment to convert that concession into a
  positive "here is when sparsification starts to pay" result.
