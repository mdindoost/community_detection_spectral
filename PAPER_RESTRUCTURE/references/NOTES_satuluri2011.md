# NOTES — Satuluri, Parthasarathy & Ruan (SIGMOD 2011), read in full and steelmanned

*Compiled 2026-07-26. Adversarial due-diligence pass on the paper our draft calls "the founding
sparsification-improves-clustering claim." Read from the primary PDFs, not from secondary
restatements. **Conclusion up front: we mischaracterized them on the single most important point,
and the paper must be re-worded before print.***

Local copies: `references/satuluri2011_lspar_sigmod.pdf` (12 pp., the SIGMOD paper),
`references/satuluri_thesis.pdf` (197 pp.; **Chapter 5 = the SIGMOD paper, essentially verbatim**,
plus thesis-only §5.2.6 on a randomized L-Spar variant for ensembles).

---

## 1. Verified / corrected table

| # | What we assert | Status | Evidence |
|---|---|---|---|
| 1 | Abstract quote accurate | **VERIFIED VERBATIM** | SIGMOD p. 721 |
| 2 | L-Spar = per-node top-d^e by Jaccard, union rule, minhash | **VERIFIED** (ceil-vs-floor unresolved, immaterial) | §3, Algorithm 2, Eqn 1 |
| 3 | Four clustering algorithms | **VERIFIED, names corrected** | Metis, **Metis+MQI**, MLR-MCL, Graclus (§4.3) |
| 4a | They score on the SPARSIFIED graph (Artifact I) | **REFUTED — they explicitly forbid it** | §4.3(i) + Table 2 caption |
| 4b | Datasets / metrics | **VERIFIED** | 7 networks; size-weighted best-match F-score (det->GT); avg conductance on original graph; cv balance |
| 5a | They don't control cluster count (Artifact II) | **REFUTED for 3 of 4 algorithms** | Table 1 `|C|`; k is an *input* to Metis/Graclus/Metis+MQI; fn. 4 |
| 5b | No chance-corrected metrics (Artifact III) | **UPHELD** | No AMI/ARI/NMI anywhere |
| 5c | They don't charge sparsification time | **REFUTED** | §4.4 + Table 2 caption |
| 6 | Retention floor behaves as we describe | **VERIFIED — but it is their documented design, not a defect** | §3 "at least one edge incident to each node" |
| 7 | Seeds / variance / significance | **NONE REPORTED** — single runs throughout | grep: no statistical claims |
| 8 | Their positives are recovery-flavoured, ours modularity | **VERIFIED AND DECISIVE** | 10W/5L on GT F-score vs 5W/5L/1T on honest-transfer conductance |

---

## 2. The verbatim abstract (SIGMOD p. 721)

> "In this paper we look at how to sparsify a graph i.e. how to reduce the edgeset while keeping
> the nodes intact, so as to enable faster graph clustering without sacrificing quality. The main
> idea behind our approach is to preferentially retain the edges that are likely to be part of the
> same cluster. We propose to rank edges using a simple similarity-based heuristic that we
> efficiently compute by comparing the minhash signatures of the nodes incident to the edge. For
> each node, we select the top few edges to be retained in the sparsified graph. Extensive
> empirical results on several real networks and using four state-of-the-art graph clustering and
> community discovery algorithms reveal that our proposed approach realizes excellent speedups
> (often in the range 10-50), with little or no deterioration in the quality of the resulting
> clusters. In fact, for at least two of the four clustering algorithms, our sparsification
> consistently enables higher clustering accuracies."

**Our quote is exact.** Note the last sentence is *precisely scoped and true of their table*: the
two algorithms are Metis (4/4 ground-truth wins) and Graclus (3/3). They do **not** claim it for
MLR-MCL (1W/3L) or Metis+MQI (2W/2L, wins ~ noise), and print those losses openly. This abstract
is careful, not oversold. We should say so.

---

## 3. ARTIFACT I — THEY EXPLICITLY AVOIDED IT (the correction that matters)

§4.3, verbatim:

> "There are two important points to note about the way we measure conductances: (i) For the
> clusters obtained from the sparsified graph, we cannot simply measure the conductance using the
> very same sparsified graph, since that would tell us nothing about how well the sparsified graph
> retained the cluster structure in the original graph. Therefore, we report the conductances of
> the clusters obtained from the sparsified graphs also using the structure of the original graph.
> (ii) The G-Spar, RandomEdge and ForestFire (but not L-Spar) often isolate a percentage of nodes
> in the graph in their sparsification. We do not include the contribution to the final average
> conductance arising from such singleton nodes. Note that this biases the comparison based on
> conductance against the results from the L-Spar and the original graph and in favor of the
> baselines."

Table 2 caption, verbatim: **"phi_avg is always calculated w.r.t. the original graph."**

**Every quality number they report is Artifact-I-free:** conductance is explicitly transferred to
the original graph; F-score is against *external* ground truth (the sparse graph plays no role);
cv is graph-independent. Point (ii) shows them *deliberately adopting a convention that
disadvantages their own method*.

**Consequence.** Our Exp L `dQ_naive`/`dQ_fixed` columns demonstrate a real pitfall — of the
literature that FOLLOWED them, not of them. The sentence "The apparent gain is entirely Artifact I"
in exp_L/SUMMARY.md V1 attributes to them an error they publicly warned against fifteen years
before us. It must go.

## 3.1 ARTIFACT II — controlled by construction, better than we control it

k is chosen once per network (Table 1) and used for **every arm**. Metis/Graclus/Metis+MQI take k
as *input*, so k is exactly matched. Footnote 4: only MLR-MCL's k "varied slightly."
Stronger: **Metis 4.0 partitions under a balance constraint**, so the flagship Wiki result
(F 12.34 -> 18.47) is controlled for granularity **and** size distribution — a tighter control than
our resolution-matched Leiden arm, which matches k but leaves sizes free.
*Residual, narrow:* cv is reported only for conductance datasets, so the three Graclus GT wins
(+2.27/+0.68/+0.37) retain small unquantified size-distribution exposure. That is the only place
Artifact II can still live, and it is not where their headline is.

## 3.2 ARTIFACT III — upheld, but narrower

No chance-corrected metric, no null model, no modularity. The *metric* half stands. But the
mechanism by which uncorrected metrics mislead in our study is granularity drift, and granularity
is pinned here. **A protocol gap, not a plausible explanation of their result.**

## 3.3 Cost accounting — honest, with disclosed failures

Table 2 caption: "Speedups are w.r.t. time for clustering the original graph and **take into
account both the sparsification as well as the clustering times** on the sparsified graph."
They report their own **slowdowns** (Metis+MQI Wiki 0.46x, Orkut 0.7x) with mechanism, and
disclose exact-Jaccard costing 240x minhash. **Our "they don't charge the Jaccard time" critique
is simply wrong and must be deleted.**

---

## 4. Complete Table 2 win/loss ledger (transcribed, p. 729)

**Metis: 6W/1L** — BioGrid F 17.78->19.71 (25x); DIP 20.04->21.58 (2x); Human 8.96->10.05 (5x);
Wiki 12.34->18.47 (52x); Orkut phi 0.85->0.76 (36x); Flickr phi 0.87->0.84 (3x);
Twitter phi 0.95->0.96 **L** (6x).

**MLR-MCL: 3W/3L/1T** — BioGrid 23.95->24.90 W; DIP 24.85->24.38 L; Human 10.55->10.43 L;
Wiki 20.22->19.30 L; Orkut phi tie (big balance win, cv 6.4->0.5); Flickr W; Twitter W.

**Metis+MQI: 3W (all ~noise) / 4L** — incl. **Wiki 0.46x slowdown**, **Orkut 0.7x slowdown**.

**Graclus: 3W/2L** (OOM on Wiki and Orkut, disclosed) — BioGrid +2.27; DIP +0.68; Human +0.37;
Flickr L; Twitter L.

### Ledger totals (26 cells)
- **Overall 15W / 10L / 1T.**
- **Ground-truth F-score (recovery): 10W / 5L.** Metis 4/4, Graclus 3/3, Metis+MQI 2/4 (~0.1),
  MLR-MCL 1/4.
- **Honest-transfer conductance (objective): 5W / 5L / 1T — a coin flip.**
- Speedups span **0.46x to 52x**; "often in the range 10-50" is honest as an "often".
- **Balance:** in every conductance cell L-Spar wins, its cv is equal or lower — the wins are not
  bought by imbalance.

### Other key numbers
- e-sweep Wiki/Metis: e=0.3 -> 7% of edges, F 17.73 (vs 12.34), **81x**.
- Minhash k-sweep: exact 18.89 vs k=30 18.47, at **240x** the cost (4 h vs 1 min).
- **Structure-clarity runtime effect:** Metis takes **80 s** on L-Spar-Wiki vs **940 s** on
  RandomEdge-Wiki **at equal edge count** — ~12x of their speedup is structure, not edge count.
- **LFR sweep (§4.5): L-Spar "actually outperforms the original clustering starting from
  degree 50."** At d_avg=50, mu=0.8: Metis+MQI F 26.95 -> **40.47**.
- Their own disclosed limitation (§5): L-Spar cannot handle bipartite/triangle-free structure
  without a bibliographic-coupling transform.

---

## 5. THE REGIME PROBLEM (the finding that changes our headline)

**Their stated threshold: d_avg >= 50. Our seven networks: d_avg 5.53-32.58. Every one below it.**

| network | d_avg | min_ret (e=0) | 2/d_avg |
|---|---|---|---|
| email-Eu-core | 32.58 | 0.0533 | 0.0614 |
| wiki-Vote | 28.51 | 0.0673 | 0.0701 |
| email-Enron | 10.73 | 0.1586 | 0.1864 |
| ca-CondMat | 8.55 | 0.1862 | 0.2340 |
| com-DBLP | 6.62 | 0.2441 | 0.3020 |
| ca-HepTh | 5.74 | 0.2760 | 0.3483 |
| com-Amazon | 5.53 | 0.3006 | 0.3617 |

min_ret ~ 1.69/d_avg — the arithmetic consequence of their stated design ("at least one edge
incident to each node"), visible in their own Table 1 (DIP, d_avg 6.4, ratio 0.53 at e=0.5).
**Their retention floor on Wiki is ~0.018 and on Twitter ~0.0015**, which is why e can reach 7%
and 4% retention there and never can on com-Amazon.

**And on their two low-degree datasets, their results are a wash — exactly like ours.** DIP
(d_avg 6.4) and Human (10.8): the eight algorithm x dataset F-deltas are +1.54, +1.09, -0.47,
-0.12, -0.16, +0.11, +0.68, +0.37. Small, mixed, no effect. Their strong wins are BioGrid (65),
Wiki (94), Orkut (76). **Where our regimes overlap, we agree with them.**

## 5.1 Why 10-50x vs our 0.87-1.35x — five documented factors, none an accounting dispute

1. **Algorithm class.** Their baselines: Metis on 53M-edge Wiki = 7,485 s; Metis+MQI = 35,511 s
   (9.9 h); MLR-MCL on 117M-edge Orkut = 21,079 s. Ours: Leiden on 1.05M-edge com-DBLP = 12.4 s.
   600-2,900x slower wall clock for 50-110x more edges, and superlinear. **You cannot get 50x off
   12 seconds of near-linear work.**
2. **Retention.** They ran 0.04-0.17 on the big graphs; we ran 0.5/0.2 with floors preventing 0.2
   on 5 of 7. Cost-proportional-to-m at 0.15 gives a 6.7x ceiling; at our realized 0.30-0.54 it is
   1.9-3.3x — and our `speedup_leiden_only` 0.92-2.59x sits inside that ceiling.
3. **Structure-clarity effect, and it REVERSES for us.** Their ~12x from "clearer structure
   converges faster" is *negative* for Leiden at aggressive retention (com-DBLP@0.2 and
   com-Amazon@0.2 are 0.96x/0.92x leiden-only — the shattered graph yields 72,736/57,788
   communities and the optimizer does more work). **This is a genuinely new observation: the sign
   of the structure-clarity runtime effect is algorithm-dependent — positive for cut-based
   partitioners, negative for modularity optimizers pushed past their natural resolution.**
4. **Minhash vs exact.** Theirs 240x cheaper. Our argument that free sparsification wouldn't rescue
   the result still holds (`T_leiden_sparse` alone exceeds `T_leiden_orig` in 2/14 cells).
5. **Accounting: identical.** Both charge sparsification time. **Delete this critique.**

---

## 6. What we have NOT tested (ordered by referee urgency)

1. **The average-degree regime.** THE big one. Need >= 2 networks with d_avg >= 50.
2. **Noise injection — the causal test of their actual mechanism (denoising).** Take a labelled
   graph, inject x% random edges, sparsify, measure GT recovery vs x. If L-Spar's gain is
   denoising, the gain should rise monotonically with injected noise and vanish at x=0. Cheap,
   decisive, and it directly interrogates our own com-Amazon anomaly.
3. **Their algorithms.** Metis and Graclus are the two the accuracy claim is about; we tested
   neither. Note the pattern in their ledger: **L-Spar helps the fixed-k balanced partitioners
   (Metis 4/4, Graclus 3/3) and does not help the free-granularity flow algorithm (MLR-MCL 1/4).**
   Leiden, Louvain and Infomap belong to the second group. That converts "we contradict them" into
   "we explain them."
4. Their metric pair (size-weighted det->GT F at fixed k; conductance transferred to original).
5. High mixing mu at high degree (their Metis+MQI 26.95 -> 40.47 at mu=0.8, d=50).
6. G-Spar (global Jaccard) — we implemented only the local half of their contribution.
7. Randomized L-Spar + ensembling (thesis §5.2.6) — directly relevant to our Exp V shuffled-weight
   null.
8. **Cluster-size balance (cv) as an outcome.** They claim L-Spar improves balance for all
   algorithms and their table supports it. We have never measured balance — a genuine benefit of
   sparsification our paper currently does not acknowledge exists.
9. Triangle-free/bipartite structure — they disclose this limitation themselves.

---

## 7. THE DISSOCIATION — their 2011 result is very likely the phenomenon we rediscovered

1. **Their metric split is our metric split.** External-reference recovery 10W/5L (7/7 for the two
   algorithms the abstract names); graph-internal objective 5W/5L/1T.
2. **Their stated mechanism only works in one direction.** §4.5 on BioGrid: high-throughput assays
   "detect many false positive interactions... all four clustering algorithms enjoy better
   clustering accuracies on the L-Spar sparsified graph... suggesting that the sparsification does
   remove many spurious interactions." That is a **denoising** claim. Denoising can improve
   agreement with an external reference; it cannot, in principle, improve an objective computed on
   the graph you just deleted edges from. **The dissociation is not an anomaly — it is what the
   denoising hypothesis predicts.**
3. **Our data reproduce their signature three times** (com-Amazon/L-Spar +0.062 recovery with
   -0.079 modularity; com-Amazon/seeding +0.036 chance-corrected with modularity loss;
   Infomap/email-Eu-core same direction).
4. **Same regime, same direction, same effect size.** Their DIP cell (ratio 0.53, d_avg 6.4) gives
   +1.54 Metis / +0.68 Graclus; our com-Amazon cell (retention 0.542, d_avg 5.53) is our one
   surviving recovery win.

**Caveats that stop this being proof:** their dissociation is *between-dataset* (conductance and
F-score are on different networks), ours is *within-cell* (stronger — say so, do not claim they
demonstrated it); conductance != modularity; their fixed-k cut partitioners never had the
granularity degree of freedom that produces ours, so the *mechanism* may differ even if the
*phenomenon* is the same.

---

## 8. Required corrections to our own files

**exp_L_lspar/SUMMARY.md** — V1 must be rescoped ("under a modularity objective and a
resolution-free optimizer, on graphs below the d_avg~50 threshold they themselves identify");
the "entirely Artifact I" sentence must carry the note that **Satuluri et al. do not commit this
error**; V4's "not reproduced under any accounting" must become "regime, not accounting";
the retention-floor finding must be presented as a scope fact, not a concealed flaw; C6 should
cite their own exact-vs-minhash measurement (18.89 vs 18.47 at 240x cost).

**STORY.md C16** — add: *"Artifact I is ours-and-the-field's, not theirs; their cluster counts are
matched by construction; their speedups charge sparsification time. Scope warning: none of our
networks reaches the d_avg ~ 50 threshold the authors state, and none of their four algorithms
appears anywhere in our study. C16 as written cannot be presented as a refutation of
Satuluri 2011."*

**EXPERIMENT_REPORT.md / EXPLORATION.md** — replace "had never been re-tested under controls" with
"had never been re-tested with a modularity optimizer, a cost-matched baseline, a chance-corrected
metric, or a configuration-model null (their own protocol already matched cluster counts and
scored conductance on the original graph)."

**HEADLINE.** "Graph sparsification does not help community detection" is **not currently supported
against Satuluri 2011**, because we never entered their regime. Two acceptable resolutions, pick
one before print:
(i) **run the high-degree / Metis-Graclus / noise-injection arm** and let the claim stand or fall;
(ii) **scope the headline explicitly** — "does not help *modularity-based* community detection on
*sparse, low-degree* graphs with *near-linear* optimizers" — and state that the founding paper's
claim is about a different objective, different algorithms, and denser graphs, and that in the
low-degree cells they report, their result already agrees with ours.
**Option (i) is much stronger and the experiment is small.**

---

## 9. The reframing this buys us

> The founding claim was about **agreement with external reference communities**, for **fixed-k,
> balance-constrained cut partitioners**, on **dense, noisy graphs**, at **hour-scale clustering
> cost**. Read on its own terms it was carefully controlled — it transferred scores to the original
> graph, matched cluster counts, and charged sparsification time — and its authors already
> identified the boundary of its validity (average degree ~50). What the fifteen years since got
> wrong was the *generalization*: the field converted a scoped denoising-plus-speed result into an
> unscoped "sparsify before you cluster" recommendation, and lost the controls on the way — most
> conspicuously the one Satuluri et al. stated first, that you must never score a partition on the
> graph you sparsified. Our contribution is (i) to show the generalization fails for
> modularity-family detection on sparse graphs with near-linear optimizers, (ii) to supply the
> controls the descendants dropped, and (iii) to recover, independently, the dissociation their
> results already hint at: sparsification can move partitions toward reference communities while
> moving them away from the objective.

That framing costs us the word "refuted" and buys a far more defensible paper — and it turns the
com-Amazon cell, currently filed as an embarrassing anomaly, into corroboration of a fifteen-year-
old result.

---

## 10. Unverified

- Ceil vs floor vs round in "top d^e edges" (paper prints no rounding operator; immaterial to
  Exp L since e is bisected to a retention target, but the footnote should be honest).
- Whether Adj(i) includes i in Eqn 1 (bounded effect on low-degree ranks; state our convention).
- Graclus 1.2's balance behaviour (determines residual exposure of its three GT wins).
- Whether the Wiki ground-truth curation preceded seeing results (unknowable; do not insinuate).
