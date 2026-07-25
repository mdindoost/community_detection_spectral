# Exp N — Anatomy of the email-Enron seeded-Leiden gain (n=33,696, m=180,811, LCC)

Design: 5 plain baselines (seeds 100-104), 5 DSpar-seeded pipelines (calibrated sampler,
alpha=0.90, true retention 0.8997; spar seeds 200-204 / leiden seeds 300-304), 20 extra plain
restarts (seeds 900-919). Best baseline = seed 101 (Q=0.609357, k=150); best seeded = seed 304
(Q=0.617876, k=141); headline dQ = +0.008519.

Files: run.py, run.log (complete, sections [1]-[8]), run2.log (earlier partial run, sections
[1]-[6] byte-identical to run.log; superseded, no discrepancy), results.json, 10 CSVs,
partitions.npz (40 arrays: memberships + kept-edge sets).

## Verdict (five parts)

**V1 — Granularity artifact: RULED OUT, decisively.** This is the strongest result of Exp N.
The seeded partition has fewer communities (k 156.2 +- 9.1 vs 181.8 +- 18.8) but it is *finer*
where it matters, not coarser: largest community 3,437 nodes (10.2% of n) vs 4,988 (14.8%) for
the best baseline and 5,149 (15.3%) for the best plain restart; 21 communities above 1% of n vs
18 / 16; top-5 communities hold 43.4% of nodes vs 47.8% / 54.7%. k falls because 43 tiny
baseline communities (median size 8, 1,265 nodes total) get absorbed, while the 15 largest
baseline communities (18,824 nodes, 55.9% of the graph) get *split*. Independent checks:
corr(k,Q) over the 25 plain restarts = -0.0009; plain restarts whose k lands inside the seeded
range [141,167] (n=7) average Q=0.608643, so the seeded mean beats *k-matched* plain restarts by
**+0.005991**; the single best plain restart has k=181, exactly the plain mean. The +0.008 is not
Artifact III.

**V2 — Statistical reality of the gain: HOLDS.** Seeded mean 0.614634 +- 0.002518 vs plain mean
0.608432 +- 0.004446 over 25 restarts: **+0.006202**, Mann-Whitney p=0.0036, 200k-permutation
p=0.0028, bootstrap 95% CI [+0.0034, +0.0089]. All 25 (baseline x seeded) pairs give dQ>0
(mean +0.0075, min +0.0015) against 3/10 for the baseline-vs-baseline control (mean -0.0015).
Seed variance is roughly halved (sd 0.0025 vs 0.0044).

**V3 — Localization: FAILS. The gain is diffuse, not concentrated in a few merges/splits.**
179 community-level events (112 matched pairs, 38 base-only, 29 seeded-only). Positive
contributions sum to +0.0824, negative to -0.0738; the net +0.0085 is an 18.3x churn residue.
The largest single event (+0.0373) is 4.4x the entire net gain. Ranking events by |dContrib|,
the running cumulative only settles within +-25% of the final dQ from rank 23 and within +-10%
from rank 29 (dq_cumulative.csv covers ranks 1-30). Region-level localization does work: the 97
stably matched pairs (Jaccard>0.5, 87 of them >0.9, covering 49.6% of nodes) contribute
**-0.0076**, i.e. nothing; the whole +0.0085 comes from the 15 low-Jaccard giant-community pairs
(+0.0227) minus the 38 dissolved small communities (-0.0096) plus 29 new ones (+0.0031).
Half the partition is untouched; the gain lives entirely in the reorganized core.

**V4 — Landscape: NO separate basin. Seeding is a better/cheaper restart, not access to
unreachable structure.** One of the 20 extra plain restarts (seed 917, k=181) reached
Q=0.617689, only **0.000186** below the best seeded partition, and its AMI to that partition is
0.7513, which sits inside the ordinary plain-plain AMI range (extra-extra mean 0.7016, max
0.8280; base-base mean 0.6805). Seeded-seeded AMI is the highest group (0.7294), consistent with
variance reduction rather than a distinct optimum. The compute statement must therefore be
phrased in expectation, not as impossibility (see Caveat C2).

**V5 — Mechanism: IDENTIFIED and quantified. DSpar thins hub-mediated boundary edges.**
Pooled over the 13 baseline communities that the seeded partition splits, DSpar removed
**16.9%** of cross-piece edges (2,187/12,924) against **6.5%** of intra-piece edges
(3,621/55,311), a **2.58x** bias against a global 10% removal rate; for the 16 seeded
communities that merge baseline communities the ratio is 2.27x (13.6% vs 6.0%). Those boundary
edges are exactly the ones DSpar is built to drop: their endpoint degree product is 2.1x to 3.1x
(median) the intra-piece value, and the ratio exceeds 1 in **29/29** parents (max 16.6).
Sparsification weakens the boundary; it never severs it (0/29 parents fragment in G_sparse, all
stay 1 component, as in G), and in 13/29 parents no cross-piece edge was removed at all, so
those splits are restart noise rather than sparsification.

## Numbered findings

1. **Compute control, verified.** One plain restart 1.236s; one seeded pipeline 2.289s (1.85x).
   20 restarts = 24.71s = **10.80x** one pipeline (results.json `granularity`). Bootstrap over
   the 25 empirical plain Q values: E[best-of-20] = 0.616322, below the best seeded 0.617876 but
   above the seeded mean 0.614634. At matched wall clock the pipeline wins: 2 restarts (2.5s,
   1.1x one pipeline) give E[best]=0.610940 vs seeded mean 0.614634 (**+0.0037** for seeded);
   9 restarts (11.4s, the cost of all 5 pipelines) give E[best]=0.614973 vs seeded best 0.617876
   (**+0.0029** for seeded). 3/25 plain restarts beat the seeded mean, 0/25 beat the seeded best,
   8/25 beat the seeded *worst* (0.610886, plain percentile 68). The seeded mean sits at the
   **88th percentile** of the plain restart distribution.
   [partition_quality.csv, landscape_ami.csv, results.json `landscape`]

2. **Where the gain is created: on the sparse graph, before refinement.** The partition found on
   G_sparse, scored on the original G with no refinement, already averages Q=0.611916 (4/5 runs
   above the plain mean; 76th percentile of the plain distribution). Refinement on G adds only
   +0.002718, i.e. **44%** of the total advantage over the plain mean; 56% is inherited from the
   sparse graph. k grows during refinement (132->141, 142->152, 152->158, 154->167, 158->163),
   so refinement re-splits rather than coarsens.
   [partition_quality.csv `Q_sparse_transfer`, partitions.npz `sparsepart_*`]

3. **Merge/split structure.** Of 150 baseline communities: 89 identical (33.3% of nodes),
   43 merged whole into a larger seeded community (3.8% of nodes), 15 split (55.9% of nodes),
   3 reshuffled. 11 seeded communities absorb >=2 whole baseline communities; 13 baseline
   communities break into >=2 pieces each >=10%. Node-mass-wise the dominant event is splitting
   of the giants, which is why k falls while the giant shrinks.
   [contingency_merge_split.csv, run.log section 3]

4. **Who moves: nobody in particular. No degree fingerprint.** 10,363 nodes (30.75%) change
   community, but baseline-vs-baseline pairs move 38.4% on average, so the seeded partition is
   *less* different from a baseline than two baselines are from each other. Hub enrichment among
   moved nodes = **0.963** (i.e. none): hub move rate 0.2962 vs non-hub 0.3077; mean degree of
   moved 10.06 vs unmoved 11.03. Move rate by degree decile is flat with a mild anti-hub tilt
   (0.351 at degree 2 down to 0.282 in the top decile [19,1383]). Hubs stay put; the low-degree
   nodes whose route into the wrong community ran through a hub-mediated shortcut are the ones
   that get reassigned once that shortcut is dropped.
   [moved_nodes_degree.csv, moved_by_degree_decile.csv, pairwise_stability.csv]

5. **Fingerprint of the top-gain communities: no distinctive structure.** The top-gain seeded
   communities are large and high-conductance (0.25-0.42) versus the all-community medians
   (baseline 0.158, seeded 0.175); their transitivity (0.02-0.33) and mean degree (4.5-16.3)
   straddle the medians (0.199/0.140 and 6.95/7.19). Nothing separates them except size and
   boundary hub-bridge ratio (3.4 to 46 versus medians 17.3/15.1). This is a null result: the
   gain is not attached to a recognizable community type.
   [topgain_fingerprint.csv]

6. **The community-level dQ decomposition is bookkeeping-limited.** 9 of the top-10 |dContrib|
   events have Jaccard < 0.5 (e.g. base community 11, 994 nodes, Hungarian-matched to seeded
   community 0, 3,437 nodes, overlap 835, "+0.0373"), so the largest entries measure how the
   matching aligned two reorganized giants, not a localized gain. Concentration statistics
   computed on that table (top1_abs_share=4.38, gini=0.924) should not be quoted as evidence of
   localization.
   [community_diff.csv]

## Adversarial granularity check (verdict: the +0.008 survives)

Three independent ways of killing it were tried and all failed:
(i) regression, corr(k,Q) = -0.0009 across 25 plain restarts, OLS slope -2.1e-7 per community,
predicted Q at k=156.2 is 0.608438 versus 0.614634 actual, residual **+0.006197**;
(ii) k-matching, plain restarts with k in [141,167] average 0.608643, gap **+0.005991**;
(iii) shape, the seeded partitions are more balanced, not coarser (smaller giant, more
communities above 1% of n, less mass in the top 5, tiny communities absorbed and giants split).
The only pro-artifact signal is corr(k,Q) = -0.69 *within* the 5 seeded runs, which is n=5,
uncorrected, and points the wrong way for an artifact story anyway (the low-k seeded runs are the
ones that split the giants hardest).

## Caveats

- **C1.** Single network, single alpha (0.90), single sampler (calibrated), n_iterations=2,
  5 seeded runs. Exp N does not re-establish the gain, it dissects the one Exp C/E/K established.
- **C2.** The tempting sentence "20 restarts cost 10.8x and still cannot reach the seeded
  solution" is only true in expectation. One of the 20 restarts landed within 0.000186 of the
  best seeded partition. Correct phrasing: *in expectation*, best-of-20 plain restarts (10.8x the
  compute) still falls short of the best seeded pipeline, and no plain restart reaches it
  reliably; but the seeded solution is not out of reach for plain Leiden.
- **C3.** The bootstrap best-of-n curve resamples from the 25 observed plain Q values, so
  P(best-of-n >= seeded best) is structurally 0 for every n. That column is an artifact of the
  empirical maximum and is not reported here.
- **C4.** The compute control is wall-clock only. There is no n_iterations-matched plain control
  (plain Leiden with n_iterations=4 costing the same as the pipeline). The matched-wall-clock
  bootstrap in Finding 1 partly covers this, but an iteration-matched arm would be cleaner.
- **C5.** The 5 baselines (seeds 100-104, mean 0.607097) were an unlucky draw relative to the 20
  extras (mean 0.608766). The "25/25 pairs positive" figure in V2 leans on that; the honest
  effect size is the seeded-vs-all-25-plain +0.006202 with p=0.003.
- **C6.** The split test compares G_sparse only for the single sparsification seed that produced
  the best seeded partition (spar seed 204).

## Paragraph for the paper's discussion section

On email-Enron, the one network where sparsify-then-detect produces a reproducible modularity
gain, the gain is a search effect with an identifiable structural cause, not the discovery of
community structure that plain Leiden cannot see. DSpar's keep-probability, proportional to
1/d_u + 1/d_v, deletes precisely the high-degree-product edges that bridge distinct groups: over
the baseline communities that the seeded partition splits, 16.9% of cross-piece edges are removed
against 6.5% of intra-piece edges, a 2.58x bias against a global 10% removal rate, and the
cross-piece edges have 2.1x to 3.1x the endpoint degree product of intra-piece edges in all 29
communities examined. With those hub-mediated shortcuts thinned, Leiden on the sparse graph
separates groups that it fuses on the full graph; transferred back to the original graph without
any refinement, that partition already scores at the 76th percentile of the plain-restart
distribution, and refinement supplies only the remaining 44% of the advantage. The result is not
a granularity artifact: the seeded partitions have fewer communities (156 versus 182) but a
smaller largest community (10.2% versus 14.8% of nodes) and more communities above 1% of the
graph, because tiny communities are absorbed while the giants are split; modularity is
uncorrelated with community count across 25 plain restarts (r = -0.001), and the seeded runs beat
count-matched plain restarts by +0.0060. Nor is it a new basin: one of 20 plain restarts came
within 0.0002 of the best seeded partition at an AMI of 0.75, well inside the ordinary
restart-to-restart spread. What seeding buys is reliability and price. It places a single
2.3-second pipeline at the 88th percentile of the plain-restart distribution and halves the seed
variance, whereas reaching the same expected quality by brute force takes roughly ten times the
compute. Finally, the gain is not localized: 179 community-level events cancel to leave a net
+0.0085 out of +0.082 of positive and -0.074 of negative contribution, and the half of the graph
whose communities are stably identified across the two partitions contributes nothing at all.
