# Exp M — Mechanism probe for delta-suppression by real community structure

Files: run.py, run_bigchunk.py, probe_results.csv (33 rows = the original 11 networks x
{real, null seed 7, null seed 8}, 35-field header, no auc_s), probe_results_batch2_small.csv
(30 rows = 10 of those 11 re-run with the current run.py, 37-field header WITH auc_s/auc_prod),
probe_results_batch2.csv (21 rows = ca-GrQc + the 6 large networks, 37-field header),
staged_results.csv (24 rows = 3 networks x 2 rewire seeds x 4 stages), batchA.log, batchC.log,
batch2_*.log, fuji_driver.log, staged.log.

MERGE RULE for analysis: concatenate the three probe CSVs and de-duplicate on
(network, arm, rewire_seed) preferring batch2 > batch2_small > probe_results. The overlap is
numerically identical (30 rows, 0 columns differing by more than 1e-6 relative), verified
2026-07-25. This yields all 17 networks WITH auc_s. Never append new rows to the 35-field
probe_results.csv.

COVERAGE: 17/17 networks (the original run covered 11; com-Amazon, com-DBLP and com-Youtube were
added locally, and wiki-Talk, cit-Patents and wiki-topcats on Fuji, 2026-07-25).

Question inherited from Exp B / claim C8: degree-preserving rewiring leaves the DSpar score
multiset `s(e)=1/d_u+1/d_v` EXACTLY unchanged, yet delta = mu_intra - mu_inter is larger on the
rewired null than on the real graph for 16/17 networks (median ratio 1.24). The difference can
therefore only come from WHICH edges each graph's Leiden partition calls intra. Exp M asks which
structural property of the real partition does that. Pre-registered hypotheses: H1 degree sorting,
H2 triangles/clustering ("triangle-rich intra edges get protected"), H3 granularity artefact.

## Verdict (four parts)

**V1 — Mechanism localized, decisively: suppression is a SORTING effect, not a granularity
artefact.** The exact identity `delta = r_pb * sd_s / sqrt(p(1-p))` decomposes
`log(delta_null/delta_real)` into sorting + scale + granularity terms (max residual 0.0017 across
all 17 networks, probe CSVs `term_sum` vs `log_ratio`). The sorting term `log(r_null/r_real)` is
POSITIVE on 16/16 networks where it is defined (median +0.574, range +0.079 to +3.424); the
granularity term is NEGATIVE on 14/16 (median -0.140) and the scale term is negative on 10/16.
`r_pb(null) > r_pb(real)` holds 17/17, including on ca-CondMat, the one network whose net delta is
NOT suppressed. So H3 is rejected: the coarser real partition (p_intra median 0.799 vs null 0.358)
actually INFLATES delta_real and partially masks the suppression; the whole effect and more sits in
partition-score alignment. The class-balance-free Mann-Whitney AUC now available for all 17
networks corroborates this independently: corr(auc_null - auc_real, log ratio) = Pearson +0.745
(p=0.001), Spearman +0.841 (p<0.001), n=16.

**V2 — The triangle/clustering hypothesis (H2) is NOT supported cross-network, and the heavy tail
killed the two correlations that once looked promising.** At full 17-network coverage, suppression
strength vs global transitivity C: Spearman -0.015 (p=0.96, n=16), Pearson +0.126 (p=0.64). Vs mean
local clustering: Spearman -0.335 — still the WRONG sign. Vs mean triangles per intra edge: Pearson
+0.187 (p=0.49), Spearman -0.224. The two headline numbers from the 11-network run, tri_intra
Pearson +0.741 and C Pearson +0.533, collapsed to +0.187 and +0.126 once the six large networks were
added — exactly as the ca-HepPh jackknife had predicted (+0.059 and +0.104). The absolute
triangle/clustering measures are dead.

ONE EXCEPTION, logged as HYPOTHESIS, not result: the intra/inter triangle RATIO
tri_intra/tri_inter is Pearson +0.654 (p=0.006, jackknife [+0.409, +0.840]) and Spearman +0.497
(p=0.050, jackknife [+0.407, +0.564]), and survives leave-TWO-out ([+0.364, +0.866] /
[+0.292, +0.644]). It is the first triangle-flavoured variable in this dataset to survive a
jackknife. It is almost certainly a partition-coarseness proxy rather than triangle evidence:
Spearman +0.71 with p_intra, +0.75 with Q, -0.83 with hb; partialling out p_intra collapses it to
Pearson +0.305 / Spearman +0.415 and partialling out auc_s to +0.363 / +0.229, while auc_s SURVIVES
partialling out the ratio (-0.415 / -0.491). wiki-Talk is a direct counterexample: its intra edges
are 3.6x LESS triangle-rich than its inter edges (ratio 0.277) and it is still suppressed (+0.482).
Needs its own pre-registered test (Exp Q) before it is anything.

**V3 — The staged causal probe cannot test H2: the design is saturated at its smallest dose.** At
stage 0.25 (2.5 double-edge swaps per edge) 99.2-100.0% of the total clustering loss AND
99.6-100.2% of the total modularity loss have already happened, simultaneously. There is no
dose-response curve; there are three replicates of a binary real-vs-randomized contrast. delta is
also not monotone in the swap count (non-monotone in 4/6 network-seed paths). Triangle loss and
community-structure loss are collinear at rho ~ 1 by construction here, so the delta rise cannot be
attributed to triangle destruction specifically.

**V4 — The best-supported NON-CIRCULAR correlate is degree sorting of the REAL partition
(H1-flavoured), and it is descriptive, not causal.** At n=16 the strongest raw correlate is auc_s of
the real graph (Spearman -0.656, p=0.006, jackknife [-0.732, -0.582]; Pearson -0.674, p=0.004,
jackknife [-0.769, -0.631]), but auc_s and r_pb(real) are Spearman +0.92 with each other and both
are near-CIRCULAR with the target: a small r_real mechanically inflates log(delta_null/delta_real).
The one survivor that is not a restatement of the target is hub_inter_lift: Spearman -0.635
(p=0.008, n=16; jackknife [-0.718, -0.579], leave-two-out [-0.798, -0.503]), Pearson -0.483
(p=0.058). This is materially WEAKER than the -0.830 measured on the 11-network subset — the heavy
tail cost it about 0.2 of Spearman — but it holds sign, significance and a jackknife interval that
excludes 0. hb = E[d_u d_v|inter]/E[d_u d_v|intra] likewise softens from -0.709 to Spearman -0.524
(p=0.037, jackknife [-0.636, -0.439]). Partial correlations still keep hubLift alive controlling for
C (Spearman -0.824) and kill C controlling for hubLift (+0.356).

Correction to the 11-network write-up: hub_inter_lift is NOT "partly a re-expression of r_real".
At n=17 Spearman(hub_inter_lift, r_pb_real) = +0.06 — they are essentially uncorrelated. That makes
hubLift the only candidate mechanism variable in this probe that is neither circular with the target
nor a coarseness proxy. It remains one variable selected post hoc out of ~21 with no multiplicity
correction, and no causal arm tests it.

**Bottom line for C8: the triangle/clustering mechanism is NOT the answer (no), but the question is
now narrower — the suppression is provably a partition-score misalignment effect, not a granularity
artefact, on the complete 17-network set including the entire heavy tail.**

## Numbered findings

1. **Exp M reproduces Exp B exactly on all 17 networks.** delta_real matches
   exp_B_config_null/results.csv to 0.0 relative error on 17/17; delta_null matches to under 1% on
   17/17 (cit-Patents 0.4%, wiki-topcats 1.0% — Exp B averaged a different number of rewire seeds
   there). Suppressed on 16/17; the single exception is ca-CondMat (0.1126 real vs 0.1018 null,
   ratio 0.904) — the same exception as in Exp B. Median ratio 1.599, identical to Exp B's.

2. **Exact decomposition (probe CSVs, columns `term_sorting`, `term_scale`, `term_granularity`,
   `term_sum`, `log_ratio`).** Per-network sorting term, descending: com-Amazon +3.424,
   cit-Patents +1.816, ca-HepPh +1.762, ca-GrQc +1.350, cit-HepPh +1.163, com-Youtube +1.061,
   com-DBLP +0.647, wiki-Vote +0.641, email-Enron +0.507, ca-HepTh +0.466, cit-HepTh +0.396,
   ca-AstroPh +0.369, wiki-Talk +0.299, wiki-topcats +0.196, ca-CondMat +0.115,
   email-Eu-core +0.079. facebook-combined is undefined because r_real = -0.0042 flips sign to
   r_null = +0.131 — the most extreme case of the same phenomenon (delta_real = -0.0014,
   delta_null = +0.0170).

3. **The two arms differ enormously in partition-score alignment while sharing the score
   multiset.** r_pb: real median 0.1382 (range -0.004 to 0.587), null median 0.2817 (range 0.131
   to 0.792). p_intra: real median 0.799 (0.585-0.961), null median 0.358 (0.246-0.492). sd_s is
   near-invariant on 16 of 17 networks (null/real 0.865-1.051), confirming the score distribution
   is essentially untouched; the exception is wiki-Talk at 1.268, which gives it the largest scale
   term in the set (+0.237). Degree-preserving rewiring fixes the degree SEQUENCE but not the
   PAIRING, and s(e) = 1/d_u + 1/d_v depends on the pairing — on a graph as degree-skewed as
   wiki-Talk that is a visible effect, not a rounding one.

4. **Clustering is destroyed by rewiring but does not predict suppression.** C real median 0.205,
   null median 0.010 (null/real median 0.088) — the rewiring works. Yet across 16 networks C ranks
   independently of suppression (Spearman -0.015). Both correlations that looked alive at n=11 were
   single-point artefacts and the six added networks confirmed it: tri_intra Pearson +0.741 -> +0.187
   and C Pearson +0.533 -> +0.126, versus the +0.059 and +0.104 that the drop-ca-HepPh jackknife had
   forecast. This is the clearest vindication of the jackknife rule in the project so far.

5. **The "triangles protect intra edges" premise is itself only weakly true, and at scale it breaks
   outright.** tri_intra > tri_inter on 13/17 real graphs; on email-Enron (11.32 vs 13.97),
   wiki-Vote (17.36 vs 19.75), email-Eu-core (19.50 vs 19.97) and wiki-Talk (3.14 vs 11.37) intra
   edges are LESS triangle-rich than inter edges. The earlier statement that the fraction of edges
   with >=1 triangle exceeds 0.88 on both sides for nearly every network is FALSE at full coverage —
   it holds on 9/17, and the large sparse graphs break it badly: com-Amazon 0.802 intra / 0.251
   inter, cit-Patents 0.506 / 0.268, com-Youtube 0.479 / 0.413, wiki-Talk 0.231 / 0.614.

6. **Staged probe (staged_results.csv), per-network, means over 2 rewire seeds:**
   - ca-GrQc: delta 0.0872 -> 0.1941 -> 0.1927 -> 0.1985 (+127.5% overall); C 0.6289 -> 0.0107 ->
     0.0103 -> 0.0101; Q 0.8514 -> 0.3765 -> 0.3754 -> 0.3748; tri_intra 11.73 -> 0.16 -> 0.14.
     96.1% of the total delta change is realized by stage 0.25.
   - ca-HepTh: delta 0.1736 -> 0.2095 -> 0.2111 -> 0.2122 (+22.2%); C 0.2811 -> 0.0030; Q 0.7619
     -> 0.4126. 93.1% of the delta change by stage 0.25.
   - email-Eu-core: delta 0.0340 -> 0.0430 -> 0.0408 -> 0.0385 (+13.3% net); C 0.2674 -> 0.1437 ->
     0.1428 -> 0.1427; Q 0.4159 -> 0.1261. delta OVERSHOOTS at stage 0.25 (199.4% of the eventual
     change) and then decays back.
   Stage 0 rows reproduce the probe real arm exactly; stage 1.0 rows land within ~5% of the probe
   null arm (different rewire-seed offsets).

7. **No dose-response even across the three staged networks.** email-Eu-core loses only 46.6% of
   its clustering (dense, n=986 — residual clustering is forced by density) and gains +13.3%
   delta; ca-GrQc loses 98.4% and gains +127.5%; but ca-HepTh loses 98.9% — indistinguishable from
   ca-GrQc — and gains only +22.2%. Amount of triangle destruction does not predict amount of
   delta rise.

8. **Within the randomized regime (stages 0.25/0.5/1.0 only) the residual C-delta relation flips
   sign**: Pearson +0.365 (ca-GrQc), +0.034 (ca-HepTh), +0.815 (email-Eu-core), i.e. more
   clustering goes with MORE delta, opposite to H2. The ranges are tiny (C varies in the 4th
   decimal) so this is noise-level, but it is certainly not support for H2.

9. **Degree-sorting correlates (V4) in full.** Against log(delta_null/delta_real), n=16:
   auc_s(real) Spearman -0.656 / Pearson -0.674; hub_inter_lift Spearman -0.635 / Pearson -0.483;
   r_pb(real) Spearman -0.571; hb Spearman -0.524 / Pearson -0.289; p_intra(real) Spearman +0.588 /
   Pearson +0.645; Q(real) Spearman +0.365; deg_R2 Spearman -0.106 (nothing); assort_deg -0.012
   (nothing). Excluding ca-CondMat (the one unsuppressed network) hubLift holds at Spearman -0.579.
   Note real hub_inter_lift (median 1.194) still EXCEEDS null (median 1.080): real partitions push
   hub edges between communities more than configuration-model partitions do relative to their own
   base rate, and the more they do so the less suppression there is.

10. **auc_s exists for all 17 networks and is the cross-check V1 was missing.** The class-balance-free
    Mann-Whitney AUC of the DSpar score against the intra-edge indicator gives a sorting comparison
    that does NOT depend on the intra/inter balance, unlike r_pb (r = delta*sqrt(p(1-p))/sd_s is
    itself balance-sensitive, so the "sorting"/"granularity" split is one algebraic partition, not
    two independent physical quantities). Two readings, both reported. The bare sign test is weak:
    auc_null > auc_real on only 11/17 (binomial p=0.33), against 16/16 for the r-based sorting term.
    The magnitude agreement is strong: corr(auc_null - auc_real, log ratio) = Pearson +0.745
    (p=0.001), Spearman +0.841 (p<0.001), n=16, and corr(auc gap, term_sorting) = Pearson +0.747 /
    Spearman +0.785. The six networks where auc_null < auc_real are exactly the weakly or
    un-suppressed ones (ca-CondMat, email-Eu-core, wiki-topcats, ca-AstroPh, cit-HepTh, wiki-Vote).
    So the balance-free measure reproduces V1's ORDERING but not a unanimous sign flip.

11. **The heavy tail is the score-blind tail.** The two most suppressed networks are the two whose
    real Leiden partition carries essentially NO information about the DSpar score: com-Amazon
    (auc_s = 0.4748, i.e. marginally below chance; r_pb = 0.0109; ratio 14.45) and cit-Patents
    (auc_s = 0.4994, dead on chance; r_pb = 0.0585; ratio 4.37), against null auc_s of 0.689 and
    0.686. com-Amazon's whole log-ratio of 2.671 is sorting (+3.424) minus granularity (-0.785) —
    the largest single sorting term in the study, and the largest granularity mask. This is the
    mechanism at its most legible: on a graph whose communities are essentially degree-independent
    (product catalogue co-purchase), Leiden's partition and the degree-driven DSpar score are
    orthogonal, while the configuration null's partition is forced to track degree.

## Caveats

- **Coverage is now complete: 17/17 networks.** The six that were missing (com-Amazon, com-DBLP,
  com-Youtube, wiki-Talk, cit-Patents, wiki-topcats) carried the heavy tail of the phenomenon — Exp B
  null/real ratios 14.45, 1.34, 2.24, 1.62, 4.39, 1.25 — and all six are now measured, reproducing
  Exp B to within 1%. Conclusions no longer rest on the mild half of the suppression distribution.
  wiki-Talk, cit-Patents and wiki-topcats were run on Fuji (62GB); the local 14GB machine OOM-killed
  them.
- **Correlation fragility.** n=16. Every headline correlation is reported with a leave-one-out
  jackknife because this project has been burned before (r=0.69 -> 0.12 on one deletion). The added
  six networks were themselves a jackknife on a grand scale, and they killed two of the previous
  run's live correlations (tri_intra +0.741 -> +0.187, C +0.533 -> +0.126) and weakened the headline
  one (hub_inter_lift -0.830 -> -0.635). What survives with a jackknife interval excluding 0 is
  auc_s(real), hub_inter_lift, p_intra(real), r_pb(real), hb, and the triangle ratio — but auc_s and
  r_pb are near-circular with the target and the triangle ratio is a coarseness proxy, leaving
  hub_inter_lift as the only non-circular, non-proxy survivor. It is still one variable selected post
  hoc out of ~21 candidates with no multiplicity correction. Treat V4 as a hypothesis for Exp Q, not
  a result.
- **The confound in the causal arm is total.** Rewiring destroys triangles and community structure
  in the same swaps. Because 99-100% of both the C loss and the Q loss occur at the smallest dose
  tested, the design has zero leverage to separate them: there is no configuration in this data
  where clustering is destroyed and modularity preserved, or vice versa. The staged probe
  therefore confirms that randomization raises delta (which Exp B already established) and adds
  nothing about WHY. A real separation needs a rewiring that targets triangles while holding the
  partition fixed (e.g. triangle-preserving vs triangle-breaking double-edge swaps at matched Q),
  or the reverse.
- **Staged coverage is 3 small networks** (ca-GrQc, ca-HepTh, email-Eu-core), 2 rewire seeds, one
  Leiden seed (42). Leiden is re-run at every stage, so the partition and the graph both change
  between stages — an additional source of the non-monotonicity in finding 6.
- **Two script versions produced these CSVs, and one bug was found.** probe_results.csv was written
  by an earlier run.py without auc_s/auc_prod (35 fields). The current run.py was verified to
  reproduce it exactly (30 rows / 10 networks, 0 columns differing by more than 1e-6 relative;
  probe_results_batch2_small.csv), so the merged 17-network table is homogeneous and auc_s is
  available throughout. Separately, wiki-topcats SEGFAULTS under run.py as committed: edge_triangles
  uses a fixed chunk of 20000 edges, and wiki-topcats has a degree-238,342 hub, so one chunk
  materialises a submatrix of up to 4.77e9 stored values, past scipy's int32 index limit. The fix is
  run_bigchunk.py, a wrapper that monkey-patches an nnz-budgeted chunker; batching is pure
  scheduling, so outputs are bit-identical (verified on ca-GrQc, wiki-Vote, email-Enron: 9 rows,
  0 columns differing by more than 1e-9). run.py itself is untouched. ANY future experiment that
  runs this triangle code on wiki-topcats or com-Orkut (Exp S) must use the wrapper.
- **facebook-combined has delta_real < 0**, so its log-ratio and sorting term are undefined and it
  is excluded from all n=10 correlations. It is the most suppressed network in the set, so its
  exclusion is not conservative.
- Modularity-only Leiden (`ModularityVertexPartition`, n_iterations=2, seed 42), undirected simple
  LCC, single Leiden seed per graph. Resolution-limit effects on p_intra are not probed.

## Effect on claim C8

C8 currently reads: *"Real structure suppresses delta vs null (16/17) — open question."*
This probe does NOT close it, and it does NOT license a triangle-based explanation. It does
narrow it, now on all 17 networks. Recommended replacement:

> **C8** Real structure suppresses delta vs null (16/17, median ratio 1.60). Mechanism localized to
> partition-score misalignment, not partition granularity (Exp M, 17/17 networks, exact
> decomposition plus a class-balance-free AUC cross-check); triangle/clustering explanation
> unsupported. Root cause still open.

Defensible sentence-level claim for the paper (safe as written):

> The suppression is not an artefact of the real partition being coarser. Writing
> `delta = r * sd(s) / sqrt(p(1-p))`, where `r` is the point-biserial correlation between the DSpar
> score and the intra-edge indicator, decomposes `log(delta_null/delta_real)` exactly into a
> sorting term `log(r_null/r_real)` and a granularity term in `p`. On all sixteen networks where it
> is defined the sorting term is positive (median +0.57), while the granularity term is negative on
> fourteen of sixteen (median -0.14): the coarser real partition inflates delta_real and partially
> masks the suppression. A class-balance-free Mann-Whitney AUC, which by construction cannot respond
> to the intra/inter balance at all, tracks the same ordering (Spearman +0.84 between the real-to-null
> AUC gap and the suppression, n=16). Real Leiden partitions simply align less with the degree-driven
> DSpar score than configuration-model partitions do, even though the two graphs share the score
> multiset exactly; in the extreme cases (com-Amazon, cit-Patents) the real partition's alignment is
> statistically indistinguishable from chance. Which structural property produces that misalignment
> we could not determine: global clustering does not predict suppression strength across networks
> (Spearman -0.02, n=16, p=0.96), and progressively rewiring three networks destroys clustering and
> modularity in the same swaps, so it cannot separate them.

Do NOT write, in any form: "real structure suppresses delta because triangle-rich intra-community
edges are protected." The data do not support it: at 11 networks the supporting correlation died on
one deletion, and at 17 networks it died outright (+0.741 -> +0.187).
