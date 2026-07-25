# Exp M — Mechanism probe for delta-suppression by real community structure

Files: run.py, probe_results.csv (33 rows = 11 networks x {real, null seed 7, null seed 8}),
staged_results.csv (24 rows = 3 networks x 2 rewire seeds x 4 stages), batchA.log, batchC.log,
staged.log.

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
all 11 networks, probe_results.csv `term_sum` vs `log_ratio`). The sorting term
`log(r_null/r_real)` is POSITIVE on 10/10 networks where it is defined (median +0.487, range
+0.079 to +1.762); the granularity term is NEGATIVE on 8/10 (median -0.132) and the scale term is
negative on 8/10 (median -0.084). `r_pb(null) > r_pb(real)` holds 11/11, including on
ca-CondMat, the one network whose net delta is NOT suppressed. So H3 is rejected: the coarser real
partition (p_intra median 0.759 vs null 0.324) actually INFLATES delta_real and partially masks
the suppression; the whole effect and more sits in partition-score alignment.

**V2 — The triangle/clustering hypothesis (H2) is NOT supported cross-network.** Suppression
strength vs global transitivity C: Spearman +0.285 (p=0.43, n=10), Pearson +0.533 (p=0.11). Vs
mean local clustering: Spearman -0.164 — the WRONG sign, and the two clustering measures disagree
with each other. Vs mean triangles per intra edge: Pearson +0.741 (p=0.014) but Spearman only
+0.273, and the Pearson collapses to **+0.059 when ca-HepPh alone is dropped** — the exact
wiki-Talk failure mode this project was burned by before. No triangle-flavoured variable survives
a jackknife.

**V3 — The staged causal probe cannot test H2: the design is saturated at its smallest dose.** At
stage 0.25 (2.5 double-edge swaps per edge) 99.2-100.0% of the total clustering loss AND
99.6-100.2% of the total modularity loss have already happened, simultaneously. There is no
dose-response curve; there are three replicates of a binary real-vs-randomized contrast. delta is
also not monotone in the swap count (non-monotone in 4/6 network-seed paths). Triangle loss and
community-structure loss are collinear at rho ~ 1 by construction here, so the delta rise cannot be
attributed to triangle destruction specifically.

**V4 — The best-supported correlate is degree sorting of the REAL partition (H1-flavoured), and it
is descriptive, not causal.** Suppression is strongest where the real Leiden partition does the
LEAST degree sorting: hub_inter_lift Spearman **-0.830** (p=0.003, n=10; jackknife range
[-0.933, -0.767], strengthens to -0.933 when ca-HepPh is dropped), hb = E[d_u d_v|inter] /
E[d_u d_v|intra] Spearman -0.709 (p=0.022, jackknife [-0.883, -0.633]). Partial correlations keep
hubLift alive controlling for C (Spearman -0.624) and kill C controlling for hubLift (+0.103). But
n=10, hubLift is partly a re-expression of r_real, and no causal arm tests it.

**Bottom line for C8: the triangle/clustering mechanism is NOT the answer (no), but the question is
now narrower — the suppression is provably a partition-score misalignment effect, not a
granularity artefact.**

## Numbered findings

1. **Exp M reproduces Exp B exactly on the 11 overlapping networks.** delta_real and delta_null
   agree to all printed digits with exp_B_config_null/results.csv (e.g. ca-GrQc 0.0872 / 0.1890;
   email-Enron 0.1501 / 0.2400). Suppressed on 10/11; the single exception is ca-CondMat
   (0.1126 real vs 0.1018 null, ratio 0.904) — the same exception as in Exp B. Median ratio on
   this subset 1.243 (vs 1.599 on the full 17).

2. **Exact decomposition (probe_results.csv, columns `term_sorting`, `term_scale`,
   `term_granularity`, `term_sum`, `log_ratio`).** Per-network sorting term: ca-HepPh +1.762,
   ca-GrQc +1.350, cit-HepPh +1.163, wiki-Vote +0.641, email-Enron +0.507, ca-HepTh +0.466,
   cit-HepTh +0.396, ca-AstroPh +0.369, ca-CondMat +0.115, email-Eu-core +0.079.
   facebook-combined is undefined because r_real = -0.0042 flips sign to r_null = +0.131 — the
   most extreme case of the same phenomenon (delta_real = -0.0014, delta_null = +0.0170).

3. **The two arms differ enormously in partition-score alignment while sharing the score
   multiset.** r_pb: real median 0.1185 (range -0.004 to 0.267), null median 0.2643 (range 0.131
   to 0.443). p_intra: real median 0.759 (0.585-0.961), null median 0.324 (0.246-0.479). sd_s is
   near-invariant as expected (null/real 0.865-1.016), confirming the score distribution is
   essentially untouched.

4. **Clustering is destroyed by rewiring but does not predict suppression.** C real median 0.267,
   null median 0.011 (null/real median 0.088) — the rewiring works. Yet across networks C ranks
   almost independently of suppression (Spearman +0.285). The strongest triangle-flavoured Pearson
   (tri_intra, +0.741) is a single-point artefact: drop ca-HepPh (tri_intra = 98.4, the largest by
   far) and it becomes +0.059; C's Pearson +0.533 likewise falls to +0.104 on the same deletion.

5. **The "triangles protect intra edges" premise is itself only weakly true in the data.**
   tri_intra > tri_inter on 8/11 real graphs, but on email-Enron (11.32 vs 13.97), wiki-Vote
   (17.36 vs 19.75) and email-Eu-core (19.50 vs 19.97) intra edges are LESS triangle-rich than
   inter edges. The fraction of edges with >=1 triangle is >0.88 on both sides for nearly every
   network — triangles are not a discriminating feature of intra edges at this granularity.

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

9. **Degree-sorting correlates (V4) in full.** Against log(delta_null/delta_real), n=10:
   hub_inter_lift Spearman -0.830 / Pearson -0.618; hb Spearman -0.709 / Pearson -0.689; r_pb
   (real) Spearman -0.636; p_intra (real) Spearman +0.515; deg_R2 Spearman +0.212 (nothing).
   Excluding ca-CondMat (the one unsuppressed network) hubLift holds at Spearman -0.767. Note
   real hub_inter_lift (median 1.247) EXCEEDS null (median 1.048): real partitions push hub edges
   between communities more than configuration-model partitions do relative to their own base
   rate, and the more they do so the less suppression there is.

## Caveats

- **The run is INCOMPLETE — 11 of 17 networks, not 17.** batchA.log ends mid-load on com-Amazon
  and batchC.log (61 bytes) contains only the com-Youtube LCC load line — both batches were killed
  by the machine restart before producing any rows. No log exists at all for the first four
  networks in the CSV (ca-GrQc, email-Eu-core, wiki-Vote, ca-HepTh); presumably a "batchB"/
  interactive log that was lost. **The six missing networks are exactly the six largest**
  (com-Amazon, com-DBLP, com-Youtube, wiki-Talk, cit-Patents, wiki-topcats) and they carry the
  heavy tail of the phenomenon: their Exp B null/real ratios are 14.45, 1.34, 2.24, 1.62, 4.39,
  1.25 (median 1.93 vs 1.24 for the covered 11). Any correlation reported here is computed on the
  mild half of the suppression distribution.
- **Correlation fragility.** n=10-11. Every headline correlation is reported with a leave-one-out
  jackknife above because this project has been burned before (r=0.69 -> 0.12 on one deletion).
  tri_intra (+0.741 -> +0.059) and C (+0.533 -> +0.104) are live examples in this very dataset.
  hub_inter_lift's Spearman is the only one whose jackknife interval excludes 0 by a wide margin
  ([-0.933, -0.767]), and even that is one variable selected post hoc out of ~20 candidates, with
  no multiplicity correction. Treat V4 as a hypothesis for a future run, not a result.
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
- **run.py on disk does not match what produced the CSVs.** `PROBE_FIELDS`/`STAGED_FIELDS` in the
  committed run.py include `auc_s` (and `auc_prod`), and `structural_metrics` computes them, but
  every data row in both CSVs has the pre-`auc` column count (35 and 21 rather than 37 and 22) and
  neither header contains `auc_s`. The runs were made with an earlier version of the script. The
  practical loss is real: the class-balance-free Mann-Whitney AUC was the one measure designed to
  separate sorting from granularity WITHOUT relying on the log decomposition, and it is
  unavailable. V1 rests on the algebraic decomposition alone (which is exact, so this is a lost
  cross-check rather than a hole).
- **facebook-combined has delta_real < 0**, so its log-ratio and sorting term are undefined and it
  is excluded from all n=10 correlations. It is the most suppressed network in the set, so its
  exclusion is not conservative.
- Modularity-only Leiden (`ModularityVertexPartition`, n_iterations=2, seed 42), undirected simple
  LCC, single Leiden seed per graph. Resolution-limit effects on p_intra are not probed.

## Effect on claim C8

C8 currently reads: *"Real structure suppresses delta vs null (16/17) — open question."*
This probe does NOT close it, and it does NOT license a triangle-based explanation. It does
narrow it, on the 11 networks it covers. Recommended replacement:

> **C8** Real structure suppresses delta vs null (16/17). Mechanism localized to partition-score
> misalignment, not partition granularity (Exp M, 11/17 networks); triangle/clustering explanation
> unsupported. Root cause still open.

Defensible sentence-level claim for the paper (safe as written):

> The suppression is not an artefact of the real partition being coarser. Writing
> `delta = r * sd(s) / sqrt(p(1-p))`, where `r` is the point-biserial correlation between the DSpar
> score and the intra-edge indicator, decomposes `log(delta_null/delta_real)` exactly into a
> sorting term `log(r_null/r_real)` and a granularity term in `p`. On all ten networks where it is
> defined the sorting term is positive (median +0.49), while the granularity term is negative on
> eight of ten (median -0.13): the coarser real partition inflates delta_real and partially masks
> the suppression. Real Leiden partitions simply align less with the degree-driven DSpar score
> than configuration-model partitions do, even though the two graphs share the score multiset
> exactly. Which structural property produces that misalignment we could not determine: global
> clustering does not predict suppression strength across networks (Spearman +0.29, n=10, p=0.43),
> and progressively rewiring three networks destroys clustering and modularity in the same swaps,
> so it cannot separate them.

Do NOT write, in any form: "real structure suppresses delta because triangle-rich intra-community
edges are protected." The data do not support it and one deletion turns the supporting correlation
into noise.
