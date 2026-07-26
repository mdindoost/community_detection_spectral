# THE STORY — written before any further tex changes (2026-07-25)

Owner: Mohammad. This document is the narrative contract: the paper gets rewritten to tell THIS
story, and no claim enters the paper without the verification listed here.

## The arc (first person, the actual research journey)

1. **The question.** DSpar (Liu et al.) proves that degree-based sampling, with importance
   reweighting, approximates spectral sparsification, so the sparsified graph preserves spectral
   properties. Spectra encode connectivity and cuts. So: *what happens to communities? Are they
   preserved, improved, or harmed?* And two follow-ups: *can we predict/distinguish when and why*,
   and *what does removing hub–hub edges do to a graph's structure?*

2. **The theory.** To answer preserve/improve/harm we need the mechanism. Modularity decomposes as
   Q = F − G; DSpar's score creates a measurable gap δ between intra- and inter-community edges;
   under δ > 0, score-proportional sampling provably removes inter-community edges preferentially
   and raises F for any fixed partition. The hub-bridge condition explains when δ > 0 arises. This
   answers the "why" at the mechanical level, exactly.

3. **The regime split (the paper's central finding), in three verified shades.**
   - **The worst-case guarantee cannot answer the question.** DSpar's spectral bound is
     (1 ± ε/α) [their Thm 2], and on community-rich graphs α collapses: measured on our networks,
     1/α = 535 (ca-GrQc), 139 (ca-CondMat), vs 4.7 (email-Eu-core) — a nontrivial bound on
     ca-GrQc would need ~10^10 samples on a 13k-edge graph. The guarantee is numerically vacuous
     exactly where communities are strong. Hence the question must be answered empirically.
   - **Weighted regime: objective PRESERVED, via unbiasedness — now fully verified (Exp K).**
     Weighted ΔQ_fixed within ±0.004 of zero in all 24 cells across 8 networks, independent of
     1/α (com-Amazon, 1/α = 10,782, tightest of all); identical topology with weights dropped
     inflates +0.013..+0.152. BUT Exp K's two surprises sharpen the claim: (i) preservation of the
     fixed partition's objective is an EVALUATION-correctness fact, not a detection benefit — at
     aggressive retention the weighted pipeline's detected partition transfers WORSE than its
     unweighted twin (6/8), and recovery is worse in every configuration; (ii) there is no speed
     payoff: 0.97–1.06× at quality-preserving retention. Final wording: weights are a correctness
     requirement for evaluation, not a performance option; sparsification is not a route to faster
     or better modularity community detection.
   - **Unweighted regime (as actually used): NOT preserved and NOT improved.** Degree mechanics
     inflate fixed-partition modularity for any partition on any heavy-tailed graph, communities or
     not; on the fixed objective, detection gets slightly worse; ground-truth recovery does not
     improve. Apparent improvements trace to three measurable evaluation artifacts.

4. **When/why — answered honestly.** The mechanical effect is exactly predictable (δ determines it,
   by theorem). The *genuine* effect — the rare case where sparsified seeding truly beats a fair
   baseline — is real (2 of 15 networks) but not predicted by any of 14 structural statistics we
   tested. Prediction of the mechanical part: solved. Prediction of the genuine part: open, and we
   show why small-sample correlation claims (including our own earlier r = 0.92) cannot settle it.

5. **Hub–hub removal has real structural consequences.** Removing hub–hub edges compresses the
   weighted degree distribution (our Lemma + variance condition), driving the ΔG channel of
   modularity change; the hub-bridge spectrum across real networks (ratios 0.5–10.5) maps where
   the mechanism bites; the single robust genuine gain occurs at its extreme. But hub-bridging
   itself is reproduced by configuration models under detected partitions — it marks degree
   structure, not community structure.

6. **LFR deviates from real graphs in exactly the dimensions that matter.** Standard LFR is nearly
   degree-homogeneous (CV ≈ 0.4 vs 1–26 in our suite) and lacks degree–community coupling
   (hb < 1, δ < 0 under planted partitions); our h-rewiring restores the coupling controllably.
   Additionally, δ measured against planted vs detected partitions is not like-for-like.

7. **The by-product.** Answering the question correctly required three controls (fixed objective,
   granularity matching, compute matching) plus a configuration-model null. We publish them as a
   protocol, because the pitfalls they close are shared by the surrounding literature — we fell
   into each one ourselves before building the control.


## P1 grounding (deep research, 2026-07-25 — full evidence in references/P1_EVIDENCE.md)

**Verdict: P1 is legitimate as a published claim and shipped tooling; NOT as routine
billion-scale practice** (the field distributes rather than sparsifies at the frontier — never
claim "practitioners routinely sparsify"). The defensible framing: a 15-year line of papers
proposes sparsify-then-detect and claims speedups AND quality gains; the methods ship
(NetworKit, CRAN, Neo4j GDS); one production system prunes before detecting (Twitter
SimClusters, KDD 2020); and nobody has tested the pipeline end-to-end against cost-matched
baselines. Top-5 citations: Satuluri+ SIGMOD 2011 ("consistently enables higher clustering
accuracies" — verbatim); Wu & Chen ICDM 2020 (GSGAN, "comparable or even better" at 5% edges);
Satuluri+ KDD 2020 (SimClusters, production); Hamann+ SNAM 2016; Chen+ PVLDB 2024.

**Prior-art obligations (must cite and position precisely):**
- Hamann+ 2016 published the NMI-inflation warning and the partition-drift observation FIRST —
  credit them; our resolution-matched control operationalizes their warning. They lack runtime,
  cost-matched baselines, config nulls, GT-recovery controls.
- Blagus+ Physica A 2015: "community structure ... merely an artifact of sampling" — closest
  relative of our artifact framing (different mechanism); cite in related work.
- Chen+ PVLDB 2024: fragmentation-with-pruning at benchmark scale; their conclusion is
  "match sparsifier to task," not "don't."
- Gottesbüren+ ESA 2025: sparsification overhead can counteract speedups (partitioning) —
  corroborates our ~1×.
- Gap statements verified: no end-to-end cost-matched study; no configuration null; no
  resolution-matched GT control; DSpar never before applied to community detection; the
  IJCAI-24 graph-reduction survey contains zero occurrences of "community."
- PRE-SUBMISSION TASK (user-side): re-run citation search with library access (Scholar/ACM
  full texts unreachable from here).

## Title candidates (question-driven)
- "Does Graph Sparsification Preserve Community Structure?"
- "Does Sparsification Preserve Communities? Spectral Guarantees, Degree Mechanics, and Honest Evaluation"

## Claim → verification table

| # | Claim (as the story states it) | Verification | Status |
|---|---|---|---|
| C1 | DSpar w/ reweighting: (1±ε/α) spectral approx of normalized Laplacian; degree scores sandwich R_e within 2/α | Source paper read (references/DSPAR_NOTES.md, Thm 1 p.4, Thm 2 p.4-5); bib correct (TMLR 2023) | **SETTLED** |
| C2 | Weighted regime preserves the fixed-partition objective (unbiasedness; bound vacuous, measured 1/α up to 10,782) — an evaluation-correctness fact, not a detection benefit: transfer/recovery do not improve, speed ≈1× | **Exp K complete**: 24/24 cells ≈0; controls attribute inflation to weight-dropping; V1–V4 in exp_K_weighted_regime/SUMMARY.md | **VERIFIED** |
| C3 | δ>0 ⟹ preferential inter-edge removal, F rises (score-proportional regime) | Thms 1–3 + supplementary proofs; scope via clipping remark (α*≈0.05 verified by direct computation) | Done, in paper |
| C4 | Mechanism verified quantitatively; two regimes (ΔF-driven vs ΔG-driven) | exp_I LCC tables (identity ≤1e-13; direction 9/11; ratio gap 0.01–0.29 measured) | Done, in paper |
| C5 | Unweighted fixed-objective answer: slightly harmed (transfer loss 0.004–0.19, 15/15) | exp4_comprehensive CSV + audit fresh test; tab_transfer_loss | Done, in paper |
| C6 | Recovery not improved (0/5 small w/ AMI + resolution control; 0/3 at scale). Sparse-graph fragmentation is a granularity effect, not benign periphery-shedding: fragments subdivide communities without mixing them, but cut the most-embedded half as readily as the least; resolution-matched original-graph partitions reproduce the profile and preserve cores better 16/20 | exp_F, exp_D CSVs; exp_O (core-preservation hypothesis dead 4/5 predictions) | Done, in paper; exp_O footnote pending tex |
| C7 | Fixed-partition gains are degree mechanics (nulls reproduce 17/17, usually stronger) | exp_B; tab_null_control | Done, in paper |
| C8 | Real structure suppresses δ vs null (16/17, median ratio 1.60). Mechanism localized to partition–score misalignment, not granularity (exp_M 17/17: sorting term positive 16/16 median +0.57; balance-free AUC cross-check Spearman +0.84; heavy tail = score-blind tail, com-Amazon/cit-Patents auc_s ≈ chance); triangle/clustering DEAD (exp_M full coverage; exp_Q causal: tri_intra moves both directions while δ falls uniformly — ca-GrQc δ +0.087→-0.874 with tri_intra -2.8%). **exp_Q causal result: hub-edge placement CONTROLS δ** — at fixed degrees, frozen partition, exactly fixed p_intra AND Q, steering degree-product mass on/off boundaries moves δ 20-450 neutral-sd in the steered direction 5/5, past the config null itself 3/5, surviving fresh re-detection (AMI 0.82-0.94). ATTRIBUTION of the real-vs-null gap to this lever NOT established (balance-free hub stats don't consistently favour the null; hubLift higher in REAL 4/5). Forbidden sentence: "δ suppressed because real partitions keep hub edges inside communities" | exp_B; exp_M SUMMARY (17/17); exp_Q SUMMARY | Done, in paper; C8 wording update pending tex |
| C9 | **Genuine gains: ONE — email-Enron only (Youtube DEMOTED by exp_X).** Enron (+0.008, 8/8 configs): survives adversarial granularity check (exp_N: k-matched +0.006, corr(k,Q)=-0.001), mechanism = 2.58x biased boundary-edge removal, diffuse, no separate basin, compute claims in expectation only; compute control two-dimensional (exp_R: iteration-matched -0.0043..-0.0048 below seeded p<0.03, convergence 4.4x cost still short directional, variance 2.2x all depths). **Youtube +0.003: statistically bulletproof (p=1.9e-5, exact replication of exp_H) but kill criterion FIRED — corr(k,Q)=-0.799 jackknife-stable, seeded k below ALL 20 plain restarts, seeded mean lands ON the plain k-Q regression line (residual -0.0001); mechanism signature present but causally inert (28 edge deletions on 3M edges); exp_H's alpha=0.95 cell annotated granularity-confounded.** Post-hoc HYPOTHESIS logged: the confound is entirely sub-20-node fragments (macro structure identical) | exp_C, exp_E, exp_H sweeps; exp_N + exp_R + exp_X SUMMARYs | Done, in paper; C9 wording + anatomy paragraph pending tex — Youtube claims must be REMOVED from tex |
| C10 | No tested statistic predicts genuine gains; small-n correlations fragile (0.69→0.12) | exp_E correlations + LOO; predictors table | Done, in paper |
| C11 | Hub–hub removal compresses weighted degrees → ΔG channel | Lemma G-change + Prop(B) + exp_I −ΔG column | Done, in paper |
| C12 | Hub-bridge spectrum real (0.48–10.5); inverted on 3 networks; nulls always >1 | exp_B table | Done, in paper |
| C13 | LFR: homogeneous + uncoupled; h-rewiring restores coupling; provenance confound | exp_A, exp1_3 CSVs, exp_D Amazon flip; regenerated figure (n=10⁴ verified) | Done, in paper |
| C14 | Artifacts I–III each reversed a conclusion; I is sparsifier-agnostic (uniform 6/6) | exp audits, exp_G | Done, in paper |
| C15 | Sampler audit: nominal α ≠ realized retention; calibrated fix | Table 1 + clip_fractions.csv + bit-identical validation | Done, in paper |
| C16 | ⚠️ **RESCOPED 2026-07-26 after reading the primary source (references/NOTES_satuluri2011.md) — this is NOT a refutation of Satuluri 2011.** They do NOT commit Artifact I (§4.3 explicitly forbids sparse-graph scoring; Table 2 caption: "phi_avg is always calculated w.r.t. the original graph"), they match cluster counts by construction (k is an input to Metis/Graclus/Metis+MQI), and they charge sparsification time to their speedups. They never claim a modularity improvement — their accuracy claim is external-ground-truth agreement for two named algorithms, and their own honest-transfer conductance is 5W/5L/1T. **CRITICAL SCOPE GAP: their §4.5 states L-Spar "actually outperforms the original clustering starting from degree 50"; ALL our networks are d_avg 5.5-32.6, and on their two low-degree datasets (DIP 6.4, Human 10.8) their results are a wash exactly like ours. None of their four algorithms (Metis, Metis+MQI, MLR-MCL, Graclus) appears anywhere in our study.** What we showed: L-Spar's modularity fails 14/14 cells under honest scoring (dQ_vs_matched -0.014..-0.404; naive scoring flips the sign in every cell = Artifact I, a pitfall of the literature that FOLLOWED them); BUT its Jaccard signal is genuinely structure-aware — the only sparsifier whose selection signal the config null does NOT reproduce (deltaJ collapses to ~0/negative on rewirings). dQ_fixed retired as evidence (null reproduces it fully). Recovery: granularity control kills 5/6 cells; com-Amazon@0.542 exception (+0.062 vs over-matched control) — promoted by exp_V's adjudication (survives +37%-k control AND chance floor; modularity loss -0.079 in same cell: objective/recovery dissociation), then SCOPED by exp_P to the modularity family: reproduces under Louvain (0.4023) but Infomap/label-prop UNSPARSIFIED baselines (0.4646/0.4801) beat it — it is granularity repair of the modularity objective's coarse default, not a capability sparsification adds. No speedup (0.87-1.35x) | exp_L SUMMARY; exp_V V5; exp_P V4 | Verified; NOT yet in paper — new section needed |
| C17 | Spending the verified signal: of four deployments (weighting w=1+J, shuffled-weight null, seeded refinement, protected deletion), NONE yields an honest modularity gain on >1/7 networks, and the one modularity gain (Enron, all three weight/seed arms ~+0.010) is reproduced BY THE SHUFFLED-WEIGHT NULL (+0.0117 — bigger) and by a config null on a structureless graph — weighting's benefit is weight-heterogeneity mechanics, indicting the edge-weighting literature's evaluation (shuffled-weight control now mandatory). Sole all-controls survivor: seeded refinement's recovery gain on com-Amazon (chance-corrected +0.036, 4.0 sd, at 1.7x cost with modularity loss in the same cell). Seeding is the only structure-dependent deployment (null worse 3/3). | exp_V SUMMARY (results/null_arm/recovery.csv, 49+18+88 rows) | Verified; NOT yet in paper — new section needed |
| C18 | Algorithm generality (exp_P): all negatives hold under Infomap/Louvain/label-prop — 60/63 cells negative vs best-of-N restarts; Artifact I is algorithm-general (sparse-graph scoring positive 60/63, sign flips vs honest transfer 45/63); reverse kill NOT triggered (max 2/7, ->1/7 under stronger control; label-prop "gains" = giant-cluster pathology relief); no speedup at quality-preserving retention (max 0.66x). Scope: no MCL/Metis/Graclus (unavailable). Live leads (unpredicted, need pre-registration): Infomap recovery gain at DSpar-0.8 (email-Eu-core AMI 0.523->0.612 at identical k, robust to trials=10 and best-of-10; third instance of objective/recovery dissociation); weak-control inversion example (label-prop looks BEST under r=1 matching, WORST under best-of-20 — the paper's thesis landing on us) | exp_P SUMMARY (results/runs/recovery/bestof/giant_share/trials_check.csv) | Verified; NOT yet in paper — new section needed |

## What changes in the tex (AFTER story approval — nothing touched yet)
1. Title → question form. Abstract → the arc above (Q first, regime-split answer, honest when/why).
2. Intro → first-person journey (paras 1–7 above), replacing the "audit of claims" framing.
   "Earlier version of this work" shrinks further: the story IS this work, one continuous project.
3. New Proposition (C2) + its 5-line proof; placed in §3 or §6.1.
4. §5–§8 content stays as verified; transitions reworded to serve the question, not the audit.
5. Discussion: answers restated per question Q1/Q2; protocol demoted from headline to by-product.
6. **Exp K (required before writing the weighted-regime claims):** audit the Feb weighted run,
   then extend: weighted DSpar + weighted Leiden across the suite (≥6 networks, calibrated
   sampler, α ∈ {0.8, 0.9}), measuring (a) weighted ΔQ_fixed (preservation), (b) the honest
   transfer metric (weighted-graph partition scored on original), (c) recovery vs baseline on
   labeled sets. Predicted outcome per unbiasedness: (a) ≈ 0; (b)(c) decide the practical
   recommendation ("keep the weights" as safe acceleration vs neutral).

## Verification rules going forward
- Every numeric claim keeps its CSV pointer (this table + audit trail in PAPER_RESTRUCTURE/).
- C1's wording is written only after Mohammad checks Liu et al.'s theorem statement.
- C2's proposition gets either the empirical check or a "theory-only" label — no silent assertion.
- Any new prose sentence with a number must name its source row before it enters the tex.

| C19 | **THE HEADLINE IS SCOPED (exp_AA, 2026-07-26).** With REAL Metis (pymetis), L-Spar crosses from harmful to helpful at **d_avg ~ 50 — exactly Satuluri et al.'s stated threshold** — and helps on BOTH metrics: honest-transfer modularity on the ORIGINAL graph positive 0/12 cells at d=11,25 → 8/12 at d=49 → 10/12 at 102 → 12/12 at 221; AMI +0.042/+0.116/+0.093. NOT seed noise (10 Metis seeds, worst-case still +0.001..+0.008 Q and +0.064..+0.134 AMI vs sd 0.002/0.011). NOT edge budget (DSpar and uniform at matched retention negative everywhere). **CANNOT be granularity — Metis takes k as input, nc_base == nc_sparse by construction.** P1 FALSIFIED; kill fired against us. FOR LEIDEN the story is unchanged: modularity gain ≤ best-of-5 restart noise, and the AMI gain dies under the resolution-matched control 21/29 cells. Denoising confirmed (Spearman(noise, gain) = 1.00 in 4/4 Leiden cells) but additive, not causal (gain does not vanish at zero noise). CONTEXT THAT MUST TRAVEL: resolution-tuned Leiden on the UNTOUCHED graph beats every sparsified route (AMI 0.956 vs Metis+L-Spar 0.754) — sparsification repairs a weaker fixed-k partitioner, it does not beat the best available route; and with exact Jaccard the pipeline is 0.58x (slower) at d=200. **Required headline qualifiers: modularity-family detection, sparse graphs (d_avg < ~50), near-linear optimizers.** | exp_AA SUMMARY (results_armA/B/C.csv, metis_noise.csv, lfr_generation.csv) | Verified; headline rescope REQUIRED before print |

| C20 | **Sparsifier coverage completed (exp_AB).** Chen's three untested top fidelity preservers (K-Neighbor, Local Degree, Local Similarity) + an untested CLASS (connectivity-preserving MST-backbone) through the honest protocol: **0 of 102 matched Leiden cells show an honest modularity gain** (best cell -0.0105; all 7 arms negative). **The backbone removes Artifact II's mechanism by construction (0.00% fragment nodes, 0 singletons, where other arms shatter up to 86%) and the conclusion does not move** — this converts "fragmentation explains the loss" from a plausible story into a TESTED AND REJECTED sufficient explanation. Random fill beats Jaccard fill 10/10 on the objective: the backbone does the work, the similarity signal subtracts. **Two qualifications: (a) kill direction 2 FIRES on recovery for Local Degree (com-Amazon, com-DBLP, chance-excess +0.021..+0.103 vs nc-matched control), and Local Similarity on com-Amazon (avgF1 0.470) beats even the best point of the entire resolution sweep on the untouched graph (0.440) — a second independent instance of the objective/recovery dissociation from a family unrelated to L-Spar's Jaccard. (b) THE exp_AA DEGREE THRESHOLD MUST BE RESTATED: real Metis on real heavy-tailed graphs shows the fixed-k gain already at d_avg 28.5-32.6 (wiki-Vote +0.0166, 15-21x seed sd, worst-case positive, 4 sparsifiers), so "d_avg < 50 is the dead zone" is an LFR statement, not universal.** No arm pays for itself end-to-end (0.58x-1.51x) | exp_AB SUMMARY (results.csv 146 rows, recovery.csv 255, analysis_full.txt) | Verified; intro B6 threshold sentence REQUIRES revision |
