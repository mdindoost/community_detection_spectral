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
   - **Weighted regime: PRESERVED, via unbiasedness, empirically confirmed.** Algorithm 1 gives
     E[A'] = A [their p.3]; measured (Feb run, α=1.0 nominal ≈55% effective retention):
     weighted ΔQ_fixed = −0.0003 (com-DBLP), +0.0005 (com-Amazon) — zero within noise — while
     the unweighted variant on the same graphs gives +0.0382 and −0.0003 respectively.
     Keeping the weights makes the inflation vanish.
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

## Title candidates (question-driven)
- "Does Graph Sparsification Preserve Community Structure?"
- "Does Sparsification Preserve Communities? Spectral Guarantees, Degree Mechanics, and Honest Evaluation"

## Claim → verification table

| # | Claim (as the story states it) | Verification | Status |
|---|---|---|---|
| C1 | DSpar w/ reweighting: (1±ε/α) spectral approx of normalized Laplacian; degree scores sandwich R_e within 2/α | Source paper read (references/DSPAR_NOTES.md, Thm 1 p.4, Thm 2 p.4-5); bib correct (TMLR 2023) | **SETTLED** |
| C2 | Weighted regime preserves fixed-partition modularity — anchored on unbiasedness E[A']=A + concentration, NOT the vacuous (1±ε/α) bound (measured 1/α: 535 ca-GrQc, 139 ca-CondMat, 4.7 email-Eu-core) | Feb data verified: weighted ΔQ_fixed −0.0003/+0.0005 vs unweighted +0.0382/−0.0003 (com-DBLP/com-Amazon, ~55% eff. retention) | Half-verified (2 datasets); **Exp K extends across suite + honest metrics** |
| C3 | δ>0 ⟹ preferential inter-edge removal, F rises (score-proportional regime) | Thms 1–3 + supplementary proofs; scope via clipping remark (α*≈0.05 verified by direct computation) | Done, in paper |
| C4 | Mechanism verified quantitatively; two regimes (ΔF-driven vs ΔG-driven) | exp_I LCC tables (identity ≤1e-13; direction 9/11; ratio gap 0.01–0.29 measured) | Done, in paper |
| C5 | Unweighted fixed-objective answer: slightly harmed (transfer loss 0.004–0.19, 15/15) | exp4_comprehensive CSV + audit fresh test; tab_transfer_loss | Done, in paper |
| C6 | Recovery not improved (0/5 small w/ AMI + resolution control; 0/3 at scale) | exp_F, exp_D CSVs | Done, in paper |
| C7 | Fixed-partition gains are degree mechanics (nulls reproduce 17/17, usually stronger) | exp_B; tab_null_control | Done, in paper |
| C8 | Real structure suppresses δ vs null (16/17) — open question | exp_B | Done, in paper |
| C9 | Genuine gains: Enron robust (+0.008, 8/8 configs); Youtube small, regime-confined (+0.003) | exp_C, exp_E, exp_H sweeps | Done, in paper |
| C10 | No tested statistic predicts genuine gains; small-n correlations fragile (0.69→0.12) | exp_E correlations + LOO; predictors table | Done, in paper |
| C11 | Hub–hub removal compresses weighted degrees → ΔG channel | Lemma G-change + Prop(B) + exp_I −ΔG column | Done, in paper |
| C12 | Hub-bridge spectrum real (0.48–10.5); inverted on 3 networks; nulls always >1 | exp_B table | Done, in paper |
| C13 | LFR: homogeneous + uncoupled; h-rewiring restores coupling; provenance confound | exp_A, exp1_3 CSVs, exp_D Amazon flip; regenerated figure (n=10⁴ verified) | Done, in paper |
| C14 | Artifacts I–III each reversed a conclusion; I is sparsifier-agnostic (uniform 6/6) | exp audits, exp_G | Done, in paper |
| C15 | Sampler audit: nominal α ≠ realized retention; calibrated fix | Table 1 + clip_fractions.csv + bit-identical validation | Done, in paper |

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
