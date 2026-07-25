# Phase 1 — Consolidated Verdict (2026-07-24)

All four decision experiments completed. No skips. Details in each `exp_*/SUMMARY.md`.

## The four verdicts

**A — Partition provenance (LFR).** δ does NOT flip sign on standard LFR under Leiden partitions
(stays −0.007…−0.003), but Leiden biases δ positive at every μ, growing as Leiden departs from the
planted truth (NMI 0.97→0.67, Δδ up to +0.0032 at μ=0.5). The paper's LFR parameterization is
nearly degree-homogeneous (max deg ~58, CV 0.40). Conclusion: δ>0 requires BOTH a Leiden-type
partition AND heavy-tailed degrees; the draft's LFR-vs-real contrast survives but is confounded by
partition provenance and must be presented with this analysis.

**B — Configuration-model null (17 networks).** The smoking gun. Degree-preserving rewired nulls
reproduce δ>0 in **17/17** and ΔQ_fixed>0 in **17/17**, and the null usually EXCEEDS the real
graph: ΔQ_fixed(null) ≥ real in 15/17, δ(null) ≥ real in 16/17, median ratio 1.24. Extremes:
com-Amazon real ΔQ=1.5e-5 vs null 0.059; cit-Patents 0.005 vs 0.077; wiki-Talk null hb=105.6;
facebook-combined real δ is **negative** (−0.0014) while its null is positive. hb>1 holds in 17/17
nulls but is erratic in real graphs (0.48–10.5). Conclusion: neither δ>0 nor ΔQ_fixed>0 is evidence
of community clarification; both follow from degree heterogeneity alone.

**C — True retention + runtime-matched seeding.** All repo samplers saturate (even
`probabilistic_no_replace` cannot exceed ~0.66 retention at α=1 due to probability clipping); a
calibrated sampler (E[ret]=α exactly) was built. Raw transfer to the original graph is negative in
45/48 cells, shrinking ~10× as retention→1; the only crossover is **email-Enron** (positive at
80–95% true retention). Seeded refinement beats the runtime-matched baseline (best-of-2 restarts)
only on email-Enron (8/8 cells, both samplers, +0.0007…+0.0149, 5–8σ; also beats best-of-5).
wiki-Vote is the negative control: beats the mean baseline 6/8 but loses runtime-matched 8/8 —
unmatched comparisons manufacture false positives.

**D — Ground-truth recovery at scale (SNAP top-5000).** Apparent F1 gains under the draft's setting
(+0.074 Amazon, +0.040 Youtube) are a granularity artifact: DSpar fragments the graph into 20–40×
more clusters and the GT communities are tiny. A resolution-matched control (Leiden on the
UNSPARSIFIED graph, γ tuned to equal cluster count) beats DSpar 2–4× everywhere (0.404 vs 0.285;
0.293 vs 0.100; 0.129 vs 0.050). Mild true pruning: −0.037…+0.001 ≈ nothing. Bonus: δ flips sign on
com-Amazon between GT (−0.004) and Leiden (+0.011) partitions; Amazon's GT hb=0.78 yet raw ΔF1
looked best there — hb does not predict recovery gains.

## What this means

**Dead:** "sparsification improves community detection/modularity" (cross-graph artifact);
r=0.92 as framework validation (near-tautological); recovery gains at scale (granularity artifact);
δ>0 or hb>1 as evidence of community-relevant structure (nulls reproduce both).

**Alive:** all theorems except Corollary 2 (fixable hypothesis gap); Theorem 3 needs restating for
the clipped/calibrated sampler; the LFR contrast with provenance caveat; email-Enron as an
existence proof of genuine, runtime-matched-surviving gains under extreme hub-bridging, with
wiki-Vote as negative control.

**New contributions Phase 1 created:**
1. Three artifact mechanisms, each with a diagnostic control: cross-graph Q (control: score on
   original / seeded refinement), granularity (control: resolution-matched Leiden), unmatched
   compute (control: runtime-matched restarts).
2. The configuration-model null table (17 networks, real vs rewired) — the paper's central figure.
3. The sampler audit: "α" means four different things across definition/text/two code paths;
   calibrated sampler provided.
4. Candidate constructive statistic for Phase 2: **excess separation** δ* = δ(real) − δ(config
   null) (equivalently the ΔQ_fixed ratio vs null). Notably, email-Eu-core is the network where
   real ΔQ_fixed most exceeds its null (ratio 0.43 null/real) — and it was the draft's only
   positive recovery dataset. Worth testing δ* as the honest predictor in Phase 2.

## Phase 2 go/no-go: GO

Paper reframes as: "When does sparsification help community detection? Artifact mechanisms,
null-model controls, and the boundary where genuine gains exist." Theory retained as the
quantitative mechanism of the artifact. Target: IEEE TNSE; arXiv on completion.
