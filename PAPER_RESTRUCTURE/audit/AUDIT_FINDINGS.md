# Audit findings — 2026-07-24

## 1. Apples-to-apples test (fresh run, DSpar `method="paper"`, nominal α=0.8, 5 seeds, LCC)

| dataset | Q_orig(P0) | Q_sparse(Pα) | Q_orig(Pα) | paper_delta | honest_delta |
|---|---|---|---|---|---|
| ca-CondMat | 0.7309±0.0017 | 0.8288±0.0017 | 0.6582±0.0013 | +0.0979 | −0.0727 |
| ca-HepTh | 0.7616±0.0004 | 0.8773±0.0023 | 0.6317±0.0042 | +0.1157 | −0.1298 |
| email-Enron | 0.6090±0.0053 | 0.7918±0.0012 | 0.5594±0.0028 | +0.1828 | −0.0496 |
| wiki-Vote | 0.4237±0.0047 | 0.4993±0.0107 | 0.3980±0.0026 | +0.0756 | −0.0257 |
| email-Eu-core | 0.4160±0.0019 | 0.5094±0.0070 | 0.4045±0.0018 | +0.0934 | −0.0115 |

Best-of-5-seeds comparison flips sign identically. Corroborated independently by
`results/exp4_comprehensive/comprehensive_alpha_results.csv` (`Q_transfer_loss` > 0 for all
15 datasets at every α; e.g. wiki-Talk α=0.8: Q_sparse 0.897 → Q_on_orig 0.417).

## 2. Config-model control (degree-preserving rewire, 3 reps, α=0.8 nominal)

| dataset | graph | Q_fixed(base) | ΔQ_fixed | δ |
|---|---|---|---|---|
| email-Enron | real | 0.606–0.613 | +0.1425±0.0033 | +0.152 |
| email-Enron | rewired | 0.242–0.244 | **+0.1559±0.0003** | **+0.250** |
| ca-CondMat | real | 0.730–0.734 | +0.0594±0.0014 | +0.112 |
| ca-CondMat | rewired | 0.311–0.313 | **+0.0651±0.0016** | **+0.098** |

Rewired graphs have no community structure yet show equal-or-larger ΔQ_fixed and δ.

## 3. Seeded refinement (repo's own `exp4_comprehensive`, `Q_final` = Leiden seeded on ORIGINAL graph)

dQ_final at α∈{0.8,0.9} across 15 datasets: range −0.0008 … +0.0078.
Only email-Enron (+0.0072/+0.0078) and com-DBLP (+0.0029/+0.0031) exceed +0.001.
Draft claims ΔQ_Leiden +0.05…+0.30 — sparse-graph self-scoring inflation.

## 4. Sampler facts

- Draft Exp 1 & Exp 3 pipelines use `experiments/dspar.py` `method="paper"` (WITH replacement);
  actual distinct-edge retention at nominal α=0.8 ≈ 0.33–0.52 (documented in the module docstring;
  visible in draft Table 5: com-Amazon m_α/m = 477,328/925,872 = 0.516).
- Draft text says "probabilistic no-replacement sampling"; formal Definition implies E[retention]=α.
  Three-way inconsistency (definition / text / code).
- `probabilistic_no_replace` also cannot reach true mild retention: probabilities clip at 1, so it
  saturates (e.g. email-Eu-core retains only 0.662 even at α=1.0). Exp C introduced a calibrated
  sampler (λ solved so E[retention]=α exactly) for the true-retention regime.
- Author convention (M. Dindoost, 2026-07-24): α=1.0 is reserved as the no-sparsification sentinel.
  Applied in `exp1_2_theoretical_predictions.py` (skips the sampler at retention=1.0) but NOT in
  `exp4_comprehensive_analysis.py` — its `alpha=1.0` rows ran the sampler (facebook-combined:
  actual_retention=0.496 at α=1.0), so those rows are ~50%-retention runs, not baselines. Use that
  file's explicit `Q_base` columns instead.

## 5. Partition-provenance observation (to be tested in exp_A)

δ>0 in draft Exp 1 is measured w.r.t. Leiden partitions; δ<0 in Exp 2 (LFR) w.r.t. planted
partitions; Exp 3 ground-truth δ mixed (3/5 negative). Rewired-graph Leiden partition shows
δ=+0.25. Hypothesis: δ>0 is chiefly a property of Leiden-found partitions on degree-heterogeneous
graphs, not of real networks.
