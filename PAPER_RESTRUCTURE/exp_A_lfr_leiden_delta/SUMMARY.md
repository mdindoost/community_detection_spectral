# Experiment A — δ under planted vs Leiden partitions on standard LFR

**Graphs:** standard LFR (networkx `LFR_benchmark_graph`), n=10000, tau1=3, tau2=1.5, <k>=15, k_max=50, community size in [20, 100] (identical to `PAPER_EXPERIMENTS/exp1_3_lfr_analysis.py`).

**Partitions:** planted ground truth vs `leidenalg` `ModularityVertexPartition` (n_iterations=-1) on the *same* graph. 3 seeds per μ, values are means over seeds.

δ = μ_intra − μ_inter with s(e)=1/d_u+1/d_v;  hb = E[d_u d_v | inter] / E[d_u d_v | intra].

| μ | δ_planted | δ_leiden | hb_planted | hb_leiden | n_comm_planted | n_comm_leiden | NMI |
|---|---|---|---|---|---|---|---|
| 0.1 | -0.00298 | -0.00280 | 0.939 | 0.942 | 223.7 | 148.0 | 0.967 |
| 0.2 | -0.00435 | -0.00416 | 0.912 | 0.916 | 231.7 | 116.7 | 0.939 |
| 0.3 | -0.00599 | -0.00573 | 0.888 | 0.893 | 229.3 | 87.3 | 0.902 |
| 0.4 | -0.00750 | -0.00677 | 0.866 | 0.876 | 228.7 | 68.0 | 0.851 |
| 0.5 | -0.01016 | -0.00696 | 0.828 | 0.869 | 236.0 | 48.3 | 0.673 |

With standard deviations over seeds:

| μ | δ_planted | δ_leiden | hb_planted | hb_leiden | Q_planted | Q_leiden |
|---|---|---|---|---|---|---|
| 0.1 | -0.00298 ± 0.00028 | -0.00280 ± 0.00027 | 0.939 ± 0.007 | 0.942 ± 0.007 | 0.8446 | 0.8453 |
| 0.2 | -0.00435 ± 0.00012 | -0.00416 ± 0.00008 | 0.912 ± 0.004 | 0.916 ± 0.003 | 0.6889 | 0.6909 |
| 0.3 | -0.00599 ± 0.00012 | -0.00573 ± 0.00014 | 0.888 ± 0.001 | 0.893 ± 0.001 | 0.5404 | 0.5439 |
| 0.4 | -0.00750 ± 0.00010 | -0.00677 ± 0.00008 | 0.866 ± 0.001 | 0.876 ± 0.001 | 0.4122 | 0.4190 |
| 0.5 | -0.01016 ± 0.00009 | -0.00696 ± 0.00010 | 0.828 ± 0.002 | 0.869 ± 0.002 | 0.2922 | 0.3098 |

### Partition-provenance shift (Leiden − planted)

| μ | Δδ = δ_leiden − δ_planted | Δhb | max degree | degree CV |
|---|---|---|---|---|
| 0.1 | +0.00019 | +0.0036 | 57 | 0.406 |
| 0.2 | +0.00019 | +0.0040 | 56 | 0.404 |
| 0.3 | +0.00026 | +0.0047 | 57 | 0.399 |
| 0.4 | +0.00073 | +0.0107 | 60 | 0.400 |
| 0.5 | +0.00320 | +0.0409 | 60 | 0.407 |

## Verdict

**No, δ does not flip sign on standard LFR.** Swapping the planted partition for a Leiden partition of the *same* graph moves δ in the predicted (positive) direction at every μ — shift Δδ = +0.00019 to +0.00320, growing monotonically with μ and reaching 32% of |δ_planted| at μ=0.5 — and raises the hub-bridge ratio in lockstep (0.887 → 0.899 on average), but δ_leiden stays negative throughout (-0.00696 to -0.00280). So partition provenance is a real, systematic bias in the direction of the hypothesis, yet on LFR it is one to two orders of magnitude too small to explain the δ ≈ +0.25 seen on a rewired email-Enron or the positive δ on all 17 real networks. The mechanism is visible in the partition itself: Leiden merges the planted communities (≈224 planted → 148 found at μ=0.1, 236 → 48 at μ=0.5; NMI 0.673–0.967), which absorbs low-s(e) boundary edges into communities — exactly the artifact direction — and the effect scales with how far Leiden departs from the ground truth. The reason it cannot go further here is that LFR as parameterised in the paper is nearly degree-homogeneous (mean max degree 58, degree CV 0.40), whereas the real/rewired networks are heavy-tailed; δ>0 therefore appears to require **both** a Leiden-found partition **and** strong degree heterogeneity, and this experiment isolates the first factor as insufficient on its own. Actionable conclusion: the estimator-artifact claim should be tested on degree-heterogeneous graphs *without* planted communities (the rewired-Enron line of evidence, exp_B), not on standard LFR; and the LFR-vs-real δ contrast in the draft remains confounded by partition provenance and should not be presented as a like-for-like comparison.

*Scale caveat:* s(e)=1/d_u+1/d_v is O(1/degree), so δ magnitudes are not directly comparable across graphs with different degree profiles. LFR here has mean degree 18.4 and a hard minimum degree well above 1, bounding |δ| near 0.1; sparse real networks with many degree-1/2 nodes admit |δ| up to ~1. The sign, not the magnitude, is the decision-relevant quantity — and the sign does not flip.
