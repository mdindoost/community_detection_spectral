# Experiment 1.3: LFR Benchmark Analysis (Theory-Aligned)

## Overview

This experiment explains why standard LFR benchmarks don't exhibit DSpar improvements, while real networks do. The key insight is that LFR places inter-community edges **uniformly**, whereas real networks exhibit **hub-bridging** (inter-community edges preferentially connect high-degree nodes).

### Key Hypothesis

DSpar works because:
1. Inter-community edges connect hubs (high-degree nodes)
2. DSpar scores: `s(e) = 1/d_u + 1/d_v` → hub edges get LOW scores → retained
3. Standard LFR lacks this property → DSpar has no preferential effect

### Theory-Aligned Evaluation

Uses **ground-truth LFR communities as FIXED partition** (not Leiden re-optimization):
- Modularity decomposition: `Q = F - G`
- Reconstruction identity: `ΔQ_fixed = ΔF_obs - ΔG_obs`
- Verified to machine precision (ε < 10⁻¹⁰)

---

## Quick Start

```bash
cd PAPER_EXPERIMENTS
python exp1_3_lfr_analysis.py
```

**Runtime**:
- Small only (RUN_LARGE=False): ~5-10 minutes
- Small + Large (RUN_LARGE=True): ~30-60 minutes

---

## Configuration

Edit constants at top of `exp1_3_lfr_analysis.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SMALL_N` | 1000 | Small network size |
| `LARGE_N` | 10000 | Large network size |
| `RUN_LARGE` | True | If True, run both small and large experiments |
| `N_REPLICATES_SMALL` | 10 | Trials per config (small networks) |
| `N_REPLICATES_LARGE` | 3 | Trials per config (large networks) |
| `MIXING_PARAMS` | [0.1, 0.2, 0.3, 0.4, 0.5] | Fraction of inter-community edges |
| `HUB_BRIDGE_STRENGTHS` | [1.0, 2.0, 4.0] | Hub preference exponent (1.0 = none) |
| `RETENTIONS` | [0.5, 0.8] | DSpar retention α values |
| `RUN_LEIDEN` | False | Enable optional Leiden evaluation |

---

## Output Structure

```
results/exp1_3_lfr/
├── lfr_analysis_raw.csv              # All trial data
├── lfr_analysis_summary.csv          # Aggregated by (network_type, n_nodes, μ, hub_strength, retention)
├── plot1_delta_vs_mu_n{N}_r{R}.pdf/png     # δ vs mixing parameter
├── plot2_dQ_vs_delta_n{N}_r{R}.pdf/png     # ΔQ vs δ scatter
├── plot3_dQ_vs_hub_ratio_n{N}_r{R}.pdf/png # ΔQ vs hub-bridge ratio
└── plot4_decomposition_mu0.3_n{N}_r{R}.pdf/png  # Bar chart decomposition
```

Where `{N}` = n_nodes (1000 or 10000) and `{R}` = retention (0.5 or 0.8)

---

## Output Columns

### Raw CSV (`lfr_analysis_raw.csv`)

| Column | Description |
|--------|-------------|
| `network_type` | 'standard_lfr' or 'modified_lfr' |
| `n_nodes` | Target network size (1000 or 10000) |
| `n_nodes_actual` | Actual size after taking LCC |
| `mu` | Mixing parameter (inter-community edge fraction) |
| `hub_strength` | Hub-bridging strength (1.0 for standard) |
| `retention` | DSpar retention α (0.5 or 0.8) |
| `hub_bridge_ratio` | E[d_u·d_v \| inter] / E[d_u·d_v \| intra] |
| `delta` | DSpar separation δ = μ_intra - μ_inter |
| `Q_orig_fixed` | Modularity on original (fixed partition) |
| `Q_sparse_fixed` | Modularity on sparse (fixed partition) |
| `dQ_fixed` | ΔQ = Q_sparse - Q_orig |
| `F_orig`, `F_sparse`, `dF_obs` | F-term values and change |
| `G_orig`, `G_sparse`, `dG_obs` | G-term values and change |
| `dQ_reconstructed` | ΔF - ΔG (should equal dQ_fixed) |
| `epsilon` | \|dQ_fixed - dQ_reconstructed\| (≈ 0) |

### Summary CSV (`lfr_analysis_summary.csv`)

Aggregated mean/std by `(network_type, n_nodes, mu, hub_strength, retention)` for:
- `hub_bridge_ratio`, `delta`, `dQ_fixed`, `dF_obs`, `dG_obs`, `epsilon`
- Plus `neg_dG_obs_mean/std` for convenience

---

## Plots

### Plot 1: δ vs μ
Shows DSpar separation across mixing parameters:
- Standard LFR: δ ≈ 0 (no separation)
- Modified LFR: δ > 0 (separation increases with hub_strength)

### Plot 2: ΔQ vs δ (Scatter)
Demonstrates correlation between DSpar separation and modularity improvement.
- Positive correlation validates the theory

### Plot 3: ΔQ vs Hub-Bridge Ratio
Shows relationship between hub-bridging property and DSpar effectiveness.

### Plot 4: Decomposition at μ=0.3
Bar chart showing ΔQ_fixed, ΔF_obs, and -ΔG_obs contributions.

---

## Key Differences from Original

| Aspect | Original | Refactored |
|--------|----------|------------|
| Partition | Leiden (re-optimized) | Ground-truth (fixed) |
| Modularity | `leidenalg.ModularityVertexPartition` | Manual Q = F - G |
| Hub strength | Fixed at 2.0 | Sweep [1.0, 2.0, 4.0] |
| Degree update | Static during rewiring | Dynamic (recomputed) |
| Verification | None | ε = \|ΔQ - (ΔF - ΔG)\| < 10⁻¹⁰ |
| DSpar | Local implementation | `experiments.dspar.dspar_sparsify` |
| Leiden | `leidenalg` library | igraph built-in (optional) |

---

## Modifying the Experiment

### Change Hub-Bridging Strengths

```python
HUB_BRIDGE_STRENGTHS = [1.0, 1.5, 2.0, 3.0, 4.0]
```

### Enable Leiden Evaluation

```python
RUN_LEIDEN = True
```

This adds `Q_orig_leiden`, `Q_sparse_leiden`, `dQ_leiden` to output.

### Change LFR Parameters

```python
N_NODES = 2000
AVG_DEGREE = 20
MAX_DEGREE = 100
MIN_COMMUNITY = 30
MAX_COMMUNITY = 200
```

### Add More Mixing Parameters

```python
MIXING_PARAMS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
```

---

## Expected Results

### Standard LFR
- Hub-bridge ratio ≈ 1.0 (uniform placement)
- δ ≈ 0 (no DSpar separation)
- ΔQ_fixed ≈ 0 (no improvement)

### Modified LFR (hub_strength = 4.0)
- Hub-bridge ratio > 1.0 (hubs connected)
- δ > 0 (DSpar separation exists)
- ΔQ_fixed > 0 (improvement observed)

### Interpretation

Standard LFR benchmarks are **not suitable** for evaluating DSpar because they don't model the hub-bridging property found in real networks. To properly evaluate DSpar on synthetic networks:
1. Use modified LFR with hub-bridging, OR
2. Use real networks (as in Experiment 1.2)

---

## Troubleshooting

### "LFR generation failed"

NetworkX LFR generator can fail for certain parameter combinations. Try:
- Reducing `mu` (e.g., max 0.4)
- Adjusting degree distribution exponents (tau1, tau2)
- Increasing community size range

### Large epsilon values

If ε > 10⁻¹⁰, check:
1. F and G computations use same membership dict
2. No floating-point overflow in degree products
3. Graph is connected (isolated nodes excluded)

### Import errors

Ensure you're running from `PAPER_EXPERIMENTS/`:
```bash
cd /path/to/community_detection_spectral/PAPER_EXPERIMENTS
python exp1_3_lfr_analysis.py
```
