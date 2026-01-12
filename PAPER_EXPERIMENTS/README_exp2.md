# Experiment 2: Ground-Truth Community Recovery

## Overview

This experiment evaluates whether DSpar sparsification improves recovery of known ground-truth communities, **conditioned on the DSpar separation statistic δ**.

### Key Hypothesis

- **DSpar improves accuracy when δ > 0** (DSpar-favorable)
- **DSpar has no effect or degrades accuracy when δ ≤ 0** (DSpar-neutral/unfavorable)

The experiment does **not** assume sparsification always helps. Instead, it explicitly reports cases where it does and does not.

---

## Quick Start

```bash
cd PAPER_EXPERIMENTS
python exp2_ground_truth_recovery.py
```

**Runtime**: ~2-5 minutes

---

## Datasets

### Small / Sanity-Check (disjoint ground truth)

| Dataset | Nodes | Edges | Communities | Source |
|---------|-------|-------|-------------|--------|
| Karate Club | 34 | 78 | 2 | Zachary (1977) |
| Dolphins | 62 | 159 | 2 | Lusseau et al. (2003) |
| Football | 115 | 613 | 12 | Girvan & Newman (2002) |
| Polbooks | 105 | 441 | 3 | Krebs (unpublished) |

### Medium / Realistic

| Dataset | Nodes | Edges | Communities | Source |
|---------|-------|-------|-------------|--------|
| email-Eu-core | 1005 | 25571 | 42 | SNAP |

### Adding New Datasets

Edit `get_all_datasets()` in `exp2_ground_truth_recovery.py`:

```python
def get_all_datasets():
    return [
        (load_karate_club, "Karate Club"),
        (load_dolphins, "Dolphins"),
        # Add your loader here:
        (load_my_dataset, "My Dataset"),
    ]
```

Loader function signature:
```python
def load_my_dataset():
    """Load graph and ground-truth labels."""
    G = nx.Graph()  # Load your graph
    ground_truth = {node: community_id for node in G.nodes()}
    return G, ground_truth, "My Dataset"
```

---

## Experimental Protocol

For each dataset:

1. **Load** graph G and ground-truth partition Y
2. **Run Leiden** on G → partition P₀
3. **Compute DSpar separation** δ = μ_intra - μ_inter using Y
4. **Apply DSpar sparsification** at α = 0.8 (main), α = 0.5 (robustness)
5. **Run Leiden** on sparsified graph → P_α
6. **Compare** partitions to ground truth using NMI and ARI
7. **Compute** ΔNMI, ΔARI, ΔQ

---

## Metrics

### DSpar Separation (δ)

```
δ = μ_intra - μ_inter

where:
  μ_intra = E[s(e) | e is intra-community]
  μ_inter = E[s(e) | e is inter-community]
  s(e) = 1/d_u + 1/d_v  (DSpar edge score)
```

**Interpretation**:
- δ > 0: Intra-community edges have higher scores → more likely to be removed → DSpar favors inter-community removal
- δ ≤ 0: No structural advantage for DSpar

### Community Detection Quality

| Metric | Range | Description |
|--------|-------|-------------|
| NMI | [0, 1] | Normalized Mutual Information |
| ARI | [-1, 1] | Adjusted Rand Index (corrected for chance) |

### Change Metrics

```
ΔNMI = NMI(P_α, Y) - NMI(P₀, Y)
ΔARI = ARI(P_α, Y) - ARI(P₀, Y)
ΔQ = Q(G_sparse) - Q(G_orig)
```

Positive values indicate improvement after sparsification.

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RETENTIONS` | [0.8, 0.5] | DSpar retention values |
| `N_REPLICATES` | 10 | Trials per configuration |
| `SEED_BASE` | 42 | Random seed base |

---

## Output Structure

```
results/exp2_ground_truth/
├── ground_truth_raw.csv          # All trial data
├── ground_truth_summary.csv      # Aggregated by (dataset, alpha)
├── ground_truth_table.tex        # LaTeX table
├── plot_delta_nmi_vs_delta.pdf/png
└── plot_delta_ari_vs_delta.pdf/png
```

---

## Output Columns

### Raw CSV (`ground_truth_raw.csv`)

| Column | Description |
|--------|-------------|
| `dataset` | Dataset name |
| `n_nodes`, `n_edges` | Graph size |
| `n_ground_truth_communities` | Number of true communities |
| `alpha` | DSpar retention |
| `delta` | DSpar separation δ |
| `hub_bridge_ratio` | E[d_u·d_v \| inter] / E[d_u·d_v \| intra] |
| `nmi_orig`, `nmi_sparse` | NMI before/after |
| `ari_orig`, `ari_sparse` | ARI before/after |
| `delta_nmi`, `delta_ari` | Changes in metrics |
| `Q_orig`, `Q_sparse`, `delta_Q` | Leiden modularity |

### Summary CSV (`ground_truth_summary.csv`)

Aggregated mean/std by `(dataset, alpha)`.

### LaTeX Table

```latex
\input{results/exp2_ground_truth/ground_truth_table.tex}
```

---

## Expected Results

### DSpar-Favorable (δ > 0)

Datasets where inter-community edges connect higher-degree nodes:
- Positive ΔNMI and ΔARI expected
- Improvement validates DSpar mechanism

### DSpar-Neutral/Unfavorable (δ ≤ 0)

Datasets without hub-bridging property:
- ΔNMI and ΔARI near zero or negative
- No improvement expected (consistent with theory)

---

## Important Notes

1. **This is not a machine learning paper**
   - No model training
   - No hyperparameter tuning
   - Fixed Leiden parameters across all runs

2. **Structural explanation, not benchmark dominance**
   - We explain *when* DSpar works, not claim it always does
   - Negative results are informative and reported

3. **Reproducibility**
   - All seeds are fixed
   - Results should be deterministic

---

## Troubleshooting

### Dataset download fails

Check network connectivity. Datasets are downloaded from:
- Newman's network data repository
- SNAP Stanford

Manual download URLs are printed on failure.

### NMI/ARI near zero

This may indicate:
- Ground truth doesn't match network structure
- Too few communities for reliable metrics
- Consider using different ground truth if available

### Large variance across replicates

DSpar is stochastic. Increase `N_REPLICATES` for more stable estimates.
