# Experiment 1.2: Theoretical Predictions Validation on Real Networks

## Overview

This experiment validates DSpar's theoretical predictions on real-world networks from the SNAP repository. Unlike Experiment 1.1 (synthetic networks with planted structure), this experiment tests whether the theory holds when community structure emerges organically.

### Key Hypothesis

DSpar preferentially removes inter-community edges, which should:
1. **Increase the F-term** (intra-community edge fraction)
2. **Decrease the G-term** (null-model penalty via degree reduction)
3. **Result in positive ΔQ** (modularity improvement)

### Theoretical Identity

Modularity decomposes as:
```
Q = F - G
```

Where:
- **F = (1/2m) × Σ_c e_c** — fraction of edges within communities
- **G = (1/4m²) × Σ_c vol_c²** — null-model penalty term

The change under sparsification satisfies:
```
ΔQ_fixed = ΔF_obs - ΔG_obs
```

This identity is verified to machine precision (~10⁻¹⁶) in every trial.

---

## File Structure

```
PAPER_EXPERIMENTS/
├── exp1_2_theoretical_predictions.py   # Core experiment script
├── run_exp1_2_all.py                   # Driver for all datasets
├── plot_exp1_2.py                      # Publication figures
├── generate_tables_exp1_2.py           # LaTeX tables
├── experiment_1_2.tex                  # LaTeX subsection scaffold
├── README_exp1_2.md                    # This file
└── results/
    └── exp1_2_theoretical/
        ├── <dataset>_theoretical_validation_FIXED.csv  # Raw trial data
        ├── <dataset>_summary_FIXED.csv                 # Aggregated stats
        ├── run_log.json                                # Execution log
        ├── figures/                                    # PNG + PDF plots
        └── tables/                                     # LaTeX tables
```

---

## Quick Start

### 1. Run All Experiments

```bash
cd PAPER_EXPERIMENTS
python run_exp1_2_all.py
```

This runs `exp1_2_theoretical_predictions.py` on all 12 datasets sequentially.

**Options:**
```bash
# Run specific datasets only
python run_exp1_2_all.py --datasets ca-AstroPh,email-Enron

# Dry run (list datasets without running)
python run_exp1_2_all.py --dry-run

# Quiet mode (suppress per-trial output)
python run_exp1_2_all.py --quiet
```

### 2. Generate Figures

```bash
python plot_exp1_2.py
```

**Options:**
```bash
# Generate combined multi-panel figure
python plot_exp1_2.py --combined

# Specific datasets
python plot_exp1_2.py --datasets ca-AstroPh,email-Enron

# Custom output directory
python plot_exp1_2.py --output-dir ./my_figures
```

### 3. Generate LaTeX Tables

```bash
python generate_tables_exp1_2.py
```

**Options:**
```bash
# Different retention value (default: 0.8)
python generate_tables_exp1_2.py --retention 0.7

# Custom output directory
python generate_tables_exp1_2.py --output-dir ./my_tables
```

### 4. Run Single Dataset (for debugging)

```bash
python exp1_2_theoretical_predictions.py ca-AstroPh
```

---

## Datasets

The experiment uses 12 SNAP networks:

| Dataset | Type | Description |
|---------|------|-------------|
| ca-AstroPh | Collaboration | Astrophysics co-authorship |
| ca-CondMat | Collaboration | Condensed matter co-authorship |
| ca-GrQc | Collaboration | General relativity co-authorship |
| ca-HepPh | Collaboration | High-energy physics co-authorship |
| ca-HepTh | Collaboration | High-energy physics theory co-authorship |
| cit-HepPh | Citation | High-energy physics citations |
| cit-HepTh | Citation | High-energy physics theory citations |
| email-Enron | Communication | Enron email network |
| email-Eu-core | Communication | EU research institution emails |
| facebook-combined | Social | Facebook ego networks combined |
| ego-Facebook | Social | Facebook ego network |
| wiki-Vote | Voting | Wikipedia admin elections |

**Adding new datasets:**

1. Add entry to `experiments/utils.py` in `SNAP_DATASETS`:
```python
'my-dataset': (
    'https://snap.stanford.edu/data/my-dataset.txt.gz',
    False,  # directed=False for undirected
    None    # skip_rows=None (or integer if header lines exist)
),
```

2. Add to `DEFAULT_DATASETS` in `run_exp1_2_all.py` and `plot_exp1_2.py`

---

## Code Details

### exp1_2_theoretical_predictions.py

The core experiment script. Key components:

#### DSpar Score
```python
# Edge score: s(e) = 1/d_u + 1/d_v
# Higher score = more likely to be removed
# Hub edges (high degree endpoints) get LOW scores → retained
# Peripheral edges get HIGH scores → removed
```

#### Two Modularity Metrics

1. **Fixed-membership (theory-aligned)**
   ```python
   # Use partition from ORIGINAL graph
   membership_fixed = {node: comm for node, comm in zip(G_original.vs['name'], partition_orig.membership)}

   # Compute modularity on SPARSIFIED graph with FIXED membership
   Q_sparse_fixed = compute_modularity_fixed(G_sparse, membership_fixed)
   ```

2. **Leiden-reoptimized (pipeline performance)**
   ```python
   # Re-run Leiden on sparsified graph
   partition_sparse = G_sparse.community_leiden(objective_function='modularity')
   Q_sparse_leiden = partition_sparse.modularity
   ```

#### G-Term Computation
```python
def compute_G_term_unweighted(G: nx.Graph, membership):
    """
    G(G) = Σ_c vol_c² / (4m²)

    Where vol_c = sum of degrees of nodes in community c
    """
    m = G.number_of_edges()
    degrees = dict(G.degree())

    # Compute volume per community
    vol = defaultdict(float)
    for node, deg in degrees.items():
        vol[membership[node]] += deg

    # G = sum(vol_c²) / (4m²)
    G_val = sum(v * v for v in vol.values()) / (4.0 * m * m)
    return G_val
```

#### Reconstruction Verification
```python
# Verify: ΔQ = ΔF - ΔG
dQ_reconstructed = dF_observed - dG_observed
error = abs(dQ_reconstructed - modularity_fixed_change)
# Error should be ~1e-15 (machine precision)
```

### run_exp1_2_all.py

Driver script that:
1. Iterates over all datasets
2. Calls `exp1_2_theoretical_predictions.py` as subprocess
3. Collects success/failure status
4. Writes `run_log.json` with timing info

### plot_exp1_2.py

Generates two figures per dataset:

**Figure A: Modularity Improvement**
- X-axis: Retention α
- Y-axis: ΔQ
- Blue line: ΔQ_fixed (theory-aligned)
- Orange line: ΔQ_Leiden (pipeline)

**Figure B: Modularity Decomposition**
- X-axis: Retention α
- Y-axis: Contribution to ΔQ
- Blue line: ΔQ_fixed (total)
- Green line: -ΔG (null-model relief, positive = good)

### generate_tables_exp1_2.py

Generates three LaTeX tables:

1. **table1_modularity_changes.tex**: Dataset stats + ΔQ at α=0.8
2. **table2_decomposition.tex**: ΔQ = ΔF - ΔG verification
3. **table3_summary.tex**: Aggregate statistics

---

## Output Files

### Raw Trial Data (`*_theoretical_validation_FIXED.csv`)

Each row = one trial. Key columns:

| Column | Description |
|--------|-------------|
| `retention` | Retention level α |
| `seed` | Random seed for trial |
| `m` | Number of edges in original graph |
| `modularity_fixed_original` | Q on original graph (fixed membership) |
| `modularity_fixed_sparse` | Q on sparse graph (fixed membership) |
| `modularity_leiden_sparse` | Q on sparse graph (Leiden re-optimized) |
| `modularity_fixed_change` | ΔQ_fixed = Q_sparse - Q_original |
| `modularity_leiden_change` | ΔQ_Leiden |
| `F_original`, `F_observed` | F-term values (original and sparse) |
| `F_improvement_observed` | ΔF = F_sparse - F_original |
| `G_original`, `G_sparse_observed` | G-term values |
| `dG_observed` | ΔG = G_sparse - G_original |
| `dQ_reconstructed` | ΔF - ΔG (equals modularity_fixed_change to machine precision) |

### Summary Data (`*_summary_FIXED.csv`)

Aggregated over 10 trials per retention level:

| Column | Description |
|--------|-------------|
| `retention` | Retention level α |
| `modularity_fixed_change_mean` | Mean ΔQ_fixed |
| `modularity_fixed_change_std` | Std dev ΔQ_fixed |
| `modularity_leiden_change_mean` | Mean ΔQ_Leiden |
| `modularity_leiden_change_std` | Std dev ΔQ_Leiden |
| `dG_observed_mean`, `dG_observed_std` | ΔG statistics |
| ... | Other metrics |

---

## Modifying the Experiment

### Change Retention Levels

In `exp1_2_theoretical_predictions.py`, modify:
```python
RETENTIONS = [0.5, 0.6, 0.7, 0.8, 0.9]  # Add/remove values
```

### Change Number of Trials

```python
N_TRIALS = 10  # Increase for more statistical power
```

### Change Sparsification Method

In `run_single_trial()`:
```python
# Current: probabilistic sampling without replacement
G_sparse = dspar_sparsify(G, alpha=retention, method='probabilistic_no_replace')

# Alternative: paper method (sampling with replacement, different semantics)
# G_sparse = dspar_sparsify(G, alpha=retention, method='paper')
```

### Change Community Detection

Currently uses igraph's built-in Leiden:
```python
partition = G_ig.community_leiden(objective_function='modularity', resolution=1.0)
```

To use different algorithm:
```python
# Louvain
partition = G_ig.community_multilevel()

# Label propagation
partition = G_ig.community_label_propagation()

# Infomap
partition = G_ig.community_infomap()
```

### Add New Metrics

In `run_single_trial()`, add to the output dict:
```python
out["my_new_metric"] = compute_my_metric(G_sparse, membership_fixed)
```

Then update the aggregation in `main()`:
```python
agg_cols = [..., "my_new_metric"]
```

---

## Troubleshooting

### "Dataset not found"

The dataset may not be in `experiments/utils.py`. Add it:
```python
SNAP_DATASETS = {
    ...
    'my-dataset': ('https://snap.stanford.edu/data/...', False, None),
}
```

### "Import error: experiments.dspar"

Run from the correct directory:
```bash
cd /path/to/community_detection_spectral/PAPER_EXPERIMENTS
python exp1_2_theoretical_predictions.py ca-AstroPh
```

Or add to PYTHONPATH:
```bash
export PYTHONPATH=/path/to/community_detection_spectral:$PYTHONPATH
```

### "Reconstruction error too large"

If `|ΔQ_reconstructed - ΔQ_fixed| > 1e-10`, there's a bug. Check:
1. Both use the same `membership_fixed`
2. Both use the same graph `G_sparse`
3. F and G terms are computed correctly

### Memory issues on large graphs

For very large graphs, reduce trials:
```bash
# Edit exp1_2_theoretical_predictions.py
N_TRIALS = 3  # Fewer trials
```

Or run datasets individually:
```bash
python exp1_2_theoretical_predictions.py email-Enron
```

---

## Interpretation Guide

### What does ΔQ_fixed > 0 mean?

DSpar is removing edges that hurt modularity (likely inter-community edges). The theory is validated.

### What does ΔQ_fixed < 0 mean?

DSpar is removing edges that help modularity. This could happen if:
- The network has unusual degree-community structure
- High-degree nodes are central to communities (not bridges)
- The planted community structure differs from degree patterns

### Why is ΔQ_Leiden sometimes different from ΔQ_fixed?

- **ΔQ_fixed**: Measures mechanism (same partition, different graph)
- **ΔQ_Leiden**: Measures outcome (different partition, different graph)

Leiden re-optimizes and may find a better partition on the sparser graph, or may fragment due to missing edges.

### What does -ΔG > 0 mean?

The null-model penalty decreased. DSpar removed edges from high-degree nodes, reducing the expected random edge count that modularity penalizes.

---

## Citation

If using this experiment, cite the DSpar paper and SNAP datasets:
```bibtex
@misc{snapnets,
  author = {Jure Leskovec and Andrej Krevl},
  title = {{SNAP Datasets}: {Stanford} Large Network Dataset Collection},
  howpublished = {\url{http://snap.stanford.edu/data}},
  month = jun,
  year = 2014
}
```
