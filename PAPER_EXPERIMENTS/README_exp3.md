# Experiment 3: Scalability and Large-Graph Tradeoffs

## Overview

This experiment evaluates runtime scalability, end-to-end speedup, and community quality tradeoffs on large real-world graphs when using DSpar sparsification compared to strong baselines.

### Key Questions

1. **Scalability**: How does sparsification time scale with graph size?
2. **Speedup**: Does sparsifying before Leiden reduce total pipeline time?
3. **Quality**: Does DSpar maintain or improve modularity vs baselines?
4. **Tradeoff**: What is the speedup-vs-quality Pareto frontier?

### Methods Compared

| Method | Description | Complexity |
|--------|-------------|------------|
| **DSpar** | Sample edges with P ∝ 1/d_u + 1/d_v (paper method) | O(m) |
| **Uniform Random** | Sample edges uniformly without replacement | O(m) |
| **Degree Sampling** | Sample edges with P ∝ deg(u) + deg(v) | O(m) |
| **Spectral** (default) | Effective resistance sampling (Julia Laplacians.jl) | O(m log² n / ε²) |

---

## File Structure

```
PAPER_EXPERIMENTS/
├── exp3_scalability.py              # Main experiment script
├── README_exp3.md                   # This file
└── results/
    └── exp3_scalability/
        ├── scalability_raw.csv              # All trial data
        ├── scalability_summary.csv          # Aggregated statistics
        ├── scalability_table_alpha0.8.tex   # LaTeX table
        └── figures/
            ├── plot1_scaling_sparsify_time.pdf/png
            ├── plot2_scaling_pipeline_time.pdf/png
            ├── plot3_quality_vs_alpha_{dataset}.pdf/png
            └── plot4_speedup_vs_quality_{dataset}.pdf/png
```

---

## Quick Start

### Basic Run (Single Dataset)

```bash
cd PAPER_EXPERIMENTS
python exp3_scalability.py --datasets com-DBLP
```

### Full Run (All Datasets)

```bash
python exp3_scalability.py
```

### Without Spectral Sparsification

```bash
python exp3_scalability.py --datasets com-DBLP --no_spectral
```

### Dry Run (Show Config Only)

```bash
python exp3_scalability.py --dry_run
```

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | com-DBLP,com-Amazon,com-Youtube,com-LiveJournal | Comma-separated list |
| `--max_edges` | None | Subsample edges for testing |
| `--replicates` | 3 | Number of replicates per config |
| `--dry_run` | False | Print config and exit |
| `--no_spectral` | False | Exclude spectral sparsification (spectral is included by default) |
| `--spectral_multiplier` | 10.0 | Spectral timeout = multiplier × max(DSpar time) |
| `--spectral_timeout` | 300 | Initial timeout (overridden by dynamic) |

---

## Datasets

Default datasets from SNAP (large real-world graphs):

| Dataset | Nodes | Edges | Type |
|---------|-------|-------|------|
| com-DBLP | ~317K | ~1M | Collaboration |
| com-Amazon | ~335K | ~926K | Co-purchase |
| com-Youtube | ~1.1M | ~3M | Social |
| com-LiveJournal | ~4M | ~35M | Social |

**Adding new datasets:**

Add to `experiments/utils.py` in `SNAP_DATASETS`, then add to `DEFAULT_DATASETS` in `exp3_scalability.py`.

---

## Metrics

### Timing Metrics

| Metric | Description |
|--------|-------------|
| `T_sparsify_sec` | Time to sparsify the graph |
| `T_leiden_orig_sec` | Leiden time on original graph |
| `T_leiden_sparse_sec` | Leiden time on sparsified graph |
| `T_pipeline_sec` | T_sparsify + T_leiden_sparse |
| `speedup` | T_leiden_orig / T_pipeline |

### Quality Metrics

| Metric | Description |
|--------|-------------|
| `Q0` | Baseline modularity (Leiden on original) |
| `dQ_fixed` | Modularity change with fixed partition |
| `dQ_leiden` | Modularity change with Leiden re-optimization |
| `nmi_P0_Palpha` | NMI between original and sparse partitions |

---

## Experiment Protocol

For each dataset:

1. **Load graph** as undirected, simple (remove self-loops, collapse multi-edges)
2. **Run Leiden on original** to get baseline partition P₀, modularity Q₀, time T_orig
3. **For each method** (dspar, uniform_random, degree_sampling, [spectral]):
   - **For each α** ∈ {0.2, 0.4, 0.6, 0.8, 1.0}:
     - **For each replicate** (3 trials):
       - Sparsify graph, record T_sparsify
       - Compute Q_fixed (modularity of P₀ on sparse graph)
       - Run Leiden on sparse graph, record T_leiden_sparse, Q_leiden
       - Compute speedup, ΔQ_fixed, ΔQ_leiden

### Spectral Dynamic Timeout

Spectral sparsification is included by default (use `--no_spectral` to exclude). The dynamic timeout mechanism:

1. **Phase 1**: Run DSpar, uniform_random, degree_sampling
2. **Measure**: max(DSpar sparsification time)
3. **Set timeout**: spectral_timeout = 10 × max(DSpar time)
4. **Phase 2**: Run spectral with dynamic timeout

This ensures spectral gets a fair chance (10× longer than DSpar) but doesn't wait forever on large graphs.

---

## Output Files

### Raw CSV (`scalability_raw.csv`)

One row per experiment:

```
dataset, method, alpha, replicate, seed,
n_nodes, m_edges, m_sparse, retention_actual,
T_sparsify_sec, T_leiden_orig_sec, T_leiden_sparse_sec, T_pipeline_sec, speedup,
Q0, Q_sparse_fixed, dQ_fixed, Q_sparse_leiden, dQ_leiden,
n_communities_orig, n_communities_sparse, nmi_P0_Palpha
```

### Summary CSV (`scalability_summary.csv`)

Aggregated by (dataset, method, alpha):

```
dataset, method, alpha,
dQ_fixed_mean, dQ_fixed_std,
dQ_leiden_mean, dQ_leiden_std,
T_sparsify_sec_mean, T_sparsify_sec_std,
T_leiden_sparse_sec_mean, T_leiden_sparse_sec_std,
T_pipeline_sec_mean, T_pipeline_sec_std,
speedup_mean, speedup_std,
T_leiden_orig_sec, Q0
```

### LaTeX Table (`scalability_table_alpha0.8.tex`)

Results at α = 0.8:

| Dataset | Method | n | m | T_spar | T_Leiden | Speedup | ΔQ_fixed | ΔQ_Leiden |
|---------|--------|---|---|--------|----------|---------|----------|-----------|

---

## Plots

### Plot 1: Sparsification Time Scaling

- **X-axis**: Number of edges m (log scale)
- **Y-axis**: Sparsification time (log scale)
- **Curves**: One per method
- **File**: `plot1_scaling_sparsify_time.pdf/png`

### Plot 2: Pipeline Time Scaling

- **X-axis**: Number of edges m (log scale)
- **Y-axis**: Pipeline time (log scale)
- **Curves**: Methods + baseline (Leiden on original)
- **File**: `plot2_scaling_pipeline_time.pdf/png`

### Plot 3: Quality vs Retention (Per Dataset)

- **X-axis**: Retention α
- **Y-axis**: ΔQ_leiden
- **Curves**: One per method
- **File**: `plot3_quality_vs_alpha_{dataset}.pdf/png`

### Plot 4: Speedup vs Quality (Per Dataset)

- **X-axis**: Speedup
- **Y-axis**: ΔQ_leiden
- **Points**: Each α level, connected per method
- **File**: `plot4_speedup_vs_quality_{dataset}.pdf/png`

---

## Code Details

### DSpar Sparsification

```python
def sparsify_dspar(G: nx.Graph, alpha: float, seed: int) -> nx.Graph:
    """
    DSpar paper method: sampling WITH replacement, returns weighted graph.
    We convert to unweighted for Leiden (keep only topology).
    """
    G_sparse_weighted = dspar_sparsify(G, retention=alpha, method='paper', seed=seed)
    # Convert to unweighted
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G_sparse_weighted.nodes())
    G_sparse.add_edges_from(G_sparse_weighted.edges())
    return G_sparse
```

### Baseline Methods

```python
# Uniform random: each edge kept with probability α
def sparsify_uniform_random(G, alpha, seed):
    keep_indices = np.random.choice(m, size=int(alpha * m), replace=False)
    # ...

# Degree sampling: P(keep edge) ∝ deg(u) + deg(v)
def sparsify_degree_sampling(G, alpha, seed):
    weights = [degrees[u] + degrees[v] for u, v in edges]
    probs = weights / sum(weights) * (alpha * m)
    # ...
```

### Spectral Sparsification

```python
def sparsify_spectral(G, alpha, seed):
    """
    Uses Julia Laplacians.jl for effective resistance sampling.
    Maps alpha to epsilon (approximation quality).
    Returns None on timeout/error.
    """
    epsilon = SPECTRAL_EPSILON_MAP[alpha]  # e.g., 0.8 → 0.3
    sparsified_edges, _ = spectral_sparsify_direct(edges, n_nodes, epsilon)
    # ...
```

### Leiden Clustering

```python
def run_leiden(G: nx.Graph):
    """Uses igraph's built-in Leiden implementation."""
    ig_graph = nx_to_igraph(G)
    partition = ig_graph.community_leiden(
        objective_function='modularity',
        resolution=1.0,
        n_iterations=2
    )
    return membership, partition.modularity, runtime
```

---

## Modifying the Experiment

### Change Retention Levels

```python
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]  # Add/remove values
```

### Change Number of Replicates

```bash
python exp3_scalability.py --replicates 5
```

### Change Spectral Epsilon Mapping

```python
SPECTRAL_EPSILON_MAP = {
    0.2: 3.0,   # Very aggressive
    0.4: 1.5,   # Aggressive
    0.6: 0.8,   # Moderate
    0.8: 0.3,   # Conservative
    1.0: 0.0,   # Skip (no sparsification)
}
```

### Add New Sparsification Method

1. Add function:
```python
def sparsify_my_method(G, alpha, seed):
    # ... implementation ...
    return G_sparse
```

2. Add to dispatcher:
```python
def sparsify(G, method, alpha, seed):
    # ...
    elif method == 'my_method':
        return sparsify_my_method(G, alpha, seed)
```

3. Add to METHODS list:
```python
METHODS = ['dspar', 'uniform_random', 'degree_sampling', 'my_method']
```

4. Add color/marker/linestyle:
```python
COLORS['my_method'] = '#FF0000'
MARKERS['my_method'] = 'v'
LINESTYLES['my_method'] = '-'
```

---

## Troubleshooting

### "Dataset not found"

Check that the dataset is in `experiments/utils.py` `SNAP_DATASETS` dict.

### "Spectral sparsification not available"

Julia is not configured. See `experiments/setup_julia.sh`.

### Spectral times out on large graphs

This is expected. Spectral has O(m log² n / ε²) complexity. Options:

1. Increase timeout: `--spectral_multiplier 20`
2. Skip spectral: use `--no_spectral`
3. Subsample graph: `--max_edges 500000`

### Memory issues

Large graphs (>10M edges) may require significant RAM. Options:

1. Subsample: `--max_edges 1000000`
2. Run datasets individually: `--datasets com-DBLP`
3. Reduce replicates: `--replicates 1`

### Import errors

Run from correct directory:
```bash
cd /path/to/community_detection_spectral/PAPER_EXPERIMENTS
python exp3_scalability.py
```

---

## Interpretation Guide

### What does speedup > 1 mean?

The sparsify + Leiden(sparse) pipeline is faster than Leiden(original).

### When is DSpar beneficial?

DSpar is beneficial when:
- **Speedup > 1**: Pipeline is faster
- **ΔQ_leiden ≥ 0**: Quality maintained or improved

### Why might ΔQ_leiden < 0?

Removing too many edges (low α) can fragment the community structure, leading to worse Leiden results.

### What does Plot 4 show?

The speedup-vs-quality Pareto frontier. Points in the upper-right (high speedup, positive ΔQ) are best. DSpar should dominate baselines in this region.

---

## Expected Results

### Small Graphs (~1M edges)

- DSpar: ~1-2s sparsification, 1.5-2× speedup
- Uniform: Similar timing, worse quality
- Spectral: ~10-20s, may have better connectivity

### Large Graphs (~35M edges)

- DSpar: ~30s sparsification, 2-4× speedup
- Uniform: Similar timing
- Spectral: Likely timeout (>300s)

### Quality Comparison

At α = 0.8:
- DSpar: ΔQ_leiden typically +0.001 to +0.01
- Uniform: ΔQ_leiden typically -0.01 to 0
- Degree: ΔQ_leiden typically -0.005 to +0.005

---

## Citation

If using this experiment, cite the SNAP datasets:

```bibtex
@misc{snapnets,
  author = {Jure Leskovec and Andrej Krevl},
  title = {{SNAP Datasets}: {Stanford} Large Network Dataset Collection},
  howpublished = {\url{http://snap.stanford.edu/data}},
  year = 2014
}
```
