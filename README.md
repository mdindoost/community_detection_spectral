# Community Detection with Spectral Graph Sparsification

This project investigates whether **spectral graph sparsification** improves community detection by removing noise while preserving community structure.

## Key Hypothesis

Spectral sparsification removes noisy inter-community edges faster than true intra-community edges, effectively **denoising** the graph and improving community detection accuracy.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/mdindoost/community_detection_spectral.git
cd community_detection_spectral
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./setup_julia.sh

# Run the DSpar vs Spectral demo
python experiments/dspar_demo.py
```

## Installation

### Python Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Required: `numpy`, `scipy`, `networkx`, `python-igraph`, `leidenalg`, `scikit-learn`, `matplotlib`

### Julia Setup

Julia is required for spectral sparsification (Spielman-Srivastava algorithm via `Laplacians.jl`).

```bash
./setup_julia.sh
```

This downloads Julia 1.10.2 and installs required packages.

## Core Files

### `experiments/dspar.py` - DSpar Implementation

Degree-based graph sparsification with three methods:

**DSpar Score Formula:**
```
s(e) = 1/d_u + 1/d_v
```
- Higher score = edge connects low-degree nodes (more important)
- Lower score = edge connects high-degree hubs (less important)

**Methods:**

| Method | Sampling | Output | `retention` Meaning |
|--------|----------|--------|---------------------|
| `"paper"` | WITH replacement | Weighted | Number of samples to draw |
| `"probabilistic_no_replace"` | WITHOUT replacement | Unweighted | Expected fraction kept |
| `"deterministic"` | Top-k selection | Unweighted | Exact fraction kept |

**Usage:**
```python
from experiments.dspar import dspar_sparsify

G_sparse = dspar_sparsify(G, retention=0.5, method="paper")
```

### `experiments/utils.py` - Utilities & Dataset Management

Comprehensive utility module (~1,450 lines) handling datasets, sparsification, and analysis.

**Dataset Loading** (100+ datasets):
```python
from experiments.utils import load_snap_dataset, SNAP_DATASETS

# Load any supported dataset (auto-downloads if needed)
edges, n_nodes, ground_truth = load_snap_dataset('cit-HepPh')

# Available: SNAP, citation (cora, citeseer, cit-HepPh, cit-HepTh),
# PPI (yeast, human), Facebook100, classic benchmarks (karate, dolphins, etc.)
print(list(SNAP_DATASETS.keys()))
```

**Spectral Sparsification** (Julia Laplacians.jl):
```python
from experiments.utils import spectral_sparsify_direct

sparsified_edges, elapsed = spectral_sparsify_direct(edges, n_nodes, epsilon)
```

**Edge Preservation Analysis**:
```python
from experiments.utils import analyze_edge_preservation, analyze_ground_truth_edge_preservation

# Analyze intra vs inter community edge preservation
stats = analyze_edge_preservation(original_edges, sparsified_edges, community_labels)
```

**LFR Benchmark Generation**:
```python
from experiments.utils import generate_lfr

edges, n_nodes, ground_truth = generate_lfr(
    n=1000, tau1=2.5, tau2=1.5, mu=0.3,
    average_degree=15, min_community=20, max_community=100, seed=42
)
```

**Other utilities**: `edges_to_adjacency()`, `adjacency_to_igraph()`, `random_sparsify()`, `add_noise_edges()`

### `experiments/dspar_demo.py` - Comparison Demo

Visualizes the difference between DSpar and Spectral sparsification on a two-cliques graph.

```bash
python experiments/dspar_demo.py
```

**Key Insight:** DSpar and Spectral sparsification are NOT equivalent:
- **DSpar** removes hub/bridge edges (high-degree nodes get low scores)
- **Spectral** keeps hub/bridge edges (high effective resistance)

Output: `experiments/dspar_vs_spectral.png`

### `experiments/cit_hepph_experiment.py` - Citation Network Experiment

Comprehensive comparison of DSpar vs Spectral on citation networks.

```bash
# Run on different datasets
python experiments/cit_hepph_experiment.py cit-HepPh
python experiments/cit_hepph_experiment.py cit-HepTh
python experiments/cit_hepph_experiment.py citeseer
```

**Metrics:**
- Edges kept (count and percentage)
- Connected components (CC)
- Communities found by Leiden
- Modularity (clustering quality)
- NMI/ARI (similarity to original clustering)
- Sparsification and Leiden timing
- CPM resolution analysis (0.1, 0.01, 0.001)

**Output columns:**
```
Method    Param    Edges    %    CC    Comm    Mod    NMI    ARI    Spar(s)    Leid(s)
```

## Testing

Comprehensive test suite with 67 tests covering functional correctness and mathematical verification.

```bash
# Run all tests
pytest tests/test_dspar.py -v

# Quick sanity check
python tests/test_dspar.py --quick

# Verbose correctness checks
python tests/test_dspar.py --verbose
```

**Test Coverage:**

| Category | Tests | What it Verifies |
|----------|-------|------------------|
| Score Computation | 7 | Formula `s(e) = 1/d_u + 1/d_v` |
| Basic Properties | 14 | Nodes preserved, no new edges, edge counts |
| Method-Specific | 13 | Paper/deterministic/probabilistic behavior |
| Edge Cases | 10 | Small graphs, invalid inputs |
| Exact Verification | 6 | Hand-computed values on tiny graphs |
| Weight Formula | 2 | `w = count / (q * p)` reconstruction |
| Spectral Properties | 4 | λ₂ preservation, effective resistance correlation |
| Statistical Rigor | 2 | Sampling distribution, chi-squared tests |
| Invariants | 2 | `sum(w * p) = 1`, score ordering |

## Parameters

### DSpar: `retention`
- Fraction of edges to sample/keep
- Example: `retention=0.5` with 100 edges → sample 50 edges

### Spectral: `epsilon`
- Approximation quality (smaller = more edges retained)
- Preserves all cuts within (1 ± epsilon) factor

| epsilon | Edges Retained |
|---------|----------------|
| 0.3 | ~80% |
| 0.5 | ~60% |
| 1.0 | ~40% |
| 2.0 | ~25% |

## DSpar Separation (δ) - Why DSpar Preserves Communities

### Overview

The DSpar separation metric δ measures whether DSpar sparsification will preserve community structure:

```
δ = μ_intra - μ_inter
```

Where:
- **μ_intra**: Mean DSpar score for intra-community edges (edges within communities)
- **μ_inter**: Mean DSpar score for inter-community edges (edges between communities)
- **DSpar score**: For edge (u,v), score = 1/d_u + 1/d_v

**Interpretation:**
- **δ > 0**: DSpar preserves community structure (good)
- **δ < 0**: DSpar damages community structure (bad)

### Step-by-Step Calculation

**Step 1: Obtain Community Assignments**

Run a community detection algorithm (e.g., Leiden) to assign each node to a community:

```python
membership = {node_id: community_id}
# Example: {A: 0, B: 0, C: 0, D: 1, E: 1, F: 1}
```

**Step 2: Classify Edges**

For each edge, check if both endpoints belong to the same community:

```python
for u, v in G.edges():
    score = 1/degree[u] + 1/degree[v]

    if membership[u] == membership[v]:
        intra_scores.append(score)   # Same community
    else:
        inter_scores.append(score)   # Different communities
```

**Step 3: Compute δ**

```python
μ_intra = mean(intra_scores)
μ_inter = mean(inter_scores)
δ = μ_intra - μ_inter
```

### Worked Example

Consider a graph with 6 nodes and 7 edges:

```
Community 0: {A, B, C}        Community 1: {D, E, F}

    A ─── B                       D ─── E
    │     │                       │     │
    C ─ ─ ┘                       F ─ ─ ┘
          \                      /
           \____________________/
               inter-edge (B-D)
```

**Edges:**
- Intra (Community 0): A-B, A-C, B-C
- Intra (Community 1): D-E, D-F, E-F
- Inter: B-D

**Degrees:**

| Node | Degree | Connections |
|------|--------|-------------|
| A | 2 | B, C |
| B | 3 | A, C, D (bridge) |
| C | 2 | A, B |
| D | 3 | B, E, F (bridge) |
| E | 2 | D, F |
| F | 2 | D, E |

**Intra-Community Scores:**

| Edge | d_u | d_v | Score = 1/d_u + 1/d_v |
|------|-----|-----|----------------------|
| A-B | 2 | 3 | 1/2 + 1/3 = 0.833 |
| A-C | 2 | 2 | 1/2 + 1/2 = 1.000 |
| B-C | 3 | 2 | 1/3 + 1/2 = 0.833 |
| D-E | 3 | 2 | 1/3 + 1/2 = 0.833 |
| D-F | 3 | 2 | 1/3 + 1/2 = 0.833 |
| E-F | 2 | 2 | 1/2 + 1/2 = 1.000 |

```
μ_intra = (0.833 + 1.000 + 0.833 + 0.833 + 0.833 + 1.000) / 6 = 0.889
```

**Inter-Community Scores:**

| Edge | d_u | d_v | Score = 1/d_u + 1/d_v |
|------|-----|-----|----------------------|
| B-D | 3 | 3 | 1/3 + 1/3 = 0.667 |

```
μ_inter = 0.667
```

**Final Result:**

```
δ = μ_intra - μ_inter = 0.889 - 0.667 = +0.222
```

### Why δ > 0 is Good

DSpar samples edges with probability proportional to `1/d_u + 1/d_v`.

In this example:
- **Intra-community edges** connect lower-degree nodes → higher scores (0.833 - 1.000)
- **Inter-community edge** connects high-degree bridge nodes → lower score (0.667)

Since δ > 0, DSpar will:
- Keep more intra-community edges (high scores)
- Remove more inter-community edges (low scores)
- **Result: Community structure is preserved**

This pattern occurs because inter-community edges often connect **hubs** (high-degree bridge nodes), while intra-community edges connect regular community members. This is known as the **hub-bridging property**.

### Computing δ for Real Datasets

Use `compute_hub_bridge_table.py` in `PAPER_EXPERIMENTS/`:

```bash
cd PAPER_EXPERIMENTS
python compute_hub_bridge_table.py
```

This computes δ for datasets and outputs results to `results/hub_bridge_exp3.csv`.

## References

- Spielman, D. A., & Srivastava, N. (2011). Graph sparsification by effective resistances. *SIAM Journal on Computing*.
- Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*.

## License

MIT
