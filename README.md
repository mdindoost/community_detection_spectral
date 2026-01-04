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

### `experiments/utils.py` - Julia Spectral Sparsification

Wrapper for Julia's `Laplacians.jl` implementing Spielman-Srivastava spectral sparsification.

**Key Function:**
```python
from experiments.utils import spectral_sparsify_direct

sparsified_edges, elapsed = spectral_sparsify_direct(edges, n_nodes, epsilon)
```

**Effective Resistance Formula:**
```
R(u,v) = L+[u,u] + L+[v,v] - 2*L+[u,v]
```
Where L+ is the Laplacian pseudoinverse. Higher ER = fewer alternate paths = more important edge.

### `experiments/dspar_demo.py` - Comparison Demo

Visualizes the difference between DSpar and Spectral sparsification on a two-cliques graph.

```bash
python experiments/dspar_demo.py
```

**Key Insight:** DSpar and Spectral sparsification are NOT equivalent:
- **DSpar** removes hub/bridge edges (high-degree nodes get low scores)
- **Spectral** keeps hub/bridge edges (high effective resistance)

Output: `experiments/dspar_vs_spectral.png`

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

## References

- Spielman, D. A., & Srivastava, N. (2011). Graph sparsification by effective resistances. *SIAM Journal on Computing*.
- Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*.

## License

MIT
