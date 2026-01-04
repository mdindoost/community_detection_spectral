# Community Detection with Spectral Graph Sparsification

This project investigates whether **spectral graph sparsification** improves community detection by removing noise while preserving community structure.

## Key Hypothesis

Spectral sparsification removes noisy inter-community edges faster than true intra-community edges, effectively **denoising** the graph and improving community detection accuracy.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mdindoost/community_detection_spectral.git
cd community_detection_spectral

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up Julia (required for spectral sparsification)
./setup_julia.sh

# Run experiment on a small dataset
python experiments/community_experiment.py --datasets email-Eu-core

# View results
cat results_report.txt
```

## Installation

### Python Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Required packages: `numpy`, `scipy`, `networkx`, `python-igraph`, `leidenalg`, `scikit-learn`

### Julia Setup

Julia is required for spectral sparsification (Spielman-Srivastava algorithm via `Laplacians.jl`).

**Option 1: Automatic installation**
```bash
./setup_julia.sh
```
This downloads Julia 1.10.2 and installs required packages.

**Option 2: Use existing Julia installation (HPC)**
```bash
# If Julia is already installed elsewhere, create symlinks:
ln -s /path/to/julia-1.10.2 julia-1.10.2
ln -s /path/to/julia_depot julia_depot

# Install Julia packages
export JULIA_DEPOT_PATH=$(pwd)/julia_depot
julia --project=JuliaProject -e 'using Pkg; Pkg.instantiate()'
```

**Option 3: Module system (HPC)**
```bash
module load julia/1.10.2
export JULIA_DEPOT_PATH=$(pwd)/julia_depot
julia --project=JuliaProject -e 'using Pkg; Pkg.instantiate()'
```

## Project Structure

```
community_detection_spectral/
├── experiments/
│   ├── community_experiment.py   # SNAP dataset experiments
│   ├── lfr_experiment.py         # LFR benchmark experiments
│   ├── noisy_lfr_experiment.py   # Noisy LFR experiments
│   ├── cluster_sparsification.py # Cluster subgraph analysis
│   ├── test_networks.py          # Synthetic test networks
│   └── utils.py                  # Shared utilities
├── sparsify_graph.jl             # Julia spectral sparsification
├── gather_results.py             # Aggregate results into report
├── run_all_parallel.sh           # Run all SNAP datasets
├── run_lfr.sh                    # Run LFR benchmarks
├── run_noisy_lfr.sh              # Run noisy LFR experiments
├── run_new_datasets.sh           # Run new datasets (citation, PPI)
├── run_social_networks.sh        # Run social network datasets
├── JuliaProject/                 # Julia dependencies
├── datasets/                     # Downloaded data (auto-created)
└── results/                      # Experiment outputs (auto-created)
```

## Experiments

### 1. Real-World Networks (SNAP)

```bash
# Single dataset
python experiments/community_experiment.py --datasets email-Eu-core

# Multiple datasets
python experiments/community_experiment.py --datasets email-Eu-core wiki-Vote com-DBLP

# All original datasets (parallel)
./run_all_parallel.sh

# New datasets (citation + PPI networks)
./run_new_datasets.sh

# Social networks (Facebook100 + Pokec + Enron)
./run_social_networks.sh
```

**Available Datasets:**

| Dataset | Nodes | Edges | Ground Truth | Memory |
|---------|-------|-------|--------------|--------|
| email-Eu-core | 1K | 16K | Yes | Low |
| wiki-Vote | 7K | 100K | No | Low |
| ca-HepPh | 12K | 118K | No | Low |
| cora | 2.7K | 5.4K | Yes | Low |
| citeseer | 3.3K | 4.7K | Yes | Low |
| yeast-ppi | 2K | 7K | No | Low |
| human-ppi | 4K | 86K | No | Low |
| facebook-combined | 4K | 88K | No | Low |
| Rice31 | 4K | 184K | No | Low |
| Texas80 | 36K | 1.6M | No | Medium |
| Penn94 | 42K | 1.4M | No | Medium |
| email-Enron | 37K | 184K | No | Low |
| soc-Epinions1 | 76K | 405K | No | Medium |
| com-DBLP | 317K | 1M | Yes | Medium |
| com-Amazon | 335K | 926K | Yes | Medium |
| com-Youtube | 1.1M | 3M | Yes | High |
| soc-Pokec | 1.6M | 30M | No | High |
| com-LiveJournal | 4M | 34M | Yes | Very High |
| com-Orkut | 3M | 117M | Yes | Very High |

**Note:** Large datasets (soc-Pokec, com-LiveJournal, com-Orkut) require 32GB+ RAM for spectral sparsification.

### 2. LFR Benchmarks (Synthetic)

Tests how sparsification affects detection across varying community strength (μ) and graph density (k_avg).

```bash
# Quick run (n=1000 nodes)
python experiments/lfr_experiment.py --n 1000 --repeats 5

# Validation (n=5000 nodes)
python experiments/lfr_experiment.py --n 5000 --repeats 3

# Or use the script
./run_lfr.sh
```

**Parameters:**
- μ (mixing): 0.1, 0.3, 0.5, 0.7 (higher = weaker communities)
- k_avg (degree): 15, 25, 50
- ε (sparsification): 0.5, 1.0, 2.0, 3.0, 4.0

### 3. Noisy LFR (Noise Removal Hypothesis)

Tests whether adding noise to clean LFR graphs makes sparsification beneficial (like real data).

```bash
python experiments/noisy_lfr_experiment.py --n 1000 --repeats 5

# Or use the script
./run_noisy_lfr.sh
```

**Noise levels:** 0%, 10%, 20%, 30%, 50%

**Expected result:** Sparsification hurts clean graphs but helps noisy graphs.

### 4. Cluster Subgraph Sparsification Analysis

Analyzes how spectral sparsification affects individual clusters/communities. Loads a pre-clustered graph, extracts each cluster as a subgraph, and examines sparsification impact on connectivity.

```bash
python experiments/cluster_sparsification.py
```

**Analysis includes:**
- **Min-cut analysis:** Finds minimum edge cut before/after sparsification
- **Well-Connected Clusters (WCC):** Identifies clusters where min_cut > log(n)
- **Effective Resistance:** Shows ER of min-cut edges to explain why some get removed
- **Degree-1 nodes:** Tracks nodes that become leaves after sparsification
- **WCC preservation:** Checks if well-connected status is maintained

**Key insight:** Spectral sparsification preserves cuts within (1±ε) factor, but individual min-cut edges may be removed if they have low effective resistance (i.e., parallel paths exist).

### 5. Synthetic Test Networks

Visualizes effective resistance and sparsification behavior on simple graph structures.

```bash
python experiments/test_networks.py
```

**Test graphs:**
- Two triangles with bridge (tests bridge preservation)
- Two stars with 3 bridges (tests parallel path behavior)
- K4 with tail (tests clique vs chain ER differences)

## DSpar: Degree-Based Graph Sparsification

DSpar is an alternative sparsification method based on edge degree scores rather than effective resistance.

### DSpar Score Formula

```
s(e) = 1/d_u + 1/d_v
```

- **Higher score** = edge connects low-degree nodes (more important to keep)
- **Lower score** = edge connects high-degree hubs (less important)

### Three Methods Available

| Method | Sampling | Output | Retention Meaning | Edges Kept |
|--------|----------|--------|-------------------|------------|
| `"paper"` | WITH replacement | Weighted | Samples to draw | < retention% |
| `"probabilistic_no_replace"` | WITHOUT replacement | Unweighted | Expected fraction | ≈ retention% |
| `"deterministic"` | Top-k selection | Unweighted | Exact fraction | = retention% |

### Usage

```python
from experiments.dspar import dspar_sparsify

# Original paper method (best for spectral properties)
G_sparse = dspar_sparsify(G, retention=0.75, method="paper")

# Probabilistic without replacement
G_sparse = dspar_sparsify(G, retention=0.75, method="probabilistic_no_replace")

# Deterministic top-k (reproducible)
G_sparse = dspar_sparsify(G, retention=0.75, method="deterministic")
```

### Method Comparison Results

Tested on Karate Club graph (34 nodes, 78 edges) with `retention=0.75`:

**Single Run (target: 59 edges)**

| Method | Edges Kept | Percentage | Notes |
|--------|------------|------------|-------|
| Paper | 42 | 53.8% | Weighted, duplicates merged |
| Probabilistic | 60 | 76.9% | Unweighted |
| Deterministic | 59 | 75.6% | Exactly as specified |

**Variability (20 runs)**

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Paper (WITH replace) | 40.6 | 2.4 | 36 | 44 |
| Probabilistic (NO replace) | 57.6 | 3.2 | 52 | 66 |
| Deterministic | 59.0 | 0.0 | 59 | 59 |

**Key Observations:**

1. **Paper method** keeps fewer edges due to sampling with replacement (duplicates get merged into higher weights)
2. **Probabilistic** varies around the target retention
3. **Deterministic** is exactly reproducible with zero variance
4. All methods share ~34 core high-score edges

### When to Use Each Method

- **Paper**: Theoretical guarantees, spectral property preservation
- **Probabilistic**: Balance of randomness and target size
- **Deterministic**: Reproducibility, benchmarking

## Metrics

| Metric | Description |
|--------|-------------|
| **Modularity** | Quality of detected communities on original graph |
| **NMI** | Normalized Mutual Information vs ground truth |
| **ARI** | Adjusted Rand Index vs ground truth |
| **CommSim** | Similarity between Leiden on original vs sparse |
| **GT_Mod** | Ground truth community modularity |
| **GT_Ratio** | Inter/intra edge preservation ratio (<1 = denoising) |
| **Misclass%** | Removed edges that were misclassified by Leiden |

## Output

```bash
# Generate combined report
python gather_results.py

# View report
cat results_report.txt
```

Results are saved to:
- `results/{dataset}/results.json` - Per-dataset detailed results
- `results/lfr/` - LFR benchmark results
- `results/noisy_lfr/` - Noisy LFR results
- `results_report.txt` - Human-readable summary

## HPC Usage

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=sparsify
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

cd /path/to/community_detection_spectral
source venv/bin/activate

# Run experiments
python experiments/community_experiment.py --datasets com-Youtube com-DBLP
python gather_results.py
```

### Memory Requirements

| Dataset Size | Recommended RAM |
|--------------|-----------------|
| Small (<100K edges) | 4GB |
| Medium (<1M edges) | 16GB |
| Large (<10M edges) | 32GB |
| Very Large (>10M edges) | 64GB+ |

## Command Line Options

### community_experiment.py
```bash
--datasets DATASET [DATASET ...]  # Datasets to run (or "all")
--epsilon EPSILON [EPSILON ...]   # Sparsification epsilon values
```

### lfr_experiment.py
```bash
--n N              # Number of nodes (default: 1000)
--repeats R        # Repetitions per config (default: 5)
--mu MU [MU ...]   # Mixing parameter values
--k_avg K [K ...]  # Average degree values
```

### noisy_lfr_experiment.py
```bash
--n N                    # Number of nodes (default: 1000)
--repeats R              # Repetitions per config (default: 5)
--noise RATIO [RATIO ...]  # Noise ratios to test
```

## References

- Spielman, D. A., & Srivastava, N. (2011). Graph sparsification by effective resistances. *SIAM Journal on Computing*.
- Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*.
- Lancichinetti, A., Fortunato, S., & Radicchi, F. (2008). Benchmark graphs for testing community detection algorithms. *Physical Review E*.

## License

MIT
