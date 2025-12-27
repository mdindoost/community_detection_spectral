# Community Detection with Spectral Graph Sparsification

This project investigates whether **spectral graph sparsification** preserves community structure in graphs. We use the Leiden algorithm for community detection and compare results on original vs. sparsified graphs.

## Hypothesis

Spectral sparsification (Spielman-Srivastava algorithm) preserves graph connectivity and effective resistances, which should maintain community structure even with significant edge reduction. In contrast, random edge removal destroys connectivity and degrades community detection quality.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/community_detection_spectral.git
cd community_detection_spectral

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up Julia (required for spectral sparsification)
./setup_julia.sh

# Run experiment on smallest dataset
python experiments/community_experiment.py --datasets email-Eu-core

# Run on all datasets
python experiments/community_experiment.py --datasets all
```

## Project Structure

```
community_detection_spectral/
├── experiments/
│   ├── community_experiment.py   # Main experiment script
│   └── utils.py                  # Utilities (download, sparsify, graph I/O)
├── sparsify_graph.jl             # Julia spectral sparsification
├── setup_julia.sh                # Julia setup script
├── requirements.txt              # Python dependencies
├── JuliaProject/                 # Julia project files
│   ├── Project.toml
│   └── Manifest.toml
├── datasets/                     # Downloaded datasets and sparsified graphs
└── results/                      # Experiment outputs (JSON results)
```

## Datasets

We use SNAP network datasets:

| Dataset | Nodes | Edges | Ground Truth |
|---------|-------|-------|--------------|
| email-Eu-core | ~1K | ~16K | Yes |
| wiki-Vote | ~7K | ~100K | No |
| ca-HepPh | ~12K | ~118K | No |
| soc-Epinions1 | ~76K | ~405K | No |
| com-DBLP | ~317K | ~1M | Yes |
| com-Amazon | ~335K | ~926K | Yes |
| com-Youtube | ~1.1M | ~3M | Yes |

Datasets are automatically downloaded on first run.

## Methods

### Spectral Sparsification

Uses Spielman-Srivastava algorithm via Julia's `Laplacians.jl`:
- Samples edges based on effective resistance
- Preserves spectral properties of the graph Laplacian
- Parameter `epsilon` controls sparsification level (smaller = more edges retained)

### Community Detection

Uses the Leiden algorithm (`leidenalg` library):
- Optimizes modularity
- Produces non-overlapping communities

### Evaluation Metrics

- **Modularity** (on original graph): How well the detected communities partition the original graph
- **NMI**: Normalized Mutual Information vs. ground truth
- **ARI**: Adjusted Rand Index vs. ground truth
- **Connected Components**: Graph connectivity after sparsification

## Usage

```bash
# Basic usage
python experiments/community_experiment.py --datasets email-Eu-core

# Multiple datasets
python experiments/community_experiment.py --datasets email-Eu-core wiki-Vote ca-HepPh

# All datasets
python experiments/community_experiment.py --datasets all

# Custom epsilon values
python experiments/community_experiment.py --epsilon 0.5 1.0 2.0 3.0

# Fewer random seeds (faster)
python experiments/community_experiment.py --num_seeds 3
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--datasets` | email-Eu-core | Datasets to run (or "all") |
| `--epsilon` | 1.0 2.0 | Sparsification epsilon values |
| `--num_seeds` | 5 | Seeds for random sparsification control |

## Output

Results are saved to `results/`:
- `results/{dataset}/results.json` - Per-dataset detailed results
- `results/summary.json` - Combined results across all datasets

Example output:
```
==========================================================================================
SUMMARY
==========================================================================================

Dataset              Config                       Edge%     CC      Mod      NMI      ARI
------------------------------------------------------------------------------------------
email-Eu-core        original                    100.0%     20   0.4174   0.5677   0.2820
email-Eu-core        spectral_eps1.0              83.7%     20   0.4168   0.5670   0.2773
email-Eu-core        spectral_eps2.0              35.2%     20   0.3926   0.6003   0.3858
email-Eu-core        random_match_eps1.0          83.7%     34   0.4144   0.5786   0.3189
email-Eu-core        random_match_eps2.0          35.2%    117   0.3809   0.5364   0.2566
------------------------------------------------------------------------------------------
```

## Requirements

### Python
- Python 3.8+
- numpy, scipy, networkx, igraph, leidenalg, scikit-learn

### Julia
- Julia 1.10+ (automatically installed by `setup_julia.sh`)
- Laplacians.jl package

## References

- Spielman, D. A., & Srivastava, N. (2011). Graph sparsification by effective resistances.
- Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities.

## License

MIT
