# Exp P — algorithm generality (pre-registered design)

Registered 2026-07-25, BEFORE any Exp P code or run existed. Greenlit by Mohammad ("go for P").

**Claim it could change:** every verdict in this paper (C5, C6, C9, C16) is Leiden/modularity-
only, while Satuluri 2011's claim covered several algorithms and referees will ask. Exp P tests
whether the paper's negative results generalize across detection algorithms — and provides the
second adjudication arm for the com-Amazon anomaly (exp_L V3 / C16). If the pattern holds, every
claim gains "across four algorithms"; if some algorithm shows controlled gains, the paper's
universal statements must be scoped per-algorithm (a finding, not a failure — report either way).

## Design

Algorithms (python-igraph implementations, defaults unless stated):
- Infomap (`community_infomap`) — map equation, the family closest to Satuluri's MLR-MCL flow
  intuition available in our stack.
- Louvain (`community_multilevel`) — modularity, the pre-Leiden standard.
- Label propagation (`community_label_propagation`) — cheapest, expected most fragile.
- (Leiden = reference, numbers already in exp_L/exp_K; not re-run.)
Note the limitation up front: no MCL/Metis/Graclus implementation in stack; scope statements to
the four tested algorithms.

Sparsifiers: DSpar calibrated (retentions ~0.8, ~0.5; 2 sparsifier seeds) and L-Spar (target
0.5; deterministic) — machinery verbatim from exp_L/exp_K conventions.
Networks: the exp_L seven (email-Eu-core, wiki-Vote, ca-HepTh, ca-CondMat, email-Enron,
com-DBLP, com-Amazon). 3 algorithm seeds per cell where the algorithm is stochastic (igraph
RNG seeding; label prop and Infomap are stochastic, Louvain too; record all).

Measurements per (algorithm, sparsifier, retention, network):
1. **Honest transfer quality**: partition detected on sparse graph, scored on the ORIGINAL
   graph by modularity Q_orig (common yardstick), against the same algorithm's partitions on
   the original graph (baseline, same seed count). Report dQ_honest_vs_mean and, runtime
   permitting, vs best-of-matched-budget (charge T_sparsify + T_jaccard as applicable).
2. **Recovery** on the 3 labelled networks: AMI/ARI (email-Eu-core), avgF1_ge3 (com-DBLP,
   com-Amazon), sparse-arm vs original-arm of the same algorithm. Granularity handling:
   Infomap/label-prop have no resolution knob, so k is REPORTED for every cell and the
   email-Eu-core/com-DBLP conclusions are drawn only where k is comparable (|Δk|/k < 25%) or
   the direction is robust to k (state which); com-Amazon gets the size-matched random-partition
   chance floor (as exp_V) so the anomaly adjudication is chance-referenced.
3. **Cost**: wall clock per stage; end-to-end speedup vs single run of the same algorithm.

## Pre-registered predictions
- P1: no (algorithm, sparsifier) pair shows honest-transfer Q_orig gain beyond seed noise on
  more than 1/7 networks — the Leiden pattern generalizes.
- P2: label propagation degrades MOST under sparsification (largest mean quality drop and/or
  fragmentation), Infomap intermediate.
- P3: com-Amazon adjudication: L-Spar@~0.5 recovery vs same-algorithm baseline on com-Amazon —
  if the avgF1 advantage reproduces under >=2 of the 3 new algorithms (above the chance floor,
  with k reported), the anomaly is algorithm-general and gets PROMOTED alongside exp_V's
  verdict; if it is Leiden-specific, it is scoped accordingly.
- P4: no end-to-end speedup >1.5x for any algorithm at quality-preserving retention.

## Kill / decision criterion
This is a robustness check, so the "kill" runs in reverse: if any algorithm shows controlled,
beyond-noise gains on >=3/7 networks, the paper's universal negatives are DEAD AS UNIVERSALS
and must be rewritten per-algorithm. Otherwise the negatives gain the four-algorithm scope.
Either outcome is reported; no rescue tweaks; post-hoc observations labeled HYPOTHESIS.

## Execution
Compute on Fuji (62GB, 24 cores; local machine is running exp_V): rsync the 7 small datasets
(~50MB) into Fuji's repo copy, run ONE job at a time (cit-Patents exp_M job may be running
concurrently — ample cores/RAM, but check free RAM before com-DBLP/com-Amazon Infomap cells),
detached with nohup + `ulimit -v` cap, incremental CSV appends, scp results back, commit after
each completed network.

## Output
PAPER_RESTRUCTURE/exp_P_algorithm_generality/{DESIGN.md,run.py,results.csv,recovery.csv,
SUMMARY.md}. Cost estimate: Infomap on com-DBLP/com-Amazon is the slow tail (minutes/run);
total a few hours.
