# Exp AB — Sparsifier coverage completion (pre-registered design)

Registered 2026-07-26 BEFORE any Exp AB code ran. Closes the two coverage gaps the literature
steelman exposed (references/NOTES_chen_and_fastcd.md §A.5, NOTES_effres_2026.md §6).

**Why this exists.** Chen et al. (PVLDB 2024) rank six sparsifiers as top clustering-fidelity
preservers: K-Neighbor, Local Degree, Local Similarity, L-Spar, ER-unweighted, ER-weighted.
**We have tested two of the six.** Separately, every sparsifier in our study can disconnect the
graph, which is why fragmentation (Artifact II) dominates our results; a **connectivity-preserving
(MST-backbone) sparsifier structurally cannot fragment**, and we have never tested that class.
Both are referee sentences we would rather pre-answer than receive.

**Claims at risk:** C5, C6, C16, C18 (sparsifier-family generality) and — after Exp AA — the
scoped headline. If a sparsifier we never tested produces a controlled gain on sparse graphs with
Leiden, the scope tightens again.

## Arms (all at matched realized retention, so the comparison is like-for-like)

1. **K-Neighbor (KN)** — Chen's best clustering preserver, and among the cheapest to compute
   (the only regime where the end-to-end cost arithmetic could plausibly work). Sadhanala et al.
   2016; NetworKit has it, else implement: each node keeps its k highest-weight/lowest-rank
   incident edges, union rule.
2. **Local Degree (LD)** — Hamann et al. 2016 (whose NMI-inflation warning we already credit).
   Each node keeps the top ceil(d^alpha) edges ranked by *neighbour degree*.
3. **Local Similarity (LSim)** — the rank-transformed cousin of L-Spar
   (`log(rank)/log(deg)`), explicitly NOT the same method as L-Spar (NOTES_satuluri2011 §3).
4. **MST-backbone + Jaccard** — connectivity-preserving: compute an MST (weights = 1 - Jaccard),
   keep it always, then fill to the retention target with the highest-Jaccard non-MST edges.
5. **MST-backbone + random** — identical pipeline, random fill. **Isolates the backbone from the
   signal**; if arm 4 beats arm 5, the similarity signal matters; if both beat plain L-Spar, the
   backbone matters.
6. **Reference arms:** L-Spar and DSpar at matched retention (already characterized; included so
   every number is directly comparable within one run).

## Networks, retention, detectors
- The seven Exp L networks (email-Eu-core, wiki-Vote, ca-HepTh, ca-CondMat, email-Enron,
  com-DBLP, com-Amazon), **streamed one at a time** (see Execution).
- Retention targets 0.5 and 0.2; realized retention always reported. Note MST-backbone arms have
  their own floor (n-1 edges) — report it.
- Leiden primary (3 seeds). If time permits, real Metis (pymetis, k fixed) on the two densest
  networks, since Exp AA showed the algorithm family decides.

## Metrics and controls (all mandatory)
Honest-transfer modularity on the ORIGINAL graph; runtime-matched best-of-N baseline AND
best-of-5 restarts (Exp AA showed the runtime-matched control is too weak when the pipeline is
fast); cluster count and **fragment count** (sub-10-node clusters) per arm — the backbone arms
should show ~0 by construction, which is the point; resolution-matched control on the original
graph wherever |dk|/k > 25%; AMI/ARI + size-matched chance floor on the labelled networks;
end-to-end wall clock including the sparsifier's own cost.

## Pre-registered predictions
- **P1:** no arm achieves honest-transfer dQ > 0 beyond best-of-5 restart noise on more than 1/7
  networks. (Prior: Exp G showed Artifact I is sparsifier-agnostic; Exp P showed the negatives
  hold across 4 algorithms; Exp AA showed low-degree graphs are the dead zone regardless.)
- **P2:** the MST-backbone arms produce near-zero fragments (<1% of nodes) at both retentions,
  where L-Spar/DSpar at matched retention produce many — confirming the backbone removes the
  Artifact II channel by construction.
- **P3:** despite P2, the backbone arms still do NOT beat a resolution-matched original-graph
  baseline on recovery — i.e. removing the fragmentation channel does not create a gain, it only
  removes an artifact.
- **P4:** K-Neighbor and Local Degree, despite being Chen's top *fidelity* preservers, show no
  honest *quality* gain — fidelity to the full-graph partition and quality are different things,
  which is our central methodological point.

## Kill criterion (bidirectional, as in Exp AA)
If any arm shows honest-transfer dQ > 0 beyond best-of-5 restart noise, OR a chance-corrected
recovery gain surviving the resolution-matched control, on >= 2/7 networks, then the scoped
headline from Exp AA tightens further and must name the exception. Report loudly either way.
If P2 holds but P3 fails (backbone arms DO gain), that is a major positive result: connectivity
preservation is the missing ingredient, and it becomes the paper's second constructive finding.

## Execution — dataset streaming (Fuji disk is ~5.5 GB free)
Do NOT stage all datasets. For each network in ascending size order:
`rsync one dataset -> run all arms -> scp results back -> DELETE the dataset from Fuji -> next`.
Check `df -h /home` before each rsync and abort with a clear report if free space < 1.5 GB.
One detached capped job at a time; incremental CSV appends; commit+push after each network.

## Output
PAPER_RESTRUCTURE/exp_AB_sparsifier_coverage/{DESIGN.md,run.py,stream_driver.sh,results.csv,
recovery.csv,SUMMARY.md}.
