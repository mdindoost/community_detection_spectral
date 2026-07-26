# Exp AA — The Satuluri regime: does sparsification help where its authors said it does?

Pre-registered 2026-07-26, BEFORE any Exp AA code ran. Triggered by the primary-source reading of
Satuluri et al. 2011 (references/NOTES_satuluri2011.md), which established that **every network in
our study sits below the average-degree threshold the original authors themselves identify** as
where L-Spar begins to beat the original graph, and that **none of their four algorithms appears
anywhere in our study**. Mohammad's instruction: "do the experiments just for have a view then we
decide to add it or not." This is an exploratory-but-registered run; inclusion in the paper is a
separate decision.

**Claims at risk:** C16 (L-Spar verdict), C18 (algorithm generality), and the paper's headline
scope. If sparsification helps at high average degree, or under fixed-k balanced partitioners, or
under injected noise, then "sparsification does not help community detection" must be scoped, and
the honest paper becomes a *when-does-it-work* result rather than a negative.

## The three questions, each with its own arm

### Arm A — the average-degree sweep (THE decisive one)
Their §4.5 (LFR, n=10^4, degree exponent 2.1, mu=0.5): *"The sparsification is more beneficial with
increasing degree, and it actually outperforms the original clustering starting from degree 50."*
Our networks: d_avg 5.53-32.58. We have never entered the regime.

- **Generator:** LFR benchmark, n = 10,000, tau1 = 2.1 (their exponent), tau2 = 1.5,
  mu in {0.3, 0.5, 0.8} (0.5 is theirs; 0.8 is the high-mixing cell where they report the largest
  gain, F 26.95 -> 40.47), min_community 20, max_community 500.
- **Average-degree sweep: d_avg in {10, 25, 50, 100, 200}.** 10 and 25 bracket our real networks;
  50 is their stated threshold; 100 and 200 are inside their strong regime (their BioGrid 65,
  Wiki 94, Orkut 76, Twitter 1139).
- 3 LFR seeds per cell.
- **Sparsifiers:** L-Spar (exact Jaccard, e bisected to retention targets), DSpar (calibrated),
  uniform random at matched retention.
- **Retention targets:** 0.5, 0.2, AND their regime 0.15 and 0.05 — reachable at high degree
  because the L-Spar floor is ~1.69/d_avg (verified in NOTES_satuluri2011 §5), i.e. ~0.03 at
  d_avg=50 and ~0.008 at d_avg=200. Report realized retention always.
- **Detectors:** Leiden (ours) + the two fixed-k families below.
- **Metrics:** honest-transfer modularity on the ORIGINAL graph; AMI/ARI against the planted
  labels (chance-corrected, and LFR labels are real ground truth, not a surrogate); cluster
  count; end-to-end wall clock including sparsification.

### Arm B — fixed-k balanced partitioners (their algorithm family)
Their ledger shows L-Spar helps the fixed-k balanced cut partitioners (Metis 4/4, Graclus 3/3) and
does NOT help the free-granularity flow algorithm (MLR-MCL 1/4). Leiden/Louvain/Infomap all belong
to the second group. **This may be the whole explanation of the disagreement.**

- **Primary:** pymetis (real Metis) if installable; otherwise `sklearn.cluster.SpectralClustering`
  (fixed k, normalized cut) as the nearest available fixed-k spectral cut partitioner, plus
  igraph `community_leading_eigenvector(clusters=k)` which also takes k.
- **Report explicitly which were available.** If neither Metis nor Graclus can be installed, say
  so plainly and label the arm a proxy — do NOT silently substitute and call it Metis.
- **Protocol:** k fixed per network across arms (their control), so granularity is pinned by
  construction. Score honestly on the original graph; AMI/ARI vs planted labels; report cluster
  size cv (balance) as an outcome — a benefit our paper has never measured.

### Arm C — noise injection (the causal test of their stated mechanism)
Their explanation is DENOISING: BioGrid's high-throughput false positives, Wiki's spurious
hyperlinks. Denoising can improve agreement with external labels; it cannot improve an objective
computed on the graph whose edges were deleted. This is the clean test nobody has run.

- Take LFR graphs at d_avg in {25, 50} and mu = 0.5, with their planted labels.
- Inject x% random (uniformly placed) edges, x in {0, 10, 25, 50, 100} percent of m.
- Sparsify (L-Spar, DSpar) to a fixed realized retention; detect; measure AMI/ARI vs the ORIGINAL
  planted labels, and honest-transfer modularity on the noisy graph and on the clean graph.
- **The prediction that makes this decisive:** if L-Spar's benefit is denoising, recovery gain must
  RISE MONOTONICALLY with x and VANISH at x = 0.

## Pre-registered predictions

- **P1 (Arm A, the crux):** honest-transfer modularity gain (vs runtime-matched Leiden baseline)
  stays <= 0 at every average degree, INCLUDING d_avg >= 50 — i.e. their degree-50 threshold is
  about their metric (external-GT F-score) and their algorithms, not about modularity.
- **P2 (Arm A, recovery):** AMI/ARI vs planted labels DOES improve at d_avg >= 50 for at least one
  sparsifier at aggressive retention, reproducing their threshold on a recovery metric. If both
  P1 and P2 hold, the objective/recovery dissociation is confirmed as degree-dependent and becomes
  the paper's central positive result.
- **P3 (Arm B):** the fixed-k balanced partitioner shows a recovery gain from sparsification on
  >= 2 of the 5 degree cells where Leiden shows none — i.e. the algorithm family, not the
  sparsifier, decides.
- **P4 (Arm C):** recovery gain rises monotonically with injected noise and is <= 0 at x = 0
  (Spearman(x, gain) > 0.8 on >= 3 of 4 network-sparsifier cells).

## Kill criteria (they run in BOTH directions — this is a scope test, not a defence)

- If **P1 fails** (positive honest modularity gain at high degree beyond seed noise on >= 2 cells),
  our headline negative is FALSE as stated and must be scoped to low-degree graphs. Report loudly.
- If **P2 and P4 both hold**, the denoising explanation is confirmed and the paper's framing
  changes from "does not help" to "helps only via denoising, only against external references,
  only above a degree threshold, and never on the objective."
- If **P2, P3 and P4 all fail**, Satuluri's degree-50 threshold does not reproduce under our
  controls even in their own regime, and C16 can be restated (still without the word "refuted" —
  their metric and algorithms differ; see NOTES_satuluri2011 §8).

## Controls (all mandatory, per EXPLORATION.md)
Honest transfer scoring on the original graph; runtime-matched best-of-N baselines; chance-corrected
AMI/ARI plus a size-matched random-partition chance floor; resolution-matched control where the
detector has a resolution knob; realized retention always reported; end-to-end cost including
sparsification; cluster-size cv reported as an outcome.

## Output
PAPER_RESTRUCTURE/exp_AA_satuluri_regime/{DESIGN.md,run.py,results_*.csv,SUMMARY.md}.
Compute: Fuji preferred (LFR generation at d_avg=200, n=10^4 is ~1M edges; cheap). One detached
capped job at a time. Commit per completed arm.
