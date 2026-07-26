# Exp AB — SUMMARY: sparsifier coverage completion (K-Neighbor, Local Degree, Local Similarity, MST-backbone)

**VERDICT — P1 and P4 hold outright and strengthen; P2 holds where it matters and fails
where it does not; P3 holds; kill direction 1 does NOT fire under Leiden but DOES fire
under fixed-k Metis; kill direction 2 FIRES for Local Degree on 2/7 networks.**

The two coverage gaps are now closed and neither one rescues sparsification.

> Adding Chen et al.'s three untested top clustering-fidelity preservers (K-Neighbor,
> Local Degree, Local Similarity) and an entire untested *class*, connectivity-preserving
> MST-backbone sparsifiers, produces **no honest modularity gain in a single one of 102
> matched-retention Leiden cells across seven networks**. The backbone removes the
> fragmentation channel exactly as designed (0.00% fragment nodes where other sparsifiers
> shatter up to 86% of the graph) and gains nothing by it: its honest dQ is negative
> everywhere and its ground-truth recovery is the *worst* of all seven arms. What does
> survive is narrow and must be named: **Local Degree beats a cluster-count-matched
> original-graph baseline on ground-truth recovery on com-Amazon and com-DBLP**, and
> **Local Similarity on com-Amazon beats even the best point of the entire resolution
> sweep**. The Metis result from Exp AA also extends downward: on real heavy-tailed
> graphs the fixed-k gain appears already at d_avg ~ 29-33, not only above 50.

Coverage: `results.csv` 146 rows (118 Leiden + 28 Metis), 7 networks, 7 arms, 2-3
retention operating points, 3 replicates each; `recovery.csv` 255 rows; `skipped.csv`
8 rows; full analysis in `analysis_full.txt`.

---

## 1. Retention floors: a structural finding that had to be measured first

Four of the seven arms have a **hard retention floor** on sparse graphs, so running
everything at 0.2 is not physically possible.

| network | n | m | d_avg | MST (n-1)/m | L-Spar | Local Degree | Local Sim | K-Neigh (k=1) |
|---|---|---|---|---|---|---|---|---|
| email-Eu-core | 986 | 16,064 | 32.58 | 0.0613 | 0.0533 | 0.0613 | 0.0534 | 0.0604 |
| ca-HepTh | 8,638 | 24,806 | 5.74 | **0.3482** | 0.2760 | 0.3466 | 0.2726 | 0.3170 |
| ca-CondMat | 21,363 | 91,286 | 8.55 | **0.2340** | 0.1862 | 0.2337 | 0.1773 | 0.2177 |
| wiki-Vote | 7,066 | 100,736 | 28.51 | 0.0701 | 0.0673 | 0.0701 | 0.0674 | 0.0691 |
| email-Enron | 33,696 | 180,811 | 10.73 | 0.1864 | 0.1586 | 0.1862 | 0.1546 | 0.1792 |
| com-Amazon | 334,863 | 925,872 | 5.53 | **0.3617** | 0.3006 | 0.3521 | 0.2981 | 0.3303 |
| com-DBLP | 317,080 | 1,049,866 | 6.62 | **0.3020** | 0.2441 | 0.3013 | 0.2337 | 0.2774 |

DESIGN deviation, disclosed: a **third operating point at the largest hard floor** was added
per network so that all seven arms are matched at one aggressive retention. `matched`=1 marks
like-for-like rows; 102/118 Leiden rows are matched and all headline statistics use only those.

Also recorded: **L-Spar's parameter grid is coarse at scale.** On com-Amazon its only
achievable retentions are 0.3006 and 0.5418. A real limitation of the local top-ceil(d^e)
rule on low-degree graphs, not a bug.

---

## 2. P1 — HOLDS, more strongly than registered. Kill direction 1 does not fire (Leiden).

- **`dQ_vs_base_best > 0` in 0 of 102 matched rows.** Best cell overall: **-0.01046**
  (K-Neighbor, email-Eu-core, retention 0.503).
- **`dQ_vs_matched > 0` (runtime-matched best-of-N) in 0 of 102 rows.**
- Per-arm best `dQ_vs_base_best`: kn -0.0105, lspar -0.0117, dspar -0.0116, mst_random
  -0.0141, lsim -0.0172, ld -0.0288, mst_jaccard -0.0290.
- Registered P1 allowed "no more than 1/7 networks". Realized: **0/7 for every arm.**

**P4 (Chen's top fidelity preservers show no honest quality gain) — HOLDS.** kn / ld / lsim
mean `dQ_vs_base_best` = -0.215 / -0.122 / -0.133; maxima -0.0105 / -0.0288 / -0.0172.
Fidelity to the full-graph partition and quality on the full graph are different things,
now demonstrated on the three sparsifiers Chen et al. rank highest.

---

## 3. P2 — the backbone works exactly as advertised where fragmentation is the problem

`frag_node_frac` (fraction of nodes in clusters of size < 10), x100:

| network \ arm | mst_jaccard | mst_random | ld | lspar | kn | dspar | lsim |
|---|---|---|---|---|---|---|---|
| ca-HepTh @0.348 | **0.00** | **0.00** | 0.69 | | 22.15 | 27.59 | 44.51 |
| ca-CondMat @0.234 | **0.00** | **0.00** | 0.35 | | 19.79 | 26.73 | 52.02 |
| com-DBLP @0.302 | **0.00** | **0.00** | 0.55 | | 20.17 | 27.53 | 51.50 |
| com-Amazon @0.362 | **0.00** | **0.00** | 4.26 | | 18.20 | 24.54 | 38.17 |
| com-Amazon @0.20 | *(floor)* | *(floor)* | | | 85.49 | 75.01 | |
| ca-CondMat @0.20 | *(floor)* | *(floor)* | | 86.30 | 39.41 | 39.35 | 67.01 |
| wiki-Vote @0.20 | **13.67** | 1.84 | 2.21 | 0.09 | 0.42 | 0.25 | 0.38 |

1. **On the four sparse networks the backbone drives fragmentation to exactly zero**
   (`frag_node_frac`=0.00000, `n_singletons`=0, by construction), where other arms lose
   18-86% of nodes to sub-10-node clusters. The cleanest possible demonstration that
   Artifact II is a *connectivity* artifact.
2. **P2 as literally registered (<1% at both retentions) FAILS on 3/7 networks**, all dense
   (wiki-Vote 13.67%, email-Eu-core 5.41%, email-Enron 2.75%).
3. **HYPOTHESIS (not a result):** on dense graphs the MST-on-(1-Jaccard) backbone *increases*
   fragmentation relative to every other arm. A similarity-chosen spanning tree is a poor
   skeleton for a dense graph: it hands Leiden a set of weakly attached tree leaves.

---

## 4. P3 — HOLDS. Removing the fragmentation channel does not create a gain.

All 14 backbone recovery cells are negative against the nc-matched control, and the backbone
arms have the *worst absolute recovery of any arm* on the overlap-ground-truth networks
(com-Amazon avgF1 0.050-0.080 vs unsparsified baseline 0.187, vs Local Similarity 0.470).

| network | arm | target | nc_arm | metric | nc_ctrl | metric_ctrl | d excess |
|---|---|---|---|---|---|---|---|
| com-Amazon | mst_jaccard | 0.50 | 692 | 0.0586 | 662 | 0.1636 | **-0.097** |
| com-Amazon | mst_random | 0.50 | 317 | 0.0798 | 311 | 0.1954 | **-0.110** |
| com-DBLP | mst_jaccard | 0.50 | 716 | 0.0605 | 728 | 0.0853 | -0.022 |
| email-Eu-core | mst_random | 0.20 | 12.0 | AMI 0.5095 | 13 | AMI 0.6365 | -0.121 |

**The connectivity guarantee buys a clean cluster count and costs everything else.** The
registered "major positive result" branch (P2 holds but P3 fails) did NOT occur.

---

## 5. Arm 4 vs arm 5: the Jaccard signal does not help the fill. Random fill is BETTER.

`dQ_vs_base_best` (mst_random minus mst_jaccard): **random fill beats Jaccard fill in 10/10
non-degenerate cells**, by +0.007 to +0.040. (Four cells at the backbone's own floor are
degenerate, where both arms reduce to the same MST; a useful internal check that the
pipelines are otherwise identical.)

Isolating the backbone from the signal shows the backbone is doing all of the work, and what
the similarity signal adds to it on the objective is negative. Same direction as Exp L and
Exp V: a verified structure-aware signal buys nothing on the objective.

---

## 6. Kill direction 2 — FIRES, for Local Degree, on 2/7 networks

Cells beating **both** the unsparsified baseline **and** the cluster-count-matched
original-graph partition, on chance-excess:

| network | arm | target | nc_arm | avgF1 | nc_ctrl | avgF1_ctrl | d excess | vs baseline |
|---|---|---|---|---|---|---|---|---|
| com-Amazon | **ld** | 0.50 | 2,762 | 0.3393 | 2,765 | 0.2249 | **+0.1034** | +0.129 |
| com-Amazon | **ld** | 0.362 | 8,004 | 0.3754 | 7,977 | 0.3272 | **+0.0391** | +0.147 |
| com-Amazon | **ld** | 0.20 | 8,856 | 0.3736 | 8,424 | 0.3302 | **+0.0353** | +0.144 |
| com-Amazon | lsim | 0.50 | 19,052 | 0.4704 | 18,345 | 0.4184 | +0.0415 | +0.220 |
| com-Amazon | lspar | 0.50 | 8,658 | 0.4019 | 8,424 | 0.3302 | +0.0619 | +0.171 |
| **com-DBLP** | **ld** | 0.302 | 1,291 | 0.1630 | 1,247 | 0.1106 | **+0.0325** | +0.028 |
| **com-DBLP** | **ld** | 0.20 | 1,408 | 0.1599 | 1,469 | 0.1231 | **+0.0205** | +0.026 |

- Under Leiden, **only Local Degree reaches the registered >=2/7 threshold**.
- **Local Similarity on com-Amazon is the strongest recovery result in the study**: avgF1
  0.4704 at retention 0.5, exceeding its nc-matched control (0.4184), the **best point of the
  entire gamma sweep on the untouched graph (0.4398)**, and the unsparsified baseline (0.1868).
- **The escape hatch that closes it on com-DBLP:** the best point of the resolution sweep
  (0.3971) beats every arm there, so the com-DBLP win is *granularity-local*.
- **com-Amazon is the network already flagged as the study's network-level anomaly**
  (EXPLORATION.md H4, Exp L). This experiment reproduces it under four more sparsifiers and
  adds the strongest instance yet.

---

## 7. Secondary detector: real Metis, and the threshold moves DOWN

`detector=metis`, k fixed = nc_base, 5 Metis option seeds:

| network | d_avg | arm | ret | dQ_vs_base_best | worst case | Q_base_std |
|---|---|---|---|---|---|---|
| wiki-Vote | 28.5 | lspar | 0.502 | **+0.0166** | **+0.0161** | 0.00077 |
| wiki-Vote | 28.5 | lsim | 0.500 | **+0.0145** | **+0.0128** | 0.00077 |
| wiki-Vote | 28.5 | dspar | 0.500 | **+0.0129** | **+0.0115** | 0.00077 |
| wiki-Vote | 28.5 | kn | 0.500 | **+0.0128** | **+0.0115** | 0.00077 |
| email-Eu-core | 32.6 | dspar | 0.498 | **+0.0156** | **+0.0143** | 0.0064 |

- On wiki-Vote the gains are **15-21x the Metis seed sd**, positive in the worst case, for
  four different sparsifiers. Both backbone arms are negative here.
- **This extends Exp AA rather than contradicting it.** Exp AA located the sign flip at
  d_avg ~ 50 on LFR; on real heavy-tailed graphs it is present already at d_avg 28.5-32.6.
- **The Exp AA context still travels.** Metis baseline Q on wiki-Vote is 0.3822 vs plain
  Leiden 0.4244; on email-Eu-core Metis+L-Spar reaches AMI 0.5467 while resolution-tuned
  Leiden on the untouched graph reaches **0.6727**. "Sparsification can repair a fixed-k cut
  partitioner", not "sparsification improves community detection".

---

## 8. Cost: no arm pays for itself end to end

`speedup_pipeline` (includes sparsifier wall clock) spans **0.58x-1.51x**; detection-only
0.79x-2.13x. The best end-to-end numbers belong to the backbone arms on sparse graphs
(1.23-1.51x) precisely because they do not fragment, bought at dQ -0.063..-0.076. K-Neighbor,
chosen because Chen et al. show it is cheapest to compute, has genuinely negligible
sparsification cost (0.1-0.9 s at 10^6 edges) and still never gains.

---

## 9. Caveats, provenance, DESIGN deviations

**Reference vs our implementations:**
1. **Local Degree and Local Similarity are NetworKit 11.2.1 reference implementations.**
2. **K-Neighbor is OUR implementation** (NetworKit has none; Chen et al. implemented theirs
   too). Follows their §2.3.2 definition, with **fractional k** alignment
   (floor(kf) + Bernoulli(kf-floor(kf)), draw fixed per node) since integer k is a coarse
   grid; pure integer-k=1 retention recorded per row as `kn_ret_k1`.
3. **Both MST-backbone arms are ours**: igraph spanning tree on (1-Jaccard), all n-1 tree
   edges kept, filled to target by highest Jaccard or uniformly at random.
4. **L-Spar and DSpar reused verbatim** from exp_L and exp_N.

**A NetworKit trap worth recording:** `nk.GraphFromCoo` given both (u,v) and (v,u) yields
**2m edges even with `directed=False`** (ca-HepTh 24,806 igraph edges -> 49,612 nk edges),
silently doubling every retention ratio. `build_nk()` now adds each edge once and asserts the
count. Anyone reproducing Chen-style pipelines through NetworKit should check this.

**Metric caveats:**
5. **`dQ_vs_resmatch` is NOT a valid quality comparison.** When nc_sparse < nc_base the match
   requires gamma < 1, and that partition scores terribly under standard modularity
   (email-Enron resmatch at gamma 0.0799 gives Q_orig 0.0696, manufacturing a spurious
   +0.517). Valid quality baselines are `dQ_vs_base_mean`, `dQ_vs_base_best`,
   `dQ_vs_matched`, all negative everywhere. The resolution-matched control's real job here
   is recovery.
6. **`avgF1_ge3` is not chance-corrected**, so a size-matched random-partition floor is
   recorded per row (`avgF1_chance`) and every com-* claim is stated on excess-over-floor.
7. **The resolution-matched control is budgeted** (1 seed, <=22 gamma evaluations, time caps).
   An under-tuned control biases *toward* the arms, so §6's positives are upper bounds.

**DESIGN deviations:** third operating point at the largest floor (necessary; four arms
cannot reach 0.2 on sparse graphs); runtime-matched restarts computed once per network and
replayed with identical seed order; Metis on the 2 densest networks only; ca-CondMat re-run
after a floor-rounding bug (documented in `fixup_condmat.sh`); 16/118 Leiden rows unmatched
and excluded from headline statistics.

**Execution:** dataset streaming as registered, one network on Fuji at a time, free space
checked before each rsync, dataset deleted immediately after, commit and push per network.
Fuji `datasets/` is empty and its free space restored.

---

## 10. Consequences for the paper

- **"You only tested 2 of the 6 best sparsifiers" is answered.** Four of six plus DSpar plus
  two backbone arms, all through the honest protocol, all negative on the objective.
- **"You never tested a connectivity-preserving sparsifier" is answered, and the answer is
  better than a null.** The backbone provably removes Artifact II's mechanism (0.00%
  fragments, 0 singletons) and the conclusion does not move. This converts "fragmentation
  explains the loss" from a plausible story into a **tested and rejected sufficient
  explanation**: modularity still falls by 0.03-0.08 with zero fragmentation.
- **C5/C6/C16/C18 strengthened for the objective, and must be QUALIFIED for recovery**:
  Local Degree beats a granularity-matched baseline on com-Amazon and com-DBLP; Local
  Similarity beats the entire resolution sweep on com-Amazon.
- **The Exp AA degree threshold must be restated.** The fixed-k gain is present at d_avg ~ 29
  on a real graph, so "d_avg < ~50 is the dead zone" is an LFR statement, not a universal one.
  Correct scope: *free-granularity modularity optimizers on sparse real graphs*.
- **Candidate C19 (objective/recovery dissociation) gains a second independent instance**,
  from a sparsifier family with nothing to do with L-Spar's Jaccard signal.
- **New open questions (HYPOTHESIS):** why does the similarity-MST backbone *increase*
  fragmentation on dense graphs? And why is random fill better than Jaccard fill on the
  objective in 10/10 cells?

## 11. Files
`DESIGN.md`, `run.py`, `stream_driver.sh`, `fixup_condmat.sh`, `analyse.py`,
`results.csv` (146 rows), `recovery.csv` (255 rows), `skipped.csv` (8 rows),
`analysis_full.txt`, `driver.log`, `fixup.log`, per-network logs.
