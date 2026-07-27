# Exp AD — SUMMARY: the fixed-$k$ arm on all seven real networks, both $k$ conventions

**VERDICT — P2 HOLDS decisively. P1 HALF FAILS and costs us a network. P3 FAILS. P4 holds on
the objective and fails on recovery. The real-network fixed-$k$ gain now rests on ONE network,
and it is not attributable to the similarity signal.**

> Five sparse networks, 90 cells, both $k$ conventions: **zero** worst-case positive. The density
> condition is now tested on real graphs from below, which it never was before. Above the
> transition, wiki-Vote reproduces (5 of 14 cells, best $+0.0124$) but **email-Eu-core does not**:
> exp_AB called it positive using three Metis seeds per arm, and at ten seeds every worst case is
> negative. On wiki-Vote the positive arms are L-Spar, Local Similarity, K-Neighbor **and DSpar**,
> so the claim that the gain belongs to the similarity signal rather than the edge budget is false
> on real graphs. Overall **5 of 170 cells** are worst-case positive and all five are on one
> network.

Ran on Fuji 2026-07-26, 19:54 to 20:31, one sequential capped job. Ten Metis option seeds per
cell. `run.py` is `exp_AB_sparsifier_coverage/run.py` with the four changes disclosed in
DESIGN.md; the diff is four hunks and nothing else is touched.

---

## The table

| $d_{avg}$ | network | $k$ convention | $k$ | cells | worst-case $>0$ | mean $>0$ | best worst-case |
|---|---|---|---|---|---|---|---|
| 5.5 | com-Amazon | gt | 5000 | 19 | 0 | 0 | $-0.047$ |
| 5.5 | com-Amazon | nc_base | 306 | 19 | 0 | 0 | $-0.028$ |
| 5.7 | ca-HepTh | nc_base | 49 | 19 | 0 | 0 | $-0.038$ |
| 6.6 | com-DBLP | gt | 5000 | 19 | 0 | 0 | $-0.034$ |
| 6.6 | com-DBLP | nc_base | 229 | 19 | 0 | 0 | $-0.030$ |
| 8.5 | ca-CondMat | nc_base | 55 | 19 | 0 | 0 | $-0.039$ |
| 10.7 | email-Enron | nc_base | 171 | 14 | 0 | 0 | $-0.017$ |
| 28.5 | wiki-Vote | nc_base | 6 | 14 | **5** | 6 | $\mathbf{+0.0124}$ |
| 32.6 | email-Eu-core | gt | 42 | 14 | 0 | 1 | $-0.005$ |
| 32.6 | email-Eu-core | nc_base | 8 | 14 | 0 | 2 | $-0.005$ |

Worst case is the minimum sparsified run minus the maximum baseline run over ten Metis seeds.

## V1 — P2 HOLDS. The density condition survives on real graphs, tested from below.

Ninety cells on the five networks between average degree 5.5 and 10.7, under both $k$
conventions, and not one is positive on either statistic. The best cell across all of them is
$-0.017$ and the worst is $-0.61$. Before tonight the claim that a fixed-$k$ partitioner needs
density rested entirely on LFR; it now rests on real networks as well.

This also extends the negative territory to a second algorithm family. Section 4 of the paper
currently delimits the negative for detectors that choose their own granularity. On sparse real
graphs a fixed-$k$ partitioner gains nothing either.

## V2 — P1 HALF FAILS. email-Eu-core does not survive ten seeds.

wiki-Vote reproduces exp_AB: five of fourteen cells worst-case positive.

email-Eu-core does not. Its means go positive in two cells, K-Neighbor $+0.0175$ and DSpar
$+0.0153$, but **every worst case over ten seeds is negative**, best $-0.0045$. exp_AB ran three
Metis seeds per arm and reported it positive. The difference is the seed count, not the data.

Consequence: any statement of the form "the fixed-$k$ gain appears on real graphs at average
degree 28.5 to 32.6" must become 28.5, on one network. This retracts part of exp_AB's headline
and standing fact 5 in the handoff.

## V3 — P3 FAILS. The gain is not the similarity signal.

wiki-Vote at retention 0.5, worst case over ten seeds:

| arm | worst case | mean |
|---|---|---|
| L-Spar | $+0.0124$ | $+0.0163$ |
| Local Similarity | $+0.0087$ | $+0.0139$ |
| K-Neighbor | $+0.0083$ | $+0.0131$ |
| **DSpar** | $\mathbf{+0.0076}$ | $+0.0126$ |
| Backbone, random fill | $-0.0150$ | $-0.0074$ |
| Backbone, Jaccard fill | $-0.0257$ | $-0.0186$ |
| Local Degree | $-0.0580$ | $-0.0344$ |

DSpar is the degree-based sampler and it produces the gain at three quarters the size of the best
similarity method. Section 5 of the paper says degree-based and uniform sparsification "do not
reproduce the effect at any degree" and concludes the gain "is a property of the similarity signal
rather than of the edge budget". That holds on LFR. It is false here.

What the data does separate is the *kind* of rule: the four methods that score individual edges
locally all produce the gain, and both connectivity-preserving backbones destroy it. Local Degree
is the exception in the other direction and is the worst arm on the network, which is worth
noting because it is a local rule too.

## V4 — P4 holds on the objective, fails on recovery.

On the objective the two conventions agree everywhere they overlap, including where $k$ differs by
a factor of sixteen (306 against 5000 on com-Amazon).

On recovery they do not. email-Eu-core, AMI against the Metis baseline at retention 0.5:

| arm | $k = 8$ (Leiden count) | $k = 42$ (ground truth) |
|---|---|---|
| L-Spar | $+0.050$ | $+0.027$ |
| Local Similarity | $+0.042$ | $+0.022$ |
| K-Neighbor | $+0.028$ | $\mathbf{-0.035}$ |
| DSpar | $+0.027$ | $\mathbf{-0.025}$ |
| Backbone, Jaccard | $+0.020$ | $+0.008$ |
| Backbone, random | $-0.006$ | $-0.040$ |
| Local Degree | $-0.236$ | $-0.285$ |

Two arms change sign. Any recovery claim under a fixed-$k$ detector must carry the convention that
produced it.

## Findings with pointers

### 1. Choosing $k$ well beats sparsifying (recovery.csv)

The email-Eu-core Metis baseline scores AMI $0.4659$ at $k = 8$ and $0.5234$ at $k = 42$. Moving
$k$ from Leiden's coarse default to the ground-truth count buys the *baseline* $+0.057$, more than
any sparsifier buys at either $k$. This is the same shape as the resolution-tuned-optimizer result
in Section 5: the cheap fix to the detector beats the pipeline.

### 2. The dissociation appears again, now under a fixed-$k$ detector (results.csv, recovery.csv)

On email-Eu-core at $k = 8$, L-Spar's worst-case modularity is $-0.033$ while its AMI gain is
$+0.050$. Objective and recovery move in opposite directions on the same partitions of the same
graph, with granularity excluded by construction. Every previous instance of the dissociation was
on a free-granularity detector where granularity was a candidate explanation. This one is not.

### 3. Recovery on the two large labelled networks is negative everywhere

com-Amazon and com-DBLP, all seven arms, both conventions: average best-match $F_1$ falls in every
cell, by $0.000$ to $0.015$. The com-Amazon recovery exception that survives every control under
Leiden does not appear under Metis.

---

## Caveats

C1. **The one surviving positive rests on one network at one $k$ convention.** wiki-Vote has no
ground-truth labels, so its $k$ could only be set from Leiden's community count, which is 6 on a
graph of 7,066 vertices. That is a very coarse partition, and the gain is measured against a
baseline using it. Whether the gain survives a different $k$ on wiki-Vote is untested and is the
obvious next experiment: a $k$ sweep on that network alone.

C2. Ten Metis option seeds. V2 shows that three was too few; ten may also be too few, and the
direction of that error is unknown.

C3. Metis imposes a balance constraint. Nothing here separates "fixed $k$" from "fixed $k$ plus
balance", and the founding claim concerns algorithms that impose both.

C4. Still no MLR-MCL and no Graclus. The exclusion in the paper must now say that no maintained
implementation exists that we could run, not that the library lacks them, because the library does
not lack Metis and we have now run it everywhere.

C5. The `gt` convention on com-DBLP and com-Amazon uses the count of SNAP overlapping communities
of at least `MIN_GT_SIZE` members, 5000 in both cases, while Metis returns a partition. The
mismatch between an overlapping reference and a partition is a known limitation of the metric, not
of this run.

C6. Recovery deltas in V4 pool the ten seeds per arm; the two conventions were separated by
cluster count because `recovery.csv` does not carry a convention column.

---

## Effect on claims

- **The negative half of the boundary is strengthened and broadened.** It now covers both
  algorithm families on real graphs, with the transition bracketed between average degree 10.7 and
  28.5 rather than bounded from one side.
- **The positive half narrows to one real network** and loses its similarity-signal attribution.
  Section 5's "the edge budget rather than the selection" paragraph is contradicted on real data
  and must be rescoped to LFR.
- **Section 4's exclusion paragraph is void.** Metis was available and is now run on all seven
  networks; the paragraph should state the result, not the excuse.
- **Introduction B6** gains a bracketed transition and loses the clause attributing the gain to the
  similarity signal.
- **exp_AB's SUMMARY overstates its own Metis result**, which was based on three seeds per arm.
  It should be annotated rather than rewritten, with a pointer here.
- **A new open question, not a claim:** on wiki-Vote the four local edge-scoring rules produce the
  gain and the two connectivity-preserving ones destroy it, while Local Degree, also local, is the
  worst arm on the network. The operative property of the selection rule is not yet identified.
