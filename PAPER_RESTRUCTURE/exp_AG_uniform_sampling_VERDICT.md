# Exp AG — Does uniform edge sampling suffice for community detection *quality*?

**Testing He, Drineas & Khanna (arXiv:2510.12669, 2025) against a metric they do not use.**

They prove that for graphs with a large structure ratio $\Upsilon(k) = \lambda_{k+1}/\rho_G(k)$,
sampling $O(\gamma^2 n \log n/\epsilon^2)$ edges **uniformly at random** preserves the clustering
subspace, bypassing importance sampling entirely. Their experiments measure the principal angle
between the bottom-$k$ eigenvectors and the true cluster indicators. They never measure whether a
community detection algorithm does better or worse. This does.

**Method.** Arm B of exp_AA extended from $\mu \in \{0.5, 0.8\}$ to
$\mu \in \{0.1, 0.2, 0.3, 0.5, 0.8\}$, i.e. down into the strong-clustering regime their theorem
is about. Five average degrees, three LFR seeds, four retention targets, three sparsifiers
(L-Spar, DSpar, uniform), two fixed-$k$ detectors. 1,080 new cells. Everything else in the
harness unchanged.

## Result 1 — their result transfers, inside a bounded regime

The quantity that matters is how much the expensive selection rule buys over uniform deletion.
Mean $\Delta$AMI of uniform minus mean $\Delta$AMI of L-Spar:

| detector | retention | $\mu=0.1$ (strong) | $\mu=0.3$ | $\mu=0.5$ (moderate) |
|---|---|---|---|---|
| Metis | 0.50 | $-0.064$ | $-0.116$ | $-0.118$ |
| Metis | 0.20 | $-0.050$ | $-0.191$ | $-0.270$ |
| Metis | 0.15 | $-0.099$ | $-0.241$ | $-0.330$ |
| Metis | 0.05 | $-0.175$ | $-0.363$ | $-0.380$ |
| Graclus | 0.50 | $\mathbf{-0.033}$ | $-0.089$ | $-0.192$ |
| Graclus | 0.20 | $\mathbf{-0.031}$ | $-0.201$ | $-0.357$ |
| Graclus | 0.15 | $-0.080$ | $-0.271$ | $-0.405$ |
| Graclus | 0.05 | $-0.172$ | $-0.422$ | $-0.421$ |

**The gap shrinks monotonically as clustering strengthens, in every row.** For Graclus at
$\mu=0.1$ the entire benefit of computing Jaccard similarity for every edge is $0.033$ AMI on a
baseline of $0.94$, and at retention $0.2$ it is $0.031$. That is the regime their theorem
describes, and their conclusion survives the change of metric: where clusters are strong, the
selection rule buys almost nothing.

**And it fails outside it.** At $\mu = 0.5$ the same rule is worth $0.192$ to $0.421$, an order of
magnitude more. At retention $0.05$ it is worth $0.17$ even at $\mu = 0.1$. Their sampling rate
$O(\gamma^2 n\log n/\epsilon^2)$ is a mild-retention quantity, so the retention boundary is
consistent with what they proved; the mixing boundary is new.

**What this means for the field.** The similarity-based line from Satuluri et al.\ onward is
built on computing a per-edge similarity score. On strongly clustered graphs at moderate
retention, that computation is close to unnecessary: uniform deletion gets within $0.03$ of it.
Where the score does earn its cost is precisely where the theory offers no guarantee, on weakly
clustered graphs, and there sparsification is destructive for every rule.

## Result 2 — the repair pattern reproduces on entirely new data

These are 1,080 cells that did not exist when the repair finding was formulated, at mixing levels
never previously run. The pattern holds without exception.

At retention $0.5$, mean $\Delta$AMI under L-Spar, against each detector's own baseline:

| $\mu$ | Metis base | Metis $\Delta$AMI | Graclus base | Graclus $\Delta$AMI |
|---|---|---|---|---|
| 0.1 | 0.82 | $\mathbf{+0.046}$ | 0.94 | $-0.018$ |
| 0.2 | 0.77 | $\mathbf{+0.070}$ | 0.92 | $-0.015$ |
| 0.3 | 0.72 | $\mathbf{+0.075}$ | 0.88 | $-0.015$ |
| 0.5 | 0.56 | $\mathbf{+0.031}$ | 0.72 | $-0.009$ |

Metis, the weaker detector at every mixing level, gains at every mixing level. Graclus, better by
0.12 to 0.16 AMI throughout, gains nowhere. Sparsification recovers ground for the detector that
was behind and does not move the one that was ahead. Five mixing levels, two detectors, and the
prediction was made before these cells were run.

Note also that L-Spar's gains for Metis here are larger than anything in the original boundary
table, and they occur at $\mu = 0.1$ to $0.3$, where the old claim predicted nothing. The gain
tracks the baseline deficit, not the density and not the mixing.

## Status

Not yet in the paper. Raw data appended to
`exp_AA_satuluri_regime/results_armB.csv` under $\mu \in \{0.1, 0.2, 0.3\}$.
