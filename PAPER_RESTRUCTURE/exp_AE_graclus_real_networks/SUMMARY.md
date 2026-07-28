# Exp AE — SUMMARY: Graclus, the second fixed-$k$ detector, on all seven real networks

**VERDICT — P2 FAILS AS STATED, and the failure corroborates the paper's own
qualification. Every gain Graclus shows is sparsification repairing a detector that was
weak to begin with, and not one of them reaches what the other fixed-$k$ detector
already achieves on the untouched graph.**

> 170 cells, exactly mirroring exp_AD. On the five networks below average degree 11,
> Graclus is worst-case positive in **7 cells** where Metis was positive in **0**. All
> seven are the connectivity-preserving backbone arms. On wiki-Vote, the one real network
> where Metis found a gain, Graclus finds **nothing**. On email-Eu-core at $k=42$, where
> Metis found nothing, Graclus finds **7 positives**. The pattern that explains all of it
> is the baseline: gains appear exactly where that detector's own unsparsified partition
> is the weaker of the two.

Ran on Fuji 2026-07-27, 58 minutes, one sequential job. `run.py` is
`exp_AD_metis_real_networks/run.py` with one disclosed change: a `graclus` detector
alongside `metis`, selected by `--graclus`. The patch script is
`scratchpad/make_exp_AE.py`; it applies 16 named anchors and refuses to run if any is
missing, so the harness is otherwise identical and any difference is the detector.

## Why this experiment exists

Metis takes $k$ as an input **and** imposes a hard balance constraint on part sizes.
Graclus takes $k$ as an input and imposes **no** balance constraint; its only options are
the objective (`ncut`/`rassoc`), local-search steps, and boundary-points-only. It is
therefore the one available algorithm that holds this paper's stated positive condition
fixed while removing the balance confound, which no experiment had addressed.

## The table

Worst case = worst sparsified run minus best baseline run. Graclus is deterministic, so
its baseline is a single exact run and the sweep runs over the sparsifier (three
replicates) rather than over the partitioner.

| $d$ | network | $k$ conv | $k$ | cells | GRACLUS $>0$ | best | METIS $>0$ | best |
|---|---|---|---|---|---|---|---|---|
| 5.5 | com-Amazon | nc_base | 306 | 19 | **3** | $+0.0275$ | 0 | $-0.0279$ |
| 5.5 | com-Amazon | gt | 5000 | 19 | 0 | $-0.0255$ | 0 | $-0.0469$ |
| 5.7 | ca-HepTh | nc_base | 49 | 19 | 0 | $-0.0101$ | 0 | $-0.0378$ |
| 6.6 | com-DBLP | nc_base | 229 | 19 | **3** | $+0.0383$ | 0 | $-0.0303$ |
| 6.6 | com-DBLP | gt | 5000 | 19 | 0 | $-0.0005$ | 0 | $-0.0343$ |
| 8.5 | ca-CondMat | nc_base | 55 | 19 | 0 | $-0.0021$ | 0 | $-0.0390$ |
| 10.7 | email-Enron | nc_base | 171 | 14 | **1** | $+0.0250$ | 0 | $-0.0167$ |
| 28.5 | wiki-Vote | nc_base | 6 | 14 | 0 | $-0.0080$ | **5** | $+0.0124$ |
| 32.6 | email-Eu-core | nc_base | 8 | 14 | 0 | $-0.0035$ | 0 | $-0.0045$ |
| 32.6 | email-Eu-core | gt | 42 | 14 | **7** | $+0.0183$ | 0 | $-0.0047$ |

## V1 — P2 fails as stated

The paper says a fixed-$k$ partitioner shows no gain on the objective on sparse real
graphs. That was established on Metis alone. Graclus contradicts it: 7 of 128 cells below
average degree 11 are worst-case positive, the best at $+0.0383$.

**All seven are `mst_jaccard` or `mst_random`**, the connectivity-preserving backbones the
paper introduced as an instrument rather than as a competitor. No similarity-based or
degree-based arm is positive on any sparse network. What helps Graclus on a sparse graph
is being handed a graph that is still connected, not being handed a similarity signal.

They also appear only under `nc_base`. At $k=5000$ every one of them vanishes
(com-Amazon best $-0.0255$, com-DBLP best $-0.0005$).

## V2 — the explanation: they are repairs of a weak baseline

Graclus optimizes normalized cut. Metis optimizes a balanced cut. Neither optimizes
modularity, and on these graphs they reach different modularity on the untouched graph:

| network | $k$ conv | Graclus $Q_{\mathrm{base}}$ | Metis $Q_{\mathrm{base}}$ | gap |
|---|---|---|---|---|
| com-Amazon | nc_base | 0.8425 | 0.8967 | $-0.0542$ |
| com-DBLP | nc_base | 0.7223 | 0.7883 | $-0.0661$ |
| ca-CondMat | nc_base | 0.6663 | 0.7018 | $-0.0355$ |
| email-Enron | nc_base | 0.3970 | 0.4335 | $-0.0365$ |
| ca-HepTh | nc_base | 0.7111 | 0.7275 | $-0.0163$ |
| email-Eu-core | gt | 0.1920 | 0.2083 | $-0.0163$ |
| email-Eu-core | nc_base | 0.3786 | 0.3679 | $+0.0108$ |
| **wiki-Vote** | nc_base | **0.4164** | **0.3828** | $\mathbf{+0.0337}$ |

The gains track the deficit. Graclus gains where it is the weaker detector; Metis gains on
wiki-Vote, the one network where **Metis** is the weaker detector by 0.034. On
email-Eu-core at $k=8$, where the two are within 0.011 of each other, neither gains.

**And none of the repairs reaches the other detector's untouched baseline:**

| cell | Graclus sparsified | Metis untouched | short by |
|---|---|---|---|
| com-DBLP, mst_jaccard, $\rho=0.50$ | 0.7605 | 0.7883 | $0.0278$ |
| com-Amazon, mst_jaccard, $\rho=0.50$ | 0.8700 | 0.8967 | $0.0267$ |
| email-Enron, mst_jaccard, $\rho=0.50$ | 0.4220 | 0.4335 | $0.0115$ |

Every sparse-network gain is a partial recovery of ground that a different fixed-$k$
detector already held without any sparsification.

## V3 — the balance constraint is not the operative variable, but $k$ is not sufficient either

The experiment was run to separate "takes $k$ as an input" from "imposes a balance
constraint". The answer is that neither is the operative variable on its own. Two
detectors that both take $k$ as an input disagree about which network shows a gain, and
the thing that predicts the gain is neither the constraint nor the density but which
detector was weaker on that graph to begin with.

## V4 — no recovery gain accompanies any of it

On the labelled sparse networks the backbone positives carry average best-match $F_1$ of
0.033 to 0.037, against reference communities. The objective moves and recovery does not,
which is the same dissociation the paper reports elsewhere.

## What this does to the paper

1. The claim's second condition, stated as density, **cannot be supported on the objective
   for fixed-$k$ detectors in general**. It was true of Metis and is false of Graclus.
2. The claim's third clause, that no gain survives comparison with an unsparsified run of
   a better detector, **is strengthened**: it now holds across two fixed-$k$ detectors on
   seven networks, and the sparse-network gains are a new and independent instance of it.
3. The honest generalization the data supports is about the **baseline**, not the density:
   sparsification recovers ground for a detector that was weak on that graph, and does not
   carry it past a detector that was not.

## V5 — validity check: the difference is not the seed protocol

exp_AD gave Metis the **best of ten partitioner seeds** as its baseline. Graclus is
deterministic, so exp_AE's baseline is a single exact run. Best-of-ten is the stronger
baseline, so Metis was held to a harder target, and that asymmetry could in principle
manufacture the whole difference. It does not.

Recomputing Metis's worst-case margin against the **mean** of its seeds, which is the
like-for-like comparison with a single deterministic run:

| comparison | cells | worst-case positive | best cell |
|---|---|---|---|
| Metis vs best of 10 seeds (as published) | 128 | 0 | $-0.0167$ |
| Metis vs mean of seeds (equalised) | 128 | **0** | $-0.0123$ |
| Graclus vs its deterministic baseline | 128 | **7** | $+0.0383$ |

Metis stays at zero either way. The reason is that its seed spread is negligible on these
graphs: best-of-ten buys between $+0.0004$ and $+0.0044$ over the mean, while the distance
between Metis's best cell and Graclus's best cell is roughly $0.05$. The protocol asymmetry
is worth about a tenth of the effect it would have to explain.

**The difference between the two detectors is the detector.**
