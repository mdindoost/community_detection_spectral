# Exp AF — Graclus on the LFR sweep: the boundary does not reproduce

**Ran 2026-07-27 on Fuji. 180 new rows appended to `exp_AA_satuluri_regime/results_armB.csv`
under `detector=graclus`. `exp_AA/run.py` gained one thing: `graclus_partition` registered in
the `DETECTORS` dict (patch: `exp_AE_graclus_real_networks/`-style, backup at
`run.py.pre_graclus_bak`). Nothing else changed, so any difference is the detector.**

## Why this was run

Table VIII and Figure 1, the entire positive half of the paper, rest on **one detector**.
Exp AE had just shown that two fixed-$k$ detectors disagree about which *real* network shows
a gain, and that the gains track each detector's own baseline weakness rather than density.
The open question was whether the synthetic transition near average degree 50 is a property
of fixed-$k$ detectors or a property of Metis.

## The answer: a property of Metis

L-Spar, $\mu = 0.5$, all statuses, which is exactly how Table VIII reproduces.

| $d_{\mathrm{avg}}$ | METIS $\Delta Q>0$ | METIS mean $\Delta Q$ | GRACLUS $\Delta Q>0$ | GRACLUS mean $\Delta Q$ |
|---|---|---|---|---|
| 11.2 | 0/12 | $-0.103$ | **3/12** | $-0.027$ |
| 24.6 | 0/12 | $-0.046$ | **8/12** | $\mathbf{+0.011}$ |
| 49.4 | 8/12 | $-0.016$ | 6/12 | $+0.006$ |
| 101.9 | 10/12 | $+0.006$ | 7/12 | $+0.001$ |
| 220.9 | 12/12 | $+0.009$ | 6/12 | $+0.002$ |

Metis rises monotonically and crosses. **Graclus does not have a transition.** It sits near
half the cells at every degree, it is already positive in the mean at $24.6$, and it is
positive in three cells at $11.2$, the degree at which Metis is 0/12 with a mean of $-0.103$.

Recovery is worse, not better:

| $d_{\mathrm{avg}}$ | METIS mean $\Delta$AMI | GRACLUS mean $\Delta$AMI |
|---|---|---|
| 11.2 | $-0.216$ | $-0.221$ |
| 24.6 | $-0.072$ | $-0.168$ |
| 49.4 | $+0.042$ | $-0.110$ |
| 101.9 | $+0.116$ | $-0.010$ |
| 220.9 | $+0.093$ | $-0.030$ |

**Graclus never gains on recovery at any degree.** Metis crosses at $49.4$ and reaches
$+0.116$; Graclus is negative in the mean everywhere and positive in at most 3 of 12 cells.

## The mechanism, and it is the same one exp_AE found

The baselines on the untouched graphs:

| $d_{\mathrm{avg}}$ | METIS $Q_b$ | GRACLUS $Q_b$ | METIS AMI$_b$ | GRACLUS AMI$_b$ |
|---|---|---|---|---|
| 11.2 | 0.3064 | 0.2243 | 0.4639 | 0.4724 |
| 24.6 | 0.2291 | 0.1636 | 0.5862 | **0.7149** |
| 49.4 | 0.1801 | 0.1512 | 0.5958 | **0.8238** |
| 101.9 | 0.1367 | 0.1365 | 0.6012 | **0.8175** |
| 220.9 | 0.1182 | 0.1212 | 0.5738 | **0.7871** |

Graclus reaches **lower modularity** than Metis on the untouched graph, so on the objective
there is something to repair, and small positives appear at every degree including the
sparsest. Graclus reaches **far higher AMI** than Metis, 0.82 against 0.60 at degree 49, so
on recovery there is nothing to repair, and no recovery gain appears anywhere.

Sparsification repairs whichever measure the detector was weak on. It does not track density.

## Consequences

1. **The transition near average degree 50 is Metis's, not the pipeline's.** Table VIII and
   Figure 1 describe one detector recovering from its own weakness as density rises. A second
   fixed-$k$ detector on the identical graphs shows no transition on either measure.
2. **The claim's second condition cannot stand.** It was inferred from a single detector on
   synthetic data, and it fails on a second detector on the same data and on real networks.
3. **The claim's third clause is now very strong.** At degree 49 the paper reports Metis with
   L-Spar reaching AMI 0.75 against its own baseline of 0.62. Graclus on the **untouched**
   graph reaches **0.8238** at the same degree, with no sparsification and no resolution
   tuning. A better unsparsified detector beats the sparsified pipeline on the synthetic
   benchmark, which the paper previously established only via a resolution-tuned optimizer.
4. **Similarity-specificity survives.** Under Graclus, L-Spar is positive in 3/12, 8/12, 6/12,
   7/12, 6/12 by degree, against DSpar 0/12, 0/12, 2/12, 3/12, 3/12 and uniform 0/12, 1/12,
   0/12, 0/12, 2/12. Which rule selects the edges still matters. What does not matter is
   density.

## What is untouched

The free-granularity negative (0 of 102 matched Leiden cells), the accounting result (45 of 63
sign flips), the mechanism of the degree-based channel, the speed results, and the exception
analysis are all unaffected. Nothing in this file bears on them.
