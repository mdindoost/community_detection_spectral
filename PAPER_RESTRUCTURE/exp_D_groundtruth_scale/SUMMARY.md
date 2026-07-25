# Experiment D — ground-truth recovery at scale (SNAP top-5000)

Average F1 (Yang & Leskovec) of Leiden partitions against overlapping ground-truth communities, on the original graph vs after DSpar.

`_ge3` = detected->gt direction averaged over clusters with >= 3 nodes; `_all` = over all clusters (literal definition).


## Graphs and ground truth

| dataset | n (LCC) | m | gt comms (>=3) | nodes covered | delta_GT | HB ratio (GT) | delta_Leiden |
|---|---|---|---|---|---|---|---|
| com-Amazon | 334,863 | 925,872 | 5000 | 16,716 | -0.00448 | 0.782 | +0.01062 |
| com-DBLP | 317,080 | 1,049,866 | 5000 | 93,432 | +0.14319 | 3.999 | +0.14727 |
| com-Youtube | 1,134,890 | 2,987,624 | 3355 | 37,476 | +0.09951 | 1.780 | +0.24898 |

## Recovery

`resmatch_X` = NO sparsification, Leiden on the original graph with the resolution tuned to the same number of clusters as condition X (granularity control).

| dataset | condition | true retention | avgF1_ge3 (mean+-std) | dF1_ge3 | avgF1_all | dF1_all | gt->det | n_clusters | gamma |
|---|---|---|---|---|---|---|---|---|---|
| com-Amazon | baseline | 1.000 | 0.2106+-0.0043 | -- | 0.2106+-0.0043 | -- | 0.1222 | 367 | 1.00 |
| com-Amazon | dspar_paper_08 | 0.516 | 0.2850+-0.0018 | +0.0744 | 0.2260+-0.0014 | +0.0155 | 0.4199 | 15496 | 1.00 |
| com-Amazon | resmatch_dspar_paper_08 | 1.000 | 0.4042+-0.0010 | +0.1936 | 0.4041+-0.0010 | +0.1936 | 0.7464 | 15015 | 1024.00 |
| com-Amazon | dspar_nr_09 | 0.764 | 0.2062+-0.0123 | -0.0043 | 0.1741+-0.0106 | -0.0365 | 0.1455 | 683 | 1.00 |
| com-Amazon | resmatch_dspar_nr_09 | 1.000 | 0.1754+-0.0017 | -0.0351 | 0.1754+-0.0017 | -0.0351 | 0.1474 | 687 | 3.67 |
| com-DBLP | baseline | 1.000 | 0.1462+-0.0126 | -- | 0.1462+-0.0126 | -- | 0.0244 | 269 | 1.00 |
| com-DBLP | dspar_paper_08 | 0.469 | 0.1000+-0.0033 | -0.0462 | 0.0616+-0.0022 | -0.0846 | 0.0873 | 10689 | 1.00 |
| com-DBLP | resmatch_dspar_paper_08 | 1.000 | 0.2926+-0.0005 | +0.1464 | 0.2915+-0.0005 | +0.1453 | 0.4108 | 10192 | 512.00 |
| com-DBLP | dspar_nr_09 | 0.651 | 0.1091+-0.0035 | -0.0372 | 0.1044+-0.0032 | -0.0418 | 0.0299 | 486 | 1.00 |
| com-DBLP | resmatch_dspar_nr_09 | 1.000 | 0.0879+-0.0020 | -0.0584 | 0.0879+-0.0020 | -0.0584 | 0.0380 | 505 | 4.00 |
| com-Youtube | baseline | 1.000 | 0.0100+-0.0005 | -- | 0.0100+-0.0005 | -- | 0.0138 | 5991 | 1.00 |
| com-Youtube | dspar_paper_08 | 0.410 | 0.0499+-0.0007 | +0.0399 | 0.0480+-0.0007 | +0.0379 | 0.0932 | 125410 | 1.00 |
| com-Youtube | resmatch_dspar_paper_08 | 1.000 | 0.1286+-0.0001 | +0.1186 | 0.1220+-0.0001 | +0.1120 | 0.2337 | 126281 | 2560.00 |
| com-Youtube | dspar_nr_09 | 0.524 | 0.0114+-0.0003 | +0.0014 | 0.0110+-0.0003 | +0.0009 | 0.0167 | 8123 | 1.00 |
| com-Youtube | resmatch_dspar_nr_09 | 1.000 | 0.0435+-0.0003 | +0.0334 | 0.0433+-0.0003 | +0.0333 | 0.0625 | 7844 | 96.00 |

### Sparsification vs its granularity-matched control

| dataset | condition | avgF1_ge3 (sparse) | avgF1_ge3 (resmatch) | diff | avgF1_all (sparse) | avgF1_all (resmatch) | diff |
|---|---|---|---|---|---|---|---|
| com-Amazon | dspar_paper_08 | 0.2850 | 0.4042 | -0.1192 | 0.2260 | 0.4041 | -0.1781 |
| com-Amazon | dspar_nr_09 | 0.2062 | 0.1754 | +0.0308 | 0.1741 | 0.1754 | -0.0013 |
| com-DBLP | dspar_paper_08 | 0.1000 | 0.2926 | -0.1926 | 0.0616 | 0.2915 | -0.2299 |
| com-DBLP | dspar_nr_09 | 0.1091 | 0.0879 | +0.0212 | 0.1044 | 0.0879 | +0.0166 |
| com-Youtube | dspar_paper_08 | 0.0499 | 0.1286 | -0.0787 | 0.0480 | 0.1220 | -0.0741 |
| com-Youtube | dspar_nr_09 | 0.0114 | 0.0435 | -0.0321 | 0.0110 | 0.0433 | -0.0324 |

## Verdict

**No. Sparsification does not improve ground-truth recovery at scale, under either condition.**
Taken at face value the draft's published setting (DSpar `method="paper"`, nominal α=0.8, true
retention 0.41–0.52) looks like a win on 2 of 3 datasets — com-Amazon +0.074 and com-Youtube
+0.040 avgF1_ge3, com-DBLP −0.046 — but that gain is *entirely* a granularity artifact: DSpar
fragments the graph and Leiden therefore returns 20–40× more, far smaller clusters (Amazon
367 → 15.5k, Youtube 6.0k → 125k), and the SNAP top-5000 communities are tiny (median 7–8
nodes), so any finer partition scores better. The control settles it: simply raising Leiden's
resolution on the **unsparsified** graph to the same cluster count beats DSpar on every dataset
and by a wide margin (Amazon 0.404 vs 0.285, DBLP 0.293 vs 0.100, Youtube 0.129 vs 0.050) — i.e.
the free resolution knob captures the whole effect and more, and the edges DSpar throws away are
pure loss. Mild true pruning (`probabilistic_no_replace`, nominal 0.9 → true retention 0.52–0.76)
changes recovery by −0.037 … +0.001 vs baseline, i.e. nothing; it does edge out its own
granularity-matched control on com-Amazon (+0.031) and com-DBLP (+0.021) but loses badly on
com-Youtube (−0.032), so even that residual is not a general effect.

Secondary observations:

- δ measured against **ground truth** is +0.143 (DBLP), +0.100 (Youtube) but **−0.004**
  (Amazon), while δ against the **Leiden** partition of the same graph is uniformly positive and
  larger (+0.011 / +0.147 / +0.249). Amazon flips sign between the two — consistent with
  exp_A's partition-provenance hypothesis: δ>0 is substantially a property of Leiden partitions,
  not of the networks' true community structure.
- The ground-truth hub-bridge ratio E[d_u d_v | inter] / E[d_u d_v | intra] is 0.78 (Amazon),
  4.00 (DBLP), 1.78 (Youtube). Amazon is *below* 1 — its ground-truth boundary edges are not
  hub-bridges at all — yet Amazon is the dataset where DSpar's raw ΔF1 looks best, so the draft's
  hub-bridging story does not predict where recovery improves.
- com-Amazon's top-5000 communities are almost perfectly edge-separated (48,691 intra vs 48
  inter edges among covered nodes), which is why its resolution-matched control reaches
  gt→det F1 = 0.75.

Caveats: Leiden is run unweighted on the sparsified graph, matching the draft's own pipeline
(`PAPER_EXPERIMENTS/exp3_scalability.py`); `method="paper"` reweighting is therefore discarded, as
in the draft. All conditions ran 3 seeds; the whole experiment takes ~13 min.

