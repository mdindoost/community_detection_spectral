# Experiment I: Tables 1-2 recomputed on the largest connected component

Pipeline: simple undirected graph -> LCC -> leidenalg `ModularityVertexPartition` (seed 42, n_iterations 2) -> fixed partition P; `experiments/dspar.py::dspar_sparsify(method="paper")` (WITH replacement, q = ceil(0.8 m) draws), weights dropped; 10 seeds (80000-80009, the same seed stream the published run used at alpha=0.8).

Everything below is on the LCC. `published` = full-graph values in `Paper_materials/main-v2.tex` Tables 1-2.

## 1. Graph size: full vs LCC

| dataset | n (full) | m (full, simple) | n (LCC) | m (LCC) | m_LCC/m_full |
|---|---:|---:|---:|---:|---:|
| ca-AstroPh | 18,771 | 198,050 | 17,903 | 196,972 | 0.9946 |
| ca-CondMat | 23,133 | 93,439 | 21,363 | 91,286 | 0.9770 |
| ca-GrQc | 5,241 | 14,484 | 4,158 | 13,422 | 0.9267 |
| ca-HepPh | 12,006 | 118,489 | 11,204 | 117,619 | 0.9927 |
| ca-HepTh | 9,875 | 25,973 | 8,638 | 24,806 | 0.9551 |
| cit-HepPh | 34,546 | 420,877 | 34,401 | 420,784 | 0.9998 |
| cit-HepTh | 27,769 | 352,285 | 27,400 | 352,021 | 0.9993 |
| email-Enron | 36,692 | 183,831 | 33,696 | 180,811 | 0.9836 |
| facebook-combined | 4,039 | 88,234 | 4,039 | 88,234 | 1.0000 |
| wiki-Vote | 7,115 | 100,762 | 7,066 | 100,736 | 0.9997 |
| email-Eu-core | 986 | 16,064 | 986 | 16,064 | 1.0000 |

## 2. Table 1 quantities: LCC vs published full-graph

| dataset | Q_orig LCC | Q_orig pub | dQ_fixed LCC | dQ_fixed pub | change | dQ_Leiden LCC | dQ_Leiden pub | change | realized ret. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ca-AstroPh | 0.6343 | 0.6393 | 0.0399 +/- 0.0007 | 0.0423 +/- 0.0008 | -0.0024 | 0.0529 +/- 0.0012 | 0.0571 +/- 0.0016 | -0.0042 | 0.4166 |
| ca-CondMat | 0.7312 | 0.7410 | 0.0614 +/- 0.0015 | 0.0638 +/- 0.0016 | -0.0024 | 0.0999 +/- 0.0014 | 0.1018 +/- 0.0012 | -0.0019 | 0.4715 |
| ca-GrQc | 0.8514 | 0.8664 | 0.0187 +/- 0.0022 | 0.0257 +/- 0.0022 | -0.0070 | 0.0540 +/- 0.0024 | 0.0628 +/- 0.0022 | -0.0088 | 0.4545 |
| ca-HepPh | 0.6512 | 0.6631 | 0.0848 +/- 0.0008 | 0.0844 +/- 0.0012 | +0.0004 | 0.1092 +/- 0.0015 | 0.1089 +/- 0.0029 | +0.0003 | 0.3399 |
| ca-HepTh | 0.7619 | 0.7760 | 0.0542 +/- 0.0015 | 0.0609 +/- 0.0023 | -0.0067 | 0.1134 +/- 0.0019 | 0.1212 +/- 0.0022 | -0.0078 | 0.4766 |
| cit-HepPh | 0.7314 | 0.7349 | 0.0120 +/- 0.0004 | 0.0136 +/- 0.0004 | -0.0016 | 0.0188 +/- 0.0022 | 0.0182 +/- 0.0019 | +0.0006 | 0.4593 |
| cit-HepTh | 0.6567 | 0.6637 | 0.0467 +/- 0.0007 | 0.0469 +/- 0.0009 | -0.0002 | 0.0582 +/- 0.0021 | 0.0546 +/- 0.0022 | +0.0036 | 0.4332 |
| email-Enron | 0.6124 | 0.6248 | 0.1429 +/- 0.0012 | 0.1507 +/- 0.0009 | -0.0078 | 0.1798 +/- 0.0020 | 0.1853 +/- 0.0008 | -0.0055 | 0.3886 |
| facebook-combined | 0.8355 | 0.8357 | 0.0253 +/- 0.0005 | 0.0259 +/- 0.0008 | -0.0006 | 0.0277 +/- 0.0010 | 0.0273 +/- 0.0008 | +0.0004 | 0.4360 |
| wiki-Vote | 0.4240 | 0.4175 | 0.0734 +/- 0.0012 | 0.0694 +/- 0.0011 | +0.0040 | 0.0827 +/- 0.0077 | 0.0932 +/- 0.0062 | -0.0105 | 0.3328 |
| email-Eu-core | 0.4159 | 0.4174 | 0.0819 +/- 0.0025 | 0.0789 +/- 0.0041 | +0.0030 | 0.0919 +/- 0.0034 | 0.0912 +/- 0.0029 | +0.0007 | 0.4429 |

## 3. Table 2 quantities: LCC vs published full-graph

| dataset | dF_obs LCC | dF_obs pub | change | -dG_obs LCC | -dG_obs pub | change |
|---|---:|---:|---:|---:|---:|---:|
| ca-AstroPh | 0.0337 +/- 0.0007 | 0.0350 +/- 0.0008 | -0.0013 | 0.0062 +/- 0.0001 | 0.0072 +/- 0.0001 | -0.0010 |
| ca-CondMat | 0.0584 +/- 0.0015 | 0.0588 +/- 0.0016 | -0.0004 | 0.0030 +/- 0.0000 | 0.0051 +/- 0.0001 | -0.0021 |
| ca-GrQc | 0.0083 +/- 0.0022 | 0.0142 +/- 0.0022 | -0.0059 | 0.0105 +/- 0.0001 | 0.0115 +/- 0.0003 | -0.0010 |
| ca-HepPh | -0.0111 +/- 0.0009 | 0.0030 +/- 0.0011 | -0.0141 | 0.0959 +/- 0.0002 | 0.0814 +/- 0.0002 | +0.0145 |
| ca-HepTh | 0.0499 +/- 0.0015 | 0.0518 +/- 0.0023 | -0.0019 | 0.0043 +/- 0.0002 | 0.0091 +/- 0.0003 | -0.0048 |
| cit-HepPh | 0.0084 +/- 0.0004 | 0.0078 +/- 0.0004 | +0.0006 | 0.0037 +/- 0.0001 | 0.0058 +/- 0.0001 | -0.0021 |
| cit-HepTh | 0.0350 +/- 0.0007 | 0.0318 +/- 0.0009 | +0.0032 | 0.0117 +/- 0.0001 | 0.0152 +/- 0.0001 | -0.0035 |
| email-Enron | 0.1086 +/- 0.0011 | 0.1186 +/- 0.0009 | -0.0100 | 0.0343 +/- 0.0002 | 0.0321 +/- 0.0003 | +0.0022 |
| facebook-combined | -0.0006 +/- 0.0006 | -0.0008 +/- 0.0007 | +0.0002 | 0.0259 +/- 0.0002 | 0.0268 +/- 0.0002 | -0.0009 |
| wiki-Vote | 0.0726 +/- 0.0014 | 0.0671 +/- 0.0016 | +0.0055 | 0.0008 +/- 0.0002 | 0.0023 +/- 0.0007 | -0.0015 |
| email-Eu-core | 0.0734 +/- 0.0027 | 0.0681 +/- 0.0040 | +0.0053 | 0.0085 +/- 0.0009 | 0.0108 +/- 0.0009 | -0.0023 |

## 4. Largest absolute change vs published, per dataset

| dataset | largest-changing quantity | LCC | published | delta |
|---|---|---:|---:|---:|
| ca-AstroPh | Q_orig | 0.6343 | 0.6393 | -0.0050 |
| ca-CondMat | Q_orig | 0.7312 | 0.7410 | -0.0098 |
| ca-GrQc | Q_orig | 0.8514 | 0.8664 | -0.0150 |
| ca-HepPh | -dG_obs | 0.0959 | 0.0814 | +0.0145 |
| ca-HepTh | Q_orig | 0.7619 | 0.7760 | -0.0141 |
| cit-HepPh | Q_orig | 0.7314 | 0.7349 | -0.0035 |
| cit-HepTh | Q_orig | 0.6567 | 0.6637 | -0.0070 |
| email-Enron | Q_orig | 0.6124 | 0.6248 | -0.0124 |
| facebook-combined | -dG_obs | 0.0259 | 0.0268 | -0.0009 |
| wiki-Vote | dQ_Leiden^(sp) | 0.0827 | 0.0932 | -0.0105 |
| email-Eu-core | dF_obs | 0.0734 | 0.0681 | +0.0053 |

## 5. Sign / regime classification

`regime` = which of the two terms is larger in absolute value.

| dataset | dQ_fixed > 0 | dF_obs sign (LCC / pub) | -dG_obs sign (LCC / pub) | regime LCC | regime pub | changed? |
|---|---|---|---|---|---|---|
| ca-AstroPh | yes | + / + | + / + | dF-dominated | dF-dominated | no |
| ca-CondMat | yes | + / + | + / + | dF-dominated | dF-dominated | no |
| ca-GrQc | yes | + / + | + / + | dG-dominated | dF-dominated | regime |
| ca-HepPh | yes | - / + | + / + | dG-dominated | dG-dominated | dF sign |
| ca-HepTh | yes | + / + | + / + | dF-dominated | dF-dominated | no |
| cit-HepPh | yes | + / + | + / + | dF-dominated | dF-dominated | no |
| cit-HepTh | yes | + / + | + / + | dF-dominated | dF-dominated | no |
| email-Enron | yes | + / + | + / + | dF-dominated | dF-dominated | no |
| facebook-combined | yes | - / - | + / + | dG-dominated | dG-dominated | no |
| wiki-Vote | yes | + / + | + / + | dF-dominated | dF-dominated | no |
| email-Eu-core | yes | + / + | + / + | dF-dominated | dF-dominated | no |

Changes: **ca-GrQc**: regime; **ca-HepPh**: dF sign

## 6. Corollary 1: predicted vs observed preservation ratio

mu_intra, mu_inter and delta are on the LCC w.r.t. the fixed partition; ratio_predicted = mu_inter/mu_intra; ratio_observed = (inter-edge survival rate)/(intra-edge survival rate) over 10 seeds.

`shrink` = |ratio_pred - 1| - |ratio_obs - 1|, i.e. how much closer to one the measured ratio is than the Bernoulli-model prediction. This is the quantity the paper's "0.07 to 0.30" sentence refers to.

| dataset | mu_intra | mu_inter | delta | ratio_predicted | ratio_observed | obs - pred | shrink |
|---|---:|---:|---:|---:|---:|---:|---:|
| ca-AstroPh | 0.100849 | 0.067650 | +0.033199 | 0.6708 | 0.8470 +/- 0.0030 | +0.1762 | +0.1762 |
| ca-CondMat | 0.261140 | 0.148564 | +0.112577 | 0.5689 | 0.7034 +/- 0.0070 | +0.1345 | +0.1345 |
| ca-GrQc | 0.318870 | 0.231635 | +0.087235 | 0.7264 | 0.9123 +/- 0.0233 | +0.1859 | +0.1859 |
| ca-HepPh | 0.099195 | 0.078114 | +0.021081 | 0.7875 | 1.0740 +/- 0.0059 | +0.2865 | +0.1385 |
| ca-HepTh | 0.383104 | 0.209457 | +0.173647 | 0.5467 | 0.7075 +/- 0.0081 | +0.1608 | +0.1608 |
| cit-HepPh | 0.083769 | 0.072753 | +0.011016 | 0.8685 | 0.9446 +/- 0.0028 | +0.0761 | +0.0761 |
| cit-HepTh | 0.084867 | 0.058393 | +0.026474 | 0.6881 | 0.8286 +/- 0.0034 | +0.1405 | +0.1405 |
| email-Enron | 0.226721 | 0.076641 | +0.150080 | 0.3380 | 0.5190 +/- 0.0043 | +0.1810 | +0.1810 |
| facebook-combined | 0.045723 | 0.047086 | -0.001364 | 1.0298 | 1.0151 +/- 0.0156 | -0.0147 | +0.0147 |
| wiki-Vote | 0.086463 | 0.035138 | +0.051325 | 0.4064 | 0.6976 +/- 0.0052 | +0.2912 | +0.2912 |
| email-Eu-core | 0.075493 | 0.041492 | +0.034000 | 0.5496 | 0.7314 +/- 0.0088 | +0.1818 | +0.1818 |

- Signed gap (observed - predicted) on the LCC: -0.0147 to +0.2912.
- Shrinkage toward one on the LCC: 0.0147 to 0.2912 (published claim on full graphs: 0.07 to 0.30). Positive on 11/11, i.e. the measured ratio is closer to one than predicted on every network.
- ratio_observed < 1 (preferential inter-community removal, as predicted) on 9/11: ca-AstroPh, ca-CondMat, ca-GrQc, ca-HepTh, cit-HepPh, cit-HepTh, email-Enron, wiki-Vote, email-Eu-core.
- ratio_observed >= 1 on 2/11: ca-HepPh (1.0740, delta=+0.02108), facebook-combined (1.0151, delta=-0.00136).

## 7. Identity check

| dataset | max |dF - dG - dQ_fixed| over 10 seeds |
|---|---:|
| ca-AstroPh | 4.30e-16 |
| ca-CondMat | 5.69e-16 |
| ca-GrQc | 6.87e-16 |
| ca-HepPh | 6.80e-16 |
| ca-HepTh | 6.52e-16 |
| cit-HepPh | 3.33e-16 |
| cit-HepTh | 3.89e-16 |
| email-Enron | 2.89e-15 |
| facebook-combined | 2.08e-16 |
| wiki-Vote | 1.67e-16 |
| email-Eu-core | 3.05e-16 |

Global maximum: 2.89e-15.

## 8. Control: is the change caused by the LCC or by the Leiden implementation?

The published tables used `igraph.community_leiden(objective_function="modularity")`; this re-run uses `leidenalg.ModularityVertexPartition` as specified. `FULL(seed42)` repeats the measurement on the full simple graph with the NEW Leiden routine, isolating the two effects. `LCC(seed7)` and `LCC(seed1234)` vary only the partition seed. See `control.csv` / `control.log` for all columns.

| dataset | variant | Q_orig | delta | dQ_fixed | dF_obs | -dG_obs | ratio_obs |
|---|---|---:|---:|---:|---:|---:|---:|
| ca-AstroPh | FULL(seed42) | 0.6307 | +0.04122 | +0.0455 | +0.0409 | +0.0047 | 0.8210 |
| ca-AstroPh | LCC(seed42) | 0.6343 | +0.03320 | +0.0399 | +0.0337 | +0.0062 | 0.8470 |
| ca-AstroPh | LCC(seed7) | 0.6312 | +0.03409 | +0.0394 | +0.0349 | +0.0045 | 0.8438 |
| ca-AstroPh | LCC(seed1234) | 0.6312 | +0.03264 | +0.0396 | +0.0331 | +0.0065 | 0.8494 |
| ca-CondMat | FULL(seed42) | 0.7379 | +0.12791 | +0.0633 | +0.0585 | +0.0049 | 0.6920 |
| ca-CondMat | LCC(seed42) | 0.7312 | +0.11258 | +0.0614 | +0.0584 | +0.0030 | 0.7034 |
| ca-CondMat | LCC(seed7) | 0.7309 | +0.11121 | +0.0615 | +0.0569 | +0.0047 | 0.7074 |
| ca-CondMat | LCC(seed1234) | 0.7312 | +0.11321 | +0.0604 | +0.0584 | +0.0021 | 0.6985 |
| ca-GrQc | FULL(seed42) | 0.8641 | +0.14563 | +0.0264 | +0.0158 | +0.0106 | 0.8183 |
| ca-GrQc | LCC(seed42) | 0.8514 | +0.08724 | +0.0187 | +0.0083 | +0.0105 | 0.9123 |
| ca-GrQc | LCC(seed7) | 0.8517 | +0.08814 | +0.0177 | +0.0085 | +0.0093 | 0.9100 |
| ca-GrQc | LCC(seed1234) | 0.8510 | +0.08826 | +0.0197 | +0.0091 | +0.0106 | 0.9033 |
| ca-HepPh | FULL(seed42) | 0.6612 | +0.03466 | +0.0809 | +0.0036 | +0.0773 | 0.9777 |
| ca-HepPh | LCC(seed42) | 0.6512 | +0.02108 | +0.0848 | -0.0111 | +0.0959 | 1.0740 |
| ca-HepPh | LCC(seed7) | 0.6596 | +0.02809 | +0.0718 | +0.0017 | +0.0701 | 0.9896 |
| ca-HepPh | LCC(seed1234) | 0.6588 | +0.03028 | +0.0732 | +0.0052 | +0.0680 | 0.9688 |
| ca-HepTh | FULL(seed42) | 0.7734 | +0.20592 | +0.0622 | +0.0530 | +0.0092 | 0.6696 |
| ca-HepTh | LCC(seed42) | 0.7619 | +0.17365 | +0.0542 | +0.0499 | +0.0043 | 0.7075 |
| ca-HepTh | LCC(seed7) | 0.7612 | +0.17170 | +0.0546 | +0.0484 | +0.0061 | 0.7120 |
| ca-HepTh | LCC(seed1234) | 0.7598 | +0.17151 | +0.0549 | +0.0482 | +0.0067 | 0.7165 |
| cit-HepPh | FULL(seed42) | 0.7324 | +0.00712 | +0.0118 | +0.0030 | +0.0087 | 0.9783 |
| cit-HepPh | LCC(seed42) | 0.7314 | +0.01102 | +0.0120 | +0.0084 | +0.0037 | 0.9446 |
| cit-HepPh | LCC(seed7) | 0.7275 | +0.01158 | +0.0151 | +0.0090 | +0.0061 | 0.9385 |
| cit-HepPh | LCC(seed1234) | 0.7325 | +0.00678 | +0.0125 | +0.0027 | +0.0098 | 0.9812 |
| cit-HepTh | FULL(seed42) | 0.6629 | +0.02600 | +0.0458 | +0.0316 | +0.0142 | 0.8435 |
| cit-HepTh | LCC(seed42) | 0.6567 | +0.02647 | +0.0467 | +0.0350 | +0.0117 | 0.8286 |
| cit-HepTh | LCC(seed7) | 0.6584 | +0.02539 | +0.0465 | +0.0333 | +0.0133 | 0.8378 |
| cit-HepTh | LCC(seed1234) | 0.6621 | +0.02440 | +0.0443 | +0.0311 | +0.0132 | 0.8448 |
| email-Enron | FULL(seed42) | 0.6168 | +0.16744 | +0.1503 | +0.1130 | +0.0373 | 0.4988 |
| email-Enron | LCC(seed42) | 0.6124 | +0.15008 | +0.1429 | +0.1086 | +0.0343 | 0.5190 |
| email-Enron | LCC(seed7) | 0.6187 | +0.16018 | +0.1416 | +0.1237 | +0.0179 | 0.4842 |
| email-Enron | LCC(seed1234) | 0.6153 | +0.15230 | +0.1417 | +0.1108 | +0.0309 | 0.5113 |
| facebook-combined | FULL(seed42) | 0.8355 | -0.00136 | +0.0253 | -0.0006 | +0.0259 | 1.0151 |
| facebook-combined | LCC(seed42) | 0.8355 | -0.00136 | +0.0253 | -0.0006 | +0.0259 | 1.0151 |
| facebook-combined | LCC(seed7) | 0.8355 | -0.00162 | +0.0256 | -0.0008 | +0.0265 | 1.0213 |
| facebook-combined | LCC(seed1234) | 0.8356 | -0.00100 | +0.0254 | -0.0005 | +0.0259 | 1.0144 |
| wiki-Vote | FULL(seed42) | 0.4257 | +0.05715 | +0.0828 | +0.0855 | -0.0027 | 0.6770 |
| wiki-Vote | LCC(seed42) | 0.4240 | +0.05132 | +0.0734 | +0.0726 | +0.0008 | 0.6976 |
| wiki-Vote | LCC(seed7) | 0.4168 | +0.05238 | +0.0699 | +0.0732 | -0.0033 | 0.6939 |
| wiki-Vote | LCC(seed1234) | 0.4270 | +0.05213 | +0.0745 | +0.0739 | +0.0006 | 0.6943 |
| email-Eu-core | FULL(seed42) | 0.4159 | +0.03400 | +0.0819 | +0.0734 | +0.0085 | 0.7314 |
| email-Eu-core | LCC(seed42) | 0.4159 | +0.03400 | +0.0819 | +0.0734 | +0.0085 | 0.7314 |
| email-Eu-core | LCC(seed7) | 0.4140 | +0.03294 | +0.0784 | +0.0706 | +0.0077 | 0.7327 |
| email-Eu-core | LCC(seed1234) | 0.4174 | +0.03364 | +0.0814 | +0.0706 | +0.0108 | 0.7331 |

### Seed stability of the near-zero terms

| dataset | dF_obs over LCC seeds {42,7,1234} | -dG_obs over LCC seeds {42,7,1234} | sign stable? |
|---|---|---|---|
| ca-AstroPh | +0.0337, +0.0349, +0.0331 | +0.0062, +0.0045, +0.0065 | yes |
| ca-CondMat | +0.0584, +0.0569, +0.0584 | +0.0030, +0.0047, +0.0021 | yes |
| ca-GrQc | +0.0083, +0.0085, +0.0091 | +0.0105, +0.0093, +0.0106 | yes |
| ca-HepPh | -0.0111, +0.0017, +0.0052 | +0.0959, +0.0701, +0.0680 | **NO** |
| ca-HepTh | +0.0499, +0.0484, +0.0482 | +0.0043, +0.0061, +0.0067 | yes |
| cit-HepPh | +0.0084, +0.0090, +0.0027 | +0.0037, +0.0061, +0.0098 | yes |
| cit-HepTh | +0.0350, +0.0333, +0.0311 | +0.0117, +0.0133, +0.0132 | yes |
| email-Enron | +0.1086, +0.1237, +0.1108 | +0.0343, +0.0179, +0.0309 | yes |
| facebook-combined | -0.0006, -0.0008, -0.0005 | +0.0259, +0.0265, +0.0259 | yes |
| wiki-Vote | +0.0726, +0.0732, +0.0739 | +0.0008, -0.0033, +0.0006 | **NO** |
| email-Eu-core | +0.0734, +0.0706, +0.0706 | +0.0085, +0.0077, +0.0108 | yes |

## 9. Verdict on Section 5.1

1. **"Positive dQ_fixed on all eleven networks"** -- HOLDS. LCC range 0.0120 to 0.1429 (published 0.0136 to 0.1507); dQ_Leiden^(sp) range 0.0188 to 0.1798. Every value moves by at most 0.008 in absolute terms and no ordering of practical interest changes.
2. **Identity dQ_fixed = dF_obs - dG_obs to machine precision** -- HOLDS (max |error| 2.89e-15 over 110 runs, still `<= 1e-14`).
3. **Two-regime story (dF-carried vs null-model-carried)** -- HOLDS in substance, with two bookkeeping changes. `ca-GrQc` moves from dF-dominated (pub 0.0142 vs 0.0115) to dG-dominated (LCC 0.0083 vs 0.0105); both terms stay positive, and the flip is stable across Leiden seeds and is caused by the LCC (the full graph with the same Leiden routine gives 0.0158 vs 0.0106). `ca-HepPh`'s dF_obs changes sign (pub +0.0030 +/- 0.0011, LCC -0.0111 +/- 0.0009), but the control shows this particular sign is NOT robust: on the same LCC with Leiden seeds 7 and 1234 it is +0.0017 and +0.0052. ca-HepPh's dF_obs is indistinguishable from zero at any preprocessing; what is robust is that ca-HepPh is overwhelmingly dG-dominated (|-dG| is 6-60x |dF| in every variant). Safest wording: `facebook-combined` and `ca-HepPh` are the two networks whose change is carried entirely by the null-model term, with dF_obs ~ 0.
4. **`facebook-combined` as the delta ~ 0, dF_obs <= 0 example** -- UNCHANGED. facebook-combined is already connected, so the LCC is the full graph; delta = -0.00136, dF_obs = -0.0006 +/- 0.0006, -dG_obs = +0.0259 +/- 0.0002, identical to the published row up to the Leiden routine.
5. **"Measured preservation ratios closer to one than predicted, by 0.07 to 0.30"** -- HOLDS in direction on 11/11, but the numeric range must be restated as **0.01 to 0.29** on the LCC. The low end moves because facebook-combined (ratio_pred 1.0298, ratio_obs 1.0151) shrinks by only 0.015.
6. **"Direction of preferential removal agrees with the prediction wherever delta is materially positive"** -- NEEDS a caveat at seed 42. ca-HepPh has delta = +0.021 yet ratio_obs = 1.074 > 1 (intra-community edges removed slightly faster). The control shows ratio_obs on ca-HepPh is 0.9777 (full), 1.0740 / 0.9896 / 0.9688 (LCC, three Leiden seeds), i.e. it straddles one; ca-HepPh has the second-smallest delta in the suite. Either keep the sentence with "delta is materially positive" meaning delta >= 0.025 (which excludes ca-HepPh and cit-HepPh), or state 9/11 explicitly.
7. **Realized-retention range in the caption** -- must change from `0.33--0.52` to `0.33--0.48` for these eleven LCCs (min wiki-Vote, max ca-HepTh).
8. **Watch item not caused by the LCC:** `wiki-Vote`'s -dG_obs is the smallest entry in Table 2 and its sign is also Leiden-seed dependent (+0.0008, -0.0033, +0.0006 across seeds 42/7/1234 on the LCC; -0.0027 on the full graph with the same routine; published +0.0023 +/- 0.0007). The delivered table uses seed 42 and is positive, but no claim should rest on the sign of -dG_obs for wiki-Vote.
9. **The disclaimer paragraph at main-v2.tex:986** ("Tables 1-2 retain the full graphs as downloaded...") can be deleted for the preprocessing half; the 15-of-17-network-subset half is unaffected by this experiment.
