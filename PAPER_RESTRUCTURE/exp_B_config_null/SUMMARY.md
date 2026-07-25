# Experiment B — configuration-model (degree-preserving rewire) null

**Question.** The draft reads DSpar separation `delta > 0` and fixed-partition
modularity gain `dQ_fixed > 0` as evidence that DSpar clarifies community
structure. A degree-preserving rewiring keeps the degree sequence exactly and
destroys community structure. If the rewired null reproduces both signals, then
neither is evidence about community structure.

**Setup.** Undirected simple graph, largest connected component. `delta = mu_intra
- mu_inter` on DSpar scores `s(e) = 1/d_u + 1/d_v`; `hb = E[d_u d_v | inter] /
E[d_u d_v | intra]`; `Q_fixed` = igraph modularity of the FIXED Leiden partition.
Sparsifier: `experiments/dspar.py` `method="paper"`, `retention=0.8`, weights
dropped. REAL arm: 1 Leiden partition, 3 sparsification seeds. NULL arm: 2 rewire
seeds (igraph simple double-edge swaps, `n_swaps = 10m`), fresh Leiden partition
per rewire, 2 sparsification seeds each (4 reps). `+-` is the std over reps.

`actual_retention` is `m_sparse / m`: the 'paper' DSpar method samples *with
replacement*, so the unique-edge retention is far below the nominal 0.8.

## Side-by-side table

| network | arm | n | m | Q_fixed_base | delta | hb | dQ_fixed | actual_retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| email-Eu-core | real | 986 | 16064 | 0.4159 | +0.0340 | 2.113 | +0.0819±0.0044 | 0.4438±0.0017 |
| email-Eu-core | null | 986 | 16064 | 0.1274±0.0013 | +0.0413±0.0016 | 1.404±0.002 | +0.0350±0.0032 | 0.4473±0.0005 |
| wiki-Vote | real | 7066 | 100736 | 0.4240 | +0.0513 | 1.902 | +0.0711±0.0012 | 0.3320±0.0007 |
| wiki-Vote | null | 7066 | 100736 | 0.1260±0.0011 | +0.1009±0.0009 | 1.689±0.027 | +0.0843±0.0012 | 0.3331±0.0009 |
| ca-GrQc | real | 4158 | 13422 | 0.8514 | +0.0872 | 0.478 | +0.0176±0.0058 | 0.4523±0.0036 |
| ca-GrQc | null | 4158 | 13422 | 0.3768±0.0013 | +0.1890±0.0009 | 1.758±0.063 | +0.0991±0.0031 | 0.4736±0.0030 |
| ca-HepTh | real | 8638 | 24806 | 0.7619 | +0.1736 | 1.581 | +0.0523±0.0022 | 0.4771±0.0023 |
| ca-HepTh | null | 8638 | 24806 | 0.4129±0.0012 | +0.2147±0.0015 | 1.929±0.047 | +0.0941±0.0016 | 0.4827±0.0014 |
| facebook-combined | real | 4039 | 88234 | 0.8355 | -0.0014 | 2.681 | +0.0256±0.0013 | 0.4368±0.0012 |
| facebook-combined | null | 4039 | 88234 | 0.1146±0.0001 | +0.0170±0.0006 | 1.391±0.042 | +0.0270±0.0005 | 0.4596±0.0012 |
| ca-CondMat | real | 21363 | 91286 | 0.7312 | +0.1126 | 2.490 | +0.0606±0.0006 | 0.4723±0.0007 |
| ca-CondMat | null | 21363 | 91286 | 0.3119±0.0001 | +0.1018±0.0003 | 1.493±0.032 | +0.0667±0.0012 | 0.4819±0.0008 |
| ca-HepPh | real | 11204 | 117619 | 0.6512 | +0.0211 | 0.501 | +0.0864±0.0016 | 0.3399±0.0008 |
| ca-HepPh | null | 11204 | 117619 | 0.1571±0.0002 | +0.0935±0.0016 | 1.872±0.010 | +0.0917±0.0016 | 0.3826±0.0003 |
| ca-AstroPh | real | 17903 | 196972 | 0.6343 | +0.0332 | 1.496 | +0.0393±0.0008 | 0.4170±0.0004 |
| ca-AstroPh | null | 17903 | 196972 | 0.1676±0.0007 | +0.0413±0.0024 | 1.294±0.007 | +0.0489±0.0011 | 0.4357±0.0007 |
| email-Enron | real | 33696 | 180811 | 0.6124 | +0.1501 | 2.974 | +0.1422±0.0004 | 0.3889±0.0003 |
| email-Enron | null | 33696 | 180811 | 0.2420±0.0005 | +0.2400±0.0036 | 4.472±0.175 | +0.1562±0.0013 | 0.3937±0.0005 |
| cit-HepTh | real | 27400 | 352021 | 0.6567 | +0.0265 | 2.173 | +0.0461±0.0002 | 0.4339±0.0004 |
| cit-HepTh | null | 27400 | 352021 | 0.1583±0.0002 | +0.0322±0.0001 | 1.778±0.149 | +0.0376±0.0006 | 0.4525±0.0001 |
| cit-HepPh | real | 34401 | 420784 | 0.7314 | +0.0110 | 1.358 | +0.0126±0.0003 | 0.4594±0.0005 |
| cit-HepPh | null | 34401 | 420784 | 0.1675±0.0003 | +0.0264±0.0006 | 1.300±0.036 | +0.0319±0.0004 | 0.4676±0.0003 |
| com-Amazon | real | 334863 | 925872 | 0.9296 | +0.0096 | 1.042 | +0.0000±0.0001 | 0.5154±0.0003 |
| com-Amazon | null | 334863 | 925872 | 0.4235±0.0002 | +0.1389±0.0014 | 2.139±0.024 | +0.0586±0.0005 | 0.5121±0.0001 |
| com-DBLP | real | 317080 | 1049866 | 0.8271 | +0.1469 | 1.583 | +0.0437±0.0002 | 0.4688±0.0003 |
| com-DBLP | null | 317080 | 1049866 | 0.3647±0.0005 | +0.1965±0.0022 | 2.308±0.078 | +0.0980±0.0005 | 0.4750±0.0003 |
| com-Youtube | real | 1134890 | 2987624 | 0.7188 | +0.2459 | 3.996 | +0.0959±0.0002 | 0.4099±0.0001 |
| com-Youtube | null | 1134890 | 2987624 | 0.4024±0.0000 | +0.5520±0.0002 | 14.066±0.094 | +0.2633±0.0002 | 0.4084±0.0001 |
| wiki-Talk | real | 2388953 | 4656682 | 0.5995 | +0.5135 | 10.477 | +0.2020±0.0002 | 0.4554±0.0000 |
| wiki-Talk | null | 2388953 | 4656682 | 0.4878±0.0000 | +0.8313±0.0000 | 105.630±0.062 | +0.3529±0.0001 | 0.4149±0.0001 |
| wiki-topcats | real | 1791489 | 25444207 | 0.6452 | +0.0229 | 5.076 | +0.0484±0.0001 | 0.4475±0.0000 |
| wiki-topcats | null* | 1791489 | 25444207 | 0.1412±0.0000 | +0.0287±0.0000 | 6.920±0.000 | +0.0538±0.0000 | 0.4348±0.0001 |
| cit-Patents | real | 3764117 | 16511740 | 0.8268 | +0.0378 | 0.723 | +0.0047±0.0001 | 0.4610±0.0000 |
| cit-Patents | null* | 3764117 | 16511740 | 0.2962±0.0000 | +0.1659±0.0000 | 1.940±0.000 | +0.0768±0.0000 | 0.4666±0.0001 |

`*` reduced null replication (1 rewire seed x 2 sparsification seeds instead
of 2 x 2) — wiki-topcats, cit-Patents are too large for the full budget.

## Null / real ratios

| network | delta_real | delta_null | dQ_real | dQ_null | dQ_null/dQ_real | delta_null/delta_real |
|---|---:|---:|---:|---:|---:|---:|
| email-Eu-core | +0.034000 | +0.041288 | +0.081933 | +0.035007 | 0.43 | 1.21 |
| wiki-Vote | +0.051325 | +0.100942 | +0.071085 | +0.084327 | 1.19 | 1.97 |
| ca-GrQc | +0.087235 | +0.188996 | +0.017610 | +0.099082 | 5.63 | 2.17 |
| ca-HepTh | +0.173647 | +0.214663 | +0.052323 | +0.094114 | 1.80 | 1.24 |
| facebook-combined | -0.001364 | +0.016971 | +0.025556 | +0.026966 | 1.06 | -12.44 |
| ca-CondMat | +0.112577 | +0.101792 | +0.060564 | +0.066655 | 1.10 | 0.90 |
| ca-HepPh | +0.021081 | +0.093537 | +0.086422 | +0.091684 | 1.06 | 4.44 |
| ca-AstroPh | +0.033199 | +0.041277 | +0.039330 | +0.048882 | 1.24 | 1.24 |
| email-Enron | +0.150080 | +0.239952 | +0.142247 | +0.156214 | 1.10 | 1.60 |
| cit-HepTh | +0.026474 | +0.032194 | +0.046123 | +0.037590 | 0.82 | 1.22 |
| cit-HepPh | +0.011016 | +0.026361 | +0.012617 | +0.031933 | 2.53 | 2.39 |
| com-Amazon | +0.009612 | +0.138908 | +0.000015 | +0.058580 | 3.82e+03 | 14.45 |
| com-DBLP | +0.146870 | +0.196526 | +0.043729 | +0.097967 | 2.24 | 1.34 |
| com-Youtube | +0.245889 | +0.551965 | +0.095856 | +0.263314 | 2.75 | 2.24 |
| wiki-Talk | +0.513457 | +0.831288 | +0.202016 | +0.352896 | 1.75 | 1.62 |
| wiki-topcats | +0.022923 | +0.028658 | +0.048358 | +0.053772 | 1.11 | 1.25 |
| cit-Patents | +0.037773 | +0.165925 | +0.004713 | +0.076767 | 16.29 | 4.39 |

Degenerate ratios: com-Amazon's real `dQ_fixed` is 1.5e-5 (indistinguishable from
zero), so its ratio is not meaningful beyond 'the null gains and the real graph
does not'. facebook-combined is the one network with `delta(real) < 0`, which
flips the sign of its delta ratio; the null there is still `delta > 0`.

## Verdict

Across all 17 networks the degree-preserving rewired null reproduces both signals: `delta > 0` in 17/17 nulls and `dQ_fixed > 0` in 17/17 nulls, on graphs that have no community structure at all (null `Q_fixed_base` collapses to the value a modularity maximiser extracts from pure degree noise). The null is not merely positive but typically *larger* than the real graph: `dQ_fixed(null) >= dQ_fixed(real)` in 15/17 networks and `delta(null) >= delta(real)` in 16/17, with a median ratio `dQ_fixed(null)/dQ_fixed(real)` of 1.24. Both quantities are therefore explained by degree heterogeneity plus the fact that DSpar preferentially keeps edges between low-degree nodes — a fixed partition always gains modularity when a degree-biased sampler thins hub-incident edges, whether or not the partition means anything. Neither `delta > 0` nor `dQ_fixed > 0` can be cited as evidence that DSpar clarifies community structure; any such claim needs a statistic that separates the real graph from its own configuration model.

**Skipped:** none — all 17 networks completed both arms.

**Budget note.** Every network except the two largest ran both arms well inside
15 min (the whole 13-network batch up to com-DBLP took 5 min; com-Youtube 6.5 min,
wiki-Talk 11 min). cit-Patents (16.5M edges) and wiki-topcats (25.4M edges)
exceeded 15 min on the null arm with 2 rewire replicates, so their nulls were
re-run with 1 rewire seed x 2 sparsification seeds (1120 s and 989 s respectively).

Per-arm wall times (s):

| network | real | null |
|---|---:|---:|
| email-Eu-core | 0.1 | 0.3 |
| wiki-Vote | 0.5 | 2.4 |
| ca-GrQc | 0.1 | 0.5 |
| ca-HepTh | 0.3 | 1.3 |
| facebook-combined | 0.3 | 1.9 |
| ca-CondMat | 1.0 | 4.7 |
| ca-HepPh | 0.7 | 3.0 |
| ca-AstroPh | 1.5 | 5.5 |
| email-Enron | 1.4 | 7.7 |
| cit-HepTh | 2.6 | 11.0 |
| cit-HepPh | 3.2 | 14.2 |
| com-Amazon | 14.8 | 93.0 |
| com-DBLP | 17.6 | 102.1 |
| com-Youtube | 71.9 | 315.1 |
| wiki-Talk | 137.7 | 529.1 |
| wiki-topcats | 281.4 | 989.0 |
| cit-Patents | 377.9 | 1119.9 |

## Reproduce

```
python run.py --network <name>            # appends real+null rows to results.csv
python run.py --network <name> --validate # cross-check vs experiments/dspar.py
python make_summary.py
```

`--validate` reproduces the repo's networkx `dspar_sparsify(method="paper")`
bit-for-bit (identical retention and dQ_fixed to 6 decimals on email-Eu-core,
ca-GrQc and facebook-combined); `run.py` uses a vectorised equivalent so the
large networks fit in memory.
