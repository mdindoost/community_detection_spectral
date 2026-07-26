#!/usr/bin/env python
"""Exp P supplementary control: best-of-N plain restarts on the ORIGINAL graph.

The pre-registered control is runtime-matched (the budget of the sparsification
pipeline buys the baseline as many restarts as fit).  For label propagation that
budget bought between 1 and 16 restarts, and label propagation turned out to be
strongly multi-modal, so the matched control is weak exactly where it matters.

This arm gives the BASELINE a strictly larger budget than the pipeline: N plain
restarts of the same algorithm on the original graph, seeds 900+.  It can only
handicap the treatment, never manufacture a gain, so it is a conservative
supplement rather than a rescue.  N=20 on the five small networks, N=10 on
com-DBLP/com-Amazon (cost).

Output: bestof.csv  (one row per network x algo)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import (ALGOS, MATCH_SEEDS, NETWORKS, append_row, detect,
                 load_lcc_graph, log)

FIELDS = ["network", "algo", "N", "Q_best_of_N", "Q_mean_N", "Q_std_N",
          "Q_min_N", "k_mean", "k_min", "k_max", "T_total"]


def main(nets):
    out = Path(__file__).resolve().parent / "bestof.csv"
    for name in nets:
        g = load_lcc_graph(name)
        N = 10 if name in ("com-DBLP", "com-Amazon") else 20
        log(f"\n=== {name}: n={g.vcount():,} m={g.ecount():,}  N={N}")
        for algo in ALGOS:
            Q, K = [], []
            t0 = time.perf_counter()
            for s in MATCH_SEEDS[:N]:
                memb, k, _ = detect(g, algo, s)
                Q.append(float(g.modularity(memb.tolist())))
                K.append(k)
            row = dict(network=name, algo=algo, N=N,
                       Q_best_of_N=float(max(Q)), Q_mean_N=float(np.mean(Q)),
                       Q_std_N=float(np.std(Q)), Q_min_N=float(min(Q)),
                       k_mean=float(np.mean(K)), k_min=int(min(K)), k_max=int(max(K)),
                       T_total=time.perf_counter() - t0)
            append_row(out, row, FIELDS)
            log(f"  {algo:10s} best={row['Q_best_of_N']:.6f} "
                f"mean={row['Q_mean_N']:.6f}+-{row['Q_std_N']:.6f} "
                f"min={row['Q_min_N']:.6f} k={row['k_mean']:.0f} "
                f"[{row['k_min']}-{row['k_max']}]  {row['T_total']:.1f}s")
        del g


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else NETWORKS)
