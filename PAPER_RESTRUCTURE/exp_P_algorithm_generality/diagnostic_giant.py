#!/usr/bin/env python
"""Exp P supplementary diagnostic (POST-HOC -> reported as HYPOTHESIS).

Why does label propagation behave differently from Infomap/Louvain under
sparsification?  Candidate: on dense graphs label propagation collapses into
one giant community (the classic monster-cluster failure mode), so ANY thinning
that breaks the collapse raises Q_orig.  This script measures the giant-cluster
share of the baseline and of the DSpar-0.5 sparse partition for all three
algorithms, on the same graphs Exp P used.

Output: giant_share.csv
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import (ALGOS, BASE_SEEDS, SPAR_SEEDS, DSPAR_ALPHAS, NETWORKS,
                 append_row, build_sparse, detect, dspar_scores, load_lcc_graph,
                 log, _probs_calibrated, sparsify_calibrated)

FIELDS = ["network", "algo", "arm", "seed", "k", "giant_share",
          "top3_share", "Q_orig"]


def shares(memb):
    cnt = np.bincount(np.asarray(memb, dtype=np.int64))
    cnt = np.sort(cnt[cnt > 0])[::-1]
    n = cnt.sum()
    return float(cnt[0] / n), float(cnt[:3].sum() / n)


def main(nets):
    out = Path(__file__).resolve().parent / "giant_share.csv"
    for name in nets:
        g = load_lcc_graph(name)
        edge_arr, scores = dspar_scores(g)
        probs = _probs_calibrated(scores, 0.5)
        kept, _ = sparsify_calibrated(edge_arr, probs, SPAR_SEEDS[0])
        gs = build_sparse(g.vcount(), kept)
        log(f"\n=== {name}: n={g.vcount():,} m={g.ecount():,} m'={gs.ecount():,}")
        for algo in ALGOS:
            for arm, graph in (("baseline", g), ("dspar_0.5", gs)):
                for s in BASE_SEEDS:
                    memb, k, _ = detect(graph, algo, s)
                    gi, t3 = shares(memb)
                    row = dict(network=name, algo=algo, arm=arm, seed=s, k=k,
                               giant_share=round(gi, 6), top3_share=round(t3, 6),
                               Q_orig=float(g.modularity(memb.tolist())))
                    append_row(out, row, FIELDS)
                    log(f"  {algo:10s} {arm:10s} s={s} k={k:7d} "
                        f"giant={gi:.4f} top3={t3:.4f} Q_orig={row['Q_orig']:.4f}")
        del g, gs


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else NETWORKS)
