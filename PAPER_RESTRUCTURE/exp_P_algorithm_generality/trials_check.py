#!/usr/bin/env python
"""Exp P robustness check for the one DESIGN deviation: INFOMAP_TRIALS=1.

igraph's community_infomap defaults to trials=10 (best of 10 internal restarts).
Exp P used trials=1 so that one seed = one restart, matching how Leiden was run
in exp_K/exp_L and making the runtime-matched control meaningful.  The cost is a
noisier Infomap baseline, which could inflate an apparent sparse-arm advantage
on the small graphs (email-Eu-core baseline sd 0.039, wiki-Vote 0.059).

This script re-runs the Infomap cells with trials=10 on the five small networks
(quality) and on the labelled email-Eu-core graph (AMI), for the baseline and
the DSpar-0.8 arm — the arm where Infomap's recovery gain appeared.

Output: trials_check.csv
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R

FIELDS = ["graph", "arm", "trials", "seed", "Q_orig", "k", "AMI", "ARI", "t"]
SMALL = ["email-Eu-core", "wiki-Vote", "ca-HepTh", "ca-CondMat", "email-Enron"]
OUT = Path(__file__).resolve().parent / "trials_check.csv"


def infomap(g, seed, trials):
    import igraph as ig
    import random
    random.seed(int(seed))
    t0 = time.perf_counter()
    vc = g.community_infomap(trials=trials)
    dt = time.perf_counter() - t0
    m = np.asarray(vc.membership, dtype=np.int64)
    return m, int(np.unique(m).size), dt


def main():
    # ---- quality side (main-stage analogue) on the five small networks
    for name in SMALL:
        g = R.load_lcc_graph(name)
        edge_arr, scores = R.dspar_scores(g)
        probs = R._probs_calibrated(scores, 0.8)
        arms = [("baseline", g)]
        for ss in R.SPAR_SEEDS:
            kept, _ = R.sparsify_calibrated(edge_arr, probs, ss)
            arms.append((f"dspar_0.8_s{ss}", R.build_sparse(g.vcount(), kept)))
        R.log(f"\n=== {name} (quality, trials=10)")
        for trials in (1, 10):
            for arm, graph in arms:
                for s in R.BASE_SEEDS:
                    memb, k, dt = infomap(graph, s, trials)
                    row = dict(graph=name, arm=arm, trials=trials, seed=s,
                               Q_orig=float(g.modularity(memb.tolist())), k=k,
                               AMI="", ARI="", t=dt)
                    R.append_row(OUT, row, FIELDS)
                    R.log(f"  trials={trials:2d} {arm:18s} s={s} k={k:6d} "
                          f"Q_orig={row['Q_orig']:.6f} ({dt:.2f}s)")
        del g, arms

    # ---- recovery side: labelled email-Eu-core
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    g, y = R.load_email_labelled()
    edge_arr, scores = R.dspar_scores(g)
    probs = R._probs_calibrated(scores, 0.8)
    arms = [("baseline", g)]
    for ss in R.SPAR_SEEDS:
        kept, _ = R.sparsify_calibrated(edge_arr, probs, ss)
        arms.append((f"dspar_0.8_s{ss}", R.build_sparse(g.vcount(), kept)))
    R.log(f"\n=== email-Eu-core LABELLED (AMI, trials sweep)")
    for trials in (1, 10):
        for arm, graph in arms:
            # 10 seeds for the baseline at trials=1 -> honest spread of the
            # noisy control; 3 seeds elsewhere
            seeds = (R.MATCH_SEEDS[:10] if (arm == "baseline" and trials == 1)
                     else R.BASE_SEEDS)
            for s in seeds:
                memb, k, dt = infomap(graph, s, trials)
                row = dict(graph="email-Eu-core-labelled", arm=arm, trials=trials,
                           seed=s, Q_orig=float(g.modularity(memb.tolist())), k=k,
                           AMI=float(adjusted_mutual_info_score(y, memb)),
                           ARI=float(adjusted_rand_score(y, memb)), t=dt)
                R.append_row(OUT, row, FIELDS)
                R.log(f"  trials={trials:2d} {arm:18s} s={s} k={k:4d} "
                      f"AMI={row['AMI']:.4f} ARI={row['ARI']:.4f}")


if __name__ == "__main__":
    main()
