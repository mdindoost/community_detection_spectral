#!/usr/bin/env python
"""
Exp K controls.  Two questions the main grid (results.csv) cannot answer alone:

 (1) WEIGHTS vs TOPOLOGY.  For every sparsified graph produced in results.csv
     (same sampler, same alpha, same sparsify seed -> bit-identical topology),
     also run Leiden with the weights DROPPED and score that partition back on
     the original graph, and also record the UNWEIGHTED fixed-partition
     modularity dQ_fixed on the same topology.  Any difference between the
     weighted and unweighted arm is attributable to the weights alone.

 (2) RETENTION vs SAMPLER.  Liu et al.'s with-replacement sampler realizes only
     33-59% distinct-edge retention at nominal alpha in {0.8, 1.0}, so its
     honest-transfer numbers are not comparable to the calibrated sampler at
     alpha=0.9.  Here the calibrated Bernoulli sampler is re-run at
     alpha = (mean realized retention of the paper arm at alpha=1.0), i.e.
     retention-matched, both weighted (HT) and unweighted.

Writes controls.csv incrementally.
"""

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/md724/community_detection_spectral/PAPER_RESTRUCTURE/exp_K_weighted_regime")
from run import (  # noqa: E402
    HERE, BASE_SEEDS, SPARSE_SEEDS, LEIDEN_SPARSE_SEEDS, P0_SEED,
    load_lcc_graph, load_email_labeled, dspar_scores, sparsify_paper,
    sparsify_calibrated, _probs_calibrated, leiden, build_sparse,
)
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score  # noqa: E402

OUT = HERE / "controls.csv"

FIELDS = [
    "network", "n", "m", "sampler", "alpha", "weighted", "rep",
    "sparsify_seed", "leiden_seed", "m_sparse", "retention",
    "Q_P0_orig", "Q_P0_sparse", "dQ_fixed", "Q_sparse_leiden",
    "Q_orig_of_P", "Q_base_mean", "Q_base_std", "Q_base_best",
    "transfer_vs_mean", "transfer_vs_best", "nc_P0", "nc_P", "nc_base_mean",
    "t_leiden", "ami_P", "ari_P", "ami_base_mean", "ari_base_mean",
]


def append_row(row):
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())


def done_keys():
    if not OUT.exists():
        return set()
    with open(OUT) as f:
        return {(r["network"], r["sampler"], r["alpha"], r["weighted"], r["rep"])
                for r in csv.DictReader(f)}


def run_network(name, g, y=None, n_reps=5, configs=None):
    n, m = g.vcount(), g.ecount()
    print(f"\n=== {name}: n={n:,} m={m:,}", flush=True)
    done = done_keys()

    qs, ncs, amis, aris = [], [], [], []
    for s in BASE_SEEDS:
        memb, q, nc, _ = leiden(g, s)
        qs.append(q); ncs.append(nc)
        if y is not None:
            amis.append(adjusted_mutual_info_score(y, memb))
            aris.append(adjusted_rand_score(y, memb))
    Q_base_mean, Q_base_std, Q_base_best = float(np.mean(qs)), float(np.std(qs, ddof=1)), float(max(qs))
    P0, Q_P0_orig, nc_P0, _ = leiden(g, P0_SEED)
    print(f"  baseline Q={Q_base_mean:.6f} best={Q_base_best:.6f} nc={np.mean(ncs):.1f}; "
          f"P0 Q={Q_P0_orig:.6f} nc={nc_P0}", flush=True)

    edge_arr, scores = dspar_scores(g)

    # retention-matched alpha for the calibrated sampler
    _k, _w = sparsify_paper(edge_arr, scores, 1.0, SPARSE_SEEDS[0])
    alpha_match = round(_k.shape[0] / m, 4)
    del _k, _w
    print(f"  retention-matched calibrated alpha = {alpha_match}", flush=True)

    if configs is None:
        configs = [("paper", 0.8), ("paper", 1.0), ("calibrated", 0.9),
                   ("calibrated_matched", alpha_match)]
    else:
        configs = [(s, alpha_match if a == "match" else a) for s, a in configs]

    cal_cache = {}
    for sampler, alpha in configs:
        base_sampler = "calibrated" if sampler.startswith("calibrated") else "paper"
        if base_sampler == "calibrated" and alpha not in cal_cache:
            cal_cache[alpha] = _probs_calibrated(scores, alpha)
        for rep in range(n_reps):
            ss, ls = SPARSE_SEEDS[rep], LEIDEN_SPARSE_SEEDS[rep]
            if base_sampler == "paper":
                kept, w = sparsify_paper(edge_arr, scores, alpha, ss)
            else:
                kept, w = sparsify_calibrated(edge_arr, cal_cache[alpha], ss)
            gs = build_sparse(n, kept)
            # weighted arm only for the retention-matched sampler (the other
            # weighted arms already live in results.csv); unweighted for all.
            arms = [False] if sampler != "calibrated_matched" else [True, False]
            for weighted in arms:
                key = (name, sampler, str(alpha), str(weighted), str(rep))
                if key in done:
                    continue
                ww = list(w) if weighted else None
                Q_P0_sparse = float(gs.modularity(P0, weights=ww))
                P, Q_sp, nc_P, t_ld = leiden(gs, ls, weights=(w if weighted else None))
                Q_orig_of_P = float(g.modularity(P))
                row = dict(
                    network=name, n=n, m=m, sampler=sampler, alpha=alpha,
                    weighted=weighted, rep=rep, sparsify_seed=ss, leiden_seed=ls,
                    m_sparse=int(gs.ecount()), retention=gs.ecount() / m,
                    Q_P0_orig=Q_P0_orig, Q_P0_sparse=Q_P0_sparse,
                    dQ_fixed=Q_P0_sparse - Q_P0_orig,
                    Q_sparse_leiden=Q_sp, Q_orig_of_P=Q_orig_of_P,
                    Q_base_mean=Q_base_mean, Q_base_std=Q_base_std,
                    Q_base_best=Q_base_best,
                    transfer_vs_mean=Q_orig_of_P - Q_base_mean,
                    transfer_vs_best=Q_orig_of_P - Q_base_best,
                    nc_P0=nc_P0, nc_P=nc_P, nc_base_mean=float(np.mean(ncs)),
                    t_leiden=t_ld,
                )
                if y is not None:
                    row.update(ami_P=float(adjusted_mutual_info_score(y, P)),
                               ari_P=float(adjusted_rand_score(y, P)),
                               ami_base_mean=float(np.mean(amis)),
                               ari_base_mean=float(np.mean(aris)))
                append_row(row)
                print(f"  {sampler} a={alpha} w={weighted} rep={rep}: "
                      f"ret={row['retention']:.4f} dQfix={row['dQ_fixed']:+.6f} "
                      f"transfer={row['transfer_vs_mean']:+.6f} nc={nc_P} "
                      f"t={t_ld:.1f}s", flush=True)


def main():
    args = sys.argv[1:]
    n_reps = 5
    if args and args[0].startswith("reps="):
        n_reps = int(args[0].split("=")[1]); args = args[1:]
    for name in args:
        if name == "email-Eu-core-labeled":
            g, y = load_email_labeled()
            run_network("email-Eu-core-labeled", g, y, n_reps=n_reps)
        else:
            run_network(name, load_lcc_graph(name), n_reps=n_reps)
    print("\nCONTROLS DONE", flush=True)


if __name__ == "__main__":
    main()
