#!/usr/bin/env python
"""
Exp X supplementary probe -- POST-HOC, HYPOTHESIS-GRADE (EXPLORATION.md rule 2).

NOT part of the pre-registered DESIGN.md.  It does NOT change the registered
verdict; it exists because the registered k-matched control came out EMPTY:
0 of the 20 plain restarts landed in the seeded k range [4981, 5399] (the
seeded k is below ALL 20 plain restarts), so "the k-matched gap" could not be
computed as specified, and the resolution-matched arm in run.py degenerated
(k(gamma) turned out to be DECREASING in gamma on com-Youtube over [0.7, 1.0],
so gamma* clipped at 0.999).

This probe pushes gamma ABOVE 1 -- the direction that empirically lowers k on
this graph -- to obtain plain-Leiden partitions at the seeded granularity
(k ~ 5259), scored with STANDARD modularity.  It answers one question:

    at matched community count, does plain Leiden reach the seeded Q?

If yes, the com-Youtube gain is granularity, constructively confirmed.
If no, the registered kill still stands but the mechanism is not granularity,
and that becomes a HYPOTHESIS needing its own pre-registered follow-up.

Outputs: kmatch_probe.csv, kmatch_probe.json
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg as la

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import load_lcc_graph, shape_stats, EDGE_FILE, N_ITER  # noqa: E402

TARGET_K = 5259.0          # seeded mean k (results.json m2_granularity.mean_k_seeded)
ANCHOR_GAMMA, ANCHOR_K = 1.0, 6427.6      # plain mean over the 20 restarts
SEEDS = [600, 601, 602, 603, 604]
Q_SEEDED_MEAN = 0.7294840057051575


def rb_leiden(g, seed, gamma):
    t0 = time.perf_counter()
    part = la.RBConfigurationVertexPartition(g, resolution_parameter=float(gamma))
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    memb = np.asarray(part.membership, dtype=np.int32)
    return memb, float(g.modularity(part.membership)), len(part), time.perf_counter() - t0


def main():
    g = load_lcc_graph(EDGE_FILE)
    n = g.vcount()
    print(f"n={n:,} m={g.ecount():,}", flush=True)

    rows, probes = [], []
    gamma = 1.5
    for i in range(3):
        memb, Q, k, dt = rb_leiden(g, 590 + i, gamma)
        sh = shape_stats(memb, n)
        probes.append(dict(gamma=float(gamma), k=int(k), Q=float(Q)))
        rows.append(dict(arm="probe", seed=590 + i, gamma=gamma, Q=Q, T=dt, **sh))
        print(f"  probe g={gamma:.4f} k={k} Q={Q:.6f} t={dt:.1f}s", flush=True)
        if abs(k - TARGET_K) <= 0.03 * TARGET_K:
            break
        a = (np.log(k) - np.log(ANCHOR_K)) / (np.log(gamma) - np.log(ANCHOR_GAMMA))
        if a == 0 or not np.isfinite(a):
            break
        gamma = float(np.clip(np.exp(np.log(TARGET_K / ANCHOR_K) / a + np.log(ANCHOR_GAMMA)),
                              1.0, 6.0))
    best = min(probes, key=lambda p: abs(p["k"] - TARGET_K))
    gstar = best["gamma"]
    print(f"  gamma* = {gstar:.4f} (probe k={best['k']}, target {TARGET_K:.0f})", flush=True)

    for s in SEEDS:
        memb, Q, k, dt = rb_leiden(g, s, gstar)
        sh = shape_stats(memb, n)
        rows.append(dict(arm="kmatched", seed=s, gamma=gstar, Q=Q, T=dt, **sh))
        print(f"  kmatched seed={s} g={gstar:.4f} k={k} Q={Q:.6f} t={dt:.1f}s", flush=True)

    with open(HERE / "kmatch_probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    km = [r for r in rows if r["arm"] == "kmatched"]
    Qk = np.array([r["Q"] for r in km])
    kk = np.array([r["k"] for r in km], float)
    out = dict(
        status="POST-HOC SUPPLEMENTARY PROBE -- HYPOTHESIS GRADE, NOT REGISTERED",
        target_k=TARGET_K, gamma_star=gstar, probes=probes,
        n_runs=len(km), mean_k=float(kk.mean()), min_k=float(kk.min()),
        max_k=float(kk.max()),
        Q_mean=float(Qk.mean()), Q_std=float(Qk.std(ddof=1)), Q_best=float(Qk.max()),
        Q_seeded_mean=Q_SEEDED_MEAN,
        gap_seeded_minus_kmatched=float(Q_SEEDED_MEAN - Qk.mean()),
        gap_seeded_minus_kmatched_best=float(Q_SEEDED_MEAN - Qk.max()),
        frac_largest_mean=float(np.mean([r["frac_largest"] for r in km])),
        n_comm_gt_1pct_mean=float(np.mean([r["n_comm_gt_1pct"] for r in km])),
        top5_share_mean=float(np.mean([r["top5_share"] for r in km])),
    )
    with open(HERE / "kmatch_probe.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
