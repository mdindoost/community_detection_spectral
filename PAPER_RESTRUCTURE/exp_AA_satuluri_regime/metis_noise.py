#!/usr/bin/env python
"""Exp AA add-on: how big is Metis run-to-run noise?

Arm B scores Metis once per graph (seed 0). The Arm B headline is a positive
honest-transfer dQ of ~+0.005..+0.011 and dAMI of ~+0.12..+0.15 at d_avg>=50,
so we must know the spread of Metis itself. This runs Metis with 10 different
option seeds on the ORIGINAL graph and on the L-Spar graph, per cell, and also
records the L-Spar sparsification seed-invariance (L-Spar is deterministic).
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import (gen_lfr, graph_from_E, graph_meta, edge_jaccard, LSpar,
                 dspar_scores, sparsify_edges, subgraph_from_eids,
                 metis_partition, evaluate, append_row, log, HERE)

SEEDS = list(range(10))
CELLS = [(0.5, d, s) for d in (25, 50, 100, 200) for s in (1, 2, 3)]
TARGETS = [0.5, 0.2, 0.15]

out = HERE / "metis_noise.csv"
for mu, d, ls in CELLS:
    try:
        E, y, meta = gen_lfr(mu, d, ls)
    except Exception as ex:
        log(f"[gen-fail] {mu} {d} {ls}: {ex}")
        continue
    g = graph_from_E(E)
    gm = graph_meta(g, y)
    k = gm["n_comm_planted"]
    log(f"\n=== mu{mu}_d{d}_s{ls} m={gm['m']:,} d_avg={gm['d_avg_real']:.1f} k={k}")

    base = [evaluate(g, metis_partition(g, k, seed=s)[0], y) for s in SEEDS]
    Qb = np.array([e["Q_orig"] for e in base])
    Ab = np.array([e["AMI"] for e in base])
    log(f"  ORIGINAL   Q={Qb.mean():.5f}+-{Qb.std():.5f} (range {np.ptp(Qb):.5f})  "
        f"AMI={Ab.mean():.4f}+-{Ab.std():.4f} (range {np.ptp(Ab):.4f})")

    Ej, J, deg = edge_jaccard(g)
    lsp = LSpar(g, J=J, E=Ej, deg=deg)
    ds = dspar_scores(g)
    for kind, t in itertools.product(("lspar", "dspar", "random"), TARGETS):
        eids, realized, knob, status, _ = sparsify_edges(
            g, kind, t, 500 + ls, lspar=lsp, dscores=ds)
        gs = subgraph_from_eids(g, eids)
        sp_ = [evaluate(g, metis_partition(gs, k, seed=s)[0], y) for s in SEEDS]
        Qs = np.array([e["Q_orig"] for e in sp_])
        As = np.array([e["AMI"] for e in sp_])
        row = dict(mu=mu, d_nom=d, lfr_seed=ls, d_avg=gm["d_avg_real"],
                   m=gm["m"], k=k, sparsifier=kind, target_ret=t,
                   realized_ret=round(realized, 5), status=status,
                   n_metis_seeds=len(SEEDS),
                   Q_base_mean=Qb.mean(), Q_base_std=Qb.std(),
                   Q_base_min=Qb.min(), Q_base_max=Qb.max(),
                   AMI_base_mean=Ab.mean(), AMI_base_std=Ab.std(),
                   AMI_base_min=Ab.min(), AMI_base_max=Ab.max(),
                   Q_sparse_mean=Qs.mean(), Q_sparse_std=Qs.std(),
                   Q_sparse_min=Qs.min(), Q_sparse_max=Qs.max(),
                   AMI_sparse_mean=As.mean(), AMI_sparse_std=As.std(),
                   AMI_sparse_min=As.min(), AMI_sparse_max=As.max(),
                   dQ=Qs.mean() - Qb.mean(), dAMI=As.mean() - Ab.mean(),
                   dQ_over_pooled_sd=(Qs.mean() - Qb.mean())
                   / max(1e-12, np.sqrt(0.5 * (Qs.var() + Qb.var()))),
                   dAMI_over_pooled_sd=(As.mean() - Ab.mean())
                   / max(1e-12, np.sqrt(0.5 * (As.var() + Ab.var()))),
                   worstcase_dQ=Qs.min() - Qb.max(),
                   worstcase_dAMI=As.min() - Ab.max())
        append_row(out, row)
        log(f"  {kind:7s} t={t:<5} ret={realized:.4f} "
            f"Q {Qb.mean():.5f}->{Qs.mean():.5f} (dQ={row['dQ']:+.5f}, "
            f"worst {row['worstcase_dQ']:+.5f})  "
            f"AMI {Ab.mean():.4f}->{As.mean():.4f} (dAMI={row['dAMI']:+.4f}, "
            f"worst {row['worstcase_dAMI']:+.4f})")
        del gs
    del g, lsp, Ej, J
