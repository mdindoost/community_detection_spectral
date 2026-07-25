#!/usr/bin/env python
"""
Control: is any change vs the published tables caused by taking the LCC, or by
the Leiden implementation (published run used igraph community_leiden; this
re-run uses leidenalg ModularityVertexPartition as specified)?

Runs the identical measurement on (a) the FULL simple graph and (b) the LCC,
with the same leidenalg partition routine, and additionally on the LCC with
three different Leiden seeds to check partition-seed sensitivity of dF_obs.

Output: control.csv + printed table.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import igraph as ig
import networkx as nx

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from run import (DATASETS, ALPHA, N_SEEDS, SEED_BASE, LEIDEN_SEED,  # noqa: E402
                 load_lcc, leiden, g_term, dspar_sparsify)


def measure(g, leiden_seed):
    n, m = g.vcount(), g.ecount()
    memb, _ = leiden(g, leiden_seed)
    memb = np.asarray(memb, dtype=np.int64)
    n_comm = int(memb.max()) + 1
    Q_orig = float(g.modularity(memb.tolist()))
    E = np.asarray([(min(u, v), max(u, v)) for u, v in g.get_edgelist()], dtype=np.int64)
    deg = np.asarray(g.degree(), dtype=np.float64)
    scores = 1.0 / deg[E[:, 0]] + 1.0 / deg[E[:, 1]]
    intra = memb[E[:, 0]] == memb[E[:, 1]]
    n1, n2 = int(intra.sum()), m - int(intra.sum())
    mu_i, mu_o = float(scores[intra].mean()), float(scores[~intra].mean())
    F_orig, G_orig = n1 / m, g_term(memb, deg, m, n_comm)
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(E)}
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(n))
    G_nx.add_edges_from((int(a), int(b)) for a, b in E)

    dQ, dF, dG, ret, rat = [], [], [], [], []
    for run in range(N_SEEDS):
        Gw = dspar_sparsify(G_nx, retention=ALPHA, method="paper", seed=SEED_BASE + run)
        kept = np.zeros(m, dtype=bool)
        for u, v in Gw.edges():
            kept[idx[(min(u, v), max(u, v))]] = True
        m_sp = int(kept.sum())
        gs = ig.Graph(n=n, edges=[tuple(map(int, e)) for e in E[kept]], directed=False)
        dQ.append(float(gs.modularity(memb.tolist())) - Q_orig)
        pi = int(kept[intra].sum())
        po = int(kept[~intra].sum())
        dF.append(pi / m_sp - F_orig)
        dG.append(g_term(memb, np.asarray(gs.degree(), dtype=np.float64), m_sp, n_comm) - G_orig)
        ret.append(m_sp / m)
        rat.append((po / n2) / (pi / n1))
    f = lambda a: (float(np.mean(a)), float(np.std(a)))  # noqa: E731
    return dict(n=n, m=m, k=n_comm, Q_orig=Q_orig, delta=mu_i - mu_o,
                ratio_pred=mu_o / mu_i, dQ=f(dQ), dF=f(dF), mdG=f([-x for x in dG]),
                ret=f(ret), ratio_obs=f(rat))


def main():
    names = sys.argv[1:] or [d for d, _ in DATASETS]
    rows = []
    print(f"{'dataset':20s} {'variant':16s} {'m':>9s} {'Q_orig':>8s} {'delta':>9s} "
          f"{'dQ_fixed':>10s} {'dF_obs':>10s} {'-dG_obs':>10s} {'ratio_obs':>10s}")
    for name, path in DATASETS:
        if name not in names:
            continue
        g_lcc, n_full, m_full = load_lcc(path)
        # rebuild the FULL simple graph
        edges, nodes = [], set()
        with open(path) as fh:
            for line in fh:
                if not line or line[0] == "#":
                    continue
                p = line.split()
                if len(p) < 2:
                    continue
                try:
                    u, v = int(p[0]), int(p[1])
                except ValueError:
                    continue
                if u != v:
                    edges.append((u, v))
                    nodes.update((u, v))
        nl = sorted(nodes)
        ix = {o: i for i, o in enumerate(nl)}
        g_full = ig.Graph(n=len(nl), edges=[(ix[u], ix[v]) for u, v in edges], directed=False)
        g_full.simplify(multiple=True, loops=True)

        variants = [("FULL(seed42)", g_full, LEIDEN_SEED), ("LCC(seed42)", g_lcc, LEIDEN_SEED),
                    ("LCC(seed7)", g_lcc, 7), ("LCC(seed1234)", g_lcc, 1234)]
        for vname, g, s in variants:
            r = measure(g, s)
            rows.append(dict(dataset=name, variant=vname, **{
                "n": r["n"], "m": r["m"], "k": r["k"], "Q_orig": r["Q_orig"],
                "delta": r["delta"], "ratio_pred": r["ratio_pred"],
                "dQ_fixed_mean": r["dQ"][0], "dQ_fixed_std": r["dQ"][1],
                "dF_obs_mean": r["dF"][0], "dF_obs_std": r["dF"][1],
                "mdG_obs_mean": r["mdG"][0], "mdG_obs_std": r["mdG"][1],
                "retention_mean": r["ret"][0], "ratio_obs_mean": r["ratio_obs"][0]}))
            print(f"{name:20s} {vname:16s} {r['m']:9,d} {r['Q_orig']:8.4f} "
                  f"{r['delta']:+9.5f} {r['dQ'][0]:+10.4f} {r['dF'][0]:+10.4f} "
                  f"{r['mdG'][0]:+10.4f} {r['ratio_obs'][0]:10.4f}", flush=True)
        with open(OUT / "control.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
