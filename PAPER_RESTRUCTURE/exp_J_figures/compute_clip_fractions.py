#!/usr/bin/env python
"""
Task 4: clipped-probability fractions for the two no-replacement DSpar samplers.

(a) repo_noreplace  : p_e = clip( s_e / sum(s) * ceil(alpha*m), 0, 1 )
(b) calibrated      : p_e = min(1, lambda_alpha * s_e), lambda by bisection s.t.
                      sum_e min(1, lambda s_e) = alpha*m

Graphs: LCC of the simple undirected graph, identical loader to
PAPER_RESTRUCTURE/exp_C_true_retention_seeded/run.py.
"""
import sys
from pathlib import Path
import numpy as np
import igraph as ig
import pandas as pd

REPO = Path("/home/md724/community_detection_spectral")
OUT = Path(__file__).resolve().parent

DATASETS = {
    "email-Eu-core": REPO / "datasets/email-Eu-core/email-Eu-core.txt",
    "wiki-Vote": REPO / "datasets/wiki-Vote/wiki-Vote.txt",
    "ca-HepTh": REPO / "datasets/ca-HepTh/ca-HepTh.txt",
    "ca-CondMat": REPO / "datasets/ca-CondMat/ca-CondMat.txt",
    "email-Enron": REPO / "datasets/email-Enron/email-Enron.txt",
    "com-DBLP": REPO / "datasets/com-DBLP/com-DBLP.txt",
}
ALPHAS = [0.7, 0.8, 0.9, 0.95]


def load_graph(path):
    edges, nodes = [], set()
    with open(path) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            edges.append((u, v))
            nodes.add(u); nodes.add(v)
    node_list = sorted(nodes)
    idx = {o: i for i, o in enumerate(node_list)}
    edges = [(idx[u], idx[v]) for u, v in edges]
    g = ig.Graph(n=len(node_list), edges=edges, directed=False)
    g.simplify(multiple=True, loops=True)
    return g.connected_components().giant()


def dspar_scores(g):
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    return 1.0 / deg[e[:, 0]] + 1.0 / deg[e[:, 1]]


def probs_repo(scores, alpha):
    m = len(scores)
    n_keep = int(np.ceil(alpha * m))
    return np.clip(scores / scores.sum() * n_keep, 0.0, 1.0)


def lambda_calibrated(scores, alpha):
    m = len(scores)
    target = alpha * m
    if alpha >= 1.0:
        return np.inf
    lo, hi = 0.0, 1.0
    while np.minimum(1.0, hi * scores).sum() < target:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.minimum(1.0, mid * scores).sum() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


rows = []
for name, path in DATASETS.items():
    g = load_graph(path)
    s = dspar_scores(g)
    m = len(s)
    print(f"{name}: n={g.vcount():,} m={m:,}  s_max={s.max():.4f} s_mean={s.mean():.6f} "
          f"s_mean/s_max={s.mean()/s.max():.4f}", flush=True)
    for alpha in ALPHAS:
        # (a) repo clipped no-replace
        raw_repo = scores_raw = s / s.sum() * int(np.ceil(alpha * m))
        p_repo = np.clip(raw_repo, 0.0, 1.0)
        clip_repo = float((raw_repo >= 1.0).mean())
        # (b) calibrated
        lam = lambda_calibrated(s, alpha)
        clip_cal = float((lam * s >= 1.0).mean())
        p_cal = np.minimum(1.0, lam * s)
        rows.append(dict(
            dataset=name, n=g.vcount(), m=m, alpha=alpha,
            clip_frac_repo_noreplace=clip_repo,
            E_retention_repo_noreplace=float(p_repo.sum() / m),
            clip_frac_calibrated=clip_cal,
            E_retention_calibrated=float(p_cal.sum() / m),
            lambda_calibrated=float(lam),
            s_max=float(s.max()), s_mean=float(s.mean()),
            unclipped_alpha_threshold=float(s.mean() / s.max()),
        ))
        print(f"   alpha={alpha}: clip_repo={clip_repo:.4f} (E[ret]={p_repo.sum()/m:.4f})  "
              f"clip_cal={clip_cal:.4f} (E[ret]={p_cal.sum()/m:.4f})", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "clip_fractions.csv", index=False)
print("\nWrote", OUT / "clip_fractions.csv")
print("\nrepo_noreplace clip fraction range: %.4f - %.4f" %
      (df.clip_frac_repo_noreplace.min(), df.clip_frac_repo_noreplace.max()))
print("calibrated clip fraction range: %.4f - %.4f" %
      (df.clip_frac_calibrated.min(), df.clip_frac_calibrated.max()))
sub = df[(df.dataset == "email-Eu-core") & (df.alpha == 0.9)]
print("email-Eu-core calibrated alpha=0.9 clip fraction: %.4f" % sub.clip_frac_calibrated.iloc[0])
