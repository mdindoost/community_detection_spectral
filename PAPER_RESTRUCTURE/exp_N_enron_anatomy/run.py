#!/usr/bin/env python
"""
Experiment N: anatomy of the email-Enron DSpar-seeding gain.

Established elsewhere (exp_C / exp_E / exp_K): on email-Enron ONLY, seeding Leiden
with a partition found on a DSpar-sparsified graph and refining on the ORIGINAL
graph beats runtime-matched plain-Leiden restarts by +0.006..+0.015 modularity,
8/8 configurations, both samplers. No graph statistic predicts why Enron.

This script dissects WHERE that gain lives:

  Step 1  Generate 5 baseline partitions (seeds 100-104) and 5 seeded partitions
          (calibrated DSpar alpha=0.9, spar seeds 200-204 / leiden seeds 300-304),
          plus 20 extra plain restarts (seeds 900-919).
  Step 2  Community-level diff (best baseline vs best seeded): Hungarian matching on
          overlap, per-community modularity contribution L_c/m - (d_c/2m)^2,
          top-10 communities by |Delta contribution|.
  Step 3  Merge / split / reshuffle structure from the contingency table.
  Step 4  Hub involvement: degree distribution of moved vs unmoved nodes.
  Step 5  Landscape probe: pairwise AMI within baselines, within seeded, across;
          20 extra restarts -> best Q and max AMI to the seeded solution.
  Step 6  Structural fingerprint of the top-gain communities, and the sparsified-graph
          test: are the groups merged-in-baseline / split-in-seeded actually
          disconnected (or weakly connected) in G_sparse?

Loaders and samplers are copied verbatim from
PAPER_RESTRUCTURE/exp_C_true_retention_seeded/run.py.

Outputs (this directory):
  partitions.npz            all memberships + the alpha=0.9 sparsified edge sets
  partition_quality.csv     Q / k / runtime for every generated partition
  community_diff.csv        every matched/unmatched community pair with dQ contribution
  contingency_merge_split.csv
  moved_nodes_degree.csv
  landscape_ami.csv
  topgain_fingerprint.csv
  sparse_split_test.csv
  SUMMARY.md
"""

import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg as la
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

REPO = Path("/home/md724/community_detection_spectral")
OUT = Path(__file__).resolve().parent
ENRON = REPO / "datasets/email-Enron/email-Enron.txt"

ALPHA = 0.90                 # calibrated true retention (exp_K calibration point)
SAMPLER = "calibrated"
N_SEEDS = 5
BASE_SEEDS = [100 + i for i in range(N_SEEDS)]
SPAR_SEEDS = [200 + i for i in range(N_SEEDS)]
LEID_SEEDS = [300 + i for i in range(N_SEEDS)]
EXTRA_SEEDS = list(range(900, 920))
N_ITER = 2


# --------------------------------------------------------------------------
# exp_C loaders / samplers (verbatim)
# --------------------------------------------------------------------------
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
            nodes.add(u)
            nodes.add(v)
    node_list = sorted(nodes)
    idx = {o: i for i, o in enumerate(node_list)}
    edges = [(idx[u], idx[v]) for u, v in edges]
    g = ig.Graph(n=len(node_list), edges=edges, directed=False)
    g.simplify(multiple=True, loops=True)
    g = g.connected_components().giant()
    return g


def dspar_scores(g):
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    return e, 1.0 / deg[e[:, 0]] + 1.0 / deg[e[:, 1]]


def _probs_calibrated(scores, alpha):
    m = len(scores)
    target = alpha * m
    if alpha >= 1.0:
        return np.ones(m)
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
    return np.minimum(1.0, 0.5 * (lo + hi) * scores)


def sparsify(g, edge_arr, scores, alpha, seed):
    probs = _probs_calibrated(scores, alpha)
    rs = np.random.RandomState(seed)
    keep = rs.random_sample(len(scores)) < probs
    kept = edge_arr[keep]
    gs = ig.Graph(n=g.vcount(), edges=[tuple(x) for x in kept], directed=False)
    return gs, kept, kept.shape[0] / edge_arr.shape[0]


def leiden(g, seed, initial_membership=None):
    t0 = time.perf_counter()
    if initial_membership is None:
        part = la.ModularityVertexPartition(g)
    else:
        _, memb = np.unique(np.asarray(initial_membership), return_inverse=True)
        part = la.ModularityVertexPartition(g, initial_membership=memb.tolist())
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    dt = time.perf_counter() - t0
    return np.asarray(part.membership), part.modularity, len(part), dt


# --------------------------------------------------------------------------
# Modularity bookkeeping
# --------------------------------------------------------------------------
def community_contributions(g, memb, edge_arr=None, deg=None):
    """Per-community (L_c/m, (d_c/2m)^2, contribution). Contributions sum to Q."""
    m = g.ecount()
    memb = np.asarray(memb)
    if edge_arr is None:
        edge_arr = np.asarray(g.get_edgelist(), dtype=np.int64)
    if deg is None:
        deg = np.asarray(g.degree(), dtype=np.float64)
    k = int(memb.max()) + 1
    cu, cv = memb[edge_arr[:, 0]], memb[edge_arr[:, 1]]
    intra = cu == cv
    L = np.bincount(cu[intra], minlength=k).astype(np.float64)
    dvol = np.bincount(memb, weights=deg, minlength=k)
    contrib = L / m - (dvol / (2.0 * m)) ** 2
    return L, dvol, contrib


def _gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    nn = len(x)
    if nn == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, nn + 1) - nn - 1).dot(x) / (nn * x.sum()))


def relabel(memb):
    _, out = np.unique(np.asarray(memb), return_inverse=True)
    return out


def contingency(a, b):
    a, b = relabel(a), relabel(b)
    ka, kb = a.max() + 1, b.max() + 1
    M = coo_matrix((np.ones(len(a)), (a, b)), shape=(ka, kb)).tocsr()
    return M


# --------------------------------------------------------------------------
def step1_generate(g):
    edge_arr, scores = dspar_scores(g)
    rows, store = [], {}

    print("  baselines...", flush=True)
    for s in BASE_SEEDS:
        memb, Q, k, dt = leiden(g, seed=s)
        store[f"base_{s}"] = memb
        rows.append(dict(kind="baseline", seed=s, spar_seed="", alpha="",
                         retention="", Q=Q, k=k, seconds=dt, Q_sparse_transfer=""))
        print(f"    base seed={s}: Q={Q:.6f} k={k} t={dt:.2f}s", flush=True)

    print("  seeded (calibrated alpha=0.90)...", flush=True)
    for ss, ls in zip(SPAR_SEEDS, LEID_SEEDS):
        t0 = time.perf_counter()
        gs, kept, ret = sparsify(g, edge_arr, scores, ALPHA, ss)
        t_spar = time.perf_counter() - t0
        memb_s, _, k_s, t_ls = leiden(gs, seed=ls)
        q_raw = g.modularity(memb_s.tolist())
        memb_f, Q, k, t_sd = leiden(g, seed=ls, initial_membership=memb_s)
        store[f"seeded_{ls}"] = memb_f
        store[f"sparsepart_{ls}"] = memb_s
        store[f"sparseedges_{ss}"] = kept
        rows.append(dict(kind="seeded", seed=ls, spar_seed=ss, alpha=ALPHA,
                         retention=ret, Q=Q, k=k, seconds=t_spar + t_ls + t_sd,
                         Q_sparse_transfer=q_raw))
        print(f"    seeded spar={ss} leiden={ls}: ret={ret:.4f} Qraw={q_raw:.6f} "
              f"Q={Q:.6f} k={k} t={t_spar+t_ls+t_sd:.2f}s", flush=True)

    print("  20 extra plain restarts (seeds 900-919)...", flush=True)
    for s in EXTRA_SEEDS:
        memb, Q, k, dt = leiden(g, seed=s)
        store[f"extra_{s}"] = memb
        rows.append(dict(kind="extra_restart", seed=s, spar_seed="", alpha="",
                         retention="", Q=Q, k=k, seconds=dt, Q_sparse_transfer=""))
        print(f"    extra seed={s}: Q={Q:.6f} k={k} t={dt:.2f}s", flush=True)

    write_csv(OUT / "partition_quality.csv", rows)
    np.savez_compressed(OUT / "partitions.npz", **store)
    return store, rows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
def step2_community_diff(g, B, S, edge_arr, deg, m):
    """Hungarian match on overlap; per-community contribution delta."""
    B, S = relabel(B), relabel(S)
    kB, kS = B.max() + 1, S.max() + 1
    M = contingency(B, S).toarray()

    LB, dB, cB = community_contributions(g, B, edge_arr, deg)
    LS, dS, cS = community_contributions(g, S, edge_arr, deg)

    # Hungarian on overlap (maximize)
    r, c = linear_sum_assignment(-M)
    matched_nodes = M[r, c].sum()
    n = g.vcount()

    # Jaccard for each matched pair
    sizeB = np.bincount(B, minlength=kB)
    sizeS = np.bincount(S, minlength=kS)

    pair_rows = []
    matchedB, matchedS = set(), set()
    for i, j in zip(r, c):
        if M[i, j] == 0:
            continue
        matchedB.add(int(i)); matchedS.add(int(j))
        inter = M[i, j]
        jac = inter / (sizeB[i] + sizeS[j] - inter)
        pair_rows.append(dict(
            base_comm=int(i), seeded_comm=int(j), size_base=int(sizeB[i]),
            size_seeded=int(sizeS[j]), overlap=int(inter), jaccard=float(jac),
            L_base=float(LB[i]), L_seeded=float(LS[j]),
            vol_base=float(dB[i]), vol_seeded=float(dS[j]),
            contrib_base=float(cB[i]), contrib_seeded=float(cS[j]),
            d_contrib=float(cS[j] - cB[i]), status="matched"))
    for i in range(kB):
        if i not in matchedB:
            pair_rows.append(dict(base_comm=int(i), seeded_comm=-1,
                                  size_base=int(sizeB[i]), size_seeded=0, overlap=0,
                                  jaccard=0.0, L_base=float(LB[i]), L_seeded=0.0,
                                  vol_base=float(dB[i]), vol_seeded=0.0,
                                  contrib_base=float(cB[i]), contrib_seeded=0.0,
                                  d_contrib=float(-cB[i]), status="base_only"))
    for j in range(kS):
        if j not in matchedS:
            pair_rows.append(dict(base_comm=-1, seeded_comm=int(j), size_base=0,
                                  size_seeded=int(sizeS[j]), overlap=0, jaccard=0.0,
                                  L_base=0.0, L_seeded=float(LS[j]), vol_base=0.0,
                                  vol_seeded=float(dS[j]), contrib_base=0.0,
                                  contrib_seeded=float(cS[j]),
                                  d_contrib=float(cS[j]), status="seeded_only"))
    pair_rows.sort(key=lambda x: -abs(x["d_contrib"]))
    write_csv(OUT / "community_diff.csv", pair_rows)

    # moved nodes: node is "unmoved" iff its base comm and seeded comm are a matched pair
    pair_map = {int(i): int(j) for i, j in zip(r, c) if M[i, j] > 0}
    tgt = np.full(kB, -1, dtype=np.int64)
    for i, j in pair_map.items():
        tgt[i] = j
    moved = tgt[B] != S

    dQ_total = float(cS.sum() - cB.sum())
    dsorted = np.array(sorted((abs(x["d_contrib"]) for x in pair_rows), reverse=True))
    signed = np.array([x["d_contrib"] for x in pair_rows])
    pos = signed[signed > 0]
    pos_sorted = np.sort(pos)[::-1]

    stats = dict(
        kB=int(kB), kS=int(kS), n=int(n), m=int(m),
        Q_base=float(cB.sum()), Q_seeded=float(cS.sum()), dQ=dQ_total,
        matched_pairs=len(pair_map),
        base_only=int(kB - len(matchedB)), seeded_only=int(kS - len(matchedS)),
        nodes_moved=int(moved.sum()), frac_moved=float(moved.mean()),
        # concentration
        top1_abs_share=float(dsorted[0] / abs(dQ_total)) if dQ_total else 0.0,
        top5_abs_share=float(dsorted[:5].sum() / abs(dQ_total)) if dQ_total else 0.0,
        top10_abs_share=float(dsorted[:10].sum() / abs(dQ_total)) if dQ_total else 0.0,
        sum_abs_dcontrib=float(np.abs(signed).sum()),
        sum_pos_dcontrib=float(pos.sum()),
        sum_neg_dcontrib=float(signed[signed < 0].sum()),
        n_pos=int((signed > 0).sum()), n_neg=int((signed < 0).sum()),
        top5_pos_share_of_pos=float(pos_sorted[:5].sum() / pos.sum()) if len(pos) else 0.0,
        top10_pos_share_of_pos=float(pos_sorted[:10].sum() / pos.sum()) if len(pos) else 0.0,
        # cancellation: how much churn per unit of net gain
        churn_ratio=float(np.abs(signed).sum() / abs(dQ_total)) if dQ_total else 0.0,
        top1_share_of_absmass=float(dsorted[0] / np.abs(signed).sum()),
        top5_share_of_absmass=float(dsorted[:5].sum() / np.abs(signed).sum()),
        top10_share_of_absmass=float(dsorted[:10].sum() / np.abs(signed).sum()),
        gini_abs_dcontrib=float(_gini(dsorted)),
    )
    return stats, pair_rows, moved, M, B, S, (cB, cS, LB, LS, dB, dS)


# --------------------------------------------------------------------------
def step3_merge_split(B, S, M, sizeB, sizeS, purity=0.90):
    """Classify each baseline community by how it maps into the seeded partition."""
    kB, kS = M.shape
    rows = []
    n_split, n_merge, n_identical, n_reshuffle, n_dissolved = 0, 0, 0, 0, 0
    # per baseline community
    for i in range(kB):
        row = M[i]
        nz = np.nonzero(row)[0]
        frac = row[nz] / sizeB[i]
        order = np.argsort(-frac)
        nz, frac = nz[order], frac[order]
        # how many seeded pieces hold >=5% of this base community
        pieces = int((frac >= 0.05).sum())
        top_frac = float(frac[0])
        j = int(nz[0])
        # how much of the receiving seeded community comes from this base community
        recv_frac = float(row[j] / sizeS[j])
        if top_frac >= purity and recv_frac >= purity:
            cls = "identical"
        elif top_frac >= purity and recv_frac < purity:
            cls = "merged_into"      # this base comm went whole into a bigger seeded comm
        elif top_frac < purity and pieces >= 2:
            cls = "split"
        else:
            cls = "reshuffled"
        rows.append(dict(base_comm=int(i), size_base=int(sizeB[i]), n_pieces_ge5pct=pieces,
                         top_seeded_comm=j, top_frac=top_frac,
                         frac_of_target_seeded=recv_frac,
                         size_top_seeded=int(sizeS[j]), classification=cls))
    cnt = Counter(x["classification"] for x in rows)
    # merges: seeded communities receiving >=2 whole base communities (>=90% each)
    merge_groups = defaultdict(list)
    for x in rows:
        if x["top_frac"] >= purity:
            merge_groups[x["top_seeded_comm"]].append(x["base_comm"])
    pure_merges = {j: v for j, v in merge_groups.items() if len(v) >= 2}
    # splits: base communities broken into >=2 seeded pieces each >=10%
    pure_splits = []
    for i in range(kB):
        row = M[i]
        nz = np.nonzero(row)[0]
        frac = row[nz] / sizeB[i]
        big = nz[frac >= 0.10]
        if len(big) >= 2:
            pure_splits.append((int(i), [int(x) for x in big],
                                float(row[big].sum() / sizeB[i])))
    write_csv(OUT / "contingency_merge_split.csv", rows)
    return rows, cnt, pure_merges, pure_splits


# --------------------------------------------------------------------------
def step4_hubs(g, deg, moved, B, S, sizeB, sizeS):
    n = g.vcount()
    q99 = np.quantile(deg, 0.99)
    hub = deg >= q99
    md, ud = deg[moved], deg[~moved]

    def pct(a, p):
        return float(np.percentile(a, p)) if len(a) else float("nan")

    rows = [dict(group="moved", count=int(moved.sum()),
                 mean_deg=float(md.mean()) if len(md) else 0.0,
                 median_deg=pct(md, 50), p90_deg=pct(md, 90), p99_deg=pct(md, 99),
                 max_deg=float(md.max()) if len(md) else 0.0,
                 n_top1pct_hubs=int(hub[moved].sum()),
                 frac_of_group_that_is_hub=float(hub[moved].mean()) if len(md) else 0.0),
            dict(group="unmoved", count=int((~moved).sum()),
                 mean_deg=float(ud.mean()) if len(ud) else 0.0,
                 median_deg=pct(ud, 50), p90_deg=pct(ud, 90), p99_deg=pct(ud, 99),
                 max_deg=float(ud.max()) if len(ud) else 0.0,
                 n_top1pct_hubs=int(hub[~moved].sum()),
                 frac_of_group_that_is_hub=float(hub[~moved].mean()) if len(ud) else 0.0),
            dict(group="all", count=n, mean_deg=float(deg.mean()),
                 median_deg=pct(deg, 50), p90_deg=pct(deg, 90), p99_deg=pct(deg, 99),
                 max_deg=float(deg.max()), n_top1pct_hubs=int(hub.sum()),
                 frac_of_group_that_is_hub=float(hub.mean()))]
    write_csv(OUT / "moved_nodes_degree.csv", rows)

    # move rate by degree decile
    dec_rows = []
    edges = np.unique(np.quantile(deg, np.linspace(0, 1, 11)))
    binidx = np.clip(np.digitize(deg, edges[1:-1]), 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        sel = binidx == b
        if sel.sum() == 0:
            continue
        dec_rows.append(dict(deg_bin=f"[{edges[b]:.0f},{edges[b+1]:.0f}]",
                             n_nodes=int(sel.sum()),
                             mean_deg=float(deg[sel].mean()),
                             frac_moved=float(moved[sel].mean())))
    # hub location
    hub_stats = dict(
        deg_threshold_top1pct=float(q99), n_hubs=int(hub.sum()),
        hub_move_rate=float(moved[hub].mean()),
        nonhub_move_rate=float(moved[~hub].mean()),
        hub_enrichment_in_moved=float(hub[moved].mean() / hub.mean()) if moved.sum() else 0.0,
        mean_deg_moved=float(deg[moved].mean()) if moved.sum() else 0.0,
        mean_deg_unmoved=float(deg[~moved].mean()),
        hubs_in_base_ncomms=int(len(set(B[hub].tolist()))),
        hubs_in_seeded_ncomms=int(len(set(S[hub].tolist()))),
        mean_size_base_comm_of_hub=float(sizeB[B[hub]].mean()),
        mean_size_seeded_comm_of_hub=float(sizeS[S[hub]].mean()),
    )
    return rows, dec_rows, hub_stats, hub


# --------------------------------------------------------------------------
def step5_landscape(store, quality_rows):
    names_b = [f"base_{s}" for s in BASE_SEEDS]
    names_s = [f"seeded_{s}" for s in LEID_SEEDS]
    names_e = [f"extra_{s}" for s in EXTRA_SEEDS]
    P = {k: store[k] for k in names_b + names_s + names_e}
    rows = []

    def amis(A, Bn, label):
        vals = []
        for i, a in enumerate(A):
            for j, b in enumerate(Bn):
                if A is Bn and j <= i:
                    continue
                v = adjusted_mutual_info_score(P[a], P[b])
                vals.append(v)
                rows.append(dict(comparison=label, part_a=a, part_b=b, ami=float(v)))
        return np.array(vals)

    v_bb = amis(names_b, names_b, "baseline-baseline")
    v_ss = amis(names_s, names_s, "seeded-seeded")
    v_bs = amis(names_b, names_s, "baseline-seeded")
    v_eb = amis(names_e, names_b, "extra-baseline")
    v_es = amis(names_e, names_s, "extra-seeded")
    v_ee = amis(names_e, names_e, "extra-extra")
    write_csv(OUT / "landscape_ami.csv", rows)

    qmap = {}
    for r in quality_rows:
        key = (("base_" if r["kind"] == "baseline" else
                "seeded_" if r["kind"] == "seeded" else "extra_") + str(r["seed"]))
        qmap[key] = r["Q"]
    Qb = np.array([qmap[k] for k in names_b])
    Qs = np.array([qmap[k] for k in names_s])
    Qe = np.array([qmap[k] for k in names_e])
    best_seeded = names_s[int(np.argmax(Qs))]
    ami_extra_to_best = np.array([adjusted_mutual_info_score(P[k], P[best_seeded])
                                  for k in names_e])
    ami_base_to_best = np.array([adjusted_mutual_info_score(P[k], P[best_seeded])
                                 for k in names_b])

    def d(v):
        return dict(mean=float(v.mean()), std=float(v.std()),
                    min=float(v.min()), max=float(v.max()))

    stats = dict(
        ami_base_base=d(v_bb), ami_seeded_seeded=d(v_ss), ami_base_seeded=d(v_bs),
        ami_extra_extra=d(v_ee), ami_extra_base=d(v_eb), ami_extra_seeded=d(v_es),
        Q_base=dict(mean=float(Qb.mean()), std=float(Qb.std()), best=float(Qb.max()),
                    min=float(Qb.min())),
        Q_seeded=dict(mean=float(Qs.mean()), std=float(Qs.std()), best=float(Qs.max()),
                      min=float(Qs.min())),
        Q_extra20=dict(mean=float(Qe.mean()), std=float(Qe.std()), best=float(Qe.max()),
                       min=float(Qe.min())),
        Q_best25_plain=float(max(Qb.max(), Qe.max())),
        best_seeded_partition=best_seeded,
        max_ami_extra_to_best_seeded=float(ami_extra_to_best.max()),
        mean_ami_extra_to_best_seeded=float(ami_extra_to_best.mean()),
        max_ami_base_to_best_seeded=float(ami_base_to_best.max()),
        gap_bestseeded_minus_best25plain=float(Qs.max() - max(Qb.max(), Qe.max())),
        n_extra_beating_seeded_mean=int((Qe > Qs.mean()).sum()),
        n_extra_beating_seeded_best=int((Qe > Qs.max()).sum()),
    )
    return stats


# --------------------------------------------------------------------------
def step6_fingerprint(g, edge_arr, deg, B, S, M, pair_rows, store, cB, cS, LB, LS, dB, dS):
    """Structural fingerprint of top-gain seeded communities + sparse-graph split test."""
    m = g.ecount()
    sizeB = np.bincount(B)
    sizeS = np.bincount(S)
    dprod = deg[edge_arr[:, 0]] * deg[edge_arr[:, 1]]

    def comm_fingerprint(memb, c, label):
        nodes = np.nonzero(memb == c)[0]
        sub = g.subgraph(nodes.tolist())
        cu, cv = memb[edge_arr[:, 0]], memb[edge_arr[:, 1]]
        intra = (cu == c) & (cv == c)
        inter = ((cu == c) ^ (cv == c))
        e_in, e_out = int(intra.sum()), int(inter.sum())
        hb = (float(dprod[inter].mean()) / float(dprod[intra].mean())
              if e_in and e_out else float("nan"))
        vol = float(deg[nodes].sum())
        cond = e_out / min(vol, 2 * m - vol) if min(vol, 2 * m - vol) > 0 else float("nan")
        return dict(which=label, comm=int(c), size=len(nodes), edges_in=e_in,
                    edges_out=e_out, conductance=float(cond),
                    density=float(2 * e_in / (len(nodes) * (len(nodes) - 1)))
                    if len(nodes) > 1 else 0.0,
                    transitivity=float(sub.transitivity_undirected(mode="zero")),
                    avg_local_clustering=float(sub.transitivity_avglocal_undirected(mode="zero")),
                    mean_degree_in_G=float(deg[nodes].mean()),
                    max_degree_in_G=float(deg[nodes].max()),
                    hub_bridge_ratio=hb,
                    mean_dprod_intra=float(dprod[intra].mean()) if e_in else float("nan"),
                    mean_dprod_inter=float(dprod[inter].mean()) if e_out else float("nan"))

    top = [x for x in pair_rows if x["d_contrib"] > 0][:10]
    fp_rows = []
    for x in top:
        if x["seeded_comm"] >= 0:
            r = comm_fingerprint(S, x["seeded_comm"], "seeded")
            r.update(d_contrib=x["d_contrib"], paired_base_comm=x["base_comm"],
                     jaccard=x["jaccard"])
            fp_rows.append(r)
        if x["base_comm"] >= 0:
            r = comm_fingerprint(B, x["base_comm"], "baseline")
            r.update(d_contrib=x["d_contrib"], paired_base_comm=x["base_comm"],
                     jaccard=x["jaccard"])
            fp_rows.append(r)
    # global reference: median fingerprint of all communities with size>=20
    for memb, lab in ((B, "baseline_ALL_median"), (S, "seeded_ALL_median")):
        vals = defaultdict(list)
        for c in np.nonzero(np.bincount(memb) >= 20)[0]:
            r = comm_fingerprint(memb, c, lab)
            for k, v in r.items():
                if k in ("which",):
                    continue
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    vals[k].append(float(v))
        row = dict(which=lab, comm=-1,
                   size=int(np.median(vals["size"])) if vals["size"] else -1)
        for k in ("conductance", "density", "transitivity", "avg_local_clustering",
                  "mean_degree_in_G", "hub_bridge_ratio"):
            row[k] = float(np.median(vals[k])) if vals[k] else float("nan")
        fp_rows.append(row)
    write_csv(OUT / "topgain_fingerprint.csv", fp_rows)

    # ---- sparse-graph split test -----------------------------------------
    # For baseline communities that the seeded partition SPLITS: are the pieces
    # disconnected / weakly connected in G_sparse but well connected in G?
    best_spar_seed = SPAR_SEEDS[LEID_SEEDS.index(int(BEST_SEEDED_SEED))]
    kept = store[f"sparseedges_{best_spar_seed}"]
    gs = ig.Graph(n=g.vcount(), edges=[tuple(x) for x in kept], directed=False)

    split_rows = []

    def group_test(parent, child, c_parent, c_child, sizeP, direction, Mx, sign=1.0):
        """For each parent community broken into >=2 child pieces, compare the
        piece-boundary in G vs G_sparse."""
        out = []
        for i in range(Mx.shape[0]):
            row = Mx[i]
            nz = np.nonzero(row)[0]
            frac = row[nz] / sizeP[i]
            big = nz[frac >= 0.10]
            if len(big) < 2 or sizeP[i] < 20:
                continue
            nodes = np.nonzero(parent == i)[0]
            sub_o = g.subgraph(nodes.tolist())
            sub_s = gs.subgraph(nodes.tolist())
            lab = child[nodes]

            def cut_stats(sg):
                el = np.asarray(sg.get_edgelist(), dtype=np.int64)
                if len(el) == 0:
                    return 0, 0
                same = lab[el[:, 0]] == lab[el[:, 1]]
                return int(same.sum()), int((~same).sum())

            in_o, cut_o = cut_stats(sub_o)
            in_s, cut_s = cut_stats(sub_s)
            # always reported as (seeded structure) - (baseline structure)
            dcon = sign * float(sum(c_child[j] for j in big) - c_parent[i])
            # hub-mediated shortcut probe: cross-piece edges' degree product vs intra
            el = np.asarray(sub_o.get_edgelist(), dtype=np.int64)
            gd = deg[nodes]
            if len(el):
                same = lab[el[:, 0]] == lab[el[:, 1]]
                dp = gd[el[:, 0]] * gd[el[:, 1]]
                hb = (float(dp[~same].mean()) / float(dp[same].mean())
                      if same.any() and (~same).any() else float("nan"))
            else:
                hb = float("nan")
            out.append(dict(
                direction=direction, parent_comm=int(i), size_parent=int(sizeP[i]),
                n_pieces_ge10pct=int(len(big)),
                piece_sizes=";".join(str(int(row[j])) for j in big),
                comps_in_G=len(sub_o.connected_components()),
                comps_in_Gsparse=len(sub_s.connected_components()),
                intra_piece_edges_G=in_o, cross_piece_edges_G=cut_o,
                intra_piece_edges_Gsparse=in_s, cross_piece_edges_Gsparse=cut_s,
                cross_frac_G=float(cut_o / (in_o + cut_o)) if (in_o + cut_o) else 0.0,
                cross_frac_Gsparse=float(cut_s / (in_s + cut_s)) if (in_s + cut_s) else 0.0,
                cross_edge_retention=float(cut_s / cut_o) if cut_o else float("nan"),
                intra_edge_retention=float(in_s / in_o) if in_o else float("nan"),
                boundary_hub_bridge_ratio=hb,
                d_contrib=dcon,
                max_degree=float(deg[nodes].max()),
                mean_degree=float(deg[nodes].mean())))
        return out

    # baseline community SPLIT by seeded partition
    split_rows += group_test(B, S, cB, cS, sizeB, "base_split_by_seeded", M)
    # seeded community MERGING several baseline communities
    split_rows += group_test(S, B, cS, cB, sizeS, "seeded_merges_base", M.T, sign=-1.0)
    split_rows.sort(key=lambda x: (x["direction"], -x["d_contrib"]))
    write_csv(OUT / "sparse_split_test.csv", split_rows)
    return fp_rows, split_rows, gs


# --------------------------------------------------------------------------
def step7_stability(g, store, quality_rows, edge_arr, deg, m):
    """Repeat the diff over all 5x5 (baseline, seeded) pairs + 5x5 (baseline, baseline)
    controls, so the best-vs-best anatomy is not a single-pair fluke."""
    rows = []

    def one(a_name, b_name, kind):
        A, Bx = relabel(store[a_name]), relabel(store[b_name])
        M = contingency(A, Bx).toarray()
        _, _, cA = community_contributions(g, A, edge_arr, deg)
        _, _, cBx = community_contributions(g, Bx, edge_arr, deg)
        r, c = linear_sum_assignment(-M)
        pm = {int(i): int(j) for i, j in zip(r, c) if M[i, j] > 0}
        tgt = np.full(M.shape[0], -1, dtype=np.int64)
        for i, j in pm.items():
            tgt[i] = j
        moved = tgt[A] != Bx
        signed = []
        for i, j in pm.items():
            signed.append(cBx[j] - cA[i])
        for i in range(M.shape[0]):
            if i not in pm:
                signed.append(-cA[i])
        for j in range(M.shape[1]):
            if j not in set(pm.values()):
                signed.append(cBx[j])
        signed = np.array(signed)
        dQ = float(cBx.sum() - cA.sum())
        sizeA, sizeBx = np.bincount(A), np.bincount(Bx)
        cnt = Counter()
        for i in range(M.shape[0]):
            row = M[i]
            nz = np.nonzero(row)[0]
            fr = row[nz] / sizeA[i]
            o = np.argsort(-fr)
            nz, fr = nz[o], fr[o]
            top, j = float(fr[0]), int(nz[0])
            recv = float(row[j] / sizeBx[j])
            if top >= 0.9 and recv >= 0.9:
                cnt["identical"] += 1
            elif top >= 0.9:
                cnt["merged_into"] += 1
            elif (fr >= 0.05).sum() >= 2:
                cnt["split"] += 1
            else:
                cnt["reshuffled"] += 1
        rows.append(dict(kind=kind, part_a=a_name, part_b=b_name, kA=int(M.shape[0]),
                         kB=int(M.shape[1]), dk=int(M.shape[1] - M.shape[0]),
                         dQ=dQ, frac_moved=float(moved.mean()),
                         churn_ratio=float(np.abs(signed).sum() / abs(dQ)) if dQ else 0.0,
                         sum_abs_dcontrib=float(np.abs(signed).sum()),
                         ami=float(adjusted_mutual_info_score(A, Bx)),
                         n_identical=cnt["identical"], n_merged_into=cnt["merged_into"],
                         n_split=cnt["split"], n_reshuffled=cnt["reshuffled"]))

    for bs in BASE_SEEDS:
        for ls in LEID_SEEDS:
            one(f"base_{bs}", f"seeded_{ls}", "base_vs_seeded")
    for i, a in enumerate(BASE_SEEDS):
        for b in BASE_SEEDS[i + 1:]:
            one(f"base_{a}", f"base_{b}", "base_vs_base")
    write_csv(OUT / "pairwise_stability.csv", rows)

    def agg(kind, key):
        v = np.array([r[key] for r in rows if r["kind"] == kind])
        return dict(mean=float(v.mean()), std=float(v.std()),
                    min=float(v.min()), max=float(v.max()))

    out = {}
    for kind in ("base_vs_seeded", "base_vs_base"):
        out[kind] = {k: agg(kind, k) for k in
                     ("dQ", "frac_moved", "churn_ratio", "dk", "ami",
                      "n_identical", "n_merged_into", "n_split", "n_reshuffled")}
    return rows, out


def main():
    t_start = time.perf_counter()
    print("Loading email-Enron LCC ...", flush=True)
    g = load_graph(ENRON)
    n, m = g.vcount(), g.ecount()
    deg = np.asarray(g.degree(), dtype=np.float64)
    edge_arr = np.asarray(g.get_edgelist(), dtype=np.int64)
    print(f"  n={n:,} m={m:,}", flush=True)

    npz = OUT / "partitions.npz"
    if npz.exists() and "--reuse" in sys.argv:
        store = dict(np.load(npz))
        quality_rows = list(csv.DictReader(open(OUT / "partition_quality.csv")))
        for r in quality_rows:
            r["Q"] = float(r["Q"]); r["k"] = int(r["k"]); r["seed"] = int(r["seed"])
        print("  reusing cached partitions", flush=True)
    else:
        print("\n[1] generating partitions", flush=True)
        store, quality_rows = step1_generate(g)

    Qb = {r["seed"]: r["Q"] for r in quality_rows if r["kind"] == "baseline"}
    Qs = {r["seed"]: r["Q"] for r in quality_rows if r["kind"] == "seeded"}
    best_b = max(Qb, key=Qb.get)
    best_s = max(Qs, key=Qs.get)
    global BEST_SEEDED_SEED
    BEST_SEEDED_SEED = best_s
    B = relabel(store[f"base_{best_b}"])
    S = relabel(store[f"seeded_{best_s}"])
    print(f"\n  best baseline seed={best_b} Q={Qb[best_b]:.6f}; "
          f"best seeded seed={best_s} Q={Qs[best_s]:.6f}; "
          f"dQ={Qs[best_s]-Qb[best_b]:+.6f}", flush=True)

    print("\n[2] community-level diff", flush=True)
    diff, pair_rows, moved, M, B, S, (cB, cS, LB, LS, dB, dS) = \
        step2_community_diff(g, B, S, edge_arr, deg, m)
    print(json.dumps(diff, indent=2), flush=True)

    sizeB, sizeS = np.bincount(B), np.bincount(S)
    print("\n[3] merge/split structure", flush=True)
    ms_rows, ms_cnt, pure_merges, pure_splits = step3_merge_split(B, S, M, sizeB, sizeS)
    print("  classification counts:", dict(ms_cnt), flush=True)
    print(f"  seeded comms absorbing >=2 whole base comms: {len(pure_merges)}", flush=True)
    print(f"  base comms split into >=2 pieces (each >=10%): {len(pure_splits)}", flush=True)

    print("\n[4] hub involvement", flush=True)
    hub_rows, dec_rows, hub_stats, hub = step4_hubs(g, deg, moved, B, S, sizeB, sizeS)
    print(json.dumps(hub_stats, indent=2), flush=True)
    write_csv(OUT / "moved_by_degree_decile.csv", dec_rows)

    print("\n[5] landscape probe", flush=True)
    land = step5_landscape(store, quality_rows)
    print(json.dumps(land, indent=2), flush=True)

    print("\n[6] structural fingerprint + sparse split test", flush=True)
    fp_rows, split_rows, gs = step6_fingerprint(g, edge_arr, deg, B, S, M, pair_rows,
                                                store, cB, cS, LB, LS, dB, dS)
    print(f"  {len(split_rows)} split baseline communities analysed on G_sparse", flush=True)

    print("\n[7] all-pairs stability", flush=True)
    stab_rows, stab = step7_stability(g, store, quality_rows, edge_arr, deg, m)
    print(json.dumps(stab, indent=2), flush=True)

    # cumulative decomposition of dQ over communities ranked by |d_contrib|
    signed = np.array([x["d_contrib"] for x in pair_rows])
    cum = np.cumsum(signed)
    cum_rows = [dict(rank=i + 1, d_contrib=float(signed[i]),
                     cum_net_dQ=float(cum[i]),
                     cum_frac_of_total_dQ=float(cum[i] / diff["dQ"]))
                for i in range(min(30, len(signed)))]
    write_csv(OUT / "dq_cumulative.csv", cum_rows)

    # granularity control: does k predict Q among the 25 plain restarts?
    plain = [r for r in quality_rows if r["kind"] in ("baseline", "extra_restart")]
    kk = np.array([r["k"] for r in plain], dtype=float)
    qq = np.array([r["Q"] for r in plain], dtype=float)
    ks = np.array([r["k"] for r in quality_rows if r["kind"] == "seeded"], dtype=float)
    gran = dict(corr_k_Q_plain25=float(np.corrcoef(kk, qq)[0, 1]),
                mean_k_plain25=float(kk.mean()), std_k_plain25=float(kk.std()),
                mean_k_seeded=float(ks.mean()), std_k_seeded=float(ks.std()),
                mean_T_plain_restart=float(np.mean(
                    [float(r["seconds"]) for r in plain])),
                mean_T_seeded_pipe=float(np.mean(
                    [float(r["seconds"]) for r in quality_rows if r["kind"] == "seeded"])))
    gran["compute_ratio_20restarts_vs_1pipe"] = (
        20 * gran["mean_T_plain_restart"] / gran["mean_T_seeded_pipe"])
    print("\n[8] granularity / compute control", flush=True)
    print(json.dumps(gran, indent=2), flush=True)

    res = dict(graph=dict(n=n, m=m), best_base_seed=best_b, best_seeded_seed=best_s,
               Q_best_base=Qb[best_b], Q_best_seeded=Qs[best_s],
               diff=diff, merge_split=dict(counts=dict(ms_cnt),
                                           n_pure_merge_targets=len(pure_merges),
                                           n_split_sources=len(pure_splits)),
               hubs=hub_stats, landscape=land, stability=stab, granularity=gran,
               elapsed_sec=time.perf_counter() - t_start)
    with open(OUT / "results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nDone in {res['elapsed_sec']:.1f}s -> results.json + CSVs", flush=True)


BEST_SEEDED_SEED = None

if __name__ == "__main__":
    main()
