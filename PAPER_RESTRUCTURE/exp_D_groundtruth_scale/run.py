#!/usr/bin/env python3
"""
Experiment D: Does DSpar sparsification improve GROUND-TRUTH recovery at scale?

Background
----------
The draft's ground-truth section tests only 5 tiny graphs (improvement on 1 of 5),
and its headline modularity gains are a sparse-graph self-scoring artifact.  Average-F1
against ground truth is a GRAPH-INDEPENDENT metric: the detected partition lives on the
node set, not on the edge set, so a partition found on a sparsified graph and one found
on the original graph are scored on exactly the same object.  It is therefore immune to
the sparse-graph self-scoring artifact.

Datasets (SNAP, with ground-truth communities): com-Amazon, com-DBLP, com-Youtube.
`datasets/<name>/<name>_labels.txt` is the SNAP top5000.cmty.txt file (5000 lines,
one community per line, tab-separated original node ids).

Design
------
For each dataset:
  graph = undirected, simple (no self-loops / multi-edges), largest connected component.
  ground truth = top-5000 communities restricted to the LCC node set, communities with
  < 3 surviving nodes dropped.  Communities OVERLAP (a node may be in several).

  Conditions (3 seeds each):
    baseline      : Leiden on the original graph
    dspar_paper_08: DSpar method="paper" (WITH replacement + reweight), retention=0.8
                    -- the draft's published setting; effective distinct-edge
                    retention is ~50%.  Leiden run UNWEIGHTED on the result, exactly as
                    PAPER_EXPERIMENTS/exp3_scalability.py does.
    dspar_nr_09   : DSpar method="probabilistic_no_replace", retention=0.9
                    -- mild TRUE pruning, no reweighting.

  Leiden: igraph community_leiden(objective_function='modularity', resolution=1.0,
  n_iterations=2), seeded through python's random module (igraph's RNG).

Metric (Yang & Leskovec style average F1)
-----------------------------------------
  gt2det   = mean over ground-truth communities  of max_c F1(gt, c)
  det2gt   = mean over detected clusters         of max_gt F1(c, gt)
  avgF1    = (gt2det + det2gt) / 2

  det2gt is reported twice: over ALL detected clusters (`_all`, the literal definition;
  dominated by the thousands of tiny/singleton clusters that neither condition can match
  to a top-5000 community) and over clusters with >= 3 nodes (`_ge3`).  Both variants of
  avgF1 are reported; the verdict requires them to agree.

  Implemented via a (cluster, gt-community) overlap table built from the node ->
  gt-communities index, so only pairs sharing >= 1 node are ever compared.

Also recorded
-------------
  delta_leiden = mu_intra - mu_inter of the DSpar score s(e)=1/d_u+1/d_v, w.r.t. the
                 BASELINE Leiden partition of the original graph (per baseline seed).
  delta_gt, hb_gt = same separation and the hub-bridge ratio
                 E[d_u d_v | inter] / E[d_u d_v | intra] w.r.t. GROUND TRUTH: an edge is
                 intra if its endpoints share >= 1 ground-truth community, inter if they
                 share none; only edges with BOTH endpoints in >= 1 community count.

Outputs (written next to this script): results.csv, SUMMARY.md
Dependencies: numpy, igraph.
"""

import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import igraph as ig

# import src/sparsifiers/dspar.py directly (importing the `src` package pulls in polars)
import importlib.util
_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("dspar", _REPO / "src" / "sparsifiers" / "dspar.py")
_dspar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dspar)
dspar_sparsify = _dspar.dspar_sparsify

OUT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets"

DATASETS = [
    ("com-Amazon", "com-amazon.ungraph.txt", "com-Amazon_labels.txt"),
    ("com-DBLP", "com-dblp.ungraph.txt", "com-DBLP_labels.txt"),
    ("com-Youtube", "com-youtube.ungraph.txt", "com-Youtube_labels.txt"),
]

CONDITIONS = [
    ("baseline", None, None),
    ("dspar_paper_08", "paper", 0.8),
    ("dspar_nr_09", "probabilistic_no_replace", 0.9),
]

SEEDS = [0, 1, 2]
MIN_GT_SIZE = 3


# =============================================================================
# Loading
# =============================================================================

def load_graph_lcc(path):
    """Return (igraph LCC, orig_id -> lcc_index dict)."""
    with open(path, "r") as f:
        raw = f.read()
    lines = [ln for ln in raw.splitlines() if ln and not ln.startswith("#")]
    arr = np.fromstring(" ".join(lines), dtype=np.int64, sep=" ").reshape(-1, 2)
    del raw, lines

    u = arr[:, 0]
    v = arr[:, 1]
    keep = u != v                                   # drop self-loops
    u, v = u[keep], v[keep]
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    span = int(max(hi.max(), lo.max())) + 1
    key = lo.astype(np.int64) * span + hi.astype(np.int64)
    key = np.unique(key)                            # drop multi-edges
    lo = key // span
    hi = key % span

    nodes = np.unique(np.concatenate([lo, hi]))
    idx = np.searchsorted(nodes, lo)
    jdx = np.searchsorted(nodes, hi)
    g = ig.Graph(n=len(nodes), edges=list(zip(idx.tolist(), jdx.tolist())), directed=False)

    comps = g.connected_components()
    sizes = comps.sizes()
    giant = int(np.argmax(sizes))
    memb = np.asarray(comps.membership)
    keep_idx = np.where(memb == giant)[0]
    if len(keep_idx) < g.vcount():
        g = g.induced_subgraph(keep_idx.tolist())
        nodes = nodes[keep_idx]
    id_map = {int(o): i for i, o in enumerate(nodes)}
    return g, id_map


def load_ground_truth(path, id_map):
    """Return list of np.ndarray of LCC indices (communities with >= MIN_GT_SIZE nodes)."""
    comms = []
    n_raw = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_raw += 1
            members = [id_map[int(t)] for t in line.split() if int(t) in id_map]
            if len(members) >= MIN_GT_SIZE:
                comms.append(np.unique(np.asarray(members, dtype=np.int64)))
    return comms, n_raw


def node_to_comms(comms, n):
    """node index -> list of ground-truth community ids."""
    n2c = defaultdict(list)
    for ci, mem in enumerate(comms):
        for node in mem:
            n2c[int(node)].append(ci)
    return n2c


# =============================================================================
# Average-F1
# =============================================================================

def average_f1(membership, n2c, gt_sizes, n_nodes):
    """
    membership : array-like of length n_nodes (detected cluster id per node)
    n2c        : dict node -> list of gt community ids
    gt_sizes   : array of ground-truth community sizes (restricted to graph)
    """
    memb = np.asarray(membership, dtype=np.int64)
    cl_sizes = np.bincount(memb)
    n_clusters = int((cl_sizes > 0).sum())
    ge3_mask = cl_sizes >= 3
    n_clusters_ge3 = int(ge3_mask.sum())

    overlap = defaultdict(int)
    for node, cids in n2c.items():
        cl = int(memb[node])
        for cm in cids:
            overlap[(cl, cm)] += 1

    best_per_cluster = defaultdict(float)
    best_per_gt = np.zeros(len(gt_sizes))
    for (cl, cm), ov in overlap.items():
        f1 = 2.0 * ov / (cl_sizes[cl] + gt_sizes[cm])
        if f1 > best_per_cluster[cl]:
            best_per_cluster[cl] = f1
        if f1 > best_per_gt[cm]:
            best_per_gt[cm] = f1

    gt2det = float(best_per_gt.mean()) if len(gt_sizes) else 0.0

    sum_all = sum(best_per_cluster.values())
    sum_ge3 = sum(v for cl, v in best_per_cluster.items() if ge3_mask[cl])
    det2gt_all = sum_all / n_clusters if n_clusters else 0.0
    det2gt_ge3 = sum_ge3 / n_clusters_ge3 if n_clusters_ge3 else 0.0

    return dict(
        gt2det=gt2det,
        det2gt_all=det2gt_all,
        det2gt_ge3=det2gt_ge3,
        avgF1_all=0.5 * (gt2det + det2gt_all),
        avgF1_ge3=0.5 * (gt2det + det2gt_ge3),
        n_clusters=n_clusters,
        n_clusters_ge3=n_clusters_ge3,
        largest_cluster=int(cl_sizes.max()),
    )


# =============================================================================
# Separation statistics
# =============================================================================

def delta_wrt_membership(g, membership):
    """DSpar separation mu_intra - mu_inter w.r.t. a disjoint partition."""
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    s = 1.0 / deg[e[:, 0]] + 1.0 / deg[e[:, 1]]
    memb = np.asarray(membership, dtype=np.int64)
    intra = memb[e[:, 0]] == memb[e[:, 1]]
    mu_i = float(s[intra].mean()) if intra.any() else float("nan")
    mu_o = float(s[~intra].mean()) if (~intra).any() else float("nan")
    return mu_i - mu_o, mu_i, mu_o


def gt_edge_stats(g, n2c):
    """
    delta and hub-bridge ratio w.r.t. OVERLAPPING ground truth.
    Edge counted only if both endpoints belong to >= 1 community.
    intra := endpoints share >= 1 community; inter := share none.
    """
    deg = np.asarray(g.degree(), dtype=np.float64)
    sets = {node: set(cids) for node, cids in n2c.items()}
    s_intra, s_inter, p_intra, p_inter = [], [], [], []
    for u, v in g.get_edgelist():
        su = sets.get(u)
        if su is None:
            continue
        sv = sets.get(v)
        if sv is None:
            continue
        s = 1.0 / deg[u] + 1.0 / deg[v]
        p = deg[u] * deg[v]
        if su & sv:
            s_intra.append(s)
            p_intra.append(p)
        else:
            s_inter.append(s)
            p_inter.append(p)
    mu_i = float(np.mean(s_intra)) if s_intra else float("nan")
    mu_o = float(np.mean(s_inter)) if s_inter else float("nan")
    hb = (float(np.mean(p_inter)) / float(np.mean(p_intra))) if (p_inter and p_intra) else float("nan")
    return dict(
        delta_gt=mu_i - mu_o,
        mu_intra_gt=mu_i,
        mu_inter_gt=mu_o,
        hb_gt=hb,
        n_intra_gt=len(s_intra),
        n_inter_gt=len(s_inter),
    )


# =============================================================================
# Runner
# =============================================================================

def leiden(g, seed, resolution=1.0):
    random.seed(seed)
    ig.set_random_number_generator(random)
    t0 = time.perf_counter()
    part = g.community_leiden(objective_function="modularity", resolution=resolution, n_iterations=2)
    return part.membership, time.perf_counter() - t0


def leiden_matched(g, seed, target_clusters, tol=0.05, max_iter=14):
    """
    Leiden on the ORIGINAL graph with the resolution tuned so that the number of
    non-empty clusters is ~= target_clusters.  This is the granularity control: DSpar
    fragments the graph and therefore returns far more, far smaller clusters, which by
    itself raises F1 against the (tiny, median ~8 node) SNAP top-5000 communities.
    """
    def n_of(gamma):
        memb, t = leiden(g, seed, resolution=gamma)
        return memb, len(set(memb)), t

    total_t = 0.0
    lo, hi = 1.0, 1.0
    memb, n_hi, t = n_of(hi)
    total_t += t
    best = (abs(n_hi - target_clusters), memb, hi, n_hi)
    it = 0
    while n_hi < target_clusters and hi < 8192 and it < max_iter:
        lo = hi
        hi *= 2.0
        memb, n_hi, t = n_of(hi)
        total_t += t
        it += 1
        if abs(n_hi - target_clusters) < best[0]:
            best = (abs(n_hi - target_clusters), memb, hi, n_hi)
    for _ in range(max_iter - it):
        if best[0] <= tol * target_clusters:
            break
        mid = 0.5 * (lo + hi)
        memb, n_mid, t = n_of(mid)
        total_t += t
        if abs(n_mid - target_clusters) < best[0]:
            best = (abs(n_mid - target_clusters), memb, mid, n_mid)
        if n_mid < target_clusters:
            lo = mid
        else:
            hi = mid
    return best[1], best[2], best[3], total_t


def main():
    rows = []
    ds_stats = {}

    for name, gfile, cfile in DATASETS:
        gpath = DATA_ROOT / name / gfile
        cpath = DATA_ROOT / name / cfile
        if not gpath.exists() or not cpath.exists():
            print(f"[skip] {name}: missing {gpath if not gpath.exists() else cpath}")
            continue

        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        g, id_map = load_graph_lcc(gpath)
        comms, n_raw = load_ground_truth(cpath, id_map)
        n2c = node_to_comms(comms, g.vcount())
        gt_sizes = np.array([len(c) for c in comms], dtype=np.float64)
        print(f"  n={g.vcount()} m={g.ecount()} | gt {len(comms)}/{n_raw} comms, "
              f"{len(n2c)} covered nodes, median size {np.median(gt_sizes):.0f} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

        gstats = gt_edge_stats(g, n2c)
        print(f"  GT: delta={gstats['delta_gt']:+.5f} hb={gstats['hb_gt']:.3f} "
              f"intra={gstats['n_intra_gt']} inter={gstats['n_inter_gt']}", flush=True)
        ds_stats[name] = dict(n=g.vcount(), m=g.ecount(), n_gt=len(comms),
                              gt_nodes=len(n2c), **gstats)

        deltas_leiden = []
        cluster_counts = defaultdict(list)
        for cond, method, retention in CONDITIONS:
            for seed in SEEDS:
                t_spars = 0.0
                if method is None:
                    gg = g
                else:
                    ts = time.perf_counter()
                    gg = dspar_sparsify(g, retention=retention, method=method, seed=seed)
                    t_spars = time.perf_counter() - ts
                true_ret = gg.ecount() / g.ecount()

                memb, t_leiden = leiden(gg, seed)
                res = average_f1(memb, n2c, gt_sizes, g.vcount())

                if method is None:
                    d, mi, mo = delta_wrt_membership(g, memb)
                    deltas_leiden.append(d)

                cluster_counts[cond].append(res["n_clusters"])
                rows.append(dict(
                    dataset=name, condition=cond, method=str(method),
                    retention_nominal=retention if retention else 1.0,
                    resolution=1.0,
                    seed=seed, n=g.vcount(), m=g.ecount(),
                    m_sparse=gg.ecount(), true_retention=round(true_ret, 4),
                    t_sparsify=round(t_spars, 2), t_leiden=round(t_leiden, 2),
                    **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()},
                ))
                print(f"  {cond:22s} seed={seed} ret={true_ret:.3f} "
                      f"avgF1_ge3={res['avgF1_ge3']:.4f} avgF1_all={res['avgF1_all']:.4f} "
                      f"gt2det={res['gt2det']:.4f} nC={res['n_clusters']} "
                      f"({t_leiden:.1f}s)", flush=True)

        # ---- granularity control: unsparsified Leiden at matched cluster count ----
        for cond, method, _ in CONDITIONS:
            if method is None:
                continue
            target = int(np.mean(cluster_counts[cond]))
            ctrl = f"resmatch_{cond}"
            for seed in SEEDS:
                memb, gamma, nc, t_leiden = leiden_matched(g, seed, target)
                res = average_f1(memb, n2c, gt_sizes, g.vcount())
                rows.append(dict(
                    dataset=name, condition=ctrl, method="none_resolution_matched",
                    retention_nominal=1.0, resolution=round(gamma, 4),
                    seed=seed, n=g.vcount(), m=g.ecount(),
                    m_sparse=g.ecount(), true_retention=1.0,
                    t_sparsify=0.0, t_leiden=round(t_leiden, 2),
                    **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()},
                ))
                print(f"  {ctrl:22s} seed={seed} gamma={gamma:.2f} target={target} "
                      f"avgF1_ge3={res['avgF1_ge3']:.4f} avgF1_all={res['avgF1_all']:.4f} "
                      f"gt2det={res['gt2det']:.4f} nC={res['n_clusters']} "
                      f"({t_leiden:.1f}s)", flush=True)

        ds_stats[name]["delta_leiden"] = float(np.mean(deltas_leiden))
        ds_stats[name]["delta_leiden_std"] = float(np.std(deltas_leiden))

    if not rows:
        print("No results.")
        return

    fields = list(rows[0].keys())
    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    write_summary(rows, ds_stats)
    print(f"\nWrote {OUT_DIR/'results.csv'} and {OUT_DIR/'SUMMARY.md'}")


def write_summary(rows, ds_stats):
    def agg(ds, cond, key):
        vals = [r[key] for r in rows if r["dataset"] == ds and r["condition"] == cond]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),) * 2

    datasets = [d for d, _, _ in DATASETS if any(r["dataset"] == d for r in rows)]
    L = []
    L.append("# Experiment D — ground-truth recovery at scale (SNAP top-5000)\n")
    L.append("Average F1 (Yang & Leskovec) of Leiden partitions against overlapping "
             "ground-truth communities, on the original graph vs after DSpar.\n")
    L.append("`_ge3` = detected->gt direction averaged over clusters with >= 3 nodes; "
             "`_all` = over all clusters (literal definition).\n")

    L.append("\n## Graphs and ground truth\n")
    L.append("| dataset | n (LCC) | m | gt comms (>=3) | nodes covered | delta_GT | HB ratio (GT) | delta_Leiden |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in datasets:
        s = ds_stats[d]
        L.append(f"| {d} | {s['n']:,} | {s['m']:,} | {s['n_gt']} | {s['gt_nodes']:,} | "
                 f"{s['delta_gt']:+.5f} | {s['hb_gt']:.3f} | {s['delta_leiden']:+.5f} |")

    order = ["baseline"]
    for cond, method, _ in CONDITIONS:
        if method is None:
            continue
        order += [cond, f"resmatch_{cond}"]

    L.append("\n## Recovery\n")
    L.append("`resmatch_X` = NO sparsification, Leiden on the original graph with the resolution "
             "tuned to the same number of clusters as condition X (granularity control).\n")
    L.append("| dataset | condition | true retention | avgF1_ge3 (mean+-std) | dF1_ge3 | "
             "avgF1_all | dF1_all | gt->det | n_clusters | gamma |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in datasets:
        b_ge3 = agg(d, "baseline", "avgF1_ge3")[0]
        b_all = agg(d, "baseline", "avgF1_all")[0]
        for cond in order:
            m3, s3 = agg(d, cond, "avgF1_ge3")
            ma, sa = agg(d, cond, "avgF1_all")
            g2d = agg(d, cond, "gt2det")[0]
            nc = agg(d, cond, "n_clusters")[0]
            ret = agg(d, cond, "true_retention")[0]
            gam = agg(d, cond, "resolution")[0]
            d3 = "--" if cond == "baseline" else f"{m3-b_ge3:+.4f}"
            da = "--" if cond == "baseline" else f"{ma-b_all:+.4f}"
            L.append(f"| {d} | {cond} | {ret:.3f} | {m3:.4f}+-{s3:.4f} | {d3} | "
                     f"{ma:.4f}+-{sa:.4f} | {da} | {g2d:.4f} | {nc:.0f} | {gam:.2f} |")

    L.append("\n### Sparsification vs its granularity-matched control\n")
    L.append("| dataset | condition | avgF1_ge3 (sparse) | avgF1_ge3 (resmatch) | diff | "
             "avgF1_all (sparse) | avgF1_all (resmatch) | diff |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in datasets:
        for cond, method, _ in CONDITIONS:
            if method is None:
                continue
            s3 = agg(d, cond, "avgF1_ge3")[0]
            c3 = agg(d, f"resmatch_{cond}", "avgF1_ge3")[0]
            sa = agg(d, cond, "avgF1_all")[0]
            ca = agg(d, f"resmatch_{cond}", "avgF1_all")[0]
            L.append(f"| {d} | {cond} | {s3:.4f} | {c3:.4f} | {s3-c3:+.4f} | "
                     f"{sa:.4f} | {ca:.4f} | {sa-ca:+.4f} |")

    L.append("\n## Verdict\n")
    L.append("_(filled in after the run — see final report)_\n")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
