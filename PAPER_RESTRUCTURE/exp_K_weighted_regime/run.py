#!/usr/bin/env python
"""
Experiment K: the WEIGHTED DSpar regime.

Question: does keeping DSpar's importance weights preserve modularity (and does
the resulting partition transfer honestly back to the original graph), even on
networks where DSpar's (1 +- eps/alpha) spectral bound is numerically vacuous
(1/alpha = 535 on ca-GrQc, 139 on ca-CondMat)?

Two weighted samplers
---------------------
(i)  "paper"      Liu et al. Algorithm 1, verbatim: draw Q = ceil(alpha*m)
                  samples WITH replacement with p_e ~ (1/d_u + 1/d_v),
                  reweight w_e = k_e / (Q p_e).  Unbiased: E[A'] = A.
                  Verified bit-identical to experiments/dspar.py method="paper"
                  (see audit_feb.txt s.1); reimplemented on numpy/igraph here
                  only for speed on the million-edge graphs.
(ii) "calibrated" exp_C's calibrated Bernoulli: p_e = min(1, lambda s_e) with
                  lambda solved by bisection so E[retention] = alpha exactly,
                  independent Bernoulli, Horvitz-Thompson weights w_e = 1/p_e.
                  Also unbiased: E[A'] = A.

Metrics per (network, sampler, alpha, sparsify seed)
----------------------------------------------------
(a) PRESERVATION   dQ_fixed_w = Q_w(P0 ; G') - Q(P0 ; G),  P0 = plain Leiden on
                   the original graph, seed 42, held FIXED.  Prediction: ~0.
(b) HONEST TRANSFER  Leiden WITH weights on G' -> P_w; score P_w UNWEIGHTED on
                   the ORIGINAL G; report Q(P_w;G) - Q_base_mean and - Q_base_best
                   where the baseline is 5-seed plain Leiden on G.
(c) RECOVERY       email-Eu-core department labels: AMI/ARI of P_w vs labels,
                   against the 5-seed plain-Leiden baseline.

Loaders: exp_E_delta_star/run.py (LCC, simple, undirected) for (a)/(b);
exp_F_gt_recheck/run.py canonicalisation (labelled nodes -> LCC) for (c).
Leiden: leidenalg ModularityVertexPartition, n_iterations=2 (repo convention),
RNG seeded.

Rows are appended to results.csv as soon as they are produced.
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg as la
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

REPO = Path("/home/md724/community_detection_spectral")
DATASETS_DIR = REPO / "datasets"
DATA_DIR = REPO / "data"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results.csv"

NETWORKS = [
    "ca-GrQc", "email-Eu-core", "wiki-Vote", "ca-HepTh",
    "ca-CondMat", "email-Enron", "com-DBLP", "com-Amazon",
]

CONFIGS = [("paper", 0.8), ("paper", 1.0), ("calibrated", 0.9)]

N_BASE_SEEDS = 5
N_SPARSE_SEEDS = 5
BASE_SEEDS = [100, 101, 102, 103, 104]
SPARSE_SEEDS = [200, 201, 202, 203, 204]
LEIDEN_SPARSE_SEEDS = [300, 301, 302, 303, 304]
P0_SEED = 42
N_ITER = 2

FIELDS = [
    "network", "n", "m", "sampler", "alpha", "rep", "sparsify_seed",
    "leiden_seed", "m_sparse", "retention", "weight_sum_ratio",
    "Q_P0_orig", "Q_P0_sparse_w", "dQ_fixed_w",
    "Q_sparse_leiden_w", "Q_orig_of_Pw", "Q_base_mean", "Q_base_std",
    "Q_base_best", "transfer_vs_mean", "transfer_vs_best",
    "nc_P0", "nc_Pw", "nc_base_mean",
    "t_sparsify", "t_leiden_sparse", "t_leiden_orig_mean",
    "ami_Pw", "ari_Pw", "ami_base_mean", "ari_base_mean",
    "ami_base_std", "ari_base_std", "nc_Pw_orig_labels",
]


# ---------------------------------------------------------------------------
# Loading (verbatim from PAPER_RESTRUCTURE/exp_E_delta_star/run.py)
# ---------------------------------------------------------------------------

def _parse_edge_file(path):
    parts = []
    leftover = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 24)
            if not chunk:
                break
            buf = leftover + chunk
            cut = buf.rfind(b"\n")
            if cut == -1:
                leftover = buf
                continue
            block, leftover = buf[:cut], buf[cut + 1:]
            if b"#" in block:
                lines = [ln for ln in block.split(b"\n")
                         if ln and not ln.lstrip().startswith(b"#")]
                block = b"\n".join(lines)
            toks = block.split()
            if toks:
                parts.append(np.array(toks, dtype=np.int64))
    if leftover.strip() and not leftover.lstrip().startswith(b"#"):
        toks = leftover.split()
        if toks:
            parts.append(np.array(toks, dtype=np.int64))
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    if flat.size % 2:
        flat = flat[: (flat.size // 2) * 2]
    return flat.reshape(-1, 2)


def load_lcc_graph(name):
    path = DATASETS_DIR / name / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = _parse_edge_file(path)
    u, v = raw[:, 0], raw[:, 1]
    del raw
    keep = u != v
    u, v = u[keep], v[keep]
    nodes = np.unique(np.concatenate([u, v]))
    n = nodes.size
    u = np.searchsorted(nodes, u)
    v = np.searchsorted(nodes, v)
    del nodes
    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    del u, v
    key = lo.astype(np.int64) * n + hi
    key = np.unique(key)
    lo = (key // n).astype(np.int32)
    hi = (key % n).astype(np.int32)
    del key
    g = ig.Graph(n=int(n))
    g.add_edges(np.column_stack([lo, hi]))
    del lo, hi
    g = g.connected_components(mode="weak").giant()
    g.simplify(multiple=True, loops=True)
    return g


def load_email_labeled():
    """exp_F_gt_recheck canonicalisation: labelled nodes only -> LCC -> 0..n-1."""
    edges = []
    with open(DATA_DIR / "email-Eu-core.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2:
                a, b = int(p[0]), int(p[1])
                if a != b:
                    edges.append((a, b))
    gt = {}
    with open(DATA_DIR / "email-Eu-core-department-labels.txt") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                gt[int(p[0])] = int(p[1])
    edges = [(a, b) for a, b in edges if a in gt and b in gt]
    nodes = sorted(set([a for a, _ in edges]) | set([b for _, b in edges]) | set(gt))
    idx = {o: i for i, o in enumerate(nodes)}
    g = ig.Graph(n=len(nodes))
    g.add_edges([(idx[a], idx[b]) for a, b in edges])
    g.simplify(multiple=True, loops=True)
    comp = g.connected_components(mode="weak")
    giant_id = int(np.argmax(comp.sizes()))
    keep = [i for i, c in enumerate(comp.membership) if c == giant_id]
    sub = g.subgraph(keep)
    y_raw = [gt[nodes[i]] for i in keep]
    _, y = np.unique(np.asarray(y_raw), return_inverse=True)
    return sub, y


# ---------------------------------------------------------------------------
# Samplers (both WEIGHTED, both unbiased)
# ---------------------------------------------------------------------------

def dspar_scores(g):
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    return e, 1.0 / deg[e[:, 0]] + 1.0 / deg[e[:, 1]]


def sparsify_paper(edge_arr, scores, alpha, seed):
    """Liu et al. Alg.1: with replacement, Q=ceil(alpha*m), w_e = k_e/(Q p_e)."""
    m = scores.size
    p = scores / scores.sum()
    q = int(np.ceil(alpha * m))
    rs = np.random.RandomState(seed)
    idx = rs.choice(m, size=q, replace=True, p=p)
    cnt = np.bincount(idx, minlength=m)
    kept = np.nonzero(cnt)[0]
    w = cnt[kept] / (q * p[kept])
    return edge_arr[kept], w


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


def sparsify_calibrated(edge_arr, probs, seed):
    """Independent Bernoulli with calibrated p_e; HT weights w_e = 1/p_e."""
    rs = np.random.RandomState(seed)
    keep = rs.random_sample(probs.size) < probs
    return edge_arr[keep], 1.0 / probs[keep]


# ---------------------------------------------------------------------------
# Leiden
# ---------------------------------------------------------------------------

def leiden(g, seed, weights=None):
    t0 = time.perf_counter()
    if weights is None:
        part = la.ModularityVertexPartition(g)
    else:
        part = la.ModularityVertexPartition(g, weights=list(weights))
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    return list(part.membership), float(part.modularity), len(part), time.perf_counter() - t0


def build_sparse(n, kept_edges):
    gs = ig.Graph(n=n)
    gs.add_edges(kept_edges)
    return gs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        return {(r["network"], r["sampler"], r["alpha"], r["rep"])
                for r in csv.DictReader(f)}


def run_network(name, g, y=None):
    n, m = g.vcount(), g.ecount()
    print(f"\n=== {name}: n={n:,} m={m:,}", flush=True)
    done = done_keys()

    # baseline: 5-seed plain Leiden on the ORIGINAL graph
    qs, ncs, ts, amis, aris = [], [], [], [], []
    for s in BASE_SEEDS:
        memb, q, nc, t = leiden(g, s)
        qs.append(q); ncs.append(nc); ts.append(t)
        if y is not None:
            amis.append(adjusted_mutual_info_score(y, memb))
            aris.append(adjusted_rand_score(y, memb))
    Q_base_mean, Q_base_std, Q_base_best = float(np.mean(qs)), float(np.std(qs, ddof=1)), float(max(qs))
    print(f"  baseline Q = {Q_base_mean:.6f} +- {Q_base_std:.6f} (best {Q_base_best:.6f}), "
          f"nc={np.mean(ncs):.1f}, t={np.mean(ts):.2f}s", flush=True)
    if y is not None:
        print(f"  baseline AMI={np.mean(amis):.4f}+-{np.std(amis,ddof=1):.4f} "
              f"ARI={np.mean(aris):.4f}+-{np.std(aris,ddof=1):.4f}", flush=True)

    # fixed partition P0 (seed 42) on the original
    P0, Q_P0_orig, nc_P0, _ = leiden(g, P0_SEED)
    print(f"  P0 (seed 42): Q={Q_P0_orig:.6f}, nc={nc_P0}", flush=True)

    edge_arr, scores = dspar_scores(g)
    cal_cache = {}

    for sampler, alpha in CONFIGS:
        if sampler == "calibrated" and alpha not in cal_cache:
            cal_cache[alpha] = _probs_calibrated(scores, alpha)
        for rep in range(N_SPARSE_SEEDS):
            key = (name, sampler, f"{alpha}", f"{rep}")
            if key in done:
                print(f"  skip (done) {sampler} a={alpha} rep={rep}", flush=True)
                continue
            ss, ls = SPARSE_SEEDS[rep], LEIDEN_SPARSE_SEEDS[rep]
            t0 = time.perf_counter()
            if sampler == "paper":
                kept, w = sparsify_paper(edge_arr, scores, alpha, ss)
            else:
                kept, w = sparsify_calibrated(edge_arr, cal_cache[alpha], ss)
            gs = build_sparse(n, kept)
            t_sp = time.perf_counter() - t0

            # (a) preservation: fixed P0, weighted modularity on G'
            Q_P0_sparse_w = float(gs.modularity(P0, weights=list(w)))

            # (b) honest transfer: weighted Leiden on G' -> P_w, scored on G
            Pw, Q_sparse_leiden_w, nc_Pw, t_ld = leiden(gs, ls, weights=w)
            Q_orig_of_Pw = float(g.modularity(Pw))

            row = dict(
                network=name, n=n, m=m, sampler=sampler, alpha=alpha, rep=rep,
                sparsify_seed=ss, leiden_seed=ls,
                m_sparse=int(gs.ecount()), retention=gs.ecount() / m,
                weight_sum_ratio=float(w.sum()) / m,
                Q_P0_orig=Q_P0_orig, Q_P0_sparse_w=Q_P0_sparse_w,
                dQ_fixed_w=Q_P0_sparse_w - Q_P0_orig,
                Q_sparse_leiden_w=Q_sparse_leiden_w, Q_orig_of_Pw=Q_orig_of_Pw,
                Q_base_mean=Q_base_mean, Q_base_std=Q_base_std, Q_base_best=Q_base_best,
                transfer_vs_mean=Q_orig_of_Pw - Q_base_mean,
                transfer_vs_best=Q_orig_of_Pw - Q_base_best,
                nc_P0=nc_P0, nc_Pw=nc_Pw, nc_base_mean=float(np.mean(ncs)),
                t_sparsify=t_sp, t_leiden_sparse=t_ld,
                t_leiden_orig_mean=float(np.mean(ts)),
            )
            if y is not None:
                row.update(
                    ami_Pw=float(adjusted_mutual_info_score(y, Pw)),
                    ari_Pw=float(adjusted_rand_score(y, Pw)),
                    ami_base_mean=float(np.mean(amis)), ari_base_mean=float(np.mean(aris)),
                    ami_base_std=float(np.std(amis, ddof=1)),
                    ari_base_std=float(np.std(aris, ddof=1)),
                    nc_Pw_orig_labels=nc_Pw,
                )
            append_row(row)
            print(f"  {sampler} a={alpha} rep={rep}: ret={row['retention']:.4f} "
                  f"sumw/m={row['weight_sum_ratio']:.4f} dQfix_w={row['dQ_fixed_w']:+.6f} "
                  f"transfer={row['transfer_vs_mean']:+.6f} nc={nc_Pw} "
                  f"t_sp={t_sp:.1f}s t_ld={t_ld:.1f}s", flush=True)


def main():
    todo = sys.argv[1:] if len(sys.argv) > 1 else NETWORKS
    for name in todo:
        if name == "email-Eu-core-labeled":
            g, y = load_email_labeled()
            run_network("email-Eu-core-labeled", g, y)
            continue
        g = load_lcc_graph(name)
        run_network(name, g)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
