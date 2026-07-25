#!/usr/bin/env python3
"""
Experiment B: configuration-model (degree-preserving rewire) null control.

Question
--------
The paper draft treats two quantities as evidence that DSpar "clarifies community
structure":

    delta   = mean DSpar score on intra-community edges
              - mean DSpar score on inter-community edges,        with s(e)=1/d_u+1/d_v
    dQ_fix  = Q(G_sparse, P) - Q(G, P)                            for a FIXED partition P

If a degree-preserving rewiring of the same graph -- which destroys all community
structure but keeps the degree sequence exactly -- reproduces delta > 0 and
dQ_fix > 0 at comparable or larger magnitude, then neither quantity is evidence
about community structure; both are consequences of degree heterogeneity alone.

Protocol (per network)
----------------------
1. Load edge list, make undirected + simple, keep the largest connected component.
2. REAL arm:
     P  = Leiden partition of G (fixed for the rest of the arm)
     delta, hb = E[d_u d_v | inter] / E[d_u d_v | intra], Q_fixed(G, P)
     for each of 3 sparsification seeds:
         G_s = DSpar(G, method="paper", retention=0.8), weights DROPPED
         dQ  = Q_fixed(G_s, P) - Q_fixed(G, P)      (igraph modularity)
         actual retention = m_sparse / m
3. NULL arm: 2 rewire seeds. For each, degree-preserving rewire (igraph, simple
   double-edge swaps, n_swaps = 10*m), fresh Leiden partition of the rewired
   graph as the fixed partition, then exactly step 2 with 2 sparsification seeds.

DSpar implementation
--------------------
`dspar_paper_sample` below is a vectorised re-implementation of
experiments/dspar.py::dspar_sparsify(method="paper"): identical scores, identical
sampling probabilities, identical q = ceil(retention*m), identical
np.random.choice(..., replace=True, p=probs) RNG stream. Only the edge ordering
fed to np.random.choice differs (array order vs networkx insertion order), which
is irrelevant to any reported statistic. Weights are dropped per protocol, so
only the set of sampled edges matters. `--validate` cross-checks the two
implementations on a network.

Usage
-----
    python run.py --network ca-GrQc
    python run.py --network ca-GrQc --validate
"""

import argparse
import csv
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_CSV = HERE / "results.csv"

RETENTION = 0.80
DSPAR_METHOD = "paper"
LEIDEN_SEED = 42
REAL_SPARSE_SEEDS = [101, 102, 103]
REWIRE_SEEDS = [7, 8]
NULL_SPARSE_SEEDS = [201, 202]
SWAPS_PER_EDGE = 10

NETWORKS = [
    "email-Eu-core", "wiki-Vote", "ca-GrQc", "ca-HepTh", "facebook-combined",
    "ca-CondMat", "ca-HepPh", "ca-AstroPh", "email-Enron", "cit-HepTh",
    "cit-HepPh", "com-Amazon", "com-DBLP", "com-Youtube", "wiki-Talk",
    "wiki-topcats", "cit-Patents",
]

CSV_FIELDS = [
    "network", "arm", "n", "m", "n_comms",
    "Q_fixed_base", "Q_fixed_base_std",
    "delta", "delta_std", "hb", "hb_std",
    "dQ_fixed", "dQ_fixed_std",
    "actual_retention", "actual_retention_std",
    "n_reps", "seconds",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_edge_file(path):
    """Chunked whitespace edge-list parser (comments '#'). Returns (M,2) int64."""
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
    ncol = 2
    if flat.size % ncol:
        flat = flat[: (flat.size // ncol) * ncol]
    return flat.reshape(-1, ncol)


def load_lcc_graph(name):
    """Undirected simple graph, largest connected component, as igraph.Graph."""
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

    comps = g.connected_components(mode="weak")
    g = comps.giant()
    g.simplify(multiple=True, loops=True)
    return g


# ---------------------------------------------------------------------------
# DSpar (vectorised re-implementation of experiments/dspar.py, method="paper")
# ---------------------------------------------------------------------------

def dspar_scores(deg, eu, ev):
    return 1.0 / deg[eu] + 1.0 / deg[ev]


def dspar_paper_sample(scores, retention, seed):
    """Return indices of unique edges retained by DSpar 'paper' sampling."""
    m = scores.size
    probs = scores / scores.sum()
    q = int(np.ceil(retention * m))
    rng = np.random.RandomState(seed)
    idx = rng.choice(m, size=q, replace=True, p=probs)
    return np.unique(idx)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def leiden_membership(g, seed=LEIDEN_SEED):
    part = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, seed=seed, n_iterations=2
    )
    return np.asarray(part.membership, dtype=np.int64)


def arm_metrics(g, memb, sparse_seeds, label=""):
    """Fixed-partition DSpar metrics for one graph + one fixed partition."""
    n = g.vcount()
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    eu, ev = E[:, 0], E[:, 1]
    m = eu.size
    deg = np.asarray(g.degree(), dtype=np.float64)

    s = dspar_scores(deg, eu, ev)
    intra = memb[eu] == memb[ev]
    n_intra, n_inter = int(intra.sum()), int((~intra).sum())

    if n_intra and n_inter:
        delta = float(s[intra].mean() - s[~intra].mean())
        prod = deg[eu] * deg[ev]
        hb = float(prod[~intra].mean() / prod[intra].mean())
    else:
        delta, hb = float("nan"), float("nan")

    q_base = float(g.modularity(memb.tolist()))

    dqs, rets = [], []
    for sd in sparse_seeds:
        keep = dspar_paper_sample(s, RETENTION, sd)
        gs = ig.Graph(n=n)
        gs.add_edges(np.column_stack([eu[keep], ev[keep]]))
        q_sp = float(gs.modularity(memb.tolist()))
        dqs.append(q_sp - q_base)
        rets.append(keep.size / m)
        del gs

    return {
        "n": n, "m": m, "n_comms": int(memb.max()) + 1,
        "n_intra": n_intra, "n_inter": n_inter,
        "Q_fixed_base": q_base, "delta": delta, "hb": hb,
        "dQ": dqs, "ret": rets,
    }


def agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def run_real(g):
    memb = leiden_membership(g)
    r = arm_metrics(g, memb, REAL_SPARSE_SEEDS, "real")
    dq_m, dq_s = agg(r["dQ"])
    rt_m, rt_s = agg(r["ret"])
    return {
        "n": r["n"], "m": r["m"], "n_comms": r["n_comms"],
        "Q_fixed_base": r["Q_fixed_base"], "Q_fixed_base_std": 0.0,
        "delta": r["delta"], "delta_std": 0.0,
        "hb": r["hb"], "hb_std": 0.0,
        "dQ_fixed": dq_m, "dQ_fixed_std": dq_s,
        "actual_retention": rt_m, "actual_retention_std": rt_s,
        "n_reps": len(r["dQ"]),
    }


def run_null(g, rewire_seeds=None):
    m = g.ecount()
    rewire_seeds = rewire_seeds or REWIRE_SEEDS
    qs, ds, hs, dqs, rets, ncs = [], [], [], [], [], []
    for rs in rewire_seeds:
        gr = g.copy()
        ig.set_random_number_generator(random.Random(rs))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gr.rewire(n=SWAPS_PER_EDGE * m, mode="simple")
        ig.set_random_number_generator(random)
        memb = leiden_membership(gr)
        r = arm_metrics(gr, memb, NULL_SPARSE_SEEDS, "null")
        qs.append(r["Q_fixed_base"])
        ds.append(r["delta"])
        hs.append(r["hb"])
        ncs.append(r["n_comms"])
        dqs += r["dQ"]
        rets += r["ret"]
        del gr
    q_m, q_s = agg(qs)
    d_m, d_s = agg(ds)
    h_m, h_s = agg(hs)
    dq_m, dq_s = agg(dqs)
    rt_m, rt_s = agg(rets)
    return {
        "n": g.vcount(), "m": m, "n_comms": int(np.mean(ncs)),
        "Q_fixed_base": q_m, "Q_fixed_base_std": q_s,
        "delta": d_m, "delta_std": d_s,
        "hb": h_m, "hb_std": h_s,
        "dQ_fixed": dq_m, "dQ_fixed_std": dq_s,
        "actual_retention": rt_m, "actual_retention_std": rt_s,
        "n_reps": len(dqs),
    }


# ---------------------------------------------------------------------------
# Validation against the repo's networkx DSpar
# ---------------------------------------------------------------------------

def validate(g):
    """Compare vectorised DSpar against experiments/dspar.py on this graph."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    import networkx as nx
    from dspar import dspar_sparsify

    memb = leiden_membership(g)
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    deg = np.asarray(g.degree(), dtype=np.float64)
    s = dspar_scores(deg, E[:, 0], E[:, 1])
    q_base = float(g.modularity(memb.tolist()))

    Gnx = nx.Graph()
    Gnx.add_nodes_from(range(g.vcount()))
    Gnx.add_edges_from(map(tuple, E))

    print(f"{'seed':>6} {'ret_repo':>10} {'ret_vec':>10} {'dQ_repo':>12} {'dQ_vec':>12}")
    for sd in REAL_SPARSE_SEEDS:
        Gs = dspar_sparsify(Gnx, retention=RETENTION, method=DSPAR_METHOD, seed=sd)
        er = np.asarray(list(Gs.edges()), dtype=np.int64)
        gr = ig.Graph(n=g.vcount())
        gr.add_edges(er)
        dq_repo = float(gr.modularity(memb.tolist())) - q_base
        ret_repo = er.shape[0] / E.shape[0]

        keep = dspar_paper_sample(s, RETENTION, sd)
        gv = ig.Graph(n=g.vcount())
        gv.add_edges(E[keep])
        dq_vec = float(gv.modularity(memb.tolist())) - q_base
        ret_vec = keep.size / E.shape[0]
        print(f"{sd:>6} {ret_repo:>10.5f} {ret_vec:>10.5f} {dq_repo:>12.6f} {dq_vec:>12.6f}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def append_row(row):
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--arms", default="real,null")
    ap.add_argument("--rewire-seeds", default=None,
                    help="comma-separated override, e.g. '7' for a reduced null "
                         "on networks too large for 2 rewire replicates")
    args = ap.parse_args()
    rseeds = ([int(x) for x in args.rewire_seeds.split(",")]
              if args.rewire_seeds else None)

    t0 = time.time()
    g = load_lcc_graph(args.network)
    print(f"[{args.network}] LCC: n={g.vcount()} m={g.ecount()} "
          f"load={time.time()-t0:.1f}s", flush=True)

    if args.validate:
        validate(g)
        return

    for arm in args.arms.split(","):
        t = time.time()
        res = run_real(g) if arm == "real" else run_null(g, rseeds)
        res["network"] = args.network
        res["arm"] = arm
        res["seconds"] = round(time.time() - t, 1)
        append_row({k: res.get(k, "") for k in CSV_FIELDS})
        print(f"[{args.network}/{arm}] Q0={res['Q_fixed_base']:.4f} "
              f"delta={res['delta']:.6f} hb={res['hb']:.3f} "
              f"dQ={res['dQ_fixed']:+.6f}+-{res['dQ_fixed_std']:.6f} "
              f"ret={res['actual_retention']:.4f} "
              f"({res['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
