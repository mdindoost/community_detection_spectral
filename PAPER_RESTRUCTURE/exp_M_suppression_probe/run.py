#!/usr/bin/env python3
"""
Experiment M: mechanism probe for DSpar-delta SUPPRESSION by real community
structure.

Phenomenon (exp_B_config_null/results.csv)
------------------------------------------
delta = mu_intra(s) - mu_inter(s),  s(e) = 1/d_u + 1/d_v, is LARGER on the
degree-preserving rewired null (with its own Leiden partition) than on the real
graph, for 16/17 networks. Since DSpar scores depend only on degrees and
rewiring preserves degrees exactly, the score multiset is IDENTICAL between the
real and rewired graph. delta can therefore differ only through WHICH edges are
intra vs inter under each graph's Leiden partition.

Exact algebraic decomposition used throughout
---------------------------------------------
Let Y = 1[edge is intra], p = P(Y=1), sd_s = std of the score multiset.
    cov(s, Y) = p(1-p) * (mu_intra - mu_inter)
=>  delta = cov(s,Y) / (p(1-p)) = r * sd_s / sqrt(p(1-p))
where r = point-biserial corr(s, Y). Because sd_s is IDENTICAL for real and
null,
    log delta_null - log delta_real
        = [log r_null - log r_real]                        <- "sorting" term
        + 0.5*[log(p_real(1-p_real)) - log(p_null(1-p_null))]  <- "granularity" term
This splits suppression exactly into (H1/H2) how strongly the partition
correlates with degree-driven scores, and (H3) how coarse/fine the partition is.

Hypotheses
----------
H1 degree sorting: null Leiden communities are more degree-stratified (hub edges
   disproportionately inter). Measures: deg_R2 (fraction of variance of log-degree
   explained by community), hub_inter_frac (top-5% degree nodes' edges that are
   inter), hb = E[d_u d_v|inter]/E[d_u d_v|intra].
H2 triangles: real communities are triangle-rich; triangles pin low-degree nodes
   to their hubs INSIDE communities, keeping both low-score AND high-score edges
   intra, compressing contrast. Measures: global transitivity, mean local
   clustering, per-edge triangle counts intra vs inter, and cross-network
   correlation of suppression with transitivity.
H3 granularity: is suppression just a p / n_comms artefact? Answered exactly by
   the decomposition above plus partial correlations.

Subcommands
-----------
    probe   --network X     : real arm + rewired null arm, full structural stats
    staged  --network X     : rewire in stages (0/25/50/100% of 10m swaps),
                              Leiden + delta + transitivity at each stage
"""

import argparse
import csv
import random
import time
import warnings
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg
import scipy.sparse as sp
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
PROBE_CSV = HERE / "probe_results.csv"
STAGED_CSV = HERE / "staged_results.csv"

LEIDEN_SEED = 42
REWIRE_SEEDS = [7, 8]
SWAPS_PER_EDGE = 10
HUB_FRAC = 0.05

PROBE_FIELDS = [
    "network", "arm", "rewire_seed", "n", "m",
    "n_comms", "Q", "p_intra", "sd_s", "mu_s",
    "delta", "mu_intra", "mu_inter", "r_pb", "auc_s", "auc_prod",
    "term_sorting", "term_scale", "term_granularity", "term_sum", "log_ratio",
    "hb", "deg_R2", "hub_inter_frac", "nonhub_inter_frac", "hub_inter_lift",
    "assort_deg", "transitivity", "avg_local_cc",
    "tri_intra", "tri_inter", "tri_frac_intra_ge1", "tri_frac_inter_ge1",
    "comm_size_mean", "comm_size_max_frac", "comm_size_gini",
    "seconds",
]

STAGED_FIELDS = [
    "network", "seed", "stage_frac", "swaps", "n", "m",
    "n_comms", "Q", "p_intra", "sd_s", "delta", "r_pb", "auc_s", "hb",
    "transitivity", "avg_local_cc", "deg_R2", "hub_inter_frac",
    "hub_inter_lift", "tri_intra", "tri_inter", "seconds",
]


# ---------------------------------------------------------------------------
# Loading (identical to exp_B_config_null/run.py)
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


def leiden_membership(g, seed=LEIDEN_SEED):
    part = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, seed=seed, n_iterations=2
    )
    return np.asarray(part.membership, dtype=np.int64)


# ---------------------------------------------------------------------------
# Structural measures
# ---------------------------------------------------------------------------

def edge_triangles(g, eu, ev, chunk=20000):
    """Per-edge triangle count via chunked sparse row intersection."""
    n = g.vcount()
    m = eu.size
    data = np.ones(2 * m, dtype=np.float32)
    A = sp.csr_matrix(
        (data, (np.concatenate([eu, ev]), np.concatenate([ev, eu]))),
        shape=(n, n), dtype=np.float32,
    )
    A.data[:] = 1.0
    out = np.empty(m, dtype=np.float64)
    for i in range(0, m, chunk):
        j = min(i + chunk, m)
        Au = A[eu[i:j]]
        Av = A[ev[i:j]]
        out[i:j] = np.asarray(Au.multiply(Av).sum(axis=1)).ravel()
    return out


def auc(x, pos):
    """P(x_pos > x_neg) + 0.5 P(=), tie-corrected Mann-Whitney AUC.

    THE class-balance-free measure of score/partition association: unlike
    delta or r_pb it does not change if the fraction of intra edges changes,
    so it isolates 'sorting' (H1/H2) from 'granularity' (H3).
    """
    n1 = int(pos.sum())
    n0 = int(x.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(x)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def score_profile(s, intra, nbins=10):
    """Inter-community rate within each decile of the DSpar score s."""
    qs = np.quantile(s, np.linspace(0, 1, nbins + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    b = np.clip(np.searchsorted(qs, s, side="left") - 1, 0, nbins - 1)
    cnt = np.bincount(b, minlength=nbins).astype(np.float64)
    inter = np.bincount(b, weights=(~intra).astype(np.float64), minlength=nbins)
    smean = np.bincount(b, weights=s, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.divide(inter, cnt, out=np.full(nbins, np.nan), where=cnt > 0),
                np.divide(smean, cnt, out=np.full(nbins, np.nan), where=cnt > 0),
                cnt)


def degree_R2(memb, deg):
    """Fraction of variance of log(degree) explained by community membership."""
    x = np.log(deg)
    k = int(memb.max()) + 1
    cnt = np.bincount(memb, minlength=k).astype(np.float64)
    ssum = np.bincount(memb, weights=x, minlength=k)
    mean_c = np.divide(ssum, cnt, out=np.zeros(k), where=cnt > 0)
    resid = x - mean_c[memb]
    sst = float(((x - x.mean()) ** 2).sum())
    sse = float((resid ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def gini(a):
    a = np.sort(np.asarray(a, dtype=np.float64))
    n = a.size
    if n == 0 or a.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def structural_metrics(g, memb, do_triangles=True, do_cc=True):
    t0 = time.time()
    n = g.vcount()
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    eu, ev = E[:, 0], E[:, 1]
    m = eu.size
    deg = np.asarray(g.degree(), dtype=np.float64)

    s = 1.0 / deg[eu] + 1.0 / deg[ev]
    intra = memb[eu] == memb[ev]
    p = float(intra.mean())
    sd_s = float(s.std(ddof=0))
    mu_s = float(s.mean())

    prof = None
    if 0 < p < 1:
        mu_in = float(s[intra].mean())
        mu_out = float(s[~intra].mean())
        delta = mu_in - mu_out
        r_pb = float(np.corrcoef(s, intra.astype(np.float64))[0, 1])
        prod = deg[eu] * deg[ev]
        hb = float(prod[~intra].mean() / prod[intra].mean())
        a_s = auc(s, intra)
        a_p = auc(prod, intra)
        prof = score_profile(s, intra)
        del prod
    else:
        mu_in = mu_out = delta = r_pb = hb = a_s = a_p = float("nan")

    # hub-incidence
    thr = np.quantile(deg, 1.0 - HUB_FRAC)
    hub = deg >= thr
    hub_edge = hub[eu] | hub[ev]
    hub_inter = float((~intra)[hub_edge].mean()) if hub_edge.any() else float("nan")
    nonhub_inter = (float((~intra)[~hub_edge].mean())
                    if (~hub_edge).any() else float("nan"))
    # lift: hub-incident edges' inter-rate relative to the graph-wide inter-rate.
    # >1 means the partition pushes hub edges between communities more than
    # average; scale-free of the overall intra/inter balance (H1's real test).
    hub_lift = hub_inter / (1.0 - p) if 0 < p < 1 else float("nan")

    # community sizes
    k = int(memb.max()) + 1
    sizes = np.bincount(memb, minlength=k).astype(np.float64)
    sizes = sizes[sizes > 0]

    res = {
        "n": n, "m": m, "n_comms": int(sizes.size),
        "Q": float(g.modularity(memb.tolist())),
        "p_intra": p, "sd_s": sd_s, "mu_s": mu_s,
        "delta": delta, "mu_intra": mu_in, "mu_inter": mu_out, "r_pb": r_pb,
        "auc_s": a_s, "auc_prod": a_p, "_profile": prof,
        "hb": hb,
        "deg_R2": degree_R2(memb, deg),
        "hub_inter_frac": hub_inter, "nonhub_inter_frac": nonhub_inter,
        "hub_inter_lift": hub_lift,
        "assort_deg": float(g.assortativity_degree(directed=False)),
        "comm_size_mean": float(sizes.mean()),
        "comm_size_max_frac": float(sizes.max() / n),
        "comm_size_gini": gini(sizes),
    }

    if do_cc:
        res["transitivity"] = float(g.transitivity_undirected(mode="zero"))
        res["avg_local_cc"] = float(g.transitivity_avglocal_undirected(mode="zero"))
    else:
        res["transitivity"] = float("nan")
        res["avg_local_cc"] = float("nan")

    if do_triangles and 0 < p < 1:
        t = edge_triangles(g, eu, ev)
        res["tri_intra"] = float(t[intra].mean())
        res["tri_inter"] = float(t[~intra].mean())
        res["tri_frac_intra_ge1"] = float((t[intra] >= 1).mean())
        res["tri_frac_inter_ge1"] = float((t[~intra] >= 1).mean())
        del t
    else:
        for kk in ("tri_intra", "tri_inter",
                   "tri_frac_intra_ge1", "tri_frac_inter_ge1"):
            res[kk] = float("nan")

    res["seconds"] = round(time.time() - t0, 1)
    return res


def rewire_copy(g, seed, n_swaps):
    gr = g.copy()
    ig.set_random_number_generator(random.Random(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gr.rewire(n=int(n_swaps), mode="simple")
    ig.set_random_number_generator(random)
    return gr


def rewire_inplace(gr, seed, n_swaps):
    ig.set_random_number_generator(random.Random(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gr.rewire(n=int(n_swaps), mode="simple")
    ig.set_random_number_generator(random)
    return gr


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def append_row(path, fields, row):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_probe(args):
    t0 = time.time()
    g = load_lcc_graph(args.network)
    print(f"[{args.network}] LCC n={g.vcount()} m={g.ecount()} "
          f"({time.time()-t0:.1f}s)", flush=True)
    do_cc = not args.no_cc
    do_tri = not args.no_tri

    memb = leiden_membership(g)
    real = structural_metrics(g, memb, do_tri, do_cc)
    real.update(network=args.network, arm="real", rewire_seed="")
    reals = real

    rows = [real]
    seeds = [int(x) for x in args.rewire_seeds.split(",") if x.strip()]
    m = g.ecount()
    for rs in seeds:
        gr = rewire_copy(g, rs, SWAPS_PER_EDGE * m)
        mb = leiden_membership(gr)
        nl = structural_metrics(gr, mb, do_tri, do_cc)
        nl.update(network=args.network, arm="null", rewire_seed=rs)
        rows.append(nl)
        del gr

    # Exact decomposition relative to the real arm:
    #   delta = r_pb * sd_s / sqrt(p(1-p))
    #   log(delta_null/delta_real) = term_sorting + term_scale + term_granularity
    # NOTE: mu_s = n/m is EXACTLY invariant under degree-preserving rewiring, but
    # sd_s is NOT (s(e)=1/d_u+1/d_v depends on the pairing, not just the degree
    # sequence) -- hence the separate term_scale.
    for r in rows:
        pr, pn = reals["p_intra"], r["p_intra"]
        with np.errstate(divide="ignore", invalid="ignore"):
            r["term_sorting"] = float(np.log(r["r_pb"] / reals["r_pb"]))
            r["term_scale"] = float(np.log(r["sd_s"] / reals["sd_s"]))
            r["term_granularity"] = 0.5 * float(
                np.log((pr * (1 - pr)) / (pn * (1 - pn)))
            )
            r["term_sum"] = (r["term_sorting"] + r["term_scale"]
                             + r["term_granularity"])
            r["log_ratio"] = float(np.log(r["delta"] / reals["delta"]))

    out = Path(args.out) if args.out else PROBE_CSV
    for r in rows:
        append_row(out, PROBE_FIELDS, r)
        print(f"  {r['arm']:5s} seed={r['rewire_seed']!s:>3} k={r['n_comms']:6d} "
              f"Q={r['Q']:.4f} p={r['p_intra']:.4f} delta={r['delta']:+.5f} "
              f"r={r['r_pb']:+.4f} degR2={r['deg_R2']:.4f} "
              f"hubLift={r['hub_inter_lift']:.4f} C={r['transitivity']:.4f} "
              f"triIn={r['tri_intra']:.2f} triOut={r['tri_inter']:.2f}",
              flush=True)
    print(f"[{args.network}] done in {time.time()-t0:.1f}s", flush=True)


def cmd_staged(args):
    t0 = time.time()
    g = load_lcc_graph(args.network)
    m = g.ecount()
    total = SWAPS_PER_EDGE * m
    stages = [float(x) for x in args.stages.split(",")]
    print(f"[{args.network}] LCC n={g.vcount()} m={m} total_swaps={total}",
          flush=True)

    for seed in [int(x) for x in args.seeds.split(",")]:
        gr = g.copy()
        done = 0
        for frac in stages:
            target = int(round(frac * total))
            if target > done:
                rewire_inplace(gr, seed + 1000 * int(frac * 100 + 1), target - done)
                done = target
            mb = leiden_membership(gr)
            r = structural_metrics(gr, mb, do_triangles=True, do_cc=True)
            r.update(network=args.network, seed=seed, stage_frac=frac, swaps=done)
            append_row(STAGED_CSV, STAGED_FIELDS, r)
            print(f"  seed={seed} stage={frac:.2f} swaps={done} "
                  f"k={r['n_comms']} Q={r['Q']:.4f} p={r['p_intra']:.4f} "
                  f"delta={r['delta']:+.5f} r={r['r_pb']:+.4f} "
                  f"C={r['transitivity']:.4f} degR2={r['deg_R2']:.4f} "
                  f"triIn={r['tri_intra']:.2f}", flush=True)
        del gr
    print(f"[{args.network}] staged done in {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--network", required=True)
    p.add_argument("--rewire-seeds", default="7,8")
    p.add_argument("--no-cc", action="store_true")
    p.add_argument("--no-tri", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_probe)

    s = sub.add_parser("staged")
    s.add_argument("--network", required=True)
    s.add_argument("--seeds", default="7,8")
    s.add_argument("--stages", default="0,0.25,0.5,1.0")
    s.set_defaults(func=cmd_staged)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
