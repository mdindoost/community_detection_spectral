#!/usr/bin/env python
"""
Experiment E: does ANY structural statistic predict where the *genuine*
(runtime-matched) DSpar-seeded Leiden gain occurs?

Background
----------
Phase 1 (PAPER_RESTRUCTURE/PHASE1_VERDICT.md, exp_B_config_null/SUMMARY.md)
showed that the raw DSpar separation delta and DeltaQ_fixed are reproduced --
often exceeded -- by degree-preserving configuration-model nulls, so neither
certifies community structure. exp_C_true_retention_seeded found exactly one
genuine effect: DSpar-seeded Leiden *refined on the ORIGINAL graph* beating a
runtime-matched plain-Leiden-restarts baseline, and only robustly on
email-Enron (1 of 6 datasets).

Question here: over a wider network set (n=15 measured, 17 with predictors),
does any structural statistic predict the honest outcome

    y = Q_seeded - Q_matched_best      (calibrated DSpar, alpha = 0.9)

Candidate predictors (from exp_B results.csv real/null arms):
    delta_real            raw DSpar separation
    delta_star            delta_real - delta_null          ("excess separation")
    dQ_ratio              dQ_fixed_real / dQ_fixed_null
    hb_real               hub-bridge ratio
    hb_excess             hb_real - hb_null
    deg_cv                degree coefficient of variation (std/mean), LCC
  (+ auxiliaries: delta_ratio, dQ_excess, hb_ratio, n, m, avg_deg, Q_fixed_base
     real/null and their ratio)

Method (steps 2-3 reuse exp_C's code verbatim in spirit)
--------------------------------------------------------
* Loader: exp_B's chunked numpy SNAP parser -> undirected simple LCC igraph.
* Sampler: exp_C's *calibrated* DSpar -- same 1/d_u + 1/d_v scores, independent
  Bernoulli, but lambda solved by bisection so sum_e min(1, lambda*s_e) = alpha*m,
  i.e. E[retention] = alpha exactly (the repo's clipped sampler saturates near
  0.45 and cannot deliver true alpha=0.9).
* Seeded refinement: Leiden on G_sparse -> membership -> la.ModularityVertexPartition
  (G_original, initial_membership=...) -> Optimiser.optimise_partition.
* Runtime-matched baseline: plain Leiden restarts on the ORIGINAL graph with a
  wall-clock budget equal to mean(T_sparsify + T_leiden_sparse + T_seed);
  at least one restart; keep best Q.

Seeds are the same as exp_C (base 100..104, sparsify 200.., leiden 300..,
matched 900..) so the alpha=0.9 calibrated rows overlap and can be cross-checked.
exp_C used 5 sparsification seeds, this uses the first 3.

Usage
-----
  run.py predictors      -> predictors.csv  (17 networks)
  run.py outcomes        -> outcomes.csv    (15 networks)
  run.py analyze         -> correlations.csv
  run.py all
"""

import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg as la

REPO = Path("/home/md724/community_detection_spectral")
DATASETS_DIR = REPO / "datasets"
HERE = Path(__file__).resolve().parent
EXP_B_CSV = REPO / "PAPER_RESTRUCTURE/exp_B_config_null/results.csv"
EXP_C_CSV = REPO / "PAPER_RESTRUCTURE/exp_C_true_retention_seeded/results_summary.csv"

# 17 networks with exp_B predictors; the last two are too slow to measure.
ALL_NETWORKS = [
    "email-Eu-core", "wiki-Vote", "ca-GrQc", "ca-HepTh", "facebook-combined",
    "ca-CondMat", "ca-HepPh", "ca-AstroPh", "email-Enron", "cit-HepTh",
    "cit-HepPh", "com-Amazon", "com-DBLP", "com-Youtube", "wiki-Talk",
    "cit-Patents", "wiki-topcats",
]
MEASURED = ALL_NETWORKS[:15]

ALPHA = 0.9
N_BASE_SEEDS = 5
N_SPARSE_SEEDS = 3
N_ITER = 2  # repo convention (run_leiden default)


# ---------------------------------------------------------------------------
# Loading (verbatim from exp_B_config_null/run.py)
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
    if flat.size % 2:
        flat = flat[: (flat.size // 2) * 2]
    return flat.reshape(-1, 2)


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
    g = g.connected_components(mode="weak").giant()
    g.simplify(multiple=True, loops=True)
    return g


# ---------------------------------------------------------------------------
# Calibrated DSpar sampler (verbatim from exp_C_true_retention_seeded/run.py)
# ---------------------------------------------------------------------------

def dspar_scores(g):
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    return e, 1.0 / deg[e[:, 0]] + 1.0 / deg[e[:, 1]]


def _probs_calibrated(scores, alpha):
    """lambda by bisection s.t. sum(min(1, lambda*s)) = alpha*m => E[retention]=alpha."""
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


def sparsify(g, edge_arr, probs, seed):
    rs = np.random.RandomState(seed)
    keep = rs.random_sample(probs.size) < probs
    kept = edge_arr[keep]
    gs = ig.Graph(n=g.vcount())
    gs.add_edges(kept)
    return gs, kept.shape[0] / edge_arr.shape[0]


# ---------------------------------------------------------------------------
# Leiden (verbatim from exp_C)
# ---------------------------------------------------------------------------

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
    return part.membership, part.modularity, len(part), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Step 1: predictors
# ---------------------------------------------------------------------------

def read_exp_b():
    d = {}
    with open(EXP_B_CSV) as f:
        for r in csv.DictReader(f):
            d.setdefault(r["network"], {})[r["arm"]] = r
    return d


def build_predictors():
    b = read_exp_b()
    rows = []
    for name in ALL_NETWORKS:
        real, null = b[name]["real"], b[name]["null"]
        g = load_lcc_graph(name)
        deg = np.asarray(g.degree(), dtype=np.float64)
        n, m = g.vcount(), g.ecount()
        del g
        dr, dn = float(real["delta"]), float(null["delta"])
        hr, hn = float(real["hb"]), float(null["hb"])
        qr, qn = float(real["dQ_fixed"]), float(null["dQ_fixed"])
        br, bn = float(real["Q_fixed_base"]), float(null["Q_fixed_base"])
        rec = dict(
            network=name, n=n, m=m, avg_deg=2.0 * m / n,
            deg_cv=float(deg.std() / deg.mean()),
            deg_max=int(deg.max()),
            delta_real=dr, delta_null=dn,
            delta_star=dr - dn,
            delta_ratio=dr / dn if dn else float("nan"),
            hb_real=hr, hb_null=hn, hb_excess=hr - hn,
            hb_ratio=hr / hn if hn else float("nan"),
            dQ_real=qr, dQ_null=qn, dQ_excess=qr - qn,
            dQ_ratio=qr / qn if qn else float("nan"),
            Qfix_real=br, Qfix_null=bn, Qfix_ratio=br / bn if bn else float("nan"),
            delta_null_std=float(null["delta_std"]),
            hb_null_std=float(null["hb_std"]),
            dQ_null_std=float(null["dQ_fixed_std"]),
        )
        rows.append(rec)
        print(f"  predictors {name:18s} n={n:>9,} m={m:>10,} "
              f"d*={rec['delta_star']:+.4f} hb_exc={rec['hb_excess']:+.4f} "
              f"dQr={rec['dQ_ratio']:.3f} cv={rec['deg_cv']:.3f}", flush=True)
    with open(HERE / "predictors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Step 2-3: outcomes
# ---------------------------------------------------------------------------

def measure(name):
    t_load = time.perf_counter()
    g = load_lcc_graph(name)
    n, m = g.vcount(), g.ecount()
    print(f"\n{'='*78}\n{name}: n={n:,} m={m:,} (load {time.perf_counter()-t_load:.1f}s)"
          f"\n{'='*78}", flush=True)

    edge_arr, scores = dspar_scores(g)
    probs = _probs_calibrated(scores, ALPHA)

    base_Q, base_T = [], []
    for s in range(N_BASE_SEEDS):
        _, Q, nc, dt = leiden(g, seed=100 + s)
        base_Q.append(Q)
        base_T.append(dt)
        print(f"  base seed={100+s}: Q={Q:.6f} k={nc} t={dt:.2f}s", flush=True)
    Q_base_mean = float(np.mean(base_Q))
    Q_base_std = float(np.std(base_Q))
    Q_base_best = float(np.max(base_Q))
    T_leiden = float(statistics.median(base_T))

    rets, Qraw, Qseed, Tpipe, kfin = [], [], [], [], []
    for s in range(N_SPARSE_SEEDS):
        t0 = time.perf_counter()
        gs, ret = sparsify(g, edge_arr, probs, seed=200 + s)
        t_spar = time.perf_counter() - t0
        memb_s, _, _, t_ls = leiden(gs, seed=300 + s)
        q_raw = g.modularity(memb_s)
        del gs
        memb_f, q_seed, nc_f, t_sd = leiden(g, seed=300 + s, initial_membership=memb_s)
        rets.append(ret); Qraw.append(q_raw); Qseed.append(q_seed); kfin.append(nc_f)
        Tpipe.append(t_spar + t_ls + t_sd)
        print(f"  spar seed={200+s}: ret={ret:.4f} Qraw={q_raw:.6f} "
              f"Qseed={q_seed:.6f} k={nc_f} T_pipe={Tpipe[-1]:.2f}s", flush=True)

    budget = float(np.mean(Tpipe))
    best_matched, n_restarts, spent = -1.0, 0, 0.0
    while True:
        _, Qm, _, dt = leiden(g, seed=900 + n_restarts)
        n_restarts += 1
        spent += dt
        best_matched = max(best_matched, Qm)
        if spent >= budget:
            break

    Q_seeded_mean = float(np.mean(Qseed))
    rec = dict(
        network=name, n=n, m=m, alpha=ALPHA,
        actual_ret_mean=float(np.mean(rets)),
        Q_base_mean=Q_base_mean, Q_base_std=Q_base_std, Q_base_best=Q_base_best,
        Q_raw_mean=float(np.mean(Qraw)),
        Q_seeded_mean=Q_seeded_mean, Q_seeded_std=float(np.std(Qseed)),
        Q_seeded_best=float(np.max(Qseed)),
        Q_matched_best=float(best_matched), n_matched_restarts=n_restarts,
        T_leiden=T_leiden, T_pipe_mean=budget, T_matched_spent=spent,
        k_seeded_mean=float(np.mean(kfin)),
        y=Q_seeded_mean - float(best_matched),
        y2=Q_seeded_mean - Q_base_mean,
        y_best=float(np.max(Qseed)) - float(best_matched),
        raw_minus_base=float(np.mean(Qraw)) - Q_base_mean,
    )
    print(f"  ==> Qbase={Q_base_mean:.6f}+-{Q_base_std:.6f} "
          f"Qseed={Q_seeded_mean:.6f} Qmatch={best_matched:.6f} (x{n_restarts}) "
          f"y={rec['y']:+.6f} y2={rec['y2']:+.6f}", flush=True)
    del g, edge_arr, scores, probs
    return rec


def build_outcomes(networks):
    out_path = HERE / "outcomes.csv"
    rows = []
    if out_path.exists():
        with open(out_path) as f:
            rows = [r for r in csv.DictReader(f) if r["network"] not in networks]
    for name in networks:
        rows.append(measure(name))
        order = {nm: i for i, nm in enumerate(ALL_NETWORKS)}
        rows.sort(key=lambda r: order.get(r["network"], 99))
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(measure_fields()))
            w.writeheader()
            w.writerows(rows)
    return rows


def measure_fields():
    return ["network", "n", "m", "alpha", "actual_ret_mean", "Q_base_mean",
            "Q_base_std", "Q_base_best", "Q_raw_mean", "Q_seeded_mean",
            "Q_seeded_std", "Q_seeded_best", "Q_matched_best",
            "n_matched_restarts", "T_leiden", "T_pipe_mean", "T_matched_spent",
            "k_seeded_mean", "y", "y2", "y_best", "raw_minus_base"]


# ---------------------------------------------------------------------------
# Step 4: correlations
# ---------------------------------------------------------------------------

PREDICTORS = ["delta_real", "delta_star", "delta_ratio", "dQ_ratio", "dQ_excess",
              "hb_real", "hb_excess", "hb_ratio", "deg_cv", "Qfix_real",
              "Qfix_ratio", "avg_deg", "n", "m"]


def analyze():
    from scipy import stats
    pred = {r["network"]: r for r in csv.DictReader(open(HERE / "predictors.csv"))}
    out = list(csv.DictReader(open(HERE / "outcomes.csv")))
    names = [r["network"] for r in out]
    y = np.array([float(r["y"]) for r in out])
    y2 = np.array([float(r["y2"]) for r in out])
    nnet = len(names)

    rows = []
    for p in PREDICTORS:
        x = np.array([float(pred[nm][p]) for nm in names])
        for label, target in (("y", y), ("y2", y2)):
            sr, sp = stats.spearmanr(x, target)
            pr, pp = stats.pearsonr(x, target)
            # rank-based binary separation: AUC of predictor vs (y>0)
            rows.append(dict(predictor=p, outcome=label, n=nnet,
                             spearman_rho=sr, spearman_p=sp,
                             pearson_r=pr, pearson_p=pp))
    with open(HERE / "correlations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # leave-one-out for the top |spearman| predictor on y
    ys = [r for r in rows if r["outcome"] == "y"]
    top = max(ys, key=lambda r: abs(r["spearman_rho"]))["predictor"]
    x = np.array([float(pred[nm][top]) for nm in names])
    print(f"\nLeave-one-out sensitivity for top predictor '{top}' (outcome y):")
    loo = []
    for i in range(nnet):
        k = [j for j in range(nnet) if j != i]
        sr, sp = stats.spearmanr(x[k], y[k])
        loo.append((names[i], sr, sp))
        print(f"  drop {names[i]:18s} rho={sr:+.4f} p={sp:.4f}")
    print(f"  range rho = [{min(r[1] for r in loo):+.4f}, {max(r[1] for r in loo):+.4f}]")
    return rows, top, loo


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    nets = sys.argv[2:] or MEASURED
    if cmd in ("predictors", "all"):
        build_predictors()
    if cmd in ("outcomes", "all"):
        build_outcomes(nets)
    if cmd in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
