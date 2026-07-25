#!/usr/bin/env python
"""
exp_G: UNIFORM random sparsification through the Phase-1 controls.

Arm (a) Artifact I: Leiden on sparsified graph, scored on the sparsified graph
        (Q_sparse, "paper" convention) AND on the original graph (Q_on_orig,
        honest convention). Compared against a 5-seed plain-Leiden baseline.
Arm (b) Null: degree-preserving rewire (10*m double-edge swaps), FIXED Leiden
        partition, Delta Q_fixed under uniform sparsification on real vs rewired.

Writes results.csv and SUMMARY.txt incrementally after EACH dataset.
"""
import csv
import os
import sys
import time
import numpy as np
import igraph as ig
import leidenalg as la

OUT = "/home/md724/community_detection_spectral/PAPER_RESTRUCTURE/exp_G_other_sparsifiers"
CSV = os.path.join(OUT, "results.csv")
SUM = os.path.join(OUT, "SUMMARY.txt")

DATASETS = [
    ("ca-CondMat", "/home/md724/community_detection_spectral/datasets/ca-CondMat/ca-CondMat.txt"),
    ("email-Enron", "/home/md724/community_detection_spectral/datasets/email-Enron/email-Enron.txt"),
    ("com-DBLP", "/home/md724/community_detection_spectral/datasets/com-DBLP/com-DBLP.txt"),
]

ALPHAS = [0.8, 0.9]
SEEDS = [0, 1, 2]
BASE_SEEDS = [0, 1, 2, 3, 4]
N_ITER = 2
REWIRE_MULT = 10

FIELDS = ["dataset", "arm", "graph", "alpha", "seed", "n", "m", "m_sparse",
          "retention", "Q_base_mean", "Q_base_std", "Q_sparse", "Q_on_orig",
          "paper_delta", "honest_delta", "Q_fixed_full", "Q_fixed_sparse", "dQ_fixed"]


def log(*a):
    print(*a, flush=True)


def load_lcc(path):
    """Load SNAP edge list -> simple undirected graph -> largest connected component."""
    raw = np.loadtxt(path, dtype=np.int64, comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(-1, 2)
    raw = raw[:, :2]
    # relabel to 0..n-1
    uniq, inv = np.unique(raw, return_inverse=True)
    e = inv.reshape(raw.shape)
    # drop self loops, canonicalise, dedupe
    e = e[e[:, 0] != e[:, 1]]
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    G = ig.Graph(n=len(uniq), edges=e.tolist(), directed=False)
    comps = G.connected_components()
    G = comps.giant()
    G.simplify()
    return G


def leiden(G, seed):
    p = la.find_partition(G, la.ModularityVertexPartition, seed=seed, n_iterations=N_ITER)
    return p.membership, G.modularity(p.membership)


def uniform_sparsify(G, alpha, seed):
    """Bernoulli(p=alpha) edge retention. Returns graph on the SAME vertex set."""
    rng = np.random.default_rng(seed)
    m = G.ecount()
    keep = rng.random(m) < alpha
    edges = np.array(G.get_edgelist(), dtype=np.int64)[keep]
    return ig.Graph(n=G.vcount(), edges=edges.tolist(), directed=False)


def already_done(dataset, arm):
    """Resume guard: True if (dataset, arm) rows are already in results.csv."""
    if not os.path.exists(CSV):
        return False
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["arm"] == arm:
                return True
    return False


def append_rows(rows):
    if not rows:
        return
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def append_summary(text):
    with open(SUM, "a") as f:
        f.write(text)


def run_dataset(name, path):
    t0 = time.time()
    log(f"\n=== {name} ===")
    G = load_lcc(path)
    n, m = G.vcount(), G.ecount()
    log(f"  LCC: n={n} m={m}  ({time.time()-t0:.1f}s)")

    rows = []

    # ---------------- baseline: 5-seed plain Leiden on original ----------------
    Qs, parts = [], []
    for s in BASE_SEEDS:
        t = time.time()
        mem, q = leiden(G, s)
        Qs.append(q)
        parts.append(mem)
        log(f"  base seed {s}: Q={q:.6f}  ({time.time()-t:.1f}s)")
    Q_base_mean = float(np.mean(Qs))
    Q_base_std = float(np.std(Qs, ddof=1))
    P0 = parts[0]
    Q_fixed_full_real = G.modularity(P0)
    log(f"  Q_base = {Q_base_mean:.6f} +- {Q_base_std:.6f}")

    for s, q in zip(BASE_SEEDS, Qs):
        rows.append(dict(dataset=name, arm="baseline", graph="real", alpha=1.0, seed=s,
                         n=n, m=m, m_sparse=m, retention=1.0,
                         Q_base_mean=Q_base_mean, Q_base_std=Q_base_std, Q_sparse=q))

    # ---------------- arm (a) Artifact I ----------------
    for alpha in ALPHAS:
        for s in SEEDS:
            t = time.time()
            Gs = uniform_sparsify(G, alpha, 1000 + s)
            ms = Gs.ecount()
            mem, Q_sparse = leiden(Gs, s)
            Q_on_orig = G.modularity(mem)
            rows.append(dict(dataset=name, arm="artifact1", graph="real", alpha=alpha, seed=s,
                             n=n, m=m, m_sparse=ms, retention=ms / m,
                             Q_base_mean=Q_base_mean, Q_base_std=Q_base_std,
                             Q_sparse=Q_sparse, Q_on_orig=Q_on_orig,
                             paper_delta=Q_sparse - Q_base_mean,
                             honest_delta=Q_on_orig - Q_base_mean))
            log(f"  artifact1 a={alpha} s={s}: ret={ms/m:.4f} Q_sparse={Q_sparse:.6f} "
                f"Q_on_orig={Q_on_orig:.6f} paper_d={Q_sparse-Q_base_mean:+.6f} "
                f"honest_d={Q_on_orig-Q_base_mean:+.6f}  ({time.time()-t:.1f}s)")

    # write arm (a) now, before the expensive rewire
    if already_done(name, "artifact1"):
        log("  [artifact1 rows already in results.csv -- not re-appending]")
    else:
        append_rows(rows)
        log(f"  [wrote {len(rows)} rows]")
    rows_a = list(rows)
    rows = []

    # ---------------- arm (b) null: degree-preserving rewire ----------------
    t = time.time()
    Grw = G.copy()
    import random as _random
    ig.set_random_number_generator(_random.Random(12345))
    Grw.rewire(n=REWIRE_MULT * m, mode="simple")
    log(f"  rewired ({REWIRE_MULT}*m={REWIRE_MULT*m} swaps) in {time.time()-t:.1f}s; "
        f"m={Grw.ecount()} degseq_match={sorted(Grw.degree())==sorted(G.degree())}")
    t = time.time()
    Prw, Q_rw = leiden(Grw, 0)
    Q_fixed_full_rw = Grw.modularity(Prw)
    log(f"  rewired Leiden Q={Q_rw:.6f}  ({time.time()-t:.1f}s)")

    for gname, GG, P, Qfull in (("real", G, P0, Q_fixed_full_real),
                                ("rewired", Grw, Prw, Q_fixed_full_rw)):
        for alpha in ALPHAS:
            for s in SEEDS:
                Gs = uniform_sparsify(GG, alpha, 2000 + s)
                ms = Gs.ecount()
                Qfs = Gs.modularity(P)
                rows.append(dict(dataset=name, arm="null", graph=gname, alpha=alpha, seed=s,
                                 n=GG.vcount(), m=GG.ecount(), m_sparse=ms,
                                 retention=ms / GG.ecount(),
                                 Q_fixed_full=Qfull, Q_fixed_sparse=Qfs,
                                 dQ_fixed=Qfs - Qfull))
                log(f"  null {gname} a={alpha} s={s}: Q_fixed_full={Qfull:.6f} "
                    f"Q_fixed_sparse={Qfs:.6f} dQ_fixed={Qfs-Qfull:+.6f}")

    if already_done(name, "null"):
        log("  [null rows already in results.csv -- not re-appending]")
    else:
        append_rows(rows)
        log(f"  [wrote {len(rows)} rows]")

    write_summary_block(name, n, m, Q_base_mean, Q_base_std, rows_a, rows)
    log(f"  === {name} done in {time.time()-t0:.1f}s ===")


def agg(rows, arm, graph, alpha, key):
    v = [r[key] for r in rows if r["arm"] == arm and r.get("graph") == graph
         and r["alpha"] == alpha]
    return (float(np.mean(v)), float(np.std(v, ddof=1))) if v else (float("nan"),) * 2


def write_summary_block(name, n, m, Qb, Qbs, rows_a, rows_b):
    L = []
    L.append(f"\n{'='*78}\n{name}   n={n}  m={m}   Q_base(5 seeds) = {Qb:.6f} +- {Qbs:.6f}\n{'='*78}\n")
    L.append("ARM (a) ARTIFACT I -- uniform Bernoulli sparsification\n")
    L.append(f"{'alpha':>6} {'ret':>7} {'Q_sparse':>12} {'Q_on_orig':>12} {'paper_delta':>14} {'honest_delta':>14}\n")
    for alpha in ALPHAS:
        rm, _ = agg(rows_a, "artifact1", "real", alpha, "retention")
        qs, qss = agg(rows_a, "artifact1", "real", alpha, "Q_sparse")
        qo, qos = agg(rows_a, "artifact1", "real", alpha, "Q_on_orig")
        pd, pds = agg(rows_a, "artifact1", "real", alpha, "paper_delta")
        hd, hds = agg(rows_a, "artifact1", "real", alpha, "honest_delta")
        L.append(f"{alpha:>6} {rm:>7.4f} {qs:>7.6f}+-{qss:.4f} {qo:>7.6f}+-{qos:.4f} "
                 f"{pd:>+9.6f}+-{pds:.4f} {hd:>+9.6f}+-{hds:.4f}\n")
    L.append("\nARM (b) NULL -- Delta Q_fixed under uniform sparsification\n")
    L.append(f"{'alpha':>6} {'dQ_fixed(real)':>22} {'dQ_fixed(rewired)':>22} {'ratio rw/real':>14}\n")
    for alpha in ALPHAS:
        dr, drs = agg(rows_b, "null", "real", alpha, "dQ_fixed")
        dw, dws = agg(rows_b, "null", "rewired", alpha, "dQ_fixed")
        ratio = dw / dr if dr else float("nan")
        L.append(f"{alpha:>6} {dr:>+15.6f}+-{drs:.5f} {dw:>+15.6f}+-{dws:.5f} {ratio:>14.3f}\n")
    append_summary("".join(L))


if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for nm, p in DATASETS:
        if only and nm not in only:
            continue
        try:
            run_dataset(nm, p)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            append_summary(f"\n### {nm} FAILED: {ex}\n")
