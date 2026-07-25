#!/usr/bin/env python
"""
Supplement: run the NULL arm for UNIFORM at the retention DSpar actually
achieved in PAPER_RESTRUCTURE/exp_B_config_null/results.csv, so uniform's
Delta Q_fixed is compared with DSpar's at MATCHED retention rather than at
alpha=0.8/0.9. Appends arm='null_retmatched' rows to results.csv.
"""
import sys, os, time, csv
import numpy as np
import igraph as ig
import random as _random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (load_lcc, leiden, uniform_sparsify, append_rows,
                           already_done, append_summary, DATASETS, SEEDS,
                           REWIRE_MULT, log)

# retention DSpar actually achieved (exp_B, 'real' rows) + DSpar's dQ_fixed
DSPAR = {
    "ca-CondMat":  dict(ret=0.4723, dq_real=0.060564, dq_null=0.066655),
    "email-Enron": dict(ret=0.3889, dq_real=0.142247, dq_null=0.156214),
    "com-DBLP":    dict(ret=0.4688, dq_real=0.043729, dq_null=0.097967),
}

PATHS = dict(DATASETS)


def run(name):
    if already_done(name, "null_retmatched"):
        log(f"{name}: null_retmatched already done, skipping")
        return
    d = DSPAR[name]
    alpha = d["ret"]
    t0 = time.time()
    G = load_lcc(PATHS[name])
    n, m = G.vcount(), G.ecount()
    P0, _ = leiden(G, 0)
    Qf_real = G.modularity(P0)
    Grw = G.copy()
    ig.set_random_number_generator(_random.Random(12345))
    Grw.rewire(n=REWIRE_MULT * m, mode="simple")
    Prw, _ = leiden(Grw, 0)
    Qf_rw = Grw.modularity(Prw)
    log(f"{name}: n={n} m={m} alpha_matched={alpha} Qf_real={Qf_real:.6f} Qf_rw={Qf_rw:.6f}")

    rows = []
    res = {}
    for gname, GG, P, Qfull in (("real", G, P0, Qf_real), ("rewired", Grw, Prw, Qf_rw)):
        vals = []
        for s in SEEDS:
            Gs = uniform_sparsify(GG, alpha, 3000 + s)
            ms = Gs.ecount()
            Qfs = Gs.modularity(P)
            vals.append(Qfs - Qfull)
            rows.append(dict(dataset=name, arm="null_retmatched", graph=gname,
                             alpha=alpha, seed=s, n=GG.vcount(), m=GG.ecount(),
                             m_sparse=ms, retention=ms / GG.ecount(),
                             Q_fixed_full=Qfull, Q_fixed_sparse=Qfs,
                             dQ_fixed=Qfs - Qfull))
        res[gname] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
        log(f"  {gname}: dQ_fixed = {res[gname][0]:+.6f} +- {res[gname][1]:.6f}")

    append_rows(rows)
    append_summary(
        f"\nRETENTION-MATCHED NULL ARM -- {name} (uniform at alpha={alpha:.4f}, "
        f"the retention DSpar actually achieved)\n"
        f"  uniform dQ_fixed   real = {res['real'][0]:+.6f} +- {res['real'][1]:.6f}   "
        f"rewired = {res['rewired'][0]:+.6f} +- {res['rewired'][1]:.6f}\n"
        f"  DSpar   dQ_fixed   real = {d['dq_real']:+.6f}                rewired = "
        f"{d['dq_null']:+.6f}   [exp_B_config_null/results.csv]\n"
        f"  ratio DSpar/uniform on real graph: "
        f"{d['dq_real']/res['real'][0] if res['real'][0] else float('nan'):.1f}x\n")
    log(f"  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or list(DSPAR)):
        run(nm)
