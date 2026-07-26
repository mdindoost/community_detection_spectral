#!/usr/bin/env python3
"""
Experiment Q -- delta-suppression root cause: causal test of hub-edge placement.

Pre-registered design: exp_Q_delta_root_cause/DESIGN.md (registered 2026-07-25,
BEFORE this file existed).

Core idea
---------
delta = mu_intra(s) - mu_inter(s), s(e) = 1/d_u + 1/d_v, depends on (a) the score
multiset and (b) which edges the partition calls intra.  Degree-preserving
double-edge swaps hold the DEGREE SEQUENCE exactly fixed (hence the score of any
given *pair* is determined by the pair alone).  Freezing the partition P0 while
swapping edges isolates (b): we steer the placement of high degree-product
("hub") edges relative to P0's boundaries and watch delta(P0) respond.

Steering variable
-----------------
    M_inter = sum_{e in E, P0(e) = inter} d_u * d_v
  UP-chain   : accept a proposed double-edge swap iff it INCREASES M_inter
  DOWN-chain : accept iff it DECREASES M_inter
  NEUTRAL    : accept unconditionally (plain degree-preserving rewiring; the
               fluctuation-band control)
Degrees never change, so d_u d_v of any candidate pair is known in O(1) and
M_inter is updated incrementally (only 2 edges change per accepted swap).

All metric definitions are imported VERBATIM from exp_M_suppression_probe/run.py
so exp_Q numbers are directly comparable with exp_M's probe CSVs.

Subcommand
----------
    chains --networks a,b,c   run Arm 1 (+ Arm 2 re-detection checkpoints)
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import igraph as ig

HERE = Path(__file__).resolve().parent
EXPM_DIR = HERE.parent / "exp_M_suppression_probe"

# exp_M's run.py is loaded VERBATIM from disk (under a distinct module name, so
# it cannot collide with this file): loaders and every metric definition are
# shared with exp_M, guaranteeing the numbers are directly comparable.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("expM_run", EXPM_DIR / "run.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["expM_run"] = M
_spec.loader.exec_module(M)

from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402

TRAJ_CSV = HERE / "trajectories.csv"
CKPT_CSV = HERE / "checkpoints.csv"

LEIDEN_SEED = M.LEIDEN_SEED          # 42
CHAIN_SEEDS = [1, 2]                 # 2 chain seeds per direction
# (name, steering sign, p_intra-preserving?)
#   the first three are the LITERAL registered arms;
#   the *_pfix three are a SUPPLEMENTARY arm added after observing that the
#   literal rule does NOT hold p_intra fixed (DESIGN says "p_intra (constant by
#   construction)" -- it is not).  The pfix arms additionally require the swap to
#   leave the number of P0-inter edges unchanged, which makes p_intra EXACTLY
#   constant, so delta moves only through sorting (r_pb) and sd_s.
DIRECTIONS = [("up", +1, False), ("down", -1, False), ("neutral", 0, False),
              ("up_pfix", +1, True), ("down_pfix", -1, True),
              ("neutral_pfix", 0, True)]
TARGET_SPE = 10.0                    # accepted swaps per edge (chain length cap)
STALL_WINDOW = 50000                 # proposals
STALL_RATE = 0.005                   # <0.5% acceptance over the window => stall
# supplementary pfix arms: the extra constraint makes the raw acceptance rate
# intrinsically low (most uniform proposals change the intra count), so the
# registered 0.5%/50k stall rule would fire immediately and would mean nothing.
# Their own (documented, non-registered) budget:
TARGET_SPE_PFIX = 2.0
STALL_WINDOW_PFIX = 2000000
STALL_RATE_PFIX = 0.00002
MAX_PROP_PFIX = 80000000
# DESIGN registers "record trajectory every 0.5 swaps/edge".  Steering turns out
# to saturate well below 0.5 swaps/edge, so the schedule below is a SUPERSET of
# the registered 0.5 grid with extra points underneath it (denser recording of
# exactly the registered quantities; no change to the arms or the acceptance
# rule).  Arm 2 (fresh re-detection) is run at EVERY checkpoint, a superset of
# the registered start/mid/end.
CKPT_SPE_GRID = ([0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
                 + [round(0.5 * k, 2) for k in range(1, 21)])

STRUCT_FIELDS = [
    "n", "m", "n_comms", "Q", "p_intra", "sd_s", "mu_s",
    "delta", "mu_intra", "mu_inter", "r_pb", "auc_s", "auc_prod",
    "hb", "deg_R2", "hub_inter_frac", "nonhub_inter_frac", "hub_inter_lift",
    "assort_deg", "transitivity", "avg_local_cc",
    "tri_intra", "tri_inter", "tri_frac_intra_ge1", "tri_frac_inter_ge1",
    "comm_size_mean", "comm_size_max_frac", "comm_size_gini",
]

TRAJ_FIELDS = (["network", "chain", "chain_seed", "spe", "spe_nom", "accepted",
                "proposals", "acc_rate_cum", "acc_rate_interval",
                "M_inter", "M_total", "final", "stalled"]
               + STRUCT_FIELDS + ["seconds"])

CKPT_FIELDS = (["network", "chain", "chain_seed", "spe", "spe_nom", "accepted",
                "ami_P0", "final", "stalled"] +
               ["fresh_" + f for f in STRUCT_FIELDS] + ["seconds"])


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def build_state(g, memb):
    E = np.asarray(g.get_edgelist(), dtype=np.int64)
    eu = E[:, 0].tolist()
    ev = E[:, 1].tolist()
    deg = [int(x) for x in g.degree()]
    comm = [int(x) for x in memb]
    N = int(g.vcount())
    eset = set()
    for a, b in zip(eu, ev):
        eset.add(a * N + b if a < b else b * N + a)
    assert len(eset) == len(eu), "duplicate edges in input"
    return eu, ev, deg, comm, N, eset


def mass_inter(eu, ev, deg, comm):
    tot = 0
    inter = 0
    for a, b in zip(eu, ev):
        w = deg[a] * deg[b]
        tot += w
        if comm[a] != comm[b]:
            inter += w
    return inter, tot


def make_graph(eu, ev, N):
    g = ig.Graph(n=N)
    g.add_edges(np.column_stack([np.asarray(eu, dtype=np.int64),
                                 np.asarray(ev, dtype=np.int64)]))
    return g


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


def completed_chains(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("final") == "1":
                done.add((r["network"], r["chain"], int(r["chain_seed"])))
    return done


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------

def run_chain(network, direction_name, sign, pfix, seed, g0, memb, log):
    t_chain = time.time()
    eu, ev, deg, comm, N, eset = build_state(g0, memb)
    m = len(eu)
    M_inter, M_total = mass_inter(eu, ev, deg, comm)

    tgt_spe = TARGET_SPE_PFIX if pfix else TARGET_SPE
    swin = STALL_WINDOW_PFIX if pfix else STALL_WINDOW
    srate = STALL_RATE_PFIX if pfix else STALL_RATE
    max_prop = MAX_PROP_PFIX if pfix else float("inf")
    target = int(round(tgt_spe * m))
    sched = [(sp, int(round(sp * m))) for sp in CKPT_SPE_GRID
             if 0 < sp <= tgt_spe]
    rng = np.random.default_rng(
        1000 * seed + {"up": 1, "down": 2, "neutral": 3,
                       "up_pfix": 4, "down_pfix": 5,
                       "neutral_pfix": 6}[direction_name])

    accepted = 0
    proposals = 0
    last_ck_acc = 0
    last_ck_prop = 0
    win_p = 0
    win_a = 0
    stalled = False

    def checkpoint(final, spe_nom):
        nonlocal last_ck_acc, last_ck_prop
        t0 = time.time()
        spe = accepted / m
        gg = make_graph(eu, ev, N)
        r = M.structural_metrics(gg, memb, do_triangles=True, do_cc=True)
        row = dict(r)
        d_acc = accepted - last_ck_acc
        d_prop = proposals - last_ck_prop
        row.update(network=network, chain=direction_name, chain_seed=seed,
                   spe=round(spe, 4), spe_nom=spe_nom,
                   accepted=accepted, proposals=proposals,
                   acc_rate_cum=(accepted / proposals) if proposals else "",
                   acc_rate_interval=(d_acc / d_prop) if d_prop else "",
                   M_inter=M_inter, M_total=M_total,
                   final=1 if final else 0, stalled=1 if stalled else 0)
        append_row(TRAJ_CSV, TRAJ_FIELDS, row)
        last_ck_acc, last_ck_prop = accepted, proposals
        print(f"  [{network}/{direction_name}/s{seed}] spe={spe:6.3f} "
              f"acc={accepted:8d} prop={proposals:10d} "
              f"accR={row['acc_rate_interval'] if d_prop else 0:.4f} "
              f"Minter/Mtot={M_inter / M_total:.4f} p={r['p_intra']:.4f} "
              f"delta={r['delta']:+.5f} r={r['r_pb']:+.4f} "
              f"hubLift={r['hub_inter_lift']:.4f} hb={r['hb']:.4f} "
              f"triIn={r['tri_intra']:.3f} triOut={r['tri_inter']:.3f} "
              f"({time.time()-t0:.1f}s)", file=log, flush=True)

        # Arm 2 -- fresh re-detection (run at every checkpoint)
        if True:
            t1 = time.time()
            mb2 = M.leiden_membership(gg, seed=LEIDEN_SEED)
            r2 = M.structural_metrics(gg, mb2, do_triangles=True, do_cc=True)
            crow = {("fresh_" + k): v for k, v in r2.items() if k in STRUCT_FIELDS}
            crow.update(network=network, chain=direction_name, chain_seed=seed,
                        spe=round(spe, 4), spe_nom=spe_nom, accepted=accepted,
                        ami_P0=float(adjusted_mutual_info_score(memb, mb2)),
                        final=1 if final else 0, stalled=1 if stalled else 0,
                        seconds=round(time.time() - t1, 1))
            append_row(CKPT_CSV, CKPT_FIELDS, crow)
            print(f"      fresh: k={r2['n_comms']} Q={r2['Q']:.4f} "
                  f"p={r2['p_intra']:.4f} delta={r2['delta']:+.5f} "
                  f"AMI(P0)={crow['ami_P0']:.4f}", file=log, flush=True)
        del gg

    checkpoint(False, 0.0)
    ck_idx = 0
    next_ck = sched[0][1]
    B = 200000

    while accepted < target and not stalled:
        I = rng.integers(0, m, size=B).tolist()
        J = rng.integers(0, m, size=B).tolist()
        F = rng.integers(0, 2, size=B).tolist()
        for t in range(B):
            i = I[t]
            j = J[t]
            proposals += 1
            win_p += 1
            if i != j:
                a = eu[i]
                b = ev[i]
                if F[t]:
                    c = ev[j]
                    d = eu[j]
                else:
                    c = eu[j]
                    d = ev[j]
                if a != d and c != b:
                    k1 = a * N + d if a < d else d * N + a
                    k2 = c * N + b if c < b else b * N + c
                    if k1 != k2 and k1 not in eset and k2 not in eset:
                        ca = comm[a]
                        cb = comm[b]
                        cc = comm[c]
                        cd = comm[d]
                        da = deg[a]
                        db = deg[b]
                        dc = deg[c]
                        dd = deg[d]
                        n1 = ca != cd
                        n2 = cc != cb
                        o1 = ca != cb
                        o2 = cc != cd
                        if pfix and (n1 + n2) != (o1 + o2):
                            ok = False
                        else:
                            dM = 0
                            if n1:
                                dM += da * dd
                            if n2:
                                dM += dc * db
                            if o1:
                                dM -= da * db
                            if o2:
                                dM -= dc * dd
                            ok = (sign == 0 or
                                  (dM > 0 if sign > 0 else dM < 0))
                        if ok:
                            eset.discard(a * N + b if a < b else b * N + a)
                            eset.discard(c * N + d if c < d else d * N + c)
                            eset.add(k1)
                            eset.add(k2)
                            eu[i] = a
                            ev[i] = d
                            eu[j] = c
                            ev[j] = b
                            M_inter += dM
                            M_total += da * dd + dc * db - da * db - dc * dd
                            accepted += 1
                            win_a += 1
                            if accepted >= next_ck:
                                checkpoint(False, sched[ck_idx][0])
                                ck_idx += 1
                                next_ck = (sched[ck_idx][1]
                                           if ck_idx < len(sched)
                                           else target + 1)
                            if accepted >= target:
                                break
            if win_p >= swin:
                if win_a < srate * win_p or proposals >= max_prop:
                    stalled = True
                    break
                win_p = 0
                win_a = 0

    checkpoint(True, round(accepted / m, 4))
    print(f"  [{network}/{direction_name}/s{seed}] DONE spe={accepted/m:.3f} "
          f"stalled={stalled} accepted={accepted} proposals={proposals} "
          f"({time.time()-t_chain:.1f}s)", file=log, flush=True)


def cmd_chains(args):
    log = sys.stdout
    done = completed_chains(TRAJ_CSV)
    for network in [x for x in args.networks.split(",") if x.strip()]:
        t0 = time.time()
        g0 = M.load_lcc_graph(network)
        memb = M.leiden_membership(g0, seed=LEIDEN_SEED)
        base = M.structural_metrics(g0, memb, do_triangles=True, do_cc=True)
        print(f"[{network}] LCC n={g0.vcount()} m={g0.ecount()} k={base['n_comms']} "
              f"Q={base['Q']:.6f} p={base['p_intra']:.6f} "
              f"delta(P0)={base['delta']:.6f} r_pb={base['r_pb']:.6f} "
              f"hubLift={base['hub_inter_lift']:.6f} hb={base['hb']:.6f} "
              f"sd_s={base['sd_s']:.6f} ({time.time()-t0:.1f}s)  "
              f"[SANITY GATE 2: compare vs exp_M delta_real]", file=log, flush=True)
        for dname, sign, pfix in DIRECTIONS:
            for seed in CHAIN_SEEDS:
                if (network, dname, seed) in done:
                    print(f"  skip {network}/{dname}/s{seed} (done)",
                          file=log, flush=True)
                    continue
                run_chain(network, dname, sign, pfix, seed, g0, memb, log)
        del g0
        print(f"[{network}] all chains done ({time.time()-t0:.1f}s)",
              file=log, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("chains")
    p.add_argument("--networks", required=True)
    p.set_defaults(func=cmd_chains)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
