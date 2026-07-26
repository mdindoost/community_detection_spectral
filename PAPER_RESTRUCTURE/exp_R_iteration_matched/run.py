#!/usr/bin/env python
"""
Experiment R: iteration-matched plain-Leiden control on email-Enron (exp_N caveat C4).

Exp N compared the DSpar-seeded pipeline (2.289 s) against plain Leiden restarts
(1.236 s each) and against restart portfolios at matched wall clock.  The missing
arm is plain Leiden given the SAME budget as one pipeline by running MORE
ITERATIONS rather than more restarts.  Design pre-registered in DESIGN.md:

  Arm A  plain Leiden, n_iterations=2, seeds 100-124   (reference; seeds 100-104
         must reproduce exp_N's baseline rows byte-for-byte)
  Arm B  plain Leiden, n_iterations calibrated so median wall clock ~= 2.289 s
         (calibration on 3 throwaway seeds 500-502; if no n_iter lands within 5%
         of the target, BOTH bracketing values are run with 25 seeds and reported)
  Arm C  plain Leiden, n_iterations=-1 (run to convergence) -- the asymptote

The seeded arm is NOT re-run: exp_N/partition_quality.csv rows kind==seeded are
read directly (same graph, same loader, same conventions).

Loader / leiden conventions copied verbatim from exp_N/run.py (which copied them
from exp_C/run.py).

Outputs (this directory):
  results.csv   one row per Leiden run (calibration rows included, phase column)
  results.json  calibration record, per-arm summaries, P1-P3 verdicts, kill status
  run.log       stdout
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg as la
from scipy.stats import mannwhitneyu

REPO = Path("/home/md724/community_detection_spectral")
OUT = Path(__file__).resolve().parent
ENRON = REPO / "datasets/email-Enron/email-Enron.txt"
EXP_N_QUALITY = REPO / "PAPER_RESTRUCTURE/exp_N_enron_anatomy/partition_quality.csv"

MAIN_SEEDS = [100 + i for i in range(25)]        # arms A, B, C
CAL_SEEDS = [500, 501, 502]                      # throwaway calibration seeds
TARGET_SECONDS = 2.289                           # one seeded pipeline (exp_N finding 1)
EXP_N_PLAIN_SECONDS = 1.236                      # one plain restart (exp_N finding 1)
LADDER = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32]
CAL_TOL = 0.05                                   # relative tolerance for a single arm B

RESULTS = OUT / "results.csv"
FIELDS = ["arm", "phase", "n_iterations", "seed", "Q", "k", "seconds", "n_iters_run"]


# --------------------------------------------------------------------------
# exp_C / exp_N loader (verbatim)
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


def leiden(g, seed, n_iter):
    """exp_N's leiden(), with n_iterations exposed.  n_iter < 0 == run to
    convergence, implemented as a manual loop of single iterations so that the
    number of iterations actually taken is observable; this is exactly what
    leidenalg's optimise_partition(n_iterations=-1) does internally (it stops on
    the first iteration with diff_inc <= 0).  Equivalence is verified in
    calibration step [2b]."""
    t0 = time.perf_counter()
    part = la.ModularityVertexPartition(g)
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    if n_iter < 0:
        iters = 0
        while True:
            d = opt.optimise_partition(part, n_iterations=1)
            iters += 1
            if d <= 0:
                break
    else:
        opt.optimise_partition(part, n_iterations=n_iter)
        iters = n_iter
    dt = time.perf_counter() - t0
    return np.asarray(part.membership), part.modularity, len(part), dt, iters


# --------------------------------------------------------------------------
def append_row(row):
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def run_arm(g, arm, phase, n_iter, seeds):
    qs, ks, ts, its = [], [], [], []
    for s in seeds:
        _, Q, k, dt, iters = leiden(g, s, n_iter)
        append_row(dict(arm=arm, phase=phase, n_iterations=n_iter, seed=s,
                        Q=f"{Q:.10f}", k=k, seconds=f"{dt:.6f}", n_iters_run=iters))
        qs.append(Q); ks.append(k); ts.append(dt); its.append(iters)
        print(f"    {arm} n_iter={n_iter} seed={s}: Q={Q:.6f} k={k} "
              f"t={dt:.3f}s iters={iters}", flush=True)
    return np.array(qs), np.array(ks), np.array(ts), np.array(its)


def describe(q, k, t, it=None):
    d = dict(n=int(len(q)), mean_Q=float(q.mean()),
             sd_Q_pop=float(q.std(ddof=0)), sd_Q_sample=float(q.std(ddof=1)),
             best_Q=float(q.max()), worst_Q=float(q.min()),
             median_Q=float(np.median(q)),
             mean_k=float(k.mean()), sd_k=float(k.std(ddof=0)),
             median_seconds=float(np.median(t)), mean_seconds=float(t.mean()),
             min_seconds=float(t.min()), max_seconds=float(t.max()))
    if it is not None:
        d["mean_iters"] = float(np.mean(it))
        d["median_iters"] = float(np.median(it))
        d["max_iters"] = int(np.max(it))
    return d


# --------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    print("Loading email-Enron LCC ...", flush=True)
    g = load_graph(ENRON)
    print(f"  n={g.vcount():,} m={g.ecount():,}", flush=True)
    assert (g.vcount(), g.ecount()) == (33696, 180811), "graph does not match exp_N"

    # ---- exp_N reference numbers ----------------------------------------
    exp_n = list(csv.DictReader(open(EXP_N_QUALITY)))
    seeded = np.array([float(r["Q"]) for r in exp_n if r["kind"] == "seeded"])
    seeded_t = np.array([float(r["seconds"]) for r in exp_n if r["kind"] == "seeded"])
    seeded_k = np.array([int(r["k"]) for r in exp_n if r["kind"] == "seeded"])
    plain_n = np.array([float(r["Q"]) for r in exp_n
                        if r["kind"] in ("baseline", "extra_restart")])
    plain_n_t = np.array([float(r["seconds"]) for r in exp_n
                          if r["kind"] in ("baseline", "extra_restart")])
    base_n = {int(r["seed"]): float(r["Q"]) for r in exp_n if r["kind"] == "baseline"}
    print(f"\nexp_N reference: seeded n={len(seeded)} mean={seeded.mean():.6f} "
          f"sd={seeded.std(ddof=0):.6f} best={seeded.max():.6f} "
          f"median_t={np.median(seeded_t):.3f}s", flush=True)
    print(f"                 plain25 mean={plain_n.mean():.6f} "
          f"sd={plain_n.std(ddof=0):.6f} median_t={np.median(plain_n_t):.3f}s", flush=True)

    # ---- [1] arm A -------------------------------------------------------
    print("\n[1] arm A: plain Leiden n_iterations=2, seeds 100-124", flush=True)
    qA, kA, tA, _ = run_arm(g, "A", "main", 2, MAIN_SEEDS)
    A = describe(qA, kA, tA)
    print("  arm A: " + json.dumps(A), flush=True)

    # sanity: seeds 100-104 must reproduce exp_N's baseline rows exactly
    repro = {}
    for i, s in enumerate(MAIN_SEEDS[:5]):
        repro[s] = dict(exp_R=float(qA[i]), exp_N=base_n[s],
                        diff=float(qA[i] - base_n[s]))
    max_repro_diff = max(abs(v["diff"]) for v in repro.values())
    print(f"  reproduction check seeds 100-104: max |dQ| = {max_repro_diff:.3e}", flush=True)
    for s, v in repro.items():
        print(f"    seed {s}: exp_R {v['exp_R']:.6f} vs exp_N {v['exp_N']:.6f} "
              f"({v['diff']:+.2e})", flush=True)
    speed_ratio = float(np.median(tA) / EXP_N_PLAIN_SECONDS)
    print(f"  machine-speed check: arm A median {np.median(tA):.3f}s vs exp_N "
          f"1.236s -> {speed_ratio:.3f}x", flush=True)

    # ---- [2] calibration -------------------------------------------------
    print(f"\n[2] arm B calibration on seeds {CAL_SEEDS}, target "
          f"{TARGET_SECONDS:.3f}s (one seeded pipeline)", flush=True)
    cal = []
    for n_iter in LADDER:
        _, _, tc, _ = run_arm(g, "B_cal", "calibration", n_iter, CAL_SEEDS)
        med = float(np.median(tc))
        cal.append(dict(n_iterations=n_iter, median_seconds=med,
                        seconds=[float(x) for x in tc]))
        print(f"  n_iter={n_iter}: median {med:.3f}s "
              f"({med / TARGET_SECONDS:.2f}x target)", flush=True)
        if med >= TARGET_SECONDS:
            break

    below = [c for c in cal if c["median_seconds"] < TARGET_SECONDS]
    above = [c for c in cal if c["median_seconds"] >= TARGET_SECONDS]
    n_lo = below[-1]["n_iterations"] if below else None
    n_hi = above[0]["n_iterations"] if above else None
    best = min(cal, key=lambda c: abs(c["median_seconds"] - TARGET_SECONDS))
    within_tol = abs(best["median_seconds"] - TARGET_SECONDS) / TARGET_SECONDS <= CAL_TOL
    if within_tol:
        b_iters = [best["n_iterations"]]
    else:
        b_iters = [x for x in (n_lo, n_hi) if x is not None and x != 2]
        if not b_iters:                      # ladder exhausted below target
            b_iters = [LADDER[-1]]
    print(f"  calibration: bracket n_lo={n_lo} n_hi={n_hi}; closest n_iter="
          f"{best['n_iterations']} at {best['median_seconds']:.3f}s "
          f"({'within' if within_tol else 'outside'} {CAL_TOL:.0%} of target); "
          f"arm B n_iterations to run: {b_iters}", flush=True)

    # ---- [2b] equivalence check for the manual convergence loop -----------
    print("\n[2b] convergence-loop equivalence check (seeds 500-501)", flush=True)
    equiv = []
    for s in CAL_SEEDS[:2]:
        _, Qm, km, _, itm = leiden(g, s, -1)
        t0 = time.perf_counter()
        part = la.ModularityVertexPartition(g)
        opt = la.Optimiser()
        opt.set_rng_seed(int(s) % (2 ** 31 - 1))
        opt.optimise_partition(part, n_iterations=-1)
        Qn = part.modularity
        equiv.append(dict(seed=s, Q_manual_loop=float(Qm), Q_native=float(Qn),
                          diff=float(Qm - Qn), iters=int(itm),
                          native_seconds=float(time.perf_counter() - t0)))
        print(f"    seed {s}: manual {Qm:.10f} vs native -1 {Qn:.10f} "
              f"(diff {Qm - Qn:+.2e}), iters={itm}", flush=True)

    # ---- [3] arm B -------------------------------------------------------
    B = {}
    for n_iter in b_iters:
        print(f"\n[3] arm B (n_iterations={n_iter}), seeds 100-124", flush=True)
        qB, kB, tB, _ = run_arm(g, f"B_n{n_iter}", "main", n_iter, MAIN_SEEDS)
        B[n_iter] = dict(summary=describe(qB, kB, tB), Q=qB)
        print(f"  arm B(n={n_iter}): " + json.dumps(B[n_iter]["summary"]), flush=True)

    # ---- [4] arm C -------------------------------------------------------
    print("\n[4] arm C: plain Leiden to convergence (n_iterations=-1), seeds 100-124",
          flush=True)
    qC, kC, tC, itC = run_arm(g, "C", "main", -1, MAIN_SEEDS)
    C = describe(qC, kC, tC, itC)
    print("  arm C: " + json.dumps(C), flush=True)

    # ---- [5] tests -------------------------------------------------------
    def compare(q, label):
        u, p = mannwhitneyu(q, seeded, alternative="two-sided")
        # permutation test on the mean difference (two-sided), 200k relabelings
        rng = np.random.default_rng(0)
        pool = np.concatenate([q, seeded])
        obs = q.mean() - seeded.mean()
        na, nt = len(q), len(pool)
        cnt, NPERM = 0, 200000
        for start in range(0, NPERM, 20000):
            blk = min(20000, NPERM - start)
            idx = np.argsort(rng.random((blk, nt)), axis=1)
            vals = pool[idx]
            diffs = vals[:, :na].mean(axis=1) - vals[:, na:].mean(axis=1)
            cnt += int((np.abs(diffs) >= abs(obs) - 1e-15).sum())
        return dict(label=label, mean=float(q.mean()),
                    mean_minus_seeded=float(q.mean() - seeded.mean()),
                    sd_pop=float(q.std(ddof=0)), sd_sample=float(q.std(ddof=1)),
                    var_ratio_pop=float(q.var(ddof=0) / seeded.var(ddof=0)),
                    var_ratio_sample=float(q.var(ddof=1) / seeded.var(ddof=1)),
                    best=float(q.max()),
                    best_minus_seeded_best=float(q.max() - seeded.max()),
                    mannwhitney_U=float(u), mannwhitney_p=float(p),
                    perm_p=float((cnt + 1) / (NPERM + 1)),
                    frac_runs_above_seeded_mean=float((q > seeded.mean()).mean()),
                    frac_runs_above_seeded_best=float((q > seeded.max()).mean()),
                    percentile_of_seeded_mean_in_arm=float((q < seeded.mean()).mean() * 100))

    tests = dict(A=compare(qA, "A (n_iter=2)"), C=compare(qC, "C (converged)"))
    for n_iter, d in B.items():
        tests[f"B_n{n_iter}"] = compare(d["Q"], f"B (n_iter={n_iter})")

    print("\n[5] comparisons vs exp_N seeded arm", flush=True)
    print(json.dumps(tests, indent=2), flush=True)

    # ---- [6] pre-registered predictions + kill criterion -----------------
    seeded_mean = float(seeded.mean())
    seeded_sd = float(seeded.std(ddof=0))
    verdicts, kill = {}, {}
    for n_iter in B:
        t = tests[f"B_n{n_iter}"]
        verdicts[f"P1_B_n{n_iter}"] = dict(
            statement="arm B mean Q < seeded mean (0.6146)",
            arm_B_mean=t["mean"], seeded_mean=seeded_mean,
            holds=bool(t["mean"] < seeded_mean))
        verdicts[f"P3_B_n{n_iter}"] = dict(
            statement="arm B variance >= 1.5x seeded variance",
            var_ratio_pop=t["var_ratio_pop"], var_ratio_sample=t["var_ratio_sample"],
            sd_B_pop=t["sd_pop"], sd_seeded_pop=seeded_sd,
            holds=bool(t["var_ratio_pop"] >= 1.5))
        k1 = bool(t["mean"] >= seeded_mean - 0.001)
        k2 = bool(t["mannwhitney_p"] > 0.10 and abs(t["mean"] - seeded_mean) < 0.002)
        kill[f"B_n{n_iter}"] = dict(
            criterion_1_meanB_ge_seededmean_minus_0p001=k1,
            criterion_2_MWU_p_gt_0p10_and_mean_within_0p002=k2,
            fired=bool(k1 or k2),
            mean_B=t["mean"], threshold=seeded_mean - 0.001,
            mannwhitney_p=t["mannwhitney_p"],
            abs_mean_gap=abs(t["mean"] - seeded_mean))
    verdicts["P2"] = dict(
        statement="arm C (convergence) mean Q < seeded mean",
        arm_C_mean=tests["C"]["mean"], seeded_mean=seeded_mean,
        holds=bool(tests["C"]["mean"] < seeded_mean))

    print("\n[6] pre-registered predictions", flush=True)
    print(json.dumps(verdicts, indent=2), flush=True)
    print("\n    kill criterion", flush=True)
    print(json.dumps(kill, indent=2), flush=True)

    res = dict(
        graph=dict(n=g.vcount(), m=g.ecount()),
        exp_N_reference=dict(
            seeded=dict(n=int(len(seeded)), mean=seeded_mean, sd_pop=seeded_sd,
                        sd_sample=float(seeded.std(ddof=1)), best=float(seeded.max()),
                        worst=float(seeded.min()),
                        mean_k=float(seeded_k.mean()),
                        median_seconds=float(np.median(seeded_t)),
                        mean_seconds=float(seeded_t.mean())),
            plain25=dict(mean=float(plain_n.mean()), sd_pop=float(plain_n.std(ddof=0)),
                         best=float(plain_n.max()),
                         median_seconds=float(np.median(plain_n_t)))),
        arm_A=A, arm_B={str(k): v["summary"] for k, v in B.items()}, arm_C=C,
        reproduction_check=dict(rows=repro, max_abs_diff=max_repro_diff,
                                machine_speed_ratio_vs_expN=speed_ratio),
        calibration=dict(target_seconds=TARGET_SECONDS, ladder=cal, n_lo=n_lo,
                         n_hi=n_hi, closest=best, within_tolerance=bool(within_tol),
                         arms_run=b_iters,
                         ratio_matched_target_seconds=float(
                             np.median(tA) * TARGET_SECONDS / EXP_N_PLAIN_SECONDS)),
        convergence_equivalence=equiv,
        tests=tests, predictions=verdicts, kill_criterion=kill,
        elapsed_sec=time.perf_counter() - t_start)
    with open(OUT / "results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nDone in {res['elapsed_sec']:.1f}s -> results.csv + results.json", flush=True)


if __name__ == "__main__":
    main()
