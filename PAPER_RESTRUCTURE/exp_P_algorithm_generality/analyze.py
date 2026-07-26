#!/usr/bin/env python
"""Exp P analysis: evaluate P1-P4 and the reverse kill criterion exactly as
pre-registered in DESIGN.md.  Reads results.csv / runs.csv / recovery.csv.

Beyond-noise gain (the operationalisation of "beyond seed noise" used for P1
and the reverse kill criterion), applied per (network, algo, sparsifier) cell:
    dQ_vs_matched > 0            (beats the runtime-matched same-algorithm best)
  AND dQ_honest_vs_mean > 2*sd   (sd = pooled sd of baseline and sparse Q_orig)
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path



class _A(list):
    """tiny numpy-free stand-in for the 1-D float arrays used below."""
    def mean(self):
        return sum(self) / len(self) if self else float("nan")
    def std(self):
        if len(self) < 2:
            return 0.0
        m = self.mean()
        return (sum((x - m) ** 2 for x in self) / len(self)) ** 0.5
    def min(self):
        return min(self) if self else float("nan")
    def max(self):
        return max(self) if self else float("nan")
    def __gt__(self, v):
        return _A(1.0 if x > v else 0.0 for x in self)
    def sum(self):
        return sum(self)


class _np:
    @staticmethod
    def array(x):
        return _A(x)
    @staticmethod
    def median(a):
        b = sorted(a)
        n = len(b)
        if not n:
            return float("nan")
        return b[n // 2] if n % 2 else 0.5 * (b[n // 2 - 1] + b[n // 2])
    @staticmethod
    def mean(a):
        a = list(a)
        return sum(a) / len(a) if a else float("nan")
    @staticmethod
    def argmax(a):
        a = list(a)
        return max(range(len(a)), key=lambda i: a[i])


np = _np()

HERE = Path(__file__).resolve().parent
NETS = ["email-Eu-core", "wiki-Vote", "ca-HepTh", "ca-CondMat", "email-Enron",
        "com-DBLP", "com-Amazon"]
ALGOS = ["infomap", "louvain", "labelprop"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def main():
    rows = load(HERE / "results.csv")
    for r in rows:
        r["_gain"] = (f(r["dQ_vs_matched"]) > 0 and
                      f(r["dQ_honest_vs_mean"]) > 2 * f(r["pooled_sd"]))
    arms = sorted({(r["sparsifier"], r["target_ret"]) for r in rows})

    print("=" * 100)
    print("TABLE 1 — honest transfer, Q_orig (per network x algo x arm)")
    print("=" * 100)
    hdr = (f"{'network':14s} {'algo':10s} {'arm':12s} {'ret':>6s} {'Q_base':>8s} "
           f"{'sd':>7s} {'Q_orig':>8s} {'dQ_mean':>9s} {'dQ_best':>9s} "
           f"{'dQ_match':>9s} {'2sd':>7s} {'k_b':>7s} {'k_s':>8s} {'r':>3s} {'GAIN':>5s}")
    print(hdr)
    for net in NETS:
        for algo in ALGOS:
            for arm, tgt in arms:
                for r in rows:
                    if (r["network"] == net and r["algo"] == algo
                            and r["sparsifier"] == arm and r["target_ret"] == tgt):
                        print(f"{net:14s} {algo:10s} {arm+'@'+tgt:12s} "
                              f"{f(r['realized_ret']):6.3f} {f(r['Q_base_mean']):8.4f} "
                              f"{f(r['Q_base_std']):7.4f} {f(r['Q_orig_mean']):8.4f} "
                              f"{f(r['dQ_honest_vs_mean']):+9.4f} "
                              f"{f(r['dQ_honest_vs_best']):+9.4f} "
                              f"{f(r['dQ_vs_matched']):+9.4f} "
                              f"{2*f(r['pooled_sd']):7.4f} {f(r['k_base']):7.0f} "
                              f"{f(r['k_sparse']):8.0f} {int(f(r['n_restarts'])):3d} "
                              f"{'YES' if r['_gain'] else '.':>5s}")

    print()
    print("=" * 100)
    print("P1 / REVERSE KILL — networks with a beyond-noise controlled gain")
    print("=" * 100)
    print(f"{'algo':10s} {'arm':14s} {'n_gain/n_cells':>16s}   networks")
    for algo in ALGOS:
        for arm, tgt in arms:
            cells = [r for r in rows if r["algo"] == algo
                     and r["sparsifier"] == arm and r["target_ret"] == tgt]
            g = [r["network"] for r in cells if r["_gain"]]
            print(f"{algo:10s} {arm+'@'+tgt:14s} {len(g):>7d}/{len(cells):<8d}   "
                  f"{', '.join(g) if g else '-'}")
    print()
    print("per-algorithm union over sparsifier arms (reverse kill is per algorithm):")
    for algo in ALGOS:
        nets_gain = sorted({r["network"] for r in rows
                            if r["algo"] == algo and r["_gain"]})
        ncov = len({r["network"] for r in rows if r["algo"] == algo})
        verdict = "REVERSE-KILL TRIGGERED" if len(nets_gain) >= 3 else "negatives hold"
        print(f"  {algo:10s} {len(nets_gain)}/{ncov} networks  {verdict}   "
              f"{', '.join(nets_gain) if nets_gain else '-'}")
    print()
    print("also: cells with dQ_vs_matched > 0 regardless of noise test:")
    for algo in ALGOS:
        pos = sorted({r["network"] for r in rows
                      if r["algo"] == algo and f(r["dQ_vs_matched"]) > 0})
        print(f"  {algo:10s} {len(pos)}: {', '.join(pos) if pos else '-'}")

    print()
    print("=" * 100)
    print("P2 — which algorithm degrades most (mean dQ over all cells) + fragmentation")
    print("=" * 100)
    print(f"{'algo':10s} {'mean dQ_match':>14s} {'median dQ_match':>16s} "
          f"{'mean dQ_mean':>13s} {'mean k_s/k_b':>13s} {'median k_s/k_b':>15s}")
    for algo in ALGOS:
        cells = [r for r in rows if r["algo"] == algo]
        dm = np.array([f(r["dQ_vs_matched"]) for r in cells])
        dn = np.array([f(r["dQ_honest_vs_mean"]) for r in cells])
        kr = np.array([f(r["k_sparse"]) / max(f(r["k_base"]), 1e-9) for r in cells])
        print(f"{algo:10s} {dm.mean():+14.4f} {np.median(dm):+16.4f} "
              f"{dn.mean():+13.4f} {kr.mean():13.2f} {np.median(kr):15.2f}")
    print()
    print("per-arm means:")
    for arm, tgt in arms:
        print(f"  {arm}@{tgt}:")
        for algo in ALGOS:
            cells = [r for r in rows if r["algo"] == algo
                     and r["sparsifier"] == arm and r["target_ret"] == tgt]
            if not cells:
                continue
            dm = np.array([f(r["dQ_vs_matched"]) for r in cells])
            print(f"    {algo:10s} mean dQ_vs_matched {dm.mean():+.4f}  "
                  f"range [{dm.min():+.4f}, {dm.max():+.4f}]  n={len(cells)}")

    print()
    print("=" * 100)
    print("Artifact I check — naive (sparse-graph) scoring vs honest transfer")
    print("=" * 100)
    for algo in ALGOS:
        cells = [r for r in rows if r["algo"] == algo]
        dn = np.array([f(r["dQ_naive"]) for r in cells])
        dh = np.array([f(r["dQ_honest_vs_mean"]) for r in cells])
        flips = sum(1 for a, b in zip(dn, dh) if a > 0 and b < 0)
        print(f"  {algo:10s} dQ_naive>0 in {int((dn>0).sum())}/{len(cells)}, "
              f"dQ_honest>0 in {int((dh>0).sum())}/{len(cells)}, sign flips {flips}")

    print()
    print("=" * 100)
    print("P4 — end-to-end speedup vs a single run of the same algorithm")
    print("=" * 100)
    print(f"{'algo':10s} {'arm':14s} {'max spd':>9s} {'median':>9s} "
          f"{'n>1.5x':>7s}  argmax")
    for algo in ALGOS:
        for arm, tgt in arms:
            cells = [r for r in rows if r["algo"] == algo
                     and r["sparsifier"] == arm and r["target_ret"] == tgt]
            if not cells:
                continue
            sp = np.array([f(r["speedup_vs_single"]) for r in cells])
            i = int(np.argmax(sp))
            print(f"{algo:10s} {arm+'@'+tgt:14s} {sp.max():9.2f} "
                  f"{np.median(sp):9.2f} {int((sp>1.5).sum()):7d}  {cells[i]['network']}")
    print()
    print("speedup where the cell is quality-preserving (dQ_vs_matched >= -0.005):")
    for algo in ALGOS:
        cells = [r for r in rows if r["algo"] == algo
                 and f(r["dQ_vs_matched"]) >= -0.005]
        if not cells:
            print(f"  {algo:10s} no quality-preserving cell")
            continue
        sp = np.array([f(r["speedup_vs_single"]) for r in cells])
        print(f"  {algo:10s} n={len(cells)} max speedup {sp.max():.2f}x "
              f"(median {np.median(sp):.2f}x)  "
              f"[{', '.join(sorted({c['network'] for c in cells}))}]")
    print()
    print("algo-only speedup (T_algo_orig / T_algo_sparse), all cells:")
    for algo in ALGOS:
        cells = [r for r in rows if r["algo"] == algo]
        sp = np.array([f(r["speedup_algo_only"]) for r in cells])
        print(f"  {algo:10s} median {np.median(sp):.2f}x  max {sp.max():.2f}x  "
              f"min {sp.min():.2f}x")

    # ------------------------------------------------------------------
    rec = load(HERE / "recovery.csv")
    if not rec:
        print("\n(no recovery.csv yet)")
        return
    print()
    print("=" * 100)
    print("RECOVERY (measurement 2) — k reported everywhere; |dk|/k<25% comparability rule")
    print("=" * 100)
    for ds in ["email-Eu-core", "com-DBLP", "com-Amazon"]:
        sub = [r for r in rec if r["dataset"] == ds]
        if not sub:
            continue
        metric = "AMI" if ds == "email-Eu-core" else "avgF1_ge3"
        kcol = "k" if ds == "email-Eu-core" else "k_ge3"
        print(f"\n--- {ds} ({metric}) ---")
        for algo in ALGOS:
            s2 = [r for r in sub if r["algo"] == algo]
            if not s2:
                continue
            byc = defaultdict(list)
            for r in s2:
                byc[r["condition"]].append(r)
            base = byc.get("baseline", [])
            bk = np.mean([f(r[kcol]) for r in base]) if base else float("nan")
            bm = np.mean([f(r[metric]) for r in base]) if base else float("nan")
            print(f"  [{algo}]")
            order = ([c for c in byc if not c.startswith("chance")]
                     + [c for c in byc if c.startswith("chance")])
            for cond in order:
                v = np.array([f(r[metric]) for r in byc[cond]])
                k = np.array([f(r[kcol]) for r in byc[cond]])
                dk = abs(k.mean() - bk) / bk if bk == bk and bk else float("nan")
                comp = "k-comparable" if dk < 0.25 else f"k-MISMATCH({dk:.0%})"
                extra = ""
                if not cond.startswith("chance") and cond != "baseline":
                    ch = byc.get(f"chance_{cond}", [])
                    if ch:
                        cv = np.mean([f(r[metric]) for r in ch])
                        extra = f" above-chance {v.mean()-cv:+.4f}"
                    extra += f"  d{metric}_vs_base {v.mean()-bm:+.4f}"
                print(f"    {cond:26s} n={len(v):2d} k={k.mean():9.1f} "
                      f"{metric}={v.mean():.4f}+-{v.std():.4f}  {comp}{extra}")


if __name__ == "__main__":
    sys.exit(main())
