#!/usr/bin/env python3
"""Build SUMMARY.md from results.csv for Experiment B (configuration-model null)."""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = [
    "email-Eu-core", "wiki-Vote", "ca-GrQc", "ca-HepTh", "facebook-combined",
    "ca-CondMat", "ca-HepPh", "ca-AstroPh", "email-Enron", "cit-HepTh",
    "cit-HepPh", "com-Amazon", "com-DBLP", "com-Youtube", "wiki-Talk",
    "wiki-topcats", "cit-Patents",
]


def main():
    rows = {}
    with open(HERE / "results.csv") as fh:
        for r in csv.DictReader(fh):
            rows[(r["network"], r["arm"])] = r

    done = [n for n in ORDER if (n, "real") in rows and (n, "null") in rows]
    skipped = [n for n in ORDER if n not in done]

    lines = []
    A = lines.append
    A("# Experiment B — configuration-model (degree-preserving rewire) null\n")
    A("**Question.** The draft reads DSpar separation `delta > 0` and fixed-partition")
    A("modularity gain `dQ_fixed > 0` as evidence that DSpar clarifies community")
    A("structure. A degree-preserving rewiring keeps the degree sequence exactly and")
    A("destroys community structure. If the rewired null reproduces both signals, then")
    A("neither is evidence about community structure.\n")
    A("**Setup.** Undirected simple graph, largest connected component. `delta = mu_intra")
    A("- mu_inter` on DSpar scores `s(e) = 1/d_u + 1/d_v`; `hb = E[d_u d_v | inter] /")
    A("E[d_u d_v | intra]`; `Q_fixed` = igraph modularity of the FIXED Leiden partition.")
    A("Sparsifier: `experiments/dspar.py` `method=\"paper\"`, `retention=0.8`, weights")
    A("dropped. REAL arm: 1 Leiden partition, 3 sparsification seeds. NULL arm: 2 rewire")
    A("seeds (igraph simple double-edge swaps, `n_swaps = 10m`), fresh Leiden partition")
    A("per rewire, 2 sparsification seeds each (4 reps). `+-` is the std over reps.\n")
    A("`actual_retention` is `m_sparse / m`: the 'paper' DSpar method samples *with")
    A("replacement*, so the unique-edge retention is far below the nominal 0.8.\n")

    A("## Side-by-side table\n")
    A("| network | arm | n | m | Q_fixed_base | delta | hb | dQ_fixed | actual_retention |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    reduced = []
    for net in done:
        for arm in ("real", "null"):
            r = rows[(net, arm)]
            if arm == "null" and int(r["n_reps"]) < 4 and net not in reduced:
                reduced.append(net)
            q, qs = float(r["Q_fixed_base"]), float(r["Q_fixed_base_std"])
            d, ds = float(r["delta"]), float(r["delta_std"])
            h, hs = float(r["hb"]), float(r["hb_std"])
            dq, dqs = float(r["dQ_fixed"]), float(r["dQ_fixed_std"])
            rt, rts = float(r["actual_retention"]), float(r["actual_retention_std"])
            qtxt = f"{q:.4f}" if arm == "real" else f"{q:.4f}±{qs:.4f}"
            dtxt = f"{d:+.4f}" if arm == "real" else f"{d:+.4f}±{ds:.4f}"
            htxt = f"{h:.3f}" if arm == "real" else f"{h:.3f}±{hs:.3f}"
            mark = "*" if (arm == "null" and int(r["n_reps"]) < 4) else ""
            A(f"| {net} | {arm}{mark} | {r['n']} | {r['m']} | {qtxt} | {dtxt} | {htxt} | "
              f"{dq:+.4f}±{dqs:.4f} | {rt:.4f}±{rts:.4f} |")
    A("")
    if reduced:
        A(f"`*` reduced null replication (1 rewire seed x 2 sparsification seeds instead")
        A(f"of 2 x 2) — {', '.join(reduced)} are too large for the full budget.\n")

    A("## Null / real ratios\n")
    A("| network | delta_real | delta_null | dQ_real | dQ_null | dQ_null/dQ_real | "
      "delta_null/delta_real |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    ratios = []
    for net in done:
        rr, nn = rows[(net, "real")], rows[(net, "null")]
        dr, dn = float(rr["delta"]), float(nn["delta"])
        qr, qn = float(rr["dQ_fixed"]), float(nn["dQ_fixed"])
        rq = qn / qr if qr != 0 else float("nan")
        rd = dn / dr if dr != 0 else float("nan")
        ratios.append((net, dr, dn, qr, qn, rq, rd))
        rqtxt = f"{rq:.2f}" if abs(rq) < 100 else f"{rq:.3g}"
        A(f"| {net} | {dr:+.6f} | {dn:+.6f} | {qr:+.6f} | {qn:+.6f} | {rqtxt} | "
          f"{rd:.2f} |")
    A("")
    A("Degenerate ratios: com-Amazon's real `dQ_fixed` is 1.5e-5 (indistinguishable from")
    A("zero), so its ratio is not meaningful beyond 'the null gains and the real graph")
    A("does not'. facebook-combined is the one network with `delta(real) < 0`, which")
    A("flips the sign of its delta ratio; the null there is still `delta > 0`.\n")

    n_null_delta_pos = sum(1 for _, _, dn, _, _, _, _ in ratios if dn > 0)
    n_null_dq_pos = sum(1 for _, _, _, _, qn, _, _ in ratios if qn > 0)
    n_null_ge_real_dq = sum(1 for _, _, _, qr, qn, _, _ in ratios if qn >= qr)
    n_null_ge_real_d = sum(1 for _, dr, dn, _, _, _, _ in ratios if dn >= dr)
    finite = [rq for _, _, _, _, _, rq, _ in ratios if rq == rq and abs(rq) < 100]
    med = sorted(finite)[len(finite) // 2] if finite else float("nan")
    n_delta_real_pos = sum(1 for _, dr, _, _, _, _, _ in ratios if dr > 0)

    A("## Verdict\n")
    A(f"Across all {len(done)} networks the degree-preserving rewired null reproduces "
      f"both signals: `delta > 0` in {n_null_delta_pos}/{len(done)} nulls and "
      f"`dQ_fixed > 0` in {n_null_dq_pos}/{len(done)} nulls, on graphs that have no "
      f"community structure at all (null `Q_fixed_base` collapses to the value a "
      f"modularity maximiser extracts from pure degree noise). The null is not merely "
      f"positive but typically *larger* than the real graph: `dQ_fixed(null) >= "
      f"dQ_fixed(real)` in {n_null_ge_real_dq}/{len(done)} networks and `delta(null) >= "
      f"delta(real)` in {n_null_ge_real_d}/{len(done)}, with a median ratio "
      f"`dQ_fixed(null)/dQ_fixed(real)` of {med:.2f}. Both quantities are therefore "
      f"explained by degree heterogeneity plus the fact that DSpar preferentially keeps "
      f"edges between low-degree nodes — a fixed partition always gains modularity when "
      f"a degree-biased sampler thins hub-incident edges, whether or not the partition "
      f"means anything. Neither `delta > 0` nor `dQ_fixed > 0` can be cited as evidence "
      f"that DSpar clarifies community structure; any such claim needs a statistic that "
      f"separates the real graph from its own configuration model.")
    A("")
    if skipped:
        A(f"**Skipped (exceeded the wall-clock / memory budget):** {', '.join(skipped)}.")
    else:
        A("**Skipped:** none — all 17 networks completed both arms.")
    A("")
    A("**Budget note.** Every network except the two largest ran both arms well inside")
    A("15 min (the whole 13-network batch up to com-DBLP took 5 min; com-Youtube 6.5 min,")
    A("wiki-Talk 11 min). cit-Patents (16.5M edges) and wiki-topcats (25.4M edges)")
    A("exceeded 15 min on the null arm with 2 rewire replicates, so their nulls were")
    A("re-run with 1 rewire seed x 2 sparsification seeds (1120 s and 989 s respectively).")
    A("")
    A("Per-arm wall times (s):")
    A("")
    A("| network | real | null |")
    A("|---|---:|---:|")
    for net in done:
        A(f"| {net} | {rows[(net,'real')]['seconds']} | {rows[(net,'null')]['seconds']} |")
    A("")
    A("## Reproduce\n")
    A("```")
    A("python run.py --network <name>            # appends real+null rows to results.csv")
    A("python run.py --network <name> --validate # cross-check vs experiments/dspar.py")
    A("python make_summary.py")
    A("```")
    A("")
    A("`--validate` reproduces the repo's networkx `dspar_sparsify(method=\"paper\")`")
    A("bit-for-bit (identical retention and dQ_fixed to 6 decimals on email-Eu-core,")
    A("ca-GrQc and facebook-combined); `run.py` uses a vectorised equivalent so the")
    A("large networks fit in memory.")

    (HERE / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"wrote SUMMARY.md: {len(done)} networks, skipped {skipped}")


if __name__ == "__main__":
    main()
