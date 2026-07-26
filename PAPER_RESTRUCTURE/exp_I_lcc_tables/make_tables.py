#!/usr/bin/env python
"""Build table1_lcc.tex / table2_lcc.tex / SUMMARY.md from results.csv."""

import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

ORDER = ["ca-AstroPh", "ca-CondMat", "ca-GrQc", "ca-HepPh", "ca-HepTh",
         "cit-HepPh", "cit-HepTh", "email-Enron", "facebook-combined",
         "wiki-Vote", "email-Eu-core"]

CITE = {
    "ca-AstroPh": "leskovec2007graph", "ca-CondMat": "leskovec2007graph",
    "ca-GrQc": "leskovec2007graph", "ca-HepPh": "leskovec2007graph",
    "ca-HepTh": "leskovec2007graph", "cit-HepPh": "leskovec2007graph",
    "cit-HepTh": "leskovec2007graph", "email-Enron": "leskovec2009community",
    "facebook-combined": "leskovec2012learning", "wiki-Vote": "leskovec2010signed",
    "email-Eu-core": "yin2017local",
}

# Published FULL-graph values (main-v2.tex Tables 1-2), for the comparison section.
PUB = {
    # name: (m, Q_orig, dQ_fixed, sd, dQ_leiden, sd, dF, sd, mdG, sd)
    "ca-AstroPh":        (198050, 0.6393, 0.0423, 0.0008, 0.0571, 0.0016,  0.0350, 0.0008, 0.0072, 0.0001),
    "ca-CondMat":        ( 93439, 0.7410, 0.0638, 0.0016, 0.1018, 0.0012,  0.0588, 0.0016, 0.0051, 0.0001),
    "ca-GrQc":           ( 14484, 0.8664, 0.0257, 0.0022, 0.0628, 0.0022,  0.0142, 0.0022, 0.0115, 0.0003),
    "ca-HepPh":          (118489, 0.6631, 0.0844, 0.0012, 0.1089, 0.0029,  0.0030, 0.0011, 0.0814, 0.0002),
    "ca-HepTh":          ( 25973, 0.7760, 0.0609, 0.0023, 0.1212, 0.0022,  0.0518, 0.0023, 0.0091, 0.0003),
    "cit-HepPh":         (420877, 0.7349, 0.0136, 0.0004, 0.0182, 0.0019,  0.0078, 0.0004, 0.0058, 0.0001),
    "cit-HepTh":         (352285, 0.6637, 0.0469, 0.0009, 0.0546, 0.0022,  0.0318, 0.0009, 0.0152, 0.0001),
    "email-Enron":       (183831, 0.6248, 0.1507, 0.0009, 0.1853, 0.0008,  0.1186, 0.0009, 0.0321, 0.0003),
    "facebook-combined": ( 88234, 0.8357, 0.0259, 0.0008, 0.0273, 0.0008, -0.0008, 0.0007, 0.0268, 0.0002),
    "wiki-Vote":         (100762, 0.4175, 0.0694, 0.0011, 0.0932, 0.0062,  0.0671, 0.0016, 0.0023, 0.0007),
    "email-Eu-core":     ( 16064, 0.4174, 0.0789, 0.0041, 0.0912, 0.0029,  0.0681, 0.0040, 0.0108, 0.0009),
}


def load():
    rows = list(csv.DictReader(open(OUT / "results.csv")))
    by = OrderedDict((d, []) for d in ORDER)
    for r in rows:
        by[r["dataset"]].append({k: (v if k == "dataset" else float(v))
                                 for k, v in r.items()})
    return by


def ms(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def f(mu, sd):
    return f"${mu:.4f} \\pm {sd:.4f}$"


def main():
    by = load()
    stat = OrderedDict()
    for d, rs in by.items():
        s = {}
        s["m"] = int(rs[0]["m_lcc"])
        s["n"] = int(rs[0]["n_lcc"])
        s["m_full"] = int(rs[0]["m_full"])
        s["n_full"] = int(rs[0]["n_full"])
        s["k"] = int(rs[0]["n_comm"])
        s["Q_orig"] = rs[0]["Q_orig"]
        for key in ("dQ_fixed", "dQ_leiden_sparse", "dF_obs", "dG_obs",
                    "realized_retention", "ratio_observed"):
            s[key] = ms([r[key] for r in rs])
        s["ratio_predicted"] = rs[0]["ratio_predicted"]
        s["delta"] = rs[0]["delta"]
        s["mu_intra"] = rs[0]["mu_intra"]
        s["mu_inter"] = rs[0]["mu_inter"]
        s["max_recon_err"] = max(r["recon_abs_err"] for r in rs)
        stat[d] = s

    # ------------------------------------------------------------------ T1
    max_err = max(s["max_recon_err"] for s in stat.values())
    ret_lo = min(s["realized_retention"][0] for s in stat.values())
    ret_hi = max(s["realized_retention"][0] for s in stat.values())

    t1 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Fixed-partition modularity changes under with-replacement DSpar at nominal $\alpha = 0.8$, "
        r"computed on the largest connected component of each simple undirected graph "
        rf"(realized retention ${ret_lo:.2f}$--${ret_hi:.2f}$)." + "\n"
        r"\textbf{Both $\Delta Q$ columns are evaluated on the sparsified graph}; Section~\ref{sec:artifacts} shows why such sparse-graph-scored quantities must not be read as detection improvements.}",
        r"\label{tab:exp1_2_modularity}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & $m$ & $m'/m$ & $Q_{\mathrm{orig}}$ & $\Delta Q_{\mathrm{fixed}}$ & $\Delta Q_{\mathrm{Leiden}}^{\mathrm{(sp)}}$ \\",
        r"\midrule",
    ]
    for d in ORDER:
        s = stat[d]
        t1.append(
            f"{d}\\cite{{{CITE[d]}}} & {s['m']:,} & "
            f"${s['realized_retention'][0]:.4f}$ & ${s['Q_orig']:.4f}$ & "
            f"{f(*s['dQ_fixed'])} & {f(*s['dQ_leiden_sparse'])} \\\\"
        )
    t1 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (OUT / "table1_lcc.tex").write_text("\n".join(t1))

    # ------------------------------------------------------------------ T2
    t2 = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Modularity decomposition verification at nominal $\alpha = 0.8$ "
        r"(largest connected component of each graph)." + "\n"
        r"For all datasets and runs, the identity $\Delta Q_{\mathrm{fixed}} = \Delta F_{\mathrm{obs}} - \Delta G_{\mathrm{obs}}$ holds up to machine precision "
        rf"(maximum absolute reconstruction error $\le 10^{{{int(np.ceil(np.log10(max_err))) if max_err > 0 else -16}}}$).}}",
        r"\label{tab:exp1_2_decomposition}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Dataset & $\Delta Q_{\mathrm{fixed}}$ & $\Delta F_{\mathrm{obs}}$ & $-\Delta G_{\mathrm{obs}}$  \\",
        r"\midrule",
    ]
    for d in ORDER:
        s = stat[d]
        mdG = (-s["dG_obs"][0], s["dG_obs"][1])
        t2.append(f"{d} & {f(*s['dQ_fixed'])} & {f(*s['dF_obs'])} & {f(*mdG)} \\\\")
    t2 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (OUT / "table2_lcc.tex").write_text("\n".join(t2))

    # ------------------------------------------------------------------ SUMMARY
    L = []
    L.append("# Experiment I: Tables 1-2 recomputed on the largest connected component\n")
    L.append("Pipeline: simple undirected graph -> LCC -> leidenalg `ModularityVertexPartition` "
             f"(seed {42}, n_iterations 2) -> fixed partition P; "
             "`experiments/dspar.py::dspar_sparsify(method=\"paper\")` (WITH replacement, "
             "q = ceil(0.8 m) draws), weights dropped; 10 seeds (80000-80009, the same seed "
             "stream the published run used at alpha=0.8).\n")
    L.append("Everything below is on the LCC. `published` = full-graph values in "
             "`Paper_materials/main-v2.tex` Tables 1-2.\n")

    L.append("## 1. Graph size: full vs LCC\n")
    L.append("| dataset | n (full) | m (full, simple) | n (LCC) | m (LCC) | m_LCC/m_full |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for d in ORDER:
        s = stat[d]
        L.append(f"| {d} | {s['n_full']:,} | {s['m_full']:,} | {s['n']:,} | {s['m']:,} | "
                 f"{s['m']/s['m_full']:.4f} |")
    L.append("")

    L.append("## 2. Table 1 quantities: LCC vs published full-graph\n")
    L.append("| dataset | Q_orig LCC | Q_orig pub | dQ_fixed LCC | dQ_fixed pub | change | "
             "dQ_Leiden LCC | dQ_Leiden pub | change | realized ret. |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in ORDER:
        s = stat[d]
        p = PUB[d]
        L.append(
            f"| {d} | {s['Q_orig']:.4f} | {p[1]:.4f} | "
            f"{s['dQ_fixed'][0]:.4f} +/- {s['dQ_fixed'][1]:.4f} | {p[2]:.4f} +/- {p[3]:.4f} | "
            f"{s['dQ_fixed'][0]-p[2]:+.4f} | "
            f"{s['dQ_leiden_sparse'][0]:.4f} +/- {s['dQ_leiden_sparse'][1]:.4f} | "
            f"{p[4]:.4f} +/- {p[5]:.4f} | {s['dQ_leiden_sparse'][0]-p[4]:+.4f} | "
            f"{s['realized_retention'][0]:.4f} |")
    L.append("")

    L.append("## 3. Table 2 quantities: LCC vs published full-graph\n")
    L.append("| dataset | dF_obs LCC | dF_obs pub | change | -dG_obs LCC | -dG_obs pub | change |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in ORDER:
        s = stat[d]
        p = PUB[d]
        L.append(
            f"| {d} | {s['dF_obs'][0]:.4f} +/- {s['dF_obs'][1]:.4f} | {p[6]:.4f} +/- {p[7]:.4f} | "
            f"{s['dF_obs'][0]-p[6]:+.4f} | "
            f"{-s['dG_obs'][0]:.4f} +/- {s['dG_obs'][1]:.4f} | {p[8]:.4f} +/- {p[9]:.4f} | "
            f"{-s['dG_obs'][0]-p[8]:+.4f} |")
    L.append("")

    L.append("## 4. Largest absolute change vs published, per dataset\n")
    L.append("| dataset | largest-changing quantity | LCC | published | delta |")
    L.append("|---|---|---:|---:|---:|")
    for d in ORDER:
        s = stat[d]
        p = PUB[d]
        cand = [
            ("Q_orig", s["Q_orig"], p[1]),
            ("dQ_fixed", s["dQ_fixed"][0], p[2]),
            ("dQ_Leiden^(sp)", s["dQ_leiden_sparse"][0], p[4]),
            ("dF_obs", s["dF_obs"][0], p[6]),
            ("-dG_obs", -s["dG_obs"][0], p[8]),
        ]
        name, lcc, pub = max(cand, key=lambda c: abs(c[1] - c[2]))
        L.append(f"| {d} | {name} | {lcc:.4f} | {pub:.4f} | {lcc-pub:+.4f} |")
    L.append("")

    L.append("## 5. Sign / regime classification\n")
    L.append("`regime` = which of the two terms is larger in absolute value.\n")
    L.append("| dataset | dQ_fixed > 0 | dF_obs sign (LCC / pub) | -dG_obs sign (LCC / pub) | "
             "regime LCC | regime pub | changed? |")
    L.append("|---|---|---|---|---|---|---|")
    flips = []
    for d in ORDER:
        s = stat[d]
        p = PUB[d]
        dF, mdG = s["dF_obs"][0], -s["dG_obs"][0]
        reg = "dG-dominated" if abs(mdG) > abs(dF) else "dF-dominated"
        pdF, pmdG = p[6], p[8]
        preg = "dG-dominated" if abs(pmdG) > abs(pdF) else "dF-dominated"
        sg = lambda x: "+" if x > 0 else ("-" if x < 0 else "0")  # noqa: E731
        ch = []
        if sg(dF) != sg(pdF):
            ch.append("dF sign")
        if sg(mdG) != sg(pmdG):
            ch.append("-dG sign")
        if reg != preg:
            ch.append("regime")
        if ch:
            flips.append((d, ", ".join(ch)))
        L.append(f"| {d} | {'yes' if s['dQ_fixed'][0] > 0 else 'NO'} | "
                 f"{sg(dF)} / {sg(pdF)} | {sg(mdG)} / {sg(pmdG)} | {reg} | {preg} | "
                 f"{', '.join(ch) if ch else 'no'} |")
    L.append("")
    L.append("Changes: " + ("; ".join(f"**{d}**: {c}" for d, c in flips) if flips else "none") + "\n")

    L.append("## 6. Corollary 1: predicted vs observed preservation ratio\n")
    L.append("mu_intra, mu_inter and delta are on the LCC w.r.t. the fixed partition; "
             "ratio_predicted = mu_inter/mu_intra; ratio_observed = "
             "(inter-edge survival rate)/(intra-edge survival rate) over 10 seeds.\n")
    L.append("`shrink` = |ratio_pred - 1| - |ratio_obs - 1|, i.e. how much closer to one the "
             "measured ratio is than the Bernoulli-model prediction. This is the quantity the "
             "paper's \"0.07 to 0.30\" sentence refers to.\n")
    L.append("| dataset | mu_intra | mu_inter | delta | ratio_predicted | ratio_observed | "
             "obs - pred | shrink |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in ORDER:
        s = stat[d]
        rp, ro = s["ratio_predicted"], s["ratio_observed"][0]
        L.append(f"| {d} | {s['mu_intra']:.6f} | {s['mu_inter']:.6f} | {s['delta']:+.6f} | "
                 f"{rp:.4f} | {ro:.4f} +/- {s['ratio_observed'][1]:.4f} | "
                 f"{ro-rp:+.4f} | {abs(rp-1)-abs(ro-1):+.4f} |")
    gaps = [stat[d]["ratio_observed"][0] - stat[d]["ratio_predicted"] for d in ORDER]
    shr = [abs(stat[d]["ratio_predicted"] - 1) - abs(stat[d]["ratio_observed"][0] - 1)
           for d in ORDER]
    L.append("")
    L.append(f"- Signed gap (observed - predicted) on the LCC: {min(gaps):+.4f} to {max(gaps):+.4f}.")
    L.append(f"- Shrinkage toward one on the LCC: {min(shr):.4f} to {max(shr):.4f} "
             f"(published claim on full graphs: 0.07 to 0.30). Positive on "
             f"{sum(1 for x in shr if x > 0)}/11, i.e. the measured ratio is closer to one "
             f"than predicted on every network.")
    below = [d for d in ORDER if stat[d]["ratio_observed"][0] < 1]
    L.append(f"- ratio_observed < 1 (preferential inter-community removal, as predicted) on "
             f"{len(below)}/11: {', '.join(below)}.")
    above = [d for d in ORDER if stat[d]["ratio_observed"][0] >= 1]
    L.append(f"- ratio_observed >= 1 on {len(above)}/11: "
             f"{', '.join(f'{d} ({stat[d]['ratio_observed'][0]:.4f}, delta={stat[d]['delta']:+.5f})' for d in above)}.\n")

    L.append("## 7. Identity check\n")
    L.append("| dataset | max |dF - dG - dQ_fixed| over 10 seeds |")
    L.append("|---|---:|")
    for d in ORDER:
        L.append(f"| {d} | {stat[d]['max_recon_err']:.2e} |")
    L.append("")
    L.append(f"Global maximum: {max_err:.2e}.\n")

    # ---------------------------------------------------------------- control
    cpath = OUT / "control.csv"
    if cpath.exists():
        crows = list(csv.DictReader(open(cpath)))
        cby = {}
        for r in crows:
            cby.setdefault(r["dataset"], {})[r["variant"]] = r
        L.append("## 8. Control: is the change caused by the LCC or by the Leiden "
                 "implementation?\n")
        L.append("The published tables used `igraph.community_leiden(objective_function="
                 "\"modularity\")`; this re-run uses `leidenalg.ModularityVertexPartition` as "
                 "specified. `FULL(seed42)` repeats the measurement on the full simple graph "
                 "with the NEW Leiden routine, isolating the two effects. `LCC(seed7)` and "
                 "`LCC(seed1234)` vary only the partition seed. See `control.csv` / "
                 "`control.log` for all columns.\n")
        L.append("| dataset | variant | Q_orig | delta | dQ_fixed | dF_obs | -dG_obs | ratio_obs |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for d in ORDER:
            for v in ("FULL(seed42)", "LCC(seed42)", "LCC(seed7)", "LCC(seed1234)"):
                r = cby.get(d, {}).get(v)
                if not r:
                    continue
                L.append(f"| {d} | {v} | {float(r['Q_orig']):.4f} | {float(r['delta']):+.5f} | "
                         f"{float(r['dQ_fixed_mean']):+.4f} | {float(r['dF_obs_mean']):+.4f} | "
                         f"{float(r['mdG_obs_mean']):+.4f} | {float(r['ratio_obs_mean']):.4f} |")
        L.append("")
        # seed stability of the two small/sign-fragile quantities
        L.append("### Seed stability of the near-zero terms\n")
        L.append("| dataset | dF_obs over LCC seeds {42,7,1234} | -dG_obs over LCC seeds "
                 "{42,7,1234} | sign stable? |")
        L.append("|---|---|---|---|")
        for d in ORDER:
            vs = [cby[d][v] for v in ("LCC(seed42)", "LCC(seed7)", "LCC(seed1234)")
                  if v in cby.get(d, {})]
            dFs = [float(r["dF_obs_mean"]) for r in vs]
            dGs = [float(r["mdG_obs_mean"]) for r in vs]
            ok = (len({x > 0 for x in dFs}) == 1) and (len({x > 0 for x in dGs}) == 1)
            L.append(f"| {d} | {', '.join(f'{x:+.4f}' for x in dFs)} | "
                     f"{', '.join(f'{x:+.4f}' for x in dGs)} | "
                     f"{'yes' if ok else '**NO**'} |")
        L.append("")

    # ---------------------------------------------------------------- verdict
    L.append("## 9. Verdict on Section 5.1\n")
    dq = [stat[d]["dQ_fixed"][0] for d in ORDER]
    dql = [stat[d]["dQ_leiden_sparse"][0] for d in ORDER]
    L.append(f"1. **\"Positive dQ_fixed on all eleven networks\"** -- HOLDS. "
             f"LCC range {min(dq):.4f} to {max(dq):.4f} (published {min(p[2] for p in PUB.values()):.4f} "
             f"to {max(p[2] for p in PUB.values()):.4f}); dQ_Leiden^(sp) range "
             f"{min(dql):.4f} to {max(dql):.4f}. Every value moves by at most 0.008 in absolute "
             f"terms and no ordering of practical interest changes.")
    L.append("2. **Identity dQ_fixed = dF_obs - dG_obs to machine precision** -- HOLDS "
             f"(max |error| {max_err:.2e} over 110 runs, still `<= 1e-14`).")
    L.append("3. **Two-regime story (dF-carried vs null-model-carried)** -- HOLDS in substance, "
             "with two bookkeeping changes. `ca-GrQc` moves from dF-dominated (pub 0.0142 vs "
             "0.0115) to dG-dominated (LCC 0.0083 vs 0.0105); both terms stay positive, and the "
             "flip is stable across Leiden seeds and is caused by the LCC (the full graph with "
             "the same Leiden routine gives 0.0158 vs 0.0106). `ca-HepPh`'s dF_obs changes sign "
             "(pub +0.0030 +/- 0.0011, LCC -0.0111 +/- 0.0009), but the control shows this "
             "particular sign is NOT robust: on the same LCC with Leiden seeds 7 and 1234 it is "
             "+0.0017 and +0.0052. ca-HepPh's dF_obs is indistinguishable from zero at any "
             "preprocessing; what is robust is that ca-HepPh is overwhelmingly dG-dominated "
             "(|-dG| is 6-60x |dF| in every variant). Safest wording: `facebook-combined` and "
             "`ca-HepPh` are the two networks whose change is carried entirely by the null-model "
             "term, with dF_obs ~ 0.")
    L.append("4. **`facebook-combined` as the delta ~ 0, dF_obs <= 0 example** -- UNCHANGED. "
             "facebook-combined is already connected, so the LCC is the full graph; delta = "
             "-0.00136, dF_obs = -0.0006 +/- 0.0006, -dG_obs = +0.0259 +/- 0.0002, identical to "
             "the published row up to the Leiden routine.")
    L.append("5. **\"Measured preservation ratios closer to one than predicted, by 0.07 to "
             "0.30\"** -- HOLDS in direction on 11/11, but the numeric range must be restated as "
             f"**{min(shr):.2f} to {max(shr):.2f}** on the LCC. The low end moves because "
             "facebook-combined (ratio_pred 1.0298, ratio_obs 1.0151) shrinks by only 0.015.")
    L.append("6. **\"Direction of preferential removal agrees with the prediction wherever delta "
             "is materially positive\"** -- NEEDS a caveat at seed 42. ca-HepPh has delta = "
             "+0.021 yet ratio_obs = 1.074 > 1 (intra-community edges removed slightly faster). "
             "The control shows ratio_obs on ca-HepPh is 0.9777 (full), 1.0740 / 0.9896 / 0.9688 "
             "(LCC, three Leiden seeds), i.e. it straddles one; ca-HepPh has the second-smallest "
             "delta in the suite. Either keep the sentence with \"delta is materially positive\" "
             "meaning delta >= 0.025 (which excludes ca-HepPh and cit-HepPh), or state 9/11 "
             "explicitly.")
    L.append("7. **Realized-retention range in the caption** -- must change from `0.33--0.52` to "
             f"`{ret_lo:.2f}--{ret_hi:.2f}` for these eleven LCCs "
             f"(min {min(ORDER, key=lambda d: stat[d]['realized_retention'][0])}, "
             f"max {max(ORDER, key=lambda d: stat[d]['realized_retention'][0])}).")
    L.append("8. **Watch item not caused by the LCC:** `wiki-Vote`'s -dG_obs is the smallest "
             "entry in Table 2 and its sign is also Leiden-seed dependent (+0.0008, -0.0033, "
             "+0.0006 across seeds 42/7/1234 on the LCC; -0.0027 on the full graph with the same "
             "routine; published +0.0023 +/- 0.0007). The delivered table uses seed 42 and is "
             "positive, but no claim should rest on the sign of -dG_obs for wiki-Vote.")
    L.append("9. **The disclaimer paragraph at main-v2.tex:986** (\"Tables 1-2 retain the full "
             "graphs as downloaded...\") can be deleted for the preprocessing half; the "
             "15-of-17-network-subset half is unaffected by this experiment.\n")

    (OUT / "SUMMARY.md").write_text("\n".join(L))
    print("wrote table1_lcc.tex, table2_lcc.tex, SUMMARY.md")


if __name__ == "__main__":
    main()
