#!/usr/bin/env python3
"""Generate publication-ready booktabs LaTeX tables for the restructured paper."""
import csv, os, statistics as st
from collections import defaultdict, OrderedDict

ROOT = "/home/md724/community_detection_spectral"
OUT = os.path.join(ROOT, "PAPER_RESTRUCTURE/phase2_writing/tables")
os.makedirs(OUT, exist_ok=True)


def rd(p):
    with open(os.path.join(ROOT, p)) as f:
        return list(csv.DictReader(f))


def esc(s):
    return s.replace("_", "\\_")


def f(x, nd, sign=False):
    v = float(x)
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "$-$")


def pm(mu, sd, nd, sign=False):
    return f"{f(mu, nd, sign)} $\\pm$ {float(sd):.{nd}f}"


def com(x):
    return f"{int(float(x)):,}"


def write(name, body):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(body)
    print("wrote", name)


# ----------------------------------------------------------------- table 1
b = rd("PAPER_RESTRUCTURE/exp_B_config_null/results.csv")
by = defaultdict(dict)
order = []
for r in b:
    by[r["network"]][r["arm"]] = r
    if r["network"] not in order:
        order.append(r["network"])

rows = []
for net in order:
    re_, nu = by[net]["real"], by[net]["null"]
    d_re = f(re_["delta"], 3, sign=True)
    if net == "facebook-combined":
        d_re = "\\textbf{" + d_re + "}"
    rows.append(" & ".join([
        esc(net), com(re_["m"]),
        f(re_["Q_fixed_base"], 4), f(nu["Q_fixed_base"], 4),
        d_re, f(nu["delta"], 3, sign=True),
        f(re_["hb"], 3), f(nu["hb"], 3),
        pm(re_["dQ_fixed"], re_["dQ_fixed_std"], 4, sign=True),
        pm(nu["dQ_fixed"], nu["dQ_fixed_std"], 4, sign=True),
        f(re_["actual_retention"], 3), f(nu["actual_retention"], 3),
    ]) + " \\\\")

t1 = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{4pt}
\caption{Configuration-model null control. For each network the degree sequence is preserved
exactly by double-edge swaps ($10m$ swaps), which destroys community structure; the sparsifier
(DSpar, \texttt{method="paper"}, nominal $\alpha=0.8$) and the fixed-partition decomposition are
then applied identically to both arms. The rewired null reproduces \emph{both} signals the draft
read as evidence of community structure: $\delta>0$ in 17/17 nulls and $\Delta Q_{\mathrm{fixed}}>0$
in 17/17 nulls, with $\Delta Q_{\mathrm{fixed}}(\mathrm{null})\ge\Delta Q_{\mathrm{fixed}}(\mathrm{real})$
in 15/17 networks (median ratio $1.24$) and $\delta(\mathrm{null})\ge\delta(\mathrm{real})$ in 16/17.
facebook-combined is the only network with $\delta<0$ on the real graph (bold), yet its null is still
$\delta>0$. Realized retention is $m'/m$: the with-replacement \texttt{paper} sampler retains far
fewer distinct edges than the nominal $\alpha$ (see Table~\ref{tab:sampler_audit}).
Real arm: 1 Leiden partition $\times$ 3 sparsification seeds; null arm: 2 rewire seeds $\times$
2 sparsification seeds (1 $\times$ 2 for cit-Patents and wiki-topcats). Standard deviations over
replicates are shown for $\Delta Q_{\mathrm{fixed}}$; for the remaining columns they are
$\le 0.002$ ($Q_{\mathrm{fixed}}^{\mathrm{base}}$), $\le 0.004$ ($\delta$), $\le 0.175$ ($\mathrm{hb}$) and
$\le 0.004$ (retention).
Source: \texttt{PAPER\_RESTRUCTURE/exp\_B\_config\_null/results.csv}.}
\label{tab:null_control}
\footnotesize
\begin{tabular}{lrrrrrrrrrrr}
\toprule
 & & \multicolumn{2}{c}{$Q_{\mathrm{fixed}}^{\mathrm{base}}$} & \multicolumn{2}{c}{$\delta$}
 & \multicolumn{2}{c}{$\mathrm{hb}$} & \multicolumn{2}{c}{$\Delta Q_{\mathrm{fixed}}$}
 & \multicolumn{2}{c}{realized ret.} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}
Network & $m$ & real & null & real & null & real & null & real & null & real & null \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
write("tab_null_control.tex", t1)

# ----------------------------------------------------------------- table 2
e4 = rd("PAPER_EXPERIMENTS/results/exp4_comprehensive/comprehensive_alpha_results.csv")
sel = [r for r in e4 if r["alpha"] in ("0.8", "0.9")]
ds = sorted({r["dataset"] for r in sel})
rows = []
for i, d in enumerate(ds):
    sub = sorted([r for r in sel if r["dataset"] == d], key=lambda r: r["alpha"])
    for j, r in enumerate(sub):
        first = "\\multirow{2}{*}{" + esc(d) + "}" if j == 0 else ""
        rows.append(" & ".join([
            first, f(r["alpha"], 1), f(r["actual_retention"], 3),
            pm(r["Q_sparse_mean"], r["Q_sparse_std"], 4),
            pm(r["Q_on_orig_mean"], r["Q_on_orig_std"], 4),
            f(r["Q_transfer_loss"], 4),
            pm(r["Q_final_mean"], r["Q_final_std"], 4),
            f(r["delta_Q_mean"], 4, sign=True),
        ]) + " \\\\")
    if i != len(ds) - 1:
        rows.append("\\addlinespace[2pt]")

t2 = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{5pt}
\caption{Artifact I --- sparse-graph self-scoring. $Q_{\mathrm{sparse}}$ is the modularity of the
partition found on the sparsified graph, \emph{scored on that sparsified graph}; $Q_{\mathrm{orig}}$
is the same partition scored on the original graph; the transfer loss is
$Q_{\mathrm{sparse}}-Q_{\mathrm{orig}}$. It is positive for all 15 datasets at both $\alpha$ values,
i.e.\ the headline ``modularity gain'' of the sparsify-then-cluster pipeline is an artifact of
comparing scores computed on two different graphs. $Q_{\mathrm{final}}$ is Leiden seeded with the
sparse-graph partition and run on the original graph, and $\Delta Q$ is that value minus the
unsparsified baseline; the honest gain is $\le 0.008$ everywhere.
Realized retention is $m'/m$ (\texttt{paper} sampler, with replacement), not the nominal $\alpha$.
The $\alpha=1.0$ rows of the source file are \emph{not} baselines --- the sampler still ran
(realized retention $0.350$--$0.589$) --- and are excluded here; the unsparsified reference is the
run's own $Q_{\mathrm{base}}$, embedded in $\Delta Q$.
5 seeds per cell; $\pm$ is the standard deviation over seeds ($\Delta Q$ and the transfer loss are
reported without one, as the source file stores no standard deviation for them).
Source: \texttt{PAPER\_EXPERIMENTS/results/exp4\_comprehensive/comprehensive\_alpha\_results.csv}.}
\label{tab:transfer_loss}
\footnotesize
\begin{tabular}{llrrrrrr}
\toprule
Dataset & $\alpha$ & realized ret. & $Q_{\mathrm{sparse}}$ & $Q_{\mathrm{orig}}$
 & transfer loss & $Q_{\mathrm{final}}$ & $\Delta Q$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
write("tab_transfer_loss.tex", t2)

# ----------------------------------------------------------------- table 3
fr = rd("PAPER_RESTRUCTURE/exp_F_gt_recheck/results.csv")
conds = ["baseline", "dspar_paper_08", "resmatch_dspar_paper_08", "dspar_cal_09"]
cname = {"baseline": "baseline (no sparsification)",
         "dspar_paper_08": "DSpar \\texttt{paper}, $\\alpha=0.8$",
         "resmatch_dspar_paper_08": "resolution-matched control",
         "dspar_cal_09": "DSpar calibrated, $\\alpha=0.9$"}
dsets = ["Karate", "Dolphins", "Football", "Polbooks", "email-Eu-core"]
rows = []
for i, d in enumerate(dsets):
    for j, c in enumerate(conds):
        sub = [r for r in fr if r["dataset"] == d and r["condition"] == c]
        assert len(sub) == 10, (d, c, len(sub))
        def agg(col, nd):
            v = [float(r[col]) for r in sub]
            return f"{st.mean(v):.{nd}f} $\\pm$ {st.pstdev(v):.{nd}f}"
        ret = st.mean([float(r["true_retention"]) for r in sub])
        gam = st.mean([float(r["resolution"]) for r in sub])
        label = d + ("$^{\\dagger}$" if d == "Dolphins" else "")
        first = "\\multirow{4}{*}{" + esc(label) + "}" if j == 0 else ""
        rows.append(" & ".join([
            first, cname[c], f"{ret:.3f}", f"{gam:.3f}",
            agg("n_clusters", 1), agg("AMI", 3), agg("ARI", 3), agg("NMI", 3),
        ]) + " \\\\")
    if i != len(dsets) - 1:
        rows.append("\\addlinespace[2pt]")

t3 = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{5pt}
\caption{Artifact II --- granularity. Ground-truth recovery on the five small benchmark graphs,
10 seeds per cell, mean $\pm$ standard deviation. \emph{baseline} is Leiden on the original graph;
\emph{DSpar \texttt{paper}} is the draft's pipeline (with-replacement sampler, nominal $\alpha=0.8$,
realized retention $0.44$--$0.55$); the \emph{resolution-matched control} applies no sparsification
at all and instead tunes the RBConfiguration resolution $\gamma$ on the original graph until the
cluster count matches the DSpar run; \emph{DSpar calibrated} uses the $\lambda$-bisection sampler
with $\mathbb{E}[\text{retention}]=0.9$ exactly. AMI is the chance-corrected primary metric; NMI is
the draft's metric and is inflated by the finer partitions sparsification produces. The only
positive cell in the draft (email-Eu-core) is reproduced and exceeded by the resolution-matched
control at 100\% of the edges ($\Delta$AMI $+0.014$ vs $+0.007$), so the score against a
granularity-matched baseline is 0/5.
Source: \texttt{PAPER\_RESTRUCTURE/exp\_F\_gt\_recheck/results.csv}.}
\label{tab:gt_recheck}
\footnotesize
\begin{tabular}{llrrrrrr}
\toprule
Dataset & Condition & true ret. & $\gamma$ & \#clusters & AMI & ARI & NMI \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
{\footnotesize\raggedright $^{\dagger}$Dolphins has no canonical ground-truth partition; the labels
used here are a surrogate two-group split obtained by Kernighan--Lin bisection, so its recovery
scores measure agreement with a heuristic cut, not with metadata.\par}
\end{table*}
"""
write("tab_gt_recheck.tex", t3)

# ----------------------------------------------------------------- table 4
eo = rd("PAPER_RESTRUCTURE/exp_E_delta_star/outcomes.csv")
rows = []
n_sig = 0
for r in eo:
    y = float(r["y"]); sd = float(r["Q_base_std"])
    sig = y > 2 * sd
    n_sig += sig
    net = esc(r["network"])
    ystr = f(y, 4, sign=True)
    if sig:
        net = "\\textbf{" + net + "}"
        ystr = "\\textbf{" + ystr + "}$^{\\ast}$"
    rows.append(" & ".join([
        net, com(r["n"]), com(r["m"]), f(r["actual_ret_mean"], 3),
        pm(r["Q_base_mean"], r["Q_base_std"], 4),
        pm(r["Q_seeded_mean"], r["Q_seeded_std"], 4),
        f(r["Q_matched_best"], 4),
        ystr, "yes" if sig else "no",
    ]) + " \\\\")
assert n_sig == 2, n_sig

t4 = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{5pt}
\caption{The boundary: runtime-matched seeded refinement, 15 networks, calibrated sampler at
$\alpha=0.9$ (realized retention $\approx 0.900$ by construction). $Q_{\mathrm{base}}$ is plain
Leiden on the original graph (5 seeds); $Q_{\mathrm{seeded}}$ is Leiden on the original graph seeded
with the sparse-graph partition (3 sparsification seeds); $Q_{\mathrm{matched}}$ is the best of the
plain-Leiden restarts affordable within the \emph{same} wall-clock budget as the full sparsify +
cluster + seed pipeline (2 restarts in all 15 cells). The honest outcome is
$y = Q_{\mathrm{seeded}} - Q_{\mathrm{matched}}$. Only 2/15 networks clear $y > 2\sigma_{\mathrm{base}}$
(starred, bold): email-Enron and com-Youtube. Genuine gains therefore exist, but they are rare and
small ($\le 0.009$).
Source: \texttt{PAPER\_RESTRUCTURE/exp\_E\_delta\_star/outcomes.csv}.}
\label{tab:boundary}
\footnotesize
\begin{tabular}{lrrrrrrrc}
\toprule
Network & $n$ & $m$ & ret. & $Q_{\mathrm{base}}$ & $Q_{\mathrm{seeded}}$ & $Q_{\mathrm{matched}}$
 & $y$ & $y>2\sigma_{\mathrm{base}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
write("tab_boundary.tex", t4)

# ----------------------------------------------------------------- table 5
co = rd("PAPER_RESTRUCTURE/exp_E_delta_star/correlations.csv")
pred_order = []
for r in co:
    if r["predictor"] not in pred_order:
        pred_order.append(r["predictor"])
pname = {
    "delta_real": r"$\delta$ (real graph)",
    "delta_star": r"$\delta^{\ast}=\delta_{\mathrm{real}}-\delta_{\mathrm{null}}$",
    "delta_ratio": r"$\delta_{\mathrm{real}}/\delta_{\mathrm{null}}$",
    "dQ_ratio": r"$\Delta Q_{\mathrm{fixed}}^{\mathrm{real}}/\Delta Q_{\mathrm{fixed}}^{\mathrm{null}}$",
    "dQ_excess": r"$\Delta Q_{\mathrm{fixed}}^{\mathrm{real}}-\Delta Q_{\mathrm{fixed}}^{\mathrm{null}}$",
    "hb_real": r"$\mathrm{hb}$ (real graph)",
    "hb_excess": r"$\mathrm{hb}_{\mathrm{real}}-\mathrm{hb}_{\mathrm{null}}$",
    "hb_ratio": r"$\mathrm{hb}_{\mathrm{real}}/\mathrm{hb}_{\mathrm{null}}$",
    "deg_cv": r"degree CV $\sigma_d/\bar d$",
    "Qfix_real": r"$Q_{\mathrm{fixed}}^{\mathrm{base}}$ (real graph)",
    "Qfix_ratio": r"$Q_{\mathrm{fixed}}^{\mathrm{base,real}}/Q_{\mathrm{fixed}}^{\mathrm{base,null}}$",
    "avg_deg": r"average degree $\bar d$",
    "n": r"$n$ (nodes)",
    "m": r"$m$ (edges)",
}
idx = {(r["predictor"], r["outcome"]): r for r in co}
rows = []
for p in pred_order:
    cells = [pname[p]]
    for out in ("y", "y2"):
        r = idx[(p, out)]
        cells += [f(r["spearman_rho"], 3, sign=True), f"{float(r['spearman_p']):.3f}",
                  f(r["pearson_r"], 3, sign=True), f"{float(r['pearson_p']):.3f}"]
    rows.append(" & ".join(cells) + " \\\\")

t5 = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{5pt}
\caption{No predictor of the genuine gain. Correlations between 14 candidate structural statistics
and the seeded-refinement outcome across the $n=15$ networks of Table~\ref{tab:boundary}.
$y = Q_{\mathrm{seeded}} - Q_{\mathrm{matched}}$ is the runtime-matched (honest) outcome;
$y_2 = Q_{\mathrm{seeded}} - Q_{\mathrm{base}}$ is the naive one. No predictor is significant for
$y$ at the $5\%$ level; the strongest is $n$ ($\rho=0.468$, $p=0.079$). Two cells reach nominal
significance against the \emph{naive} outcome only ($\mathrm{hb}$ excess, $p=0.026$; $n$, $p=0.044$), and
neither survives correction for the 28 tests reported here. $\delta^{\ast}$ --- the null-corrected
separation statistic --- is wrong-signed for both outcomes and is therefore retained as a diagnostic
only. With $n=15$, the $95\%$ CI of a single Pearson $r$ spans roughly $\pm 0.5$; see the wiki-Talk
case study in the text, where dropping one network moves $r(\mathrm{deg\ CV})$ from $0.69$ to $0.12$.
Source: \texttt{PAPER\_RESTRUCTURE/exp\_E\_delta\_star/correlations.csv}.}
\label{tab:predictors}
\footnotesize
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{$y$ (runtime-matched)} & \multicolumn{4}{c}{$y_2$ (naive)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Predictor & $\rho$ & $p$ & $r$ & $p$ & $\rho$ & $p$ & $r$ & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
write("tab_predictors.tex", t5)

# ----------------------------------------------------------------- table 6
# realized-retention ranges, assembled from the measured files
def rng(vals):
    return f"{min(vals):.3f}--{max(vals):.3f}"

paper = defaultdict(list)
for r in e4:                                    # 15 networks, alpha 0.8/0.9/1.0
    paper[r["alpha"]].append(float(r["actual_retention"]))
for r in b:                                     # 17 networks x 2 arms, nominal 0.8
    paper["0.8"].append(float(r["actual_retention"]))
fpaper = defaultdict(list)
for r in fr:
    if r["condition"] == "dspar_paper_08":
        fpaper[r["dataset"]].append(float(r["true_retention"]))
for d, v in fpaper.items():
    paper["0.8"].append(st.mean(v))

ec = rd("PAPER_RESTRUCTURE/exp_C_true_retention_seeded/results_summary.csv")
norep = defaultdict(list); cal = defaultdict(list)
for r in ec:
    (norep if r["sampler"] == "repo_noreplace" else cal)[r["alpha"]].append(float(r["actual_ret_mean"]))
for r in eo:
    cal["0.9"].append(float(r["actual_ret_mean"]))
fcal = defaultdict(list)
for r in fr:
    if r["condition"] == "dspar_cal_09":
        fcal[r["dataset"]].append(float(r["true_retention"]))
for d, v in fcal.items():
    cal["0.9"].append(st.mean(v))

n_paper08 = len(paper["0.8"])
t6_rows = [
    ("\\multirow{3}{*}{\\texttt{paper} (w/ repl.)}", "0.8",
     rng(paper["0.8"]), f"{n_paper08}", "exp4, exp\\_B (both arms), exp\\_F"),
    ("", "0.9", rng(paper["0.9"]), "15", "exp4"),
    ("", "1.0", rng(paper["1.0"]), "15", "exp4"),
    ("\\multirow{3}{*}{\\texttt{prob.\\_no\\_replace}}", "0.8",
     rng(norep["0.8"]), "6", "exp\\_C"),
    ("", "0.9", rng(norep["0.9"]), "6", "exp\\_C"),
    ("", "1.0", "0.662", "1", "audit (email-Eu-core)"),
    ("\\multirow{3}{*}{calibrated ($\\lambda$-bisection)}", "0.8", rng(cal["0.8"]), "6", "exp\\_C"),
    ("", "0.9", rng(cal["0.9"]), "26", "exp\\_C, exp\\_E, exp\\_F"),
    ("", "1.0", "--", "--", "not run"),
]
rows = []
for i, (s, a, r_, k, src) in enumerate(t6_rows):
    rows.append(" & ".join([s, a, r_, k, src]) + " \\\\")
    if i in (2, 5):
        rows.append("\\addlinespace[2pt]")

t6 = r"""\begin{table}[t]
\centering
\setlength{\tabcolsep}{4pt}
\caption{Sampler audit: nominal $\alpha$ versus realized edge retention $m'/m$, over all runs on
disk. The three samplers used in (or implied by) the pipeline are not interchangeable.
\texttt{paper} samples $\lceil \alpha m\rceil$ edges \emph{with} replacement, so distinct-edge
retention collapses to roughly $\alpha/2$ and never approaches $\alpha$ --- even at $\alpha=1.0$.
\texttt{probabilistic\_no\_replace} sets $p_e=\mathrm{clip}(s_e/\sum s\cdot\lceil\alpha m\rceil,0,1)$;
$13$--$35\%$ of edges hit the clip, so $\mathbb{E}[\text{kept}]<\alpha m$ always and the sampler
saturates below $0.7$. The calibrated sampler solves $\sum_e \min(1,\lambda s_e)=\alpha m$ by
bisection, giving $\mathbb{E}[\text{retention}]=\alpha$ exactly; it is the only variant for which the
nominal knob means what the formal definition says. ``\#'' is the number of network$\times$sampler
runs pooled into the range. Ranges are min--max of the per-network mean realized retention.
Sources: \texttt{PAPER\_RESTRUCTURE/audit/AUDIT\_FINDINGS.md},
\texttt{exp\_C/results\_summary.csv}, \texttt{exp\_B/results.csv}, \texttt{exp\_E/outcomes.csv},
\texttt{exp\_F/results.csv},
\texttt{PAPER\_EXPERIMENTS/results/exp4\_comprehensive/comprehensive\_alpha\_results.csv}.
Exp\_C additionally ran $\alpha\in\{0.7,0.95\}$ (\texttt{no\_replace}: $0.366$--$0.572$ and
$0.447$--$0.678$; calibrated: $0.699$--$0.700$ and $0.950$--$0.951$). The calibrated
$\alpha=1.0$ cell was never run, as it is the no-sparsification sentinel.}
\label{tab:sampler_audit}
\footnotesize
\begin{tabular}{llrrl}
\toprule
Sampler & nominal $\alpha$ & realized $m'/m$ & \# & source \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
write("tab_sampler_audit.tex", t6)
print("paper08 n =", n_paper08, "cal09 n =", len(cal["0.9"]))
