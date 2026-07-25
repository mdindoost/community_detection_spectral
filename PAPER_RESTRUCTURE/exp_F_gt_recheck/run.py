#!/usr/bin/env python3
"""
Experiment F: does the draft's Experiment-3 email-Eu-core "positive" ground-truth
result survive (a) chance-adjusted metrics and (b) a resolution-matched control?

Background
----------
Paper_materials/main-tnse.tex, tab:ground_truth_summary reports that DSpar at
alpha=0.8 improves ground-truth recovery on email-Eu-core (dNMI=+0.025,
dARI=+0.071) -- the only positive dataset of the 5 small ground-truth graphs.

PAPER_RESTRUCTURE/exp_D_groundtruth_scale showed at scale (com-Amazon/DBLP/Youtube)
that such gains are a GRANULARITY artifact: DSpar "paper" sampling with replacement
shreds the graph (true distinct-edge retention ~0.5 at nominal alpha=0.8), Leiden
then returns many more, much smaller clusters, and
  (i)  plain NMI is upward-biased for partitions with many clusters
       (Vinh, Epps & Bailey, JMLR 2010),
  (ii) a resolution-matched control -- Leiden on the UNSPARSIFIED graph with the
       resolution tuned to the same cluster count -- reproduces the whole effect.

This script re-runs the draft's five small ground-truth datasets under exactly
those two controls.

Datasets / ground truth (loaders mimic PAPER_EXPERIMENTS/exp2_ground_truth_recovery.py)
--------------------------------------------------------------------------------------
  Karate         nx.karate_club_graph(), gt = 'club' attribute (2 factions)
  Dolphins       data/dolphins.gml, gt = Kernighan-Lin bisection (as in exp2; NOTE:
                 this is a *surrogate* ground truth, not published labels)
  Football       data/football.gml (igraph read + simplify), gt = 'value' (conference)
  Polbooks       data/polbooks.gml, gt = 'value' in {l,n,c}
  email-Eu-core  data/email-Eu-core.txt + data/email-Eu-core-department-labels.txt

Every graph is then made undirected + simple, restricted to labelled nodes, and
reduced to its largest connected component; labels are restricted accordingly.

Conditions (10 seeds each)
--------------------------
  baseline           Leiden (leidenalg ModularityVertexPartition) on the original graph.
  dspar_paper_08     DSpar experiments/dspar.py method="paper", retention=0.8, edge
                     WEIGHTS DROPPED -- bit-for-bit the draft's Experiment-3 pipeline.
                     Leiden on the result.
  resmatch_paper_08  NO sparsification. Leiden on the ORIGINAL graph with
                     RBConfigurationVertexPartition, resolution gamma tuned by
                     bisection so the mean cluster count matches dspar_paper_08.
  dspar_cal_09       Calibrated true-retention sampler (from exp_C_true_retention_seeded):
                     same DSpar scores, independent Bernoulli, lambda solved so that
                     sum_e min(1, lambda*s_e) = 0.9*m  =>  E[retention] = 0.90 exactly.
  resmatch_cal_09    Resolution-matched control for dspar_cal_09; only run when that
                     condition's mean cluster count differs from baseline by > 20%.

Metrics (sklearn, over all LCC nodes): AMI (primary, chance-adjusted), ARI, NMI.

Outputs (next to this script): results.csv (per-seed rows), SUMMARY.md
"""

import csv
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import igraph as ig
import leidenalg as la
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

REPO = Path("/home/md724/community_detection_spectral")
OUT = Path(__file__).resolve().parent
DATA = REPO / "data"
sys.path.insert(0, str(REPO))
from experiments.dspar import dspar_sparsify  # noqa: E402

SEEDS = list(range(10))
N_ITER = 2
ALPHA_PAPER = 0.8
ALPHA_CAL = 0.9
RESMATCH_TRIGGER = 0.20  # relative cluster-count gap vs baseline needed to run ctrl 5


# =============================================================================
# Loaders (mimic PAPER_EXPERIMENTS/exp2_ground_truth_recovery.py)
# =============================================================================

def load_karate():
    G = nx.karate_club_graph()
    gt = {n: (0 if G.nodes[n]["club"] == "Mr. Hi" else 1) for n in G.nodes()}
    return G, gt


def load_dolphins():
    G = nx.read_gml(DATA / "dolphins.gml", label="id")
    from networkx.algorithms.community import kernighan_lin_bisection
    comms = kernighan_lin_bisection(G, seed=0)
    gt = {}
    for i, c in enumerate(comms):
        for n in c:
            gt[n] = i
    return G, gt


def load_football():
    gi = ig.Graph.Read_GML(str(DATA / "football.gml"))
    gi.simplify(multiple=True, loops=True)
    G = nx.Graph()
    for v in gi.vs:
        G.add_node(v.index, **v.attributes())
    for e in gi.es:
        G.add_edge(e.source, e.target)
    gt = {n: G.nodes[n]["value"] for n in G.nodes()}
    return G, gt


def load_polbooks():
    G = nx.read_gml(DATA / "polbooks.gml", label="id")
    lm = {"l": 0, "n": 1, "c": 2}
    gt = {n: lm.get(G.nodes[n].get("value", "n"), 1) for n in G.nodes()}
    return G, gt


def load_email():
    G = nx.Graph()
    with open(DATA / "email-Eu-core.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2:
                u, v = int(p[0]), int(p[1])
                if u != v:
                    G.add_edge(u, v)
    gt = {}
    with open(DATA / "email-Eu-core-department-labels.txt") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                gt[int(p[0])] = int(p[1])
    return G, gt


DATASETS = [
    ("Karate", load_karate),
    ("Dolphins", load_dolphins),
    ("Football", load_football),
    ("Polbooks", load_polbooks),
    ("email-Eu-core", load_email),
]


def canonicalise(G_raw, gt_raw):
    """undirected simple -> labelled nodes only -> LCC -> relabel 0..n-1.

    Returns (nx.Graph G, igraph g, np.array true_labels aligned to 0..n-1).
    """
    G = nx.Graph()
    G.add_nodes_from(n for n in G_raw.nodes() if n in gt_raw)
    for u, v in G_raw.edges():
        if u != v and u in gt_raw and v in gt_raw:
            G.add_edge(u, v)
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()
    order = sorted(G.nodes())
    remap = {o: i for i, o in enumerate(order)}
    G = nx.relabel_nodes(G, remap, copy=True)
    labels_raw = [gt_raw[o] for o in order]
    _, y = np.unique(np.asarray(labels_raw, dtype=object).astype(str), return_inverse=True)
    g = ig.Graph(n=G.number_of_nodes(),
                 edges=[(u, v) for u, v in G.edges()], directed=False)
    return G, g, y


# =============================================================================
# Sparsifiers
# =============================================================================

def dspar_paper_unweighted(G, retention, seed):
    """experiments/dspar.py method='paper' (with replacement + reweight), weights
    then DROPPED -- exactly what PAPER_EXPERIMENTS/exp2_ground_truth_recovery.py does."""
    Gw = dspar_sparsify(G, retention=retention, method="paper", seed=seed)
    gs = ig.Graph(n=G.number_of_nodes(),
                  edges=[(u, v) for u, v in Gw.edges()], directed=False)
    return gs


def _probs_calibrated(scores, alpha):
    """lambda s.t. sum(min(1, lambda*s_e)) = alpha*m  =>  E[retention] = alpha.
    Copied from PAPER_RESTRUCTURE/exp_C_true_retention_seeded/run.py."""
    m = len(scores)
    if alpha >= 1.0:
        return np.ones(m)
    target = alpha * m
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


def dspar_calibrated(g, alpha, seed):
    deg = np.asarray(g.degree(), dtype=np.float64)
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    scores = 1.0 / deg[e[:, 0]] + 1.0 / deg[e[:, 1]]
    probs = _probs_calibrated(scores, alpha)
    rs = np.random.RandomState(seed)
    kept = e[rs.random_sample(len(scores)) < probs]
    return ig.Graph(n=g.vcount(), edges=[tuple(x) for x in kept], directed=False)


# =============================================================================
# Leiden
# =============================================================================

def leiden_mod(g, seed):
    part = la.ModularityVertexPartition(g)
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    return np.asarray(part.membership)


def leiden_rb(g, seed, gamma):
    part = la.RBConfigurationVertexPartition(g, resolution_parameter=gamma)
    opt = la.Optimiser()
    opt.set_rng_seed(int(seed) % (2 ** 31 - 1))
    opt.optimise_partition(part, n_iterations=N_ITER)
    return np.asarray(part.membership)


def leiden_matched(g, seed, target, tol=0.05, max_iter=40):
    """Leiden on the ORIGINAL graph, resolution tuned so #clusters ~= target.
    Structure borrowed from exp_D_groundtruth_scale/run.py (bisection on gamma)."""
    def n_of(gamma):
        memb = leiden_rb(g, seed, gamma)
        return memb, len(set(memb.tolist()))

    lo, hi = 1.0, 1.0
    memb, n_hi = n_of(hi)
    best = (abs(n_hi - target), memb, hi, n_hi)
    it = 0
    # if gamma=1 already overshoots, search downward
    if n_hi > target:
        while n_hi > target and lo > 1e-4 and it < max_iter:
            hi = lo
            lo /= 2.0
            memb, n_hi = n_of(lo)
            it += 1
            if abs(n_hi - target) < best[0]:
                best = (abs(n_hi - target), memb, lo, n_hi)
    else:
        while n_hi < target and hi < 1e5 and it < max_iter:
            lo = hi
            hi *= 2.0
            memb, n_hi = n_of(hi)
            it += 1
            if abs(n_hi - target) < best[0]:
                best = (abs(n_hi - target), memb, hi, n_hi)
    for _ in range(max_iter - it):
        if best[0] <= max(1.0, tol * target):
            break
        mid = 0.5 * (lo + hi)
        memb, n_mid = n_of(mid)
        if abs(n_mid - target) < best[0]:
            best = (abs(n_mid - target), memb, mid, n_mid)
        if n_mid < target:
            lo = mid
        else:
            hi = mid
    return best[1], best[2], best[3]


def score(y_true, memb):
    return dict(
        n_clusters=int(len(set(memb.tolist()))),
        AMI=float(adjusted_mutual_info_score(y_true, memb)),
        ARI=float(adjusted_rand_score(y_true, memb)),
        NMI=float(normalized_mutual_info_score(y_true, memb)),
    )


# =============================================================================
# Runner
# =============================================================================

def main():
    rows = []
    meta = {}

    for name, loader in DATASETS:
        G_raw, gt_raw = loader()
        G, g, y = canonicalise(G_raw, gt_raw)
        n, m = g.vcount(), g.ecount()
        k_gt = len(set(y.tolist()))
        meta[name] = dict(n=n, m=m, k_gt=k_gt)
        print(f"\n=== {name}: n={n} m={m} gt_communities={k_gt}", flush=True)

        nclust = {}

        def add(cond, seed, memb, true_ret, gamma):
            r = score(y, memb)
            rows.append(dict(dataset=name, n=n, m=m, k_gt=k_gt, condition=cond,
                             seed=seed, true_retention=round(true_ret, 4),
                             resolution=round(gamma, 5), **r))
            nclust.setdefault(cond, []).append(r["n_clusters"])
            return r

        # --- 1. baseline ---
        for s in SEEDS:
            memb = leiden_mod(g, seed=100 + s)
            r = add("baseline", s, memb, 1.0, 1.0)
        print(f"  baseline        k={np.mean(nclust['baseline']):.1f} "
              f"AMI={np.mean([x['AMI'] for x in rows if x['dataset']==name and x['condition']=='baseline']):.4f}",
              flush=True)

        # --- 2. draft condition: DSpar paper alpha=0.8, weights dropped ---
        for s in SEEDS:
            gs = dspar_paper_unweighted(G, ALPHA_PAPER, seed=200 + s)
            memb = leiden_mod(gs, seed=100 + s)
            add("dspar_paper_08", s, memb, gs.ecount() / m, 1.0)

        # --- 4. calibrated true-retention 0.9 ---
        for s in SEEDS:
            gs = dspar_calibrated(g, ALPHA_CAL, seed=200 + s)
            memb = leiden_mod(gs, seed=100 + s)
            add("dspar_cal_09", s, memb, gs.ecount() / m, 1.0)

        k_base = float(np.mean(nclust["baseline"]))

        # --- 3./5. resolution-matched controls on the ORIGINAL graph ---
        for cond in ("dspar_paper_08", "dspar_cal_09"):
            k_t = float(np.mean(nclust[cond]))
            gap = abs(k_t - k_base) / max(k_base, 1.0)
            if cond == "dspar_cal_09" and gap <= RESMATCH_TRIGGER:
                print(f"  [skip resmatch_{cond}] cluster count {k_t:.1f} within "
                      f"{gap*100:.1f}% of baseline {k_base:.1f}", flush=True)
                continue
            target = int(round(k_t))
            for s in SEEDS:
                memb, gamma, nc = leiden_matched(g, seed=100 + s, target=target)
                add(f"resmatch_{cond}", s, memb, 1.0, gamma)
            print(f"  resmatch_{cond}: target k={target} achieved "
                  f"{np.mean(nclust['resmatch_'+cond]):.1f}", flush=True)

        for cond in nclust:
            sub = [x for x in rows if x["dataset"] == name and x["condition"] == cond]
            print(f"  {cond:22s} k={np.mean([x['n_clusters'] for x in sub]):6.1f} "
                  f"ret={np.mean([x['true_retention'] for x in sub]):.3f} "
                  f"AMI={np.mean([x['AMI'] for x in sub]):.4f} "
                  f"ARI={np.mean([x['ARI'] for x in sub]):.4f} "
                  f"NMI={np.mean([x['NMI'] for x in sub]):.4f}", flush=True)

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    write_summary(rows, meta)
    print(f"\nWrote {OUT/'results.csv'} and {OUT/'SUMMARY.md'}")


def write_summary(rows, meta):
    def agg(ds, cond, key):
        v = [r[key] for r in rows if r["dataset"] == ds and r["condition"] == cond]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))

    def conds(ds):
        seen, out = set(), []
        order = ["baseline", "dspar_paper_08", "resmatch_dspar_paper_08",
                 "dspar_cal_09", "resmatch_dspar_cal_09"]
        for c in order:
            if any(r["dataset"] == ds and r["condition"] == c for r in rows) and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    datasets = [d for d, _ in DATASETS if any(r["dataset"] == d for r in rows)]
    L = []
    L.append("# Experiment F — does the draft's Exp-3 ground-truth result survive "
             "chance correction and a granularity control?\n")
    L.append("`baseline` = Leiden (ModularityVertexPartition) on the original graph. "
             "`dspar_paper_08` = the draft's pipeline: DSpar method=\"paper\" "
             "(with replacement + reweight), nominal retention 0.8, weights dropped. "
             "`resmatch_X` = **no sparsification**, Leiden on the ORIGINAL graph with "
             "RBConfiguration resolution tuned so the cluster count matches X "
             "(granularity control). `dspar_cal_09` = calibrated sampler with "
             "E[true retention] = 0.90 exactly.\n")
    L.append("10 seeds per cell. AMI = adjusted mutual information (chance-corrected, "
             "the metric that is *not* inflated by returning more clusters); "
             "NMI is the draft's metric.\n")

    L.append("\n## Graphs\n")
    L.append("| dataset | n (LCC) | m | ground-truth communities |")
    L.append("|---|---|---|---|")
    for d in datasets:
        s = meta[d]
        L.append(f"| {d} | {s['n']} | {s['m']} | {s['k_gt']} |")

    L.append("\n## Results (mean ± std over 10 seeds; Δ vs baseline)\n")
    L.append("| dataset | condition | true ret. | γ | n_clusters | AMI | ΔAMI | "
             "ARI | ΔARI | NMI | ΔNMI |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in datasets:
        b = {k: agg(d, "baseline", k)[0] for k in ("AMI", "ARI", "NMI")}
        for c in conds(d):
            ret = agg(d, c, "true_retention")[0]
            gam = agg(d, c, "resolution")[0]
            kc, ks = agg(d, c, "n_clusters")
            cells = []
            for k in ("AMI", "ARI", "NMI"):
                mu, sd = agg(d, c, k)
                delta = "—" if c == "baseline" else f"{mu-b[k]:+.4f}"
                cells += [f"{mu:.4f}±{sd:.4f}", delta]
            L.append(f"| {d} | {c} | {ret:.3f} | {gam:.3f} | {kc:.1f}±{ks:.1f} | "
                     + " | ".join(cells) + " |")

    L.append("\n## Sparsification vs its granularity-matched control\n")
    L.append("If `resmatch` matches or beats the sparsified run, the sparsifier "
             "contributed nothing beyond changing partition granularity.\n")
    L.append("| dataset | condition | AMI (sparse) | AMI (resmatch) | diff | "
             "ARI (sparse) | ARI (resmatch) | diff | NMI (sparse) | NMI (resmatch) | diff |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in datasets:
        for c in ("dspar_paper_08", "dspar_cal_09"):
            rc = f"resmatch_{c}"
            if not any(r["dataset"] == d and r["condition"] == rc for r in rows):
                continue
            cells = []
            for k in ("AMI", "ARI", "NMI"):
                a = agg(d, c, k)[0]
                bb = agg(d, rc, k)[0]
                cells += [f"{a:.4f}", f"{bb:.4f}", f"{a-bb:+.4f}"]
            L.append(f"| {d} | {c} | " + " | ".join(cells) + " |")

    # ---- seed-paired deltas (same Leiden seed index in both arms) ------------
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r["condition"]), {})[int(r["seed"])] = r

    def paired(ds, a, b, key):
        A, B = by.get((ds, a)), by.get((ds, b))
        if not A or not B:
            return None
        seeds = sorted(set(A) & set(B))
        v = np.array([A[s][key] - B[s][key] for s in seeds])
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        t = float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else float("nan")
        return v.mean(), sd, t, int((v > 0).sum()), len(v)

    L.append("\n## Seed-paired deltas (10 matched seeds, paired t)\n")
    L.append("`vs baseline` answers \"is the sparsified run better than plain Leiden?\"; "
             "`vs resmatch` answers \"is it better than plain Leiden made equally fine-grained?\"\n")
    L.append("| dataset | comparison | metric | mean Δ | std | t | wins |")
    L.append("|---|---|---|---|---|---|---|")
    for d in datasets:
        for a, b, lab in [("dspar_paper_08", "baseline", "dspar_paper_08 vs baseline"),
                          ("dspar_cal_09", "baseline", "dspar_cal_09 vs baseline"),
                          ("dspar_paper_08", f"resmatch_dspar_paper_08",
                           "dspar_paper_08 vs resmatch"),
                          ("resmatch_dspar_paper_08", "baseline", "resmatch vs baseline")]:
            for k in ("AMI", "ARI", "NMI"):
                p = paired(d, a, b, k)
                if p is None:
                    continue
                mu, sd, t, w, nn = p
                ts = "n/a" if np.isnan(t) else f"{t:+.2f}"
                L.append(f"| {d} | {lab} | {k} | {mu:+.4f} | {sd:.4f} | {ts} | {w}/{nn} |")

    L.append("\n## Verdict\n")
    L.append(VERDICT)
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")


VERDICT = """
**(a) Does email-Eu-core's gain survive chance-adjusted metrics?**
Only in the weakest sense. Under the draft's own metric the effect shrinks by half at the
LCC-restricted, 10-seed replication (draft: ΔNMI=+0.025, ΔARI=+0.071; here ΔNMI=+0.009,
ΔARI=+0.037). Under the chance-corrected AMI it is ΔAMI=+0.007 ± 0.019 (paired t=+1.26,
7/10 seeds), i.e. statistically indistinguishable from zero. ΔARI=+0.037 ± 0.045
(t=+2.63) is the only nominally significant piece, and it is not robust to (b).

**(b) Does it survive the resolution-matched control?**
No. Leiden on the UNSPARSIFIED graph, with the resolution tuned to the same cluster
count DSpar produces (7.7 -> 8.6 clusters), reproduces the entire gain and slightly
exceeds it: ΔAMI=+0.014, ΔARI=+0.039, ΔNMI=+0.015 vs baseline — larger than DSpar's
+0.007/+0.037/+0.009. Sparsified minus resolution-matched is NEGATIVE on all three
metrics (ΔAMI=-0.006, ΔARI=-0.001, ΔNMI=-0.006; |t| <= 0.9). The email-Eu-core
"improvement" is a granularity effect available for free by turning one knob, at 100%
of the edges; DSpar throws away 56% of the distinct edges to buy nothing.

**(c) Do the four negative datasets stay negative?**
Yes, all four, on every metric, and more strongly under AMI than under NMI.
ΔAMI: Karate -0.191, Polbooks -0.122, Dolphins -0.055, Football -0.037
(paired t = -8.1, -7.3, -3.4, -3.5; 0-2 winning seeds out of 10). The calibrated
true-retention-0.90 sampler — which does not shred the graph — is also negative on all
four (ΔAMI -0.065, -0.060, -0.009, -0.032) and negative on email-Eu-core too
(ΔAMI=-0.009), so the single positive result depends on the with-replacement sampler's
~44% true retention, not on DSpar's degree-based scores.

**(d) Implication for the Exp-3 rewrite.**
Score is 0/5, not 1/5: once recovery is measured with a chance-corrected metric and
compared against an equally fine-grained unsparsified partition, DSpar does not improve
ground-truth recovery on any of the draft's five datasets — the email-Eu-core cell
should be reported as a partition-granularity artifact of with-replacement sampling
(nominal alpha=0.8 -> 0.44 true retention), with the resolution-matched control as the
correct baseline, which retires the delta>0 "favorable regime" claim built on it.
"""


if __name__ == "__main__":
    main()
