# Exp K — Weighted-regime verification (full grid: 8 networks x 2 samplers, 5 seeds/cell)

Files: audit_feb.txt, run.py, results.csv (135 rows), controls.py, controls.csv (205 rows),
spectral_gap.py/csv, make_summary.py, tables.md (full machine-generated tables).

## Verdict (four parts)

**V1 — Preservation: YES, decisively.** Weighted dQ_fixed within +-0.0041 of zero in all 24 cells,
INDEPENDENT of 1/alpha (com-Amazon, 1/alpha = 10,782, is the tightest at |dQ| <= 1e-4).
Controlled attribution: identical topology with weights dropped inflates +0.013..+0.152.
Unbiasedness E[A']=A carries preservation where the (1+-eps/alpha) bound is vacuous.

**V2 — Transfer: baseline-level only at mild (90%) retention; at aggressive retention the weights
HURT transfer (worse than unweighted twin 6/8).** Key sentence: the weights preserve a fixed
partition's objective value perfectly while the detection problem on the sparse graph is destroyed.
Enron's genuine +0.011 reproduced (beats best-of-5).

**V3 — Recovery: weighted is WORSE than baseline in every configuration and worse than its
unweighted twin on identical topology every time; seed variance ~triples. Not a granularity
artifact (cluster counts within 20%).**

**V4 — "Safe acceleration": "safe" yes, "acceleration" no.** Weights are computationally free
(0.92-1.06x) and MANDATORY for honest fixed-partition evaluation on sparsified graphs; but
end-to-end speedup at quality-preserving retention (alpha=0.9) is 0.97-1.06x — within noise of 1x.
Caveats: single-threaded leidenalg, n_iterations=2; recovery tested on one labeled network.

## Paper wording (agent-suggested, adopted into STORY):
Weights are a correctness requirement for evaluation, not a performance option. Sparsification is
not a route to faster or better modularity community detection: no quality gain, ~1x speed at
quality-preserving retention, and detection/recovery degrade beyond ~10% edge removal.

## New numbers for the suite
1/alpha (normalized Laplacian): ca-GrQc 535.5, com-Amazon 10,782, com-DBLP 373, ca-HepTh 320,
email-Enron 284, ca-CondMat 139.2, wiki-Vote 9.9, email-Eu-core 4.7.
Feb run audited: numbers reproduce verbatim; n=1 replicate, salted-hash seeds (non-reproducible),
alpha=0.8 never ran, dQ_leiden was cross-graph (Artifact I). Sampler verified bit-identical to
Liu et al. Algorithm 1.
