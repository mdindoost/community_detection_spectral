#!/usr/bin/env python
"""Supplementary (NOT pre-registered, added after the main run): arm B at
n_iterations=5.  Reason: calibration on the 3 throwaway seeds put n_iter=4 at
2.518 s (1.10x the 2.289 s pipeline budget), but over the 25 main seeds its
median came in at 2.057 s (0.90x).  n_iter=5 therefore supplies the strictly
ABOVE-budget bracket the design asks for.  This arm can only help plain Leiden,
so it is adversarial to Exp R's own conclusion.  Appends to results.csv."""
import json
import numpy as np
from scipy.stats import mannwhitneyu
import run as R

g = R.load_graph(R.ENRON)
print(f"n={g.vcount():,} m={g.ecount():,}", flush=True)
q, k, t, _ = R.run_arm(g, "B_n5_supp", "supplementary", 5, R.MAIN_SEEDS)
import csv
exp_n = list(csv.DictReader(open(R.EXP_N_QUALITY)))
seeded = np.array([float(r["Q"]) for r in exp_n if r["kind"] == "seeded"])
u, p = mannwhitneyu(q, seeded, alternative="two-sided")
out = dict(summary=R.describe(q, k, t), mean_minus_seeded=float(q.mean() - seeded.mean()),
           var_ratio_pop=float(q.var(ddof=0) / seeded.var(ddof=0)),
           var_ratio_sample=float(q.var(ddof=1) / seeded.var(ddof=1)),
           mannwhitney_p=float(p),
           median_cost_vs_pipeline=float(np.median(t) / R.TARGET_SECONDS),
           frac_above_seeded_mean=float((q > seeded.mean()).mean()),
           frac_above_seeded_best=float((q > seeded.max()).mean()),
           kill_c1_meanB_ge_seededmean_minus_0p001=bool(q.mean() >= seeded.mean() - 0.001),
           kill_c2=bool(p > 0.10 and abs(q.mean() - seeded.mean()) < 0.002))
print(json.dumps(out, indent=2), flush=True)
json.dump(out, open(R.OUT / "results_supp_n5.json", "w"), indent=2)
