# Exp R — iteration-matched Enron control (pre-registered design)

Registered 2026-07-25, BEFORE any Exp R code or run existed. Fills exp_N caveat C4.

**Claim it could change:** C9 (Enron genuine gain) as refined by Exp N. Exp N's compute control
is wall-clock only: the seeded pipeline (2.29s) was compared against plain restarts (1.24s each)
and restart portfolios. Missing arm: plain Leiden given the SAME budget as one pipeline via more
iterations rather than more restarts. If deeper plain Leiden closes the gap, the honest sentence
"seeding places one pipeline at the 88th percentile of the plain distribution at ~1/10 the
compute of a matching restart search" must be weakened to "seeding matches deeper plain Leiden
at equal cost" — a much weaker claim.

**Design.** email-Enron LCC (n=33,696, m=180,811), same loader/conventions as exp_N (read its
run.py; reuse code verbatim where possible). Arms, 25 seeds each (seeds 100-124):
  A. plain Leiden, n_iterations=2 (reference; must reproduce exp_N's plain distribution —
     sanity check against partition_quality.csv, mean 0.608432).
  B. plain Leiden, n_iterations chosen so median wall-clock per run ≈ one seeded pipeline
     (2.289s; calibrate n_iter on 3 throwaway seeds first, report the calibration).
  C. plain Leiden, n_iterations=-1 (run to convergence), wall-clock recorded — the asymptote.
Compare against exp_N's seeded distribution (partition_quality.csv, mean 0.614634 ± 0.0025,
best 0.617876) — do NOT re-run the seeded arm; same graph, same conventions make the numbers
directly comparable.

**Pre-registered predictions:**
- P1: arm B mean Q < 0.6146 (seeded mean) − i.e. iteration-deepening does not close the gap
  (basis: Exp N's matched-wall-clock restart bootstrap gave +0.0037 for seeded).
- P2: arm C (convergence) mean Q < seeded mean — the gap is not an early-stopping artifact.
- P3: arm B variance remains ≥ 1.5x the seeded variance (sd 0.0025) — seeding's variance
  reduction is not reproduced by deeper iteration.

**Kill criterion:** if arm B mean ≥ seeded mean − 0.001, OR Mann-Whitney between arm B and the
seeded 5 runs is non-significant at p>0.10 with arm B mean within 0.002, the "reliability and
price" framing of C9/Exp N is overturned for the iteration dimension: report it, revise the
Exp N discussion paragraph accordingly, no rescue tweaks.

**Output:** PAPER_RESTRUCTURE/exp_R_iteration_matched/{DESIGN.md,run.py,results.csv,SUMMARY.md}.
Cost estimate: 25 seeds x 3 arms x ~2-8s = minutes. Local machine, MemoryMax=3G, one job.
