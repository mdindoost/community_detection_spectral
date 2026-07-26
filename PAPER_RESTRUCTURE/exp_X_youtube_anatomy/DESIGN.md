# Exp X — com-Youtube mini-anatomy (pre-registered design)

Registered 2026-07-25 before any Exp X code/run. Greenlit by Mohammad (small threads approved).

**Claim it could change:** C9's second genuine gain — com-Youtube +0.003 at calibrated
alpha>=0.9 (exp_H: survives Bonferroni at 0.95) — has never received the scrutiny Enron got
(exp_N + exp_R). Either it survives an exp_N-style granularity/statistics check and C9's
second leg hardens, or it dies and C9 becomes "one genuine gain, Enron only." Both outcomes
matter for the synthesis phase.

**Design (exp_N-lite, on Fuji).** com-Youtube LCC (n~1.13M, m~2.99M), loader conventions from
exp_N/exp_L. Calibrated DSpar sampler at the exp_H regime that produced the finding (read
exp_H_youtube_sweep/ for the exact alpha and configuration; use the strongest surviving cell,
expected alpha=0.95). Arms:
  A. 20 plain Leiden restarts (seeds 100-119), n_iterations=2.
  B. 5 seeded pipelines (spar seeds 200-204, leiden seeds 300-304): DSpar -> Leiden on sparse
     -> refine on full graph (exp_N pipeline shape), wall-clock recorded per stage.
Measurements (each mirrors exp_N):
  1. Statistical reality: seeded-vs-plain mean gap, Mann-Whitney + permutation test,
     bootstrap CI; seeded mean's percentile in the plain distribution.
  2. Granularity: corr(k,Q) across plain restarts; k-matched comparison (plain restarts with
     k in the seeded range); largest-community and >1%-of-n shape stats.
  3. Mechanism: cross-boundary vs intra-community edge-removal rates for the communities the
     seeded partition splits/merges (the exp_N 2.58x analogue), with endpoint degree-product
     ratios.
  4. Compute: one pipeline vs one restart wall clock; matched-wall-clock bootstrap
     E[best-of-n] as in exp_N finding 1. Phrase any compute claim IN EXPECTATION only.

**Pre-registered predictions:**
- P1: the seeded mean exceeds the plain mean by +0.002..+0.005 with p<0.05 (reproducing exp_H).
- P2: corr(k,Q) across plain restarts is |r|<0.3 and the k-matched gap remains positive
  (granularity does not explain the gain).
- P3: boundary-edge removal bias >1.5x in split communities (mechanism continuity with Enron).

**Kill criterion:** if the k-matched gap <= 0, or corr(k,Q) < -0.5 with the seeded k
systematically lower, the Youtube gain is a granularity artifact: C9 is DEMOTED to
"one genuine gain (Enron)" and exp_H's cell is annotated. If P1 fails to reproduce at all
(p>0.10), same demotion with "non-reproducible" annotation. No rescue tweaks.

**Output:** PAPER_RESTRUCTURE/exp_X_youtube_anatomy/{DESIGN.md,run.py,CSVs,results.json,
SUMMARY.md}. Compute on FUJI (62GB; ~30-60s per Leiden run x ~30 runs + refinements; expect
~1-2h). Dataset com-Youtube must be rsynced to Fuji first. One detached job, ulimit-capped.
