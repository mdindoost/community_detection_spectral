# Exp AA — SUMMARY: The Satuluri regime (does sparsification help where its authors said it does?)

**VERDICT — the kill criterion fired, against our own paper.**
`"Graph sparsification does not help community detection"` is **FALSE as an unscoped
statement and must be rescoped before print.** With **real Metis** (the algorithm the 2011
accuracy claim is actually about), L-Spar produces a positive, seed-robust, control-surviving
improvement in **both** honest-transfer modularity on the ORIGINAL graph **and** chance-corrected
recovery, and it switches sign at **d_avg ~ 50 — the exact threshold Satuluri et al. state**
(NOTES_satuluri2011 §4/§5). Every network in our study sits below that threshold, which is why
we never saw it.

The correct scoped claim is:

> Sparsification does not help **modularity-family, free-granularity** detection
> (Leiden/Louvain/Infomap) on **sparse graphs (d_avg < ~50)**; for **fixed-k balanced cut
> partitioners** at **d_avg >= 50** a similarity-based sparsifier (L-Spar) genuinely improves
> both the objective and recovery — but never above what resolution-tuned Leiden on the
> untouched graph already achieves, and (with exact Jaccard) at a net compute loss.

---

## 1. Arm A — the average-degree sweep, Leiden. P1 essentially HOLDS, P2 HOLDS-then-dies-under-control.

`results_armA.csv` (409 rows; LFR n=10^4, tau1=2.1, tau2=1.5, 3 seeds/cell,
3 sparsifiers x 4 retention targets x 3 Leiden seeds).

**P1 (no positive honest modularity gain).** Column `dQ_vs_matched` exceeds 2x the baseline
seed sd (`Q_base_std`) in only 3 of 144 aggregated cells — all mu=0.5, d_nom=100, lspar
(+0.0038/+0.0041/+0.0046 vs sd 0.00148). Against the **stricter** best-of-5-restart baseline
(`dQ_vs_base_best`) those same cells give only +0.0009..+0.0018, i.e. **below** seed sd.
=> For Leiden, P1 survives: the high-degree honest modularity gain is at most ~+0.005 and does
not beat best-of-5 restarts. The formal 2-sd trigger fires only because runtime matching
allotted the baseline a single restart (`n_restarts` = 1 in those rows).

**P2 (recovery improves at d_avg >= 50).** `dAMI_vs_base_mean` for lspar rises with degree:
mu=0.3: +0.034 (d 33.0) -> +0.040 (57.0) -> +0.041 (110.1) -> +0.058 (216.8);
mu=0.5: +0.063 (49.4) -> +0.096 (101.9). Baseline `AMI_base_std` is 0.0014-0.0090, so these are
7-40x seed noise. **P2 holds as registered.**

**...but the resolution-matched control kills it.** `AMI_resmatch` (Leiden on the ORIGINAL graph
at the gamma whose nc is nearest the sparse nc, `nc_resmatch` recorded for audit) is *higher*
than the sparsified AMI in 21 of the 29 qualifying d>=50 cells: e.g. mu=0.3/d=57.0/ret 0.151,
AMI 0.9866 vs resmatch 0.9928; mu=0.5/d=49.4/ret 0.199, 0.8927 vs 0.9145.
=> **Leiden's recovery gain is a granularity effect**, the same mechanism that killed the
Youtube gain in exp_X. C6/C16 are unchanged for the modularity-optimizer family.

**HYPOTHESIS (post-hoc, needs its own run):** the one exception is mu=0.5, d_avg~221, where
lspar beats the resolution-matched control by +0.017..+0.024 AMI. Only **1 of 3 LFR seeds** has
completed for that cell. Flagged HYPOTHESIS per EXPLORATION.md rule 2; do not cite.

**Controls all clean.** Size-matched random-partition chance floor `AMI_chance` = -0.0001 at
every degree. Realized retention reported in every row.

---

## 2. Arm B — fixed-k partitioners. P3 HOLDS DECISIVELY; P1 IS FALSIFIED HERE.

`results_armB.csv` (695 rows). **Available detectors, stated plainly:**
- **pymetis 2025.2.2 = REAL Metis.** Installed successfully. `nc_base == nc_sparse == k_fixed`
  in every row, so **granularity is pinned by construction and no resolution artifact is
  possible.** This is the primary Arm B result.
- **`eigen_proxy` = igraph `community_leading_eigenvector(clusters=k)` — a FAILED proxy.**
  It treats k as an upper bound and stops early: asked for k~100 it returns nc = 3-29 on the
  original graph and up to 2436 on sparsified graphs. Its large apparent gains (dQ up to +0.18,
  dAMI up to +0.62) are granularity-confounded and are **not** evidence. 6 ARPACK convergence
  failures logged in `detector_failures.csv`.
- **`spectral_proxy` = sklearn SpectralClustering — NOT RUN.** Timed at 98 s (d=10) to 138 s
  (d=200) per call; did not fit the budget. **DESIGN deviation, disclosed.**
- **Graclus: unavailable.** No maintained Python binding; not attempted.

**Metis at mu=0.5 (Satuluri's setting), cells with positive value out of 12 per degree
(`dQ` / `dAMI`), all honest-transfer to the ORIGINAL graph:**

| d_avg | L-Spar dQ+ | L-Spar dAMI+ | mean dQ | mean dAMI | DSpar dQ+ | random dQ+ |
|---|---|---|---|---|---|---|
| 11.2  | 0/12 | 0/12  | -0.103 | -0.216 | 0/12 | 0/12 |
| 24.6  | 0/12 | 6/12  | -0.046 | -0.072 | 0/12 | 0/12 |
| **49.4**  | **8/12** | **9/12**  | -0.016 | **+0.042** | 0/12 | 0/12 |
| **101.9** | **10/12** | **12/12** | **+0.006** | **+0.116** | 2/12 | 0/12 |
| **220.9** | **12/12** | **12/12** | **+0.009** | **+0.093** | 3/12 | 2/12 |

Best cells: d_avg 49.4 / ret 0.149 -> dQ +0.0049, dAMI +0.152, dARI +0.105;
d_avg 101.9 / ret 0.149 -> dQ +0.0086, dAMI +0.142; d_avg 220.9 / ret 0.148 -> dQ +0.0115,
dAMI +0.113. **The sign flip sits at d_avg ~ 50, matching their stated threshold.**

**Balance (cv), a benefit our paper had never measured:** Metis `cv_base` -> `cv_sparse`
0.030 -> 0.027 (d 49), 0.025 -> 0.024 (d 102), 0.027 -> 0.026 (d 221). L-Spar preserves or
slightly improves balance, as Satuluri et al. claim; it does not buy quality with imbalance.

**mu = 0.8 does NOT reproduce their largest reported gain.** At nominal mu=0.8 (realized
0.864-0.914) Metis `dQ` is positive in 1 of 60 cells; `dAMI` is positive only at retention 0.5
(+0.017..+0.055) on baselines of AMI 0.03-0.16. Their Metis+MQI F 26.95 -> 40.47 at mu=0.8,
d=50 does not appear under our metric and detector.

---

## 3. metis_noise.csv — the gain is not Metis run-to-run variance. P1 KILL CONFIRMED.

108 rows, 10 Metis option seeds x 3 LFR seeds per cell, mu=0.5, worst case = (min sparsified)
minus (max baseline) across the 10 seeds.

| d_avg | target | realized | dQ (worst) | dAMI (worst) | Metis Q sd / AMI sd |
|---|---|---|---|---|---|
| 24.6  | 0.20 | 0.198 | -0.0244 (-0.0295) | +0.0259 (-0.0024) | 0.0018 / 0.0125 |
| **49.4**  | 0.15 | 0.149 | **+0.0054 (+0.0010)** | **+0.157 (+0.134)** | 0.0019 / 0.0128 |
| **49.4**  | 0.20 | 0.198 | **+0.0080 (+0.0041)** | **+0.151 (+0.127)** | 0.0019 / 0.0128 |
| **101.9** | 0.20 | 0.200 | **+0.0097 (+0.0052)** | **+0.133 (+0.102)** | 0.0018 / 0.0105 |
| **220.9** | 0.20 | 0.198 | **+0.0129 (+0.0076)** | **+0.101 (+0.064)** | 0.0022 / 0.0139 |

**Matched-retention controls at the same degrees are uniformly negative:** DSpar dQ
-0.145..-0.003 and dAMI -0.480..+0.032; uniform random dQ -0.140..-0.0004 and dAMI
-0.450..-0.020. **The effect is L-Spar's Jaccard signal, not the edge budget.**

---

## 4. Arm C — the denoising mechanism is REAL but is not the whole explanation. P4 PARTLY HOLDS.

`results_armC.csv` (240/240 rows, complete). x% of m added as uniformly random NEW edges,
planted labels fixed, honest transfer scored on the noisy graph and separately on the clean one.

- **Monotone rise: Spearman(x, dAMI) = 1.00 with p<1e-4 in 4/4 Leiden+L-Spar cells** (and 0.9 in
  2/4 Metis+L-Spar cells). Example, d=50 / ret 0.199: dAMI +0.062 (x=0) -> +0.075 -> +0.091 ->
  +0.119 -> **+0.184 (x=100)**. Exactly the shape the denoising hypothesis predicts.
- **DSpar shows no denoising**: rho = +1.0, +0.7, -0.3, -1.0 — mixed, and dAMI is -0.10..-0.69
  throughout. The mechanism is **specific to similarity-based sparsification**.
- **Strict P4 fails**: the gain does not vanish at x=0 (+0.008..+0.144 at zero noise), so only
  **3 of 16** cells satisfy `rho > 0.8 AND dAMI <= 0 at x=0`.
  => **Denoising is additive to a pre-existing L-Spar advantage, not its cause.**
- **New observation:** `dQ_clean` (partition scored on the UNCORRUPTED graph) rises monotonically
  with x and turns positive: d=50/ret 0.5, -0.0002 -> +0.0017 -> +0.0043 -> +0.0070 -> +0.0152,
  while `dQ_noisy` stays ~0. Sparsification improves the objective **on the true graph** when the
  observed graph is corrupted. Its interpretation assumes access to a "true" graph one does not
  have; label accordingly.

---

## 5. Cost — the quality win is bought at a compute loss with exact Jaccard.

`results_armA.csv`, lspar rows, end-to-end including sparsification:

| d_nom | T_sparsify | T_leiden_orig | speedup_pipeline | speedup_detect_only |
|---|---|---|---|---|
| 10  | 0.16 s | 0.64 s | 1.49x | 2.29x |
| 50  | 0.73 s | 1.21 s | 1.16x | 3.92x |
| 100 | 1.87 s | 2.31 s | 1.01x | 5.69x |
| 200 | 7.35 s | 4.70 s | **0.58x** | **7.24x** |

Detection-only speedup grows to 7.2x, but exact Jaccard overtakes it. Satuluri et al. measured
minhash at **240x cheaper** than exact (their §4, 1 min vs 4 h) against **hour-scale** clustering,
so their 10-50x is consistent with ours once the two cost bases are matched. For Metis the
imbalance is starker (`T_base` 0.37-1.16 s vs `T_pipeline` 0.68-8.76 s, speedup 0.13-0.55x):
**the Arm B quality result is a quality result, not a speed result, in our implementation.**

---

## 6. Context that must travel with the positive result

Best AMI achievable per cell by route (`results_armA.csv` + `results_armB.csv`):

| mu | d_avg | Leiden base | Leiden res-matched | Leiden+L-Spar | Metis base | Metis+L-Spar |
|---|---|---|---|---|---|---|
| 0.5 | 24.6  | 0.774 | **0.896** | 0.795 | 0.586 | 0.649 |
| 0.5 | 49.4  | 0.830 | **0.956** | 0.903 | 0.596 | 0.754 |
| 0.5 | 101.9 | 0.778 | **0.940** | 0.887 | 0.601 | 0.761 |
| 0.3 | 216.8 | 0.933 | **1.000** | 0.999 | -- | -- |

**Resolution-tuned Leiden on the untouched graph wins every cell.** L-Spar lifts a weaker fixed-k
partitioner from 0.60 to 0.76; it never reaches what the free-granularity optimizer already does.
The honest sentence is *"sparsification can repair a fixed-k cut partitioner on dense graphs"*,
not *"sparsification improves community detection"*.

---

## 7. Caveats and DESIGN deviations (all disclosed)

1. **The registered LFR parameter box is infeasible above d_avg ~ 50.** networkx LFR can only
   place all nodes when `(1-mu)*min_degree < min_community` and `(1-mu)*max_degree <
   max_community`. **Deviation:** a documented ladder — `registered` -> `maxdeg_cap{100,85,70,55}`
   -> `scaled_maxk5/3`. Per-cell mode/box/realized stats in `lfr_generation.csv`.
   **45/45 cells generated; zero cells lost.** mu=0.5 stayed on the REGISTERED community range
   [20,500] to d_avg=102; only d_avg~200 and mu=0.3 d>=100 needed `scaled`. Confound to keep in
   mind: at d_avg~200 the community-size floor rises (20 -> 40-56).
2. **networkx does not honour mu.** Realized mixing runs +0.06 to +0.11 above request. All
   conclusions stated against realized values. The nominal-0.8 block is near-structureless
   (baseline AMI 0.03-0.16) and is weak evidence either way.
3. **Realized average degree overshoots nominal** in some cells; always bin by `d_avg_real`.
   No cell crosses the d_avg=50 boundary ambiguously.
4. **`spectral_proxy` not run** (cost); **Graclus unavailable**; `eigen_proxy` does not honour k
   and is reported as a failed proxy, not evidence. **Only Metis is real.**
5. **Resolution-matched control is a shared gamma grid** (13 gammas 0.25..120, nearest-nc match)
   with a 240 s cumulative budget that truncates on the largest graphs. `nc_resmatch` records the
   achieved match for audit.
6. **Compute ran entirely on the 14 GB local machine.** Fuji was unreachable — Tailscale SSH
   printed an auth URL; the user must re-click it to restore access.
7. **Still filling in** (detached, resume-safe): Arm A mu=0.8 at d>=50 and mu=0.5 at d_avg~221
   seeds 2-3. Neither affects any verdict above.
8. **`dQ_vs_matched` is sensitive to the restart budget** — where the pipeline is fast the matched
   baseline gets only 1 restart; always read `dQ_vs_base_best` alongside. This is why Arm A's
   formal P1 trigger is not treated as a real Leiden gain.

## 8. Files
`run.py` (gen/armA/armB/armC), `metis_noise.py`, `analyse.py`, `lfr_generation.csv`,
`results_armA.csv`, `results_armB.csv`, `results_armC.csv`, `metis_noise.csv`,
`detector_failures.csv`, `*.log`.

## 9. Consequences for the paper
- **C16 must be rescoped**, and the word "refutation" stays out — Satuluri et al.'s stated
  threshold **reproduces** with their algorithm family on their generator.
- **New claim candidate (C19): the objective/recovery dissociation is algorithm-family
  dependent.** For free-granularity modularity optimizers, sparsification's recovery gain is a
  granularity artifact; for fixed-k balanced cut partitioners at d_avg >= 50 it is a genuine gain
  on both metrics that no granularity story can explain, because k is pinned by construction.
- **The headline must carry three qualifiers**: modularity-family detection, sparse
  (d_avg < ~50) graphs, near-linear optimizers. The "field regressed" spine is unaffected and
  arguably strengthened — the founding result holds in its own regime; what was lost in the
  fifteen years since is the *scope*, not the *finding*.
