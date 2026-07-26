# Exp AC — SUMMARY: does the transition depend on degree heterogeneity? An SBM arm.

**VERDICT — P1 and P4 HOLD. P2 is FALSIFIED in the opposite direction. P3 is FALSIFIED as
vacuous. Neither kill direction fires, and the promotion criterion does not fire either:
degree heterogeneity is EXCLUDED as the variable that sets the transition location.**

> The fixed-$k$ gain reproduces on plain SBM and on degree-corrected SBM, so it is not an
> LFR artifact and the boundary now holds across two generator families. But the transition
> does **not** move with degree heterogeneity. All four SBM variants cross at average degree
> 24.5 to 24.7 while degree CV ranges over an order of magnitude (0.14 to 1.47), and LFR
> crosses at 49.4. Across 60 cells, Spearman(deg_cv, dQ) = **-0.260** and
> Spearman(d_avg, dQ) = **+0.652**. The suspected confound is not a confound. What remains
> unexplained is why LFR sits at twice the SBM value; the leading candidate is baseline
> detectability, not heterogeneity.

Ran on Fuji 2026-07-26, 16:12 to 18:43, one sequential capped job (EXPLORATION rule 7).
Files: `DESIGN.md` (pre-registered before any code), `run.py`, `driver.sh`, `analyse.py`
(written before the run finished), `analysis_full.txt`, 8 arm CSVs at 180 data rows each,
`metis_seeds.csv` (540), `sbm_generation.csv` (60), 12 logs.

Coverage: 4 generators x 5 nominal degrees (10/25/50/100/200) x 3 graphs per cell x
3 sparsifiers (L-Spar, DSpar, uniform random) x 4 retention targets (0.05/0.15/0.20/0.50)
x 2 arms (A = Leiden with free granularity, B = real Metis with $k$ pinned to the planted
block count). Registered generators `sbm` and `dcsbm`; `sbm_ps` and `dcsbm_ps` are the
disclosed 2x2 controls adding community-size heterogeneity. Sampler calibration succeeded on
every row (`cal_status = ok`, 180/180 per file); 9 rows per file are `floor` cases where the
requested retention is below the sparsifier's hard floor, as anticipated.

---

## V1 — P1 HOLDS. The fixed-$k$ sign flip is not LFR-specific. Kill direction 1 does not fire.

Under real Metis with $k$ pinned by construction, L-Spar produces positive honest-transfer
modularity on every generator. Counts of positive cells out of twelve, by realized degree
(`analysis_full.txt`, ARM B):

| generator | d≈11 | d≈25 | d≈49 | d≈102 | d≈220 |
|---|---|---|---|---|---|
| LFR (exp_AA) | 0/12 | 0/12 | 8/12 | 10/12 | 12/12 |
| sbm | 0/12 | 5/12 | **12/12** | 11/12 | 4/12 |
| sbm_ps | 0/12 | 3/12 | 9/12 | **12/12** | **12/12** |
| dcsbm | 0/12 | 9/12 | 6/12 | 6/12 | 6/12 |
| dcsbm_ps | 0/12 | 3/12 | 3/12 | 6/12 | 9/12 |

The registered prediction was that the flip would occur on both SBM variants because the
mechanism does not require heterogeneous degrees. It does. **The kill criterion that would
have restated the boundary as a property of power-law benchmark graphs is not triggered.**

Two irregularities are recorded rather than smoothed. Plain SBM peaks at 12/12 at d≈49 and
falls to 4/12 at d≈220. dcSBM peaks at 9/12 at d≈25 and sits at 6/12 above it. The count is
not monotone in degree on either, unlike LFR.

## V2 — P2 is FALSIFIED, and in the opposite direction.

P2 predicted the transition would occur at a **higher** average degree under plain SBM than
under LFR, because a homogeneous degree sequence gives a degree-based or similarity-based
selector less signal at a given density. It occurs at **half** the degree.

| generator | deg_cv at d≈50 | transition (dQ) | transition (dAMI) | max dQ | max dAMI |
|---|---|---|---|---|---|
| LFR (exp_AA) | 1.367 | **49.4** | 24.6 | 0.0122 | 0.1522 |
| dcsbm | 1.387 | **24.6** | 24.6 | 0.0138 | 0.0377 |
| dcsbm_ps | 1.469 | **24.6** | 24.6 | 0.0183 | 0.1082 |
| sbm | 0.137 | **24.7** | 24.7 | 0.0121 | 0.0207 |
| sbm_ps | 0.136 | **24.5** | 11.2 | 0.0202 | 0.0798 |

Transition is operationalized as the smallest realized average degree at which the mean
L-Spar + Metis gain is positive at the best retention point.

## V3 — P3 is FALSIFIED as vacuous, and the heterogeneity hypothesis is excluded.

P3 predicted that the degree-corrected SBM transition would fall closer to the LFR value than
the plain SBM one does, identifying degree heterogeneity rather than generator family as the
operative variable. Plain SBM lands at 24.7 and degree-corrected SBM at 24.6. They are
indistinguishable while their degree CV differs by a factor of ten.

The 2x2 factorial says the same at every degree (`analysis_full.txt`, HETEROGENEITY AXIS).
Best L-Spar + Metis dQ across the four heterogeneity cells at d≈50 is 0.0121 / 0.0073 /
0.0097 / 0.0075, with no consistent ordering; at d≈100 it is 0.0096 / 0.0151 / 0.0138 /
0.0127.

Rank correlations over the 60 generator-by-degree cells, with leave-one-out jackknife:

| pair | Spearman | jackknife |
|---|---|---|
| deg_cv vs dQ | **-0.260** | [-0.293, -0.231] |
| d_avg vs dQ | **+0.652** | [+0.636, +0.693] |
| deg_cv vs dAMI | -0.068 | [-0.102, -0.022] |
| d_avg vs dAMI | +0.385 | [+0.354, +0.423] |

Heterogeneity is not merely a weak correlate of the gain; its sign is negative. **The
DESIGN's promotion criterion ("if P2 and P3 both hold, degree heterogeneity is promoted from
a suspected confound to a named axis") does not fire, and the evidence runs the other way.**
Within this generator family, average degree is the strong correlate.

## V4 — P4 HOLDS. No honest Leiden modularity gain on any SBM variant.

Fifteen of 240 Arm A cells beat the compute-matched control by more than twice the baseline
seed standard deviation. **Zero of 240 survive both that control and the stricter best-of-5
restart baseline.** The largest cell against best-of-5 is +0.0053 on plain SBM at d≈11
against a seed sd of 0.0072. Kill direction 2, which would have required a generator-scoped
exception to the negative half of the boundary, does not fire.

## V5 — A nuance that resists V3: seed-robust gains occur only where there is heterogeneity.

Of 540 Metis option-seed cells (10 seeds; worst case = worst sparsified run minus best
baseline run), 42 are positive in the worst case. Every one of them is on `dcsbm`,
`dcsbm_ps` or `sbm_ps`. **None is on plain SBM.** Degree heterogeneity does not move the
transition location, but it may govern whether a gain past that location survives the
partitioner's own run-to-run variation. This is an observation from a subset, not a
pre-registered test, and it needs its own design before it becomes a claim.

---

## Findings with pointers

### 1. Arm A recovery and the granularity control (`results_armA_*.csv`)

Across 684 usable Arm A cells, AMI exceeds the unsparsified baseline in 194 and exceeds the
resolution-matched partition of the original graph in 133 (19%). The granularity control
therefore removes about a third of the apparent recovery gains here, where on LFR (exp_AA)
it reversed 21 of 29. The residue is concentrated at mild retention and high degree. This is
weaker than exp_AA's result in the same arm and should be characterized before any statement
about Leiden recovery on SBM enters the paper.

### 2. Chance floors are at zero, so AMI here is uninflated (`analysis_full.txt`, sanity block)

`AMI_chance` and `ARI_chance` are 0.0000 to 0.0002 in every generator-degree cell, against
planted labels. Artifact III is not operating on these graphs, unlike the real-network
recovery measurements where the floor moves with $k$.

### 3. Granularity blow-up reproduces on SBM

Leiden returns far more communities on the sparsified graph: on plain SBM at d≈11, 20 on the
original against 2,069 on the sparsified; at d≈220, 94 against 77. The mechanism behind
Artifact II is present on SBM exactly as on real graphs, which is what makes the fixed-$k$
arm necessary.

### 4. Cost on these graphs is NOT comparable to exp_AA or to real networks

End-to-end pipeline speedups here are 1.43x to 2.26x, and detection-only speedups 1.56x to
4.71x, i.e. the pipeline is *faster* than the baseline. exp_AA reports 0.58x at d=200 on LFR
with exact Jaccard, and the real-network sweeps report 0.87x to 1.35x. The discrepancy is not
explained and these numbers must not be quoted as speed evidence until it is. Note also that
the compute-matched control bought `n_restarts` = 1 in nearly every cell here, so the Arm A
comparisons rest on the best-of-5 baseline rather than on the matched one.

---

## Caveats

C1. **The transition figure depends on its definition.** At d≈25 the positive gains are small
(0.0046 to 0.0125) even where a majority of cells are positive. A stricter rule, for instance
requiring positivity in 8 of 12 cells, would move plain SBM's transition to d≈49 and leave
dcSBM at 24.6. The qualitative conclusion of V3, that heterogeneity does not order the
transitions, is unchanged under either rule; the specific value 24.6 is not robust to it.

C2. **The SBM graphs are easier than the LFR graphs at matched mixing.** Baseline Metis AMI at
d≈25 is 0.887 on dcSBM and 0.969 on plain SBM, against 0.586 on LFR. Degree and mixing were
calibrated per cell to exp_AA's realized values (realized mu 0.577 to 0.676 across both
families), so this is a difference in what the generators build at the same mixing parameter,
not a calibration failure. It is the leading candidate explanation for the LFR-versus-SBM
transition gap and it is not tested here.

C3. Three graphs per cell; the correlations in V3 are over 60 cell means, not 720 rows.

C4. The 42 worst-case-positive cells in V5 come from a 540-row subset run at three retention
targets, not from the full grid.

C5. `analyse.py` was run on Fuji because the local environment has no pandas; exp_AA's
`results_armA.csv`, `results_armB.csv`, `lfr_generation.csv` and `metis_noise.csv` were copied
there so the LFR reference is recomputed from exp_AA's own data rather than from its SUMMARY.

C6. The LFR reference is exp_AA's mu = 0.5 arm (realized mu 0.579 to 0.672), chosen because
Exp AC's mixing was calibrated to it.

---

## Effect on claims

- **C19 (the boundary) is STRENGTHENED.** The fixed-$k$ gain and the free-granularity negative
  both reproduce on a second generator family. The referee question "is this an LFR artifact?"
  is answered with a pre-registered experiment.

- **The introduction's B6 must change.** It currently reads "Degree heterogeneity and the noise
  present in how a graph was collected are further correlates of that quantity, and we do not
  separate them here." Degree heterogeneity has now been separated and is not a correlate; its
  rank correlation with the gain is -0.26 while density's is +0.65. The sentence should record
  the exclusion, which is stronger than the hedge it replaces.

- **A third transition location is available.** LFR ≈ 50, SBM ≈ 25, real networks at or below
  29. Three generator families, three values, which supports stating a condition rather than a
  threshold more firmly than two did.

- **New open question, not a claim.** Why LFR's transition sits at twice the SBM value. Neither
  degree heterogeneity nor community-size heterogeneity accounts for it (sbm_ps behaves like
  sbm). Baseline detectability at matched mixing is the candidate (C2) and would need its own
  pre-registered test.

- **Nothing here licenses a speed statement** (see finding 4).
