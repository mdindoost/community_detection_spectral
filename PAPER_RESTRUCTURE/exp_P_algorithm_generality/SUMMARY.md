# Exp P — algorithm generality: DSpar and L-Spar under Infomap, Louvain and label propagation

Files: run.py, results.csv (63 cells), runs.csv (378 individual detection runs),
recovery.csv (284 rows), bestof.csv, giant_share.csv, trials_check.csv, analyze.py,
bestof.py, diagnostic_giant.py, trials_check.py. Logs: main_email-Eu-core.log,
main_small.log, main_large.log, recovery.log, bestof.log, diagnostic_giant.log,
trials_check.log.
Coverage: main arm 7/7 networks x 3 algorithms x 3 sparsifier arms (COMPLETE);
recovery 3/3 labelled networks x 3 algorithms, chance floors on every arm, Louvain
resolution-matched controls wherever |dk|/k > 25% (COMPLETE). Compute on Fuji.

## VERDICT (five parts)

**V1 — The paper's negatives are NOT Leiden artifacts. The reverse kill criterion is NOT
triggered; every verdict now carries four-algorithm scope.**
Registered reverse-kill: an algorithm showing controlled beyond-noise gains on >=3/7 networks
would kill the universal negatives. Maximum observed is 2/7 (label propagation), and it falls
to 1/7 under a stronger control. Against the pre-registered runtime-matched control
(`dQ_vs_matched`, results.csv): Infomap negative in **21/21** cells, Louvain negative in
**21/21**, label propagation negative in 10/21. Against best-of-N plain restarts (bestof.csv,
N=20 small / N=10 large — a strictly more generous baseline), **60/63 cells are negative**;
the only 3 positives are label propagation on email-Eu-core, where the baseline is pathological
(V3). Mean dQ vs best-of-N: Infomap -0.0820 (0/21 positive), Louvain -0.0463 (0/21),
label propagation -0.1056 (3/21).

**V2 — Artifact I (scoring on the sparse graph) is algorithm-independent and larger than the
Leiden-only evidence suggested.** `dQ_naive` (partition scored on the sparsified graph) is
positive in **60/63** cells, up to +0.5365; the same partitions scored honestly on the
ORIGINAL graph are positive in only 15/63, and beat their runtime-matched control in 11/63.
The sign flips between the two accountings in **45/63** cells (Infomap 14/21, Louvain 20/21,
label propagation 11/21). Reading modularity off the sparsified graph inverts the result under
every algorithm we tested, not just Leiden. (results.csv `dQ_naive` vs `dQ_honest_vs_mean`.)

**V3 — P2 is confirmed, but only after the baseline is given a fair number of restarts; the
raw ordering is inverted and the reason is a label-propagation failure mode, not a
sparsification benefit.** Under the registered runtime-matched control label propagation looks
like the BEST algorithm under sparsification (mean dQ_vs_matched +0.0093 vs Infomap -0.0783,
Louvain -0.0443) — the exact opposite of prediction P2. The control is at fault: it bought only
r=1..16 restarts (results.csv `n_restarts`) of a strongly multi-modal algorithm. Given 20
restarts, label propagation is the WORST of the three (mean dQ_bestN -0.1056, bestof.csv).
Mechanism (giant_share.csv): label propagation's baseline giant-cluster share is 1.0000 on
email-Eu-core, 0.9993 on wiki-Vote, 0.6551 on email-Enron, 0.2771 on ca-HepTh, and <=0.108 on
ca-CondMat / com-DBLP / com-Amazon. Sparsification "helps" exactly where the monster cluster
dominates, and only email-Eu-core resists all 20 restarts (best-of-20 Q=0.0792 against
Louvain's 0.4164 on the same graph). One network out of 63 cells, and it is a known algorithm
pathology being partially relieved — not evidence that sparsification improves detection.

**V4 — The com-Amazon anomaly (exp_L V3) is scoped, not promoted: it reproduces in the
modularity family only, and both non-modularity algorithms beat it outright without any
sparsification.** Pre-registered rule: promotion required reproduction under >=2 of the 3 new
algorithms. Observed 1 of 3. On com-Amazon at L-Spar retention 0.5418 (recovery.csv,
avgF1_ge3): Louvain 0.4023 +- 0.0004 vs its own baseline 0.1416 — a near-exact replication of
Leiden's 0.4019 in exp_L — and it survives an over-matched Louvain resolution control
(0.3190 at k=8,316; sweep 0.3326 / 0.3491 / 0.3604 at k=9,143 / 10,517 / 11,688, i.e. up to
35% MORE clusters than L-Spar's 8,655). But Infomap scores 0.4585 with L-Spar against its own
baseline 0.4646 (**-0.0061**), and label propagation 0.4469 against 0.4801 (**-0.0332**). The
sharpest reading: Infomap's and label propagation's *unsparsified* partitions (0.4646, 0.4801
at k>=3 of 15,367 and 22,480) both beat the best L-Spar+Louvain number. The com-Amazon
"anomaly" is Leiden/Louvain's coarse default granularity being partially repaired by
shattering, in a metric that rewards fine granularity — not a recovery capability that
sparsification adds.

**V5 — Cost: no algorithm buys speed at a quality-preserving retention.** Maximum end-to-end
`speedup_vs_single` among cells that preserve quality (dQ vs best-of-N >= -0.005) is **0.66x**
(n=4 cells). 12/63 cells exceed 1.5x, and every one of them loses modularity
(dQ_vs_matched -0.0153 .. -0.1618). The fastest cell in the grid — label propagation with
L-Spar on com-DBLP, 3.78x — costs -0.1303. Algorithm-only speedups (excluding sparsification
cost) are median 1.21x Infomap, 1.63x Louvain, 1.15x label propagation. P4 holds under all
three algorithms.

## Findings with pointers

### 1. Honest transfer, all 63 cells (results.csv)

Per-algorithm summary of `dQ_vs_matched` (sparse-arm mean Q_orig minus the best of as many
plain restarts as the pipeline's wall clock buys):

| algo | cells | negative | mean | median | worst cell | mildest cell |
|---|---|---|---|---|---|---|
| infomap | 21 | **21/21** | -0.0783 | -0.0582 | email-Enron dspar@0.5 -0.1806 | email-Eu-core dspar@0.8 -0.0002 |
| louvain | 21 | **21/21** | -0.0443 | -0.0292 | ca-HepTh dspar@0.5 -0.1239 | email-Eu-core dspar@0.8 -0.0040 |
| labelprop | 21 | 10/21 | +0.0093 | +0.0056 | com-Amazon dspar@0.5 -0.2093 | email-Eu-core lspar@0.5 +0.2279 |

Per-arm means of `dQ_vs_matched` (n=7 networks each):

| arm | infomap | louvain | labelprop |
|---|---|---|---|
| dspar@0.8 | -0.0296 | -0.0121 | +0.0247 |
| dspar@0.5 | -0.1113 | -0.0651 | +0.0089 |
| lspar@0.5 | -0.0941 | -0.0556 | -0.0058 |

Retention 0.8 is mild for Infomap and Louvain (-0.0002..-0.0582 and -0.0040..-0.0240) and
retention 0.5 is not: both sparsifiers cost 0.05-0.18 modularity at half the edges under every
algorithm. L-Spar is not better than calibrated DSpar at matched retention for Infomap
(-0.0941 vs -0.1113) or Louvain (-0.0556 vs -0.0651) — the differences are small and go the
same direction as Leiden's in exp_L.

### 2. The beyond-noise test and the reverse kill criterion (analyze.py)

Operationalisation of "beyond seed noise" used for P1 and the reverse kill, applied per cell:
`dQ_vs_matched > 0` AND `dQ_honest_vs_mean > 2 * pooled_sd` (pooled sd of the baseline and
sparse-arm Q_orig; column `pooled_sd` in results.csv).

| algo | networks with a beyond-noise gain | under best-of-N |
|---|---|---|
| infomap | 0/7 | 0/7 |
| louvain | 0/7 | 0/7 |
| labelprop | 2/7 (email-Eu-core, email-Enron) | **1/7** (email-Eu-core) |

Reverse-kill threshold is >=3/7 for some algorithm. NOT reached. email-Enron's label-prop cells
(+0.1460 and +0.1608 vs the runtime-matched control) flip to -0.0682 and -0.0534 once the
baseline is given 20 restarts (best-of-20 Q=0.5226 vs the matched control's 0.3084) — the
matched control had simply not sampled label propagation's good mode.

### 3. Recovery under the |dk|/k < 25% comparability rule (recovery.csv)

Only 7 of the 27 arm-cells are k-comparable, and all 7 are at DSpar retention 0.8. Every
L-Spar and DSpar-0.5 cell fragments the partition beyond the comparability rule, so per DESIGN
no recovery conclusion is drawn from them except through the granularity-matched controls.

| dataset | algo | arm | k_base | k_arm | \|dk\|/k | metric | base | arm | delta | chance floor |
|---|---|---|---|---|---|---|---|---|---|---|
| email-Eu-core | infomap | dspar@0.8 | 15.3 | 15.3 | 0.0% | AMI | 0.5233 | **0.6118** | **+0.0885** | 0.0029 |
| email-Eu-core | louvain | dspar@0.8 | 7.3 | 8.3 | 13.6% | AMI | 0.5468 | 0.5676 | +0.0208 | 0.0037 |
| com-DBLP | infomap | dspar@0.8 | 14292 | 15527 | 8.6% | avgF1_ge3 | 0.3260 | **0.3358** | **+0.0098** | 0.0868 |
| com-DBLP | louvain | dspar@0.8 | 220 | 202 | 8.5% | avgF1_ge3 | 0.1019 | 0.0660 | -0.0359 | 0.0118 |
| com-DBLP | labelprop | dspar@0.8 | 23260 | 27909 | 20.0% | avgF1_ge3 | 0.3822 | 0.3835 | +0.0013 | 0.1072 |
| com-Amazon | infomap | dspar@0.8 | 15367 | 17105 | 11.3% | avgF1_ge3 | 0.4646 | 0.4547 | -0.0099 | 0.0676 |
| com-Amazon | labelprop | dspar@0.8 | 22480 | 26366 | 17.3% | avgF1_ge3 | 0.4801 | 0.4729 | -0.0073 | 0.0797 |

Four positive, three negative; the only substantial one is Infomap on email-Eu-core. It
survives every check we ran (see finding 5): a 10-seed baseline (mean AMI 0.5398, best 0.5881,
against the DSpar arm's 0.6152), igraph's default `trials=10` (baseline 0.5825 vs DSpar 0.6379,
+0.0554 at k 14.7 vs 16.7), the k-comparability rule, and a chance floor of 0.0029.
**Status: this is the outcome of a pre-registered measurement, but no prediction covered it
(P1-P4 say nothing about recovery under Infomap outside com-Amazon). Per EXPLORATION rule 2 it
is reported as a measurement result that requires its own pre-registered follow-up before it can
become a claim** — one network with a strong effect, one with +0.0098, one negative.

### 4. The com-DBLP granularity artifact reproduces exactly under Louvain (recovery.csv)

| condition | seeds | k | k>=3 | avgF1_ge3 | gamma |
|---|---|---|---|---|---|
| baseline | 3 | 220.3 | 220 | 0.1019 +- 0.0050 | 1.0 |
| lspar_0.5 | 3 | 10947.0 | 10947 | 0.1930 +- 0.0001 | 1.0 |
| **resmatch_lspar_0.5** | 1 | 11159 | 11159 | **0.2996** | 576 |
| dspar_0.5 | 6 | 1684.3 | 1684 | 0.0979 +- 0.0023 | 1.0 |
| **resmatch_dspar_0.5** | 1 | 1657 | 1657 | **0.1246** | 40 |

Louvain's L-Spar number (0.1930 at k=10,947) is byte-comparable to Leiden's in exp_L
(0.1930 at k=10,948), and the resolution-matched control beats it by the same margin
(0.2996 vs exp_L's 0.2971). Artifact II is a property of the modularity objective's default
granularity, not of Leiden's optimiser.

The chance floors quantify the artifact directly: on com-Amazon under Louvain, a size-matched
random partition scores 0.0086 at k=231 and **0.0567 at k=8,651** — a 6.6x rise in the chance
level from fragmentation alone (recovery.csv `chance_*` rows). Any avgF1_ge3 comparison across
different k is reading that slope.

### 5. The one DESIGN deviation is conservative (trials_check.csv)

Exp P ran `community_infomap(trials=1)`; igraph's default is `trials=10`. Re-running the
Infomap cells at both settings on the five small networks:

| network | arm | trials=1 mean Q_orig | trials=10 mean Q_orig |
|---|---|---|---|
| email-Eu-core | baseline | 0.3475 | **0.4035** |
| email-Eu-core | dspar_0.8 | 0.4016 | 0.3993 |
| wiki-Vote | baseline | 0.3535 | **0.4152** |
| wiki-Vote | dspar_0.8 | 0.4167 | 0.4182 |
| ca-HepTh | baseline | 0.6781 | 0.6789 |
| ca-CondMat | baseline | 0.6437 | 0.6451 |
| email-Enron | baseline | 0.5311 | 0.5352 |

`trials=1` depresses the Infomap BASELINE (by 0.056 on email-Eu-core, 0.062 on wiki-Vote) and
leaves the sparse arm essentially unchanged. The deviation therefore makes sparsification look
BETTER than it is; with igraph's defaults Infomap's email-Eu-core dspar@0.8 cell goes from
`dQ_honest_vs_mean` +0.0553 to -0.0042. Every Infomap negative in V1 is strengthened, not
threatened, by restoring the default. The email-Eu-core recovery gain (finding 3) was
re-checked the same way and persists (+0.0554 at trials=10).

### 6. Cost decomposition (results.csv)

Median `speedup_vs_single` (whole pipeline vs one plain run of the same algorithm):
Infomap 0.99-1.38x by arm, Louvain 0.88-1.29x, label propagation 0.83-1.05x. The only cells
above 1.5x are on com-DBLP/com-Amazon (12/63), all of them quality-losing. `speedup_algo_only`
(detection time only, sparsification free) is median 1.21x / 1.63x / 1.15x — even discounting
the sparsifier entirely, halving the edges does not halve the work, because all three
algorithms return many more communities on the sparse graph (mean k_sparse/k_base 1.40 Infomap,
8.92 Louvain, 2.30 label propagation) and do more work per pass.

## Caveats

C1. **Scope limitation, stated up front (as DESIGN requires): no MCL, Metis or Graclus.**
python-igraph has no implementation of any of them, and MLR-MCL — the algorithm Satuluri 2011's
strongest claim is about — is therefore untested. Every statement here is scoped to the four
algorithms Leiden / Infomap / Louvain / label propagation. Infomap (map equation, flow-based) is
the nearest available relative of the MCL flow intuition, and it is the algorithm that loses
most, but this is not a substitute for testing MLR-MCL itself.
C2. DESIGN deviations, all recorded: (a) `INFOMAP_TRIALS=1` instead of igraph's `trials=10`, so
that one seed equals one restart and the runtime-matched control is meaningful across the three
algorithms — shown conservative in finding 5; (b) DSpar sparse graphs are detected UNWEIGHTED
(the Horvitz-Thompson weights are computed and discarded), following the paper's main protocol
and exp_K's verdict that the weights are an evaluation-correctness device — honest transfer
scores on the original unweighted graph regardless; (c) Louvain resolution-matched controls
were ADDED (DESIGN only required reporting k, but `community_multilevel` has a resolution knob,
so it was used wherever |dk|/k > 25%); (d) chance floors were computed for all three labelled
networks and all arms, not only com-Amazon; (e) the best-of-N control (bestof.csv) is a
post-hoc SUPPLEMENT added after seeing that the registered runtime-matched control bought only
r=1..16 restarts — it can only handicap the treatment, never manufacture a gain, and it is
reported alongside, not instead of, the registered control.
C3. The registered runtime-matched control is weak on the large networks: `n_restarts` = 1 in
most com-DBLP/com-Amazon cells. Every conclusion that depends on it has been re-checked against
best-of-N. Two cells would have read as gains on the registered control alone (ca-CondMat
label-prop, where the single restart drew Q=0.4192 against a 3-seed baseline mean of 0.6252;
and email-Enron label-prop) and both flip negative under best-of-N.
C4. Seeds per cell: baseline 3 (100-102), DSpar arms 2 sparsifier seeds x 3 algorithm seeds = 6,
L-Spar 3 (deterministic sparsifier, exp_L finding 6). Label propagation's spread is genuinely
bimodal on several graphs, so 3-seed sd understates its variance — this is why V3 rests on
best-of-20 rather than on sd.
C5. Louvain `resmatch` on com-DBLP/com-Amazon is n=1 seed per gamma (as in exp_L). The
com-Amazon over-matched sweep is 3 single-seed points; its sd across the sweep is 0.0114.
C6. Wall-clock numbers were measured on Fuji while a concurrent exp_M wiki-topcats job held the
load average at 33-48 on 24 cores. All timings are relative-only. Both arms of every comparison
were measured interleaved under the same conditions, so the speedup ratios are usable and the
absolute seconds are not.
C7. igraph 1.0.0 / Python 3.10 on Fuji. RNG: `ig.set_random_number_generator(random)` once at
import, then `random.seed(seed)` immediately before every detection call. Determinism verified
by `run.py selftest` (identical membership on a repeated seed for all three algorithms;
different membership on a different seed for Infomap and Louvain; exact reproduction after 1000
intervening foreign `random.random()` draws). Label propagation returns k=1 on the selftest
graph for both seeds, so its "different seed differs" check is vacuous there — its seed
sensitivity is instead documented on the real graphs in runs.csv.
C8. One retention target for L-Spar (0.5, as registered) and two for DSpar (0.8, 0.5). The
L-Spar retention floor documented in exp_L finding 2 is not probed here.

## Effect on claims

- **C5, C6, C9, C16 all gain four-algorithm scope.** "Sparsify-then-detect does not improve
  modularity under honest transfer scoring" is now verified for Leiden, Infomap, Louvain and
  label propagation across 7 networks and 63 controlled cells, with 60/63 negative against a
  best-of-N baseline. The referee question "is this just Leiden?" is answered.
- **Artifact I is upgraded from a Leiden observation to an algorithm-general one**: sparse-graph
  scoring is positive in 60/63 cells and flips sign against honest transfer in 45/63, under
  three additional algorithms (finding 2). This is the strongest single piece of evidence in the
  study for the paper's central methodological point.
- **The exp_L com-Amazon lead (C16 / exp_L V3) is DEMOTED to modularity-family scope.** It
  reproduces under Louvain (0.4023 vs an over-matched 0.3326-0.3604) and fails under both
  non-modularity algorithms, whose unsparsified baselines (0.4646, 0.4801) beat it. Wording for
  the paper: the com-Amazon effect is a granularity repair specific to the modularity objective's
  coarse default, not a recovery capability conferred by sparsification. Exp V's verdict on the
  same cell should be read together with this one.
- **P2's prediction survives, with a caveat worth a sentence in the paper**: label propagation is
  the most fragile of the three, but only measurable as such once the baseline gets enough
  restarts; under a single-restart baseline the same data say the opposite. This is a small,
  concrete example of the paper's own thesis (weak controls invert conclusions) landing on us.
- **Live lead, NOT a claim: Infomap recovery at mild retention.** DSpar-0.8 raises Infomap's AMI
  on email-Eu-core from 0.5233 to 0.6118 at identical k, above a 0.0029 chance floor, robust to
  restarts and to `trials=10`; com-DBLP shows a small same-signed effect (+0.0098) and com-Amazon
  a small opposite one (-0.0099). Unpredicted, so it needs its own pre-registration (suggested:
  retention sweep 0.9/0.8/0.7, more labelled networks, and whether the effect tracks Infomap's
  known resolution behaviour on dense small graphs) before it can be claimed. Note it coexists
  with a modularity loss of -0.0002 in the same cell — a second instance of the
  objective/recovery dissociation exp_L reported for com-Amazon.
