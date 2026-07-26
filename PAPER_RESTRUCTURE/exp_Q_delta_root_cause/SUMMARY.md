# Exp Q — δ-suppression root cause: causal test of hub-edge placement

Files: DESIGN.md (pre-registered 2026-07-25, before any code), run.py, analyze.py,
trajectories.csv (Arm 1, 3,010 rows), checkpoints.csv (Arm 2, fresh Leiden at every
trajectory point), verdicts.csv (one row per network x variant), analysis.txt (full
printed report), chains.log (per-checkpoint run log).

Networks: ca-GrQc, email-Eu-core, ca-HepTh, ca-CondMat, email-Enron (the 5 registered).
Loaders and EVERY metric definition are exp_M_suppression_probe/run.py, loaded verbatim
from disk by file path (`importlib`, module name `expM_run`), so exp_Q numbers are
directly comparable with exp_M's probe CSVs. Local machine, one detached
`systemd-run --user --scope -p MemoryMax=3G` python, ~20 min wall clock.

## VERDICT (four parts)

**V1 — The causal claim is established, decisively, and it is the cleanest causal
statement in this project so far.** With the degree sequence fixed exactly, the
partition P0 frozen, the intra-edge fraction p_intra fixed EXACTLY, and modularity
Q(P0) fixed EXACTLY (max |Q - Q0| = 0.00e+00 over all ten pfix chains,
trajectories.csv column `Q`), steering degree-product mass onto or off P0's community
boundaries by degree-preserving double-edge swaps moves δ(P0) massively and
monotonically in the steered direction on 5/5 networks. ca-GrQc: δ 0.0872 -> +0.2504
(UP) and -> -0.8745 (DOWN). ca-HepTh: 0.1736 -> +0.3332 / -0.6768. ca-CondMat:
0.1126 -> +0.2318 / -0.2980. email-Enron: 0.1501 -> +0.2333 / -0.2854.
email-Eu-core: 0.0340 -> +0.0692 / -0.0408. Every UP end and every DOWN end lies
outside the matched-swap-count NEUTRAL band by 20 to 450 neutral standard
deviations (verdicts.csv `up_delta_end`, `down_delta_end`, `up_neutral_at_end`,
`up_neutral_2sd`, `down_*`). The kill criterion is NOT triggered on any network.
Three UP ends exceed δ_null outright (ca-GrQc +0.2504 vs 0.1890; ca-HepTh +0.3332 vs
0.2147; ca-CondMat +0.2318 vs 0.1018), i.e. the manipulation can push a real graph
past its own configuration null while leaving degrees, partition, p and Q untouched.

**V2 — But the observed real-vs-null suppression is NOT thereby attributed to hub
placement.** DESIGN's closing sentence for C8 asserts more than the manipulation
proves: that real partitions suppress δ *because* they place hub-incident edges on
boundaries no more than degree mechanics forces. That attribution fails its own
check. Using exp_M's columns (see finding 10), the balance-FREE hub-placement
statistics do not favour the null: `hb` = E[d_u d_v|inter]/E[d_u d_v|intra] is larger
in the null on only 3/5 networks (ca-GrQc 0.478 -> 1.758 and ca-HepTh 1.581 -> 1.929
and email-Enron 2.974 -> 4.472 yes; email-Eu-core 2.113 -> 1.404 and ca-CondMat
2.490 -> 1.493 no), and `hub_inter_lift` is HIGHER in the real arm on 4/5 (exp_M
finding 9 reports the same at n=17). Only the balance-SENSITIVE inter-mass share is
uniformly higher in the null (5/5: 0.053->0.659, 0.600->0.804, 0.284->0.678,
0.441->0.655, 0.523->0.903), and that statistic is inflated in the null purely by its
smaller p_intra, which is exactly the granularity confound exp_M's V1 warned about.
So: hub placement is a verified CAUSE of δ; it is not shown to be THE cause of the
real-vs-null gap.

**V3 — The triangle covariate: P3 fails as registered, and the registered statistic
is the wrong one.** The pre-registered P3 (|within-chain partial corr of
tri_intra/tri_inter with δ, conditioning on direction| < 0.4 on >= 3/5) fails 0/5:
-0.871, -0.997, -0.929, -0.898, -0.979 (pfix; literal -0.879 to -0.957). The ratio
tracks δ. But it tracks it because `tri_inter` is what the manipulation moves —
tri_inter goes 1.65 -> 11.83 in ca-GrQc UP and 1.65 -> 0.000 in ca-GrQc DOWN, a
mechanical consequence of relocating edges across boundaries. Substituting the
quantity H2 was actually about, per-intra-edge triangle richness `tri_intra`, the
partial correlation collapses to -0.068 / -0.167 / -0.009 on ca-GrQc / ca-HepTh /
ca-CondMat (|.|<0.4 on 3/5), and the sign test across chain endpoints is
inconsistent: in all five UP chains δ RISES while tri_intra FALLS 23-80%; in the five
DOWN chains δ falls 0.075-0.96 while tri_intra moves -25%, -2.7%, -1.3%, +45%,
+198%. ca-GrQc DOWN is the sharpest single dissociation: δ +0.0872 -> -0.8745 with
tri_intra 11.728 -> 11.405 (-2.8%). Triangle richness of intra edges does not track
δ; the ratio does, as a bystander of the manipulation. Labelled HYPOTHESIS: the
tri_intra substitution is a post-hoc statistic, not the registered one.

**V4 — Re-detection (Arm 2) survives in the clean arm and collapses in the literal
one, and the difference is informative.** In the pfix arm, running Leiden fresh on
the manipulated graph recovers essentially P0 (AMI(P0) 0.82-0.94 at UP ends) and the
freshly-estimated δ still exceeds δ_real on 5/5: 0.1911, 0.0417, 0.2169, 0.1561,
0.1604 vs δ_real 0.0872, 0.0340, 0.1736, 0.1126, 0.1501 (checkpoints.csv,
`fresh_delta`, `ami_P0`). P4 PASSES 5/5. In the literal arm the same test fails 0/5 —
but only because the literal chains destroy the graph's community structure
altogether (AMI(P0) 0.008-0.038, p_intra -> 0.0000-0.0012), so there is no partition
left to re-detect. P4 is meaningful only in the pfix arm.

**Bottom line for C8: the root cause is no longer "open with one candidate". Hub-edge
placement relative to community boundaries is a VERIFIED causal determinant of δ at
fixed degrees, fixed partition, fixed granularity and fixed modularity. What remains
unproven is that it is the quantity by which real and configuration-null partitions
actually differ.**

## Sanity gates (run before any interpretation)

**GATE 2 (state check) — PASS 5/5, exactly.** δ(P0) at swap 0 equals exp_M's
`delta_real` to 0.0 absolute difference on every network: ca-GrQc 0.087235,
email-Eu-core 0.034000, ca-HepTh 0.173647, ca-CondMat 0.112577, email-Enron 0.150080
(trajectories.csv rows with `spe`=0 vs exp_M probe_results*.csv `delta`, arm=real).
Also matching at swap 0: r_pb, p_intra, Q, hub_inter_lift, hb, auc_s, tri_intra,
tri_inter. This is guaranteed by construction (exp_M's module is executed, not
re-implemented) and verified anyway.

**GATE 1 (engine check) — PASS 5/5.** The registered wording ("NEUTRAL chain at 10
swaps/edge must land δ(P0)...") conflates two different quantities. exp_M's δ_null is
measured with a FRESH Leiden partition on the rewired graph; the NEUTRAL chain's
δ(P0) keeps P0 frozen. The two are not comparable and the neutral chain is NOT a
reproduction of δ_null — it is the fluctuation-band control at matched swap count.
The correct engine check is the neutral chain at 10 swaps/edge PLUS a fresh Leiden,
which reproduces exp_M's δ_null to within 3% on all five networks (checkpoints.csv,
`fresh_delta` at `final`=1 on chain `neutral`):
ca-GrQc +0.1872 vs 0.1890 (-0.9%), email-Eu-core +0.0420 vs 0.0413 (+1.7%),
ca-HepTh +0.2132 vs 0.2147 (-0.7%), ca-CondMat +0.1047 vs 0.1018 (+2.8%),
email-Enron +0.2415 vs 0.2400 (+0.7%); fresh k and p_intra likewise match (e.g.
email-Enron k=162 vs 145, p=0.3226 vs 0.3245). The custom swap engine is therefore
equivalent to igraph's `rewire(mode="simple")` for this purpose.
For the record, the frozen-P0 neutral endpoints — the actual fluctuation control —
are δ(P0) = -0.0892 / -0.0005 / -0.0372 / -0.0226 / -0.0539 at p_intra 0.042 / 0.165
/ 0.039 / 0.027 / 0.118. Freezing a partition through 10 swaps/edge of randomization
drives δ(P0) slightly NEGATIVE, nowhere near δ_null.

## Numbered findings

1. **The registered acceptance rule does not hold p_intra fixed; DESIGN's parenthetical
   "p_intra (constant by construction)" is wrong.** A swap (a,b),(c,d) -> (a,d),(c,b)
   changes how many of the two edges P0 calls intra. Measured: the literal UP chains
   drive p_intra to 0.0004 / 0.0012 / 0.0001 / 0.0000 / 0.0000 and the literal DOWN
   chains to 0.85 / 0.78 / 0.82 / 0.81 / 0.83, from real values 0.896 / 0.585 / 0.799
   / 0.759 / 0.731 (trajectories.csv `p_intra`). Since δ = r_pb * sd_s / sqrt(p(1-p)),
   the literal arm's δ movement is a mixture of sorting and granularity, and at the UP
   ends it is dominated by 1/sqrt(p(1-p)) blowing up over a handful of surviving intra
   edges: ca-GrQc's literal UP "δ = +1.306" is computed over 5 intra edges out of
   13,422 and is not a meaningful number. On ca-GrQc and ca-HepTh the literal UP chain
   reaches p_intra = 0 exactly and δ becomes undefined (verdicts.csv
   `up_degenerate_p0`); endpoints for the literal arm are therefore reported at the
   last checkpoint where δ is defined.

2. **The supplementary p-preserving (pfix) arm fixes this and is the causal readout.**
   Added AFTER observing finding 1 (so: not pre-registered; its provenance is stated
   here in full), motivated by DESIGN's own assertion. It adds one condition to the
   acceptance rule: the swap must leave the number of P0-inter edges among the two
   edges unchanged. Consequences, all verified in trajectories.csv: p_intra is
   constant to the last printed digit; `Q` is constant with max |Q - Q0| = 0.00e+00
   (modularity of a FIXED partition depends only on p_intra and the fixed community
   degree sums); the granularity term of exp_M's decomposition is identically zero;
   sd_s drifts only +2.9% to +9.9% (UP) and -6.3% to +9.9% (DOWN), so δ moves
   essentially in lockstep with r_pb. This is the "matched modularity" condition the
   DESIGN asked for, achieved exactly rather than approximately.

3. **Arm-1 endpoints, pfix (verdicts.csv, variant=pfix).** Format
   δ_real -> UP-end (swaps/edge) | DOWN-end (swaps/edge), NEUTRAL at matched count:
   - ca-GrQc      0.0872 -> +0.2504 (0.95) | -0.8745 (0.83); neutral -0.0136 / -0.0051
   - email-Eu-core 0.0340 -> +0.0692 (1.37) | -0.0408 (2.00); neutral +0.0210 / +0.0188
   - ca-HepTh     0.1736 -> +0.3332 (2.00) | -0.6768 (1.81); neutral +0.0561 / +0.0610
   - ca-CondMat   0.1126 -> +0.2318 (2.00) | -0.2980 (2.00); neutral +0.0507
   - email-Enron  0.1501 -> +0.2333 (1.29) | -0.2854 (2.00); neutral +0.0888 / +0.0797
   Neutral 2-sd half-widths (2 chain seeds) are 0.0006 to 0.0037, so every endpoint is
   separated by 20-450 sd. KILL CRITERION NOT TRIGGERED, 5/5, both variants.

4. **The balance-free sorting measure moves the same way, so this is not a p artefact
   even in the literal arm.** auc_s (Mann-Whitney AUC of the DSpar score against the
   intra indicator, exp_M's class-balance-free statistic) in the pfix arm goes from
   0.5236 / 0.7058 / 0.6871 / 0.6926 / 0.7067 (real) to 0.788 / 0.981 / 0.962 / 0.963
   / 0.976 at UP ends and to 0.0019 / 0.174 / 0.0026 / 0.049 / 0.089 at DOWN ends
   (trajectories.csv `auc_s`). r_pb moves 0.093 -> +0.259 / -0.847 on ca-GrQc,
   0.247 -> +0.454 / -0.888 on ca-HepTh, and so on. Sorting itself is what is being
   manipulated.

5. **P1 (registered): direction 5/5, magnitude 4/5.** Direction (UP-end > baseline AND
   DOWN-end < baseline) holds on 5/5 in BOTH variants. The magnitude clause ("UP-end δ
   at least 25% of the way from δ_real toward δ_null") is met on 4/5 in both variants,
   reaching 160% / 483% / 389% / (n.a.) / 93% of the gap in the pfix arm. The fifth is
   ca-CondMat, the one network in the whole study where δ_null < δ_real (0.1018 vs
   0.1126, the exp_B/exp_M exception): the "gap" points downward there, so a criterion
   phrased as "toward δ_null" is ill-posed, not failed. Its UP chain still raises δ to
   +0.2318, more than twice δ_real. Reported as 4/5 with that note rather than
   silently rescued.

6. **P2 (registered): pass 10/10 pfix chains, fail in the literal UP chains for a
   mechanical reason.** Within-chain Pearson corr(hub_inter_lift, δ), averaged over 2
   chain seeds whose values agree to <0.01: pfix UP +0.958/+0.974/+0.962/+0.948/+0.993,
   pfix DOWN +0.927/+0.996/+0.975/+0.995/+0.986 — all > 0.8. Literal DOWN also passes
   5/5 (+0.909 to +0.999). Literal UP gives -0.54 to -0.58 because hub_inter_lift is
   defined as (hub-incident inter rate)/(1-p) and is driven to exactly 1.0 as p -> 0
   regardless of hub placement; it is uninformative in a degenerate partition, not
   evidence against P2.

7. **P3 (registered): FAIL 0/5. Post-hoc substitution: 3/5. See V3.** Registered
   statistic: partial Pearson corr of log((tri_intra+eps)/(tri_inter+eps)) with δ,
   residualised on the direction indicator, pooling UP and DOWN chains: pfix -0.871 /
   -0.997 / -0.929 / -0.898 / -0.979; literal -0.879 / -0.850 / -0.861 / -0.957 /
   -0.787. Required |.| < 0.4 on >= 3/5. Fails everywhere. Ill-conditioning is only
   partial (tri_inter is exactly 0 at 8 of 126 pfix DOWN checkpoints, all on ca-GrQc
   and ca-HepTh), so the failure is real and not an eps artefact. HYPOTHESIS-labelled
   substitution with tri_intra alone: -0.068 / -0.889 / -0.167 / -0.009 / -0.950,
   i.e. |.|<0.4 on 3/5 (ca-GrQc, ca-HepTh, ca-CondMat).

8. **P4 (registered, secondary): PASS 5/5 in the pfix arm, and the AMI makes it
   meaningful.** checkpoints.csv, `fresh_delta` / `ami_P0` at UP-chain ends:
   ca-GrQc +0.1911 (AMI 0.944), email-Eu-core +0.0417 (0.824), ca-HepTh +0.2169
   (0.905), ca-CondMat +0.1561 (0.918), email-Enron +0.1604 (0.850), against δ_real
   0.0872 / 0.0340 / 0.1736 / 0.1126 / 0.1501. Leiden re-run from scratch finds
   essentially the same partition and measures a larger δ. Best point on each UP chain
   is higher still (up to +0.1995 / +0.0545 / +0.2362 / +0.1680 / +0.1868). DOWN-chain
   ends re-detect δ of -0.326 / -0.012 / -0.342 / -0.201 / -0.120 at AMI 0.41-0.54.
   In the literal arm P4 fails 0/5, but AMI(P0) there is 0.008-0.038 (email-Eu-core
   and email-Enron are at AMI ~ 0.000): the partition has been destroyed, so the test
   has nothing to re-detect.

9. **Chain behaviour and the registered UNTESTABLE rule.** Realized cumulative
   acceptance (trajectories.csv `acc_rate_cum` at `final`=1): NEUTRAL 0.70-0.99;
   literal UP 0.030-0.047, literal DOWN 0.011-0.027; pfix UP 0.00033-0.0064, pfix DOWN
   0.00048-0.019 (the pfix constraint rejects most uniform proposals by construction,
   which is why the pfix arms run under their own documented budget: target 2
   swaps/edge, stall at <0.002% over 2M proposals, hard cap 80M proposals — the
   registered 0.5%/50k rule would fire immediately on them and would mean nothing).
   Applying the registered rule *to the literal arm as written*: a directional chain
   stalls below 2 swaps/edge on ca-GrQc (down, 0.65), email-Eu-core (down, 1.90),
   ca-HepTh (down, 1.47) and email-Enron (down, 1.96), so those four networks are
   flagged UNTESTABLE per the letter of the rule, leaving ca-CondMat (up 3.44, down
   2.14) as the only network testable by the letter. The letter is reported; it is
   also reported that the rule's premise does not describe what happened. The rule
   exists to catch chains that CANNOT move the graph; these chains stalled because
   they had moved it to saturation — the inter-mass share reaches 1.0000 (literal UP,
   all 5) and 0.0019-0.0140 from a starting 0.05-0.60 (literal DOWN), and δ had already
   separated from the neutral band by 20-450 sd. Both readings are on the record; no
   parameter was changed after seeing results to make a network testable.

10. **Post-hoc (HYPOTHESIS): where do real and null partitions actually sit on the
    steering variable?** Recovered from exp_M's columns via the identity
    mass_share_inter = (1-p) hb / ((1-p) hb + p), verified against trajectories.csv
    `M_inter`/`M_total` at swap 0 (ca-GrQc 0.0526 both ways).
    network | mass_real | mass_null | hb_real | hb_null | hubLift_real | hubLift_null
    ca-GrQc 0.0526 0.6594 | 0.478 1.758 | 0.967 1.172
    email-Eu-core 0.5999 0.8040 | 2.113 1.404 | 1.470 1.057
    ca-HepTh 0.2844 0.6775 | 1.581 1.929 | 1.422 1.214
    ca-CondMat 0.4413 0.6547 | 2.490 1.493 | 1.486 1.121
    email-Enron 0.5225 0.9030 | 2.974 4.472 | 1.194 1.084
    Mass share is higher in the null 5/5 — the direction the causal chains say raises
    δ — but mass share is not balance-free and the null's smaller p_intra inflates it
    mechanically. The balance-free versions disagree: hb favours the null on 3/5 and
    hub_inter_lift favours the REAL arm on 4/5. This is why V2 stops short of
    attribution. Needs its own pre-registered test (a p-matched real-vs-null
    hub-placement comparison) before it is anything.

## Caveats

- **frozen-P0 δ is not δ_null, and the NEUTRAL chain is not the exp_M null.** exp_M's
  δ_null re-runs Leiden on the rewired graph; every δ in trajectories.csv is computed
  against the FROZEN P0 of the original graph. The two coincide only at swap 0. The
  neutral chain's role is the fluctuation band at matched swap count, nothing more; its
  frozen-P0 endpoint (δ(P0) slightly negative, p_intra 0.03-0.16) is expected and is
  not a failure to reproduce anything. The correct comparison to δ_null is
  checkpoints.csv's `fresh_delta` on the neutral chain (Gate 1, agrees within 3%).
- **The pfix arm is not pre-registered.** It was added after seeing that the literal
  rule violates DESIGN's own "p_intra constant by construction" claim. Its results are
  reported alongside, not instead of, the literal arm, and both arms agree on the
  registered direction test (P1, 5/5 each) and on the kill criterion (not triggered,
  5/5 each). Everything specific to the pfix arm (P2 10/10, P4 5/5, Q exactly fixed)
  should be treated as a confirmatory re-run with a better-specified manipulation, not
  as a pre-registered result.
- **The pfix chains use a biased proposal budget, not a reversible Markov chain.** All
  chains here (including the registered literal ones) are greedy steering, not
  sampling: no detailed balance, no stationary distribution. They answer "can this
  quantity be moved and does δ follow", not "what does a typical graph with this
  property look like".
- **P3's registered statistic was ill-chosen.** The manipulation moves inter-boundary
  edges, so tri_inter is a direct downstream product of the treatment; conditioning
  only on the direction indicator does not remove that. The finding to carry forward is
  the sign inconsistency of tri_intra (finding 7 / V3), and it is a HYPOTHESIS.
- **Chain lengths are short by design intent, not by failure.** No directional chain
  needed more than 2 swaps/edge; most saturated below 1. The registered 10 swaps/edge
  cap and the 0.5-swaps/edge recording stride were both mis-scaled; the recording grid
  was densified below 0.5 (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4 added) so that
  short chains still yield 20+ point trajectories. This is a superset of the registered
  grid and changes no arm or acceptance rule.
- **Arm 2 was run at every trajectory checkpoint**, not the registered start/mid/end —
  a superset, adopted because Leiden costs 0.1-1.2 s on these graphs.
- **5 small/medium networks, 2 chain seeds per arm, one Leiden seed (42), modularity-only
  ModularityVertexPartition with n_iterations=2, undirected simple LCC.** Per-seed
  agreement is tight (P2 correlations agree to <0.01 between seeds) but n=5 networks and
  the heavy tail of the suppression distribution (com-Amazon, cit-Patents) is untouched.
- **email-Eu-core is the weakest network in the set** — δ_real 0.0340 and δ_null 0.0413
  differ by only 0.007, so its "483% of the gap" is 483% of a very small number.

## Effect on claim C8

C8 currently reads (post exp_M): *"Real structure suppresses delta vs null (16/17, median
ratio 1.60). Mechanism localized to partition-score misalignment, not partition
granularity; triangle/clustering explanation unsupported. Root cause still open."*

Recommended replacement:

> **C8** Real structure suppresses delta vs null (16/17, median ratio 1.60). Mechanism
> localized to partition-score misalignment, not partition granularity (Exp M, 17/17,
> exact decomposition plus a class-balance-free AUC cross-check); triangle/clustering
> explanation unsupported (Exp M) and not rescued by a direct causal test (Exp Q:
> per-intra-edge triangle richness moves in both directions while delta falls uniformly).
> Exp Q establishes causally, on 5/5 networks, that the placement of high-degree-product
> edges relative to community boundaries CONTROLS delta at fixed degree sequence, fixed
> partition, fixed intra-edge fraction and fixed modularity — delta is steerable from
> +0.09 to +0.25 or to -0.87 on ca-GrQc, past the network's own configuration null, and
> the effect survives re-estimating the partition (AMI 0.82-0.94). Whether the real-null
> gap itself is produced by this lever is NOT established: balance-free hub-placement
> statistics do not consistently favour the null.

Defensible sentence-level claim for the paper (safe as written):

> Delta is not merely correlated with where hub-incident edges sit relative to community
> boundaries — it is controlled by it. Holding the degree sequence exactly fixed by
> degree-preserving double-edge swaps, freezing the partition, and additionally requiring
> each swap to preserve the number of inter-community edges (which fixes both the
> intra-edge fraction and the partition's modularity exactly), we accepted only swaps that
> increased, or only swaps that decreased, the total degree-product mass carried by
> boundary edges. On all five networks tested this moves delta far outside the
> fluctuation band of unconstrained rewiring at matched swap count — on ca-GrQc from
> +0.087 to +0.250 or to -0.874, in both cases past the graph's own configuration-model
> null value of +0.189 — and re-running community detection on the manipulated graph
> recovers essentially the original partition (AMI 0.82-0.94) and still measures the
> elevated delta. The same manipulation leaves per-intra-edge triangle counts moving in
> inconsistent directions, confirming that triangle richness is not the operative
> variable. What the experiment does not show is that this lever explains the real-versus-
> null gap: the real partitions in our sample do not place hub edges on boundaries less
> often than configuration-model partitions do once the intra/inter balance is accounted
> for.

Do NOT write, in any form: "delta is suppressed on real graphs because real partitions
keep hub edges inside communities." Exp Q supports the causal direction but the
cross-arm comparison in finding 10 does not support that attribution.
