# Exp Q — δ-suppression root cause: causal test of hub-edge placement (pre-registered design)

Registered 2026-07-25 after exp_M's full-coverage correlations, BEFORE any Exp Q code/run.
Greenlit by Mohammad ("for Q we should do"). Target updated per exp_M's n=16 analysis: the
causal candidate is **hub_inter_lift** (the only non-circular, non-proxy jackknife survivor,
Spearman -0.635, and now known to be uncorrelated with r_real: Spearman +0.06), NOT triangles
(dead at full coverage: C Spearman -0.015; tri_intra Pearson +0.187). The triangle RATIO rides
along as a pre-registered coarseness-confounded covariate only.

**Claim it could change:** C8's "root cause still open". If hub-edge placement causally moves
δ at fixed degree sequence and fixed partition, C8 closes: real partitions suppress δ because
they place hub-incident edges across community boundaries no more than (or independently of)
degree mechanics, whereas configuration-model partitions are forced to track degree. If the
manipulation does not move δ, hub_inter_lift dies as mechanism and C8 stays open with one
fewer candidate.

## Core idea

δ = mean_s(intra) − mean_s(inter) depends only on (a) the score multiset {s(e)} and (b) which
edges the partition calls intra. Degree-preserving double-edge swaps preserve (a) EXACTLY.
Holding the partition P0 FIXED while swapping edges therefore isolates (b): we can causally
steer the placement of high-degree-product (hub) edges relative to P0's boundaries and watch
δ(P0) respond, with zero confounding from Leiden re-optimization, triangles-as-cause, or score
changes.

## Design

Networks (small/medium, local machine): ca-GrQc, email-Eu-core, ca-HepTh, email-Enron,
ca-CondMat (the 3 exp_M staged networks + 2 more; loaders verbatim from exp_M run.py).
P0 = Leiden(g, seed 42, n_iter=2) on the real graph — computed once, then FROZEN.

Arm 1 — targeted placement steering (the causal arm), per network:
  Degree-preserving double-edge swaps ACCEPTED only when they move hub-mass in the desired
  direction relative to frozen P0:
  - UP-chain: accept swaps that increase the total degree-product mass on P0-inter edges
    (push hub edges to boundaries); REJECT others.
  - DOWN-chain: the reverse (pull hub edges inside communities).
  - NEUTRAL-chain (control): accept unconditionally (standard rewiring) — reproduces the
    exp_M null trajectory for reference.
  Chain length: up to 10 accepted swaps/edge or until acceptance stalls (<0.5% acceptance over
  50k proposals); record trajectory every 0.5 swaps/edge: hub_inter_lift(P0), hb(P0), δ(P0),
  r_pb(P0), p_intra (constant by construction), tri_intra/tri_inter (covariate), realized
  acceptance rate. 2 chain seeds per direction.

Arm 2 — re-detection check (secondary): at 3 checkpoints per chain (start / mid / end), run
  Leiden fresh (seed 42) on the modified graph and record δ(P_fresh), Q, k — does the causal
  effect on δ(P0) survive when the partition is re-estimated? (This re-introduces the usual
  confounds; it is reported as secondary, not the causal readout.)

## Pre-registered predictions

- P1 (the causal claim): δ(P0) moves MONOTONICALLY with the steering — UP-chains raise δ(P0),
  DOWN-chains lower it — on >= 4/5 networks, with the UP-end δ(P0) moving at least 25% of the
  way from δ_real toward δ_null's level (exp_M values).
- P2: hub_inter_lift(P0) is the tracking variable: within chains, corr(hub_inter_lift, δ) > 0.8.
  (Near-mechanical if P1 holds; registered to quantify it.)
- P3: the triangle-ratio covariate does NOT track δ within chains once direction is conditioned
  on (|within-chain partial corr| < 0.4 on >= 3/5 networks) — triangles are a bystander.
- P4 (secondary): the re-detected δ(P_fresh) at UP-chain ends exceeds δ_real on >= 3/5 networks
  (the effect is not an artifact of freezing P0).

## Kill criterion

If UP and DOWN chains fail to separate δ(P0) beyond the NEUTRAL chain's fluctuation band
(±2 sd of the neutral trajectory at matched swap count) on >= 3/5 networks, hub-edge placement
is NOT the mechanism: report "hub_inter_lift killed as cause; C8 root cause remains open",
no rescue tweaks. If chains stall (acceptance < 0.5%) before reaching 2 swaps/edge on a
network, that network is reported UNTESTABLE, not as support.

## Output
PAPER_RESTRUCTURE/exp_Q_delta_root_cause/{DESIGN.md,run.py,trajectories.csv,checkpoints.csv,
SUMMARY.md}. Local machine, one detached capped python (MemoryMax=3G), ~1-2h.
