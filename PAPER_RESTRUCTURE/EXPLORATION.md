# EXPLORATION.md — contract for the exploration phase

Agreed by Mohammad Dindoost + Claude, 2026-07-25. This file is the constitution for the
free-rein exploration sessions (including any fresh session on Fuji, which will have NO memory
of prior chats — this repo is the single source of truth; start by reading this file, then
phase2_writing/STORY.md, then the exp_*/SUMMARY.md files).

## Purpose

Explore every idea that could make the paper better, ignoring page limits and writing concerns.
We are NOT finishing the paper during exploration; we are generating verified pieces. Assembly
into prose happens afterwards, story-first, per the discipline in STORY.md.

## Rules (non-negotiable — this project got burned twice by skipping them)

1. **Pre-registration.** Before ANY experiment runs, its folder gets a DESIGN.md stating:
   (a) which claim (STORY.md C-number, or a new numbered claim) the result could change;
   (b) explicit predictions, written before results are seen;
   (c) a kill criterion — the concrete outcome under which we declare the idea dead and stop.
   No DESIGN.md, no run.
2. **Post-hoc findings are hypotheses, never results.** Anything noticed after looking at data
   gets labeled HYPOTHESIS in the SUMMARY and needs its own pre-registered follow-up to be
   promoted. (History: r=0.92 draft claim; r=0.69→0.12 deg-CV collapse; tri_intra +0.74→+0.06.)
3. **Every numeric claim carries its CSV pointer** (file + column, row identifiable).
4. **Controls checklist** — each experiment states which apply and runs them:
   configuration-model null arm; resolution/granularity-matched control; chance-corrected
   metrics (AMI not NMI); runtime/cost matching for any speed or quality claim; leave-one-out
   jackknife on every correlation (report the interval, not just r).
5. **Verdict discipline.** Each experiment ends with SUMMARY.md: VERDICT up top, numbered
   findings with pointers, explicit caveats. Model: exp_K_weighted_regime/SUMMARY.md.
6. **Commit after every completed network/stage** (OOM/restart resilience — we lost work twice
   on 2026-07-25). Push to refactor_v2 (standing authorization from Mohammad).
7. **Memory/infra discipline.** ONE heavy python job at a time per machine. Detach with
   setsid/nohup (survives harness death). Cap RAM: `systemd-run --user --scope -p MemoryMax=NG`
   (fallback `ulimit -v`). All CSV writes incremental/append. Per-run logs. Check `free -g`
   before large graphs. Local machine = 14GB RAM (OOM-killed us once); Fuji = 62GB / 24 cores /
   Python 3.10 / disk was 98% full — check `df -h` before writing anything big.
8. **No .tex edits during exploration.** STORY.md: evidence-backed claim-table updates that
   record verified experiment verdicts are applied directly and flagged in the report
   (Mohammad, 2026-07-25: "why are you waiting for my sign off if it is worth doing");
   genuine narrative pivots (thesis/framing changes) still need his explicit sign-off.
9. **α=1.0 is the no-sparsification sentinel** in all of Mohammad's experiment configs.
10. **Scale to the claim.** Cheap version first; scale up only if the cheap version says the
    idea is alive. Kill criteria are honored — no "one more tweak" after a kill.

## In-flight work (state as of 2026-07-25 ~19:15, after the OOM kill)

- **Exp L (L-Spar, exp_L_lspar/)**: main arm COMPLETE 7/7 networks (results.csv); null arm
  COMPLETE 3/3 (null_arm.csv — headline: L-Spar's Jaccard signal is structure-aware, deltaJ
  +0.09..+0.18 real vs ~0/negative in config nulls; first sparsifier in the study whose signal
  the null does NOT reproduce). Quality verdict shaping up: dQ_vs_matched negative in all
  completed cells (−0.014..−0.40) — Satuluri 2011's quality claim does not survive controls.
  REMAINING: recovery stage incomplete (email-Eu-core done; com-DBLP baseline/lspar done,
  resmatch arms + com-Amazon pending). CRITICAL: com-DBLP avgF1 RISES 0.109→0.193→0.376 while
  k explodes 227→10,950→72,736 — must be granularity-controlled before any interpretation
  (avgF1 best-match favors fragmentation; see Artifact III). Also: com-DBLP lspar rows are
  byte-identical across Leiden seeds — L-Spar is deterministic; check seed-independence claim.
  Then write SUMMARY.md (verdict on Satuluri claim + structure-awareness + recovery + ~1x cost).
- **Exp M (δ-suppression, exp_M_suppression_probe/)**: SUMMARY.md committed (verdict: triangle
  hypothesis dead; suppression = partition–score misalignment/sorting, not granularity; run
  INCOMPLETE 11/17). Batch-2 completion state: validation vs old CSV PASSED on small networks
  (probe_results_batch2_small.csv); new rows with auc_s columns for com-Amazon, com-DBLP,
  com-Youtube in probe_results_batch2.csv. REMAINING: wiki-Talk, cit-Patents, wiki-topcats
  (the last two need Fuji's RAM), then recompute all correlations at n≈16 with jackknives +
  auc_s cross-check, then revise SUMMARY.md numbers. NOTE: on-disk run.py is newer than the
  original probe_results.csv (adds auc_s/auc_prod, 37 vs 35 fields) — never append new rows to
  the OLD csv; batch2 files use the new header.
- **Exp N (Enron anatomy, exp_N_enron_anatomy/)**: COMPLETE, SUMMARY.md committed. Gain
  survives adversarial granularity check; mechanism = 2.58x biased removal of hub-mediated
  boundary edges; no separate basin; compute claim must be phrased in expectation only.
- **STORY.md claims table updated directly (2026-07-25)**: C8 (exp_M mechanism), C9 (exp_N
  anatomy + exp_R iteration control), C6 (exp_O verdict), new C16 (exp_L), C17 reserved for
  exp_V. Remaining tex-side integration happens in the assembly phase after exploration.

## Experiment roster

### Tier 1 — closes real holes in this paper
- **Exp O — core-preservation / periphery-fragmentation** (Mohammad's hypothesis; design below).
- **Exp P — algorithm generality**: DSpar + L-Spar under Infomap, Louvain, label propagation
  (python-igraph has all three) on ~6 networks, honest-transfer + recovery protocol. Claim at
  risk: entire story is Leiden-only; Satuluri's claim was about other algorithms. Biggest
  remaining referee hole.
- **Exp Q — δ root-cause causal test**: triangle-preserving vs triangle-breaking double-edge
  swaps at matched modularity; direct test of hub_inter_lift hypothesis (exp_M V4). Claim: C8.
  GREENLIT by Mohammad 2026-07-25 ("we should do"); design AFTER exp_M heavy-tail correlations
  land (they decide whether hub_inter_lift enters the causal arms).
- **Exp R — iteration-matched Enron control**: plain Leiden n_iterations matched to pipeline
  cost (exp_N caveat C4). Small.
- **Exp S — com-Orkut feasibility demo** (117M edges, needs Fuji, dataset must be downloaded):
  memory-bound regime where edge reduction = feasibility, not speed. The one unconditional
  "sparsification pays" statement. BLOCKED on Fuji disk cleanup — Mohammad will say when.
- **Small threads APPROVED by Mohammad 2026-07-25** (do if they help the paper — they do):
  H-O1 leaf-shedding (pre-registered follow-up of exp_O, cheap, sharpens the exp_O negative
  into a degree/k-core statement) and a Youtube mini-anatomy (C9's second gain gets the exp_N
  treatment: statistical reality + granularity check + boundary-removal mechanism; run on
  Fuji, com-Youtube dataset needs rsync ~35MB). Queue after P/Q are in flight.
- **Exp T decision deferred until after Exp P** (Mohammad 2026-07-25).

- **Exp V — "spending the signal"** (greenlit, DESIGN.md pre-registered, commit ba85b8a):
  can L-Spar's verified structure-aware Jaccard signal be deployed for an honest gain —
  weighting (w=1+J, J-shuffle sharp null), seeding (sparse->refine), protected deletion —
  under full controls; adjudicates the com-Amazon anomaly. Constructive positive => new C17;
  negative => strengthened thesis. Prior art to position against: Khadivi 2011, Berry 2011,
  adaptive-modularity-weighting 2017 (their evaluations have the artifact shapes we audit).

### Tier 2 — raises the ceiling
- **Exp T — predictor at scale**: 100+ networks (Netzschleuder), statistics pre-registered
  BEFORE any results are read (list them in DESIGN.md and freeze). Claim: C10 becomes a real
  negative result or an actual predictor.
- **Exp U — modern-reduction taste**: 2-3 representatives (one coarsening, one
  effective-resistance/spectral, one k-core pruning) through the protocol. Seeds paper 2
  (IJCAI-24 survey gap: zero "community" occurrences in graph-reduction literature).

### Tier 3 — paper 2, do NOT fold into this paper
Full modern-reduction benchmark; δ-suppression theory; ABCD hub-bridging at 10^6; dynamic/
streaming amortization study.

## Exp O — pre-registered design (greenlit 2026-07-25)

**Hypothesis (Mohammad's, stated before any dedicated experiment):** under edge sparsification
the sparse-graph partition shatters into many small clusters, but these are loose/peripheral
connections; the CORES of the main communities are preserved. Fragments of support already in
hand: nc explosion (exp_L results.csv: ca-CondMat 55→503→5157; com-DBLP 227→10,950→72,736);
Exp N finding 4 (movers are periphery-tilted, hubs stay); Chen PVLDB 2024
"fragmentation-with-pruning".

**Claim it could change:** upgrades C5/C6's "slightly harmed / not improved" into a structural
statement — "sparsification preserves community cores and reorganizes hub-mediated boundaries
and loose periphery" — unifying with Exp N's mechanism. New claim number C16 if it survives.

**Design.** Networks: email-Eu-core, wiki-Vote, ca-HepTh, ca-CondMat, email-Enron (small five
first; com-DBLP, com-Amazon only after the small five pass sanity). Sparsifiers: DSpar
(calibrated sampler, retentions ~0.5 and ~0.8) and L-Spar (targets 0.5, 0.2; reuse
exp_L_lspar/run.py machinery). Seeds: 3 Leiden seeds per arm minimum.
Per node, computed on the ORIGINAL graph + its partition P0: embeddedness = fraction of
neighbors in own community; k-core index; degree. Then:
(1) **Agreement-by-embeddedness**: node-level agreement between P0 and sparse-graph partition
    P' (community-matched by Hungarian/plurality mapping), stratified by embeddedness decile.
(2) **Core cohesion**: for each P0 community ≥20 nodes, core = top-50%-embeddedness members;
    cohesion = largest fraction of the core landing in one P' community.
(3) **Fragment composition**: nodes in P' clusters of size <10 — their embeddedness decile
    distribution vs the graph's.
**Controls:** (a) config-null arm — same pipeline on degree-preserving rewired graph (its own
P0_null); structure-specificity check. (b) Resolution-matched control — partition the ORIGINAL
graph with resolution tuned to match P''s community count; compute (1)-(3) identically; the
hypothesis requires sparsification's fragmentation to be MORE embeddedness-selective than mere
resolution. (c) Chance correction: permutation baseline for agreement.
**Pre-registered predictions:**
P1: agreement rises monotonically with embeddedness decile; top-3-decile agreement >0.85 at
    retention 0.5 for DSpar.
P2: sub-10-node P' fragments are ≥2x over-represented in the bottom-2 embeddedness deciles.
P3: median core cohesion ≥0.8 at retention 0.5, both sparsifiers.
P4: null arm shows a materially flatter agreement-embeddedness gradient (top-minus-bottom
    decile gap at most half the real graph's).
P5: resolution-matched control does NOT reproduce the embeddedness selectivity of the
    fragmentation (its fragment composition is closer to uniform).
**Kill criterion:** if the agreement gradient is flat (top-3 minus bottom-3 decile gap <0.1) OR
the null arm reproduces the gradient within noise on ≥half the networks, the hypothesis is dead
and the observation stays a descriptive footnote. No rescue tweaks.
**Output:** PAPER_RESTRUCTURE/exp_O_core_preservation/{DESIGN.md,run.py,results CSVs,SUMMARY.md}.

## After exploration: SYNTHESIS, not writing (Mohammad's rule, 2026-07-25)

When the roster is done, the next phase is NOT tex assembly. It is a deliberate synthesis
pass over ALL the pieces — we know far more now than when the story was written. Re-ask,
with the full evidence map on the table:
1. Does the story still hold, or has it become a different (better) story? The thesis has
   already moved twice tonight: "sparsification doesn't help" -> "seeing communities is
   necessary but not sufficient" -> "even a verified signal buys nothing on the objective;
   what it can rarely buy is recovery, and the objective and recovery dissociate."
2. What NEW questions did the answers create? (Candidates already visible: why is com-Amazon
   the network-level exception — H4; what makes weight heterogeneity help Leiden — H3; the
   δ root cause if Q lands; the objective/recovery dissociation as a phenomenon in itself.)
3. Which claims are load-bearing vs decorative? Which experiments changed a conclusion vs
   confirmed one? What would we cut?
4. "Would we still start this project today?" — and what is the ONE sentence the paper
   exists to defend?
5. Only after that conversation with Mohammad: decide research-or-writing, and if writing,
   what the paper's spine is. Writing decisions are narrative pivots = his sign-off.

## Session-death recovery (quota end, OOM, restart)

Long runs are launched DETACHED and survive the death of any Claude session. To recover:
1. `git pull` (every completed network is committed+pushed immediately).
2. Read this file + the exp_*/SUMMARY.md files.
3. Check Fuji jobs: `ssh mayooran@100.88.245.65 'ls -lt md724/community_detection_spectral/PAPER_RESTRUCTURE/exp_M_suppression_probe/ | head; tail -5 md724/community_detection_spectral/PAPER_RESTRUCTURE/exp_M_suppression_probe/*.log'`
   (Tailscale SSH; if it prints a login.tailscale.com URL, Mohammad must click it — auth
   expires ~12h.) Copy finished CSVs back with scp, merge/dedupe, commit.
4. Check local detached jobs: `ls -lt PAPER_RESTRUCTURE/exp_*/` for fresh logs/CSVs; a run
   whose log stopped growing and whose process is gone (ps aux | grep run.py) died — resume
   from its incremental CSV (all runners append; never re-run completed rows).

**Standing authorization (Mohammad, 2026-07-25): run any needed experiments on Fuji.** Keep
its disk in mind (~8GB free); com-Orkut (Exp S) does NOT fit until Fuji is cleaned.

## Environment

- venv recipe: `python3 -m venv venv && pip install numpy scipy python-igraph leidenalg networkx
  scikit-learn pandas` (Fuji: Python 3.10, works; local 3.14, works).
- Datasets: `datasets/` (6.3GB, SNAP edge lists; loader code copied between exp run.py files).
- com-Orkut for Exp S: not downloaded yet (~1.7GB gz from SNAP).
- Repo: github.com:mdindoost/community_detection_spectral.git, branch refactor_v2.
