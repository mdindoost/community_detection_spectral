# Phase 2 — New abstract + intro draft (v1, 2026-07-24)

Working title (options):
1. **When Does Sparsification Help Community Detection? Separating Genuine Gains from Degree-Mechanical Artifacts**
2. Sparsification and Community Detection: Null Models, Measurement Artifacts, and the Limits of Modularity Gains
3. (keep-the-brand option) Less is More? Sparsification and the Illusion of Clearer Community Structure

## Abstract (draft v1)

> Graph sparsification is increasingly reported to not merely accelerate community detection but to
> improve it. We show that, for degree-based sparsification, the standard evidence for such
> improvements is largely an artifact of evaluation methodology — and we characterize the narrow
> conditions under which genuine gains exist. We first develop a fixed-partition theory of
> degree-based edge sampling: when intra-community edges receive higher DSpar scores than
> inter-community edges (positive separation, δ > 0), sparsification provably removes
> inter-community edges preferentially and increases the intra-community edge fraction. We then
> show empirically that this mechanism operates independently of community structure: on 17
> real-world networks, degree-preserving configuration-model nulls reproduce both δ > 0 and
> fixed-partition modularity gains on every network — typically exceeding the real graph (median
> ratio 1.24) — so neither signal distinguishes a network from its own null model. We further
> identify three evaluation artifacts that generate apparent improvements: (i) comparing modularity
> across different graphs — partitions optimized on sparsified graphs lose 0.01–0.19 modularity
> when evaluated on the original network, on all 15 networks tested; (ii) partition granularity —
> apparent ground-truth-recovery gains vanish against resolution-matched controls run on the
> unsparsified graph, on all eight labeled datasets tested; and (iii) unmatched computation —
> gains attributed to sparsification-seeded optimization disappear against runtime-matched restart
> baselines on most networks. After controlling for all three, a genuine benefit remains on 2 of 15
> networks, where seeding Leiden with the sparsified-graph partition and refining on the original
> graph beats runtime-matched baselines by small but robust margins (+0.008 modularity); no
> structural statistic we tested predicts where these gains occur. Our results yield a practical
> evaluation protocol — null-model, resolution-matched, and runtime-matched controls — for any
> claim that preprocessing improves community detection.

(~250 words; TNSE structured-abstract friendly; every number traceable to
PHASE1_VERDICT.md / exp_E / exp_F.)

## Introduction — skeleton (v1)

1. **Hook.** Sparsification as standard preprocessing for scale; growing claims (incl. our own
   earlier work-in-progress and the DSpar line) that it can *improve* downstream community
   detection, not just preserve it. One-sentence thesis: those claims mostly do not survive
   controlled evaluation, but something real and small remains — and both facts matter.
2. **The mechanism (real).** Hub-bridge intuition; DSpar scores; δ > 0; fixed-partition theory
   summary (Thms 1–3 retained). The mechanism is provable and measurable.
3. **The artifacts (the trap).** Three ways the mechanism *appears* to improve detection:
   sparse-graph self-scoring modularity, granularity, unmatched compute. One paragraph each with the control
   that exposes it. Key sentence: "a degree-biased sampler raises the fixed-partition modularity of
   *any* partition on *any* degree-heterogeneous graph — including one with no communities at all."
4. **The null-model result.** 17/17 networks: configuration-model rewiring reproduces both signals,
   usually more strongly. hb > 1 in every null; hb erratic in real graphs. Consequence: δ and hb
   are degree statistics, not community statistics. (Also: δ measured w.r.t. Leiden partitions vs
   planted/ground-truth partitions differs systematically — partition provenance.)
5. **What survives.** Runtime-matched seeded refinement: 2/15 networks, +0.008; no predictor among
   δ, δ*, hb, degree-CV, size (with the 14-network deg-CV correlation collapse as a cautionary
   case study directly relevant to prior small-n correlation claims, including r = 0.92 in an
   earlier version of this work).
6. **Contributions list.**
   (a) Fixed-partition theory of degree-based sampling (mechanism, incl. corrected sampler
       formulation with calibrated retention);
   (b) Null-model characterization: δ > 0 and ΔQ_fixed > 0 are configuration-model properties;
   (c) Three-control evaluation protocol (null / resolution-matched / runtime-matched) with
       open implementations;
   (d) Boundary result: existence proof of genuine gains (2/15) + demonstration that no tested
       statistic predicts them;
   (e) Sampler audit: nominal α vs realized retention across published DSpar variants
       (with-replacement collapse; clipping saturation).
7. **Roadmap paragraph.**

## Framing rules (agreed)

- Never present as self-correction melodrama; present as "the evaluation methodology this area
  lacks." Prior-draft errors appear once, factually, as motivation.
- Every artifact claim paired with its control and its number.
- "Improve community detection" appears only inside quotation marks or with explicit metric+graph
  qualification.
- Honest scope: modularity/Leiden/DSpar family; no claims about other sparsifiers beyond what was
  tested (uniform/degree/spectral baselines from draft Exp 4 to be redone or dropped).

## Next writing steps

1. User feedback on title + abstract v1 → v2.
2. Restructure skeleton: map every draft section/table/figure → keep / rewrite / drop.
3. Rewrite §Experiments around Exps B/C/D/E/F as the primary evidence; draft Exp 1–2 tables
   become the "mechanism" subsection.
