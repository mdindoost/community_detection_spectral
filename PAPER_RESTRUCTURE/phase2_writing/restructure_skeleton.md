# Restructure skeleton — v1 (2026-07-24)

**Title (decided):** When Does Sparsification Help Community Detection? Separating Genuine Gains
from Degree-Mechanical Artifacts
**Abstract:** v1 in `abstract_intro_draft.md` (approved direction; final polish at assembly).

Legend: KEEP = essentially verbatim · REWRITE = same material, new framing · NEW = written from
Phase-1 outputs · DROP = removed. Every NEW table/figure names its data source (already on disk).

---

## §1 Introduction — REWRITE (7-point skeleton in abstract_intro_draft.md)

## §2 Background — KEEP with fixes
- 2.1 Modularity: KEEP. (Draft L135–179)
- 2.2 Spectral sparsification: KEEP; fix complexity contradiction (L234 vs L269).
- 2.3 DSpar: KEEP; **add the missing score equation** s(e)=1/d_u+1/d_v (used before defined, L256);
  add one paragraph: nominal α vs realized retention (sampler audit teaser, forward-ref §4.1).
- ADD 2.4 Related work: Satuluri+ 2011 (local sparsification for clustering), Serrano+ 2009
  disparity filter, backboning (Coscia–Neffke), Guimerà+ 2004 (modularity of random graphs),
  Fortunato–Barthélemy 2007 (resolution limit), Vinh+ 2010 (AMI), Peel+ 2017 / Hric+ 2014
  (metadata vs ground truth), Liu+ 2023 (DSpar), Park+ / CM++ (well-connectedness).

## §3 Theory: The mechanism — KEEP with three repairs
- Defs 1–6, Assumption 1: KEEP. **Replace Def "DSpar Sparsification"** with the calibrated sampler
  p(e)=min(1, λs(e)), λ solved so Σp=αm (matches what we can actually run; exp_C/run.py).
- Thm 1 + Cor 1 + Remark 1: KEEP (proof unchanged under monotone p; add half-sentence on clipping).
- Thm 2, Lemma 1, Prop 2: KEEP.
- **Cor 2: fix hypothesis** — require weighted balance D_c^(s)=2m/k (or restate as approximate
  bound with imbalance error term).
- **Thm 3: restate** for the calibrated sampler; qualitative conclusion unchanged.
- §3.3 spectral connection: KEEP but trim ~40%; stays explicitly heuristic.
- **Reframe throughout**: theory predicts *mechanical* fixed-partition effects; explicitly states
  it makes NO claim that δ>0 indicates community structure (forward-ref §5 null results).
- Hub-Bridge Hypothesis: REWRITE as a *conditional mechanism premise*, not an empirical law;
  the empirical status (erratic hb in real graphs, hb>1 in all nulls) moves to §5.

## §4 The sampler audit — NEW (short section)
- Nominal α vs realized retention: with-replacement collapse (0.33–0.55 at α=0.8),
  no-replace clipping saturation (≤0.66 at α=1), calibrated fix.
  Source: audit/AUDIT_FINDINGS.md, exp_C SUMMARY §0. One table + one paragraph.
- States the α=1.0 sentinel convention and the three-sampler inconsistency in prior pipelines
  (factually, one sentence each).

## §5 Mechanism verification + null models — the paper's core
- 5.1 Fixed-partition decomposition on real networks: KEEP draft Tables 1–3 (Exp 1 data) as
  *mechanism verification* (ΔQ_fixed = ΔF−ΔG to machine precision; two regimes), now WITH a
  realized-retention column. Figures cit-HepTh A/B: KEEP.
- 5.2 **Configuration-model null — NEW central table**: 17 networks × {real, rewired}: Q_base, δ,
  hb, ΔQ_fixed. Verdict: 17/17 nulls reproduce both signals, median ratio 1.24; facebook real
  δ<0; hb>1 in 17/17 nulls. Source: exp_B_config_null/results.csv + SUMMARY.md.
  NEW figure: paired real-vs-null bars (or slope chart) for ΔQ_fixed and δ.
- 5.3 Partition provenance: NEW short subsection. δ_planted vs δ_leiden on LFR (no sign flip, but
  systematic positive bias growing with μ; δ>0 needs Leiden + heavy tails). Amazon GT vs Leiden
  sign flip. Source: exp_A + exp_D SUMMARY.
- 5.4 LFR hub-bridging (old Exp 2): KEEP one figure + condensed text, WITH provenance caveat and
  degree-homogeneity note; define rewiring operator h formally (draft never did).

## §6 Three evaluation artifacts, three controls — NEW (from Phase 1)
- 6.1 Artifact I — cross-graph modularity. Transfer loss: partitions from sparsified graphs lose
  0.01–0.19 Q on the original graph, 15/15. Source: exp4_comprehensive CSV (Q_transfer_loss) +
  audit fresh test. Table: Q_sparse vs Q_on_orig vs Q_base at α∈{0.8,0.9}.
- 6.2 Artifact II — granularity. Resolution-matched controls beat DSpar on recovery everywhere:
  large scale (exp_D: 0.404 vs 0.285 etc., avgF1) and small graphs (exp_F: email-Eu-core dies
  under AMI + γ-matched control; 0/5). AMI primary, NMI demoted (Vinh+ 2010). Fix/flag Dolphins
  GT surrogate. Tables from exp_D + exp_F results.csv.
- 6.3 Artifact III — unmatched computation. wiki-Vote case: seeded beats mean baseline 6/8 but
  loses runtime-matched 8/8. Source: exp_C.
- Each subsection ends with the control stated as a protocol rule.

## §7 The boundary: where genuine gains exist — NEW
- 7.1 Runtime-matched seeded refinement, 15 networks, calibrated α=0.9: 2/15 genuine
  (email-Enron +0.0077, com-Youtube +0.0086); full table. Source: exp_E outcomes.csv.
- 7.2 No predictor: 14 candidates, none significant (best n: ρ=0.47, p=0.079); δ* wrong-signed
  (diagnostic only); **wiki-Talk collapse case study** (deg_cv r=0.69→0.12) as a self-contained
  caution about small-n correlations — including the r=0.92 style of claim. Source:
  exp_E correlations.csv + SUMMARY.
- 7.3 True-retention sweep: transfer loss shrinks ~10× as retention→1; crossover only on Enron.
  Source: exp_C.

## §8 Runtime reality — REWRITE of old Exp 4 (much shorter)
- Honest timing: Leiden speedups mostly <1 at α=0.8; fragmentation regime; pipeline overhead.
  KEEP fragmentation diagnostics narrative; DROP the speed–quality "favorable regime" claims.
- **DROP the spectral baseline entirely** (returned original graph — broken), or rerun properly
  on Fuji if we want the comparison; decision deferred, default DROP.
- Fix: 8-vs-6 dataset mismatch, com-Orkut phantom mention, cit-Patents speedup arithmetic.

## §9 Discussion + Conclusion — REWRITE
- What the theory does and does not license; protocol summary (the three controls + null model);
  implications for sparsification-pipeline papers and for benchmark design (LFR degree
  homogeneity); limitations (modularity/Leiden/DSpar family; 2/15 existence proof unexplained;
  n=15 power); future work: predictor search at scale (100+ networks, pre-registered), ABCD
  hub-bridging sweeps, well-connectedness angle (Park/CM++) as open link.

## Appendix
- Full per-α tables; sampler equivalence verification; reproducibility statement (repo + PAPER_RESTRUCTURE
  scripts); FILL OR DELETE the currently-empty appendix promise (NMI trends L1538).

## Draft tables/figures disposition
| Draft item | Fate |
|---|---|
| Tab 1–3 (Exp1 modularity/decomposition/summary) | KEEP → §5.1 (+retention column; ΔQ_Leiden column REMOVED or replaced by Q_on_orig) |
| Tab 4 (hub_bridge 17 nets) | KEEP → §5.2 merged into real-vs-null table |
| Tab 5 (ground truth 5 small) | REPLACE with exp_F version (AMI, controls, Dolphins fixed) |
| Tab 6 (scalability) | REWRITE → §8, spectral rows dropped |
| Fig cit-HepTh A/B | KEEP §5.1 |
| LFR figs (4) | KEEP 2 (dQ vs hub ratio; delta vs mu), DROP 2 |
| ΔNMI/ΔARI vs δ figs | DROP (superseded by exp_F) |
| pipeline-time, speedup-vs-quality figs | DROP or redo honest |
| NEW figs | real-vs-null paired chart (exp_B); transfer-loss chart (6.1); wiki-Talk collapse inset (7.2) |

## Writing order (proposed)
1. §5.2 null table + §6 artifacts (the spine — data ready)
2. §3 theory repairs (Cor 2, Thm 3, sampler def)
3. §7 boundary, §4 audit
4. §1 intro, §8, §9
5. Abstract final polish, related work, appendix, bib additions
