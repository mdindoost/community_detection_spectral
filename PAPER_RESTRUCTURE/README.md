# Paper Restructure — "When Does Sparsification Help Community Detection?"

**Date started:** 2026-07-24
**Reason:** Audit of the TNSE draft (`Paper_materials/main-tnse.tex`) found the headline claim
("sparsification improves community detection") rests on comparing modularity across different
graphs, which is not a valid comparison. Confirmed by three independent pipelines (see
`audit/AUDIT_FINDINGS.md`).

## New thesis

> Apparent modularity gains under sparsification are largely a measurement artifact with a
> precise, provable mechanism (our Theorems 1–3 explain it). Genuine gains exist only under
> narrow, identifiable conditions — which we characterize.

The theory section survives unchanged; it becomes the *explanation of the artifact* rather than
the prediction of an improvement.

## Key audit facts (established 2026-07-24)

1. `ΔQ_Leiden` in the draft = Q(P_α on sparsified G) − Q(P_0 on original G) → sparse-graph self-scoring (Artifact I), invalid.
2. Scored honestly on the original graph, DSpar at effective ~40% retention **loses** 0.01–0.19
   modularity on all datasets tested (fresh test + repo's own `results/exp4_comprehensive/`
   `Q_transfer_loss` column, positive on all 15 datasets, all α).
3. Seeded refinement on the original graph (`Q_final`) recovers to baseline; best genuine gains:
   email-Enron +0.007, com-DBLP +0.003. ~40× smaller than the draft's claimed gains.
4. Degree-preserving rewired graphs (no communities, Q_fixed≈0.24–0.31) reproduce **both** signals
   (δ>0 and ΔQ_fixed>0) at equal or larger magnitude → δ>0 does not certify community structure.
5. Sampler inconsistency: draft's Definition (E[retention]=α) and text ("no-replacement") vs code
   (`method="paper"`, with-replacement → actual retention ≈ 0.33–0.55 at nominal α=0.8; documented
   in `experiments/dspar.py` docstring and visible in draft Table 5 `m_α` column).
6. Draft Exp 4 table: spectral baseline returns the original graph (m_α = m, identical times);
   Leiden "speedups" at α=0.8 are mostly < 1; text says 8 datasets / cites com-Orkut, table has 6.

## Phase 1 — decision experiments

| Dir | Experiment | Question it decides | Status |
|---|---|---|---|
| `exp_A_lfr_leiden_delta/` | δ on standard LFR w.r.t. **Leiden** partition vs planted partition | Is δ>0 an estimator artifact (partition provenance) or a real-network property? Decides how Exp 2's story is rewritten. | ✅ done — no sign flip, but provenance bias real; δ>0 needs Leiden + heavy tails |
| `exp_B_config_null/` | Config-model null (degree-preserving rewire) across the 17 networks | Smoking-gun table: real vs rewired ΔQ_fixed and δ side by side. | ✅ done — nulls reproduce both signals 17/17, usually LARGER (median ratio 1.24) |
| `exp_C_true_retention_seeded/` | True no-replacement retention α ∈ {0.7–0.95} + seeded Leiden vs **runtime-matched** restarts baseline | The only remaining shot at a genuine positive headline result. Untested territory (all prior runs were effective ~40% retention). | ✅ done — gains survive runtime-matching only on email-Enron; wiki-Vote = negative control |
| `exp_D_groundtruth_scale/` | Ground-truth recovery at scale (com-Amazon/DBLP/Youtube, SNAP top-5000) | How big the "when it genuinely helps" section is. Currently rests on one small dataset. | ✅ done — apparent gains = granularity artifact; resolution-matched control wins 2–4× |

**Consolidated verdict: `PHASE1_VERDICT.md`. Big-machine needs: none (`BIG_MACHINE_TODO.md`).**

Each experiment dir gets: the script, results CSVs, and a `SUMMARY.md` with the verdict.

## Phase 2 — paper rewrite (after Phase 1 verdicts)

- Rewrite abstract/intro around new thesis; retitle.
- Theory: keep; add artifact-mechanism paragraph; fix Corollary 2 hypothesis (weighted balance).
- Promote `Q_on_orig` / `Q_final` (already in `results/exp4_comprehensive/`) to the main table.
- Exp 2 (LFR): add partition-provenance analysis from exp_A.
- Exp 4: fix spectral baseline or drop it; honest speed numbers; repair table/text mismatches.
- One sampler consistently; report effective retention everywhere.

## Phase 3 — polish

Minor-issues list from the audit: DSpar score equation missing in Background; stale "Experiment
1.2/1.3" references; empty appendix with a forward reference; missing citations (Satuluri+ 2011,
disparity filter/Serrano+ 2009, Guimerà+ 2004 random-graph modularity, Fortunato–Barthélemy
resolution limit); Theorem 3(a) quantifiers; define hub-bridging strength h operationally;
data/code availability statement.

## Environment note

`venv/` is broken (built for Python 3.12; system is 3.14). Working env:
`/tmp/claude-1001/-home-md724-community-detection-spectral/d102f690-a31a-4e32-8710-5abe2b39ee56/scratchpad/v14`
— rebuild `venv` properly when convenient.
