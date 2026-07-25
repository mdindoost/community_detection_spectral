# Panel Review — Judge's Ruling (2026-07-25)

Panel: Referee B (Fable, theory/framing) — full report delivered, verdict **major revision**.
Referee A (Opus, methodology/stats) — stalled after ~1h with no findings produced; killed.
Decision: proceed on B + judge verification; run a narrow Opus **numbers audit** on the revised
paper afterward (cheaper and more useful than a second full review).

## Adjudication of Referee B's major concerns

| # | Finding | Ruling | Disposition |
|---|---|---|---|
| MC1 | Unclipped-regime hypothesis unsatisfiable at experimental α (α* ≈ 0.05 on email-Eu-core; ~65% clip at α=0.9); direction unproven in clipped regime | **CONFIRMED** | Remark 1 rewritten with true numbers + FOSD sufficient condition; abstract scoped to "score-proportional sampling"; intro claim rewritten |
| MC2 | "Verified to high accuracy" rests on an algebraic identity; no ratio validation shown | **CONFIRMED — and worse**: judge checked exp1_2 CSVs; Cor. 1 ratio predictions miss by 0.07–0.30 under the with-replacement sampler | Claim deleted; §5.1 now reports the 0.07–0.30 discrepancy as a finding (signature of concave retention); "theory's predictions hold" header replaced |
| MC3 | com-Youtube gain fragile (2.5σ, single config, matched baseline one unlucky draw; "5–8σ" wrong for the 15-net table) | **CONFIRMED** (verified from outcomes.csv: Q_matched 0.7209 < base mean 0.7245) | Downgraded to "one robust gain + one candidate" in abstract, intro, §7.1, tab_boundary caption; correct σ figures inserted; multiplicity note added |
| MC4 | "Cannot distinguish from null" overstates: δ_null ≥ δ_real 16/17 makes the pair a discriminator with opposite sign | **CONFIRMED** | Slogans replaced ("does not certify"); suppression observation added as a finding in §5.2 and as an open problem in §9 |
| MC5 | Proof defects: Thm 1(b) s̄* limit, hidden hypothesis, Cor 1 limit-mixing, Assumption 1 star counterexample, Prop 1(B) equivalence, Def 5 well-posedness, Slutsky misuse | **CONFIRMED** (all checked) | All repaired: hypotheses condensed to (i)–(iii) incl. unclipped-along-sequence; proof rewritten with cancelling s̄^(n); Cor 1 restated; Assumption 1 justification fixed; Prop 1(B) now "equivalent"; Def 5 existence/uniqueness added; Slutsky removed |
| MC6 | m inconsistencies (full graph vs LCC), two different 15-suites, 0.01–0.19 vs 0.004–0.185 | **CONFIRMED** | Preprocessing note added in §5.1; suite-membership note added; ranges unified to 0.004–0.19 in abstract and intro. (Full LCC recompute of Tables 1–2 deferred; documented honestly) |
| MC7 | tab_transfer_loss caption misdefines transfer loss | **CONFIRMED** (verified against column values) | Caption formula corrected to Q_base − Q_orig; explicit definition added to §6.1 prose |
| MC8 | +0.0149 misattributed to with-replacement sampler (actually clipped no-replacement) | **CONFIRMED** (verified in exp_C SUMMARY) | Sentence corrected; sampler pair named explicitly |
| MC9 | Hypothesis 1 framed as empirical law, falsified by own data; pigeonhole stated as expectation | **CONFIRMED** | Reframed as Hub–Bridge **Condition** on (G, C) pairs; pigeonhole fixed to deterministic bound; closing sentence added noting 3/17 violations + all nulls satisfying it |
| MC10 | 8+ unverifiable references to "an earlier version"; "draft" language in captions | **CONFIRMED for captions; PARTIALLY for text** | Captions neutralized (4 fixes); text mentions reduced and qualified ("unpublished manuscript"); kept where they carry scientific content (r=0.92 case study, spectral-bug disclosure, §9 candor) |

Minor issues fixed: notation footnote (bare δ vs Kronecker), preservation-ratio domain guard,
best-of-2 budget-slack disclosure, abstract "most networks"→"13 of 15", "eight labeled datasets"→
"every labeled dataset", LFR homogeneity now relative (CV 0.4 vs 1.0–26), F1-granularity mechanism
sentence, runtime-numbers provenance pointer, per-setting clip percentages.

Deferred (tracked, not blocking): vector figures; repository URL/DOI in appendix; LCC recompute of
Tables 1–2; compressing spectral background; supplementary-izing tab_predictors; Youtube matched-
baseline distribution rerun + Enron-style sweep on Youtube (would upgrade or kill the candidate).

## Judge's assessment of the panel

Referee B was accurate on every claim spot-checked (8/8 verified against tex/CSVs). Its report
exceeded its assigned lens and covered most of Referee A's mandate. The review-panel exercise
caught: one wrong formula in a headline table caption, one misattributed result, one overstated
statistical claim, one theory-scope gap material to the abstract, and four pre-existing proof
defects — none of which the authors (human or AI) had caught in three prior passes.
