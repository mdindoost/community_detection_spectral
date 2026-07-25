# Exp C — True retention + seeded Leiden vs runtime-matched baseline

Files: `run.py`, `results.csv` (240 per-run rows), `results_summary.csv` (48 aggregated rows), `run.log`.
Datasets: email-Eu-core, wiki-Vote, ca-HepTh, ca-CondMat, email-Enron, com-DBLP (LCC). 5 seeds.

## 0. Sampler audit — changes the premise

`experiments/dspar.py::dspar_sparsify(method="probabilistic_no_replace")` computes
`p_e = clip(score_e/Σscore · ⌈αm⌉, 0, 1)`; 13–35% of edges hit the clip, so **E[kept] < α·m always
and it saturates** — it cannot produce true 70–95% retention at any α, not even α=1.0
(email-Eu-core: 0.662 at α=1.0; wiki-Vote: 0.447 at α=0.95).

Two samplers were therefore run:
- `repo_noreplace` — bit-for-bit the repo formula (verified via `run.py --verify-sampler`); actual retention 0.37–0.68.
- `calibrated` — same DSpar scores/Bernoulli scheme, λ solved by bisection so Σ min(1, λ·s_e) = α·m
  ⟹ E[retention] = α exactly (measured 0.699–0.951).

**Consequence for the paper:** the draft's formal Definition (E[ret]=α), `probabilistic_no_replace`
(0.37–0.68), and `paper` (0.33–0.55) are three different samplers. Measured m'/m must be reported everywhere.

## 1. Calibrated sampler (true retention) — seeded vs baselines

Runtime-matched budget bought exactly **2 restarts** in all 48 cells (T_pipe = 1.55–1.90 × T_leiden;
sparsification is only 4–8% of pipeline time), so the fair baseline is best-of-2 plain Leiden.

| dataset | α | Q_base | Q_orig(P_α) raw | Q_seeded | Q_matched | seeded−base | seeded−matched |
|---|---|---|---|---|---|---|---|
| email-Eu-core | 0.95 | 0.4160 | 0.4151 | 0.4155 | 0.4174 | −0.0005 | −0.0018 |
| wiki-Vote | 0.95 | 0.4244 | 0.4219 | 0.4222 | 0.4294 | −0.0022 | −0.0072 |
| ca-HepTh | 0.95 | 0.7614 | 0.7579 | 0.7619 | 0.7621 | +0.0006 | −0.0002 |
| ca-CondMat | 0.95 | 0.7308 | 0.7284 | 0.7324 | 0.7294 | +0.0016 | **+0.0030** |
| **email-Enron** | 0.80 | 0.6071 | **0.6103** | **0.6155** | 0.6051 | **+0.0084** | **+0.0104** |
| **email-Enron** | 0.90 | 0.6071 | **0.6119** | **0.6146** | 0.6051 | **+0.0075** | **+0.0095** |
| com-DBLP | 0.95 | 0.8257 | 0.8247 | 0.8284 | 0.8275 | +0.0027 | +0.0009 |

(Full 48-row table in `results_summary.csv`; α ∈ {0.7, 0.8, 0.9, 0.95} × 2 samplers × 6 datasets.)

## Verdict (i): where does raw-transfer loss vanish?

**Negative in 45/48 cells.** Loss shrinks ~10× at mild true retention (ca-HepTh −0.083 → −0.0035;
com-DBLP −0.059 → −0.0010) and only extrapolates to zero at retention → 1 — i.e. "don't sparsify" —
on 5/6 datasets. **Exception: email-Enron**, which crosses positive between 70% and 80% true
retention (+0.0032…+0.0048 raw transfer at 0.80–0.95).

## Verdict (ii): does seeded beat runtime-matched restarts?

Yes in 12/48 configs, on 3/6 datasets, **convincingly only on email-Enron**:
- **email-Enron: all 8 cells, both samplers, all α, +0.0007…+0.0149** (≈5–8 σ_base); seeded *mean*
  also beats best-of-**5** plain Leiden by +0.006…+0.011. The harsher repo sampler wins *bigger*
  (+0.0149) — Enron's gain comes from aggressive hub-edge removal, not mild pruning.
- ca-CondMat: calibrated α≥0.9 only (+0.0014/+0.0030); does not beat best-of-5.
- com-DBLP: edge of noise vs best-of-2 (+0.0002/+0.0009); beats best-of-5 slightly.
- email-Eu-core, ca-HepTh, wiki-Vote: never. **wiki-Vote is the key counter-example** — seeded beats
  the *mean* baseline in 6/8 cells but loses to runtime-matched in all 8. Reporting `seeded −
  Q_base_mean` without the runtime match manufactures a false positive.

Aggregate: seeded > Q_base_mean in 21/48; seeded > matched in 12/48.

## Framing implication

No general gain claim survives. Defensible claims: (a) transfer loss is governed by *true* retention
and shrinks ~10× toward zero as retention → 1, becoming a gain only on email-Enron; (b) seeded
refinement survives a runtime-matched baseline only on email-Enron (the extreme hub-bridging/δ
network), with wiki-Vote as the explicit negative control.
