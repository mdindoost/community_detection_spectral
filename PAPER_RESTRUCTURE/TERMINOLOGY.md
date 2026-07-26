# TERMINOLOGY — fixed definitions, do not drift

Written 2026-07-26 after the phrase "cross-graph" caused a real misunderstanding between
Mohammad and Claude. **The phrase "cross-graph" is RETIRED. It reads both ways and must never
appear in the paper, in summaries, or in conversation.**

---

## The two things that must never be confused

### ✅ HONEST TRANSFER SCORING — the correct method
Mohammad's standing principle from the beginning: *"we remove edges, so we must go back to the
MAIN graph to compare."*

```
compare   Q(P_sparse , G_ORIGINAL)   vs   Q(P_baseline , G_ORIGINAL)
                      ^                                ^
              both partitions scored on the SAME original graph
```

Detect on whatever graph you like; **score on the original**. One yardstick, two partitions.
This is what our entire protocol runs on (`dQ_honest_vs_mean`, `dQ_honest_vs_best`,
`dQ_vs_matched` in every results.csv).

Prior art: **Satuluri, Parthasarathy & Ruan, SIGMOD 2011, §4.3** — *"we cannot simply measure the
conductance using the very same sparsified graph, since that would tell us nothing about how well
the sparsified graph retained the cluster structure in the original graph. Therefore, we report the
conductances of the clusters obtained from the sparsified graphs also using the structure of the
original graph."* Table 2 caption: *"phi_avg is always calculated w.r.t. the original graph."*
They stated it first. We restore it; we do not claim it.

### ❌ SPARSE-GRAPH SELF-SCORING — the error (= our Artifact I)

```
compare   Q(P_sparse , G_SPARSE)     vs   Q(P_baseline , G_ORIGINAL)
                      ^                                ^
          scored on the sparsified graph      scored on the original
```

Two numbers computed **on two different graphs**. The sparsified graph has had its
inter-community edges preferentially removed, so it flatters any partition scored on it.
Column name in our CSVs: `dQ_naive`. Related trap: `dQ_fixed` (a fixed partition re-scored on the
sparse graph), which the configuration null reproduces in full (exp_L finding 3) and which is
therefore evidence of nothing.

**Measured cost of the error:** the sign of the conclusion flips in **45 of 63** controlled cells
across four detection algorithms (exp_P V2). It is not a subtle bias; it inverts the answer.

---

## Where each appears in the literature (verified from primary sources)

| Source | Which method | Evidence |
|---|---|---|
| Satuluri et al. 2011 (SIGMOD) | ✅ honest transfer | §4.3 + Table 2 caption, quoted above |
| Chen et al. 2024 (PVLDB), the *paper* | n/a — computes no graph-scored objective | "modularity" appears 0 times |
| Chen et al. 2024, the *released code* | ❌ self-scoring | `src/metrics_nk.py:414` — `Modularity().getQuality(C, Graph)` with `Graph` = sparsified |
| Pari et al. 2026 (ERSCD, arXiv 2606.26766) | ❌ self-scoring | Reports Q=0.87 on football; that graph's maximum modularity over ALL partitions is 0.6046. Impossible on the original. Honest transfer of their own partition: 0.598 |
| Setiadi et al. 2025 (IJAIN, CPSK) | ❌ self-scoring | Reports Q~0.63 on karate; karate's maximum is 0.4198. Reproduction yields Q~0.97 on random graphs with NO communities |
| **Our own February draft (`main-tnse.tex`)** | ❌ self-scoring **in the headline** | The honest measure (`Q_transfer_loss`) was ALREADY COMPUTED in our own `exp4_comprehensive` and showed loss on all 15 datasets. The method was in the work; the abstract reported the other number. |

---

## The narrative this supports

The discipline was published correctly in 2011 by the paper that started this line, and was
progressively lost — to the point where a 2026 paper reports a modularity above its graph's
mathematical maximum, and the field's most rigorous benchmark computes the artifact by default in
its own tooling. **We do not claim to have discovered the artifact. We claim to have measured what
its absence costs (45/63 sign flips) and to supply the protocol that prevents it.**

---

## Other terms, fixed

- **Artifact I** = sparse-graph self-scoring (above).
- **Artifact II** = granularity: fragmentation mechanically inflates best-match F1, purity, NMI.
  Control = resolution-matched partition of the ORIGINAL graph at the same cluster count.
- **Artifact III** = uncorrected metrics: NMI/F1 without a chance floor. Control = AMI/ARI plus a
  size-matched random-partition floor.
- **Artifact IV (candidate, from exp_AB/K-core reading)** = node-set truncation: node-removing
  sparsifiers (k-core, peeling) score recovery on a smaller, denser, easier subpopulation.
  Control = report |V_sparse|/|V| and match coverage.
- **Runtime-matched baseline** = best of as many plain restarts as fit the pipeline's wall clock.
  **Known weakness (exp_AA):** when the pipeline is fast this buys only 1 restart and inflates
  apparent gains. **Always report `dQ_vs_base_best` (best-of-5) alongside it.**
- **delta (δ)** = mean sparsifier score on intra-community edges minus mean on inter-community
  edges, for a given partition. The mechanism quantity.
- **alpha (α)** = in DSpar's theory, the normalized spectral gap (their bound is (1±ε/α)).
  In our experiment configs, α=1.0 is Mohammad's **no-sparsification sentinel**. These are
  different αs; disambiguate wherever both appear.
