# NOTES — Effective Resistance-Based Graph Sparsification and Community Detection (ERSCD)

**Audit date:** 2026-07-26 · **Auditor:** adversarial due-diligence pass, steelman-first
**Status:** headline claim does not survive; one genuinely untested regime identified (see §6)

---

## 1. Bibliographic details

- **Title:** Effective Resistance-Based Graph Sparsification and Community Detection
- **Authors:** Jayanta Pari, Pratibha Bhandari, Soumyendu Raha
- **Affiliation:** Department of Computational and Data Science, Indian Institute of Science (IISc), Bangalore, India
- **arXiv:** 2606.26766 (submitted 25 June 2026)
- **Categories:** cs.SI (primary); cs.CG; math.DS
- **DOI:** https://doi.org/10.48550/arXiv.2606.26766 · **License:** CC BY 4.0
- **Method acronym:** ERSCD
- **Local copy:** references/effres_sparsification_2026.pdf
- **Sections:** 1 Intro · 2 Related Work · 3 Effective Resistance preliminaries · 4 Methodology · 5 Experimental Setup · 6 Results · 7 Evaluation Against Existing Methods · 8 Conclusion

---

## 2. Steelman — the strongest version of their case

### 2.1 Claim, stated precisely

Sparsifying a graph by (a) converting effective resistance into an edge similarity, (b) retaining a
minimum spanning tree as a mandatory backbone, and (c) deleting a `pe` fraction of the
lowest-similarity non-MST edges, then running Clauset-Newman-Moore weighted modularity
maximisation on the resulting weighted sparse graph, yields **better community recovery than
Louvain, Infomap and Ricci Flow** while remaining computationally practical. Verbatim from §8:
*"ERSCD consistently performs better in terms of accuracy and also takes reasonably less computing
time than the other existing methods"* and *"in most cases ERSCD outperforms all the other
methods."* Stated contributions: *"A novel sparsification technique using spanning tree that
preserves community structure while reducing computational complexity"* and *"Extensive
experimental evaluation ... demonstrating the superiority of our approach compared to
state-of-the-art methods."*

### 2.2 Method

1. **Effective resistance.** Exact via `R_uv = (e_u - e_v)^T L^+ (e_u - e_v) = L^+_uu - 2L^+_uv - L^+_vv`
   (their Thm 3.1); pseudoinverse computed once, then O(1) per pair. They also cite the
   Spielman-Srivastava Johnson-Lindenstrauss approximation as an alternative.
2. **Weight transform.** Gaussian kernel `w_uv = exp(-R_uv^2 / 2 sigma)`, with sigma = 1 in experiments.
3. **Restriction / normalisation.** `R^mod_H = sqrt(R^mod o A)` (Hadamard with adjacency), then
   `R^mod_H = 0.5(R^mod_H/max + (R^mod_H)^T/max)`.
4. **Similarity.** `S = J - R^mod_H`, J the all-ones matrix.
5. **Sparsification.** Prim MST using effective resistance as edge weight -> separate MST from
   non-MST edges -> sort non-MST edges *ascending by similarity* -> delete `pe` fraction -> rebuild
   graph **with similarity values as edge weights**.
6. **Detection.** Clauset-Newman-Moore greedy weighted modularity, `Q = (1/2m)Sum(A_uv - d_u d_v/2m)delta(c_u,c_v)`
   with `A_uv` the similarity weights.
7. **Complexity claimed:** `O(n^omega + m log n + m log^2 n)`, omega <= 3.

Only one free parameter (`pe`), which is a genuine merit of the design.

### 2.3 Datasets

- **Synthetic:** SBM (250 nodes) and LFR (250 nodes; avg degree 5, degree exponent 3, community
  exponent 1.5, community sizes 10-50, mu in [0.1, 0.6]).
- **Real:** Zachary Karate Club, American College Football, Political Books, Political Blogs, Cora.
- **Edge counts are not reported for any dataset.** Largest network is Cora (n = 2708).

### 2.4 Metrics and baselines

- **Metrics:** ARI, NMI, modularity. (ARI is chance-corrected - genuinely better than much of this
  literature.)
- **Baselines:** Louvain, Infomap, Ricci Flow, run by the authors. Synthetic baselines averaged over
  10 simulations. Results presented as figures (Figs 17-22); **no numeric baseline tables**.
- **`pe` selection:** swept 0-0.5 in steps of 0.05; best cell reported. §8 concedes *"further work is
  needed to define a more general rule for selecting the pe value."*

### 2.5 Their evidence at its strongest

| Dataset | `pe`* | ARI | NMI | Modularity |
|---|---|---|---|---|
| Karate Club | 0.40 | 0.74 | 0.71 | 0.42 |
| American College Football | 0.35 | **0.87** | **0.91** | **0.87** |
| Political Books | 0.20 | 0.70 | 0.59 | 0.51 |
| Political Blogs | **0.00** | 0.80 | 0.67 | 0.42 |
| Cora | **0.00** | 0.31 | 0.46 | 0.90 |

Football ARI 0.87 / NMI 0.91 is a strong recovery result in absolute terms, and modularity 0.87
would be a remarkable score if it meant what it appears to mean. On synthetic data, ERSCD and
Louvain are the only methods holding up at mu > 0.4. The single-parameter design and the
connectivity-guaranteeing MST backbone are real methodological virtues.

---

## 3. Reproduction

Their pipeline was reimplemented and run on karate, football and polbooks (all three obtainable
locally). Two readings of the ambiguous §4 similarity definition were tested; the **resistance
reading** (`S = J - normalised sqrt(R o A)`, so high S <=> low resistance <=> similar, which is the only
reading under which "remove lowest similarity" is sensible) reproduces the published numbers:

| Network | Paper reports | Reproduced (`res` variant) |
|---|---|---|
| football @ pe~0.35-0.40 | ARI 0.87, NMI 0.91, Q 0.87 | ARI 0.869, NMI 0.916, **Q_sparse 0.848** |
| polbooks @ pe=0.20 | ARI 0.70, NMI 0.59, Q 0.51 | ARI 0.680, NMI 0.561, **Q_sparse 0.510** |
| karate @ pe=0.40 | ARI 0.74, NMI 0.71, Q 0.42 | ARI 0.709, NMI 0.703, Q_sparse 0.398 |

The literal Gaussian reading inverts the deletion order and destroys performance (football ARI
0.04-0.11), confirming the resistance reading is the intended one. Reproduction fidelity is what
makes the audit below load-bearing rather than speculative.

**Validation of the harness:** my modularity optimiser recovers the known literature optima exactly -
football 0.6046, karate (unweighted) 0.4198.

---

## 4. Audit against our controls

| # | Artifact / control | Verdict | Evidence |
|---|---|---|---|
| **I** | **Scoring on the sparse graph** | **COMMITTED - provable** | Reported football Q = 0.87. Maximum achievable modularity on the **original** football graph is **0.6046**; the ground-truth partition scores **0.554**. 0.87 is arithmetically impossible on the original graph. Cora's reported Q = 0.90 likewise exceeds that graph's achievable maximum (~0.81). Reproduction confirms 0.87 is the *weighted modularity of the resistance-reweighted sparse graph*. **Honest transfer** of their own football partition to the original graph = **0.598**, which is *below* plain Louvain best-of-20 (0.6046) -> sign flip. Signature is textbook: as `pe` rises 0->0.5, Q_sparse climbs 0.675->0.874 while Q_honest is flat/declining 0.595->0.591. *Caveat: the paper never states in words which graph it scores on (§6 is silent); this finding rests on arithmetic impossibility + reproduction, not a quotable sentence.* |
| **II** | **Granularity** | **SILENT, and it is doing the work** | Cluster counts are never reported as a control variable, only as prose ("detects all the communities properly"). Forced-k control on the **original** graph: football ERSCD k=11 ARI 0.869; Louvain-on-original at k=11 gets **0.857**, at k=12 **0.897**, at k=13 **0.912**. Karate: ERSCD 0.709 at k=3 vs Louvain-on-original **0.882** at k=2. Polbooks: ERSCD 0.680 at k=4 vs **0.694** at k=4. ERSCD loses to a granularity-swept original-graph baseline on **3/3** networks. |
| **III** | **Uncorrected metrics** | **PARTIALLY CONTROLLED - credit due** | They report **ARI**, which is chance-corrected, alongside NMI. This is better than most of the surrounding literature and better than our own February draft. AMI is absent but ARI carries the load. **Not a fatal flaw here.** |
| **Cost accounting** | End-to-end cost incl. sparsifier | **COMMITTED (self-contradicted)** | Abstract claims *"maintaining computational efficiency"*; §8 claims *"reasonably less computing time than the other existing methods"*. But their own Fig 22 text says *"least time is taken by Infomap and Louvain method, and our method's execution time lies in between"* - i.e. **ERSCD is slower than both**, and is only faster than Ricci Flow. Exact `L^+` is **O(n^3) time, O(n^2) memory**; on com-Amazon (335k nodes) that is ~10^16 flops and ~900 GB. Method as published cannot reach even the small end of our suite. §8 concedes *"We haven't used GPU acceleration yet."* |
| **Baseline strength** | Best-of-N at matched wall clock | **COMMITTED - the decisive omission** | **Plain unsparsified CNM - their own detector - is never a baseline.** This is the one control that isolates the sparsification contribution. Measured: football plain CNM ARI **0.474** -> ERSCD 0.869 looks like a huge win; but Louvain best-of-20 on the **original** graph gets **0.807** untuned and **0.912** granularity-swept. The "gain" is escaping CNM's resolution-limit merge pathology, **not** a capability sparsification adds - exactly our Exp P com-Amazon finding (C16/C18). Real-network baselines are single-run; only synthetic baselines are averaged (10 sims). No variance reported for ERSCD anywhere. |
| **Config-model null** | Degree-preserving rewiring | **ABSENT** | Not attempted. |
| **Shuffled-weight null** (our Exp V, C17) | Is the benefit weight *information* or weight *heterogeneity*? | **ABSENT, and it fires** | Permuting the similarity weights across the same retained topology reproduces essentially all of the sparse-graph modularity at high `pe`: football pe=0.50, ERSCD Q_sparse **0.8737** vs shuffled-weight null **0.8719**. On karate the null *exceeds* ERSCD in most cells (e.g. pe=0.15: null 0.4729 vs 0.4099). Same indictment as Exp V. |
| **Retention regime** | Is this even aggressive sparsification? | **NO** | The mandatory MST backbone plus `pe <= 0.5` on non-MST edges floors retention at **59-88%** (karate 77% @ pe=0.4; football 67% @ pe=0.4, 59% @ pe=0.5; polbooks 85% @ pe=0.2). This is a mild-retention regime, far from where speedups are claimed to live. |
| **Their own null result** | - | **UNREMARKED** | Their own tuning selects **pe = 0** - *no sparsification at all* - on **2 of 5** real networks (Political Blogs, Cora). At pe = 0 the only intervention is resistance reweighting. The paper does not comment on this. |

### 4.1 Additional finding: it is not a spectral sparsifier

The paper cites Spielman-Srivastava in §4 and frames itself as effective-resistance sparsification,
but the actual algorithm **discards the sampling-with-reweighting** that makes SS a spectral
sparsifier. ERSCD deletes edges *deterministically* by a similarity threshold and reweights by
similarity, not by inverse sampling probability. It therefore carries **no spectral guarantee
whatsoever** - it is resistance-derived *reweighting + backbone extraction*. The legitimacy of the
SS guarantee is borrowed but not inherited. This should be stated explicitly if we cite the paper.

---

## 5. What genuinely survives

**Effective resistance carries real community signal.** A **random-score null** pushed through the
*identical* MST + threshold + reweight pipeline (resistance replaced by random scores, 10 reps):

| Network | ERSCD ARI | Random-score null ARI | Gap |
|---|---|---|---|
| football | 0.869 | 0.638 +- 0.046 | **+0.231** |
| karate | 0.709 | 0.512 +- 0.072 | +0.197 |
| polbooks | 0.680 | 0.547 +- 0.057 | +0.133 |

This is the same status L-Spar's Jaccard signal earned in Exp L (C16): the selection signal is
genuinely structure-aware and is **not** reproduced by a null. It just fails to convert into a win
over an honest baseline. Worth saying so plainly - it is a real property of effective resistance,
not an artifact.

---

## 6. Verdict

**Primarily (d) - nothing in the headline claim survives our controls - with a genuine (c) component.**

- **(a) Valid result surviving our controls?** *No.* The modularity claim is Artifact I and is
  arithmetically impossible on the original graph. The recovery claim fails against plain Louvain
  best-of-N on the unsparsified graph on 3/3 reproducible networks. The efficiency claim is
  contradicted by the paper's own Figure 22.
- **(b) Untested regime?** *Partly, but not usefully.* Everything is <= 2708 nodes, so it is a
  small-graph regime we deliberately do not target; and the method cannot scale past it (O(n^3)/O(n^2)).
  Cora is attributed, but attributes are not used. Not a regime worth chasing.
- **(c) Method fundamentally different from DSpar / L-Spar / uniform - YES, in one specific respect
  that matters.** The **MST backbone constraint** is a class of sparsifier we have never tested.
  Every sparsifier in our suite deletes edges without a connectivity guarantee, which is *why*
  fragmentation (Artifact II) dominates our results. A backbone-constrained sparsifier **structurally
  cannot fragment**. Our Exp G touched spectral/effective-resistance methods only lightly and tested
  nothing connectivity-constrained. This is a real hole in our coverage, independent of whether this
  particular paper is sound.
- **(d)** Applies to the paper's claims as stated.

### Honest limits of this audit

- The paper never states in prose which graph modularity is measured on; the Artifact I finding is
  inference from arithmetic impossibility plus faithful reproduction, not a quoted sentence.
- Political Blogs and Cora were not reproduced locally - those rows are unaudited except for Cora's
  impossible modularity value.
- Numeric values were extracted via automated fetch of the arXiv HTML and cross-checked against a
  second source; per-baseline numeric tables do not exist in the paper (figures only), so the
  baseline comparison could not be read cell-by-cell.
- Karate reproduction used the unweighted graph; the paper does not state whether it uses Zachary's
  weighted or unweighted version.

---

## 7. Follow-up experiment this implies

**Exp Y - Connectivity-preserving (backbone-constrained) sparsification.**

*Motivation:* every sparsifier we have tested can disconnect the graph, and fragmentation drives
Artifact II. A mandatory-backbone sparsifier removes that failure mode by construction. If
sparsification ever helps, this is the most plausible place for it, and we have not looked.

*Design:*
1. **Arms:** (i) ERSCD-style resistance-similarity + MST backbone; (ii) Jaccard-similarity + MST
   backbone (isolates the backbone from the resistance signal); (iii) random-score + MST backbone
   (the null that fired above); (iv) plain DSpar / L-Spar at **matched retention** (no backbone) -
   the comparison that isolates the backbone's contribution.
2. **Networks:** our standard suite; use the JL-approximate resistance
   (`R = ||W^{1/2} M L^+ (e_u - e_v)||^2`) since exact `L^+` is infeasible above ~10^4 nodes. Report the
   approximation's own cost inside the cost accounting.
3. **Retention:** sweep including the 59-88% band the paper actually occupies, *and* push into the
   aggressive band the backbone permits, to separate "mild retention" from "backbone" as
   explanations.
4. **Controls, all mandatory:** honest transfer scoring on the original graph; k-matched /
   resolution-matched baselines on the original; best-of-N at matched wall clock; AMI + ARI with
   chance floor; configuration-model null; **shuffled-weight null**; **random-score null** through
   the identical pipeline.
5. **Detectors:** Leiden, Louvain, Infomap, and **CNM specifically** - CNM's resolution-limit
   pathology is what manufactures this paper's apparent gain, and including it lets us demonstrate
   that mechanism directly rather than only asserting it.

*Pre-registered prediction (from C16/C17/C18):* the backbone removes the fragmentation channel and
therefore neutralises Artifact II, but honest dQ stays <= 0 and recovery does not beat a
granularity-matched original-graph baseline; the resistance arm beats the random-score arm
(structure-aware signal, as in Exp L) without beating the unsparsified baseline. *Kill criterion:*
if the backbone arm produces a positive honest dQ or a chance-corrected recovery gain that survives
k-matching and best-of-N on >= 3 networks, our headline conclusion needs an explicit carve-out for
connectivity-preserving sparsifiers.

---

## 8. How to cite this paper in our related work

Use it as a **contemporary, independent instance of Artifact I plus the missing-own-detector
baseline** - it is a 2026 paper, so it demonstrates the pitfalls are live, not historical. Precise
and fair phrasing:

> A 2026 effective-resistance method (Pari et al., arXiv:2606.26766) reports modularity 0.87 on the
> American College Football network, a value that exceeds the maximum modularity attainable on that
> graph (0.6046); the score is computed on the reweighted sparse graph. Transferred honestly to the
> original graph the same partition scores 0.598, below a plain best-of-N Louvain baseline. The
> reported recovery gain is measured against the method's own detector (CNM) at its default
> granularity and does not survive a resolution-matched comparison on the unsparsified graph.

Credit where due: they use chance-corrected ARI, expose a single interpretable parameter, and their
resistance similarity is genuinely structure-aware under a random-score null. Also note that despite
citing Spielman-Srivastava, the method is not a spectral sparsifier - it drops the
sampling-with-reweighting step and inherits no guarantee.
