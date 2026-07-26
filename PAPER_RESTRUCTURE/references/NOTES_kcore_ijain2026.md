# NOTES — Setiadi et al. 2025, "Community preserving sparsification based on K-core" (IJAIN)

*Adversarial due-diligence review. Compiled 2026-07-26. Reviewer: steelman-then-audit protocol.*
*Status: **KILLED — reproduced and falsified.** Two genuine coverage gaps identified for follow-up.*

---

## 1. Bibliography

- **Setiadi, T., Yaakub, M. R., & Abu Bakar, A.** (2025). "Community preserving sparsification
  based on K-core for enhanced community detection in attributed networks."
  *International Journal of Advances in Intelligent Informatics*, **11**(4), 550-566.
  DOI: 10.26555/ijain.v11i4.2209. CC-BY-SA 4.0.
- Affiliations: Universitas Ahmad Dahlan (Yogyakarta, ID); Universiti Kebangsaan Malaysia (MY).
- Timeline: received 2025-09-05, accepted 2025-10-17, online 2025-11-17 (~6-week review).
  Funded by FRGS/1/2020/ICT02/UKM/01/2.
- **No code or data release.** No supplementary material. Method is described in prose + 7 equations.
- Local copy: references/kcore_ijain2026.pdf

---

## 2. STEELMAN — what they claim and their strongest evidence

### 2.1 The claim
Sparsification, done in a *community-preserving and attribute-aware* way, **improves** community
detection quality (modularity, NMI) **and** semantic cohesion (purity, entropy) **while** cutting
runtime, on attributed networks from 34 to 107,614 nodes. This is a direct contradiction of our
"sparsification does not help" conclusion, in a regime (attributed networks) and with an operator
family (k-core) we never tested.

### 2.2 The method (CPSK), as written
Given `G = (V, E, A)`, hyperparameters `k` (core threshold), `theta in (0,1]` (retention),
`alpha in [0,1]` (attribute/structure trade-off):

1. **Attribute weighting** (Eq. 1-3): `w(u,v) = alpha*base_weight + (1-alpha)*Jaccard(A_u, A_v)`.
2. **Preliminary partition** (§3.2.1.2): run **Louvain on the full weighted graph G** -> `Pi_pre`.
   "This partition serves only as a heuristic and does not constrain the final community structure."
3. **k-core** (Eq. 4): `V_k = {v : deg_{G_k}(v) >= k}`.
4. **Intra-community restriction** (Eq. 5): `E_intra = {(u,v) in E : Pi_pre(u) = Pi_pre(v)}`.
   *All inter-community edges of `Pi_pre` are discarded here.*
5. **theta-sampling** (Eq. 6): keep a fraction theta of `E_intra`, **uniformly at random**.
6. **Build H** from `V_k` + retained edges.
7. **Core-periphery reconnection** (§3.2.2.5): "each core node `u in V_k` is reconnected with its
   original neighbors missing in `H`, along with their edges."
8. **Detect on H** (§3.2.3) with Louvain; report metrics.

### 2.3 Datasets (6)
Karate (34/78), Polblogs (1,222/16,714), Citeseer (3,327/4,732), Cora (2,708/5,429),
IMDB (~19,000/96,000), Gplus (107,614/13,673,453).
**Karate and Polblogs attributes are never specified.**

### 2.4 Metrics
Modularity (Eq. 7, standard Newman); Purity (Eq. 8); Entropy (Eq. 9);
**NMI** (Eq. 10, plain `2I(A,B)/(H(A)+H(B))`); wall-clock time, stated to cover
"the efficiency of the entire pipeline." Results averaged over 10 runs.

### 2.5 Baselines
SAC, EVA, MAGCN, KnnA, CDSSA, SAS-LP — six *attributed community detection* methods.
**There is no unsparsified baseline.** Plain Louvain on G is never reported, despite being
step 2 of their own pipeline.

### 2.6 Strongest evidence, with numbers
- **Modularity (Fig. 2)** — CPSK best on 6/6: Karate ~0.63, Polblogs ~0.50, Citeseer ~0.98,
  Cora ~0.96, IMDB ~0.89, Gplus ~0.98.
- **NMI (Fig. 3)** — CPSK best on 5/6.
- **Purity (Table 2)** — 1.0000 / 0.9964 / 0.9427 / 0.9677 / 0.9541 / 0.8105; best on 4/6.
- **Entropy (Table 3)** — 0.0014 / 0.0163 / 0.1417 / 0.1166 / 0.1236 / 0.5639; best on 3/6.
- **Runtime (Fig. 4)** — fastest on Karate, Citeseer, Cora, Gplus.
- **Sensitivity (Figs. 5-7)** — "robust": Q flat across **k = 1...72** and across
  **alpha = 1.0...0.1**; as theta decreases, purity -> 1 and entropy -> 0 while Q stays high.

**The steelman in one sentence:** *a deterministic, structure-and-semantics-aware preprocessing
step yields uniformly better modularity, better ground-truth agreement, better attribute
homogeneity, and lower runtime than six specialised attributed methods, across three orders of
magnitude of network size, and is insensitive to all three of its hyperparameters.*

That last property is where it dies.

---

## 3. AUDIT against our artifact list

| # | Control | Status | Evidence |
|---|---------|--------|----------|
| **I** | Score partition on ORIGINAL graph | **COMMITTED** (provably) | §3.2.3: detection "performed on the sparsified graph H, yielding the final partition"; Eq. 7 defines Q with "M is the total number of edges in **the graph**". No transfer-back sentence exists anywhere. Proof by arithmetic: reported karate Q ~0.63, but the exact maximum modularity of the karate graph over all partitions is **0.4198** (verified: best-of-500 Louvain = 0.4198). Also Polblogs ~0.50 vs known max ~0.427. Aggravating factor: Eq. 5 deletes *every inter-community edge of Pi_pre*, i.e. exactly the edges forming Q's penalty term. Artifact I in its most extreme possible form. |
| **II** | Granularity / resolution-matched control | **COMMITTED + SILENT** | Number of detected communities **never reported for any dataset or parameter setting**. Purity and entropy are monotone in granularity by construction — a singleton partition scores purity 1.0, entropy 0.0. CPSK reports purity **1.0000** on Karate. §4.6.1 reads the artifact's signature backwards: "increasing the level of sparsification leads to purity approaching 1 and entropy decreasing significantly toward zero... This indicates that... the community structure remains well preserved." |
| **III** | Chance-corrected metrics | **COMMITTED** | Eq. 10 is plain NMI. AMI, ARI, and "chance" appear nowhere. Given the reproduction produces thousands of clusters, the chance floor is doing most of the work. |
| **Cost accounting** | Sparsifier's own cost | **COMMITTED — worse than silent** | §3.2.1.2 runs **Louvain on the full graph G** as preprocessing. §4.5 lists this as "linear in the number of edges" — never counted as a detector run. §3.3 states timing covers "the efficiency of **the entire pipeline**." A pipeline containing a complete run of the baseline detector cannot be faster than that detector. Gplus (13.67M edges) at ~2-5 s for Jaccard weighting + Louvain + k-core + sampling + second Louvain is not credible. |
| **Baseline strength** | Fair / best-of-N | **COMMITTED** | No no-sparsification baseline exists. 10 runs averaged (credit) but no variance, CI, or significance test; no best-of-N; no matched wall clock. |
| **Config-model null** | Randomised control | **ABSENT** | Not run. **We ran it for them (§4) and the method fails catastrophically.** |
| **Label leakage** | Attributes != ground truth | **CANNOT DETERMINE (suspicious)** | Attributes specified for Citeseer/Cora/IMDB/Gplus but **not Karate or Polblogs**. Polblogs' only standard node attribute is political affiliation, which is also its ground-truth label; if used, alpha<1 injects the NMI target into the sparsifier. Karate has no standard attributes; if the club split was used, purity (1.0000) and NMI (~0.70) are circular. |
| **Node-set truncation** (*new — Artifact IV candidate*) | Partition covers same node set as reference | **SILENT** | k-core removes **nodes**, not just edges. `|V_H|` never reported. NMI/purity computed on a reduced, denser node set are not comparable to a partition of all of V. Our Artifacts I-III all implicitly assume matched node coverage. See §6. |
| **Ingredient ablation** | Do the novel components do anything? | **SELF-REFUTED** | Fig. 6: metrics flat to 3 d.p. across **k = 1...72** — the namesake parameter has **zero** measurable effect. Fig. 7: alpha = 1.0 (attributes entirely ignored) gives the same numbers as alpha = 0.5 — the attribute component also inert. Authors read both as "robustness." The only active ingredient is theta, i.e. uniform-random deletion of intra-community edges of a leaked Louvain partition. |

**Internal inconsistency.** Fig. 6(b) reports Gplus purity ~0.42 / entropy ~0.81 at theta = 0.5,
while Table 2/3 report 0.8105 / 0.5639 for the same dataset. The headline table therefore uses a
more aggressive theta than the sensitivity analysis. theta is never stated for the main results.

---

## 4. REPRODUCTION — the falsification

CPSK was reimplemented from the prose (no code exists). §3.2.2.5 is ambiguous, so **both readings
were tested**; they bracket the method and neither is a useful sparsifier.

**Reading A ("maximal reconnection")**: `m_H = m_G` in every case; theta has zero effect. The
method is a no-op returning G. Does not reproduce the paper's numbers.

**Reading B ("minimal reconnection")**: reproduces the paper closely; taken as operative.

### B.1 — Does the headline modularity require community structure? NO.

| Graph | m_G | m_H | #comms | plain Louvain #comms | **Q reported (on H)** | **Q honest (on G)** | Q plain Louvain (G) |
|---|---|---|---|---|---|---|---|
| karate (real) | 78 | 29 | 13 | 4 | **0.6831** | 0.2720 | **0.4198** |
| ER n=2708 m=5429 (**no structure**) | 5,429 | 1,631 | 1,018 | 84 | **0.9941** | 0.3026 | 0.5243 |
| ER n=19000 m=96000 (**no structure**) | 96,000 | 14,739 | 4,832 | 25 | **0.9699** | 0.1600 | 0.2557 |
| BA n=5000 m=15k (**no structure**) | 14,991 | 3,346 | 1,779 | 25 | **0.9744** | 0.2295 | 0.3872 |

- The paper's headline values (Citeseer ~0.98, Gplus ~0.98, Cora ~0.96, IMDB ~0.89) are
  **reproduced on graphs containing no community structure at all.**
- karate: reproduction gives 0.6831 vs the paper's ~0.63 — within 0.05, strong evidence the reading
  is correct. Both are far above the graph's exact optimum, 0.4198.
- Under **honest transfer scoring**, CPSK is *worse than plain Louvain in every row*
  (ER-19k: 0.1600 vs 0.2557, delta = -0.096). **The sign flips.**

### B.2 — The theta sweep reproduces the paper's Fig. 5 on a structure-free graph
(ER n=19,000, m=96,000, k=2)

| theta | m_H | #communities | Q reported | Q honest |
|---|---|---|---|---|
| 1.0 | 29,472 | 25 | 0.9438 | 0.2557 |
| 0.8 | 23,579 | 822 | 0.9446 | 0.2459 |
| 0.6 | 17,684 | 2,940 | 0.9519 | 0.2052 |
| 0.4 | 11,794 | 7,312 | 0.9903 | 0.1244 |
| 0.2 | 5,901 | 13,100 | 0.9994 | 0.0615 |
| 0.1 | 2,954 | 16,046 | 0.9994 | 0.0307 |

The paper's Fig. 5 pattern is exactly this: the partition shatters from 25 to **16,046** pieces on
19,000 nodes, and `Q -> 1 - 1/n_c` is a mathematical certainty, not a finding. Honest Q falls by
88% over the same sweep. Artifact II in its purest recorded form.

### B.3 — The k sweep reproduces Fig. 6: the k-core is inert
(same ER graph, theta = 0.5)

| k | m_H | #communities | Q reported |
|---|---|---|---|
| 1 | 14,736 | 4,834 | 0.9700 |
| 2 | 14,739 | 4,832 | 0.9699 |
| 3 | 14,755 | 4,821 | 0.9695 |
| 5 | 14,932 | 4,644 | 0.9702 |
| 8 | 14,736 | 1,591 | 0.9700 |

Q varies by 0.0007 across k. **The k-core component of "K-core-based sparsification" does nothing.**

### Reproduction script (preserved — scratchpad is session-scoped)

```python
import networkx as nx, random
from networkx.algorithms.community import louvain_communities, modularity

def pmapof(cs):
    d = {}
    for i, c in enumerate(cs):
        for v in c: d[v] = i
    return d

def cpsk_min(G, k=2, theta=0.5, seed=0):
    """CPSK, minimal reading of the core-periphery reconnection step (3.2.2.5)."""
    rng = random.Random(seed)
    pre = louvain_communities(G, seed=seed, weight=None); pm = pmapof(pre)   # step 2: Louvain on FULL G
    Vk = set(nx.k_core(G, k=k).nodes())                                      # step 3
    E_intra = [(u, v) for u, v in G.edges() if pm[u] == pm[v]]               # step 4: drop ALL inter edges
    rng.shuffle(E_intra); E_keep = E_intra[:int(round(theta * len(E_intra)))]# step 5: uniform random
    H = nx.Graph(); H.add_nodes_from(Vk); H.add_edges_from(E_keep)           # step 6
    for u in list(Vk):                                                       # step 7: reconnection
        for w in G.neighbors(u):
            if w not in H: H.add_node(w); H.add_edge(u, w)
    final = louvain_communities(H, seed=seed, weight=None); fm = pmapof(final)
    Q_H = modularity(H, final, weight=None)                                  # what the paper reports
    grp = {}
    for v in G.nodes(): grp.setdefault(fm.get(v, ('s', v)), set()).add(v)
    Q_G = modularity(G, list(grp.values()), weight=None)                     # honest transfer scoring
    plain = louvain_communities(G, seed=seed, weight=None)
    return dict(mG=G.number_of_edges(), mH=H.number_of_edges(), kH=len(final),
                kG=len(plain), Q_H=Q_H, Q_G=Q_G,
                Q_plain=modularity(G, plain, weight=None))

# karate exact optimum check (nx karate graph IS weighted; must pass weight=None)
K = nx.karate_club_graph()
print(max(modularity(K, louvain_communities(K, seed=s, weight=None), weight=None)
          for s in range(500)))   # -> 0.4198, matching the literature exact optimum
```

---

## 5. VERDICT

**(d) Nothing survives** — the central claim is an artifact, reproduced on structure-free nulls.
**With genuine (b)/(c) caveats that this paper does not fill.**

Ranked failure list:
1. **Artifact I, provably.** Reported modularity exceeds the mathematical maximum of the graph
   being described (karate 0.63 vs 0.4198). Not a matter of interpretation.
2. **Partition leakage disguised as sparsification.** The "sparsifier" deletes the inter-community
   edges of a Louvain partition of the *full* graph. The quality metric's penalty term is removed
   by construction. Q ~0.97 follows on graphs with no communities.
3. **Artifact II, unguarded and misread.** Cluster counts never reported; purity/entropy monotone
   in granularity; the theta-sweep figure is the artifact's signature, interpreted as a result.
4. **Artifact III.** Plain NMI, no chance correction, against partitions with thousands of clusters.
5. **Cost claim structurally impossible.** The pipeline contains a full run of the detector it
   claims to accelerate, and claims to time "the entire pipeline."
6. **No unsparsified baseline exists.**
7. **Both title ingredients are empirically inert** by the paper's own Figs. 6 and 7.

Honest concessions to the authors:
- Averaging over 10 runs is more than most comparable papers do.
- The *architecture* (cheap first-pass partition -> guided reduction -> re-detect) is a close cousin
  of our seeding pipeline (Exp C/N/R) — the one place in our whole study where a genuine gain
  (email-Enron, +0.008) survived every control. The idea is not stupid; the evaluation destroys it.
- The attributed regime and coreness-based pruning *are* real gaps in our coverage. See §6.

**Do not cite this as "a claim we refuted."** Cite it as a contemporary worked example of all three
artifacts co-occurring, and — for a single devastating exhibit — as the case where the reported
modularity provably exceeds the graph's maximum.

---

## 6. Is k-core pruning different in kind? — and the Exp W interaction

**Yes, the operator is genuinely different in kind** from DSpar / L-Spar / uniform / ER-sampling:

| Property | DSpar / L-Spar / ER-sampling | k-core pruning |
|---|---|---|
| Randomness | stochastic edge sampling | **deterministic** |
| Unit removed | edges | **nodes** (whole low-coreness periphery), edges follow |
| Locality of score | local (endpoint degrees, Jaccard) | **globally recursive** (coreness is a fixed point of peeling) |
| Structure | flat retention rate alpha | **nested hierarchy** (k-cores nested; monotone in k) |
| Relation to degree | degree *is* the signal (DSpar) | coreness != degree — a hub whose neighbours are all leaves has coreness 1 |

That last row matters: coreness measures *mutually reinforcing* density, so a coreness sparsifier is
not a repackaged degree sparsifier, and our Exp B result (degree-preserving nulls reproduce DSpar's
gain 17/17) does **not** automatically transfer. A coreness-preserving null would be needed.

**But this paper provides zero evidence about k-core pruning**, because its own Fig. 6 shows the k
parameter has no effect across k = 1...72 — confirmed in our reproduction (Q varies by 0.0007
across k = 1...8). The paper is named after its inert component.

**Interaction with Exp W and Exp O — a falsifiable prediction.**
- CPSK's stated premise (§4.5) is verbatim **Mohammad's core-preservation hypothesis from Exp O**,
  which failed 4 of 5 pre-registered predictions.
- **Exp W** showed fragments under sparsification are **mid-degree**, hubs are never shed, and the
  configuration-model null is **more** leaf-selective than the real graph in 8/8 cells:
  **real community structure holds low-degree nodes in place.**
- k-core pruning deliberately removes the low-coreness periphery, i.e. precisely the nodes Exp W
  says real structure is actively holding. **Prediction (pre-registerable):** coreness-based pruning
  at matched retention should damage recovery *more* on real graphs than on their degree-preserving
  nulls — the opposite sign to CPSK's claim, and the opposite sign to the DSpar pattern from Exp B.

**Candidate new artifact — IV: node-set truncation.** Artifacts I-III all assume the partition
covers the same node set as the reference. Node-removing sparsifiers (k-core, and any peeling
method) break that: recovery scored on the surviving, denser subpopulation is measured on an easier
problem. CPSK never reports `|V_H|`. If we test coreness pruning, coverage must be an explicit
reported quantity and a matched-coverage control is mandatory. This is a genuine extension of our
framework that we would not have found without reading this paper.

---

## 7. Implied follow-up

### Exp Y2 — Coreness-based sparsification under the honest protocol *(the real gap)*
- **Operator.** Coreness-threshold retention: keep edges with `min(core(u), core(v)) >= k`; sweep k
  to hit our existing retention grid, directly comparable with DSpar / L-Spar / uniform / ER.
  Deterministic — no seed variance.
- **Networks.** The existing 15-17, plus attributed ones (Cora, Citeseer, Pubmed, ogbn-arxiv) to
  enter the untested regime.
- **Mandatory controls.** (i) honest transfer scoring [I]; (ii) resolution-matched original-graph
  control at matched cluster count [II]; (iii) AMI/ARI with explicit chance floor [III];
  (iv) **configuration-model null AND a coreness-preserving null** — the latter is new and is what
  makes this a test of coreness rather than of degree; (v) best-of-N at matched wall clock;
  (vi) end-to-end cost including peeling; (vii) **reported node coverage `|V_H|/|V|` and a
  matched-coverage control** [IV].
- **Pre-registered prediction (from Exp W).** Real graphs damaged more than degree-matched nulls;
  dQ negative at quality-preserving retention; no end-to-end speedup beyond 1.35x.
- **Kill criterion.** If coreness pruning shows a positive honest dQ or a chance-corrected recovery
  gain surviving the resolution-matched control on >=3 networks, our headline needs qualification.

### Exp Z — Granularity control for attribute-based metrics *(framework extension)*
- **Observation.** Purity and entropy are monotone in cluster count by construction; the entire
  attributed-CD literature (SAC, EVA, MAGCN, KnnA, CDSSA, SAS-LP, CPSK) reports them without a
  granularity control or a chance floor. Nobody runs the check.
- **Deliverable.** Define expected purity/entropy under a random partition with the same block-size
  distribution; report chance-corrected purity and a resolution-matched purity baseline. Re-score
  the published attributed-CD tables (Table 2/3 of this paper are a ready-made target).
- **Predicted result.** Ranking reshuffles; "high purity" collapses toward the chance line once
  cluster counts are matched. Artifact II/III generalised to a subfield that has never had it, and
  cheap — no large-scale compute needed.

### Exp Z2 — The pre-partition leakage control *(one paragraph, high value)*
- **Rule.** Any pipeline that computes a partition on G and then deletes inter-community edges
  before re-detecting must be compared against **simply reporting the pre-partition**, scored
  honestly on G.
- **Already demonstrated here.** karate: pre-partition (= plain Louvain) honest Q = 0.4198 vs CPSK
  post-sparsification honest Q = 0.2720. The sparsification step *destroys* 35% of the modularity
  its own preprocessing already found.
- Joins the **shuffled-weight null** (Exp V) as a second "mandatory check nobody runs."

### Where this lands in the paper
Add to Related Work / artifact catalogue as the most recent and most complete worked example of
Artifacts I+II+III co-occurring, with the karate arithmetic (0.63 > 0.4198) as the single-sentence
exhibit. Add Artifact IV (node-set truncation) to the framework. Attributed networks move from
"not covered" to "covered by Exp Y2/Z" if those run; otherwise state the limitation explicitly.
