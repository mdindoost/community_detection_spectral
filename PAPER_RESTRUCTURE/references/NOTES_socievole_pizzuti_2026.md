# NOTES — Socievole & Pizzuti 2026, "Effective resistance and kernel-based graph
# sparsification for community detection in complex networks"

**Read in full 2026-07-27 from the publisher PDF supplied by Mohammad.**
Local copy: `references/socievole_pizzuti_softcomputing2026.pdf`
*Soft Computing* 30:2109–2133, 2026. DOI 10.1007/s00500-025-10734-5.
Accepted 25 April 2025, published online 18 February 2026. ICAR-CNR, Rende, Italy.

**Verdict: NOT a D1 instance. Credit them, do not prosecute them. And their own tables
independently corroborate this paper's central negative.**

---

## 1. What the method is

`KOmeGAnet+`, a two-step pre-processing plus a genetic algorithm:

1. weight $G$ by effective resistance between each pair of nodes; apply a kernel $k$ to the
   resistance matrix to get $K$; sparsify $K$ by **weight thresholding**, cutting a fraction $nr$
   of lowest-weight edges, with $nr$ chosen by maximizing the minimum absolute spectral
   similarity (MASS, Yan et al. 2018);
2. run a GA on the sparsified weighted matrix, maximizing weighted modularity.

So it both **reweights and deletes**. §4.1, verbatim: "Given the input graph $G$ and its reduced
graph $G'$, find a partition $C = \{C_1,...,C_k\}$ of $G'$ in $k$ communities such that the
weighted modularity of $C$ is maximized."

## 2. Their evaluation, and why it clears our central control

§5.2, verbatim: **"To assess the quality of the partitions found by the community detection, the
Normalized Mutual Information (NMI) is used."**

NMI is computed against ground-truth labels, not against a graph. **Artifact I does not apply to
their headline metric.** They also report modularity $Q$ and the community count $k$ in every
results table, and they compare against unsparsified Louvain and Infomap throughout.

On which graph $Q$ is computed the paper is silent, but the numbers indicate the **original**
graph, which is the honest accounting. Table 14 gives Louvain/Infomap on LFR3 as
$[1, 0.405, 2]$ and Table 16 gives KOmeGAnet+ on LFR3 as $[1, 0.405, 2]$: identical NMI, identical
$Q$, identical $k$. Two methods that recover the same partition score the same $Q$, which only
happens if $Q$ is evaluated on a common graph. **[INFERRED, not stated by them — say "consistent
with" and not "they do".]**

**What is fair to say about their methodology:** they use NMI, which is not chance-corrected, on
partitions whose community counts differ from the reference (their own Table 10 shows ERKGA
drifting to 5, 6, 7 and 9 communities against a ground truth of 4 as $\mu$ rises). Hamann et al.
2016 warned specifically that NMI is inflated when sparsification fragments a graph. That is a
methodological point our §3.2 and §3.3 own, and it can be made without accusing anyone of the
self-scoring artifact.

## 3. THE CORROBORATION — their own tables show our negative

This is the reason to read the paper. Across every LFR-1000 experiment they run, **unsparsified
Louvain matches or beats the best sparsified pipeline**, on both NMI and modularity.

**Table 11** (LFR-1000, average edge density 0.06), NMI:

| $\mu$ | OmeGAnet (sparsified) | Louvain (untouched) |
|---|---|---|
| 0.1 | 1 | 1 |
| 0.2 | 0.9687 | **1** |
| 0.3 | 0.6012 | **1** |
| 0.4 | 0.4445 | **1** |
| 0.5 | 0.266 | **1** |
| 0.6 | 0.1637 | **0.4779** |

**Table 12**, modularity on the same networks: Louvain is $\geq$ OmeGAnet at every $\mu$
(0.6864/0.6864, 0.5421/0.532, 0.4462/0.296, 0.3452/0.1815, 0.2414/0.105, 0.1582/0.073).

**Table 14 and Table 16**, LFR-1000 at three densities, $\mu = 0.1$, as [NMI, $Q$, $k$]:

| Network | Density | best sparsified | Louvain/Infomap untouched |
|---|---|---|---|
| LFR1 | 0.08 | $[1, 0.5576, 3]$ | $[1, 0.5576, 3]$ |
| LFR2 | 0.25 | $[1, 0.5667, 3]$ | $[1, 0.5667, 3]$ |
| LFR3 | 0.49 | $[1, 0.405, 2]$ (KOmeGAnet+) | $[1, 0.405, 2]$ |

The sparsified pipeline never exceeds an unsparsified run of a standard detector. It ties it at
best. Their own §6.2 says so in words about the densest case: **"for the LFR3 network, OmeGANet
achieves only 0.3522 as NMI value, while Louvain and Infomap are able to perfectly match the
ground-truth."**

**What their gains actually are.** The improvements they report are of kernels and the GA over
ERGA, their own no-kernel baseline (e.g. LFR-1000 $\mu=0.1$: ERGA 0.099 against ERKGA 0.7813).
That is a comparison between two of their own variants, not between sparsify-then-detect and
detect. Read that way there is no conflict with our results at all.

## 4. Their density evidence, which is our second condition

Their LFR-1000 family spans **density 0.08 ($d=80$), 0.25 ($d=250$) and 0.49 ($d=500$)**, all far
above the transition our §4.6 locates. They state that sparsification is what rescues the dense
case: "the sparsification is needed as well when networks are dense as the results provided by
OmeGAnet suggest," and their abstract claims the method "is especially suited for dense graphs."

On the sparse side their LFR-128 networks have $d=8$, and there sparsification hurts: Table 8
gives ERKGA (no sparsification) above OmeGAnet (with it) at $\mu = 0.3, 0.4, 0.5, 0.6$
(0.8894/0.8049, 0.3791/0.3308, 0.2123/0.1314, 0.2005/0.0698). **Caveat: ERKGA and OmeGAnet differ
in kernels as well as in sparsification, so this contrast is confounded and must not be quoted as
a clean sparsification effect.** The clean contrasts are OmeGAnet vs OmeGAnet+ and KOmeGAnet vs
KOmeGAnet+ in Tables 15 and 16.

## 5. How to use this in §2

1. **Cite as current practice, credited.** A 2026 journal paper on exactly our pipeline, with an
   honest external metric and unsparsified baselines. It is the best evidence that the question is
   live, and it is not a straw man.
2. **Their dense-graph claim is our positive condition, independently arrived at.** "Especially
   suited for dense graphs" is their sentence, not ours.
3. **Their tables are corroboration for our negative**, and stronger for being someone else's
   data. Use with care: they never frame it as a comparison against an unsparsified baseline, so
   the observation is ours to make from their numbers.
4. **The methodological gap to name is NMI without chance correction or count matching**, not
   self-scoring. Do not put them in the D1 bucket.

## 6. Other things worth knowing

- Networks are small: Karate 34, Football 115, Dolphins 62, LFR-128, LFR-1000.
- No runtime or speedup is reported anywhere, despite the storage/computation motivation. Their
  conclusion concedes the MASS sweep itself is costly: "the procedure to choose the percentage of
  nodes to remove needs to compute the MASS and the connected components for several possible
  percentages of edge cuts."
- Their future work lists effective-resistance sampling (Spielman & Srivastava) as not yet tried.
- Companion paper: Socievole & Pizzuti, ASONAM 2024, DOI 10.1007/978-3-031-85386-9_14.
