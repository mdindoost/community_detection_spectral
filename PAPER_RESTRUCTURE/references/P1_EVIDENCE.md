# P1 EVIDENCE — Does anyone actually sparsify/subsample before community detection?

**Purpose.** Ground the Introduction/Background of a paper whose central question is
*"should you sparsify before community detection?"* (our experimental answer: no quality gain,
~1× speed, evaluation hazards). This file collects **real, citable, quoted** evidence that the
practice/claim exists. Nothing here is invented; anything not directly verified is marked
**[UNVERIFIED]** or **[PARAPHRASE]**.

Compiled 2026-07-25. Verification method: primary PDFs downloaded and text-extracted where
possible (arXiv, VLDB, WWW archives, author copies); publisher landing pages and Semantic Scholar
API otherwise. ACM DL and Springer block automated fetch (403) — items sourced only from those
are marked accordingly.

Legend for the "P1 relevance" line:
- **SUPPORTS** = establishes that the sparsify-before-community-detection practice/claim is real.
- **COMPLICATES** = the source is weaker/narrower than convenient, or partially pre-empts us.
- **PRIOR-ART RISK** = overlaps with something we claim to show.

---

## CATEGORY 1 — Sparsification-for-clustering claims (the direct ancestor of P1)

### 1.1 Satuluri, Parthasarathy & Ruan (2011) — the origin claim

**Citation.** Venu Satuluri, Srinivasan Parthasarathy, Yiye Ruan. "Local graph sparsification for
scalable clustering." *Proceedings of the 2011 ACM SIGMOD International Conference on Management of
Data (SIGMOD '11)*, pp. 721–732.
**DOI.** 10.1145/1989323.1989399 · https://dl.acm.org/doi/10.1145/1989323.1989399
**Impact.** 206 citations per the Semantic Scholar API (paperId
`eb66482fe6a11ddd83dcf9886e66a609eda6fd9c`); 169 per OpenAlex. Both retrieved 2026-07-25 — quote
whichever, but say which, and do not quote a Google Scholar figure we could not verify.

**Quotes — VERBATIM ABSTRACT.** ACM DL returns HTTP 403 to automated fetch; the abstract below was
retrieved from the **OpenAlex API** (`https://api.openalex.org/works/doi:10.1145/1989323.1989399`,
`abstract_inverted_index` reconstructed, retrieved 2026-07-25). Punctuation of the reconstructed
text is imperfect (OpenAlex strips some marks) but wording is the publisher's:

> "In this paper we look at how to sparsify a graph i.e. how to reduce the edgeset while keeping
> the nodes intact, so as to enable **faster graph clustering without sacrificing quality**. The
> main idea behind our approach is to preferentially retain the edges that are likely to be part of
> the same cluster. We propose to rank edges using a simple similarity-based heuristic that we
> efficiently compute by comparing the minhash signatures of the nodes incident to the edge. For
> each node, we select the top few edges to be retained in the sparsified graph. Extensive empirical
> results on several real networks and using four state-of-the-art graph clustering and community
> discovery algorithms reveal that our proposed approach **realizes excellent speedups (often in the
> range 10-50), with little or no deterioration in the quality of the resulting clusters. In fact,
> for at least two of the four clustering algorithms, our sparsification consistently enables higher
> clustering accuracies.**"

(Bold added by us. The last sentence is *the* claim P1 tests: not merely "no loss" but a **quality
gain** from sparsification, for community-discovery algorithms specifically.)

**Independently verified restatement of this paper's claim** (from a source we *did* read in full —
Hamann et al. 2016, arXiv:1601.00286, §"Related Work"):

> "One line of research attempts to sparsify graphs with the goal of speeding up data mining
> algorithms. [Satuluri et al., 2011] propose a local graph sparsification method with the
> intention of speedup and quality improvement of community detection. They suggest reducing the
> edge set to 10-20% of the original graph and use the Jaccard [similarity]..."

and

> "[Satuluri et al., 2011] face a similar problem as with their sparsification technique based on
> Jaccard Similarity ... they want to preserve the community structure. They propose a different
> solution: Each node u keeps the top ⌊d(u)^α⌋ edges incident to u, ranked according to their
> similarity."

**P1 relevance. SUPPORTS (strongest single citation).** This is the canonical published statement
of exactly the proposition P1 tests: prune 80–90% of edges, get 10–50× speedup, lose ~nothing in
quality. Our result (~1× speed, no quality gain) is a direct empirical response. The Hamann quote
lets us attribute "speedup **and quality improvement** of community detection" to Satuluri et al.
without relying on a paywalled abstract.

---

### 1.2 Hamann, Lindner, Meyerhenke, Staudt & Wagner (2016) — the systematic comparison

**Citation.** Michael Hamann, Gerd Lindner, Henning Meyerhenke, Christian L. Staudt, Dorothea
Wagner. "Structure-preserving sparsification methods for social networks." *Social Network Analysis
and Mining* 6(1):22, 2016. (Earlier version: ASONAM 2015; arXiv:1601.00286.)
**DOI.** 10.1007/s13278-016-0332-2 · https://arxiv.org/abs/1601.00286
**Verification.** Full PDF downloaded and text-extracted. All quotes below are verbatim.

**Quotes.**

> "Sparsification reduces the size of networks while preserving structural and statistical
> properties of interest. ... We contribute the first systematic conceptual and experimental
> comparison of edge sparsification methods on a diverse set of network properties."

> "All methods are evaluated on a set of social networks from Facebook, Google+, Twitter and
> LiveJournal with respect to network properties including diameter, connected components,
> **community structure**, multiple node centrality measures and the behavior of epidemic
> simulations. In order to assess the preservation of the community structure, we also include
> experiments on synthetically generated networks with ground truth communities."

> "Considering the preservation of the community structure we show that some of the newly
> introduced variants with local filtering are best for preserving the community structure while
> the variants without local filtering do not preserve the community structure in our experiments."

> "Sparsification can also be applied as an acceleration technique: By disregarding a large
> fraction of edges that are unimportant for the task, running times of graph and network analysis
> algorithms can be reduced."

**The evaluation-hazard warning they already published (important):**

> "Note that the frequently used normalized mutual information (NMI) measure reports higher
> similarity values for a larger number of found communities (see e.g. [Vinh et al., 2009]). This
> makes it unsuitable for comparing partitions on sparsified networks as we have to expect many
> small communities when a lot of edges are removed. As [Vinh et al., 2009] also show in their
> experiments, the adjusted rand index does not have these properties as it has an expected value
> of 0 for random partitions."

**Their partition-instability finding:**

> "Filtering the edges such that we can measure that the conductance of one of these many community
> structures is decreased most probably does not just make this structure clearer but does also
> lead the algorithm into finding different community structures. Therefore most methods lead to
> significantly different community structures."

> "Simmelian Backbones, Jaccard Similarity and algebraic distance prefer intra-cluster edges and
> thus do not keep global structures but with the added local filtering step they are able to
> enforce and retain a community structure as it was already shown for Jaccard Similarity.
> **However, the preserved community structure is not necessarily the same as the one the Louvain
> algorithm finds.**"

> "Random edge deletion performs surprisingly well and retains a wide range of properties, but more
> targeted methods can perform even better."

**P1 relevance. SUPPORTS + PRIOR-ART RISK (moderate).** Supports: it documents that a whole
literature of edge-sparsification methods exists and is explicitly evaluated on *community
structure*, and it independently attests Satuluri et al.'s quality-improvement claim.
Complicates/pre-empts: (a) they already warn that **NMI is inflated by fragmentation** — a cousin
of our granularity-artifact result, so we must cite them and position our resolution-matched
control as the *fix*, not the discovery of the hazard; (b) they already report that sparsification
"lead[s] the algorithm into finding different community structures" and that improved conductance
of a *fixed* partition does not mean the structure got clearer — conceptually adjacent to our
δ-null result, though they show it descriptively and offer no null model. They do **not** measure
end-to-end runtime, and they do **not** ask whether sparsify+detect beats detect-on-full-graph at
matched cost. Our contribution survives; the framing must acknowledge them.

---

### 1.3 Sotiropoulos & Tsourakakis (2021) — sparsify *because* you want communities

**Citation.** Konstantinos Sotiropoulos, Charalampos E. Tsourakakis. "Triangle-aware Spectral
Sparsifiers and Community Detection." *KDD '21: Proceedings of the 27th ACM SIGKDD Conference on
Knowledge Discovery & Data Mining*, pp. 1501–1509, 2021.
**DOI.** 10.1145/3447548.3467260 · author PDF:
https://tsourakakis.com/wp-content/uploads/2021/06/triangle-aware-spectral-sparsifiers-and-community-detection.pdf
**Verification.** Author PDF downloaded and text-extracted. Quotes verbatim.

**Quotes.**

> "Triangle-aware graph partitioning has proven to be a successful approach to finding communities
> in real-world data [8, 40, 51, 54]. But how can we explain its empirical success?"

> "For instance, Satuluri, Parthasarathy, and Ruan designed an elegant method for finding
> communities [40]. The core idea lies in reweighing each edge e = (u, v) in the graph by the
> Jaccard similarity coefficient, i.e., a function of the number of triangles t(u, v) the edge is
> contained in, and the endpoints' degrees deg(u), deg(v)."

> "In this work we advance the understanding of triangle-based graph partitioning in two ways.
> First, we introduce a novel triangle-aware sparsification scheme. Our scheme provably produces a
> spectral sparsifier with high probability [46, 47] on graphs that exhibit strong triadic closure,
> a hallmark property of real-world networks."

> "we observe that the Jaccard similarity of an edge used by Satuluri [40], and the closely related
> Tectonic similarity measure introduced by Tsourakakis et al. [51] provide consistently good
> signals of whether an edge is contained within a community or not."

**P1 relevance. SUPPORTS (strong).** A top-tier venue paper, ten years after Satuluri, whose entire
premise is that sparsification is a legitimate route to community detection and whose open question
is *why* it works — i.e., the community treats "sparsify then detect" as an established, effective
pipeline in need of explanation, not as a proposition in need of testing. This is exactly the
premise P1 challenges.

---

### 1.4 Chen, Ye, Vedula, Bronstein, Dreslinski, Mudge & Talati (2023/24) — the VLDB benchmark

**Citation.** Yuhan Chen, Haojie Ye, Sanketh Vedula, Alex Bronstein, Ronald Dreslinski, Trevor
Mudge, Nishil Talati. "Demystifying Graph Sparsification Algorithms in Graph Properties
Preservation." *Proceedings of the VLDB Endowment* 17(3):427–440, 2023 (presented VLDB 2024).
**DOI.** 10.14778/3632093.3632106 · arXiv:2311.12314 · PDF: https://www.vldb.org/pvldb/vol17/p427-chen.pdf
**Verification.** VLDB PDF downloaded and text-extracted. Quotes verbatim.

**Quotes.**

> "Graph sparsification can be applied to greatly reduce the run time of graph algorithms by
> substituting the full graph with a much smaller sparsified graph, without significantly degrading
> the output quality."

> "Our study shows that there is no one sparsifier that performs the best in preserving all graph
> properties"

On community detection specifically (they use Louvain, and an F1 clustering-similarity metric):

> "We employ the widely recognized Louvain method [12] for community detection, assuming the number
> of communities is unknown, and use the number detected in the original graph as the ground truth.
> ... As the prune rate increases, the graph becomes increasingly disconnected, and the number of
> communities consistently rises."

> "For all sparsifiers, F1 similarity decreases as the prune rate increases."

> "These sparsifiers share a focus on local edges, and locally similar vertices more likely to
> belong to the same community."

**P1 relevance. SUPPORTS + PRIOR-ART RISK (moderate).** Supports: a 2023/24 VLDB paper states the
sparsify-to-accelerate premise flatly ("greatly reduce the run time ... without significantly
degrading the output quality"), confirming the claim is live and mainstream in the data-management
community. Pre-empts partially: they already report monotone degradation of clustering agreement
with prune rate and the fragmentation-driven explosion in community count — the same mechanism
behind our granularity artifact. They do **not** report wall-clock end-to-end speedups for the
community-detection pipeline, do **not** use a resolution-matched control, and do **not** test
against a null model; and their framing is "pick the right sparsifier per task," not "don't."

---

### 1.5 Downstream/derivative works that adopt the practice

These matter because they show the practice propagating, not just the original claim.

- **Banf (2018), SparseClust.** "Network Module Detection using Recursive Local Graph Sparsification
  and Clustering." Preprints 2018, DOI 10.20944/PREPRINTS201808.0421.V1. Verbatim abstract (via
  Semantic Scholar API): *"Here we present a fast and highly scalable community structure preserving
  network module detection that recursively integrates graph sparsification and clustering. Our
  algorithm, called SparseClust, participated in the most recent DREAM community challenge on
  disease module identification..."* Code: https://github.com/mbanf/NetworkModuleDetection ("Highly
  scalable network clustering based on recursive jaccard similarity graph sparsification").
  **SUPPORTS** — sparsify-then-cluster deployed in a real biological-network challenge.

- **`backbone` R package (CRAN).** Domagalski, Neal & Sagan, "backbone: An R package to extract
  network backbones," *PLoS ONE* 17(5):e0269137, 2022 (61 citations per Semantic Scholar).
  The package ships `sparsify.with.lspar()`, documented verbatim as *"the L-spar backbone described
  by Satuluri et al. (2011)"*, equivalent to `sparsify(escore = "jaccard", normalize = "rank",
  filter = "degree", umst = FALSE)`; the manual's worked example annotates the sparsified plot with
  *"Clearly visible communities."*
  https://search.r-project.org/CRAN/refmans/backbone/html/sparsify.with.lspar.html
  **SUPPORTS (software guidance).** Satuluri's method is a one-line call in a mainstream CRAN
  package, and the documentation's own example frames the payoff as community visibility.

- **NetworKit `sparsification` module.** Ships `LocalDegreeSparsifier`, `LocalSimilaritySparsifier`
  ("the Local Similarity sparsification approach introduced by Satuluri et al."), `SCANSparsifier`,
  Simmelian backbones, etc. Docs: https://networkit.github.io/dev-docs/python_api/sparsification.html
  and tutorial https://networkit.github.io/dev-docs/notebooks/Sparsification.html
  These are the reference implementations from Hamann et al. 2016 (their paper: *"we have published
  efficient parallelized implementations and a framework for such methods as part of the NetworKit
  open-source tool suite"*). **SUPPORTS (software availability).**
  *Caveat, stated honestly:* the NetworKit sparsification tutorial itself does **not** instruct
  users to sparsify before community detection; it documents edge scorers. The community-detection
  connection there is via the Hamann et al. paper, not the tutorial text.

- **Serrano, Boguñá & Vespignani (2009) disparity filter.** "Extracting the multiscale backbone of
  complex weighted networks," *PNAS* 106(16):6483–6488. DOI 10.1073/pnas.0808904106.
  Reported property list (from the PNAS abstract, via search index — **[VERIFY VERBATIM]**): the
  filter *"reduces the number of edges in the original network significantly keeping at the same
  time almost all the weight and a large fraction of nodes, while preserving the cut-off of the
  degree distribution, the form of the weight distribution, and the clustering coefficient."*
  **COMPLICATES.** Notably, the disparity filter's own stated guarantees are about *weights,
  degree/weight distributions and clustering coefficient* — **not** community structure. Papers
  that combine it with community detection are downstream (e.g. Ghalmane et al., "Extracting
  backbones in weighted modular complex networks," *Scientific Reports* 10:15095, 2020,
  https://www.nature.com/articles/s41598-020-71876-0 — which proposes backbone extraction methods
  that *exploit* community structure rather than feeding a detector). Use the disparity filter as
  context for "network filtering is standard practice," not as a sparsify-then-detect claim.

- **Recent (2025–2026) sparsify-for-community-detection papers** — the practice is still being
  published:
  - Jayanta Pari, Pratibha Bhandari, Soumyendu Raha. "Effective Resistance-Based Graph
    Sparsification and Community Detection." arXiv:2606.26766v1, June 2026.
    Verbatim motivation: *"Many real-world networks typically have high edges density... Due to
    their dense nature, the analysis of these types of networks is computationally heavy. So some
    preprocessing of data is required to solve the problem."*
    Verbatim negative note in their own results: *"It appears that for the synthetic data,
    sparsification does not play much of a role because the networks are inherently generated with
    community structure; so the edges are not noisy, and they follow well-defined probabilities."*
    **SUPPORTS + small PRIOR-ART note** — an active 2026 paper still selling the pipeline, but which
    itself concedes sparsification is inert on synthetic (LFR/SBM-like) benchmarks. That concession
    is a useful corroboration of our LFR finding and should be cited as such.
  - "Effective resistance and kernel-based graph sparsification for community detection in complex
    networks," *Soft Computing*, 2025, DOI 10.1007/s00500-025-10734-5.
    **[UNVERIFIED — Springer blocked automated fetch; title/venue confirmed via search index only,
    abstract not captured. Do not quote without retrieving the paper.]**
  - Setiadi et al. "Community preserving sparsification based on K-core for enhanced community
    detection in attributed networks," *International Journal of Advances in Intelligent
    Informatics* 11(4):550–566, 2025. https://ijain.org/index.php/IJAIN/article/view/2209
    **[UNVERIFIED — title/venue confirmed; abstract not captured verbatim.]**
  - Jesse Laeuchli. "Fast Community Detection with Graph Sparsification." *PAKDD 2020*, LNAI,
    DOI 10.1007/978-3-030-47426-3_23 (open access via PMC7206315). Verified: examines "the types of
    errors that can be tolerated using spectral methods while still recovering the communities" from
    "dropping edges using different sparsification strategies," reporting "essentially an order of
    magnitude speed-up." Cites Spielman–Srivastava, **not** Satuluri.
    **SUPPORTS** — an explicit "sparsify to make community detection fast" paper with a claimed 10×.

- **Gionis, Rozenshtein & Tatti (2017), community-aware sparsification.** "Community-aware network
  sparsification." *SIAM International Conference on Data Mining (SDM) 2017*; arXiv:1701.07221.
  Verbatim opening: *"Network sparsification aims to reduce the number of edges of a network while
  maintaining its structural properties: shortest paths, cuts, spectral measures, **or network
  modularity**. Sparsification has multiple applications, such as, speeding up graph-mining
  algorithms, graph visualization, as well as identifying the important network edges."*
  **SUPPORTS** — a top data-mining venue lists *modularity* among the properties sparsification is
  supposed to maintain, and *speeding up graph-mining algorithms* as its purpose.
  *Note:* their formulation takes communities as **input**, so it is not itself a
  sparsify-then-detect pipeline. Cite for the framing sentence, not as a detector.

- **Wu & Chen (2020), GSGAN.** "Graph Sparsification with Generative Adversarial Network."
  *2020 IEEE International Conference on Data Mining (ICDM)*, DOI 10.1109/ICDM50108.2020.00172; arXiv:2009.11736. (Venue/DOI verified via OpenAlex.) Verified from the arXiv PDF; verbatim:
  > "Graph sparsification aims to reduce the number of edges of a network while maintaining its
  > accuracy for given tasks. In this study, we propose a novel method called GSGAN, which is able
  > to sparsify networks for community detection tasks."

  > "Despite of only using far fewer edges (e.g., 5%), applying clustering algorithms of community
  > detection on our sparsified graph receives comparable or even better results than on the
  > original graph. Moreover, the execution time of community detection can be considerably reduced
  > (e.g., by nearly an order of magnitude) when applying on the sparsified graph."

  > "L-Spar [1] develops a sparsification algorithm specifically for graph clustering."

  **SUPPORTS (very strong).** This is the cleanest modern restatement of the exact proposition P1
  tests: 5% of edges, *better* community detection, ~10× faster. If our result is "no quality gain,
  ~1× speed," this is the sentence we are contradicting.

---

## CATEGORY 2 — Does sampling preserve community structure? (the sampling line)

### 2.1 Leskovec & Faloutsos (2006) — the graph-sampling starting point

**Citation.** Jure Leskovec, Christos Faloutsos. "Sampling from large graphs." *KDD '06:
Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data
Mining*, pp. 631–636, 2006. DOI 10.1145/1150402.1150479.
PDF: https://cs.stanford.edu/people/jure/pubs/sampling-kdd06.pdf (verified, quotes verbatim)

**Quotes.**

> "Given a huge real graph, how can we derive a representative sample? There are many known
> algorithms to compute interesting measures (shortest paths, centrality, betweenness, etc.), but
> several of them become impractical for large graphs. Thus graph sampling is essential."

> "the practical conclusions from our work are: Sampling strategies based on edge selection do not
> perform well; simple uniform random node selection performs surprisingly well. Overall, best
> performing methods are the ones based on random-walks and 'forest fire'; they match very
> accurately both static as well as evolutionary graph patterns, with sample sizes down to about
> 15% of the original graph."

**P1 relevance. SUPPORTS + COMPLICATES.** Supports: this is the paper that legitimized "shrink the
graph, then analyse it" and is the citation Neo4j GDS uses to justify its sampler (see §4.1).
Complicates, and we must be honest: the property list they check is degree/eigenvalue/clustering-
coefficient/hop-plot distributions — **community structure is not among the evaluated properties**.
So Leskovec & Faloutsos should be cited as the *provenance of the practice*, not as a claim that
sampling preserves communities.

### 2.2 Maiya & Berger-Wolf (2010) — sampling *for* community structure

**Citation.** Arun S. Maiya, Tanya Y. Berger-Wolf. "Sampling community structure." *WWW '10:
Proceedings of the 19th International Conference on World Wide Web*, pp. 701–710, 2010.
DOI 10.1145/1772690.1772762 ·
PDF: https://archives.iw3c2.org/www2010/proceedings/www/p701.pdf (verified, verbatim)

**Quotes.**

> "We propose a novel method, based on concepts from expander graphs, to sample communities in
> networks. We show that our sampling method, unlike previous techniques, produces subgraphs
> representative of community structure in the original network. These generated subgraphs may be
> viewed as stratified samples in that they consist of members from most or all communities in the
> network. Using samples produced by our method, we show that the problem of community detection
> may be recast into a case of statistical relational learning. We empirically evaluate our approach
> against several real-world datasets and demonstrate that our sampling method can effectively be
> used to infer and approximate community affiliation in the larger network."

> "The networks of today can be so large that analysis of the network in its entirety can be
> intractable and impractical. How, then, should one proceed in analyzing and mining these
> networks? Traditional approaches include designing more efficient algorithms or leveraging
> computing power through parallelization or distributed computing. Unfortunately, these existing
> methods are not always easily available as an option. Another approach that has received very
> little attention is sampling."

**P1 relevance. SUPPORTS (strong, for the *node*-sampling half of P1).** An explicit WWW paper
whose thesis is "the network is too big to run community detection on, so sample first and infer
community affiliation from the sample." Note the phrase "unlike previous techniques" — a direct
admission that generic samplers do *not* preserve community structure, which is itself useful to us.

### 2.3 Blagus, Šubelj, Weiss & Bajec (2015) — sampling *manufactures* community structure

**Citation.** Neli Blagus, Lovro Šubelj, Gregor Weiss, Marko Bajec. "Sampling promotes community
structure in social and information networks." *Physica A: Statistical Mechanics and its
Applications* 432:206–215, 2015. arXiv:1504.03097. (Verified from arXiv PDF; verbatim.)

**Quotes.**

> "Any network studied in the literature is inevitably just a sampled representative of its
> real-world analogue. Additionally, network sampling is lately often applied to large networks to
> allow for their faster and more efficient analysis. Nevertheless, the changes in network structure
> introduced by sampling are still far from understood. ... However, despite these notable
> differences, the structure of sampled networks exhibits **stronger characterization by
> community-like groups than the original networks**, irrespective of their type and consistently
> across various sampling techniques. Hence, **rich community structure commonly observed in social
> and information networks is to some extent merely an artifact of sampling.**"

> "After applying network sampling techniques, sampled networks expectedly contain fewer and
> smaller groups. However, the sampled networks exhibit stronger characterization by community-like
> groups than the original networks. We have shown that the changes in the node group structure
> introduced by sampling occur regardless of the network type and consistently across different
> sampling techniques."

**P1 relevance. SUPPORTS OUR CONCLUSION + PRIOR-ART RISK (real, but narrow).** This is the closest
published relative of our "evaluation hazard" finding: apparent community structure gets
*stronger* after subsampling, and that strengthening is an artifact. Differences that keep our
contribution distinct: (i) they study *node/link sampling of the whole network* as a data-collection
issue, not sparsification as a deliberate accelerator; (ii) their diagnostic is group-structure
type (community vs. module) via a mixture/blockmodel-style analysis, **not** modularity deltas,
**not** a degree-preserving configuration null, and **not** ground-truth recovery; (iii) they draw
no conclusion about runtime or about whether one *should* sparsify. We should cite them prominently
and honestly as the antecedent of the artifact framing.

### 2.4 Related sampling-preserves-communities line (secondary, lower priority)

- Ruohan Gao et al., "Graph Property Preservation under Community-Based Sampling," *GLOBECOM 2015*.
  https://ruohangao.github.io/assets/papers/CBS_globecom2015.pdf — proposes Community-Based
  Sampling; claim (per abstract, **[PARAPHRASE — not fetched verbatim]**) that CBS "preserves
  community-related graph properties very well." **SUPPORTS (weak).**
- Sarma Goyal et al. / Bhattacharya et al., "ComPAS: Community Preserving Sampling for Streaming
  Graphs," arXiv:1802.01614 (AAMAS 2018). **SUPPORTS (weak).** **[UNVERIFIED verbatim.]**
- Mall, Langone & Suykens, "FURS: Fast and Unique Representative Subset selection retaining
  large-scale community structure," *Social Network Analysis and Mining* 3(4):1075–1095, 2013.
  DOI 10.1007/s13278-013-0144-6. **SUPPORTS (weak).** **[UNVERIFIED verbatim.]**
- Network-epidemiology / survey-sampling branch: Rocha, Thorson, Lambiotte & Liljeros,
  "Respondent-driven sampling bias induced by community structure and response rates in social
  networks," *J. Royal Statistical Society Series A* 180(1):99–118, 2017; arXiv:1503.05826.
  **SUPPORTS THE HAZARD FRAMING** — community structure and sampling interact and bias estimates.
  Note this is about *estimation bias from* community structure, not about detecting communities on
  a sample, so it is context, not a core citation. **[UNVERIFIED verbatim.]**

---

## CATEGORY 3 — Practice at scale (industry and HPC)

### 3.1 Twitter/X SimClusters — production sparsification before community detection

**Citation.** Venu Satuluri, Yao Wu, Xun Zheng, Yilei Qian, Brian Wichers, Qieyun Dai, Gui Ming
Tang, Jerry Jiang, Jimmy Lin. "SimClusters: Community-Based Representations for Heterogeneous
Recommendations at Twitter." *KDD '20: Proceedings of the 26th ACM SIGKDD International Conference
on Knowledge Discovery & Data Mining*, pp. 3183–3193, 2020. DOI 10.1145/3394486.3403370.
(Author list and pages verified via the Crossref API, 2026-07-25.)
(53 citations, Semantic Scholar. ACM DL 403s automated fetch; the paper text was not retrieved.)

**Verified primary source instead: Twitter's own open-sourced production code documentation**
(`twitter/the-algorithm`, `src/scala/com/twitter/simclusters_v2/README.md`, fetched raw from
GitHub 2026-07-25). Verbatim:

> "Producer-producer similarity is computed as the cosine similarity between users who follow each
> producer. The resulting cosine similarity values can be used to construct a producer-producer
> similarity graph, where the nodes are producers and edges are weighted by the corresponding cosine
> similarity value. **Noise removal is performed, such that edges with weights below a specified
> threshold are deleted from the graph.**"

> "**After noise removal has been completed, Metropolis-Hastings sampling-based community detection
> is then run on the Producer-Producer similarity graph** to identify a community affiliation for
> each producer. This algorithm takes in a parameter *k* for the number of communities to be
> detected."

> "In production, the Known For dataset covers the top 20M producers and k ~= 145000. In other
> words, we discover around 145k communities based on Twitter's user follow graph."

**P1 relevance. SUPPORTS (strongest "practice at scale" citation).** A deployed, billion-user
system performs an explicit edge-pruning ("noise removal") step **immediately before** running
community detection, and says so in its own production documentation. Bonus narrative: the first
author of SimClusters is **Venu Satuluri**, first author of the 2011 SIGMOD sparsification paper —
the same person carried the sparsify-then-cluster idea from SIGMOD 2011 into Twitter production.

*Honest caveat:* Twitter's pruning is **weight thresholding for noise removal on a derived
similarity graph**, not topology-driven sparsification of a raw interaction graph for speed. It
proves practitioners prune before detecting; it does not prove they do so to buy speed on an
unweighted graph. State this distinction if we cite it.

### 3.2 Peng, Kolda & Pinar (2014) — reduce the graph, then detect

**Citation.** Chengbin Peng, Tamara G. Kolda, Ali Pinar. "Accelerating Community Detection by Using
K-core Subgraphs." arXiv:1403.2226v3, 2014. **Preprint only** — OpenAlex shows no journal/conference DOI, so cite as an arXiv preprint (Kolda and Pinar were at Sandia National Laboratories). (Verified from arXiv PDF; verbatim.)

**Quotes.**

> "Community detection is expensive, and the cost generally depends at least linearly on the number
> of vertices in the graph. We propose working with a reduced graph that has many fewer nodes but
> nonetheless captures key community structure. The K-core of a graph is the largest subgraph within
> which each node has at least K connections. We propose a framework that accelerates community
> detection by applying an expensive algorithm (modularity optimization, the Louvain method,
> spectral clustering, etc.) to the K-core and then using an inexpensive heuristic (such as local
> modularity maximization) to infer community labels for the remaining nodes. **Our experiments
> demonstrate that the proposed framework can reduce the running time by more than 80% while
> preserving the quality of the solutions.**"

**P1 relevance. SUPPORTS (strong).** Sandia-authored, explicitly framed as *reduce-then-detect for
speed*, with a quantified >80% runtime reduction and a "quality preserved" claim. This is a
node-reduction rather than edge-sparsification variant of P1's proposition, and it directly
contradicts our "~1× speed" finding — so it must be engaged with, not merely cited.

### 3.3 Billion-scale / distributed community detection — what they actually do

Searched: distributed Louvain (Ghosh et al., Zeng & Yu, Que et al.), GossipMap, GVE-Louvain,
multi-GPU Louvain, and the "real-time community detection on a laptop" line (Traag,
*PLoS ONE* 12(11):e0188702, 2017).

**Finding (negative, and we should report it as such): the distributed/HPC community-detection
literature scales *up* rather than sampling *down*.** These systems partition and parallelize the
full graph; the pruning they discuss is *algorithmic* pruning inside the Louvain iteration (e.g.,
skipping vertices whose neighborhoods have not changed), not a preprocessing sparsification of the
input. Representative pointers, all **[UNVERIFIED verbatim — abstracts only]**:
- Ghosh et al., "Scalable distributed Louvain algorithm for community detection in large graphs,"
  *The Journal of Supercomputing*, 2022, DOI 10.1007/s11227-021-04224-2.
- Zeng & Yu, "A Scalable Distributed Louvain Algorithm for Large-Scale Graph Community Detection,"
  *IEEE Cluster* 2018. https://cse.unl.edu/~yu/homepage/publications/paper/2018.A%20Scalable%20Distributed%20Louvain%20Algorithm%20for%20Large-scale%20Graph%20Community%20Detection.pdf
- Bhowmick et al. / OSTI, "Distributed Multi-GPU Community Detection on Exascale Computing
  Platforms," https://www.osti.gov/servlets/purl/2372978

**P1 relevance. COMPLICATES — and this is important for honesty.** At the true frontier of scale,
the answer to "how do you run community detection on a 3-billion-edge graph?" has been *distribute
it*, not *sparsify it*. P1's legitimacy therefore rests on the mid-scale / single-machine
practitioner and on the sparsification literature's own claims, **not** on the HPC literature. Do
not write "practitioners at scale routinely sparsify before community detection" — that
over-claims. Write "the sparsification literature repeatedly proposes it, tooling makes it a
one-liner, and at least one production system prunes before detecting."

### 3.4 GNN-era sparsification (DSpar and its citing literature)

**DSpar.** Zirui Liu, Kaixiong Zhou, Zhimeng Jiang, Li Li, Rui Chen, Soo-Hyun Choi, Xia Hu.
"DSpar: An Embarrassingly Simple Strategy for Efficient GNN Training and Inference via Degree-Based
Sparsification." *Transactions on Machine Learning Research (TMLR)*, 07/2023.
https://openreview.net/forum?id=SaVEXFuozg (local copy + detailed notes:
`PAPER_RESTRUCTURE/references/dspar_source_paper.pdf`, `DSPAR_NOTES.md`).
Claim (per the paper, cross-checked against our own DSPAR_NOTES.md): 1.1–5.9× faster than
baselines with "almost no accuracy drop," 30–90% inference-latency reduction; §5.2.2 asserts
"the cluster structure is well-preserved by DSpar."

**Who cites DSpar, and for what.** Checked via OpenAlex full-text search and targeted web search.
Citing works are uniformly **GNN-efficiency** papers, e.g.:
- "Towards Lightweight Graph Neural Network Search with Curriculum Graph Sparsification,"
  *KDD 2024*, DOI 10.1145/3637528.3671706.
- "The Heterophilic Snowflake Hypothesis: Training and Empowering GNNs for Heterophilic Graphs,"
  arXiv:2406.12539 (2024).
- "Two Heads Are Better Than One: Boosting Graph Sparse Training via Semantic and Topological
  Awareness," *ICML 2024*, arXiv:2402.01242.
- "Large-Scale Spectral Graph Neural Networks via Laplacian Sparsification," *KDD 2025*,
  DOI 10.1145/3690624.3709241.

**Finding (negative, report honestly): we found NO paper applying DSpar to community detection,
graph clustering, Louvain or Leiden.** A dedicated search for exactly that returned nothing.
**P1 relevance. COMPLICATES.** DSpar's own claims (near-free speedup, cluster structure preserved)
are a legitimate antecedent to test, but "people use DSpar before community detection" is **not**
supportable. Frame our use of DSpar as *taking a state-of-the-art, theoretically grounded
sparsifier from the GNN literature and asking whether its promise transfers to community
detection* — that framing is defensible and is also a small novelty claim.

---

## CATEGORY 4 — Software and practitioner guidance

### 4.1 Neo4j Graph Data Science — a first-class "Sampling" step in graph creation

**Source.** Neo4j GDS Library Manual v2026.06, *Graph management → Creating graphs → Sampling*.
https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/sampling/
(fetched 2026-07-25 with a browser user-agent; publisher 403s plain fetch. Quotes verbatim.)

> "Graph sampling algorithms can be used to reduce the size of large and complex graphs while
> preserving structural properties. This can help to reduce bias, and ensure privacy, and making
> graph analysis more scalable. Sampling algorithms are widely used in machine learning, social
> network analysis, and many other applications."

From the *Random walk with restarts sampling* page:

> "Sometimes it may be useful to have a smaller but structurally representative sample of a given
> graph. For instance, such a sample could be used to train an inductive embedding algorithm (such
> as a graph neural network, like GraphSAGE). The training would then be faster than when training
> on the entire graph, and then the trained model could still be used to predict embeddings on the
> entire graph."

> "It was shown by Leskovec et al. in the paper 'Sampling from Large Graphs' that RWR is a very good
> sampling algorithm for preserving structural features of the original graph that was sampled
> from."

From the *Common Neighbour Aware Random Walk sampling* page:

> "The Common Neighbour Aware Random Walk (CNARW) is a graph sampling technique that involves
> optimizing the selection of the next-hop node. It takes into account the number of common
> neighbours between the current node and the next-hop candidates. According to the paper, a major
> reason why simple random walks tend to converge slowly is due to the high clustering feature that
> is typical for some kinds of graphs e.g. for online social networks (OSNs)."

**P1 relevance. SUPPORTS (moderate) + COMPLICATES (must be stated).** Supports: the single most
widely deployed commercial graph-analytics platform ships graph **sampling as a standard
pre-analysis step**, in the same product whose "Community detection" chapter contains Louvain,
Leiden, Label Propagation, Modularity Optimization, etc. — so "sample it, then run Louvain" is a
two-line workflow a GDS user can and does write.
Complicates, honestly: the GDS docs **motivate sampling for GNN/embedding training and for
scalability in general, not for community detection specifically**, and nothing in the sampling
docs claims community structure is preserved. Do not attribute to Neo4j a recommendation they do
not make. The defensible sentence is: *"mainstream graph platforms expose graph sampling as a
first-class preprocessing step alongside their community-detection algorithms."*
**[Also checked and found NOT to recommend sparsification before community detection: igraph's
community-detection chapter, cuGraph's community API docs, NetworkX. No such guidance located.]**

### 4.2 NetworKit `sparsification` module

**Source.** https://networkit.github.io/dev-docs/python_api/sparsification.html and tutorial
https://networkit.github.io/dev-docs/notebooks/Sparsification.html
Ships `LocalSimilaritySparsifier` — documented as "an implementation of the Local Similarity
sparsification approach introduced by Satuluri et al." — plus `LocalDegreeSparsifier`,
`SCANSparsifier`, Simmelian backbones, Forest Fire, Random Edge, etc.
These are the artifacts of Hamann et al. 2016 (§1.2), whose own words: *"we have published
efficient parallelized implementations and a framework for such methods as part of the NetworKit
open-source tool suite."*
**P1 relevance. SUPPORTS (tooling availability).** Satuluri's method is `pip install networkit`
away, in the same library that ships PLM/PLP community detection.
**Honest caveat:** the tutorial text does **not** tell users to sparsify before community
detection.

### 4.3 `backbone` R package (CRAN) — see §1.5

Ships `sparsify.with.lspar()` = "the L-spar backbone described by Satuluri et al. (2011)".
Package paper: Domagalski, Neal & Sagan, *PLoS ONE* 17(5):e0269137, 2022 (61 citations).
**SUPPORTS (tooling availability, R/social-science side).**

### 4.4 Applied social-network-analysis guidance: filter before you look for communities

**Citation.** Carlos Henrique Gomes Ferreira, Fabricio Murai, Ana P. C. Silva, Martino Trevisan,
Luca Vassio, Idilio Drago, Marco Mellia, Jussara M. Almeida. "On network backbone extraction for
modeling online collective behavior." *PLOS ONE* 17(9):e0274218, 2022.
DOI 10.1371/journal.pone.0274218 (open access; verified from the PLOS PDF; verbatim).

**Quotes.**

> "only a fraction of edges contribute to the actual investigation. Even worse, the often large
> number of non-relevant edges may obfuscate the salient interactions, **blurring the underlying
> structures and user communities** that capture the collective behavior patterns driving the target
> phenomenon. To solve this issue, researchers have proposed several network backbone extraction
> techniques to obtain a reduced and representative version of the network that better explains the
> phenomenon of interest."

> "We characterize ten state-of-the-art techniques in terms of their assumptions, requirements, and
> other aspects that one must consider to apply them in practice. ... We show that **each method can
> produce very different backbones**, underlying that the choice of an adequate method is of utmost
> importance to reveal valuable knowledge about the particular phenomenon under investigation."

> "community detection algorithms [14, 15] unveil groups of tightly connected users in a network,
> letting collective behavior emerge by exploring the topological properties of the network itself"

**P1 relevance. SUPPORTS (strong, and from a different community than CS-systems).** In applied
computational social science, edge filtering *before* community detection is not an optimization —
it is treated as a methodological necessity, on the theory that unfiltered edges "blur" the
communities. This is precisely the "sparsification clarifies community structure" belief our δ /
configuration-null result attacks, and it is held by a whole applied field. Their own finding that
"each method can produce very different backbones" is a mild antecedent to our
partition-instability point.

---

## CATEGORY 5 — Coarsening and condensation

### 5.1 IJCAI 2024 survey — the modern graph-reduction frame

**Citation.** Mohammad Hashemi, Shengbo Gong, Juntong Ni, Wenqi Fan, B. Aditya Prakash, Wei Jin.
"A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation."
*IJCAI-24, Survey Track*. https://www.ijcai.org/proceedings/2024/0891.pdf (verified; verbatim).

**Quotes.**

> "the increasing complexity and size of graph datasets present significant challenges for analysis
> and computation. In response, graph reduction, or graph summarization, has gained prominence for
> simplifying large graphs while preserving essential properties."

> "Traditional sparsification target at preserving essential graph properties including pairwise
> distances, cuts, and spectrum."

> "With the emergence of GNNs, a new goal of graph sparsification has arisen: maintaining the
> prediction performance of GNNs trained on the sparsified graph."

**P1 relevance. COMPLICATES — a load-bearing negative finding.** We text-searched the full survey:
it contains **four** occurrences of "graph classification," **one** of "node classification," and
**one** of "clustering"; **zero** occurrences of "community" or "community detection." The modern
graph-reduction field has reoriented entirely around GNN prediction tasks and has *dropped*
community detection as an evaluation target. This is genuinely good for us: it means the
community-detection question was asked in 2011–2016, answered optimistically, and then abandoned
rather than settled. That is a clean gap statement for the Introduction, and it is defensible
because we can point at the survey's own task list.

### 5.2 Coarsening-adjacent items (secondary)

- Peng, Kolda & Pinar (2014) k-core reduction — see §3.2 (the strongest coarsening-style
  accelerate-community-detection claim).
- CoVeC: "CoVeC: Coarse-grained vertex clustering for efficient community detection in sparse
  complex networks," *Information Sciences*, 2020 (16 citations; cites Satuluri 2011).
  **[UNVERIFIED verbatim — title/venue only.]** **SUPPORTS (weak).**
- "A community detection algorithm based on graph compression for large-scale social networks,"
  *Information Sciences*, 2020 (108 citations; cites Satuluri 2011).
  **[UNVERIFIED verbatim.]** **SUPPORTS (weak)** — the highest-cited work citing Satuluri, and it
  is a *compress-then-detect* method, i.e. the practice propagating.
- Sotiropoulos & Tsourakakis, "A Unifying Framework for Spectrum-Preserving Graph Sparsification
  and Coarsening," *NeurIPS 2019* (80 citations; cites Satuluri 2011).
  **[UNVERIFIED verbatim.]** Context only.
- Chen, Kolda-adjacent hypergraph line and multilevel-partitioning coarsening (e.g. "Improving
  Coarsening Schemes for Hypergraph Partitioning by Exploiting Community Structure," SEA 2017) go
  the *other* direction — they use community detection to improve coarsening. Not P1 evidence.

---

## (a) VERDICT — is "should you sparsify before community detection?" a real question?

**Yes, with one qualification that must be stated honestly.**

The question is unambiguously real *as a published claim and as available tooling*. It is
**less** established *as documented routine practice at the billion-edge frontier*, where the
dominant answer has been distribution and parallelism rather than sparsification (§3.3). The
defensible framing is therefore:

> A fifteen-year line of work — from SIGMOD 2011 through ICDM 2020, KDD 2021, VLDB 2024 and a 2026
> arXiv preprint — has proposed sparsifying graphs before community detection, claiming order-of-
> magnitude speedups and, in several cases, *improved* cluster quality; the methods ship in
> NetworKit, CRAN and (as generic sampling) Neo4j GDS; and at least one production system at
> Twitter/X prunes edges immediately before running community detection at 20M-node scale. What has
> **not** happened is a controlled test of whether the pipeline actually pays, end to end, against a
> cost-matched baseline.

That last sentence is our gap, and it survives everything found here.

**The five strongest citations, ranked.**

1. **Satuluri, Parthasarathy & Ruan, SIGMOD 2011** (§1.1) — the origin, 206 citations, and the only
   source that states the *quality-gain* claim outright: "for at least two of the four clustering
   algorithms, our sparsification consistently enables higher clustering accuracies," alongside
   "speedups (often in the range 10-50)". Non-negotiable citation; it is the proposition itself.
2. **Wu & Chen, GSGAN, IEEE ICDM 2020** (§1.5, last bullet) — the sharpest modern restatement:
   at 5% of edges, "comparable or even better results than on the original graph," with execution
   time "reduced ... by nearly an order of magnitude." Shows the claim is not a 2011 relic.
3. **Twitter/X SimClusters** (§3.1) — production evidence. The open-sourced code documentation says
   edges below a weight threshold are deleted and *then* community detection is run, at 20M
   producers / ~145k communities. Also the same first author as (1). This is the "practice"
   citation; use it with the weight-thresholding caveat stated.
4. **Hamann et al., SNAM 2016** (§1.2) — the systematic study that both attests Satuluri's claim
   ("intention of speedup and quality improvement of community detection ... reducing the edge set
   to 10-20%") and supplies the NetworKit implementations everyone uses. Also our most important
   prior-art acknowledgement.
5. **Chen, Ye, Vedula, Bronstein, Dreslinski, Mudge & Talati, PVLDB 17(3), 2023/24** (§1.4) — the
   most recent peer-reviewed statement of the premise from a top systems venue: "Graph
   sparsification can be applied to greatly reduce the run time of graph algorithms by substituting
   the full graph with a much smaller sparsified graph, without significantly degrading the output
   quality." Doubles as prior art (see §b) — cite it for both jobs.

*Runners-up worth a citation each:* **Peng, Kolda & Pinar, arXiv:1403.2226 (2014)** (§3.2) —
"Community detection is expensive ... the proposed framework can reduce the running time by more
than 80% while preserving the quality of the solutions"; quantitative and squarely contradicted by
our ~1× speed result, which makes it a productive foil, **but it is an unrefereed preprint, so do
not lean on it alone**. Also: Sotiropoulos & Tsourakakis KDD 2021 (§1.3, sparsify-for-communities
treated as established and in need of *explanation*); Gomes Ferreira et al. PLOS ONE 2022 (§4.4,
the applied-CSS belief that unfiltered edges "blur" communities); Blagus et al. Physica A 2015
(§2.3, the artifact framing).

---

## (b) PRIOR-ART RISK — who has already answered part of our question?

Ranked by how much of our claim they take. **None of them takes the whole thing**, but three take
real pieces, and two of those we currently do not cite.

**HIGH — must cite, must position against.**

1. **Hamann, Lindner, Meyerhenke, Staudt & Wagner, SNAM 2016 (arXiv:1601.00286).** Already
   published: (i) the explicit warning that **NMI is inflated when sparsification fragments the
   graph into more, smaller communities**, and the recommendation to use ARI instead; (ii) the
   finding that improving a *fixed* partition's conductance under sparsification "does not just make
   this structure clearer but does also lead the algorithm into finding different community
   structures"; (iii) "the preserved community structure is not necessarily the same as the one the
   Louvain algorithm finds"; (iv) "Random edge deletion performs surprisingly well."
   **What they did NOT do:** no end-to-end runtime measurement; no cost-matched baseline; no
   degree-preserving configuration null; no ground-truth-recovery experiment with a
   resolution-matched control; no verdict on whether to sparsify. Our Phase-1 results B, C and D
   are all outside their scope. **Action: cite them early and explicitly, credit the NMI/granularity
   warning to them, and frame our resolution-matched control as operationalizing it.** Failing to
   cite this paper would look like we missed the closest relative.

2. **Blagus, Šubelj, Weiss & Bajec, Physica A 2015 (arXiv:1504.03097).** Already published:
   subsampled networks show **stronger** community-like structure than the originals, "consistently
   across various sampling techniques," and therefore "rich community structure ... is to some
   extent merely an artifact of sampling."
   **What they did NOT do:** they treat sampling as a data-collection fact, not as a deliberate
   accelerator; no modularity-delta analysis; no configuration-model null; no runtime; no
   recommendation. **Action: cite as the antecedent of our artifact framing; do not claim to be
   first to observe that reduction inflates apparent community structure.**

**MEDIUM — cite, but our scope is clearly larger.**

3. **Chen, Ye, Vedula, Bronstein, Dreslinski, Mudge & Talati, PVLDB 17(3), 2023/24
   (arXiv:2311.12314).** Already published: across 12 sparsifiers and 14 graphs, "For all
   sparsifiers, F1 similarity decreases as the prune rate increases," and "As the prune rate
   increases, the graph becomes increasingly disconnected, and the number of communities
   consistently rises." That is the fragmentation mechanism behind our granularity artifact, at
   benchmark scale. **What they did NOT do:** no wall-clock end-to-end community-detection pipeline
   timing; no resolution-matched control; no null model; and their conclusion is "match the
   sparsifier to the task," not "sparsification does not pay." **Action: cite as the strongest
   existing evidence of quality degradation, then say what it leaves open.**

**LOW — narrow overlaps worth a sentence.**

4. **Gottesbüren, Maas, Rosch, Sanders & Seemaier, "Linear-Time Multilevel Graph Partitioning via
   Edge Sparsification," ESA 2025 (arXiv:2504.17615).** Verbatim: *"Since sparsification itself
   introduces computational overhead which can counteract potential speedups, particularly when the
   size reduction is modest"* and *"Since sparsification itself introduces computational overhead, we
   only apply it if the potential edge reduction is significant."* This anticipates, in the
   *partitioning* setting and as an engineering aside, the accounting behind our ~1× speed result.
   **Action: cite in the runtime discussion as independent corroboration of the overhead argument.
   It is not a threat — they are optimizing partitioning, not evaluating community detection.**

5. **Pari, Bhandari & Raha, arXiv:2606.26766 (June 2026).** Their own concession: *"It appears that
   for the synthetic data, sparsification does not play much of a role because the networks are
   inherently generated with community structure; so the edges are not noisy."* A one-line
   corroboration of our LFR finding, published by authors who are otherwise advocating the pipeline.
   **Action: cite as a friendly-witness corroboration.**

**SEARCHED FOR AND NOT FOUND (i.e. the space appears open).** Multiple query formulations found
**no** paper that: (i) reports the **end-to-end wall-clock** cost of sparsify-then-detect against
detect-on-the-full-graph and concludes it does not pay for community detection; (ii) uses a
**degree-preserving configuration-model null** to show that post-sparsification modularity/δ gains
are reproduced by degree heterogeneity alone; (iii) uses a **resolution-matched** unsparsified
control to show that ground-truth F1 gains are a cluster-granularity artifact; or (iv) applies
**DSpar** (or any effective-resistance GNN sparsifier) to community detection at all.
*This is a negative search result, not a proof of absence — Google Scholar's true "cited by" graph
was not traversable from this environment, and ACM DL / Springer / ScienceDirect full texts were
not reachable. Before submission, re-run these checks with library access.*

---

## (c) SUGGESTED BACKGROUND PASSAGE (3–4 sentences, drop-in)

> Sparsifying a graph before clustering it has been an explicit recommendation since Satuluri et
> al. reported that retaining only 10–20% of edges yields "excellent speedups (often in the range
> 10-50), with little or no deterioration in the quality of the resulting clusters" and, for some
> algorithms, "consistently enables higher clustering accuracies" [Satuluri et al., SIGMOD 2011].
> The claim has been renewed rather than retired: sparsifiers built specifically for community
> detection report "comparable or even better results than on the original graph" using 5% of the
> edges [Wu and Chen, ICDM 2020], k-core reduction reports "more than 80%" runtime savings "while
> preserving the quality of the solutions" [Peng et al., 2014], and the practice is packaged in
> NetworKit, CRAN's `backbone`, and — as generic graph sampling — Neo4j GDS, while Twitter's
> production SimClusters pipeline deletes low-weight edges immediately before running community
> detection on 20M nodes [Satuluri et al., KDD 2020]. Yet the systematic evaluations that exist
> stop short of the operational question: Hamann et al. show that sparsification "lead[s] the
> algorithm into finding different community structures" and warn that NMI is inflated when
> fragmentation multiplies small communities [SNAM 2016], and Chen et al. find that "for all
> sparsifiers, F1 similarity decreases as the prune rate increases" [PVLDB 2024] — but neither
> measures whether the pipeline is faster end to end, nor compares it against an unsparsified
> baseline given the same compute budget. We ask exactly that question.

*(Note: the Peng et al. k-core result is an arXiv preprint; if a referee-proof citation is needed
for the runtime claim, substitute Chen et al. PVLDB 2024 or Wu & Chen ICDM 2020. Adjust bracketed
keys to the repo's `refs.bib`. Every quoted fragment above appears verbatim in
this file with its verification note; the Satuluri fragments come from the OpenAlex-reconstructed
abstract and should be spot-checked against the PDF before camera-ready.)*

---

## Appendix — sources that could NOT be verified from this environment

| Source | Why | Risk if cited |
|---|---|---|
| ACM DL full texts (Satuluri 2011; SimClusters KDD'20; Sotiropoulos KDD'21 official version) | HTTP 403 | Abstracts obtained elsewhere (OpenAlex / author PDF / Twitter repo); page-level quotes unverified |
| Springer (Soft Computing 2025 effective-resistance CD paper; SNAM journal version of Hamann) | 303 redirect to auth | arXiv version of Hamann used instead and is safe; Soft Computing paper should not be quoted |
| ScienceDirect (CoVeC; graph-compression CD, *Information Sciences* 2020) | paywall | Titles/venues only — do not quote |
| Google Scholar "cited by" graph | not accessible | Citation counts taken from Semantic Scholar / OpenAlex APIs instead |
| export.arxiv.org API | network-blocked in this sandbox | Worked around by downloading arXiv PDFs directly |
