# P2 RESEARCH — new material for Section 2, Background and Related Work

**Purpose.** §2 has to introduce every part the rest of the paper uses, define "removable
redundancy" (which §1 uses as a technical term and nothing defines), and hold the contemporary
instances of the missing control per decision D1. This file collects material that
`P1_EVIDENCE.md` does not already cover. Read that file first; it is not superseded.

**Compiled 2026-07-27.** Same discipline as P1_EVIDENCE.md and the same reason: nothing here is
written from recall. Every bibliographic field below was read off a fetched record, and the record
is named. Where a claim describes what a paper *argues* rather than that it exists, the
verification level says whether the full text was read.

## Verification legend

- **[META-VERIFIED]** — bibliographic fields read off a publisher record, DOI resolver, arXiv, or
  the Semantic Scholar API. Safe to cite. Says nothing about whether we have read the paper.
- **[FULL-TEXT]** — the PDF or the full HTML was read. Safe to characterize what it claims.
- **[SECOND-HAND]** — content described via a fetched page summary, not read directly. **Do not
  put a characterization of this paper into the tex until it is upgraded to FULL-TEXT.**
- **[UNVERIFIED]** — could not confirm. Not citable.

---

## FINDING 1 — the paper's statement about MLR-MCL and Graclus is wrong as written

This is the most consequential item in this file and it contradicts text already in §4.1:

> "For Graclus and MLR-MCL we could find no maintained implementation that we were able to run,
> so MLR-MCL, the algorithm that claim is largest about, remains untested by us."

Both have source code published by their own authors, findable in one search.

### 1.1 MLR-MCL — source available, BSD licence  [META-VERIFIED]

**Paper.** Venu Satuluri, Srinivasan Parthasarathy. "Scalable graph clustering using stochastic
flows: applications to community discovery." *KDD '09*, pp. 737–746, 2009.
DOI 10.1145/1557019.1557101.
*Source of fields:* Semantic Scholar Graph API keyed by DOI, retrieved 2026-07-27:
`https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/1557019.1557101` — returned
`"pages": "737-746"`, `"DBLP": "conf/kdd/SatuluriP09"`.

**Code.** `https://sites.google.com/site/stochasticflowclustering/` (Satuluri's own project page),
fetched 2026-07-27. Offers `mlrmcl1.2.tar.gz` covering the first two papers, plus
`ECCB2012_SR-MCL.tar.gz` and `nrmcl.zip` for later variants. The page states the code is
**"released open source under a BSD license."** Most recent associated publication on the page is
2014 (HiPC); a 2014 CIKM directed-network variant is listed as "awaiting public release."

### 1.2 Graclus — source available, GPL  [META-VERIFIED]

**Paper.** Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis. "Weighted graph cuts without
eigenvectors: A multilevel approach." *IEEE Transactions on Pattern Analysis and Machine
Intelligence* 29(11):1944–1957, November 2007.
*Source of fields:* the software page's own recommended citation, `https://cs.utexas.edu/users/dml/Software/graclus.html`, fetched 2026-07-27.

**Code.** Same page. Versions 1.0, 1.1 and 1.2 downloadable; v1.2 adds 64-bit Linux and a Windows
MATLAB interface. Licence: **"You are welcome to use the code under the terms of the GNU Public
License (GPL), however please acknowledge its use with a citation."** The page states the code was
compiled with **gcc 3.0.3** on Solaris and Linux, which is a toolchain from the early 2000s and is
the honest basis for a "does not build on a current compiler" claim if that turns out to be true.

Graclus also appears as a pooling operator in maintained deep-learning libraries
(`torch_geometric.nn.pool.graclus`, `rusty1s/pytorch_cluster`), which is the coarsening step rather
than the full clustering tool. **[SECOND-HAND** — seen in search results, not fetched.**]**

### What this forces

"We could find no maintained implementation" is not defensible; both are published by their
authors and reachable in one search. Two honest options:

1. **Run them.** MLR-MCL is the algorithm the founding accuracy claim is largest about. If it
   builds, the most consequential gap in the paper closes.
2. **Say precisely what happened.** "The published implementations date from 2008 and 2014 and do
   not build under $X$; we did not port them" is a defensible sentence. "We could find no
   implementation" is not.

Either way §4.1's sentence and §5's planned discussion have to change. Flagged, not edited.

---

## FINDING 2 — "removable redundancy" has an established name: the metric backbone

§1 uses "removable redundancy" as a technical term, the claim's second condition rests on it, and
nothing in the paper defines it. There is a named, formal concept for very nearly this quantity,
and one of the two papers below is a NeurIPS 2024 theory result about exactly our question.

### 2.1 Dreveton, Chucri, Grossglauser, Thiran (NeurIPS 2024)  [META-VERIFIED + abstract FULL-TEXT]

**Citation.** Maximilien Dreveton, Charbel Chucri, Matthias Grossglauser, Patrick Thiran. "Why the
Metric Backbone Preserves Community Structure." *Advances in Neural Information Processing Systems
38 (NeurIPS 2024)*. arXiv:2406.03852.
*Source of fields:* `https://arxiv.org/abs/2406.03852`, fetched 2026-07-27.

**Verbatim abstract** (read directly, quotable):

> "The metric backbone of a weighted graph is the union of all-pairs shortest paths. It is obtained
> by removing all edges (u,v) that are not the shortest path between u and v. In networks with
> well-separated communities, the metric backbone tends to preserve many inter-community edges,
> because these edges serve as bridges connecting two communities, but tends to delete many
> intra-community edges because the communities are dense. This suggests that the metric backbone
> would dilute or destroy the community structure of the network. However, this is not borne out by
> prior empirical work, which instead showed that the metric backbone of real networks preserves the
> community structure of the original network well. In this work, we analyze the metric backbone of
> a broad class of weighted random graphs with communities, and we formally prove the robustness of
> the community structure with respect to the deletion of all the edges that are not in the metric
> backbone. An empirical comparison of several graph sparsification techniques confirms our
> theoretical finding and shows that the metric backbone is an efficient sparsifier in the presence
> of communities."

**Their experimental setup.** **[SECOND-HAND** — from a fetch of the ar5iv HTML summarised by a
small model, NOT read directly.**]** Reported as: real networks with known ground truth; Bayesian
MCMC via graph-tool, Leiden, and spectral clustering; clustering run on each sparsification and
compared against ground truth; the unsparsified graph is included as a baseline; **no runtime or
speedup is reported.**

**Why this matters to us, and the care it needs.**
- *Prior-art risk, real but bounded.* They prove robustness of community structure under one
  specific sparsifier on a random-graph model. We measure whether a detector run on a sparsified
  graph beats the same detector on the original, across seven sparsifiers and five detectors, and
  we price the accounting. Different questions. But a reviewer who knows this paper will ask, and
  §2 must cite it and say what is different.
- *It is evidence for our second condition, not against it.* A formal result that community
  structure survives deleting every non-shortest-path edge is a statement that graphs carry
  removable redundancy. That is the concept §1 needs and does not define.
- *Do NOT claim they omit our controls* until someone reads the paper properly. The recovery metric
  is agreement with ground-truth labels, which is not scored "on" a graph at all, so Artifact I
  does not straightforwardly apply to it. Whether they control granularity under Leiden is a real
  question and is unanswered. **Get the PDF before writing a sentence about their protocol.**

### 2.2 Correia, Barrat, Rocha (PLOS Computational Biology 2023)  [META-VERIFIED]

**Citation.** Rion Brattig Correia, Alain Barrat, Luis M. Rocha. "Contact networks have small
metric backbones that maintain community structure and are primary transmission subgraphs."
*PLOS Computational Biology* 19(2):e1010854, 2023. DOI 10.1371/journal.pcbi.1010854.
*Source of fields:* the PLOS article page, fetched 2026-07-27.

This is the empirical predecessor Dreveton et al. refer to. Reported backbone sizes are roughly
6–20% of edges on most of nine contact networks, one at 49%, and 10–30% is quoted for social
networks generally. **[SECOND-HAND** for those percentages — from search-result summaries, not the
PDF. Read before quoting a number.**]**

### The definitional opening this gives §2

"Removable redundancy" can be introduced against a concept that already exists: edges absent from
the metric backbone are redundant in a precise, checkable sense, and real networks have a great
many of them. Our condition is broader than the metric one, and §2 should say in what way rather
than inventing a term with no anchor.

---

## FINDING 3 — the degree-preserving rewiring null has a canonical citation we do not cite

§3.5 and §4.7 rest on a degree-preserving rewiring built by double edge swaps. We cite
`molloy1995critical` for the configuration model, which is the right citation for the *model* but
not for the *rewiring procedure*.

**Citation.** Sergei Maslov, Kim Sneppen. "Specificity and stability in topology of protein
networks." *Science* 296(5569):910–913, 2002. DOI 10.1126/science.1065103.  [META-VERIFIED]
*Source of fields:* science.org DOI record and consistent secondary records, fetched 2026-07-27.

This is the standard citation for the degree-preserving edge-swap randomization we actually run,
and it is also the paper that established using such a rewiring as a *null model* for detecting
structure beyond the degree sequence, which is exactly our use.

---

## FINDING 4 — the objective/recovery dissociation has adjacent prior work

§1's fourth contribution bullet reports the dissociation "as an observation rather than a result"
and calls it the most consequential open question the boundary exposes. There is adjacent
literature and the novelty claim should be checked against it before §5 is written.

**[SECOND-HAND, all of the following** — from search-result summaries only. None fetched, none
read. Listed as leads, **not citable in this state**.**]**

- A "ground truth contest between modularity maximization and modularity density maximization",
  reporting that ground-truth-based quality tracks modularity density but not standard modularity.
- A line of work reporting that *exact* modularity maximization outperforms many methods at
  recovering ground truth on LFR and ABCD, which cuts the other way.
- The known framing that modularity's suitability depends on whether it is used as an objective or
  as a quality function, and on how maximization is operationalized.

The honest reading is that "modularity optima need not be the best recoveries" is not new in
general. What may be new is our specific form: the two criteria moving in *opposite directions
within the same configuration*, on the same partition of the same graph. §2 or §5 has to make that
distinction explicitly or the contribution bullet will be read as overclaiming.

---

## FINDING 5 — recent sparsification-and-community-detection work for the reference list

Added at Mohammad's request: reviewers weigh the recency of a reference list, and §2 is where that
is satisfied honestly. Everything here is on the paper's own subject, sparsification and community
detection. Nothing was added merely to raise the year count.

| Key | Work | Year | Verification |
|---|---|---|---|
| `sotiropoulos2021triangle` | Sotiropoulos & Tsourakakis, Triangle-aware spectral sparsifiers and community detection, KDD, pp. 1501--1509, doi 10.1145/3447548.3467260 | 2021 | **[FULL-TEXT]** author PDF read, quotes in P1_EVIDENCE §1.3 |
| `liu2023dspar` *(already present)* | DSpar, TMLR | 2023 | [FULL-TEXT] local PDF + DSPAR_NOTES.md |
| `correia2023contact` | Correia, Barrat & Rocha, Contact networks have small metric backbones..., PLOS Comput Biol 19(2):e1010854 | 2023 | **[META-VERIFIED]** PLOS article page |
| `chen2024demystifying` *(already present)* | Chen et al., PVLDB 17(3) | 2024 | [FULL-TEXT] local PDF + NOTES_chen_and_fastcd.md |
| `dreveton2024metric` | Dreveton, Chucri, Grossglauser & Thiran, Why the metric backbone preserves community structure, NeurIPS, arXiv:2406.03852 | 2024 | **[META-VERIFIED]** + abstract read verbatim |
| `hashemi2024survey` | Hashemi, Gong, Ni, Fan, Prakash & Jin, A comprehensive survey on graph reduction, IJCAI Survey Track | 2024 | **[FULL-TEXT]** title page read; quotes in P1_EVIDENCE §5.1 |
| `socievole2024community` | Socievole & Pizzuti, Community detection ... exploiting spectral graph sparsification for efficient disaster response, ASONAM, doi 10.1007/978-3-031-85386-9_14 | 2024 | **[META-VERIFIED]** Semantic Scholar Graph API by DOI; DBLP `conf/asunam/SocievoleP24` |
| `gottesburen2025linear` *(already present)* | Gottesbueren et al., ESA | 2025 | [META-VERIFIED] |
| `setiadi2025community` | Setiadi, Yaakub & Abu Bakar, Community preserving sparsification based on K-core, IJAIN 11(4):550--566, doi 10.26555/ijain.v11i4.2209 | 2025 | **[FULL-TEXT]** local PDF; reproduced and falsified, see NOTES_kcore_ijain2026.md |
| `socievole2026effective` | Socievole & Pizzuti, Effective resistance and kernel-based graph sparsification for community detection, Soft Computing 30:2109--2133, doi 10.1007/s00500-025-10734-5 | 2026 | **[META-VERIFIED]** Semantic Scholar Graph API by DOI; DBLP `journals/soco/SocievoleP26` |
| `pari2026effective` | Pari, Bhandari & Raha, Effective resistance-based graph sparsification and community detection, arXiv:2606.26766 | 2026 | **[FULL-TEXT]** local PDF; audited in NOTES_effres_2026.md |

That is 2021, 2023 x2, 2024 x4, 2025 x2, 2026 x2 on the paper's exact subject, six of them read in
full. Two of the recent ones, `setiadi2025community` and `pari2026effective`, are the contemporary
instances of the missing control that decision D1 assigns to related work with the arithmetic, and
both have already been reproduced and audited in this repo.

**One caution.** Socievole and Pizzuti appear twice, and the 2026 Soft Computing paper is on
exactly our question, using effective resistance for community detection. Neither has been read.
Before §2 characterizes either, someone has to check whether they score quality on the sparsified
graph or the original: if they score on the sparsified graph, they belong with the D1 instances and
the paper should say so with the arithmetic; if they do not, the related-work paragraph has to
credit them. **Do not write either sentence from the abstract.**

---

## Still to do

1. Get the Dreveton et al. PDF and settle their evaluation protocol.
2. Decide MLR-MCL and Graclus: build attempt, or a precise sentence.
3. Fetch and read the modularity-versus-ground-truth items in Finding 4 before §5 is drafted.
4. Read Socievole & Pizzuti 2026 (Soft Computing) and settle whether it carries the transfer
   control. It is the closest recent work to our question.
5. Not yet searched: the granularity/resolution-comparison literature that motivates §3.2 beyond
   Hamann et al.; the fixed-$k$ balanced-partitioning literature that §4.1's Metis balance
   constraint sits in; anything 2025–26 that post-dates P1_EVIDENCE.md's compilation.
