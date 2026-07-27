# Exp AD — the fixed-$k$ arm on all seven real networks, both $k$ conventions

Pre-registered 2026-07-26 (evening) BEFORE any Exp AD code was written or run.

## Why this exists

Exp AB ran a real-Metis arm on **two** networks only, email-Eu-core ($d_{avg}$ 32.6) and
wiki-Vote (28.5), and got 8 positive cells out of 28. Its DESIGN says why the other five were
skipped: *"Leiden primary (3 seeds). If time permits, real Metis (pymetis, k fixed) on the two
densest networks."* It was a time budget, not a blocker. `pymetis` is installed and has been used
in exp_AA, exp_AB and exp_AC; a Metis baseline on wiki-Vote with ten seeds takes 0.068 s.

The draft of Section 4 currently states that Metis "has no implementation in the library used
here". **That sentence is false**, and it discards the strongest real-network evidence we have.
This experiment removes the exclusion rather than rewording it.

## What is at stake

The fixed-$k$ half of the boundary presently rests on LFR (exp_AA), SBM and dcSBM (exp_AC), and
two real networks that both sit **above** the transition. Nothing tests the fixed-$k$ arm below
it on real data. The two conditions in the paper's claim are the detector's degrees of freedom
and the graph's removable redundancy; on real networks we have varied the first and not the
second.

Five of our seven networks sit at average degree 5.5 to 10.7, well below the two already run.
Running the same arm there tests the density condition on real graphs from the other side.

## Design

Everything is held identical to Exp AB so the two are comparable cell for cell. `run.py` is
`exp_AB_sparsifier_coverage/run.py` **copied verbatim** with four disclosed changes, listed at
the end of this file.

- **Networks:** all seven (email-Eu-core, wiki-Vote, ca-HepTh, ca-CondMat, email-Enron,
  com-DBLP, com-Amazon). The two already covered by Exp AB are re-run so that all seven receive
  identical treatment; their agreement with Exp AB is a consistency check.
- **Detector:** real Metis through `pymetis`, $k$ fixed by construction, so the sparsified and
  unsparsified arms return the same number of communities and no granularity effect is possible.
- **Two $k$ conventions, both reported** (Mohammad, 2026-07-26):
  - `nc_base`: $k$ = the community count Leiden returns on the original graph. This is Exp AB's
    convention and keeps the two experiments comparable.
  - `gt`: $k$ = the number of ground-truth communities, on the three labelled networks only
    (email-Eu-core hard labels; com-DBLP and com-Amazon SNAP overlapping communities of at
    least `MIN_GT_SIZE` members). This is the convention a practitioner with metadata would use,
    and it differs from `nc_base` by more than an order of magnitude on the large networks.
- **Sparsifiers:** the same seven arms as Exp AB (K-Neighbor, Local Degree, Local Similarity,
  L-Spar, DSpar, backbone with random fill, backbone with Jaccard fill), at matched realized
  retention, at the same operating points (0.5, 0.2, plus the largest floor for that network).
- **Seeds:** ten Metis option seeds for the baseline and for every arm. Exp AB used five for the
  baseline and three per arm; ten is the exp_AA `metis_noise` protocol and makes the worst-case
  comparison stronger, not weaker.
- **Scoring:** honest transfer throughout. Modularity of every partition is computed on the
  ORIGINAL graph. The headline statistic is the worst case, the minimum sparsified run minus the
  maximum baseline run over the ten seeds.
- **Recovery:** AMI, ARI and average best-match $F_1$ with size-matched chance floors on the
  three labelled networks, under both $k$ conventions.
- **Cost:** end to end, including the sparsifier's own time.

## Pre-registered predictions

- **P1:** the two dense networks reproduce Exp AB. Worst-case positive cells appear on wiki-Vote
  and email-Eu-core under the `nc_base` convention.
- **P2:** the five sparse networks ($d_{avg}$ 5.5 to 10.7) show no worst-case positive cell under
  either convention. This is the density condition tested on real graphs.
- **P3:** where a gain occurs, the similarity-based arms (L-Spar, Local Similarity, K-Neighbor)
  produce it and the backbone arms do not, matching Exp AB's pattern on the two dense networks.
- **P4:** the two $k$ conventions agree in sign on the three labelled networks. Magnitudes may
  differ; the direction should not.

## Kill criteria, bidirectional

- **If P2 fails** and Metis gains on sparse real graphs, then on real networks the fixed-$k$ gain
  does not require density, and the paper's second condition must be restated or dropped for real
  graphs. This is the outcome that would most damage the current claim, and it must be reported
  first if it occurs.
- **If P1 fails** and the two dense networks show nothing at ten seeds, then Exp AB's 8 positive
  cells were an artifact of its seed count or its arm-level seed reuse, and the real-network half
  of the boundary collapses to nothing.
- **If P4 fails** and the conventions disagree in sign, then $k$ selection is a confound, every
  fixed-$k$ statement in the paper needs its convention attached, and the boundary claim becomes
  conditional on how a practitioner picks $k$.

## Disclosed changes to the copied Exp AB code

1. `METIS_SEEDS` extended from five to ten, and the per-arm loop uses all ten rather than the
   first three.
2. `run_network` takes a `k_mode` argument; the Metis block selects $k$ from `nc_base` or from
   the ground-truth community count, and every row records which convention produced it.
3. A `--metis-only` flag skips the Leiden arm sweep, which Exp AB already reported, while keeping
   the Leiden baseline because it supplies `nc_base`.
4. Outputs land in this folder.

Nothing else is touched: loaders, sparsifiers, floors, operating points, scoring, recovery
evaluation and chance floors are byte-identical to Exp AB.

## Output

`PAPER_RESTRUCTURE/exp_AD_metis_real_networks/{DESIGN.md, run.py, driver.sh, results.csv,
recovery.csv, <network>.log, SUMMARY.md}`
