# Exp W — leaf-shedding (H-O1 follow-up, pre-registered design)

Registered 2026-07-25 before any Exp W code/run. Promotes exp_O's post-hoc hypothesis H-O1 to
a pre-registered test. Greenlit by Mohammad (small threads approved "if they help the paper").

**Claim it could change:** the descriptive account of WHAT sparsification's fragmentation does
at node level (feeds C6's footnote and the discussion). Exp O killed "cores preserved /
periphery shed" measured by EMBEDDEDNESS; H-O1 says the real selectivity may be by DEGREE /
K-CORE: fragments concentrate in low-degree, low-coreness nodes (embeddedness was an anti-core
proxy — leaves score 1.0). If it survives its granularity control, the paper gains one honest
descriptive sentence ("sparsification's fragments are the graph's low-coreness leaves — but so
are a resolution-matched partition's" OR "...and that selectivity is sparsification-specific").

**Design.** Reuse exp_O machinery/data verbatim (exp_O_core_preservation/run.py, results.csv,
node_attrs.csv already hold degree, k-core, per-decile agreement columns ahc_d*/ahd_d*).
Networks: the exp_O five. Cells: the moderate-fragmentation cells exp_O identified (dspar 0.5,
lspar 0.5, lspar 0.2 where fragments exist) + their resmatch controls + config-null arm.
Measurements: (1) fragment-node composition by DEGREE decile and K-CORE decile (enrichment in
bottom-2 vs top-2, mirroring exp_O's P2 but on the right axes); (2) agreement gradient by
k-core/degree decile (already recorded — analyze, don't re-run); (3) same statistics for the
resolution-matched control partitions and the config-null arm. Recompute fragment composition
from stored partitions if exp_O saved them; otherwise re-run ONLY the needed cells (cheap,
small five, one capped python at a time).

**Pre-registered predictions:**
- P1: fragment nodes are >=2x over-represented in the bottom-2 DEGREE deciles and bottom-2
  K-CORE deciles, in >= 2/3 of testable fragmentation cells per network.
- P2: the resolution-matched control shows the SAME direction of concentration (because
  granularity effects also shed leaves) — the honest question is magnitude: sparsification's
  enrichment exceeds the control's by >=1.5x in fewer than half the cells (i.e. we predict
  this is NOT sparsification-specific, matching exp_O V3's pattern).
- P3: config-null arm shows weaker degree-selectivity than the real graph.

**Kill criterion:** if P1 fails (no >=2x bottom-2 enrichment in a majority of testable cells),
H-O1 is dead outright. If P1 holds but the resmatch control matches or exceeds the enrichment
in >= half the cells, the finding is recorded as a GRANULARITY fact, not a sparsification
fact, and enters the paper only in that form. No rescue tweaks.

**Output:** PAPER_RESTRUCTURE/exp_W_leaf_shedding/{DESIGN.md,analysis or run.py,CSVs,SUMMARY.md}.
Cost: mostly analysis of existing exp_O data + a few cheap re-runs. Local machine, MemoryMax=3G.
