# Big-machine TODO

**Status after Phase 1: NOTHING REQUIRED.** All 17 networks completed both arms on the local
machine (cit-Patents and wiki-topcats null arms used 1 rewire seed × 2 sparsification seeds instead
of 2×2, due to the 15-min per-arm budget; results are unambiguous regardless).

## Reserve machine (verified 2026-07-24)

Fuji — Ubuntu 22.04, 62 GB RAM (51 free), 24-core Xeon w5-3423, Python 3.10, venv OK, PyPI
reachable. ⚠️ Disk 98% full (11 GB free) — enough for ~3 GB jobs; clean before anything bigger.
Access: user has SSH; for agent-driven runs, key-based auth needed (`ssh-copy-id mayooran@Fuji`).

## Optional items (only if wanted / if referees ask)

1. Re-run cit-Patents + wiki-topcats null arms with full seeds (2 rewire × 3 sparsification) for
   tighter error bars in the final Exp B table. ~1–2 h on Fuji. Cosmetic.
2. com-Orkut (117M edges) / com-LiveJournal (34M) — NOT in the draft's tables and not needed for
   any claim. Add only if a reviewer explicitly requests larger scale in revision. Fuji's 62 GB
   handles both.
3. Rebuild the local repo venv (broken: built for Python 3.12, system now 3.14). Working temp env:
   scratchpad/v14 (see PAPER_RESTRUCTURE/README.md).
