#!/usr/bin/env python3
"""Build exp_AE (Graclus) from exp_AD (Metis) by targeted patching.

exp_AD/run.py is itself exp_AB/run.py with four disclosed changes. This adds a
fifth detector and changes nothing else, so any difference in the results is the
detector and not the harness.

WHY GRACLUS: Metis takes k as an input AND imposes a hard balance constraint on
part sizes. Graclus takes k as an input and has NO balance constraint (its only
options are the objective, local-search steps and boundary-points-only). It is
therefore the one available algorithm that holds this paper's stated positive
condition fixed while removing the balance confound.
"""
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
s = SRC.read_text()
orig_len = len(s)
applied = []


def sub1(pattern, repl, name, count=1, regex=False):
    global s
    if regex:
        s2, n = re.subn(pattern, repl, s, count=count)
    else:
        n = s.count(pattern)
        if count and n > count:
            raise SystemExit(f"PATCH {name}: expected <={count} matches, found {n}")
        s2 = s.replace(pattern, repl, count if count else -1)
    if n == 0:
        raise SystemExit(f"PATCH {name}: anchor not found")
    s = s2
    applied.append(f"{name} ({n})")


# --- 1. os import guard ------------------------------------------------------
if "\nimport os" not in s:
    sub1("import numpy as np", "import os\nimport shutil\nimport subprocess\nimport tempfile\n"
         "import numpy as np", "imports")
else:
    sub1("import numpy as np", "import shutil\nimport subprocess\nimport tempfile\n"
         "import numpy as np", "imports")

# --- 2. detector globals -----------------------------------------------------
sub1('METIS_ONLY = False',
     'METIS_ONLY = False\n'
     'DETECTOR = "metis"            # exp_AE: "metis" or "graclus"\n'
     'GRACLUS_BIN = os.environ.get("GRACLUS_BIN", str(REPO / "bin" / "graclus"))\n'
     '# Graclus is deterministic, so a seed sweep over the partitioner is vacuous.\n'
     '# For it the sweep runs over the SPARSIFIER instead (REP_SEEDS, three\n'
     '# replicates) and the baseline is a single exact run, which is its own best.\n'
     'GRACLUS_SEEDS = [0, 1, 2]',
     "detector globals")

# --- 3. the graclus wrapper --------------------------------------------------
WRAPPER = '''

def graclus_partition(g, k, seed=0):
    """Graclus 1.2 (Dhillon, Guan & Kulis 2007), fixed k, normalized cut.

    Unlike Metis it imposes NO balance constraint on part sizes; that is the
    reason this arm exists. Deterministic, so `seed` is accepted for signature
    compatibility with metis_partition and ignored.
    """
    n = g.vcount()
    t0 = time.perf_counter()
    tmpd = tempfile.mkdtemp(prefix="graclus_")
    try:
        gf = os.path.join(tmpd, "g.graph")
        # METIS adjacency format: header "n m" (m = undirected edge count), then
        # one line per vertex listing its 1-indexed neighbours.
        with open(gf, "w") as f:
            f.write("%d %d\\n" % (n, g.ecount()))
            for nb in g.get_adjlist():
                f.write(" ".join(str(x + 1) for x in nb))
                f.write("\\n")
        # graclus strips the directory from its argument and writes
        # "<basename>.part.<k>" into the CURRENT WORKING DIRECTORY, so it must be
        # run with cwd set to the scratch directory and given a bare filename.
        r = subprocess.run([GRACLUS_BIN, "g.graph", str(int(k))], cwd=tmpd,
                           capture_output=True, text=True, timeout=10800)
        pf = os.path.join(tmpd, "g.graph.part.%d" % int(k))
        if not os.path.exists(pf):
            raise RuntimeError("graclus wrote no partition (rc=%s)\\nSTDOUT:%s\\nSTDERR:%s"
                               % (r.returncode, r.stdout[-500:], r.stderr[-500:]))
        memb = np.loadtxt(pf, dtype=np.int64)
        memb = np.atleast_1d(memb)
        if memb.shape[0] != n:
            raise RuntimeError("graclus returned %d labels for %d vertices"
                               % (memb.shape[0], n))
        return memb, time.perf_counter() - t0
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def detect_fixed_k(g, k, seed=0):
    """Dispatch to the selected fixed-k detector."""
    if DETECTOR == "graclus":
        return graclus_partition(g, k, seed=seed)
    return metis_partition(g, k, seed=seed)


def fixed_k_seeds():
    return GRACLUS_SEEDS if DETECTOR == "graclus" else METIS_SEEDS

'''
sub1("\n\n# ---------------------------------------------------------------------------\n"
     "# L-Spar (exp_L_lspar/run.py verbatim)",
     WRAPPER + "\n# ---------------------------------------------------------------------------\n"
     "# L-Spar (exp_L_lspar/run.py verbatim)",
     "graclus wrapper")

# --- 4. seed lists and call sites -------------------------------------------
sub1("for s in METIS_SEEDS:", "for s in fixed_k_seeds():", "baseline seed loop")
sub1("for i, s in enumerate(METIS_SEEDS):", "for i, s in enumerate(fixed_k_seeds()):",
     "arm seed loop")
sub1("p, dt = metis_partition(g, k_fix, seed=s)", "p, dt = detect_fixed_k(g, k_fix, seed=s)",
     "baseline call")
sub1("ms, T_ms = metis_partition(gs, k_fix, seed=s)",
     "ms, T_ms = detect_fixed_k(gs, k_fix, seed=s)", "arm call")

# --- 5. detector label in every emitted row ---------------------------------
sub1('dict(resolution="", detector="metis")', 'dict(resolution="", detector=DETECTOR)',
     "rec_row detector", count=2)
sub1('detector="metis", stochastic=int(arm in STOCHASTIC)',
     'detector=DETECTOR, stochastic=int(arm in STOCHASTIC)', "results detector")

# --- 6. logs ----------------------------------------------------------------
sub1('log(f"\\n  ==== Metis (k_mode={k_mode}, fixed k={k_fix}) on {name} ====")',
     'log(f"\\n  ==== {DETECTOR} (k_mode={k_mode}, fixed k={k_fix}) on {name} ====")',
     "header log")
sub1('log(f"    metis base k={k_fix} ', 'log(f"    {DETECTOR} base k={k_fix} ', "base log")
sub1('log(f"    metis {arm:12s} ', 'log(f"    {DETECTOR} {arm:12s} ', "arm log")

# --- 7. CLI -----------------------------------------------------------------
sub1('    METIS_ONLY = "--metis-only" in sys.argv',
     '    METIS_ONLY = ("--metis-only" in sys.argv) or ("--graclus-only" in sys.argv)\n'
     '    if "--graclus" in sys.argv or "--graclus-only" in sys.argv:\n'
     '        DETECTOR = "graclus"\n'
     '        if not os.path.exists(GRACLUS_BIN):\n'
     '            raise SystemExit("graclus binary not found at %s (set GRACLUS_BIN)" % GRACLUS_BIN)',
     "CLI flag")
sub1('    run_network(sys.argv[2], do_metis=("--metis" in sys.argv or METIS_ONLY), k_mode=km)',
     '    run_network(sys.argv[2],\n'
     '                do_metis=("--metis" in sys.argv or "--graclus" in sys.argv or METIS_ONLY),\n'
     '                k_mode=km)',
     "CLI dispatch")
sub1('raise SystemExit("usage: run.py net <network> [--metis] [--metis-only] [--k-mode nc_base|gt]")',
     'raise SystemExit("usage: run.py net <network> [--metis|--graclus] '
     '[--metis-only|--graclus-only] [--k-mode nc_base|gt]")',
     "usage string")

# DETECTOR is rebound in __main__, so it must be declared global there.
sub1('if __name__ == "__main__":', 'if __name__ == "__main__":\n    pass',
     "main guard placeholder")
s = s.replace('if __name__ == "__main__":\n    pass', 'if __name__ == "__main__":')

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(s)
print("patched %s -> %s  (%d -> %d bytes)" % (SRC, DST, orig_len, len(s)))
for a in applied:
    print("   applied:", a)
