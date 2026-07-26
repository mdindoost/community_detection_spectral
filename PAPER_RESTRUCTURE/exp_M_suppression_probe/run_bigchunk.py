#!/usr/bin/env python3
"""Wrapper for exp_M run.py that survives mega-hub graphs (wiki-topcats).

run.py's edge_triangles() uses a FIXED chunk of 20000 edges.  On wiki-topcats
(max degree 238,342) one such chunk materialises A[eu[i:j]] with up to
4.77e9 stored values -- past scipy's int32 index limit (2^31-1) -- which
segfaults (observed: rc=139, peak RSS 22.8GB, so not OOM).

This wrapper monkey-patches edge_triangles with a version that picks chunk
boundaries adaptively so each batch stays under an nnz budget.  The per-edge
result is computed independently for every edge, so batching is a pure
scheduling choice: outputs are numerically IDENTICAL to run.py's.
run.py itself is left untouched (md5 unchanged).

Also arms faulthandler with a periodic stack dump so a hang/crash is localised.
"""
import faulthandler
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("expm_run", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

NNZ_BUDGET = int(4e7)


def edge_triangles_budgeted(g, eu, ev, chunk=20000):
    n = g.vcount()
    m = eu.size
    deg = np.asarray(g.degree(), dtype=np.int64)
    A = sp.csr_matrix(
        (np.ones(2 * m, dtype=np.float32),
         (np.concatenate([eu, ev]), np.concatenate([ev, eu]))),
        shape=(n, n), dtype=np.float32,
    )
    A.data[:] = 1.0
    w = deg[eu] + deg[ev]          # nnz that edge i contributes to a batch
    cw = np.cumsum(w)
    out = np.empty(m, dtype=np.float64)
    i = 0
    nb = 0
    t0 = time.time()
    while i < m:
        base = cw[i - 1] if i > 0 else 0
        j = int(np.searchsorted(cw, base + NNZ_BUDGET, side="right"))
        j = max(j, i + 1)
        j = min(j, m)
        Au = A[eu[i:j]]
        Av = A[ev[i:j]]
        out[i:j] = np.asarray(Au.multiply(Av).sum(axis=1)).ravel()
        del Au, Av
        i = j
        nb += 1
        if nb % 200 == 0:
            print(f"    [tri] {i}/{m} edges, {nb} batches, {time.time()-t0:.0f}s",
                  flush=True)
    print(f"    [tri] done {m} edges in {nb} batches, {time.time()-t0:.0f}s",
          flush=True)
    return out


R.edge_triangles = edge_triangles_budgeted

if __name__ == "__main__":
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)
    sys.argv[0] = str(HERE / "run.py")
    R.main()
