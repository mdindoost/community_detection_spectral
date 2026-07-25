"""Normalized-Laplacian spectral gap alpha (smallest nonzero eigenvalue of
L = I - D^-1/2 A D^-1/2) per network, i.e. Liu et al.'s alpha; reports 1/alpha,
the factor by which their (1 +- eps/alpha) bound is inflated."""
import sys, time, csv
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
sys.path.insert(0, "/home/md724/community_detection_spectral/PAPER_RESTRUCTURE/exp_K_weighted_regime")
from run import load_lcc_graph, load_email_labeled, HERE

rows = []
for name in sys.argv[1:]:
    t0 = time.time()
    if name == "email-Eu-core-labeled":
        g, _ = load_email_labeled()
    else:
        g = load_lcc_graph(name)
    n = g.vcount()
    e = np.asarray(g.get_edgelist(), dtype=np.int64)
    d = np.asarray(g.degree(), dtype=np.float64)
    dm = 1.0 / np.sqrt(d)
    r = np.concatenate([e[:, 0], e[:, 1]]); c = np.concatenate([e[:, 1], e[:, 0]])
    v = dm[r] * dm[c]
    M = csr_matrix((v, (r, c)), shape=(n, n))       # D^-1/2 A D^-1/2
    # lambda_2(M) = 1 - alpha
    vals = eigsh(M, k=2, which="LA", return_eigenvectors=False, tol=1e-6, maxiter=20000)
    lam2 = float(np.sort(vals)[0])
    alpha = 1.0 - lam2
    rows.append(dict(network=name, n=n, m=g.ecount(), alpha=alpha, inv_alpha=1.0 / alpha,
                     secs=time.time() - t0))
    print(f"{name:24s} alpha={alpha:.6g}  1/alpha={1/alpha:.1f}  ({time.time()-t0:.1f}s)", flush=True)
    with open(HERE / "spectral_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
