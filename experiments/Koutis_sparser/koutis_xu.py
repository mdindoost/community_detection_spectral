"""
Koutis & Xu (2016)
"Simple Parallel and Distributed Algorithms for Spectral Graph Sparsification"
ACM Trans. Parallel Comput. 3, 2, Article 14.

Algorithms:
  - compute_spanner_greedy     (Section 2, stretch definition p.14:4)
  - compute_t_bundle_spanner   (Definition 2.1, p.14:4)
  - half_sparsify              (Algorithm 1, Section 3.1, p.14:6)
  - sparsify                   (Algorithm 3, Section 3.3, p.14:9)
  - bundle_decomposition       (Section 3.4, p.14:9)
  - sparsify_bundle            (Section 3.4, Theorem 3.6, p.14:10)
"""

import networkx as nx
import numpy as np
from math import log, ceil


# ─────────────────────────────────────────────────────────────────────
# SPANNER CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────

def compute_spanner_greedy(G, stretch_bound):
    """
    Greedy spanner with weighted stretch (Section 2, p.14:4).

    Stretch of edge e over path p:
        st_p(e) = w_e * sum_{e' in p} (1 / w_{e'})

    A spanner H satisfies st_H(e) <= stretch_bound for all e in G.
    The paper uses stretch_bound = 2*log(n) ("log n-spanner").
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)

        # Shortest path in H using resistance 1/w as edge length
        try:
            dist = nx.shortest_path_length(H, u, v, weight='resistance')
            stretch = w * dist
        except nx.NetworkXNoPath:
            stretch = float('inf')

        if stretch > stretch_bound:
            H.add_edge(u, v, weight=w, resistance=1.0 / w)

    return H


# ─────────────────────────────────────────────────────────────────────
# DEFINITION 2.1: t-BUNDLE SPANNER
# ─────────────────────────────────────────────────────────────────────

def compute_t_bundle_spanner(G, t, stretch_bound=None):
    """
    Definition 2.1 (p.14:4):
        H_1,...,H_t subgraphs of G where H_i is a spanner for
        G - sum_{j=1}^{i-1} H_j.
        H = union(H_j) is the t-bundle spanner.

    Returns (bundle, components, G_remaining).
    """
    if stretch_bound is None:
        n = G.number_of_nodes()
        stretch_bound = 2 * log(max(n, 2))

    components = []
    G_remaining = G.copy()

    for _ in range(t):
        if G_remaining.number_of_edges() == 0:
            break
        spanner_i = compute_spanner_greedy(G_remaining, stretch_bound)
        components.append(spanner_i)
        G_remaining.remove_edges_from(spanner_i.edges())

    # Build union (only carry 'weight', not internal 'resistance')
    bundle = nx.Graph()
    bundle.add_nodes_from(G.nodes())
    for comp in components:
        for u, v, data in comp.edges(data=True):
            bundle.add_edge(u, v, weight=data.get('weight', 1.0))

    return bundle, components, G_remaining


# ─────────────────────────────────────────────────────────────────────
# ALGORITHM 1: HALFSPARSIFY  
# ─────────────────────────────────────────────────────────────────────

def half_sparsify(G, epsilon, rng=None):
    """
    Algorithm 1 — HALFSPARSIFY (p.14:6)

    1. Compute a (48 * log^2(n) / eps^2)-bundle spanner H for G
    2. G_tilde := H
    3. For each edge e not in H:
           add e to G_tilde with probability 1/4 and weight 4*w_e

    Theorem 3.2 (p.14:7):
        (1 - eps) G  <=  G_tilde  <=  (1 + eps) G
        with probability  1 - 1/n^2.
        Expected edges:  O(s * t  +  m / 2).

    Proof sketch:
        With t = 48*log^2(n)/eps^2, Corollary 2.5 gives
        4*w_e*B_e <= eps^2/(6*log n) * G  for edges outside the bundle.
        This satisfies Theorem 3.1 (matrix Chernoff) with R = eps^2/(6*log n),
        yielding  n * exp(-3*log n) = 1/n^2  failure probability.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = G.number_of_nodes()
    ln_n = log(max(n, 2))
    t = max(1, int(ceil(48 * ln_n ** 2 / epsilon ** 2)))

    bundle, components, G_remaining = compute_t_bundle_spanner(G, t)

    G_tilde = bundle.copy()

    for u, v, data in G_remaining.edges(data=True):
        if rng.random() < 0.25:
            w = data.get('weight', 1.0)
            G_tilde.add_edge(u, v, weight=4.0 * w)

    return G_tilde


# ─────────────────────────────────────────────────────────────────────
# ALGORITHM 3: SPARSIFY  
# ─────────────────────────────────────────────────────────────────────

def sparsify(G, epsilon, rho, rng=None):
    """
    Algorithm 3 — SPARSIFY (p.14:9)

    G_0 := G
    For i = 1 to ceil(log2(rho)):
        G_i := HALFSPARSIFY(G_{i-1},  eps / ceil(log2(rho)))
    Return G_{ceil(log2(rho))}

    Theorem 3.4 (p.14:9):
        (1 - eps) G  <=  G_tilde  <=  (1 + eps) G   w.h.p.
        Expected edges:  O(s * t  +  m / rho),
        where  t = O(log^2(n) * log^2(rho) / eps^2).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    num_iters = max(1, int(ceil(log(rho) / log(2))))
    eps_per_iter = epsilon / num_iters

    G_curr = G.copy()
    for _ in range(num_iters):
        G_curr = half_sparsify(G_curr, eps_per_iter, rng=rng)

    return G_curr


# ─────────────────────────────────────────────────────────────────────
# BUNDLE DECOMPOSITION 
# ─────────────────────────────────────────────────────────────────────

def bundle_decomposition(G, stretch_bound=None):
    """
    Full bundle decomposition: peel spanners until G is exhausted (G = H).
    This is the t-bundle decomposition from Definition 2.1 with t chosen
    so that all edges are covered.

    Returns (components, edge_to_component).
    """
    if stretch_bound is None:
        n = G.number_of_nodes()
        stretch_bound = 2 * log(max(n, 2))

    components = []
    edge_to_component = {}
    G_remaining = G.copy()
    idx = 0

    while G_remaining.number_of_edges() > 0:
        spanner_i = compute_spanner_greedy(G_remaining, stretch_bound)
        if spanner_i.number_of_edges() == 0:
            # Safety: assign leftover edges to current component
            for u, v in G_remaining.edges():
                edge_to_component[(min(u, v), max(u, v))] = idx
            components.append(G_remaining.copy())
            break
        components.append(spanner_i)
        for u, v in spanner_i.edges():
            edge_to_component[(min(u, v), max(u, v))] = idx
        G_remaining.remove_edges_from(spanner_i.edges())
        idx += 1

    return components, edge_to_component


# ─────────────────────────────────────────────────────────────────────
# SPARSIFYBUNDLE  
# ─────────────────────────────────────────────────────────────────────

def sparsify_bundle(G, epsilon, rng=None):
    """
    SPARSIFYBUNDLE (Theorem 3.6, p.14:10)

    1. Full bundle decomposition of G.
    2. Group i = { H_{(i-1)k+1}, ..., H_{ik} },  k = 4*log(n).
       (k derived from Lemma 3.5 via Lemma 2.4, p.14:5,9:
        need  2*log(n) / ((i-1)*k)  <=  1/i  =>  k >= 4*log(n).)
    3. For each edge in group i:  u_e = 1/i   (Lemma 3.5).
    4. T = sum(u_e).
    5. Sampling (p.14:9-10):
       N = ceil(6*log(n) / eps^2) copies per edge.
       Copy weight:  w'_e = w_e * eps^2 / (6*log(n)).
       Each copy enters independently with prob u_e / T.
       Entering copy reweighted by  T / u_e.
       Multiple copies merged by summing weights.

    Theorem 3.6:
        (1 - eps) G  <=  G_tilde  <=  (1 + eps) G
        with probability  1 - 1/n^2.
        Expected edges:  O(s * log^2(n) * log(t) / eps^2).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = G.number_of_nodes()
    ln_n = log(max(n, 2))

    # k = 4*log(n):  Lemma 3.5 via Lemma 2.4
    # With 2t = (i-1)*k components before group i,
    # Lemma 2.4 gives  w_e*R_e <= log(n) / t = 2*log(n) / ((i-1)*k).
    # For bound <= 1/i:  k >= 2*i*log(n)/(i-1).  Max at i=2 => k >= 4*log(n).
    k = max(1, int(ceil(4 * ln_n)))

    components, edge_to_comp = bundle_decomposition(G)

    # Assign groups (Lemma 3.5, p.14:9)
    edge_to_group = {}
    for edge, comp_idx in edge_to_comp.items():
        edge_to_group[edge] = (comp_idx // k) + 1

    # u_e = 1/group  (Lemma 3.5: w_e * R_eff <= 1/group)
    edge_ue = {e: 1.0 / g for e, g in edge_to_group.items()}
    T = sum(edge_ue.values())

    # Multi-copy sampling (p.14:9-10)
    N = max(1, int(ceil(6 * ln_n / epsilon ** 2)))

    G_tilde = nx.Graph()
    G_tilde.add_nodes_from(G.nodes())

    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        ue = edge_ue.get(key, 1.0)
        w = data.get('weight', 1.0)

        copy_weight = w * epsilon ** 2 / (6 * ln_n)
        prob_per_copy = min(1.0, ue / T)

        num_entered = rng.binomial(N, prob_per_copy)

        if num_entered > 0:
            G_tilde.add_edge(u, v, weight=num_entered * copy_weight * T / ue)

    return G_tilde
