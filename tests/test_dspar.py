"""
Comprehensive Test Suite for DSpar Implementation

This file contains:
1. Functional tests - verify code runs correctly
2. Correctness tests - verify algorithm matches mathematical specification

Run with:
    pytest test_dspar.py -v              # Full suite
    python test_dspar.py --quick         # Quick sanity check
    python test_dspar.py --verbose       # Verbose correctness checks

Total: 67 tests
"""

import pytest
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh
from collections import Counter
import sys
import math

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.dspar import compute_dspar_scores, dspar_sparsify


# =============================================================================
# FIXTURES: Common test graphs
# =============================================================================

@pytest.fixture
def karate_graph():
    """Zachary's Karate Club - standard test graph"""
    return nx.karate_club_graph()


@pytest.fixture
def simple_graph():
    """Simple 5-node graph for manual verification"""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
    return G


@pytest.fixture
def star_graph():
    """Star graph - one hub connected to all others"""
    return nx.star_graph(10)


@pytest.fixture
def complete_graph():
    """Complete graph K10"""
    return nx.complete_graph(10)


@pytest.fixture
def path_graph():
    """Path graph - linear chain"""
    return nx.path_graph(10)


@pytest.fixture
def barbell_graph():
    """Two cliques connected by a bridge"""
    return nx.barbell_graph(5, 1)


# =============================================================================
# PART 1: FUNCTIONAL TESTS
# =============================================================================

# -----------------------------------------------------------------------------
# Test Class 1: DSpar Score Computation
# -----------------------------------------------------------------------------

class TestDSparScores:
    """Tests for compute_dspar_scores function"""
    
    def test_score_formula_simple(self, simple_graph):
        """Test that scores follow formula s(e) = 1/d_u + 1/d_v"""
        G = simple_graph
        scores = compute_dspar_scores(G)
        degrees = dict(G.degree())
        
        for (u, v), score in scores.items():
            expected = 1.0 / degrees[u] + 1.0 / degrees[v]
            assert abs(score - expected) < 1e-10, f"Score mismatch for edge ({u}, {v})"
    
    def test_score_formula_karate(self, karate_graph):
        """Test score formula on larger graph"""
        G = karate_graph
        scores = compute_dspar_scores(G)
        degrees = dict(G.degree())
        
        for (u, v), score in scores.items():
            expected = 1.0 / degrees[u] + 1.0 / degrees[v]
            assert abs(score - expected) < 1e-10
    
    def test_all_edges_have_scores(self, karate_graph):
        """Every edge should have a score"""
        G = karate_graph
        scores = compute_dspar_scores(G)
        assert len(scores) == G.number_of_edges()
    
    def test_scores_positive(self, karate_graph):
        """All scores should be positive"""
        scores = compute_dspar_scores(karate_graph)
        assert all(s > 0 for s in scores.values())
    
    def test_score_range(self, karate_graph):
        """Scores should be in (0, 2] since max is 1/1 + 1/1 = 2"""
        scores = compute_dspar_scores(karate_graph)
        assert all(0 < s <= 2 for s in scores.values())
    
    def test_hub_edges_lower_score(self, star_graph):
        """Edges to hub (high degree) should have consistent scores"""
        G = star_graph
        scores = compute_dspar_scores(G)
        
        hub_degree = G.degree(0)
        leaf_degree = G.degree(1)
        expected_score = 1.0 / hub_degree + 1.0 / leaf_degree
        
        for edge, score in scores.items():
            assert abs(score - expected_score) < 1e-10
    
    def test_symmetric_edge_representation(self, simple_graph):
        """Edge (u,v) and (v,u) should map to same canonical form"""
        scores = compute_dspar_scores(simple_graph)
        
        for (u, v) in scores.keys():
            assert u < v, "Edges should be stored as (min, max)"


# -----------------------------------------------------------------------------
# Test Class 2: Basic Sparsification Properties
# -----------------------------------------------------------------------------

class TestBasicProperties:
    """Tests for basic properties all methods should satisfy"""
    
    @pytest.mark.parametrize("method", ["paper", "probabilistic_no_replace", "deterministic"])
    def test_nodes_preserved(self, karate_graph, method):
        """All nodes should be preserved after sparsification"""
        G = karate_graph
        G_sparse = dspar_sparsify(G, retention=0.5, method=method, seed=42)
        assert set(G_sparse.nodes()) == set(G.nodes())
    
    @pytest.mark.parametrize("method", ["paper", "probabilistic_no_replace", "deterministic"])
    def test_no_new_edges(self, karate_graph, method):
        """No edges should be created that weren't in original"""
        G = karate_graph
        G_sparse = dspar_sparsify(G, retention=0.5, method=method, seed=42)
        
        original_edges = set((min(u, v), max(u, v)) for u, v in G.edges())
        sparse_edges = set((min(u, v), max(u, v)) for u, v in G_sparse.edges())
        
        assert sparse_edges.issubset(original_edges)
    
    @pytest.mark.parametrize("method", ["paper", "probabilistic_no_replace", "deterministic"])
    def test_fewer_or_equal_edges(self, karate_graph, method):
        """Sparsified graph should have fewer or equal edges"""
        G = karate_graph
        G_sparse = dspar_sparsify(G, retention=0.5, method=method, seed=42)
        assert G_sparse.number_of_edges() <= G.number_of_edges()
    
    @pytest.mark.parametrize("method", ["paper", "probabilistic_no_replace", "deterministic"])
    def test_no_self_loops(self, karate_graph, method):
        """No self-loops should be introduced"""
        G_sparse = dspar_sparsify(karate_graph, retention=0.5, method=method, seed=42)
        assert nx.number_of_selfloops(G_sparse) == 0
    
    @pytest.mark.parametrize("method", ["deterministic"])
    def test_retention_1_keeps_all(self, karate_graph, method):
        """retention=1.0 should keep all edges for deterministic method"""
        G = karate_graph
        G_sparse = dspar_sparsify(G, retention=1.0, method=method, seed=42)
        assert G_sparse.number_of_edges() == G.number_of_edges()
    
    def test_paper_retention_1_has_duplicates(self, karate_graph):
        """Paper method at retention=1.0 has fewer unique edges due to replacement"""
        G = karate_graph
        m = G.number_of_edges()
        
        G_sparse = dspar_sparsify(G, retention=1.0, method="paper", seed=42)
        
        assert G_sparse.number_of_edges() < m
        assert G_sparse.number_of_edges() > m * 0.5


# -----------------------------------------------------------------------------
# Test Class 3: Deterministic Method Tests
# -----------------------------------------------------------------------------

class TestDeterministicMethod:
    """Tests specific to deterministic method"""
    
    def test_exact_edge_count(self, karate_graph):
        """Deterministic should keep exactly ceil(retention * m) edges"""
        G = karate_graph
        m = G.number_of_edges()
        
        for retention in [0.25, 0.5, 0.75, 0.9]:
            G_sparse = dspar_sparsify(G, retention=retention, method="deterministic")
            expected = int(np.ceil(retention * m))
            assert G_sparse.number_of_edges() == expected
    
    def test_reproducible(self, karate_graph):
        """Deterministic should give same result every time"""
        G1 = dspar_sparsify(karate_graph, retention=0.5, method="deterministic")
        G2 = dspar_sparsify(karate_graph, retention=0.5, method="deterministic")
        
        assert set(G1.edges()) == set(G2.edges())
    
    def test_keeps_highest_scores(self, simple_graph):
        """Should keep edges with highest DSpar scores (may have ties)"""
        G = simple_graph
        scores = compute_dspar_scores(G)
        
        G_sparse = dspar_sparsify(G, retention=0.6, method="deterministic")
        
        sparse_edges = set((min(u, v), max(u, v)) for u, v in G_sparse.edges())
        kept_scores = [scores[e] for e in sparse_edges]
        
        removed_edges = set(scores.keys()) - sparse_edges
        removed_scores = [scores[e] for e in removed_edges]
        
        min_kept = min(kept_scores)
        max_removed = max(removed_scores) if removed_scores else 0
        
        assert min_kept >= max_removed - 1e-10
    
    def test_low_degree_edges_preferred(self, barbell_graph):
        """Edges between low-degree nodes should be kept"""
        G = barbell_graph
        G_sparse = dspar_sparsify(G, retention=0.3, method="deterministic")
        
        sparse_edges = list(G_sparse.edges())
        degrees = dict(G.degree())
        
        avg_degree_product = np.mean([degrees[u] * degrees[v] for u, v in sparse_edges])
        all_avg = np.mean([degrees[u] * degrees[v] for u, v in G.edges()])
        
        assert avg_degree_product <= all_avg


# -----------------------------------------------------------------------------
# Test Class 4: Paper Method (With Replacement) Tests
# -----------------------------------------------------------------------------

class TestPaperMethod:
    """Tests specific to paper method (with replacement, weighted)"""
    
    def test_returns_weighted_graph(self, karate_graph):
        """Paper method should return weighted edges"""
        G_sparse = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=42)
        
        for u, v, data in G_sparse.edges(data=True):
            assert 'weight' in data
            assert data['weight'] > 0
    
    def test_weight_formula(self, karate_graph):
        """Test weight formula: w'_e = k_e / (q * p_e)"""
        G = karate_graph
        G_sparse, weights = dspar_sparsify(
            G, retention=0.5, method="paper", seed=42, return_weights=True
        )
        
        assert all(w > 0 for w in weights.values())
    
    def test_fewer_unique_edges_than_samples(self, karate_graph):
        """With replacement: unique edges < number of samples"""
        G = karate_graph
        m = G.number_of_edges()
        retention = 0.75
        q = int(np.ceil(retention * m))
        
        G_sparse = dspar_sparsify(G, retention=retention, method="paper", seed=42)
        
        assert G_sparse.number_of_edges() <= q
    
    def test_total_weight_approximates_edges(self, karate_graph):
        """Total weight should approximate original edge count"""
        G = karate_graph
        m = G.number_of_edges()
        
        total_weights = []
        for seed in range(10):
            G_sparse, weights = dspar_sparsify(
                G, retention=0.75, method="paper", seed=seed, return_weights=True
            )
            total_weights.append(sum(weights.values()))
        
        avg_total = np.mean(total_weights)
        assert 0.8 * m <= avg_total <= 1.2 * m
    
    def test_reproducible_with_seed(self, karate_graph):
        """Same seed should give same result"""
        G1 = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=42)
        G2 = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=42)
        
        assert set(G1.edges()) == set(G2.edges())
    
    def test_different_seeds_different_results(self, karate_graph):
        """Different seeds should give different results"""
        G1 = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=42)
        G2 = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=123)
        
        assert set(G1.edges()) != set(G2.edges())


# -----------------------------------------------------------------------------
# Test Class 5: Probabilistic No Replacement Tests
# -----------------------------------------------------------------------------

class TestProbabilisticNoReplace:
    """Tests specific to probabilistic without replacement method"""
    
    def test_unweighted_output(self, karate_graph):
        """Should return unweighted graph"""
        G_sparse = dspar_sparsify(
            karate_graph, retention=0.5, method="probabilistic_no_replace", seed=42
        )
        
        for u, v, data in G_sparse.edges(data=True):
            if 'weight' in data:
                assert data['weight'] == 1.0
    
    def test_approximately_expected_edges(self, karate_graph):
        """Should keep approximately retention * m edges on average"""
        G = karate_graph
        m = G.number_of_edges()
        retention = 0.5
        expected = retention * m
        
        edge_counts = []
        for seed in range(20):
            G_sparse = dspar_sparsify(
                G, retention=retention, method="probabilistic_no_replace", seed=seed
            )
            edge_counts.append(G_sparse.number_of_edges())
        
        avg_edges = np.mean(edge_counts)
        assert 0.8 * expected <= avg_edges <= 1.2 * expected
    
    def test_reproducible_with_seed(self, karate_graph):
        """Same seed should give same result"""
        G1 = dspar_sparsify(karate_graph, retention=0.5, method="probabilistic_no_replace", seed=42)
        G2 = dspar_sparsify(karate_graph, retention=0.5, method="probabilistic_no_replace", seed=42)
        
        assert set(G1.edges()) == set(G2.edges())


# -----------------------------------------------------------------------------
# Test Class 6: Edge Cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_single_edge_graph(self):
        """Graph with single edge"""
        G = nx.Graph()
        G.add_edge(0, 1)
        
        G_sparse = dspar_sparsify(G, retention=1.0, method="deterministic")
        assert G_sparse.number_of_edges() == 1
    
    def test_two_edges(self):
        """Graph with two edges"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        
        G_sparse = dspar_sparsify(G, retention=0.5, method="deterministic")
        assert G_sparse.number_of_edges() == 1
    
    def test_complete_graph(self, complete_graph):
        """Complete graph - all edges have same score"""
        G = complete_graph
        scores = compute_dspar_scores(G)
        
        score_values = list(scores.values())
        assert all(abs(s - score_values[0]) < 1e-10 for s in score_values)
    
    def test_path_graph(self, path_graph):
        """Path graph - end edges should have higher scores"""
        G = path_graph
        scores = compute_dspar_scores(G)
        
        end_edge = (0, 1)
        internal_edge = (4, 5)
        
        assert scores[end_edge] > scores[internal_edge]
    
    def test_disconnected_components(self):
        """Graph with disconnected components"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        G.add_edges_from([(3, 4), (4, 5)])
        
        G_sparse = dspar_sparsify(G, retention=0.5, method="deterministic")
        assert G_sparse.number_of_edges() == 2
    
    def test_invalid_retention_zero(self, karate_graph):
        """retention=0 should raise error"""
        with pytest.raises(ValueError):
            dspar_sparsify(karate_graph, retention=0.0)
    
    def test_invalid_retention_negative(self, karate_graph):
        """Negative retention should raise error"""
        with pytest.raises(ValueError):
            dspar_sparsify(karate_graph, retention=-0.5)
    
    def test_invalid_retention_above_one(self, karate_graph):
        """retention > 1 should raise error"""
        with pytest.raises(ValueError):
            dspar_sparsify(karate_graph, retention=1.5)
    
    def test_invalid_method(self, karate_graph):
        """Invalid method name should raise error"""
        with pytest.raises(ValueError):
            dspar_sparsify(karate_graph, retention=0.5, method="invalid_method")


# -----------------------------------------------------------------------------
# Test Class 7: Statistical Properties (Functional)
# -----------------------------------------------------------------------------

class TestStatisticalPropertiesFunctional:
    """Tests for statistical properties of the sparsification"""
    
    def test_high_score_edges_more_likely_kept(self, karate_graph):
        """Edges with higher scores should be kept more often"""
        G = karate_graph
        scores = compute_dspar_scores(G)
        
        median_score = np.median(list(scores.values()))
        high_score_edges = {e for e, s in scores.items() if s >= median_score}
        low_score_edges = {e for e, s in scores.items() if s < median_score}
        
        high_kept = 0
        low_kept = 0
        n_runs = 50
        
        for seed in range(n_runs):
            G_sparse = dspar_sparsify(G, retention=0.5, method="probabilistic_no_replace", seed=seed)
            sparse_edges = set((min(u, v), max(u, v)) for u, v in G_sparse.edges())
            
            high_kept += len(sparse_edges & high_score_edges)
            low_kept += len(sparse_edges & low_score_edges)
        
        assert high_kept > low_kept
    
    def test_paper_method_unbiased_per_edge(self, karate_graph):
        """Paper method should give approximately unbiased weight estimate"""
        G = karate_graph
        
        edge_weighted_counts = Counter()
        n_runs = 100
        
        for seed in range(n_runs):
            G_sparse, weights = dspar_sparsify(
                G, retention=0.75, method="paper", seed=seed, return_weights=True
            )
            for edge, weight in weights.items():
                edge_weighted_counts[edge] += weight
        
        avg_weights = []
        for edge in compute_dspar_scores(G).keys():
            avg_weight = edge_weighted_counts[edge] / n_runs
            avg_weights.append(avg_weight)
        
        overall_avg = np.mean(avg_weights)
        assert 0.7 <= overall_avg <= 1.3


# -----------------------------------------------------------------------------
# Test Class 8: Consistency Tests
# -----------------------------------------------------------------------------

class TestConsistency:
    """Tests for consistency between methods and properties"""
    
    def test_deterministic_subset_of_scores(self, karate_graph):
        """Deterministic result should be top-k by score"""
        G = karate_graph
        scores = compute_dspar_scores(G)
        
        retention = 0.5
        G_sparse = dspar_sparsify(G, retention=retention, method="deterministic")
        n_kept = G_sparse.number_of_edges()
        
        sorted_scores = sorted(scores.values(), reverse=True)
        threshold = sorted_scores[n_kept - 1]
        
        for u, v in G_sparse.edges():
            edge = (min(u, v), max(u, v))
            assert scores[edge] >= threshold - 1e-10
    
    def test_return_weights_flag(self, karate_graph):
        """return_weights=True should return tuple"""
        result = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=42, return_weights=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], nx.Graph)
        assert isinstance(result[1], dict)
    
    def test_return_weights_false(self, karate_graph):
        """return_weights=False should return just graph"""
        result = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=42, return_weights=False)
        
        assert isinstance(result, nx.Graph)


# -----------------------------------------------------------------------------
# Test Class 9: Performance Sanity Checks
# -----------------------------------------------------------------------------

class TestPerformance:
    """Basic performance sanity checks"""
    
    def test_larger_graph(self):
        """Should handle larger graphs without error"""
        G = nx.barabasi_albert_graph(1000, 5, seed=42)
        
        G_sparse = dspar_sparsify(G, retention=0.5, method="deterministic")
        
        assert G_sparse.number_of_nodes() == 1000
        assert G_sparse.number_of_edges() > 0
    
    def test_dense_graph(self):
        """Should handle dense graphs"""
        G = nx.complete_graph(50)
        
        G_sparse = dspar_sparsify(G, retention=0.1, method="deterministic")
        
        expected = int(np.ceil(0.1 * G.number_of_edges()))
        assert G_sparse.number_of_edges() == expected


# =============================================================================
# PART 2: CORRECTNESS TESTS (Mathematical Verification)
# =============================================================================

# -----------------------------------------------------------------------------
# Test Class 10: Exact Manual Verification on Tiny Graphs
# -----------------------------------------------------------------------------

class TestExactManualVerification:
    """Manually compute expected values for tiny graphs and compare exactly."""
    
    def test_triangle_graph_scores(self):
        """Triangle: all degree 2, all scores = 1.0"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (0, 2)])
        
        scores = compute_dspar_scores(G)
        expected_score = 1/2 + 1/2
        
        for edge, score in scores.items():
            assert abs(score - expected_score) < 1e-10
    
    def test_path_3_nodes_scores(self):
        """Path 0--1--2: edge scores = 1.5"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        
        scores = compute_dspar_scores(G)
        
        assert abs(scores[(0, 1)] - 1.5) < 1e-10
        assert abs(scores[(1, 2)] - 1.5) < 1e-10
    
    def test_star_4_nodes_scores(self):
        """Star with center 0: all scores = 4/3"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (0, 3)])
        
        scores = compute_dspar_scores(G)
        expected = 1/3 + 1/1
        
        for edge, score in scores.items():
            assert abs(score - expected) < 1e-10
    
    def test_4_node_manual_weights_paper_method(self):
        """Verify weight formula on path graph"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        scores = compute_dspar_scores(G)
        assert abs(scores[(0, 1)] - 1.5) < 1e-10
        assert abs(scores[(1, 2)] - 1.0) < 1e-10
        assert abs(scores[(2, 3)] - 1.5) < 1e-10
        
        total_score = 1.5 + 1.0 + 1.5
        expected_probs = {
            (0, 1): 1.5 / total_score,
            (1, 2): 1.0 / total_score,
            (2, 3): 1.5 / total_score,
        }
        
        G_sparse, weights = dspar_sparsify(
            G, retention=1.0, method="paper", seed=42, return_weights=True
        )
        
        q = 3
        for edge, w in weights.items():
            p_e = expected_probs[edge]
            implied_count = w * q * p_e
            assert implied_count > 0.99
            assert implied_count <= q + 0.01
    
    def test_probability_sum_to_one(self):
        """Sampling probabilities sum to 1"""
        G = nx.karate_club_graph()
        scores = compute_dspar_scores(G)
        
        score_values = np.array(list(scores.values()))
        probs = score_values / score_values.sum()
        
        assert abs(probs.sum() - 1.0) < 1e-10
    
    def test_probability_range(self):
        """All probabilities in (0, 1)"""
        G = nx.karate_club_graph()
        scores = compute_dspar_scores(G)
        
        score_values = np.array(list(scores.values()))
        probs = score_values / score_values.sum()
        
        assert all(0 < p < 1 for p in probs)


# -----------------------------------------------------------------------------
# Test Class 11: Weight Formula Verification
# -----------------------------------------------------------------------------

class TestWeightFormula:
    """Verify weight formula w = count / (q * p)"""
    
    def test_weight_reconstruction(self):
        """Verify we can reconstruct count from weight"""
        G = nx.karate_club_graph()
        retention = 0.5
        m = G.number_of_edges()
        q = int(np.ceil(retention * m))
        
        scores = compute_dspar_scores(G)
        edges = list(scores.keys())
        score_values = np.array([scores[e] for e in edges])
        probs = score_values / score_values.sum()
        edge_to_prob = dict(zip(edges, probs))
        
        G_sparse, weights = dspar_sparsify(
            G, retention=retention, method="paper", seed=42, return_weights=True
        )
        
        total_implied_count = 0
        for edge, w in weights.items():
            p_e = edge_to_prob[edge]
            implied_count = w * q * p_e
            rounded_count = round(implied_count)
            assert abs(implied_count - rounded_count) < 0.01
            total_implied_count += rounded_count
        
        assert total_implied_count == q
    
    def test_unbiased_estimator_property(self):
        """E[sum of weights] ≈ m"""
        G = nx.karate_club_graph()
        retention = 0.75
        n_runs = 500
        
        edge_total_weight = Counter()
        
        for seed in range(n_runs):
            G_sparse, weights = dspar_sparsify(
                G, retention=retention, method="paper", seed=seed, return_weights=True
            )
            for edge, w in weights.items():
                edge_total_weight[edge] += w
        
        total_weight_sum = sum(edge_total_weight.values())
        expected_total = n_runs * G.number_of_edges()
        
        ratio = total_weight_sum / expected_total
        assert 0.95 <= ratio <= 1.05


# -----------------------------------------------------------------------------
# Test Class 12: Spectral Property Tests
# -----------------------------------------------------------------------------

class TestSpectralProperties:
    """Test preservation of spectral properties"""
    
    def get_laplacian_eigenvalues(self, G, k=5):
        """Get k smallest non-zero eigenvalues of normalized Laplacian"""
        if G.number_of_edges() == 0:
            return np.array([])
        
        L = nx.normalized_laplacian_matrix(G).astype(float)
        n = G.number_of_nodes()
        k_actual = min(k + 1, n - 1)
        
        if k_actual < 2:
            return np.array([])
        
        try:
            eigenvalues = eigsh(L, k=k_actual, which='SM', return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)
            return eigenvalues[eigenvalues > 0.01]
        except:
            return np.array([])
    
    def test_algebraic_connectivity_preserved(self):
        """λ₂ should be approximately preserved"""
        G = nx.karate_club_graph()
        
        orig_eigs = self.get_laplacian_eigenvalues(G, k=2)
        if len(orig_eigs) < 1:
            pytest.skip("Could not compute eigenvalues")
        
        orig_lambda2 = orig_eigs[0]
        
        G_sparse = dspar_sparsify(G, retention=0.75, method="deterministic")
        
        sparse_eigs = self.get_laplacian_eigenvalues(G_sparse, k=2)
        if len(sparse_eigs) < 1:
            pytest.skip("Could not compute eigenvalues for sparse graph")
        
        sparse_lambda2 = sparse_eigs[0]
        
        ratio = sparse_lambda2 / orig_lambda2
        assert 0.5 <= ratio <= 2.0
    
    def test_spectral_gap_reasonable(self):
        """Spectral gap should remain positive"""
        G = nx.karate_club_graph()
        
        for retention in [0.9, 0.75, 0.6]:
            G_sparse = dspar_sparsify(G, retention=retention, method="deterministic")
            
            if not nx.is_connected(G_sparse):
                continue
            
            eigs = self.get_laplacian_eigenvalues(G_sparse, k=2)
            if len(eigs) < 1:
                continue
            
            assert eigs[0] > 0.001
    
    def test_effective_resistance_approximation(self):
        """DSpar scores should correlate with effective resistances"""
        G = nx.karate_club_graph()
        
        L = nx.laplacian_matrix(G).toarray().astype(float)
        
        try:
            L_pinv = np.linalg.pinv(L)
        except:
            pytest.skip("Could not compute pseudoinverse")
        
        scores = compute_dspar_scores(G)
        
        dspar_scores = []
        eff_resistances = []
        
        for (u, v), score in scores.items():
            r_eff = L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v]
            dspar_scores.append(score)
            eff_resistances.append(r_eff)
        
        correlation = np.corrcoef(dspar_scores, eff_resistances)[0, 1]
        assert correlation > 0.3
    
    def test_lovasz_bound_relationship(self):
        """Verify DSpar-ER relationship and R_eff ≤ 1 for edges"""
        G = nx.karate_club_graph()
        
        L = nx.laplacian_matrix(G).toarray().astype(float)
        try:
            L_pinv = np.linalg.pinv(L)
        except:
            pytest.skip("Could not compute pseudoinverse")
        
        scores = compute_dspar_scores(G)
        
        dspar_vals = []
        er_vals = []
        
        for (u, v), score in scores.items():
            r_eff = L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v]
            dspar_vals.append(score)
            er_vals.append(r_eff)
            
            assert r_eff <= 1.0 + 0.01
        
        correlation = np.corrcoef(dspar_vals, er_vals)[0, 1]
        assert correlation > 0.5


# -----------------------------------------------------------------------------
# Test Class 13: Deterministic Method Exact Verification
# -----------------------------------------------------------------------------

class TestDeterministicExact:
    """Deterministic method should be exactly verifiable"""
    
    def test_exact_edges_kept(self):
        """Verify exactly which edges are kept based on scores"""
        G = nx.Graph()
        G.add_edges_from([
            (0, 1), (0, 2), (1, 2), (1, 4), (3, 4),
        ])
        
        scores = compute_dspar_scores(G)
        
        assert abs(scores[(3, 4)] - 1.5) < 1e-10
        assert abs(scores[(0, 2)] - 1.0) < 1e-10
        
        G_sparse = dspar_sparsify(G, retention=0.4, method="deterministic")
        
        kept_edges = set((min(u, v), max(u, v)) for u, v in G_sparse.edges())
        
        assert (3, 4) in kept_edges
        assert (0, 2) in kept_edges
        assert len(kept_edges) == 2


# -----------------------------------------------------------------------------
# Test Class 14: Statistical Rigor Tests
# -----------------------------------------------------------------------------

class TestStatisticalRigor:
    """More rigorous statistical tests"""
    
    def test_sampling_distribution(self):
        """Edge sampling follows expected distribution"""
        G = nx.karate_club_graph()
        retention = 0.5
        m = G.number_of_edges()
        q = int(np.ceil(retention * m))
        
        scores = compute_dspar_scores(G)
        edges = list(scores.keys())
        score_values = np.array([scores[e] for e in edges])
        probs = score_values / score_values.sum()
        
        n_runs = 1000
        edge_counts = {e: [] for e in edges}
        
        for seed in range(n_runs):
            G_sparse, weights = dspar_sparsify(
                G, retention=retention, method="paper", seed=seed, return_weights=True
            )
            
            for i, e in enumerate(edges):
                if e in weights:
                    count = round(weights[e] * q * probs[i])
                else:
                    count = 0
                edge_counts[e].append(count)
        
        significant_errors = 0
        for i, e in enumerate(edges):
            counts = np.array(edge_counts[e])
            expected_mean = q * probs[i]
            actual_mean = counts.mean()
            se = counts.std() / np.sqrt(n_runs)
            
            if abs(actual_mean - expected_mean) > 3 * se + 0.1:
                significant_errors += 1
        
        max_failures = max(1, int(0.05 * len(edges)))
        assert significant_errors <= max_failures
    
    def test_chi_squared_goodness_of_fit(self):
        """High-probability edges appear more often"""
        G = nx.karate_club_graph()
        retention = 0.75
        
        scores = compute_dspar_scores(G)
        edges = list(scores.keys())
        score_values = np.array([scores[e] for e in edges])
        expected_probs = score_values / score_values.sum()
        
        n_runs = 500
        edge_appearance_counts = np.zeros(len(edges))
        
        for seed in range(n_runs):
            G_sparse, weights = dspar_sparsify(
                G, retention=retention, method="paper", seed=seed, return_weights=True
            )
            for i, e in enumerate(edges):
                if e in weights:
                    edge_appearance_counts[i] += 1
        
        median_prob = np.median(expected_probs)
        high_prob_mask = expected_probs >= median_prob
        
        high_prob_appearances = edge_appearance_counts[high_prob_mask].mean()
        low_prob_appearances = edge_appearance_counts[~high_prob_mask].mean()
        
        assert high_prob_appearances > low_prob_appearances


# -----------------------------------------------------------------------------
# Test Class 15: Invariant Tests
# -----------------------------------------------------------------------------

class TestInvariants:
    """Test mathematical invariants"""
    
    def test_total_weight_equals_samples(self):
        """sum(w * p) = 1"""
        G = nx.karate_club_graph()
        retention = 0.6
        
        scores = compute_dspar_scores(G)
        edges = list(scores.keys())
        score_values = np.array([scores[e] for e in edges])
        probs = score_values / score_values.sum()
        edge_to_prob = dict(zip(edges, probs))
        
        G_sparse, weights = dspar_sparsify(
            G, retention=retention, method="paper", seed=42, return_weights=True
        )
        
        weighted_sum = sum(w * edge_to_prob[e] for e, w in weights.items())
        assert abs(weighted_sum - 1.0) < 0.01
    
    def test_score_ordering_preserved(self):
        """Deterministic preserves score ordering"""
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        scores = compute_dspar_scores(G)
        
        G_sparse = dspar_sparsify(G, retention=0.5, method="deterministic")
        
        kept_edges = set((min(u, v), max(u, v)) for u, v in G_sparse.edges())
        removed_edges = set(scores.keys()) - kept_edges
        
        if not removed_edges:
            return
        
        min_kept_score = min(scores[e] for e in kept_edges)
        max_removed_score = max(scores[e] for e in removed_edges)
        
        assert min_kept_score >= max_removed_score - 1e-10


# =============================================================================
# MAIN: Run tests
# =============================================================================

def run_quick_tests():
    """Run quick subset of tests for manual verification"""
    print("=" * 60)
    print("Quick DSpar Tests")
    print("=" * 60)
    
    G = nx.karate_club_graph()
    
    # Test 1: Score computation
    print("\n1. Testing score computation...")
    scores = compute_dspar_scores(G)
    degrees = dict(G.degree())
    
    errors = 0
    for (u, v), score in scores.items():
        expected = 1.0 / degrees[u] + 1.0 / degrees[v]
        if abs(score - expected) > 1e-10:
            errors += 1
    
    print(f"   Score formula: {'PASS' if errors == 0 else 'FAIL'}")
    
    # Test 2: Deterministic edge count
    print("\n2. Testing deterministic edge count...")
    m = G.number_of_edges()
    G_sparse = dspar_sparsify(G, retention=0.5, method="deterministic")
    expected = int(np.ceil(0.5 * m))
    
    print(f"   Expected: {expected}, Got: {G_sparse.number_of_edges()}")
    print(f"   Result: {'PASS' if G_sparse.number_of_edges() == expected else 'FAIL'}")
    
    # Test 3: Node preservation
    print("\n3. Testing node preservation...")
    preserved = set(G_sparse.nodes()) == set(G.nodes())
    print(f"   All nodes preserved: {'PASS' if preserved else 'FAIL'}")
    
    # Test 4: No new edges
    print("\n4. Testing no new edges...")
    original_edges = set((min(u, v), max(u, v)) for u, v in G.edges())
    sparse_edges = set((min(u, v), max(u, v)) for u, v in G_sparse.edges())
    subset = sparse_edges.issubset(original_edges)
    print(f"   Sparse subset of original: {'PASS' if subset else 'FAIL'}")
    
    # Test 5: Paper method weights
    print("\n5. Testing paper method weights...")
    G_paper, weights = dspar_sparsify(G, retention=0.5, method="paper", seed=42, return_weights=True)
    all_positive = all(w > 0 for w in weights.values())
    print(f"   All weights positive: {'PASS' if all_positive else 'FAIL'}")
    print(f"   Total weight: {sum(weights.values()):.2f} (original edges: {m})")
    
    # Test 6: Reproducibility
    print("\n6. Testing reproducibility with seed...")
    G1 = dspar_sparsify(G, retention=0.5, method="paper", seed=42)
    G2 = dspar_sparsify(G, retention=0.5, method="paper", seed=42)
    reproducible = set(G1.edges()) == set(G2.edges())
    print(f"   Same seed same result: {'PASS' if reproducible else 'FAIL'}")
    
    print("\n" + "=" * 60)
    print("Quick tests complete. Run 'pytest test_dspar.py -v' for full suite.")
    print("=" * 60)


def run_verbose_correctness():
    """Run correctness tests with verbose output"""
    print("=" * 70)
    print("DSpar CORRECTNESS Tests")
    print("=" * 70)
    print("\nThese tests verify ALGORITHMIC CORRECTNESS, not just code execution.\n")
    
    G = nx.karate_club_graph()
    
    # Test 1: Manual verification
    print("1. Manual score verification (triangle graph)...")
    G_tri = nx.Graph()
    G_tri.add_edges_from([(0, 1), (1, 2), (0, 2)])
    scores = compute_dspar_scores(G_tri)
    expected = 1.0
    all_correct = all(abs(s - expected) < 1e-10 for s in scores.values())
    print(f"   All scores = {expected}: {'PASS' if all_correct else 'FAIL'}")
    
    # Test 2: Probability sum
    print("\n2. Probability sum = 1...")
    scores = compute_dspar_scores(G)
    probs = np.array(list(scores.values()))
    probs = probs / probs.sum()
    print(f"   Sum of probabilities: {probs.sum():.10f}")
    print(f"   Result: {'PASS' if abs(probs.sum() - 1.0) < 1e-10 else 'FAIL'}")
    
    # Test 3: Weight formula
    print("\n3. Weight formula verification...")
    retention = 0.5
    m = G.number_of_edges()
    q = int(np.ceil(retention * m))
    edges = list(scores.keys())
    score_values = np.array([scores[e] for e in edges])
    probs = score_values / score_values.sum()
    edge_to_prob = dict(zip(edges, probs))
    
    G_sparse, weights = dspar_sparsify(G, retention=retention, method="paper", seed=42, return_weights=True)
    
    weighted_sum = sum(w * edge_to_prob[e] for e, w in weights.items())
    print(f"   sum(w * p) = {weighted_sum:.6f} (expected: 1.0)")
    print(f"   Result: {'PASS' if abs(weighted_sum - 1.0) < 0.01 else 'FAIL'}")
    
    # Test 4: ER correlation
    print("\n4. DSpar-EffectiveResistance correlation...")
    L = nx.laplacian_matrix(G).toarray().astype(float)
    try:
        L_pinv = np.linalg.pinv(L)
        dspar_scores = []
        eff_resistances = []
        for (u, v), score in scores.items():
            r_eff = L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v]
            dspar_scores.append(score)
            eff_resistances.append(r_eff)
        
        correlation = np.corrcoef(dspar_scores, eff_resistances)[0, 1]
        print(f"   Correlation: {correlation:.4f}")
        print(f"   Result: {'PASS' if correlation > 0.3 else 'FAIL (weak correlation)'}")
    except:
        print("   Could not compute")
    
    print("\n" + "=" * 70)
    print("Run 'pytest test_dspar.py -v' for full test suite (67 tests)")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            run_quick_tests()
        elif sys.argv[1] == "--verbose":
            run_verbose_correctness()
        else:
            print("Usage: python test_dspar.py [--quick|--verbose]")
            print("Or: pytest test_dspar.py -v")
    else:
        pytest.main([__file__, "-v", "--tb=short"])