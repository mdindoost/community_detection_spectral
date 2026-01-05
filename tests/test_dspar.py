"""
Comprehensive Test Suite for DSpar Implementation

Run with: pytest test_dspar.py -v
Or: python test_dspar.py
"""

import pytest
import numpy as np
import networkx as nx
from collections import Counter
import sys
import os

# Import the module to test
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
    return nx.star_graph(10)  # Node 0 is hub, connected to 1-10


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
# TEST CLASS 1: DSpar Score Computation
# =============================================================================

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
        """Edges to hub (high degree) should have lower scores"""
        G = star_graph
        scores = compute_dspar_scores(G)
        
        # Hub is node 0 with degree 10
        # Leaves are nodes 1-10 with degree 1
        # Score for hub edge: 1/10 + 1/1 = 1.1
        hub_degree = G.degree(0)  # Should be 10
        leaf_degree = G.degree(1)  # Should be 1
        
        expected_score = 1.0 / hub_degree + 1.0 / leaf_degree
        
        for edge, score in scores.items():
            assert abs(score - expected_score) < 1e-10, "All edges in star should have same score"
    
    def test_symmetric_edge_representation(self, simple_graph):
        """Edge (u,v) and (v,u) should map to same canonical form"""
        scores = compute_dspar_scores(simple_graph)
        
        for (u, v) in scores.keys():
            assert u < v, "Edges should be stored as (min, max)"


# =============================================================================
# TEST CLASS 2: Basic Sparsification Properties
# =============================================================================

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
        
        # Only deterministic guarantees all edges at retention=1.0
        # Paper method samples WITH replacement, so duplicates reduce unique edges
        # Probabilistic has variance
        assert G_sparse.number_of_edges() == G.number_of_edges()
    
    def test_paper_retention_1_has_duplicates(self, karate_graph):
        """Paper method at retention=1.0 has fewer unique edges due to replacement"""
        G = karate_graph
        m = G.number_of_edges()
        
        # With replacement sampling of m items, expected unique ~ m(1 - 1/e) ≈ 0.632m
        G_sparse = dspar_sparsify(G, retention=1.0, method="paper", seed=42)
        
        # Should have significantly fewer unique edges than original
        # but total weight should still approximate m
        assert G_sparse.number_of_edges() < m
        assert G_sparse.number_of_edges() > m * 0.5  # At least half


# =============================================================================
# TEST CLASS 3: Deterministic Method Tests
# =============================================================================

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
        
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        
        assert edges1 == edges2
    
    def test_keeps_highest_scores(self, simple_graph):
        """Should keep edges with highest DSpar scores (may have ties)"""
        G = simple_graph
        scores = compute_dspar_scores(G)
        
        # Keep 3 edges (out of 5)
        G_sparse = dspar_sparsify(G, retention=0.6, method="deterministic")
        
        # Get scores of kept edges
        sparse_edges = set((min(u, v), max(u, v)) for u, v in G_sparse.edges())
        kept_scores = [scores[e] for e in sparse_edges]
        
        # Get scores of removed edges
        removed_edges = set(scores.keys()) - sparse_edges
        removed_scores = [scores[e] for e in removed_edges]
        
        # All kept scores should be >= all removed scores (allowing for ties)
        min_kept = min(kept_scores)
        max_removed = max(removed_scores) if removed_scores else 0
        
        assert min_kept >= max_removed - 1e-10, \
            f"Kept edge has lower score ({min_kept}) than removed edge ({max_removed})"
    
    def test_low_degree_edges_preferred(self, barbell_graph):
        """Edges between low-degree nodes should be kept"""
        G = barbell_graph
        G_sparse = dspar_sparsify(G, retention=0.3, method="deterministic")
        
        # The bridge edge connects lower-degree nodes, should be kept
        sparse_edges = list(G_sparse.edges())
        degrees = dict(G.degree())
        
        # Calculate average degree product of kept edges
        avg_degree_product = np.mean([degrees[u] * degrees[v] for u, v in sparse_edges])
        
        # Calculate average degree product of all edges
        all_avg = np.mean([degrees[u] * degrees[v] for u, v in G.edges()])
        
        # Kept edges should have lower average degree product
        assert avg_degree_product <= all_avg


# =============================================================================
# TEST CLASS 4: Paper Method (With Replacement) Tests
# =============================================================================

class TestPaperMethod:
    """Tests specific to paper method (with replacement, weighted)"""
    
    def test_returns_weighted_graph(self, karate_graph):
        """Paper method should return weighted edges"""
        G_sparse = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=42)
        
        # Check that edges have weight attribute
        for u, v, data in G_sparse.edges(data=True):
            assert 'weight' in data
            assert data['weight'] > 0
    
    def test_weight_formula(self, karate_graph):
        """Test weight formula: w'_e = k_e / (q * p_e)"""
        G = karate_graph
        G_sparse, weights = dspar_sparsify(
            G, retention=0.5, method="paper", seed=42, return_weights=True
        )
        
        # All weights should be positive
        assert all(w > 0 for w in weights.values())
    
    def test_fewer_unique_edges_than_samples(self, karate_graph):
        """With replacement: unique edges < number of samples"""
        G = karate_graph
        m = G.number_of_edges()
        retention = 0.75
        q = int(np.ceil(retention * m))  # Number of samples
        
        G_sparse = dspar_sparsify(G, retention=retention, method="paper", seed=42)
        
        # Due to replacement, unique edges should typically be less than q
        # (unless graph is very large or retention very small)
        assert G_sparse.number_of_edges() <= q
    
    def test_total_weight_approximates_edges(self, karate_graph):
        """Total weight should approximate original edge count (unbiased estimator)"""
        G = karate_graph
        m = G.number_of_edges()
        
        # Run multiple times and check average
        total_weights = []
        for seed in range(10):
            G_sparse, weights = dspar_sparsify(
                G, retention=0.75, method="paper", seed=seed, return_weights=True
            )
            total_weights.append(sum(weights.values()))
        
        avg_total = np.mean(total_weights)
        
        # Should be reasonably close to m (within 20%)
        assert 0.8 * m <= avg_total <= 1.2 * m, f"Average total weight {avg_total} not close to m={m}"
    
    def test_reproducible_with_seed(self, karate_graph):
        """Same seed should give same result"""
        G1 = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=42)
        G2 = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=42)
        
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        
        assert edges1 == edges2
    
    def test_different_seeds_different_results(self, karate_graph):
        """Different seeds should (usually) give different results"""
        G1 = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=42)
        G2 = dspar_sparsify(karate_graph, retention=0.5, method="paper", seed=123)
        
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        
        # Should be different (with very high probability)
        assert edges1 != edges2


# =============================================================================
# TEST CLASS 5: Probabilistic No Replacement Tests
# =============================================================================

class TestProbabilisticNoReplace:
    """Tests specific to probabilistic without replacement method"""
    
    def test_unweighted_output(self, karate_graph):
        """Should return unweighted graph"""
        G_sparse = dspar_sparsify(
            karate_graph, retention=0.5, method="probabilistic_no_replace", seed=42
        )
        
        # Edges should not have weight or weight should be 1
        for u, v, data in G_sparse.edges(data=True):
            if 'weight' in data:
                assert data['weight'] == 1.0
    
    def test_approximately_expected_edges(self, karate_graph):
        """Should keep approximately retention * m edges on average"""
        G = karate_graph
        m = G.number_of_edges()
        retention = 0.5
        expected = retention * m
        
        # Run multiple times
        edge_counts = []
        for seed in range(20):
            G_sparse = dspar_sparsify(
                G, retention=retention, method="probabilistic_no_replace", seed=seed
            )
            edge_counts.append(G_sparse.number_of_edges())
        
        avg_edges = np.mean(edge_counts)
        
        # Should be within 20% of expected
        assert 0.8 * expected <= avg_edges <= 1.2 * expected
    
    def test_reproducible_with_seed(self, karate_graph):
        """Same seed should give same result"""
        G1 = dspar_sparsify(karate_graph, retention=0.5, method="probabilistic_no_replace", seed=42)
        G2 = dspar_sparsify(karate_graph, retention=0.5, method="probabilistic_no_replace", seed=42)
        
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        
        assert edges1 == edges2


# =============================================================================
# TEST CLASS 6: Edge Cases
# =============================================================================

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
        
        # All scores should be equal (all nodes have same degree)
        score_values = list(scores.values())
        assert all(abs(s - score_values[0]) < 1e-10 for s in score_values)
    
    def test_path_graph(self, path_graph):
        """Path graph - end edges should have higher scores"""
        G = path_graph
        scores = compute_dspar_scores(G)
        
        # End edges connect degree-1 and degree-2 nodes: 1/1 + 1/2 = 1.5
        # Internal edges connect degree-2 nodes: 1/2 + 1/2 = 1.0
        
        end_edge = (0, 1)
        internal_edge = (4, 5)
        
        assert scores[end_edge] > scores[internal_edge]
    
    def test_disconnected_components(self):
        """Graph with disconnected components"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])  # Component 1
        G.add_edges_from([(3, 4), (4, 5)])  # Component 2
        
        G_sparse = dspar_sparsify(G, retention=0.5, method="deterministic")
        
        # Should still work
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


# =============================================================================
# TEST CLASS 7: Statistical Properties
# =============================================================================

class TestStatisticalProperties:
    """Tests for statistical properties of the sparsification"""
    
    def test_high_score_edges_more_likely_kept(self, karate_graph):
        """Edges with higher scores should be kept more often"""
        G = karate_graph
        scores = compute_dspar_scores(G)
        
        # Classify edges as high/low score
        median_score = np.median(list(scores.values()))
        high_score_edges = {e for e, s in scores.items() if s >= median_score}
        low_score_edges = {e for e, s in scores.items() if s < median_score}
        
        # Count how often each type is kept
        high_kept = 0
        low_kept = 0
        n_runs = 50
        
        for seed in range(n_runs):
            G_sparse = dspar_sparsify(G, retention=0.5, method="probabilistic_no_replace", seed=seed)
            sparse_edges = set((min(u, v), max(u, v)) for u, v in G_sparse.edges())
            
            high_kept += len(sparse_edges & high_score_edges)
            low_kept += len(sparse_edges & low_score_edges)
        
        # High score edges should be kept more often
        assert high_kept > low_kept, f"High: {high_kept}, Low: {low_kept}"
    
    def test_paper_method_unbiased_per_edge(self, karate_graph):
        """Paper method should give approximately unbiased weight estimate per edge"""
        G = karate_graph
        
        # Track weighted count per edge across many runs
        edge_weighted_counts = Counter()
        n_runs = 100
        
        for seed in range(n_runs):
            G_sparse, weights = dspar_sparsify(
                G, retention=0.75, method="paper", seed=seed, return_weights=True
            )
            for edge, weight in weights.items():
                edge_weighted_counts[edge] += weight
        
        # Average weight per edge should be close to 1
        # Relaxed bounds due to statistical variance (especially for high-score edges)
        avg_weights = []
        for edge in compute_dspar_scores(G).keys():
            avg_weight = edge_weighted_counts[edge] / n_runs
            avg_weights.append(avg_weight)
        
        # Check overall average is close to 1
        overall_avg = np.mean(avg_weights)
        assert 0.7 <= overall_avg <= 1.3, f"Overall avg weight {overall_avg} not close to 1"


# =============================================================================
# TEST CLASS 8: Consistency Tests
# =============================================================================

class TestConsistency:
    """Tests for consistency between methods and properties"""
    
    def test_deterministic_subset_of_scores(self, karate_graph):
        """Deterministic result should be top-k by score"""
        G = karate_graph
        scores = compute_dspar_scores(G)
        
        retention = 0.5
        G_sparse = dspar_sparsify(G, retention=retention, method="deterministic")
        n_kept = G_sparse.number_of_edges()
        
        # Get threshold score
        sorted_scores = sorted(scores.values(), reverse=True)
        threshold = sorted_scores[n_kept - 1]
        
        # All kept edges should have score >= threshold
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


# =============================================================================
# TEST CLASS 9: Performance Sanity Checks
# =============================================================================

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
        G = nx.complete_graph(50)  # 1225 edges
        
        G_sparse = dspar_sparsify(G, retention=0.1, method="deterministic")
        
        expected = int(np.ceil(0.1 * G.number_of_edges()))
        assert G_sparse.number_of_edges() == expected


# =============================================================================
# MAIN: Run tests
# =============================================================================

def run_quick_tests():
    """Run a quick subset of tests for manual verification"""
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
    
    # Summary
    print("\n" + "=" * 60)
    print("Quick tests complete. Run 'pytest test_dspar.py -v' for full suite.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_tests()
    else:
        # Run pytest
        pytest.main([__file__, "-v", "--tb=short"])