import random
import time

import numpy as np
import pytest

from clay.geometry import point_to_segment_distance
from clay.graph import Graph, Node
from clay.penalties.chain_collinearity import ChainCollinearity


# =============================================================================
# Reference Implementation (non-vectorized)
# =============================================================================

def chain_collinearity_reference(g: Graph, centers: np.ndarray) -> float:
    """Reference implementation of ChainCollinearity penalty (non-vectorized)."""
    total_penalty = 0.0

    for b_name in g.name2idx:
        b_idx = g.name2idx[b_name]
        predecessors = g.incoming[b_name]
        successors = g.outgoing[b_name]

        for a_name in predecessors:
            for c_name in successors:
                if a_name == c_name:
                    continue

                a_idx = g.name2idx[a_name]
                c_idx = g.name2idx[c_name]

                ax, ay = centers[2 * a_idx], centers[2 * a_idx + 1]
                bx, by = centers[2 * b_idx], centers[2 * b_idx + 1]
                cx, cy = centers[2 * c_idx], centers[2 * c_idx + 1]

                # Distance from B to segment AC
                dist = point_to_segment_distance(ax, ay, cx, cy, bx, by)
                total_penalty += 0.5 * dist ** 2

    return total_penalty


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestChainCollinearityEdgeCases:
    """Edge case tests for ChainCollinearity penalty."""

    def test_no_edges_zero_penalty(self):
        """Graph with nodes but no edges should have zero penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10), Node("c", 10, 10)],
            edges=[]
        )
        penalty = ChainCollinearity(g)
        centers = np.array([0.0, 0.0, 50.0, 50.0, 100.0, 100.0])
        assert penalty.compute(centers) == 0.0

    def test_single_edge_zero_penalty(self):
        """Single edge means no chains, so zero penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[("a", "b")]
        )
        penalty = ChainCollinearity(g)
        centers = np.array([0.0, 0.0, 100.0, 100.0])
        assert penalty.compute(centers) == 0.0

    def test_star_graph_zero_penalty(self):
        """Star graph (hub with spokes) has no chains."""
        g = Graph(
            nodes=[
                Node("hub", 10, 10),
                Node("a", 10, 10), Node("b", 10, 10), Node("c", 10, 10)
            ],
            edges=[("hub", "a"), ("hub", "b"), ("hub", "c")]
        )
        penalty = ChainCollinearity(g)
        centers = np.array([50.0, 50.0, 0.0, 0.0, 100.0, 0.0, 50.0, 100.0])
        assert penalty.compute(centers) == 0.0

    def test_single_chain_collinear_zero_penalty(self):
        """Single chain A->B->C with B on line AC should have zero penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10), Node("c", 10, 10)],
            edges=[("a", "b"), ("b", "c")]
        )
        penalty = ChainCollinearity(g)
        # A at (0,0), B at (50,50), C at (100,100) - all collinear
        centers = np.array([0.0, 0.0, 50.0, 50.0, 100.0, 100.0])
        assert abs(penalty.compute(centers)) < 1e-10

    def test_single_chain_non_collinear_positive_penalty(self):
        """Single chain A->B->C with B off line AC should have positive penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10), Node("c", 10, 10)],
            edges=[("a", "b"), ("b", "c")]
        )
        penalty = ChainCollinearity(g)
        # A at (0,0), B at (50,100), C at (100,0) - B is above line AC
        centers = np.array([0.0, 0.0, 50.0, 100.0, 100.0, 0.0])
        assert penalty.compute(centers) > 0.0

    def test_chain_with_same_start_end_skipped(self):
        """Chain where A==C should be skipped (avoid self-reference)."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[("a", "b"), ("b", "a")]  # Bidirectional
        )
        penalty = ChainCollinearity(g)
        # This creates potential chains A->B->A and B->A->B, both should be skipped
        assert penalty.n_chains == 0
        centers = np.array([0.0, 0.0, 100.0, 100.0])
        assert penalty.compute(centers) == 0.0

    def test_multiple_chains_sum_penalties(self):
        """Multiple chains should sum their penalties."""
        g = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10),
                Node("c", 10, 10), Node("d", 10, 10)
            ],
            edges=[("a", "b"), ("b", "c"), ("c", "d")]  # Linear chain
        )
        penalty = ChainCollinearity(g)
        # Should have 2 chains: A->B->C and B->C->D
        assert penalty.n_chains == 2

        # All collinear
        centers_collinear = np.array([0.0, 0.0, 33.0, 33.0, 66.0, 66.0, 100.0, 100.0])
        assert abs(penalty.compute(centers_collinear)) < 1e-6

        # B off line (creates non-zero penalty for A->B->C)
        centers_bent = np.array([0.0, 0.0, 33.0, 50.0, 66.0, 66.0, 100.0, 100.0])
        assert penalty.compute(centers_bent) > 0.0

    def test_more_deviation_higher_penalty(self):
        """Greater deviation from collinearity should give higher penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10), Node("c", 10, 10)],
            edges=[("a", "b"), ("b", "c")]
        )
        penalty = ChainCollinearity(g)

        # A at (0,0), C at (100,0), B at varying heights
        centers_small_dev = np.array([0.0, 0.0, 50.0, 10.0, 100.0, 0.0])
        centers_large_dev = np.array([0.0, 0.0, 50.0, 50.0, 100.0, 0.0])

        penalty_small = penalty.compute(centers_small_dev)
        penalty_large = penalty.compute(centers_large_dev)

        assert penalty_large > penalty_small


# =============================================================================
# Reference Comparison Tests
# =============================================================================

class TestChainCollinearityReferenceComparison:
    """Compare vectorized implementation against reference."""

    @pytest.fixture
    def random_graphs(self):
        """Generate random DAGs with varying sizes."""
        graphs = []
        random.seed(42)

        for n_nodes in [5, 8, 12, 15, 20]:
            nodes = [Node(f"n{i}", 10, 10) for i in range(n_nodes)]

            # Create random DAG edges (only i -> j where i < j)
            edge_density = random.uniform(0.2, 0.4)
            edges = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if random.random() < edge_density:
                        edges.append((f"n{i}", f"n{j}"))

            if len(edges) < 2:
                edges = [(f"n0", f"n1"), (f"n1", f"n2")]

            g = Graph(nodes=nodes, edges=edges)
            centers = np.array([random.uniform(0, 200) for _ in range(n_nodes * 2)])
            graphs.append((g, centers))

        return graphs

    def test_matches_reference_implementation(self, random_graphs):
        """Vectorized implementation should match reference for random graphs."""
        for g, centers in random_graphs:
            penalty = ChainCollinearity(g)
            vectorized_result = penalty.compute(centers)
            reference_result = chain_collinearity_reference(g, centers)

            if abs(reference_result) > 1e-10:
                rel_diff = abs(vectorized_result - reference_result) / abs(reference_result)
                assert rel_diff < 1e-9, \
                    f"Mismatch for graph with {len(g.nodes)} nodes, {len(g.edges)} edges: " \
                    f"{vectorized_result} vs {reference_result} (rel_diff={rel_diff})"
            else:
                assert abs(vectorized_result - reference_result) < 1e-10

    def test_performance_improvement(self, random_graphs):
        """Vectorized implementation should be faster than reference."""
        random.seed(123)
        large_graphs = []

        for n_nodes in [30, 50, 80]:
            nodes = [Node(f"n{i}", 10, 10) for i in range(n_nodes)]
            # Create DAG with decent chain count
            edges = [(f"n{i}", f"n{j}")
                     for i in range(n_nodes) for j in range(i + 1, n_nodes)
                     if random.random() < 0.15]
            g = Graph(nodes=nodes, edges=edges)
            centers = np.array([random.uniform(0, 300) for _ in range(n_nodes * 2)])
            large_graphs.append((g, centers))

        n_iterations = 30

        for g, centers in large_graphs:
            penalty = ChainCollinearity(g)

            # Time reference
            start = time.perf_counter()
            for _ in range(n_iterations):
                chain_collinearity_reference(g, centers)
            ref_time = time.perf_counter() - start

            # Time vectorized
            start = time.perf_counter()
            for _ in range(n_iterations):
                penalty.compute(centers)
            vec_time = time.perf_counter() - start

            speedup = ref_time / vec_time
            print(f"\nNodes: {len(g.nodes):3d}, Edges: {len(g.edges):4d}, "
                  f"Chains: {penalty.n_chains:4d} | "
                  f"Ref: {ref_time:.4f}s, Vec: {vec_time:.4f}s, Speedup: {speedup:.2f}x")

            assert speedup > 1.5, "Vectorized implementation should be faster than reference"

            # Verify correctness
            vec_result = penalty.compute(centers)
            ref_result = chain_collinearity_reference(g, centers)
            if abs(ref_result) > 1e-10:
                rel_diff = abs(vec_result - ref_result) / abs(ref_result)
                assert rel_diff < 1e-9
            else:
                assert abs(vec_result - ref_result) < 1e-10
