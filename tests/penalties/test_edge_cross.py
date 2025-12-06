import random
import time

import numpy as np
import pytest

from clay.geometry import point_to_segment_distance
from clay.graph import Graph, Node, Edge
from clay.penalties.edge_cross import EgdeCross, segment_cross


# =============================================================================
# Reference Implementation (non-vectorized)
# =============================================================================

def _softmin_ref(values: list[float], sharpness: float = 1.0) -> float:
    """Reference softmin implementation."""
    values = np.asarray(values)
    scale = values.max() or 1.0
    v = -sharpness * values / scale
    v_max = v.max()
    return scale * (-(v_max + np.log(np.exp(v - v_max).sum())) / sharpness)


def _crossing_penalty_ref(
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
    dx: float, dy: float
) -> float:
    """Reference crossing penalty implementation (matches original exactly)."""
    if not segment_cross(ax, ay, bx, by, cx, cy, dx, dy):
        return 0.0

    # Match original argument order exactly
    d1 = point_to_segment_distance(ax, ay, cx, cy, dx, dy)
    d2 = point_to_segment_distance(bx, by, cx, cy, dx, dy)
    d3 = point_to_segment_distance(cx, cy, ax, ay, bx, by)
    d4 = point_to_segment_distance(dx, dy, ax, ay, bx, by)

    min_dist = _softmin_ref([d1, d2, d3, d4])
    return 0.5 * min_dist ** 2


def edge_cross_reference(g: Graph, centers: np.ndarray) -> float:
    """Reference implementation of EdgeCross penalty (non-vectorized)."""
    n_edges = len(g.edges)
    total_penalty = 0.0

    for i in range(n_edges):
        for j in range(i + 1, n_edges):
            (a_name, b_name) = g.edges[i]
            (c_name, d_name) = g.edges[j]

            a_idx = g.name2idx[a_name]
            b_idx = g.name2idx[b_name]
            c_idx = g.name2idx[c_name]
            d_idx = g.name2idx[d_name]

            ax, ay = centers[2 * a_idx], centers[2 * a_idx + 1]
            bx, by = centers[2 * b_idx], centers[2 * b_idx + 1]
            cx, cy = centers[2 * c_idx], centers[2 * c_idx + 1]
            dx, dy = centers[2 * d_idx], centers[2 * d_idx + 1]

            penetration = _crossing_penalty_ref(ax, ay, bx, by, cx, cy, dx, dy)
            total_penalty += 0.5 * penetration ** 2

    return total_penalty


# =============================================================================
# Helper Function Tests (existing)
# =============================================================================

class TestSegmentCross:
    def test_no_cross(self):
        assert not segment_cross(0, 0, 4, 0, 0, 1, 4, 1)

    def test_cross(self):
        assert segment_cross(0, 0, 4, 4, 0, 4, 4, 0)

    def test_touching_at_endpoint(self):
        assert not segment_cross(0, 0, 4, 4, 4, 4, 8, 8)

    def test_collinear_non_overlapping(self):
        assert not segment_cross(0, 0, 4, 4, 5, 5, 6, 6)

    def test_collinear_overlapping(self):
        assert not segment_cross(0, 0, 4, 4, 2, 2, 6, 6)


# =============================================================================
# EdgeCross.compute Edge Case Tests
# =============================================================================

class TestEdgeCrossEdgeCases:
    """Edge case tests for EdgeCross penalty."""

    def test_no_edges_zero_penalty(self):
        """Graph with nodes but no edges should have zero penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[]
        )
        penalty = EgdeCross(g)
        centers = np.array([0.0, 0.0, 100.0, 100.0])
        assert penalty.compute(centers) == 0.0

    def test_single_edge_zero_penalty(self):
        """Single edge means no pairs, so zero penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[Edge("a", "b")]
        )
        penalty = EgdeCross(g)
        centers = np.array([0.0, 0.0, 100.0, 100.0])
        assert penalty.compute(centers) == 0.0

    def test_parallel_edges_zero_penalty(self):
        """Two parallel horizontal edges should not cross."""
        g = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10),
                Node("c", 10, 10), Node("d", 10, 10)
            ],
            edges=[Edge("a", "b"), Edge("c", "d")]
        )
        penalty = EgdeCross(g)
        # a--b on y=0, c--d on y=50 (parallel)
        centers = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 50.0, 100.0, 50.0])
        assert penalty.compute(centers) == 0.0

    def test_crossing_edges_positive_penalty(self):
        """X-pattern crossing edges should have positive penalty."""
        g = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10),
                Node("c", 10, 10), Node("d", 10, 10)
            ],
            edges=[Edge("a", "b"), Edge("c", "d")]
        )
        penalty = EgdeCross(g)
        # a(0,0)--b(100,100) crosses c(0,100)--d(100,0)
        centers = np.array([0.0, 0.0, 100.0, 100.0, 0.0, 100.0, 100.0, 0.0])
        result = penalty.compute(centers)
        assert result > 0.0

    def test_shared_endpoint_zero_penalty(self):
        """Edges sharing an endpoint should not count as crossing."""
        g = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10), Node("c", 10, 10)
            ],
            edges=[Edge("a", "b"), Edge("a", "c")]
        )
        penalty = EgdeCross(g)
        # a at origin, b and c spread out
        centers = np.array([50.0, 50.0, 0.0, 0.0, 100.0, 0.0])
        assert penalty.compute(centers) == 0.0

    def test_mixed_one_crossing_pair(self):
        """Three edges where only one pair crosses."""
        g = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10),
                Node("c", 10, 10), Node("d", 10, 10),
                Node("e", 10, 10), Node("f", 10, 10)
            ],
            edges=[Edge("a", "b"), Edge("c", "d"), Edge("e", "f")]
        )
        penalty = EgdeCross(g)
        # a-b and c-d cross (X pattern), e-f is far away (parallel)
        centers = np.array([
            0.0, 0.0,      # a
            100.0, 100.0,  # b
            0.0, 100.0,    # c
            100.0, 0.0,    # d
            200.0, 0.0,    # e
            300.0, 0.0     # f (parallel, far away)
        ])
        result = penalty.compute(centers)
        assert result > 0.0

    def test_more_crossings_higher_penalty(self):
        """More crossing pairs should result in higher penalty."""
        # Graph with 1 crossing
        g1 = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10),
                Node("c", 10, 10), Node("d", 10, 10)
            ],
            edges=[Edge("a", "b"), Edge("c", "d")]
        )
        # X pattern
        centers1 = np.array([0.0, 0.0, 100.0, 100.0, 0.0, 100.0, 100.0, 0.0])
        penalty1 = EgdeCross(g1).compute(centers1)

        # Graph with 2 crossings (add another crossing pair)
        g2 = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10),
                Node("c", 10, 10), Node("d", 10, 10),
                Node("e", 10, 10), Node("f", 10, 10)
            ],
            edges=[Edge("a", "b"), Edge("c", "d"), Edge("e", "f")]
        )
        # Two X patterns
        centers2 = np.array([
            0.0, 0.0,      # a
            100.0, 100.0,  # b
            0.0, 100.0,    # c
            100.0, 0.0,    # d
            50.0, 0.0,     # e
            50.0, 100.0    # f (vertical, crosses both diagonals)
        ])
        penalty2 = EgdeCross(g2).compute(centers2)

        assert penalty2 > penalty1


# =============================================================================
# Reference Comparison Tests
# =============================================================================

class TestEdgeCrossReferenceComparison:
    """Compare vectorized implementation against reference."""

    @pytest.fixture
    def random_graphs(self):
        """Generate random graphs with varying edge counts."""
        graphs = []
        random.seed(42)

        for n_nodes in [4, 6, 8, 10, 12, 15, 18]:
            nodes = [Node(f"n{i}", 10, 10) for i in range(n_nodes)]

            # Random edges with decent density to get some crossings
            edge_density = random.uniform(0.2, 0.4)
            edges = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if random.random() < edge_density:
                        edges.append(Edge(f"n{i}", f"n{j}"))

            if len(edges) < 2:
                # Ensure at least 2 edges
                edges = [Edge("n0", "n1"), Edge("n2", "n3")]

            g = Graph(nodes=nodes, edges=edges)

            # Random positions (spread out to get some crossings)
            centers = np.array([random.uniform(0, 200) for _ in range(n_nodes * 2)])

            graphs.append((g, centers))

        return graphs

    def test_matches_reference_implementation(self, random_graphs):
        """Vectorized implementation should match reference for random graphs."""
        for g, centers in random_graphs:
            penalty = EgdeCross(g)
            vectorized_result = penalty.compute(centers)
            reference_result = edge_cross_reference(g, centers)

            # Use relative tolerance for large values
            rel_diff = abs(vectorized_result - reference_result) / max(abs(reference_result), 1e-10)
            assert rel_diff < 1e-9, \
                f"Mismatch for graph with {len(g.nodes)} nodes, {len(g.edges)} edges: " \
                f"{vectorized_result} vs {reference_result} (rel_diff={rel_diff})"

    def test_performance_improvement(self, random_graphs):
        """Vectorized implementation should be faster than reference."""
        random.seed(123)
        large_graphs = []

        for n_nodes in [20, 35, 50]:
            nodes = [Node(f"n{i}", 10, 10) for i in range(n_nodes)]
            # Higher edge density for more pairs
            edges = [Edge(f"n{i}", f"n{j}")
                     for i in range(n_nodes) for j in range(i + 1, n_nodes)
                     if random.random() < 0.25]
            g = Graph(nodes=nodes, edges=edges)
            centers = np.array([random.uniform(0, 300) for _ in range(n_nodes * 2)])
            large_graphs.append((g, centers))

        n_iterations = 30

        for g, centers in large_graphs:
            penalty = EgdeCross(g)

            # Time reference
            start = time.perf_counter()
            for _ in range(n_iterations):
                edge_cross_reference(g, centers)
            ref_time = time.perf_counter() - start

            # Time vectorized
            start = time.perf_counter()
            for _ in range(n_iterations):
                penalty.compute(centers)
            vec_time = time.perf_counter() - start

            speedup = ref_time / vec_time
            print(f"\nNodes: {len(g.nodes):3d}, Edges: {len(g.edges):4d} | "
                  f"Ref: {ref_time:.4f}s, Vec: {vec_time:.4f}s, Speedup: {speedup:.2f}x")

            assert speedup > 1.5, "Vectorized implementation should be faster than reference"

            # Verify correctness (relative tolerance)
            vec_result = penalty.compute(centers)
            ref_result = edge_cross_reference(g, centers)
            rel_diff = abs(vec_result - ref_result) / max(abs(ref_result), 1e-10)
            assert rel_diff < 1e-9
