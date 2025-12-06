import random
import time

import numpy as np
import pytest

from clay.geometry import point_to_segment_distance
from clay.graph import Graph, Node, Edge
from clay.penalties.node_edge import (
    NodeEdge,
    segment_intersects_rect,
    segment_rectangle_penetration,
)


# =============================================================================
# Reference Implementation (non-vectorized)
# =============================================================================

def _segment_intersects_rect_ref(
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
    w: float, h: float
) -> bool:
    """Reference implementation of segment-rectangle intersection."""
    hw, hh = w / 2, h / 2

    left, right = cx - hw, cx + hw
    bottom, top = cy - hh, cy + hh

    def inside(px, py):
        return left <= px <= right and bottom <= py <= top

    if inside(ax, ay) or inside(bx, by):
        return True

    dx, dy = bx - ax, by - ay
    t_min, t_max = 0.0, 1.0

    for p, q in [
        (-dx, ax - left),
        (dx, right - ax),
        (-dy, ay - bottom),
        (dy, top - ay),
    ]:
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t_min = max(t_min, t)
            else:
                t_max = min(t_max, t)

        if t_min > t_max:
            return False

    return True


def _segment_rectangle_penetration_ref(
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
    w: float, h: float
) -> float:
    """Reference implementation of segment-rectangle penetration."""
    if not _segment_intersects_rect_ref(ax, ay, bx, by, cx, cy, w, h):
        return 0.0

    d = point_to_segment_distance(ax, ay, bx, by, cx, cy)
    half_diag = float((w**2 + h**2) ** 0.5 / 2)

    return half_diag - d


def node_edge_reference(g: Graph, centers: np.ndarray) -> float:
    """Reference implementation of NodeEdge penalty (non-vectorized)."""
    total_penalty = 0.0
    for i, node in enumerate(g.nodes):
        for e in g.edges:
            if node.name in e:
                continue  # Skip edges connected to this node

            cx, cy = centers[2 * i], centers[2 * i + 1]
            w, h = node.width, node.height

            src_name, dst_name = e
            src_idx = g.name2idx[src_name]
            dst_idx = g.name2idx[dst_name]
            ax, ay = centers[2 * src_idx], centers[2 * src_idx + 1]
            bx, by = centers[2 * dst_idx], centers[2 * dst_idx + 1]

            penetration = _segment_rectangle_penetration_ref(
                ax, ay, bx, by, cx, cy, w, h
            )
            total_penalty += 0.5 * penetration**2

    return total_penalty


# =============================================================================
# Helper Function Tests (existing)
# =============================================================================

class TestSegmentIntersectsRect:
    def test_no_intersection(self):
        assert not segment_intersects_rect(1, 1, 5, 3, 2, 5, 6, 2)

    def test_inside(self):
        assert segment_intersects_rect(0, 0, 6, 6, 3, 3, 2, 2)

    def test_crossing_corner(self):
        assert segment_intersects_rect(1, 1, 5, 5, 1, 4, 5, 1)

    def test_crossing_middle(self):
        assert segment_intersects_rect(1, 1, 5, 5, 0, 3, 5, 2)

    def test_touching_edge(self):
        assert segment_intersects_rect(0, 0, 4, 0, 2, 2, 4, 4)

    def test_touching_corner(self):
        assert segment_intersects_rect(0, 0, 4, 4, 2, 2, 4, 4)


class TestSegmentRectanglePenetration:
    def test_no_intersection(self):
        penetration = segment_rectangle_penetration(0, 0, 4, 0, 2, 5, 2, 2)
        assert penetration == 0.0

    def test_intersection(self):
        penetration = segment_rectangle_penetration(0, 0, 4, 4, 2, 2, 2, 2)
        assert penetration > 0.0


# =============================================================================
# NodeEdge.compute Edge Case Tests
# =============================================================================

class TestNodeEdgeEdgeCases:
    """Edge case tests for NodeEdge penalty."""

    def test_no_edges_zero_penalty(self):
        """Graph with nodes but no edges should have zero penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[]
        )
        penalty = NodeEdge(g)
        centers = np.array([0.0, 0.0, 100.0, 100.0])
        assert penalty.compute(centers) == 0.0

    def test_single_edge_two_nodes_zero_penalty(self):
        """Single edge connecting both nodes - no unconnected node-edge pairs."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[Edge("a", "b")]
        )
        penalty = NodeEdge(g)
        centers = np.array([0.0, 0.0, 100.0, 100.0])
        assert penalty.compute(centers) == 0.0

    def test_node_far_from_edge_zero_penalty(self):
        """Third node far from edge should have zero penalty."""
        g = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10), Node("c", 10, 10)
            ],
            edges=[Edge("a", "b")]
        )
        penalty = NodeEdge(g)
        # Edge from (0,0) to (100,0), node C at (50, 100) - far above
        centers = np.array([0.0, 0.0, 100.0, 0.0, 50.0, 100.0])
        assert penalty.compute(centers) == 0.0

    def test_node_on_edge_positive_penalty(self):
        """Node sitting on an edge should have positive penalty."""
        g = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10), Node("c", 20, 20)
            ],
            edges=[Edge("a", "b")]
        )
        penalty = NodeEdge(g)
        # Edge from (0,0) to (100,0), node C at (50, 0) - right on the edge
        centers = np.array([0.0, 0.0, 100.0, 0.0, 50.0, 0.0])
        assert penalty.compute(centers) > 0.0

    def test_node_intersecting_edge_positive_penalty(self):
        """Node rectangle intersecting edge should have positive penalty."""
        g = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10), Node("c", 30, 30)
            ],
            edges=[Edge("a", "b")]
        )
        penalty = NodeEdge(g)
        # Edge from (0,0) to (100,100), node C at (50,50) with large size - intersection
        centers = np.array([0.0, 0.0, 100.0, 100.0, 50.0, 50.0])
        assert penalty.compute(centers) > 0.0

    def test_connected_node_skipped(self):
        """Edge endpoints should not be penalized for their own edge."""
        g = Graph(
            nodes=[
                Node("a", 50, 50),  # Large node
                Node("b", 50, 50),
            ],
            edges=[Edge("a", "b")]
        )
        penalty = NodeEdge(g)
        # Even with overlapping large nodes, no penalty since they're connected
        centers = np.array([0.0, 0.0, 30.0, 30.0])
        assert penalty.compute(centers) == 0.0

    def test_multiple_edges_multiple_nodes(self):
        """Complex graph with multiple intersections."""
        g = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10),
                Node("c", 10, 10), Node("d", 10, 10),
                Node("e", 30, 30)  # Large node in the middle
            ],
            edges=[Edge("a", "b"), Edge("c", "d")]
        )
        penalty = NodeEdge(g)
        # Two crossing edges with node E in the middle
        centers = np.array([
            0.0, 0.0,      # a
            100.0, 100.0,  # b
            0.0, 100.0,    # c
            100.0, 0.0,    # d
            50.0, 50.0     # e - in the middle, intersects both edges
        ])
        assert penalty.compute(centers) > 0.0

    def test_more_intersections_higher_penalty(self):
        """More intersections should result in higher penalty."""
        # Graph with 1 node-edge intersection
        g1 = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10), Node("c", 30, 30)
            ],
            edges=[Edge("a", "b")]
        )
        centers1 = np.array([0.0, 0.0, 100.0, 100.0, 50.0, 50.0])
        penalty1 = NodeEdge(g1).compute(centers1)

        # Graph with 2 edges, node intersects both
        g2 = Graph(
            nodes=[
                Node("a", 10, 10), Node("b", 10, 10),
                Node("c", 10, 10), Node("d", 10, 10),
                Node("e", 30, 30)
            ],
            edges=[Edge("a", "b"), Edge("c", "d")]
        )
        # X pattern with node E in middle
        centers2 = np.array([
            0.0, 0.0,      # a
            100.0, 100.0,  # b
            0.0, 100.0,    # c
            100.0, 0.0,    # d
            50.0, 50.0     # e
        ])
        penalty2 = NodeEdge(g2).compute(centers2)

        assert penalty2 > penalty1


# =============================================================================
# Reference Comparison Tests
# =============================================================================

class TestNodeEdgeReferenceComparison:
    """Compare vectorized implementation against reference."""

    @pytest.fixture
    def random_graphs(self):
        """Generate random graphs with varying node/edge counts."""
        graphs = []
        random.seed(42)

        for n_nodes in [4, 6, 8, 10, 12, 15]:
            # Random node sizes
            nodes = [
                Node(f"n{i}", random.randint(10, 30), random.randint(10, 30))
                for i in range(n_nodes)
            ]

            # Random edges with moderate density
            edge_density = random.uniform(0.15, 0.3)
            edges = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if random.random() < edge_density:
                        edges.append(Edge(f"n{i}", f"n{j}"))

            if len(edges) < 1:
                edges = [Edge("n0", "n1")]

            g = Graph(nodes=nodes, edges=edges)

            # Random positions (spread out, some may intersect)
            centers = np.array([random.uniform(0, 200) for _ in range(n_nodes * 2)])

            graphs.append((g, centers))

        return graphs

    def test_matches_reference_implementation(self, random_graphs):
        """Vectorized implementation should match reference for random graphs."""
        for g, centers in random_graphs:
            penalty = NodeEdge(g)
            vectorized_result = penalty.compute(centers)
            reference_result = node_edge_reference(g, centers)

            # Use relative tolerance for potentially large values
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

        for n_nodes in [25, 40, 60]:
            nodes = [
                Node(f"n{i}", random.randint(10, 30), random.randint(10, 30))
                for i in range(n_nodes)
            ]
            # Moderate edge density
            edges = [Edge(f"n{i}", f"n{j}")
                     for i in range(n_nodes) for j in range(i + 1, n_nodes)
                     if random.random() < 0.2]
            g = Graph(nodes=nodes, edges=edges)
            centers = np.array([random.uniform(0, 300) for _ in range(n_nodes * 2)])
            large_graphs.append((g, centers))

        n_iterations = 30

        for g, centers in large_graphs:
            penalty = NodeEdge(g)

            # Time reference
            start = time.perf_counter()
            for _ in range(n_iterations):
                node_edge_reference(g, centers)
            ref_time = time.perf_counter() - start

            # Time vectorized
            start = time.perf_counter()
            for _ in range(n_iterations):
                penalty.compute(centers)
            vec_time = time.perf_counter() - start

            speedup = ref_time / vec_time
            print(f"\nNodes: {len(g.nodes):3d}, Edges: {len(g.edges):4d} | "
                  f"Ref: {ref_time:.4f}s, Vec: {vec_time:.4f}s, Speedup: {speedup:.2f}x")

            # Vectorized should be faster
            assert speedup > 1.5, "Vectorized implementation should be faster than reference"

            # Verify correctness
            vec_result = penalty.compute(centers)
            ref_result = node_edge_reference(g, centers)
            if abs(ref_result) > 1e-10:
                rel_diff = abs(vec_result - ref_result) / abs(ref_result)
                assert rel_diff < 1e-9
            else:
                assert abs(vec_result - ref_result) < 1e-10
