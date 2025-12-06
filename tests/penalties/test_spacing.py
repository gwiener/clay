import random
import time

import numpy as np
import pytest

from clay.graph import Graph, Node, Edge
from clay.penalties.spacing import Spacing


def _signed_distance_ref(cx1, cy1, w1, h1, cx2, cy2, w2, h2):
    """Reference implementation of signed distance between two nodes."""
    dx_gap = abs(cx2 - cx1) - (w1 + w2) / 2
    dy_gap = abs(cy2 - cy1) - (h1 + h2) / 2

    if dx_gap >= 0 or dy_gap >= 0:
        return (max(0, dx_gap)**2 + max(0, dy_gap)**2) ** 0.5
    else:
        return -(dx_gap**2 + dy_gap**2) ** 0.5


def spacing_reference(
    g: Graph,
    centers: np.ndarray,
    D: int = 50,
    k_edge: float = 1.0,
    k_repel: float = 10.0
) -> float:
    """Reference implementation of spacing penalty (non-vectorized)."""
    n_nodes = len(g.nodes)
    energy = 0.0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            cx1, cy1 = centers[2*i], centers[2*i + 1]
            cx2, cy2 = centers[2*j], centers[2*j + 1]
            w1, h1 = g.nodes[i].width, g.nodes[i].height
            w2, h2 = g.nodes[j].width, g.nodes[j].height
            d = _signed_distance_ref(cx1, cy1, w1, h1, cx2, cy2, w2, h2)
            delta = d - D
            name_i, name_j = g.nodes[i].name, g.nodes[j].name
            is_edge = any((e.src == name_i and e.dst == name_j) or (e.src == name_j and e.dst == name_i) for e in g.edges)
            added_energy = 0.0
            if is_edge:
                added_energy = 0.5 * k_edge * delta ** 2
            elif delta < 0:
                added_energy = 0.5 * k_repel * delta ** 2
            energy += added_energy
    return energy


class TestSpacingEdgeCases:
    """Edge case tests for Spacing penalty."""

    def test_unconnected_nodes_far_apart_zero_penalty(self):
        """Non-connected nodes far apart should have exactly zero penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[]
        )
        penalty = Spacing(g)
        centers = np.array([100.0, 100.0, 200.0, 200.0])
        assert penalty.compute(centers) == 0.0

    def test_unconnected_nodes_overlapping_positive_penalty(self):
        """Overlapping non-connected nodes should have positive repulsion penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[]
        )
        penalty = Spacing(g)
        centers = np.array([100.0, 100.0, 105.0, 105.0])
        result = penalty.compute(centers)
        assert result > 0.0

    def test_connected_nodes_at_desired_distance_zero_penalty(self):
        """Connected nodes at exactly desired distance D should have zero penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[Edge("a", "b")]
        )
        penalty = Spacing(g, D=50)
        # Place nodes so signed_distance == 50
        # signed_distance = sqrt(dx_gap^2 + dy_gap^2) when separated
        # For horizontal placement: dx_gap = |cx2-cx1| - (w1+w2)/2 = |cx2-cx1| - 10
        # We want dx_gap = 50, so |cx2-cx1| = 60
        centers = np.array([100.0, 100.0, 160.0, 100.0])
        result = penalty.compute(centers)
        assert abs(result) < 1e-10

    def test_connected_nodes_closer_than_desired_positive_penalty(self):
        """Connected nodes closer than D should have positive spring penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[Edge("a", "b")]
        )
        penalty = Spacing(g, D=50)
        centers = np.array([100.0, 100.0, 115.0, 100.0])
        result = penalty.compute(centers)
        assert result > 0.0

    def test_connected_nodes_farther_than_desired_positive_penalty(self):
        """Connected nodes farther than D should have positive spring penalty."""
        g = Graph(
            nodes=[Node("a", 10, 10), Node("b", 10, 10)],
            edges=[Edge("a", "b")]
        )
        penalty = Spacing(g, D=50)
        centers = np.array([100.0, 100.0, 300.0, 100.0])
        result = penalty.compute(centers)
        assert result > 0.0

    def test_three_nodes_mixed_connectivity(self):
        """A-B connected, C isolated overlapping A: verify correct penalty sources."""
        g = Graph(
            nodes=[Node("A", 10, 10), Node("B", 10, 10), Node("C", 10, 10)],
            edges=[Edge("A", "B")]
        )
        penalty = Spacing(g, D=50)
        # A at (100,100), B at (160,100) -> A-B at desired distance
        # C at (105,100) -> overlaps with A
        centers = np.array([100.0, 100.0, 160.0, 100.0, 105.0, 100.0])
        result = penalty.compute(centers)
        # Should have penalty from A-C overlap (repulsion), B-C is far (zero)
        # A-B is at desired distance (zero)
        assert result > 0.0

    def test_single_node_zero_penalty(self):
        """Single node should have zero penalty (no pairs)."""
        g = Graph(
            nodes=[Node("a", 10, 10)],
            edges=[]
        )
        penalty = Spacing(g)
        centers = np.array([100.0, 100.0])
        assert penalty.compute(centers) == 0.0


class TestSpacingReferenceComparison:
    """Compare vectorized implementation against reference."""

    @pytest.fixture
    def random_graphs(self):
        """Generate random graphs of various sizes."""
        graphs = []
        random.seed(42)

        for n_nodes in [2, 3, 5, 8, 10, 15, 20]:
            nodes = [Node(f"n{i}", 10 + random.randint(0, 20), 10 + random.randint(0, 20))
                     for i in range(n_nodes)]

            # Random edges (sparse to dense)
            edge_density = random.uniform(0.1, 0.5)
            edges = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if random.random() < edge_density:
                        edges.append(Edge(f"n{i}", f"n{j}"))

            g = Graph(nodes=nodes, edges=edges)

            # Random positions
            centers = np.array([random.uniform(50, 450) for _ in range(n_nodes * 2)])

            graphs.append((g, centers))

        return graphs

    def test_matches_reference_implementation(self, random_graphs):
        """Vectorized implementation should match reference for random graphs."""
        for g, centers in random_graphs:
            penalty = Spacing(g)
            vectorized_result = penalty.compute(centers)
            reference_result = spacing_reference(g, centers)

            assert abs(vectorized_result - reference_result) < 1e-6, \
                f"Mismatch for graph with {len(g.nodes)} nodes: {vectorized_result} vs {reference_result}"

    def test_performance_improvement(self, random_graphs):
        """Vectorized implementation should be faster than reference."""
        # Use larger graphs for timing
        random.seed(123)
        large_graphs = []
        for n_nodes in [30, 50, 80]:
            nodes = [Node(f"n{i}", 15, 15) for i in range(n_nodes)]
            edges = [Edge(f"n{i}", f"n{j}") for i in range(n_nodes) for j in range(i+1, n_nodes)
                     if random.random() < 0.2]
            g = Graph(nodes=nodes, edges=edges)
            centers = np.array([random.uniform(50, 650) for _ in range(n_nodes * 2)])
            large_graphs.append((g, centers))

        n_iterations = 50

        for g, centers in large_graphs:
            penalty = Spacing(g)

            # Time reference
            start = time.perf_counter()
            for _ in range(n_iterations):
                spacing_reference(g, centers)
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
            # Verify correctness
            assert abs(penalty.compute(centers) - spacing_reference(g, centers)) < 1e-6
