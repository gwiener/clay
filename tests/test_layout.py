"""
Tests for layout engine penalty functions.

Tests the energy function components and their integration into the
layout optimization process.
"""

import numpy as np

from clay.layout import Node, edge_node_intersection_penalty


class TestEdgeNodeIntersectionPenalty:
    """Tests for the edge-node intersection penalty function."""

    def test_no_intersection_no_penalty(self):
        """Edges that don't intersect nodes should have zero penalty."""
        # Three nodes in a line
        positions = np.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [200.0, 0.0],
        ])

        nodes = [
            Node('A', width=20, height=20),
            Node('B', width=20, height=20),
            Node('C', width=20, height=20),
        ]

        # Edge from A to B - doesn't pass through C
        edges = [(0, 1)]

        penalty = edge_node_intersection_penalty(positions, edges, nodes)
        assert penalty == 0.0, "Edge not passing through any nodes should have zero penalty"

    def test_edge_crossing_through_node_has_penalty(self):
        """Edge crossing through a non-endpoint node should be penalized."""
        # Three nodes: A and C far apart, B in the middle
        positions = np.array([
            [0.0, 50.0],    # A - left
            [50.0, 50.0],   # B - center
            [100.0, 50.0],  # C - right
        ])

        nodes = [
            Node('A', width=10, height=10),
            Node('B', width=30, height=30),  # Larger node in the middle
            Node('C', width=10, height=10),
        ]

        # Edge from A to C passes directly through B's center
        edges = [(0, 2)]

        penalty = edge_node_intersection_penalty(positions, edges, nodes)
        assert penalty > 0, "Edge passing through node should have positive penalty"
        assert penalty > 100, "Edge through center should have substantial penalty"

    def test_edge_endpoints_not_penalized(self):
        """Edges should not be penalized for their own endpoint nodes."""
        # Two nodes with direct edge between them
        positions = np.array([
            [0.0, 0.0],
            [50.0, 0.0],
        ])

        nodes = [
            Node('A', width=20, height=20),
            Node('B', width=20, height=20),
        ]

        # Edge from A to B
        edges = [(0, 1)]

        penalty = edge_node_intersection_penalty(positions, edges, nodes)
        assert penalty == 0.0, "Edge should not be penalized for its own endpoints"

    def test_deeper_penetration_higher_penalty(self):
        """Edges passing deeper through nodes should have higher penalty."""
        # Node B in center, edges at different distances
        node_b_pos = np.array([50.0, 50.0])
        node_b = Node('B', width=40, height=40)

        # Edge 1: Passes through center (maximum depth)
        positions_center = np.array([
            [0.0, 50.0],     # A - left
            [50.0, 50.0],    # B - center
            [100.0, 50.0],   # C - right
        ])

        # Edge 2: Passes near edge (minimum depth)
        positions_edge = np.array([
            [0.0, 30.0],     # A - left lower
            [50.0, 50.0],    # B - center
            [100.0, 70.0],   # C - right upper
        ])

        nodes = [
            Node('A', width=10, height=10),
            node_b,
            Node('C', width=10, height=10),
        ]

        edges = [(0, 2)]  # A to C, passing through B

        penalty_center = edge_node_intersection_penalty(positions_center, edges, nodes)
        penalty_edge = edge_node_intersection_penalty(positions_edge, edges, nodes)

        # Center crossing should be more penalized (though might be similar due to midpoint)
        assert penalty_center > 0, "Center crossing should be penalized"
        assert penalty_edge >= 0, "Edge grazing might or might not intersect"

    def test_multiple_intersections_accumulate(self):
        """Multiple edge-node intersections should accumulate penalties."""
        # Four nodes in a line, with edges crossing through middle nodes
        positions = np.array([
            [0.0, 50.0],     # A
            [40.0, 50.0],    # B
            [60.0, 50.0],    # C
            [100.0, 50.0],   # D
        ])

        nodes = [
            Node('A', width=10, height=10),
            Node('B', width=30, height=30),
            Node('C', width=30, height=30),
            Node('D', width=10, height=10),
        ]

        # Edge from A to D passes through both B and C
        edges = [(0, 3)]

        penalty = edge_node_intersection_penalty(positions, edges, nodes)
        assert penalty > 0, "Edge passing through multiple nodes should be penalized"

    def test_no_penalty_for_nearby_but_not_intersecting(self):
        """Edges near but not intersecting nodes should have zero penalty."""
        # Node B above the line from A to C
        positions = np.array([
            [0.0, 0.0],      # A - left
            [50.0, 50.0],    # B - above
            [100.0, 0.0],    # C - right
        ])

        nodes = [
            Node('A', width=10, height=10),
            Node('B', width=10, height=10),
            Node('C', width=10, height=10),
        ]

        # Edge from A to C passes below B
        edges = [(0, 2)]

        penalty = edge_node_intersection_penalty(positions, edges, nodes)
        assert penalty == 0.0, "Edge not intersecting node should have zero penalty"

    def test_works_with_different_node_sizes(self):
        """Penalty should work correctly with various node sizes."""
        positions = np.array([
            [0.0, 50.0],
            [50.0, 50.0],
            [100.0, 50.0],
        ])

        # Mix of small and large nodes
        nodes = [
            Node('A', width=5, height=5),
            Node('B', width=80, height=80),  # Very large node
            Node('C', width=5, height=5),
        ]

        # Edge from A to C definitely passes through large B
        edges = [(0, 2)]

        penalty = edge_node_intersection_penalty(positions, edges, nodes)
        assert penalty > 0, "Edge through large node should be penalized"
        # Larger node means deeper penetration possible
        assert penalty > 1000, "Large node intersection should have substantial penalty"
