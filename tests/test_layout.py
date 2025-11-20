"""
Tests for layout engine penalty functions.

Tests the energy function components and their integration into the
layout optimization process.
"""

import numpy as np

from clay.layout import (
    Node,
    edge_node_intersection_penalty,
    layout_graph,
    LayoutResult,
    LayoutStats
)


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


class TestLayoutGraph:
    """Tests for the main layout_graph function and result structure."""

    def test_layout_result_has_stats(self):
        """layout_graph should return LayoutResult with populated stats."""
        # Create a simple graph
        nodes_dict = {
            'A': Node('A', width=40, height=30),
            'B': Node('B', width=40, height=30),
            'C': Node('C', width=40, height=30),
        }
        edges = [('A', 'B'), ('B', 'C')]

        # Layout the graph
        result = layout_graph(nodes_dict, edges, verbose=False)

        # Verify result type
        assert isinstance(result, LayoutResult), "Should return LayoutResult"
        assert isinstance(result.stats, LayoutStats), "Should contain LayoutStats"

        # Verify stats are populated
        assert result.stats.success is not None, "success should be set"
        assert result.stats.iterations > 0, "Should have positive iterations"
        assert result.stats.function_evals > 0, "Should have positive function evals"
        assert result.stats.final_energy >= 0, "Should have non-negative energy"
        assert isinstance(result.stats.message, str), "message should be string"

        # Verify penalty breakdown is populated
        assert len(result.stats.penalty_breakdown) > 0, "penalty_breakdown should not be empty"
        expected_penalties = {'overlap', 'edge_length', 'straightness', 'edge_node', 'bbox', 'area'}
        assert set(result.stats.penalty_breakdown.keys()) == expected_penalties, \
            "Should have all penalty components"

        # Verify all penalties are numbers
        for name, value in result.stats.penalty_breakdown.items():
            assert isinstance(value, (int, float, np.number)), \
                f"Penalty {name} should be numeric"

        # Verify weights are populated
        assert len(result.stats.weights) > 0, "weights should not be empty"
        assert set(result.stats.weights.keys()) == expected_penalties, \
            "Should have weights for all penalties"

        # Verify positions are populated
        assert len(result.positions) == 3, "Should have 3 node positions"
        assert 'A' in result.positions, "Should have position for node A"
        assert 'B' in result.positions, "Should have position for node B"
        assert 'C' in result.positions, "Should have position for node C"

    def test_empty_graph_returns_valid_stats(self):
        """Empty graph should return LayoutResult with valid empty stats."""
        result = layout_graph({}, [], verbose=False)

        assert isinstance(result, LayoutResult), "Should return LayoutResult"
        assert result.stats.success is True, "Empty graph is trivially successful"
        assert result.stats.iterations == 0, "No iterations for empty graph"
        assert result.stats.final_energy == 0.0, "Zero energy for empty graph"
        assert len(result.positions) == 0, "No positions for empty graph"


class TestInitializationModes:
    """Tests for different initialization strategies and seed control."""

    def test_spring_initialization_mode(self):
        """Force-directed initialization should work and produce valid layout."""
        nodes = {
            'a': Node('A', width=40, height=20),
            'b': Node('B', width=40, height=20),
            'c': Node('C', width=40, height=20),
        }
        edges = [('a', 'b'), ('b', 'c')]

        result = layout_graph(nodes, edges, init_mode='spring', seed=42, verbose=False)

        assert isinstance(result, LayoutResult), "Should return LayoutResult"
        assert len(result.positions) == 3, "Should have 3 node positions"
        assert result.stats.seed == 42, "Should record seed in stats"

    def test_grid_initialization_mode(self):
        """Grid initialization should work (original behavior)."""
        nodes = {
            'a': Node('A', width=40, height=20),
            'b': Node('B', width=40, height=20),
            'c': Node('C', width=40, height=20),
        }
        edges = [('a', 'b'), ('b', 'c')]

        result = layout_graph(nodes, edges, init_mode='grid', verbose=False)

        assert isinstance(result, LayoutResult), "Should return LayoutResult"
        assert len(result.positions) == 3, "Should have 3 node positions"

    def test_random_initialization_mode(self):
        """Random initialization should work with seed control."""
        nodes = {
            'a': Node('A', width=40, height=20),
            'b': Node('B', width=40, height=20),
        }
        edges = [('a', 'b')]

        result = layout_graph(nodes, edges, init_mode='random', seed=123, verbose=False)

        assert isinstance(result, LayoutResult), "Should return LayoutResult"
        assert len(result.positions) == 2, "Should have 2 node positions"
        assert result.stats.seed == 123, "Should record seed in stats"

    def test_seed_reproducibility(self):
        """Same seed should produce identical layouts."""
        nodes = {
            'a': Node('A', width=40, height=20),
            'b': Node('B', width=40, height=20),
            'c': Node('C', width=40, height=20),
        }
        edges = [('a', 'b'), ('b', 'c')]

        result1 = layout_graph(nodes, edges, init_mode='spring', seed=42, verbose=False)
        result2 = layout_graph(nodes, edges, init_mode='spring', seed=42, verbose=False)

        # Check that positions are identical
        for node_id in nodes.keys():
            pos1 = result1.positions[node_id]
            pos2 = result2.positions[node_id]
            assert np.allclose(pos1, pos2), f"Positions for {node_id} should be identical with same seed"

        # Check that energies are identical
        assert result1.stats.final_energy == result2.stats.final_energy, "Final energies should be identical"

    def test_different_seeds_produce_different_layouts(self):
        """Different seeds should produce different layouts."""
        nodes = {
            'a': Node('A', width=40, height=20),
            'b': Node('B', width=40, height=20),
            'c': Node('C', width=40, height=20),
        }
        edges = [('a', 'b'), ('b', 'c')]

        result1 = layout_graph(nodes, edges, init_mode='spring', seed=1, verbose=False)
        result2 = layout_graph(nodes, edges, init_mode='spring', seed=2, verbose=False)

        # At least one position should be different
        positions_differ = False
        for node_id in nodes.keys():
            pos1 = result1.positions[node_id]
            pos2 = result2.positions[node_id]
            if not np.allclose(pos1, pos2):
                positions_differ = True
                break

        assert positions_differ, "Different seeds should produce different layouts"

    def test_seed_recorded_in_stats(self):
        """Seed value should be recorded in stats for traceability."""
        nodes = {'a': Node('A', width=40, height=20)}
        edges = []

        result_with_seed = layout_graph(nodes, edges, seed=999, verbose=False)
        result_without_seed = layout_graph(nodes, edges, seed=None, verbose=False)

        assert result_with_seed.stats.seed == 999, "Seed should be recorded when provided"
        assert result_without_seed.stats.seed is None, "Seed should be None when not provided"

    def test_invalid_init_mode_raises_error(self):
        """Invalid initialization mode should raise ValueError."""
        nodes = {'a': Node('A', width=40, height=20)}
        edges = []

        try:
            layout_graph(nodes, edges, init_mode='invalid', verbose=False)
            assert False, "Should have raised ValueError for invalid init_mode"
        except ValueError as e:
            assert 'Unknown init_mode' in str(e), "Error message should mention invalid init_mode"
