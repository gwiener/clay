import numpy as np

from clay.graph import Graph, Node
from clay.penalties.node_edge import (
    NodeEdge,
    segment_intersects_rect,
    segment_rectangle_penetration,
)


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


class TestNodeEdgePenalty:
    def test_compute_no_penalty(self):
        nodes = [
            Node(name='A', width=2, height=2),
            Node(name='B', width=2, height=2)
        ]
        edges = [('A', 'B')]
        g = Graph(nodes=nodes, edges=edges)
        penalty = NodeEdge(g, w=1.0)

        centers = np.array([0.0, 0.0, 5.0, 5.0])  # A at (0,0), B at (5,5)
        computed_penalty = penalty.compute(centers)
        assert abs(computed_penalty - 0.0) < 1e-6
    
    def test_compute_with_penalty(self):
        nodes = [
            Node(name='A', width=2, height=2),
            Node(name='B', width=2, height=2),
            Node(name='C', width=2, height=2)
        ]
        edges = [('A', 'B')]
        g = Graph(nodes=nodes, edges=edges)
        penalty = NodeEdge(g, w=1.0)
        centers = np.array([0.0, 0.0, 10.0, 10.0, 5.0, 5.0])  # A at (0,0), B at (10,10), C at (5,5)
        computed_penalty = penalty.compute(centers)
        assert computed_penalty > 0.0
        