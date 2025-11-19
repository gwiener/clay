"""
Tests for geometry utilities (CCW and segment intersection).

These tests verify the low-level geometric primitives used for
collision detection, independent of the layout engine.
"""

import pytest

from clay.geometry import ccw, segments_intersect


class TestCCW:
    """Tests for the CCW (counter-clockwise) orientation test."""

    def test_ccw_point_above_horizontal_line(self):
        """Point above a horizontal line should be counter-clockwise (left)."""
        a = (0, 0)
        b = (4, 0)
        c = (2, 1)  # Above the line
        assert ccw(a, b, c) is True

    def test_ccw_point_below_horizontal_line(self):
        """Point below a horizontal line should be clockwise (right)."""
        a = (0, 0)
        b = (4, 0)
        c = (2, -1)  # Below the line
        assert ccw(a, b, c) is False

    def test_ccw_point_left_of_vertical_line(self):
        """Point to the left of a vertical line should be counter-clockwise."""
        a = (2, 0)
        b = (2, 4)
        c = (1, 2)  # Left of the line
        assert ccw(a, b, c) is True

    def test_ccw_point_right_of_vertical_line(self):
        """Point to the right of a vertical line should be clockwise."""
        a = (2, 0)
        b = (2, 4)
        c = (3, 2)  # Right of the line
        assert ccw(a, b, c) is False

    def test_ccw_diagonal_line(self):
        """Test CCW with diagonal line."""
        a = (0, 0)
        b = (4, 4)
        c_left = (0, 2)   # Left side
        c_right = (2, 0)  # Right side

        assert ccw(a, b, c_left) is True
        assert ccw(a, b, c_right) is False

    def test_ccw_collinear_points(self):
        """Collinear points should return False (not counter-clockwise)."""
        a = (0, 0)
        b = (4, 0)
        c = (2, 0)  # On the line
        # Collinear gives cross product = 0, which is not > 0
        assert ccw(a, b, c) is False


class TestSegmentsIntersect:
    """Tests for segment-segment intersection detection."""

    def test_crossing_segments_x_shape(self):
        """Two segments forming an X should intersect."""
        # Horizontal segment
        p1, p2 = (0, 0), (4, 0)
        # Vertical segment crossing through middle
        p3, p4 = (2, -1), (2, 1)

        assert segments_intersect(p1, p2, p3, p4) is True

    def test_parallel_horizontal_segments(self):
        """Parallel horizontal segments should not intersect."""
        p1, p2 = (0, 0), (4, 0)
        p3, p4 = (0, 1), (4, 1)

        assert segments_intersect(p1, p2, p3, p4) is False

    def test_parallel_vertical_segments(self):
        """Parallel vertical segments should not intersect."""
        p1, p2 = (0, 0), (0, 4)
        p3, p4 = (1, 0), (1, 4)

        assert segments_intersect(p1, p2, p3, p4) is False

    def test_segments_touching_at_endpoint(self):
        """Segments that touch at an endpoint should intersect (T-shape)."""
        # Horizontal segment
        p1, p2 = (0, 0), (4, 0)
        # Vertical segment touching at midpoint
        p3, p4 = (2, 0), (2, 2)

        assert segments_intersect(p1, p2, p3, p4) is True

    def test_segments_sharing_endpoint(self):
        """Segments that share an endpoint (L-shape) - edge case.

        Note: The CCW straddle test doesn't detect this as intersection
        because the shared endpoint makes the segments collinear at that
        point. This is a known limitation for degenerate cases.
        """
        p1, p2 = (0, 0), (4, 0)
        p3, p4 = (0, 0), (0, 4)

        # This is False due to the collinearity at shared endpoint
        assert segments_intersect(p1, p2, p3, p4) is False

    def test_non_intersecting_segments_far_apart(self):
        """Segments that are far apart should not intersect."""
        p1, p2 = (0, 0), (1, 0)
        p3, p4 = (5, 5), (6, 6)

        assert segments_intersect(p1, p2, p3, p4) is False

    def test_non_intersecting_segments_nearly_touching(self):
        """Segments that are close but don't touch should not intersect."""
        # Horizontal segment
        p1, p2 = (0, 0), (4, 0)
        # Vertical segment just to the right
        p3, p4 = (5, -1), (5, 1)

        assert segments_intersect(p1, p2, p3, p4) is False

    def test_diagonal_intersecting_segments(self):
        """Two diagonal segments crossing should intersect."""
        # Diagonal from bottom-left to top-right
        p1, p2 = (0, 0), (4, 4)
        # Diagonal from bottom-right to top-left
        p3, p4 = (4, 0), (0, 4)

        assert segments_intersect(p1, p2, p3, p4) is True

    def test_collinear_overlapping_segments(self):
        """Collinear overlapping segments should not intersect (limitation)."""
        # Both horizontal on same line, overlapping
        p1, p2 = (0, 0), (4, 0)
        p3, p4 = (2, 0), (6, 0)

        # Note: This algorithm doesn't detect collinear overlap as intersection
        # This is a known limitation of the straddle test
        assert segments_intersect(p1, p2, p3, p4) is False

    def test_collinear_non_overlapping_segments(self):
        """Collinear non-overlapping segments should not intersect."""
        p1, p2 = (0, 0), (2, 0)
        p3, p4 = (3, 0), (5, 0)

        assert segments_intersect(p1, p2, p3, p4) is False

    def test_perpendicular_segments_not_touching(self):
        """Perpendicular segments that don't reach each other."""
        # Short horizontal segment
        p1, p2 = (0, 0), (1, 0)
        # Vertical segment to the right
        p3, p4 = (3, -1), (3, 1)

        assert segments_intersect(p1, p2, p3, p4) is False

    def test_segments_with_negative_coordinates(self):
        """Test with negative coordinates."""
        p1, p2 = (-2, -2), (2, 2)
        p3, p4 = (-2, 2), (2, -2)

        assert segments_intersect(p1, p2, p3, p4) is True

    def test_segments_with_float_coordinates(self):
        """Test with floating point coordinates."""
        p1, p2 = (0.5, 0.5), (4.5, 0.5)
        p3, p4 = (2.5, -0.5), (2.5, 1.5)

        assert segments_intersect(p1, p2, p3, p4) is True

    def test_very_short_segments_intersecting(self):
        """Test with very short segments that intersect."""
        p1, p2 = (1.0, 1.0), (1.1, 1.0)
        p3, p4 = (1.05, 0.9), (1.05, 1.1)

        assert segments_intersect(p1, p2, p3, p4) is True
