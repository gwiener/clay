"""
Tests for geometry utilities (CCW and segment intersection).

These tests verify the low-level geometric primitives used for
collision detection, independent of the layout engine.
"""

import pytest

from clay.geometry import ccw, segments_intersect, segment_intersects_rectangle


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


class TestSegmentIntersectsRectangle:
    """Tests for segment-rectangle intersection detection."""

    def test_horizontal_segment_crossing_through_rectangle(self):
        """Horizontal segment passing through rectangle center."""
        # Rectangle centered at (5, 5), size 4x4
        # Segment from left to right, passing through center
        seg_start = (0, 5)
        seg_end = (10, 5)
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_vertical_segment_crossing_through_rectangle(self):
        """Vertical segment passing through rectangle center."""
        seg_start = (5, 0)
        seg_end = (5, 10)
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_diagonal_segment_crossing_through_rectangle(self):
        """Diagonal segment passing through rectangle."""
        seg_start = (0, 0)
        seg_end = (10, 10)
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_segment_entirely_outside_rectangle(self):
        """Segment that doesn't touch rectangle at all."""
        # Segment far to the left of rectangle
        seg_start = (0, 0)
        seg_end = (1, 0)
        rect_center = (5, 5)
        rect_width = 2
        rect_height = 2

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is False

    def test_segment_entirely_inside_rectangle(self):
        """Small segment completely contained within rectangle."""
        # Rectangle 10x10 centered at (5, 5)
        # Small segment inside
        seg_start = (4, 5)
        seg_end = (6, 5)
        rect_center = (5, 5)
        rect_width = 10
        rect_height = 10

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_segment_touching_rectangle_edge(self):
        """Segment touching but not crossing rectangle edge."""
        # Rectangle 4x4 centered at (5, 5)
        # Edges at x=3,7 and y=3,7
        # Segment along top edge
        seg_start = (4, 7)
        seg_end = (6, 7)
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_segment_touching_rectangle_corner(self):
        """Segment touching rectangle at a corner point - edge case.

        Note: This is a degenerate case where the segment endpoint is exactly
        at a corner. The CCW straddle test may not detect this as intersection
        because the corner is collinear with both adjacent edges. This is a
        known limitation for corner-touching cases.
        """
        # Rectangle 4x4 centered at (5, 5)
        # Corners at (3,3), (7,3), (7,7), (3,7)
        # Segment ending at top-right corner
        seg_start = (8, 8)
        seg_end = (7, 7)
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        # This may return False due to corner degeneracy
        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is False

    def test_segment_starting_inside_ending_outside(self):
        """Segment starts inside rectangle, ends outside."""
        seg_start = (5, 5)  # Inside
        seg_end = (10, 10)  # Outside
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_segment_starting_outside_ending_inside(self):
        """Segment starts outside rectangle, ends inside."""
        seg_start = (0, 0)  # Outside
        seg_end = (5, 5)    # Inside
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_segment_parallel_to_rectangle_edge_outside(self):
        """Segment parallel to rectangle edge but outside."""
        # Rectangle 4x4 centered at (5, 5)
        # Right edge at x=7
        # Segment parallel to right edge, to the right of it
        seg_start = (8, 4)
        seg_end = (8, 6)
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is False

    def test_segment_near_but_not_touching_rectangle(self):
        """Segment very close to rectangle but not intersecting."""
        # Rectangle 4x4 centered at (5, 5)
        # Top edge at y=7
        # Segment just above top edge
        seg_start = (4, 7.1)
        seg_end = (6, 7.1)
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is False

    def test_segment_crossing_two_opposite_edges(self):
        """Segment entering one edge and exiting opposite edge."""
        # Segment from left to right, longer than rectangle
        seg_start = (2, 5)
        seg_end = (8, 5)
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_segment_crossing_two_adjacent_edges(self):
        """Segment entering one edge and exiting adjacent edge (corner cut)."""
        # Diagonal segment cutting through corner
        seg_start = (3, 6)
        seg_end = (6, 3)
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_very_small_rectangle(self):
        """Test with a very small rectangle."""
        seg_start = (5, 4)
        seg_end = (5, 6)
        rect_center = (5, 5)
        rect_width = 0.5
        rect_height = 0.5

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_very_long_segment_crossing_small_rectangle(self):
        """Very long segment crossing a small rectangle."""
        seg_start = (-100, 5)
        seg_end = (100, 5)
        rect_center = (5, 5)
        rect_width = 2
        rect_height = 2

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_segment_with_negative_coordinates(self):
        """Test with negative coordinates."""
        seg_start = (-5, -5)
        seg_end = (5, 5)
        rect_center = (0, 0)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_rectangle_with_different_width_height(self):
        """Test with non-square rectangle (wide)."""
        # Wide rectangle 10x2
        seg_start = (3, 5)
        seg_end = (7, 5)
        rect_center = (5, 5)
        rect_width = 10
        rect_height = 2

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_rectangle_tall_narrow(self):
        """Test with non-square rectangle (tall)."""
        # Tall rectangle 2x10
        seg_start = (5, 3)
        seg_end = (5, 7)
        rect_center = (5, 5)
        rect_width = 2
        rect_height = 10

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True

    def test_segment_grazing_rectangle_edge_horizontally(self):
        """Segment just grazing along the edge of rectangle."""
        # Rectangle 4x4 centered at (5, 5)
        # Bottom edge at y=3
        # Segment along bottom edge
        seg_start = (4, 3)
        seg_end = (6, 3)
        rect_center = (5, 5)
        rect_width = 4
        rect_height = 4

        assert segment_intersects_rectangle(
            seg_start, seg_end, rect_center, rect_width, rect_height
        ) is True
