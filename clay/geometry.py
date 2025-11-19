"""
Geometry utilities for collision detection and intersection testing.

This module provides low-level geometric primitives used by the layout engine
for detecting intersections between edges and nodes.
"""

from typing import Tuple

import numpy as np


def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    """
    Test if point C is counter-clockwise (left) of the line from A to B.

    Uses the cross product to determine orientation. Geometrically, this
    computes the signed area of the parallelogram formed by vectors AB and AC.

    Args:
        a: First point defining the line (x, y)
        b: Second point defining the line (x, y)
        c: Test point (x, y)

    Returns:
        True if C is to the left of (counter-clockwise from) line A→B
        False if C is to the right of (clockwise from) line A→B

    Examples:
        >>> ccw((0, 0), (4, 0), (2, 1))   # C above horizontal line
        True
        >>> ccw((0, 0), (4, 0), (2, -1))  # C below horizontal line
        False

    Note:
        The cross product formula:
        cross = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)

        If cross > 0: C is left of AB (counter-clockwise)
        If cross < 0: C is right of AB (clockwise)
        If cross = 0: C is on the line AB (collinear)
    """
    # Calculate cross product using the determinant formula
    # This gives us the signed area of the parallelogram
    cross_product = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return cross_product > 0


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float]
) -> bool:
    """
    Test if two line segments intersect.

    Two segments (P1→P2) and (P3→P4) intersect if their endpoints are on
    opposite sides of each other. This is the classic "straddle test" using
    the CCW (counter-clockwise) orientation check.

    Args:
        p1: First endpoint of segment 1 (x, y)
        p2: Second endpoint of segment 1 (x, y)
        p3: First endpoint of segment 2 (x, y)
        p4: Second endpoint of segment 2 (x, y)

    Returns:
        True if the segments intersect (including touching at endpoints)
        False if the segments do not intersect

    Algorithm:
        For segments to intersect, both of these must be true:
        1. P3 and P4 are on opposite sides of line P1→P2
        2. P1 and P2 are on opposite sides of line P3→P4

        This ensures the segments "straddle" each other.

    Examples:
        >>> # Crossing segments (X shape)
        >>> segments_intersect((0, 0), (4, 0), (2, -1), (2, 1))
        True

        >>> # Parallel segments
        >>> segments_intersect((0, 0), (4, 0), (0, 1), (4, 1))
        False

        >>> # T-shape (touching at endpoint)
        >>> segments_intersect((0, 0), (4, 0), (2, 0), (2, 2))
        True

    Note:
        This implementation treats segments that touch at endpoints as
        intersecting. For stricter intersection (excluding endpoints),
        the CCW tests would need to use strict inequality (!=).
    """
    # Check if P3 and P4 are on opposite sides of line P1→P2
    # If ccw(P1, P2, P3) != ccw(P1, P2, P4), they're on opposite sides
    side1 = ccw(p1, p2, p3) != ccw(p1, p2, p4)

    # Check if P1 and P2 are on opposite sides of line P3→P4
    # If ccw(P3, P4, P1) != ccw(P3, P4, P2), they're on opposite sides
    side2 = ccw(p3, p4, p1) != ccw(p3, p4, p2)

    # Both conditions must be true for intersection
    return side1 and side2
