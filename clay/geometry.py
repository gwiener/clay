import numpy as np


def point_to_segment_distance(
    ax: int, ay: int,
    bx: int, by: int,
    px: int, py: int
) -> float:
    """
    Calculate minimum distance from a point to a line segment.

    Projects the point onto the infinite line through the segment,
    then clamps to the segment endpoints if projection falls outside.

    Args:
        point: Point coordinates as array [x, y]
        seg_start: Segment start point [x, y]
        seg_end: Segment end point [x, y]

    Returns:
        Minimum distance from point to segment (float)
    """
    seg_start = np.array([ax, ay])
    seg_end = np.array([bx, by])
    point = np.array([px, py])
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq < 1e-10:
        # Segment is essentially a point
        return np.linalg.norm(point - seg_start)

    # Project point onto infinite line: t is parameter (0=start, 1=end)
    t = np.dot(point - seg_start, seg_vec) / seg_len_sq

    # Clamp t to [0, 1] to stay within segment bounds
    t = max(0.0, min(1.0, t))

    # Closest point on segment
    closest = seg_start + t * seg_vec

    return np.linalg.norm(point - closest)
