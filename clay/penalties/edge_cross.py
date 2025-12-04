import numpy as np

from clay.penalties import Penalty


def segment_cross(
    ax: int, ay: int,
    bx: int, by: int,
    cx: int, cy: int,
    dx: int, dy: int
) -> bool:
    """Check if segment AB crosses segment CD (scalar version for tests)."""
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(cx, cy, dx, dy, ax, ay)
    o2 = orient(cx, cy, dx, dy, bx, by)
    o3 = orient(ax, ay, bx, by, cx, cy)
    o4 = orient(ax, ay, bx, by, dx, dy)

    return o1 * o2 < 0 and o3 * o4 < 0


def _segment_cross_vectorized(ax, ay, bx, by, cx, cy, dx, dy):
    """Vectorized orientation test for segment crossing."""
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(cx, cy, dx, dy, ax, ay)
    o2 = orient(cx, cy, dx, dy, bx, by)
    o3 = orient(ax, ay, bx, by, cx, cy)
    o4 = orient(ax, ay, bx, by, dx, dy)

    return (o1 * o2 < 0) & (o3 * o4 < 0)


def _point_to_segment_distance_vectorized(seg_ax, seg_ay, seg_bx, seg_by, px, py):
    """
    Vectorized point-to-segment distance.

    Args:
        seg_ax, seg_ay: Segment start coordinates (arrays)
        seg_bx, seg_by: Segment end coordinates (arrays)
        px, py: Point coordinates (arrays)

    Returns:
        Array of distances from each point to corresponding segment.
    """
    # Segment vector
    seg_vx = seg_bx - seg_ax
    seg_vy = seg_by - seg_ay

    # Segment length squared
    seg_len_sq = seg_vx * seg_vx + seg_vy * seg_vy

    # Vector from segment start to point
    to_point_x = px - seg_ax
    to_point_y = py - seg_ay

    # Project point onto infinite line: t is parameter (0=start, 1=end)
    # Handle degenerate case where segment is a point
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (to_point_x * seg_vx + to_point_y * seg_vy) / seg_len_sq
        t = np.where(seg_len_sq < 1e-10, 0.0, t)

    # Clamp t to [0, 1]
    t = np.clip(t, 0.0, 1.0)

    # Closest point on segment
    closest_x = seg_ax + t * seg_vx
    closest_y = seg_ay + t * seg_vy

    # Distance from point to closest point
    dx = px - closest_x
    dy = py - closest_y

    return np.sqrt(dx * dx + dy * dy)


def _softmin_vectorized(values, axis=0, sharpness=1.0):
    """
    Vectorized softmin along an axis.

    Args:
        values: Array of shape (n_values, n_items)
        axis: Axis along which to compute softmin
        sharpness: Sharpness parameter

    Returns:
        Array of softmin values.
    """
    scale = np.maximum(values.max(axis=axis, keepdims=True), 1e-10)
    v = -sharpness * values / scale
    v_max = v.max(axis=axis, keepdims=True)
    result = scale * (-(v_max + np.log(np.exp(v - v_max).sum(axis=axis, keepdims=True))) / sharpness)
    return result.squeeze(axis=axis)


def _crossing_penalty_vectorized(ax, ay, bx, by, cx, cy, dx, dy):
    """
    Vectorized crossing penalty for multiple segment pairs.

    Returns:
        Array of crossing penalties (0.5 * min_dist^2).
    """
    # Match original argument order exactly
    d1 = _point_to_segment_distance_vectorized(ax, ay, cx, cy, dx, dy)
    d2 = _point_to_segment_distance_vectorized(bx, by, cx, cy, dx, dy)
    d3 = _point_to_segment_distance_vectorized(cx, cy, ax, ay, bx, by)
    d4 = _point_to_segment_distance_vectorized(dx, dy, ax, ay, bx, by)

    # Stack distances: shape (4, n_crossings)
    distances = np.stack([d1, d2, d3, d4], axis=0)

    # Compute softmin along axis 0
    min_dist = _softmin_vectorized(distances, axis=0)

    return 0.5 * min_dist ** 2


class EgdeCross(Penalty):
    def __init__(self, g, w=1.0):
        super().__init__(g, w)

        # Pre-compute edge endpoint indices: shape (m, 2)
        self.edge_indices = np.array([
            (g.name2idx[src], g.name2idx[dst]) for src, dst in g.edges
        ], dtype=np.int32) if g.edges else np.empty((0, 2), dtype=np.int32)

        # Pre-compute edge pair indices (upper triangle)
        m = len(g.edges)
        if m >= 2:
            self.i_indices, self.j_indices = np.triu_indices(m, k=1)
        else:
            self.i_indices = np.array([], dtype=np.int32)
            self.j_indices = np.array([], dtype=np.int32)

    def compute(self, centers: np.ndarray) -> float:
        if len(self.g.edges) < 2:
            return 0.0

        # Reshape centers to (n_nodes, 2)
        coords = centers.reshape(-1, 2)

        # Get all edge endpoints
        edge_starts = coords[self.edge_indices[:, 0]]  # (m, 2)
        edge_ends = coords[self.edge_indices[:, 1]]    # (m, 2)

        # Get pairs of edges
        i_idx, j_idx = self.i_indices, self.j_indices

        # Edge i: A→B, Edge j: C→D
        ax, ay = edge_starts[i_idx, 0], edge_starts[i_idx, 1]
        bx, by = edge_ends[i_idx, 0], edge_ends[i_idx, 1]
        cx, cy = edge_starts[j_idx, 0], edge_starts[j_idx, 1]
        dx, dy = edge_ends[j_idx, 0], edge_ends[j_idx, 1]

        # Vectorized crossing detection
        crosses = _segment_cross_vectorized(ax, ay, bx, by, cx, cy, dx, dy)

        if not np.any(crosses):
            return 0.0

        # Compute penalty only for crossing pairs
        penalties = _crossing_penalty_vectorized(
            ax[crosses], ay[crosses], bx[crosses], by[crosses],
            cx[crosses], cy[crosses], dx[crosses], dy[crosses]
        )

        # Final penalty: 0.5 * penalty^2 (quadratic)
        return float(np.sum(0.5 * penalties ** 2))
