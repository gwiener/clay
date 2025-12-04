import numpy as np

from clay.graph import Graph
from clay.penalties import Penalty


def segment_intersects_rect(
    ax: int, ay: int,
    bx: int, by: int,
    cx: int, cy: int,
    w: int, h: int
) -> bool:
    """
    Check if segment AB intersects rectangle centered at C with size (w, h).
    Scalar version for tests.
    """

    hw, hh = w / 2, h / 2

    # Rectangle bounds
    left, right = cx - hw, cx + hw
    bottom, top = cy - hh, cy + hh

    # Check if either endpoint is inside
    def inside(px, py):
        return left <= px <= right and bottom <= py <= top

    if inside(ax, ay) or inside(bx, by):
        return True

    # Direction vector
    dx, dy = bx - ax, by - ay

    # Liang-Barsky clipping
    t_min, t_max = 0.0, 1.0

    for p, q in [
        (-dx, ax - left),    # left
        (dx, right - ax),    # right
        (-dy, ay - bottom),  # bottom
        (dy, top - ay),      # top
    ]:
        if p == 0:
            # Parallel to this edge
            if q < 0:
                return False  # Outside and parallel
        else:
            t = q / p
            if p < 0:
                t_min = max(t_min, t)  # Entering
            else:
                t_max = min(t_max, t)  # Exiting

        if t_min > t_max:
            return False

    return True


def segment_rectangle_penetration(
    ax: int, ay: int,
    bx: int, by: int,
    cx: int, cy: int,
    w: int, h: int
) -> float:
    """
    Scalar version for tests.
    A segment from A to B, rectangle centered at C with width w, height h.
    Calculates the distance in which the center has to be moved to avoid intersection.
    """
    from clay.geometry import point_to_segment_distance

    if not segment_intersects_rect(ax, ay, bx, by, cx, cy, w, h):
        return 0.0

    d = point_to_segment_distance(ax, ay, bx, by, cx, cy)
    half_diag = float((w**2 + h**2) ** 0.5 / 2)

    return half_diag - d


# =============================================================================
# Vectorized helper functions
# =============================================================================

def _segment_intersects_rect_vectorized(
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
    cx: np.ndarray, cy: np.ndarray,
    hw: np.ndarray, hh: np.ndarray
) -> np.ndarray:
    """
    Vectorized segment-rectangle intersection using Liang-Barsky clipping.

    Args:
        ax, ay, bx, by: Segment endpoints (arrays of shape (n_pairs,))
        cx, cy: Rectangle centers (arrays of shape (n_pairs,))
        hw, hh: Half-widths and half-heights (arrays of shape (n_pairs,))

    Returns:
        Boolean array of shape (n_pairs,) indicating intersection.
    """
    # Rectangle bounds
    left = cx - hw
    right = cx + hw
    bottom = cy - hh
    top = cy + hh

    # Check if either endpoint is inside
    a_inside = (left <= ax) & (ax <= right) & (bottom <= ay) & (ay <= top)
    b_inside = (left <= bx) & (bx <= right) & (bottom <= by) & (by <= top)
    endpoint_inside = a_inside | b_inside

    # Direction vector
    dx = bx - ax
    dy = by - ay

    # Liang-Barsky: compute t_min, t_max for each edge
    # Initialize
    t_min = np.zeros_like(ax)
    t_max = np.ones_like(ax)

    # Process each of the 4 edges
    edges = [
        (-dx, ax - left),    # left
        (dx, right - ax),    # right
        (-dy, ay - bottom),  # bottom
        (dy, top - ay),      # top
    ]

    for p, q in edges:
        # Parallel case: p == 0
        parallel = np.abs(p) < 1e-10
        outside_parallel = parallel & (q < 0)

        # Non-parallel case
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(parallel, 0.0, q / p)

        # Update t_min (entering, p < 0) or t_max (exiting, p > 0)
        entering = (p < 0) & ~parallel
        exiting = (p > 0) & ~parallel

        t_min = np.where(entering, np.maximum(t_min, t), t_min)
        t_max = np.where(exiting, np.minimum(t_max, t), t_max)

        # Mark as non-intersecting if outside and parallel
        t_min = np.where(outside_parallel, 2.0, t_min)  # Force t_min > t_max

    # Intersection if t_min <= t_max (and not outside_parallel)
    liang_barsky_intersects = t_min <= t_max

    return endpoint_inside | liang_barsky_intersects


def _point_to_segment_distance_vectorized(
    seg_ax: np.ndarray, seg_ay: np.ndarray,
    seg_bx: np.ndarray, seg_by: np.ndarray,
    px: np.ndarray, py: np.ndarray
) -> np.ndarray:
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


class NodeEdge(Penalty):
    def __init__(
        self,
        g: Graph,
        w: float = 1.0,
    ):
        super().__init__(g, w)

        n = len(g.nodes)
        m = len(g.edges)

        # Pre-compute node dimensions
        self.widths = np.array([node.width for node in g.nodes], dtype=np.float64)
        self.heights = np.array([node.height for node in g.nodes], dtype=np.float64)
        self.half_widths = self.widths / 2
        self.half_heights = self.heights / 2
        self.half_diags = np.sqrt(self.widths**2 + self.heights**2) / 2

        # Pre-compute edge endpoint indices: shape (m, 2)
        if m > 0:
            self.edge_indices = np.array([
                (g.name2idx[src], g.name2idx[dst]) for src, dst in g.edges
            ], dtype=np.int32)
        else:
            self.edge_indices = np.empty((0, 2), dtype=np.int32)

        # Pre-compute valid (node_idx, edge_idx) pairs
        # where node is NOT an endpoint of the edge
        if n > 0 and m > 0:
            # Create all (node, edge) pairs
            node_indices = np.arange(n)
            edge_indices = np.arange(m)

            # Meshgrid to get all pairs
            node_idx_grid, edge_idx_grid = np.meshgrid(node_indices, edge_indices, indexing='ij')
            node_idx_flat = node_idx_grid.ravel()
            edge_idx_flat = edge_idx_grid.ravel()

            # Get edge endpoints for each pair
            edge_src = self.edge_indices[edge_idx_flat, 0]
            edge_dst = self.edge_indices[edge_idx_flat, 1]

            # Valid pairs: node is not src or dst of edge
            valid_mask = (node_idx_flat != edge_src) & (node_idx_flat != edge_dst)

            self.valid_node_indices = node_idx_flat[valid_mask]
            self.valid_edge_indices = edge_idx_flat[valid_mask]
        else:
            self.valid_node_indices = np.array([], dtype=np.int32)
            self.valid_edge_indices = np.array([], dtype=np.int32)

    def compute(self, centers: np.ndarray) -> float:
        if len(self.valid_node_indices) == 0:
            return 0.0

        # Reshape centers to (n_nodes, 2)
        coords = centers.reshape(-1, 2)

        # Get node centers for valid pairs
        cx = coords[self.valid_node_indices, 0]
        cy = coords[self.valid_node_indices, 1]
        hw = self.half_widths[self.valid_node_indices]
        hh = self.half_heights[self.valid_node_indices]
        half_diag = self.half_diags[self.valid_node_indices]

        # Get edge endpoints for valid pairs
        edge_src = self.edge_indices[self.valid_edge_indices, 0]
        edge_dst = self.edge_indices[self.valid_edge_indices, 1]
        ax = coords[edge_src, 0]
        ay = coords[edge_src, 1]
        bx = coords[edge_dst, 0]
        by = coords[edge_dst, 1]

        # Vectorized intersection test
        intersects = _segment_intersects_rect_vectorized(ax, ay, bx, by, cx, cy, hw, hh)

        if not np.any(intersects):
            return 0.0

        # Compute penetration only for intersecting pairs
        ax_int = ax[intersects]
        ay_int = ay[intersects]
        bx_int = bx[intersects]
        by_int = by[intersects]
        cx_int = cx[intersects]
        cy_int = cy[intersects]
        half_diag_int = half_diag[intersects]

        # Distance from rectangle center to segment
        dist = _point_to_segment_distance_vectorized(
            ax_int, ay_int, bx_int, by_int, cx_int, cy_int
        )

        # Penetration = half_diagonal - distance
        penetration = half_diag_int - dist

        # Quadratic penalty: 0.5 * penetration^2
        return float(np.sum(0.5 * penetration**2))
