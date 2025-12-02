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
    Check if segment AB intersects rectangle centered at C with size (w, h)
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


def segment_rectangle_penetration(
    ax: int, ay: int,
    bx: int, by: int,
    cx: int, cy: int,
    w: int, h: int
) -> float:
    """
    A segment from A to B, rectangle centered at C with width w, height h.
    Calculates the distance in which the center has to be moved to avoid intersection,
    using the half-diagonal as a reference.
    """
    # Check if segment intersects rectangle
    if not segment_intersects_rect(ax, ay, bx, by, cx, cy, w, h):
        return 0.0
    
    # Penetration: distance from center to edge, inverted
    d = point_to_segment_distance(ax, ay, bx, by, cx, cy)
    half_diag = float((w**2 + h**2) ** 0.5 / 2)
    
    penetration = half_diag - d
    return penetration


def edge_to_segment(
    centers: np.ndarray,
    e: tuple[str, str],
    g: Graph
) -> tuple[int, int, int, int]:
    """
    Convert graph edge to a segment based on node centers.
    
    Args:
        centers: Array of node center positions [cx0, cy0, cx1, cy1, ...]
        g: Graph object containing nodes and edges.
    
    Returns:
        List of segments as tuples (ax, ay, bx, by)
    """
    src_name, dst_name = e
    src_idx = g.name2idx[src_name]
    dst_idx = g.name2idx[dst_name]
    ax, ay = centers[2 * src_idx], centers[2 * src_idx + 1]
    bx, by = centers[2 * dst_idx], centers[2 * dst_idx + 1]
    return (ax, ay, bx, by)


class NodeEdge(Penalty):
    def __init__(
        self,
        g: Graph,
        w: float = 1.0,
    ):
        super().__init__(g, w)
    
    def compute(self, centers: np.ndarray) -> float:
        total_penalty = 0.0
        for i, node in enumerate(self.g.nodes):
            for e in self.g.edges:
                if node.name in e:
                    continue  # Skip edges connected to this node
                cx, cy = centers[2 * i], centers[2 * i + 1]
                w, h = node.width, node.height
                ax, ay, bx, by = edge_to_segment(centers, e, self.g)
                penetration = segment_rectangle_penetration(
                    ax, ay, bx, by,
                    cx, cy,
                    w, h
                )
                total_penalty += 1/2 * penetration**2  # Quadratic penalty
        return total_penalty