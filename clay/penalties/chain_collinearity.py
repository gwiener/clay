import numpy as np

from clay.graph import Graph
from clay.penalties import Penalty


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
    seg_vx = seg_bx - seg_ax
    seg_vy = seg_by - seg_ay
    seg_len_sq = seg_vx * seg_vx + seg_vy * seg_vy

    to_point_x = px - seg_ax
    to_point_y = py - seg_ay

    with np.errstate(divide='ignore', invalid='ignore'):
        t = (to_point_x * seg_vx + to_point_y * seg_vy) / seg_len_sq
        t = np.where(seg_len_sq < 1e-10, 0.0, t)

    t = np.clip(t, 0.0, 1.0)

    closest_x = seg_ax + t * seg_vx
    closest_y = seg_ay + t * seg_vy

    dx = px - closest_x
    dy = py - closest_y

    return np.sqrt(dx * dx + dy * dy)


class ChainCollinearity(Penalty):
    """
    Penalty for non-collinear 3-node chains.

    For every chain A->B->C in the graph (where edges A->B and B->C both exist),
    penalizes the distance of B from the line segment connecting A and C.
    This encourages paths to be visually straight.
    """

    def __init__(self, g: Graph, w: float = 1.0):
        super().__init__(g, w)

        # Enumerate all (A, B, C) chains where A->B and B->C edges exist
        chains: list[tuple[int, int, int]] = []

        for b_name in g.name2idx:
            predecessors = g.incoming[b_name]
            successors = g.outgoing[b_name]

            for a_name in predecessors:
                for c_name in successors:
                    # Avoid degenerate A == C case
                    if a_name != c_name:
                        chains.append((
                            g.name2idx[a_name],
                            g.name2idx[b_name],
                            g.name2idx[c_name]
                        ))

        # Pre-compute index arrays for vectorized access
        if chains:
            chains_arr = np.array(chains, dtype=np.int32)
            self.a_indices = chains_arr[:, 0]
            self.b_indices = chains_arr[:, 1]
            self.c_indices = chains_arr[:, 2]
            self.n_chains = len(chains)
        else:
            self.a_indices = np.array([], dtype=np.int32)
            self.b_indices = np.array([], dtype=np.int32)
            self.c_indices = np.array([], dtype=np.int32)
            self.n_chains = 0

    def compute(self, centers: np.ndarray) -> float:
        """
        Compute collinearity penalty for all chains.

        Returns sum of squared distances from middle nodes to
        their endpoint segments.
        """
        if self.n_chains == 0:
            return 0.0

        coords = centers.reshape(-1, 2)

        # Segment endpoints (A to C)
        ax = coords[self.a_indices, 0]
        ay = coords[self.a_indices, 1]
        cx = coords[self.c_indices, 0]
        cy = coords[self.c_indices, 1]

        # Middle point B
        bx = coords[self.b_indices, 0]
        by = coords[self.b_indices, 1]

        # Distance from B to segment AC
        distances = _point_to_segment_distance_vectorized(ax, ay, cx, cy, bx, by)

        # Quadratic penalty
        return float(0.5 * np.sum(distances ** 2))
