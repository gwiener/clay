import numpy as np

from clay.geometry import point_to_segment_distance
from clay.penalties import Penalty


def segment_cross(
    ax: int, ay: int,
    bx: int, by: int,
    cx: int, cy: int,
    dx: int, dy: int
) -> bool:
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    
    o1 = orient(cx, cy, dx, dy, ax, ay)
    o2 = orient(cx, cy, dx, dy, bx, by)
    o3 = orient(ax, ay, bx, by, cx, cy)
    o4 = orient(ax, ay, bx, by, dx, dy)
    
    return o1 * o2 < 0 and o3 * o4 < 0


def softmin(values: list[float], sharpness=1.0) -> float:
    values = np.asarray(values)
    scale = values.max() or 1.0
    v = -sharpness * values / scale
    v_max = v.max()
    return scale * (-(v_max + np.log(np.exp(v - v_max).sum())) / sharpness)


def crossing_penalty(
    ax: int, ay: int,
    bx: int, by: int,
    cx: int, cy: int,
    dx: int, dy: int
) -> float:
    if not segment_cross(ax, ay, bx, by, cx, cy, dx, dy):
        return 0.0
    
    d1 = point_to_segment_distance(ax, ay, cx, cy, dx, dy)
    d2 = point_to_segment_distance(bx, by, cx, cy, dx, dy)
    d3 = point_to_segment_distance(cx, cy, ax, ay, bx, by)
    d4 = point_to_segment_distance(dx, dy, ax, ay, bx, by)
    
    min_dist = softmin([d1, d2, d3, d4])
    
    return 0.5 * min_dist ** 2


class EgdeCross(Penalty):
    def __init__(self, g, w=1.0):
        super().__init__(g, w)
    
    def compute(self, centers: np.ndarray) -> float:
        n_edges = len(self.g.edges)
        total_penalty = 0.0
        for i in range(n_edges):
            for j in range(i + 1, n_edges):
                (a_name, b_name) = self.g.edges[i]
                (c_name, d_name) = self.g.edges[j]
                
                a_idx = self.g.name2idx[a_name]
                b_idx = self.g.name2idx[b_name]
                c_idx = self.g.name2idx[c_name]
                d_idx = self.g.name2idx[d_name]
                
                ax, ay = centers[2 * a_idx], centers[2 * a_idx + 1]
                bx, by = centers[2 * b_idx], centers[2 * b_idx + 1]
                cx, cy = centers[2 * c_idx], centers[2 * c_idx + 1]
                dx, dy = centers[2 * d_idx], centers[2 * d_idx + 1]
                
                penetration = crossing_penalty(
                    ax, ay,
                    bx, by,
                    cx, cy,
                    dx, dy
                )
                total_penalty += 1/2 * penetration ** 2  # Quadratic penalty
        return total_penalty