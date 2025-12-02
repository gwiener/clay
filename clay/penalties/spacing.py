from clay.penalties import Penalty, Graph
import numpy as np


def _signed_distance(cx1, cy1, w1, h1, cx2, cy2, w2, h2):
    dx_gap = abs(cx2 - cx1) - (w1 + w2) / 2
    dy_gap = abs(cy2 - cy1) - (h1 + h2) / 2
    
    if dx_gap >= 0 or dy_gap >= 0:
        # Separated: boundary distance
        return (max(0, dx_gap)**2 + max(0, dy_gap)**2) ** 0.5
    else:
        # Overlapping: negative penetration
        return -(dx_gap**2 + dy_gap**2) ** 0.5


class Spacing(Penalty):
    def __init__(
        self,
        g: Graph,
        w: float = 1.0,
        D: int = 50,
        k_edge: float = 1.0,
        k_repel: float = 10.0
    ):
        super().__init__(g, w)
        self.D = D
        self.k_edge = k_edge
        self.k_repel = k_repel

    def compute(self, centers: np.ndarray) -> float:
        n_nodes = len(self.g.nodes)
        energy = 0.0
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                cx1, cy1 = centers[2*i], centers[2*i + 1]
                cx2, cy2 = centers[2*j], centers[2*j + 1]
                w1, h1 = self.g.nodes[i].width, self.g.nodes[i].height
                w2, h2 = self.g.nodes[j].width, self.g.nodes[j].height
                d = _signed_distance(cx1, cy1, w1, h1, cx2, cy2, w2, h2)
                delta = d - self.D
                name_i, name_j = self.g.nodes[i].name, self.g.nodes[j].name
                is_edge = (name_i, name_j) in self.g.edges or (name_j, name_i) in self.g.edges
                added_energy = 0.0
                if is_edge:
                    added_energy = 0.5 * self.k_edge * delta ** 2
                elif delta < 0:
                    added_energy = 0.5 * self.k_repel * delta ** 2
                energy += added_energy
        return energy
