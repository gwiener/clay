import numpy as np

from clay.penalties import Graph, Penalty


class Area(Penalty):
    def __init__(
        self,
        g: Graph,
        w: float = 1.0,
        ideal_area_factor: float = 1.15,
    ):
        super().__init__(g, w)
        self.ideal_area_factor = ideal_area_factor
        total_nodes_area = sum(node.width * node.height for node in g.nodes)
        ideal_area = total_nodes_area * self.ideal_area_factor
        self.ideal_dim = np.sqrt(ideal_area)

        # Pre-compute half dimensions
        self.half_widths = [node.width / 2 for node in g.nodes]
        self.half_heights = [node.height / 2 for node in g.nodes]
 

    def compute(self, centers: np.ndarray) -> float:
        """
        Area penalty to encourage compact layouts.
        """
        n = len(self.g.nodes)
        min_x = min(centers[2*i] - self.half_widths[i] for i in range(n))
        max_x = max(centers[2*i] + self.half_widths[i] for i in range(n))
        min_y = min(centers[2*i + 1] - self.half_heights[i] for i in range(n))
        max_y = max(centers[2*i + 1] + self.half_heights[i] for i in range(n))
        actual_width = max_x - min_x
        actual_height = max_y - min_y
        delta_w = max(0, actual_width - self.ideal_dim)
        delta_h = max(0, actual_height - self.ideal_dim)
        diag_sq = delta_w ** 2 + delta_h ** 2
        return np.sqrt(diag_sq)
    
