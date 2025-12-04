import numpy as np

from clay.penalties import Graph, Penalty


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

        # Pre-compute node dimensions
        self.widths = np.array([node.width for node in g.nodes])
        self.heights = np.array([node.height for node in g.nodes])

        # Pre-compute edge set for O(1) lookup
        self.edge_set = set(g.edges)

        # Pre-compute edge matrix (symmetric)
        n = len(g.nodes)
        self.edge_matrix = np.zeros((n, n), dtype=bool)
        for src, dst in g.edges:
            i, j = g.name2idx[src], g.name2idx[dst]
            self.edge_matrix[i, j] = True
            self.edge_matrix[j, i] = True

        # Pre-compute upper triangle indices for pair iteration
        self.i_indices, self.j_indices = np.triu_indices(n, k=1)

    def compute(self, centers: np.ndarray) -> float:
        n = len(self.g.nodes)
        if n < 2:
            return 0.0

        # Reshape centers to (n, 2)
        coords = centers.reshape(-1, 2)
        cx = coords[:, 0]
        cy = coords[:, 1]

        # Get pairs
        i_idx, j_idx = self.i_indices, self.j_indices

        # Compute pairwise differences
        dx = np.abs(cx[j_idx] - cx[i_idx])
        dy = np.abs(cy[j_idx] - cy[i_idx])

        # Compute half-widths and half-heights for each pair
        half_w_sum = (self.widths[i_idx] + self.widths[j_idx]) / 2
        half_h_sum = (self.heights[i_idx] + self.heights[j_idx]) / 2

        # Compute gaps
        dx_gap = dx - half_w_sum
        dy_gap = dy - half_h_sum

        # Compute signed distance
        # Separated: sqrt(max(0, dx_gap)^2 + max(0, dy_gap)^2)
        # Overlapping: -sqrt(dx_gap^2 + dy_gap^2)
        separated = (dx_gap >= 0) | (dy_gap >= 0)

        signed_dist = np.where(
            separated,
            np.sqrt(np.maximum(0, dx_gap)**2 + np.maximum(0, dy_gap)**2),
            -np.sqrt(dx_gap**2 + dy_gap**2)
        )

        # Compute delta from desired distance
        delta = signed_dist - self.D

        # Get edge mask for pairs
        is_edge = self.edge_matrix[i_idx, j_idx]

        # Compute energy contributions
        # Connected pairs: always penalized (spring)
        # Non-connected pairs: only penalized when overlapping (delta < 0)
        edge_energy = np.where(is_edge, 0.5 * self.k_edge * delta**2, 0.0)
        repel_energy = np.where(~is_edge & (delta < 0), 0.5 * self.k_repel * delta**2, 0.0)

        return float(np.sum(edge_energy) + np.sum(repel_energy))
