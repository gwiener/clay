from dataclasses import dataclass

import numpy as np

from clay.graph import Graph
from clay.penalties import LocalPenalty, LocalEnergies


@dataclass(frozen=True)
class NodePairKey:
    """Key for node pair contributions (Spacing penalty)."""
    node1: str
    node2: str

    def __str__(self) -> str:
        return f"{self.node1} <-> {self.node2}"


class Spacing(LocalPenalty):
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

    def compute_local_energies(self, centers: np.ndarray) -> LocalEnergies:
        n = len(self.g.nodes)
        if n < 2:
            return LocalEnergies(np.array([]), [])

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
        total_energy = edge_energy + repel_energy

        # Build keys for all pairs
        keys = [
            NodePairKey(self.g.nodes[i_idx[k]].name, self.g.nodes[j_idx[k]].name)
            for k in range(len(i_idx))
        ]

        return LocalEnergies(total_energy, keys)
