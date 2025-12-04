import random as random_module
from abc import abstractmethod

from clay import graph


class LayoutEngine:
    """Base class for layout engines."""

    def compute_variable_limits(self, g: graph.Graph) -> list[tuple[float, float]]:
        """Compute position bounds for each node, using graph's padding."""
        return compute_variable_limits(g, padding=g.padding)

    @abstractmethod
    def fit(self, g: graph.Graph) -> "Result":
        """Compute layout for the given graph."""
        pass


class Result:
    """Layout result container."""

    def __init__(
        self,
        layout: graph.Layout,
        metadata: dict | None = None
    ):
        self.layout = layout
        self.metadata = metadata or {}


def compute_variable_limits(g: graph.Graph, padding: int = 0) -> list[tuple[float, float]]:
    """
    Compute variable limits for a graph layout.

    Args:
        g: Graph object containing nodes.
        padding: Margin from canvas edges where nodes cannot be placed.

    Returns:
        A list of tuples representing variable limits [(x_min, x_max), (y_min, y_max), ...] for each node.
    """
    limits = []
    for node in g.nodes:
        limits.extend([
            (node.width / 2 + padding, g.canvas.width - node.width / 2 - padding),
            (node.height / 2 + padding, g.canvas.height - node.height / 2 - padding)
        ])
    return limits


def init_random(
    g: graph.Graph,
    limits: list[tuple[int, int]]
) -> list[float]:
    """Initialize random center positions within given limits."""
    centers = []
    for i in range(len(g.nodes)):
        x_min, x_max = limits[i * 2]
        y_min, y_max = limits[i * 2 + 1]
        cx = random_module.uniform(x_min, x_max)
        cy = random_module.uniform(y_min, y_max)
        centers.extend([cx, cy])
    return centers


def init_from_ranked(g: graph.Graph, direction: str = "TB") -> list[float]:
    """
    Initialize layout using Sugiyama-style ranked algorithm.

    Args:
        g: Graph object containing nodes and edges.
        direction: Layout direction - "TB" (top-to-bottom) or "LR" (left-to-right).

    Returns:
        A flat list of center coordinates [x0, y0, x1, y1, ...].
    """
    from clay.layout.ranked import Ranked
    result = Ranked(direction=direction).fit(g)
    return result.layout.centers


__all__ = [
    "LayoutEngine",
    "Result",
    "compute_variable_limits",
    "init_random",
    "init_from_ranked",
]
