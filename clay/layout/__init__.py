import random as random_module
from abc import abstractmethod

from clay import graph


class LayoutEngine:
    """Base class for layout engines."""

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


def compute_variable_limits(g: graph.Graph) -> list[tuple[int, int]]:
    """
    Compute variable limits for a graph layout.

    Args:
        g: Graph object containing nodes.

    Returns:
        A list of tuples representing variable limits [(x_min, x_max), (y_min, y_max), ...] for each node.
    """
    limits = []
    for node in g.nodes:
        limits.extend([
            (node.width / 2, g.canvas.width - node.width / 2),
            (node.height / 2, g.canvas.height - node.height / 2)
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


# Import engines after base class to avoid circular imports
from clay.layout.random import Random
from clay.layout.energy import Energy
from clay.layout.ranked import Ranked

# Registry for CLI lookup
ENGINES: dict[str, type[LayoutEngine]] = {
    "random": Random,
    "energy": Energy,
    "ranked": Ranked,
}


def get_engine(name: str) -> type[LayoutEngine]:
    """Get engine class by name."""
    if name not in ENGINES:
        raise ValueError(f"Unknown layout engine: {name}. Available: {list(ENGINES.keys())}")
    return ENGINES[name]


__all__ = [
    "LayoutEngine",
    "Result",
    "Random",
    "Energy",
    "Ranked",
    "ENGINES",
    "get_engine",
    "compute_variable_limits",
    "init_random",
]
