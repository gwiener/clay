from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from clay.graph import Graph


@dataclass
class LocalEnergies:
    """Container for per-element energy contributions.

    The energies and keys arrays are matched by position:
    energies[i] corresponds to keys[i].
    """
    energies: np.ndarray
    keys: list  # list of key objects with __str__


class Penalty(ABC):
    """Base class for all penalty functions."""

    def __init__(self, g: Graph, w: float = 1.0):
        self.g = g
        self.w = w

    @abstractmethod
    def compute(self, centers: np.ndarray) -> float:
        """Compute unweighted penalty energy."""
        ...

    def compute_contributions(self, centers: np.ndarray) -> dict:
        """Return per-element energy contributions.

        Override in subclasses to provide detailed breakdown.
        Keys are objects with __str__ for formatting.
        Values are unweighted energy contributions.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support contribution analysis")

    def __call__(self, centers: np.ndarray) -> float:
        return self.w * self.compute(centers)


class LocalPenalty(Penalty):
    """Base class for penalties that are sums of per-element energies.

    Subclasses implement compute_local_energies() which returns both
    the per-element energies and their keys. The base class provides
    compute() and compute_contributions() implementations.
    """

    @abstractmethod
    def compute_local_energies(self, centers: np.ndarray) -> LocalEnergies:
        """Compute per-element energies.

        Returns:
            LocalEnergies with energies array and corresponding keys.
        """
        ...

    def compute(self, centers: np.ndarray) -> float:
        result = self.compute_local_energies(centers)
        return float(np.sum(result.energies))

    def compute_contributions(self, centers: np.ndarray) -> dict:
        result = self.compute_local_energies(centers)
        return {k: float(e) for k, e in zip(result.keys, result.energies, strict=True) if e > 0}
