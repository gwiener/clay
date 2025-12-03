import numpy as np
from scipy.optimize import minimize

from clay import graph
from clay.layout import LayoutEngine, Result, compute_variable_limits, init_random
from clay.penalties.area import Area
from clay.penalties.edge_cross import EgdeCross
from clay.penalties.node_edge import NodeEdge
from clay.penalties.spacing import Spacing


class _EnergyFunction:
    """Internal class for computing energy during optimization."""

    def __init__(self, g: graph.Graph):
        self.g = g
        self.penalties = [
            Spacing(g),
            NodeEdge(g, w=2.0),
            EgdeCross(g, w=0.5),
            Area(g)
        ]
        self.history: list[dict[str, float]] = []

    def compute(self, centers: np.ndarray) -> float:
        return sum(penalty(centers) for penalty in self.penalties)

    def callback(self, centers: np.ndarray) -> None:
        record = {p.__class__.__name__: p(centers) for p in self.penalties}
        total = sum(record.values())
        record['Total'] = total
        self.history.append(record)


class Energy(LayoutEngine):
    """Energy-based layout engine using optimization."""

    def __init__(self, max_iter: int = 2000, ftol: float = 1e-6):
        self.max_iter = max_iter
        self.ftol = ftol

    def fit(self, g: graph.Graph) -> Result:
        """
        Optimize the layout of the graph using energy minimization.

        Args:
            g: Graph object containing nodes and edges.

        Returns:
            A Result object with optimized center positions for each node.
        """
        limits = compute_variable_limits(g)
        x0 = init_random(g, limits)

        energy_func = _EnergyFunction(g)
        opt_result = minimize(
            energy_func.compute,
            x0=np.array(x0),
            args=(),
            bounds=limits,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'ftol': self.ftol},
            callback=energy_func.callback
        )

        optimized_centers = opt_result.x.tolist()
        layout = graph.Layout(g, optimized_centers)

        return Result(
            layout,
            metadata={
                "optimization_result": opt_result,
                "history": energy_func.history
            }
        )
