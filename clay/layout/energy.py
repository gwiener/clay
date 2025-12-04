import numpy as np
from scipy.optimize import basinhopping, minimize
from tqdm import tqdm

from clay import graph
from clay.layout import LayoutEngine, Result, compute_variable_limits, init_random
from clay.penalties.area import Area
from clay.penalties.edge_cross import EgdeCross
from clay.penalties.node_edge import NodeEdge
from clay.penalties.spacing import Spacing


class _EnergyFunction:
    """Internal class for computing energy during optimization."""

    def __init__(
        self,
        g: graph.Graph,
        progress: bool = False,
        total_iters: int = 100
    ):
        self.g = g
        self.penalties = [
            Spacing(g),
            NodeEdge(g, w=2.0),
            EgdeCross(g, w=0.5),
            Area(g, w=0.5)
        ]
        self.history: list[dict[str, float]] = []
        self.pbar: tqdm | None = None
        if progress:
            self.pbar = tqdm(total=total_iters, desc="Optimizing")

    def compute(self, centers: np.ndarray) -> float:
        return sum(penalty(centers) for penalty in self.penalties)

    def callback(self, centers: np.ndarray) -> None:
        """Callback for L-BFGS-B optimizer."""
        record = {p.__class__.__name__: p(centers) for p in self.penalties}
        total = sum(record.values())
        record['Total'] = total
        self.history.append(record)
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix(energy=f"{total:.1f}")

    def callback_bh(self, centers: np.ndarray, f: float, accept: bool) -> None:
        """Callback for basinhopping (different signature)."""
        self.history.append({"energy": f, "accepted": accept})
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix(energy=f"{f:.1f}", accepted=accept)

    def close(self) -> None:
        """Close progress bar if open."""
        if self.pbar:
            self.pbar.close()


class Energy(LayoutEngine):
    """Energy-based layout engine using optimization."""

    def __init__(
        self,
        max_iter: int = 2000,
        ftol: float = 1e-6,
        init_layout: graph.Layout | None = None,
        optimizer: str = "basinhopping",
        progress: bool = False
    ):
        self.max_iter = max_iter
        self.ftol = ftol
        self.init_layout = init_layout
        self.optimizer = optimizer
        self.progress = progress

    def fit(self, g: graph.Graph) -> Result:
        """
        Optimize the layout of the graph using energy minimization.

        Args:
            g: Graph object containing nodes and edges.

        Returns:
            A Result object with optimized center positions for each node.
        """
        if self.init_layout is not None:
            g = self.init_layout.graph
        limits = compute_variable_limits(g)
        if self.init_layout is not None:
            x0 = self.init_layout.centers
        else:
            x0 = init_random(g, limits)

        # Determine total iterations for progress bar
        match self.optimizer:
            case "basinhopping":
                total_iters = 100  # niter
            case "L-BFGS-B":
                total_iters = self.max_iter
            case _:
                total_iters = self.max_iter

        energy_func = _EnergyFunction(g, progress=self.progress, total_iters=total_iters)

        try:
            match self.optimizer:
                case "L-BFGS-B":
                    opt_result = minimize(
                        energy_func.compute,
                        x0=np.array(x0),
                        bounds=limits,
                        method='L-BFGS-B',
                        options={'maxiter': self.max_iter, 'ftol': self.ftol},
                        callback=energy_func.callback
                    )
                case "basinhopping":
                    opt_result = basinhopping(
                        energy_func.compute,
                        x0=np.array(x0),
                        niter=100,
                        T=1.0,
                        stepsize=50,
                        minimizer_kwargs={
                            'method': 'L-BFGS-B',
                            'bounds': limits,
                            'options': {'maxiter': self.max_iter, 'ftol': self.ftol}
                        },
                        callback=energy_func.callback_bh
                    )
                case _:
                    raise ValueError(f"Unknown optimizer: {self.optimizer}")
        finally:
            energy_func.close()

        optimized_centers = opt_result.x.tolist()
        layout = graph.Layout(g, optimized_centers)

        return Result(
            layout,
            metadata={
                "optimization_result": opt_result,
                "history": energy_func.history
            }
        )
