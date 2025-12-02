import random

import numpy as np
from scipy.optimize import OptimizeResult

from clay import graph
from clay.penalties.area import Area
from clay.penalties.node_edge import NodeEdge
from clay.penalties.spacing import Spacing


def compute_variable_limits(g: graph.Graph) -> list[tuple[int, int]]:
    """
    Compute variable limits for a graph layout.
    
    Args:
        graph: Graph object containing nodes.
    
    Returns:
        A list of integers representing variable limits [x_min0, y_min0, x_max0, y_max0, ...] for a layout algorithm.
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
    centers = []
    for i in range(len(g.nodes)):
        x_min, x_max = limits[i * 2]
        y_min, y_max = limits[i * 2 + 1]
        cx = random.uniform(x_min, x_max)
        cy = random.uniform(y_min, y_max)
        centers.extend([cx, cy])
    return centers


def random_layout(g: graph.Graph) -> graph.Layout:
    """
    Generate a random layout for the given graph.
    
    Args:
        graph: Graph object containing nodes.
    
    Returns:
        A Layout object with random center positions for each node.
    """
    limits = compute_variable_limits(g)
    centers = init_random(g, limits)
    return graph.Layout(g, centers)


class Energy(object):
    def __init__(
        self,
        g: graph.Graph
    ):
        self.g = g
        self.penalties = [
            Spacing(g),
            NodeEdge(g, w=2.0),
            Area(g, w=0.5)
        ]
        self.history = []

    def compute(self, centers: np.ndarray) -> float:
        return sum(penalty(centers) for penalty in self.penalties)
    
    def callback(self, centers: np.ndarray):
        record = {p.__class__.__name__: p(centers) for p in self.penalties}
        total = sum(record.values())
        record['Total'] = total
        self.history.append(record)


class Result(object):
    def __init__(
        self,
        layout: graph.Layout,
        optimization_result: OptimizeResult,
        history: list[dict[str, float]]
    ):
        self.layout = layout
        self.optimization_result = optimization_result
        self.history = history


def fit(g: graph.Graph) -> Result:
    """
    Optimize the layout of the graph using energy minimization.
    
    Args:
        g: Graph object containing nodes and edges.
    
    Returns:
        A Layout object with optimized center positions for each node.
    """
    from scipy.optimize import minimize

    limits = compute_variable_limits(g)
    x0 = init_random(g, limits)
    
    energy = Energy(g)
    result = minimize(
        energy.compute,
        x0=np.array(x0),
        args=(),
        bounds=limits,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'ftol': 1e-6},
        callback=energy.callback
    )
    
    optimized_centers = result.x.tolist()
    layout = graph.Layout(g, optimized_centers)
    return Result(layout, result, energy.history)