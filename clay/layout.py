import random
from abc import abstractmethod

import numpy as np
from scipy.optimize import OptimizeResult

from clay import graph


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


def signed_distance(cx1, cy1, w1, h1, cx2, cy2, w2, h2):
    dx_gap = abs(cx2 - cx1) - (w1 + w2) / 2
    dy_gap = abs(cy2 - cy1) - (h1 + h2) / 2
    
    if dx_gap >= 0 or dy_gap >= 0:
        # Separated: boundary distance
        return (max(0, dx_gap)**2 + max(0, dy_gap)**2) ** 0.5
    else:
        # Overlapping: negative penetration
        return -(dx_gap**2 + dy_gap**2) ** 0.5


class Penalty(object):
    def __init__(
        self,
        g: graph.Graph,
        w: float = 1.0,
    ):
        self.g = g
        self.w = w
    
    @abstractmethod
    def compute(self, centers: np.ndarray) -> float:
        pass

    def __call__(self, centers: np.ndarray) -> float:
        return self.w * self.compute(centers)


class Spacing(Penalty):
    def __init__(
        self,
        g: graph.Graph,
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
                d = signed_distance(cx1, cy1, w1, h1, cx2, cy2, w2, h2)
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


class Area(Penalty):
    def __init__(
        self,
        g: graph.Graph,
        w: float = 1.0,
        ideal_area_factor: float = 1.15,
    ):
        super().__init__(g, w)
        self.ideal_area_factor = ideal_area_factor
        total_nodes_area = sum(node.width * node.height for node in g.nodes)
        ideal_area = total_nodes_area * self.ideal_area_factor
        self.ideal_dim = np.sqrt(ideal_area)
 

    def compute(self, centers: np.ndarray) -> float:
        """
        Area penalty to encourage compact layouts.
        """
        min_x = min(centers[2*i] - self.g.nodes[i].width / 2 for i in range(len(self.g.nodes)))
        max_x = max(centers[2*i] + self.g.nodes[i].width / 2 for i in range(len(self.g.nodes)))
        min_y = min(centers[2*i + 1] - self.g.nodes[i].height / 2 for i in range(len(self.g.nodes)))
        max_y = max(centers[2*i + 1] + self.g.nodes[i].height / 2 for i in range(len(self.g.nodes)))
        actual_width = max_x - min_x
        actual_height = max_y - min_y
        delta_w = max(0, actual_width - self.ideal_dim)
        delta_h = max(0, actual_height - self.ideal_dim)
        diag_sq = delta_w ** 2 + delta_h ** 2
        return np.sqrt(diag_sq)
    

class Energy(object):
    def __init__(
        self,
        g: graph.Graph
    ):
        self.g = g
        self.penalties = [
            Spacing(g),
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