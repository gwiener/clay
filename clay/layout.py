import random

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


def boundary_distance(cx1, cy1, w1, h1, cx2, cy2, w2, h2):
    """Distance between rectangle boundaries (0 if overlapping)"""
    dx = max(0, abs(cx2 - cx1) - (w1 + w2) / 2)
    dy = max(0, abs(cy2 - cy1) - (h1 + h2) / 2)
    return (dx**2 + dy**2) ** 0.5


def spacing_penalty(
    centers: np.ndarray,
    g: graph.Graph,
    D: int = 50,
    k_edge: float = 1.0,
    k_repel: float = 10.0
) -> float:
    """
    positions: flat array [x0, y0, x1, y1, ...]
    sizes: list of (width, height) per node
    edges: set of (i, j) tuples for connected pairs
    D: desired minimum distance between boundaries
    """
    n_nodes = len(g.nodes)
    energy = 0.0
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            cx1, cy1 = centers[2*i], centers[2*i + 1]
            cx2, cy2 = centers[2*j], centers[2*j + 1]
            w1, h1 = g.nodes[i].width, g.nodes[i].height
            w2, h2 = g.nodes[j].width, g.nodes[j].height
            
            d = boundary_distance(cx1, cy1, w1, h1, cx2, cy2, w2, h2)
            delta = d - D
            name_i, name_j = g.nodes[i].name, g.nodes[j].name
            is_edge = (name_i, name_j) in g.edges or (name_j, name_i) in g.edges
            added_energy = 0.0
            if is_edge:
                added_energy = 0.5 * k_edge * delta ** 2
            elif delta < 0:
                added_energy = 0.5 * k_repel * delta ** 2
            energy += added_energy
    
    return energy


def area_penalty(
    centers: np.ndarray,
    g: graph.Graph,
    ideal_area_factor: float = 1.15
) -> float:
    """
    Area penalty to encourage compact layouts.
    """
    total_nodes_area = sum(node.width * node.height for node in g.nodes)
    ideal_area = total_nodes_area * ideal_area_factor
    ideal_dim = np.sqrt(ideal_area)
    min_x = min(centers[2*i] - g.nodes[i].width / 2 for i in range(len(g.nodes)))
    max_x = max(centers[2*i] + g.nodes[i].width / 2 for i in range(len(g.nodes)))
    min_y = min(centers[2*i + 1] - g.nodes[i].height / 2 for i in range(len(g.nodes)))
    max_y = max(centers[2*i + 1] + g.nodes[i].height / 2 for i in range(len(g.nodes)))
    actual_width = max_x - min_x
    actual_height = max_y - min_y
    delta_w = max(0, actual_width - ideal_dim)
    delta_h = max(0, actual_height - ideal_dim)
    diag_sq = delta_w ** 2 + delta_h ** 2
    return np.sqrt(diag_sq)
    

def energy(
    centers: np.ndarray,
    g: graph.Graph,
) -> float:
    """
    Energy function for graph layout optimization.
    
    Args:
        centers: Flat array of node center positions [x0, y0, x1, y1, ...].
        g: Graph object containing nodes and edges.
    
    Returns:
        A float representing the total energy of the layout.
    """
    return spacing_penalty(centers, g) + 0.5*area_penalty(centers, g)


class Result(object):
    def __init__(
        self,
        layout: graph.Layout,
        optimization_result: OptimizeResult
    ):
        self.layout = layout
        self.optimization_result = optimization_result


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
    
    result = minimize(
        energy,
        x0=np.array(x0),
        args=(g,),
        bounds=limits,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'ftol': 1e-6}
    )
    
    optimized_centers = result.x.tolist()
    layout = graph.Layout(g, optimized_centers)
    return Result(layout, result)