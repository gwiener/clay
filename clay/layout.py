#!/usr/bin/env python3
"""
Automatic Graph Layout Engine
Implements constrained optimization for compact, orderly diagram layouts.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch


class Node:
    """Represents a graph node with label and dimensions."""

    def __init__(
        self,
        label: str,
        width: Optional[float] = None,
        height: Optional[float] = None,
        padding_x: float = 2,
        padding_y: float = 1,
        fontsize: int = 10
    ):
        self.label = label
        self.fontsize = fontsize
        self.padding_x = padding_x
        self.padding_y = padding_y

        # Calculate size if not provided
        if width is None or height is None:
            from matplotlib.textpath import TextPath
            from matplotlib.font_manager import FontProperties

            font_props = FontProperties(weight='bold', size=fontsize)
            tp = TextPath((0, 0), label, size=fontsize, prop=font_props)
            bbox = tp.get_extents()

            self.width = width if width is not None else bbox.width + 2 * padding_x
            self.height = height if height is not None else bbox.height + 2 * padding_y
        else:
            self.width = width
            self.height = height


# ============================================================================
# GEOMETRY UTILITIES
# ============================================================================

def rect_edge_point(
    rect: Tuple[float, float, float, float],
    center: Tuple[float, float],
    target: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Find point where line from center to target intersects rectangle boundary.

    Args:
        rect: (x, y, width, height) - bottom-left corner and dimensions
        center: (cx, cy) - center of rectangle
        target: (tx, ty) - target point to aim at

    Returns:
        (x, y) point on rectangle edge
    """
    x, y, w, h = rect
    cx, cy = center
    tx, ty = target

    # Direction vector
    dx, dy = tx - cx, ty - cy

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return center

    # Check intersection with each edge
    # Right edge
    if dx > 0:
        t = (x + w - cx) / dx
        py = cy + t * dy
        if y <= py <= y + h:
            return (x + w, py)

    # Left edge
    if dx < 0:
        t = (x - cx) / dx
        py = cy + t * dy
        if y <= py <= y + h:
            return (x, py)

    # Bottom edge
    if dy > 0:
        t = (y + h - cy) / dy
        px = cx + t * dx
        if x <= px <= x + w:
            return (px, y + h)

    # Top edge
    if dy < 0:
        t = (y - cy) / dy
        px = cx + t * dx
        if x <= px <= x + w:
            return (px, y)

    return center


def point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray
) -> float:
    """
    Calculate distance from a point to a line defined by two points.

    Args:
        point: (x, y) point to measure from
        line_start: (x, y) start of line
        line_end: (x, y) end of line

    Returns:
        distance (float)
    """
    AC = line_end - line_start
    AC_len = np.linalg.norm(AC)

    if AC_len < 1e-6:
        # Line start and end are same point
        return np.linalg.norm(point - line_start)

    # Unit vector along the line
    u = AC / AC_len

    # Project point onto line
    AB = point - line_start
    projection_length = np.dot(AB, u)
    point_projected = line_start + projection_length * u

    # Distance from point to its projection
    return np.linalg.norm(point - point_projected)


# ============================================================================
# ENERGY FUNCTION COMPONENTS
# ============================================================================

def overlap_penalty(positions: np.ndarray, nodes: List[Node]) -> float:
    """
    Penalize overlapping nodes (hard constraint via high weight).

    Args:
        positions: array of shape (n, 2)
        nodes: list of Node objects

    Returns:
        penalty value (float)
    """
    penalty = 0
    n = len(nodes)
    MARGIN = 10  # Minimum spacing between nodes

    for i in range(n):
        for j in range(i + 1, n):
            # Distance between centers
            dist = np.linalg.norm(positions[i] - positions[j])

            # Minimum required distance (half-widths + margin)
            min_dist_x = (nodes[i].width + nodes[j].width) / 2 + MARGIN
            min_dist_y = (nodes[i].height + nodes[j].height) / 2 + MARGIN

            # Use the smaller of the two (conservative check)
            min_dist = min(min_dist_x, min_dist_y)

            # Smooth penalty that grows as overlap increases
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2

    return penalty


def edge_length_penalty(
    positions: np.ndarray,
    edges: List[Tuple[int, int]]
) -> float:
    """
    Penalize long edges (encourages connected nodes to be close).

    Args:
        positions: array of shape (n, 2)
        edges: list of (from_idx, to_idx) tuples

    Returns:
        penalty value (float)
    """
    total = 0
    for (u, v) in edges:
        dist = np.linalg.norm(positions[v] - positions[u])
        total += dist ** 2
    return total


def straightness_penalty(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    nodes: List[Node]
) -> float:
    """
    Penalize non-straight paths through nodes.
    For any path A->B->C, penalizes if B is not collinear with A and C.

    Args:
        positions: array of shape (n, 2)
        edges: list of (from_idx, to_idx) tuples
        nodes: list of Node objects

    Returns:
        penalty value (float)
    """
    penalty = 0

    # Build adjacency lists
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for (u, v) in edges:
        outgoing[u].append(v)
        incoming[v].append(u)

    # For each node B with both incoming and outgoing edges
    for B_idx in range(len(nodes)):
        for A_idx in incoming[B_idx]:
            for C_idx in outgoing[B_idx]:
                # How far is B from the line AC?
                dist = point_to_line_distance(
                    positions[B_idx],
                    positions[A_idx],
                    positions[C_idx]
                )
                penalty += dist ** 2

    return penalty


def bounding_box_penalty(
    positions: np.ndarray,
    nodes: List[Node],
    target_bbox: Tuple[float, float]
) -> float:
    """
    Penalize if diagram exceeds target bounding box.

    Args:
        positions: array of shape (n, 2)
        nodes: list of Node objects
        target_bbox: (width, height) tuple

    Returns:
        penalty value (float)
    """
    if len(positions) == 0:
        return 0

    # Calculate actual bounding box
    min_x = min(positions[i][0] - nodes[i].width/2 for i in range(len(nodes)))
    max_x = max(positions[i][0] + nodes[i].width/2 for i in range(len(nodes)))
    min_y = min(positions[i][1] - nodes[i].height/2 for i in range(len(nodes)))
    max_y = max(positions[i][1] + nodes[i].height/2 for i in range(len(nodes)))

    actual_width = max_x - min_x
    actual_height = max_y - min_y

    # Penalize if too large
    width_penalty = max(0, actual_width - target_bbox[0]) ** 2
    height_penalty = max(0, actual_height - target_bbox[1]) ** 2

    return width_penalty + height_penalty


def area_penalty(positions: np.ndarray, nodes: List[Node]) -> float:
    """
    Penalize large bounding box area (encourages compactness).

    Args:
        positions: array of shape (n, 2)
        nodes: list of Node objects

    Returns:
        penalty value (float)
    """
    if len(positions) == 0:
        return 0

    # Calculate bounding box
    min_x = min(positions[i][0] - nodes[i].width/2 for i in range(len(nodes)))
    max_x = max(positions[i][0] + nodes[i].width/2 for i in range(len(nodes)))
    min_y = min(positions[i][1] - nodes[i].height/2 for i in range(len(nodes)))
    max_y = max(positions[i][1] + nodes[i].height/2 for i in range(len(nodes)))

    area = (max_x - min_x) * (max_y - min_y)
    return area


# ============================================================================
# MAIN ENERGY FUNCTION
# ============================================================================

def energy_function(
    positions_flat: np.ndarray,
    nodes: List[Node],
    edges: List[Tuple[int, int]],
    target_bbox: Tuple[float, float]
) -> float:
    """
    Combined energy function for optimization.

    Args:
        positions_flat: flattened array of positions (for optimizer)
        nodes: list of Node objects
        edges: list of (from_idx, to_idx) tuples
        target_bbox: (width, height) tuple

    Returns:
        total energy value (float)
    """
    positions = positions_flat.reshape(-1, 2)

    E = 0

    # Weights (tunable parameters)
    W_OVERLAP = 1000      # Hard constraint
    W_EDGE_LENGTH = 10    # Keep connected nodes close
    W_STRAIGHTNESS = 5    # Encourage straight-through paths
    W_BBOX = 100          # Stay within target box
    W_AREA = 1            # Minimize total area

    E += W_OVERLAP * overlap_penalty(positions, nodes)
    E += W_EDGE_LENGTH * edge_length_penalty(positions, edges)
    E += W_STRAIGHTNESS * straightness_penalty(positions, edges, nodes)
    E += W_BBOX * bounding_box_penalty(positions, nodes, target_bbox)
    E += W_AREA * area_penalty(positions, nodes)

    return E


# ============================================================================
# LAYOUT ENGINE
# ============================================================================

def layout_graph(
    nodes_dict: Dict[str, Node],
    edges: List[Tuple[str, str]],
    target_bbox: Tuple[float, float] = (800, 600),
    verbose: bool = True
) -> Dict[str, Tuple[float, float]]:
    """
    Layout a graph using constrained optimization.

    Args:
        nodes_dict: dict of {node_id: Node}
        edges: list of (from_id, to_id) tuples
        target_bbox: (width, height) tuple for bounding box
        verbose: print optimization progress

    Returns:
        dict of {node_id: (x, y)} positions
    """
    # Create ordered list of node IDs and index mapping
    node_ids = list(nodes_dict.keys())
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Convert edges to indices
    edges_idx = [(id_to_idx[u], id_to_idx[v]) for u, v in edges]

    # Create nodes list in same order
    nodes_list = [nodes_dict[node_id] for node_id in node_ids]

    n = len(nodes_list)

    if n == 0:
        return {}

    # Initial layout - simple grid
    grid_size = int(np.ceil(np.sqrt(n)))
    x0 = np.zeros((n, 2))
    spacing_x = target_bbox[0] / (grid_size + 1)
    spacing_y = target_bbox[1] / (grid_size + 1)

    for i in range(n):
        x0[i] = [
            (i % grid_size + 1) * spacing_x,
            (i // grid_size + 1) * spacing_y
        ]

    if verbose:
        print(f"Laying out {n} nodes with {len(edges)} edges...")
        print(f"Target bounding box: {target_bbox[0]}x{target_bbox[1]}")

    # Optimize
    result = minimize(
        energy_function,
        x0.flatten(),
        args=(nodes_list, edges_idx, target_bbox),
        method='L-BFGS-B',
        bounds=[(0, target_bbox[0]), (0, target_bbox[1])] * n,
        options={'maxiter': 2000, 'ftol': 1e-6}
    )

    if verbose:
        print(f"Optimization {'converged' if result.success else 'terminated'}")
        print(f"Final energy: {result.fun:.2f}")
        print(f"Iterations: {result.nit}")

    # Convert result back to dict of positions by ID
    positions_array = result.x.reshape(-1, 2)
    positions_dict = {
        node_id: positions_array[idx]
        for node_id, idx in id_to_idx.items()
    }

    return positions_dict


# ============================================================================
# RENDERING
# ============================================================================

def render_graph_matplotlib(
    nodes_dict: Dict[str, Node],
    edges: List[Tuple[str, str]],
    positions_dict: Dict[str, Tuple[float, float]],
    output_file: str = 'graph.png'
) -> None:
    """
    Render graph using matplotlib.

    Args:
        nodes_dict: dict of {node_id: Node}
        edges: list of (from_id, to_id) tuples
        positions_dict: dict of {node_id: (x, y)}
        output_file: path to save image
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw edges first (so they're behind nodes)
    for (u_id, v_id) in edges:
        u_pos = positions_dict[u_id]
        v_pos = positions_dict[v_id]
        u_node = nodes_dict[u_id]
        v_node = nodes_dict[v_id]

        # Calculate edge points on rectangles
        u_rect = (u_pos[0] - u_node.width/2, u_pos[1] - u_node.height/2,
                  u_node.width, u_node.height)
        v_rect = (v_pos[0] - v_node.width/2, v_pos[1] - v_node.height/2,
                  v_node.width, v_node.height)

        start = rect_edge_point(u_rect, u_pos, v_pos)
        end = rect_edge_point(v_rect, v_pos, u_pos)

        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='->,head_width=0.4,head_length=0.8',
            color='#333333',
            linewidth=1.5,
            zorder=1
        )
        ax.add_patch(arrow)

    # Draw nodes
    for node_id, node in nodes_dict.items():
        x, y = positions_dict[node_id]

        # Rectangle centered at (x, y)
        rect = patches.Rectangle(
            (x - node.width/2, y - node.height/2),
            node.width, node.height,
            linewidth=2,
            edgecolor='black',
            facecolor='lightblue',
            zorder=2
        )
        ax.add_patch(rect)

        # Label
        ax.text(
            x, y, node.label,
            ha='center', va='center',
            fontsize=node.fontsize, fontweight='bold',
            zorder=3
        )

    ax.set_aspect('equal')
    ax.autoscale()
    ax.margins(0.1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graph saved to {output_file}")


def render_graph_svg(
    nodes_dict: Dict[str, Node],
    edges: List[Tuple[str, str]],
    positions_dict: Dict[str, Tuple[float, float]],
    output_file: str = 'graph.svg'
) -> None:
    """
    Render graph as SVG.

    Args:
        nodes_dict: dict of {node_id: Node}
        edges: list of (from_id, to_id) tuples
        positions_dict: dict of {node_id: (x, y)}
        output_file: path to save SVG
    """
    # Find bounding box
    if not positions_dict:
        print("No nodes to render")
        return

    positions_list = [(node_id, positions_dict[node_id])
                      for node_id in nodes_dict.keys()]

    min_x = min(pos[0] - nodes_dict[node_id].width/2
                for node_id, pos in positions_list)
    max_x = max(pos[0] + nodes_dict[node_id].width/2
                for node_id, pos in positions_list)
    min_y = min(pos[1] - nodes_dict[node_id].height/2
                for node_id, pos in positions_list)
    max_y = max(pos[1] + nodes_dict[node_id].height/2
                for node_id, pos in positions_list)

    margin = 20
    width = max_x - min_x + 2 * margin
    height = max_y - min_y + 2 * margin

    offset_x = -min_x + margin
    offset_y = -min_y + margin

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<defs>',
        '  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        '    <polygon points="0 0, 10 3.5, 0 7" fill="#333" />',
        '  </marker>',
        '</defs>',
    ]

    # Draw edges
    for (u_id, v_id) in edges:
        u_pos = (positions_dict[u_id][0] + offset_x,
                 positions_dict[u_id][1] + offset_y)
        v_pos = (positions_dict[v_id][0] + offset_x,
                 positions_dict[v_id][1] + offset_y)
        u_node = nodes_dict[u_id]
        v_node = nodes_dict[v_id]

        u_rect = (u_pos[0] - u_node.width/2, u_pos[1] - u_node.height/2,
                  u_node.width, u_node.height)
        v_rect = (v_pos[0] - v_node.width/2, v_pos[1] - v_node.height/2,
                  v_node.width, v_node.height)

        start = rect_edge_point(u_rect, u_pos, v_pos)
        end = rect_edge_point(v_rect, v_pos, u_pos)

        svg.append(
            f'  <line x1="{start[0]:.2f}" y1="{start[1]:.2f}" '
            f'x2="{end[0]:.2f}" y2="{end[1]:.2f}" '
            f'stroke="#333" stroke-width="2" marker-end="url(#arrowhead)" />'
        )

    # Draw nodes
    for node_id, node in nodes_dict.items():
        x = positions_dict[node_id][0] + offset_x
        y = positions_dict[node_id][1] + offset_y

        svg.append(
            f'  <rect x="{x - node.width/2:.2f}" y="{y - node.height/2:.2f}" '
            f'width="{node.width}" height="{node.height}" '
            f'fill="lightblue" stroke="black" stroke-width="2" />'
        )

        # Escape special characters in label
        label = node.label.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        svg.append(
            f'  <text x="{x:.2f}" y="{y:.2f}" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'font-family="Arial" font-size="{node.fontsize}" font-weight="bold">'
            f'{label}</text>'
        )

    svg.append('</svg>')

    with open(output_file, 'w') as f:
        f.write('\n'.join(svg))
    print(f"✓ SVG saved to {output_file}")
