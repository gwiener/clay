#!/usr/bin/env python3
"""
Automatic Graph Layout Engine
Implements constrained optimization for compact, orderly diagram layouts.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch


@dataclass
class LayoutStats:
    """Statistics and metadata from layout optimization."""
    success: bool
    iterations: int
    function_evals: int
    final_energy: float
    penalty_breakdown: Dict[str, float]
    weights: Dict[str, float]
    target_bbox: Tuple[float, float]
    message: str
    seed: Optional[int]  # Random seed used for initialization (None = non-deterministic)


@dataclass
class LayoutResult:
    """Result from layout optimization containing positions and statistics."""
    positions: Dict[str, Tuple[float, float]]
    stats: LayoutStats


def measure_text_in_data_coords(
    label: str,
    fontsize: int,
    target_bbox: Tuple[float, float] = (800, 600)
) -> Tuple[float, float]:
    """
    Measure text dimensions in data coordinates (font points).

    This function properly converts text dimensions from font units (points)
    to data coordinates by rendering text in a temporary figure with DPI=72.
    This ensures 1 font point = 1 data coordinate unit = 1 pixel.

    Args:
        label: Text to measure
        fontsize: Font size in points
        target_bbox: Target coordinate system in font points (width, height)

    Returns:
        (width, height) in data coordinates (font points)
    """
    # Create temporary figure with DPI=72 for 1:1 font point mapping
    dpi = 72
    figsize = (target_bbox[0] / dpi, target_bbox[1] / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, target_bbox[0])
    ax.set_ylim(0, target_bbox[1])

    # Add text at center
    text = ax.text(
        target_bbox[0] / 2,
        target_bbox[1] / 2,
        label,
        ha='center',
        va='center',
        fontsize=fontsize,
        fontweight='bold'
    )

    # Draw to generate renderer
    fig.canvas.draw()

    # Get bounding box in display coordinates
    bbox_display = text.get_window_extent(renderer=fig.canvas.renderer)

    # Transform to data coordinates
    bbox_data = bbox_display.transformed(ax.transData.inverted())

    width = bbox_data.width
    height = bbox_data.height

    # Clean up
    plt.close(fig)

    return width, height


class Node:
    """Represents a graph node with label and dimensions."""

    def __init__(
        self,
        label: str,
        width: Optional[float] = None,
        height: Optional[float] = None,
        padding_x: float = 0.5,
        padding_y: float = 0.3,
        fontsize: int = 10,
        target_bbox: Tuple[float, float] = (800, 600)
    ):
        self.label = label
        self.fontsize = fontsize
        self.padding_x = padding_x
        self.padding_y = padding_y

        # Calculate size if not provided
        if width is None or height is None:
            # Measure text in data coordinates
            text_width, text_height = measure_text_in_data_coords(
                label, fontsize, target_bbox
            )

            self.width = width if width is not None else text_width + 2 * padding_x
            self.height = height if height is not None else text_height + 2 * padding_y
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
# INITIALIZATION STRATEGIES
# ============================================================================

def simple_spring_layout(
    edges: List[Tuple[int, int]],
    n: int,
    target_bbox: Tuple[float, float],
    iterations: int = 50,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Lightweight force-directed layout using Fruchterman-Reingold-style algorithm.

    Implements a simple spring/force-directed layout to generate better initial
    positions than a rigid grid. Connected nodes attract each other (spring forces),
    while all nodes repel each other (electrostatic forces).

    This provides structure-aware initialization that:
    - Places connected nodes closer together
    - Spreads unconnected nodes apart
    - Breaks symmetry that causes optimizer failures
    - Gives L-BFGS-B a much better starting point

    Args:
        edges: List of (from_idx, to_idx) tuples using node indices
        n: Number of nodes
        target_bbox: (width, height) for scaling output
        iterations: Number of force-directed iterations (default: 50)
        seed: Random seed for reproducibility (None = non-deterministic)

    Returns:
        Array of shape (n, 2) with (x, y) positions scaled to target_bbox

    Algorithm:
        1. Initialize nodes randomly in center region
        2. For each iteration:
           - Compute repulsive forces between all node pairs
           - Compute attractive forces along edges
           - Update positions with damped force application
        3. Scale positions to fit target_bbox with margins

    Complexity: O(iterations * (n^2 + m)) where m = number of edges
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize positions randomly near center
    # Use 20% of smaller bbox dimension as initial spread
    center = np.array(target_bbox) / 2
    spread = min(target_bbox) * 0.2
    pos = center + np.random.randn(n, 2) * spread

    # Force-directed iterations
    k_repel = 1000.0  # Repulsion strength
    k_attract = 0.1   # Spring constant for edges
    damping = 0.01    # Step size (prevent oscillation)

    for _ in range(iterations):
        forces = np.zeros((n, 2))

        # Repulsive forces between all pairs (inverse square law)
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist = np.linalg.norm(delta)

                if dist > 0.01:  # Avoid division by zero
                    # Force magnitude inversely proportional to distance squared
                    force_mag = k_repel / (dist ** 2)
                    force_dir = delta / dist

                    forces[i] += force_mag * force_dir
                    forces[j] -= force_mag * force_dir

        # Attractive forces along edges (Hooke's law)
        for u, v in edges:
            delta = pos[v] - pos[u]
            dist = np.linalg.norm(delta)

            if dist > 0.01:
                # Spring force proportional to distance
                force_mag = k_attract * dist
                force_dir = delta / dist

                forces[u] += force_mag * force_dir
                forces[v] -= force_mag * force_dir

        # Update positions with damping
        pos += forces * damping

    # Scale positions to fit target_bbox with margins
    # Add 10% margin on each side
    margin_factor = 0.1

    # Find current bounds
    min_pos = np.min(pos, axis=0)
    max_pos = np.max(pos, axis=0)
    current_size = max_pos - min_pos

    # Avoid division by zero if all nodes at same position
    current_size = np.maximum(current_size, 1.0)

    # Scale to fit within (1-2*margin) of target_bbox
    scale_factor = np.array(target_bbox) * (1 - 2 * margin_factor) / current_size
    scale = min(scale_factor)  # Uniform scaling to maintain aspect ratio

    # Center in target_bbox
    pos_scaled = (pos - min_pos) * scale
    offset = (np.array(target_bbox) - (max_pos - min_pos) * scale) / 2
    pos_final = pos_scaled + offset

    return pos_final


# ============================================================================
# ENERGY FUNCTION COMPONENTS
# ============================================================================

def overlap_penalty(positions: np.ndarray, nodes: List[Node]) -> float:
    """
    Smooth exponential overlap penalty to prevent node overlaps.

    Uses a smooth exponential barrier function instead of discontinuous max(0, ...)
    to avoid numerical instability in L-BFGS-B optimization. The penalty:
    - Is C∞ continuous (infinitely differentiable)
    - Provides long-range repulsion before nodes touch
    - Has bounded maximum value (~1500 for complete overlap)
    - Produces stable gradients for the optimizer

    The penalty grows exponentially as separation decreases below 1.0
    (where 1.0 = just touching). This maintains strong enforcement of
    non-overlap while avoiding the extreme values (billions) that caused
    "ABNORMAL termination" issues.

    Args:
        positions: array of shape (n, 2)
        nodes: list of Node objects

    Returns:
        penalty value (float), bounded and smooth
    """
    penalty = 0.0
    n = len(nodes)
    MARGIN = 10  # Minimum spacing between nodes
    STRENGTH = 10.0  # Base penalty magnitude
    SHARPNESS = 5.0  # Steepness of exponential (higher = more aggressive)
    REPULSION_RANGE = 1.5  # Start repelling at this multiple of minimum distance

    for i in range(n):
        for j in range(i + 1, n):
            # Distance between centers in each dimension
            dx = abs(positions[i][0] - positions[j][0])
            dy = abs(positions[i][1] - positions[j][1])

            # Minimum required distance in each dimension (rectangle half-widths + margin)
            min_dx = (nodes[i].width + nodes[j].width) / 2 + MARGIN
            min_dy = (nodes[i].height + nodes[j].height) / 2 + MARGIN

            # Separation ratios (1.0 = just touching, <1.0 = overlapping)
            sep_x = dx / min_dx if min_dx > 0 else 10.0
            sep_y = dy / min_dy if min_dy > 0 else 10.0

            # Only penalize if close in BOTH dimensions (rectangles near/overlapping)
            if sep_x < REPULSION_RANGE and sep_y < REPULSION_RANGE:
                # For rectangles to overlap, they must overlap in BOTH dimensions
                # So we need BOTH sep_x < 1.0 AND sep_y < 1.0
                # Use geometric mean to combine both constraints smoothly
                sep = np.sqrt(sep_x * sep_y)

                # Exponential barrier function:
                # - sep ≥ 1.0: penalty ≈ STRENGTH (light repulsion)
                # - sep < 1.0: penalty grows exponentially
                # - sep → 0: penalty → STRENGTH * exp(SHARPNESS) ≈ 1484
                violation = 1.0 - sep
                penalty += STRENGTH * np.exp(SHARPNESS * violation)

    return penalty


def edge_length_penalty(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    target_bbox: Tuple[float, float]
) -> float:
    """
    Penalize long edges (encourages connected nodes to be close).

    Normalized by target_bbox diagonal to make penalty scale-invariant.

    Args:
        positions: array of shape (n, 2)
        edges: list of (from_idx, to_idx) tuples
        target_bbox: (width, height) tuple for normalization

    Returns:
        penalty value (float), normalized to [0, 1] range approximately
    """
    if len(edges) == 0:
        return 0.0

    # Normalize distances by diagonal of target bbox
    diagonal = np.sqrt(target_bbox[0]**2 + target_bbox[1]**2)

    total = 0
    for (u, v) in edges:
        dist = np.linalg.norm(positions[v] - positions[u])
        normalized_dist = dist / diagonal
        total += normalized_dist ** 2

    return total


def straightness_penalty(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    nodes: List[Node],
    target_bbox: Tuple[float, float]
) -> float:
    """
    Penalize non-straight paths through nodes.
    For any path A->B->C, penalizes if B is not collinear with A and C.

    Normalized by target_bbox diagonal to make penalty scale-invariant.

    Args:
        positions: array of shape (n, 2)
        edges: list of (from_idx, to_idx) tuples
        nodes: list of Node objects
        target_bbox: (width, height) tuple for normalization

    Returns:
        penalty value (float), normalized to [0, 1] range approximately
    """
    penalty = 0

    # Build adjacency lists
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for (u, v) in edges:
        outgoing[u].append(v)
        incoming[v].append(u)

    # Normalize distances by diagonal of target bbox
    diagonal = np.sqrt(target_bbox[0]**2 + target_bbox[1]**2)

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
                normalized_dist = dist / diagonal
                penalty += normalized_dist ** 2

    return penalty


def edge_node_intersection_penalty(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    nodes: List[Node]
) -> float:
    """
    Penalize edges that pass through nodes that are not their endpoints.

    For each edge, checks if it intersects with any node rectangle (except
    the edge's source and destination nodes). If an intersection is detected,
    calculates how deeply the edge penetrates through the node and applies
    a quadratic penalty.

    Uses the segment midpoint depth as a proxy for penetration severity:
    edges passing through the center of a node are penalized more heavily
    than edges grazing the edge of a node.

    Args:
        positions: array of shape (n, 2) with (x, y) coordinates
        edges: list of (from_idx, to_idx) tuples
        nodes: list of Node objects with width and height attributes

    Returns:
        penalty value (float)

    Complexity:
        O(m * n) where m = number of edges, n = number of nodes
        Acceptable for typical graphs with dozens of nodes and edges.

    Note:
        Requires segment_intersects_rectangle from clay.geometry module.
    """
    from clay.geometry import segment_intersects_rectangle

    penalty = 0.0

    for (u, v) in edges:
        edge_start = positions[u]
        edge_end = positions[v]
        edge_midpoint = (edge_start + edge_end) / 2.0

        # Check intersection with all nodes except edge endpoints
        for i in range(len(nodes)):
            if i == u or i == v:
                continue  # Skip edge's source and destination nodes

            # Fast binary check: does edge intersect this node's rectangle?
            if segment_intersects_rectangle(
                tuple(edge_start),
                tuple(edge_end),
                tuple(positions[i]),
                nodes[i].width,
                nodes[i].height
            ):
                # Calculate penetration depth at edge midpoint
                rect_center = positions[i]
                half_width = nodes[i].width / 2.0
                half_height = nodes[i].height / 2.0

                # Distance from midpoint to rectangle center in each dimension
                dx = abs(edge_midpoint[0] - rect_center[0])
                dy = abs(edge_midpoint[1] - rect_center[1])

                # Depth from each edge (how far inside from the boundary)
                depth_x = half_width - dx
                depth_y = half_height - dy

                # Use minimum depth (distance to nearest rectangle edge)
                # This represents how deeply the edge penetrates
                if depth_x > 0 and depth_y > 0:
                    depth = min(depth_x, depth_y)
                    # Quadratic penalty: deeper penetration = much worse
                    penalty += depth ** 2

    return penalty


def _point_to_segment_distance(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray
) -> float:
    """
    Calculate minimum distance from a point to a line segment.

    Projects the point onto the infinite line through the segment,
    then clamps to the segment endpoints if projection falls outside.

    Args:
        point: Point coordinates as array [x, y]
        seg_start: Segment start point [x, y]
        seg_end: Segment end point [x, y]

    Returns:
        Minimum distance from point to segment (float)
    """
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq < 1e-10:
        # Segment is essentially a point
        return np.linalg.norm(point - seg_start)

    # Project point onto infinite line: t is parameter (0=start, 1=end)
    t = np.dot(point - seg_start, seg_vec) / seg_len_sq

    # Clamp t to [0, 1] to stay within segment bounds
    t = max(0.0, min(1.0, t))

    # Closest point on segment
    closest = seg_start + t * seg_vec

    return np.linalg.norm(point - closest)


def _segment_to_segment_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray
) -> float:
    """
    Calculate minimum distance between two line segments.

    The minimum distance is either:
    - Zero if segments intersect
    - Distance from an endpoint of one segment to the other segment

    Args:
        p1, p2: Endpoints of first segment
        p3, p4: Endpoints of second segment

    Returns:
        Minimum distance between segments (float)
    """
    from clay.geometry import segments_intersect

    # Check if segments intersect
    if segments_intersect(tuple(p1), tuple(p2), tuple(p3), tuple(p4)):
        return 0.0

    # Segments don't intersect - find minimum distance
    # Check distance from each endpoint to the other segment
    distances = [
        _point_to_segment_distance(p1, p3, p4),
        _point_to_segment_distance(p2, p3, p4),
        _point_to_segment_distance(p3, p1, p2),
        _point_to_segment_distance(p4, p1, p2),
    ]

    return min(distances)


def edge_crossing_penalty(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    target_bbox: Tuple[float, float]
) -> float:
    """
    Penalize edges that cross or come close to crossing each other.

    Uses smooth exponential penalty based on minimum distance between
    edge segments. This guides the optimizer away from crossings while
    maintaining differentiability for L-BFGS-B.

    The penalty:
    - Is zero when edges are far apart (> 5% of bbox diagonal)
    - Grows exponentially as edges approach each other
    - Is bounded (maximum ~150 for complete overlap)
    - Skips edges that share a common endpoint

    Args:
        positions: array of shape (n, 2) with node coordinates
        edges: list of (from_idx, to_idx) tuples using node indices
        target_bbox: (width, height) tuple for normalization

    Returns:
        penalty value (float), smooth and differentiable

    Complexity:
        O(m²) where m = number of edges
    """
    if len(edges) <= 1:
        return 0.0

    penalty = 0.0
    n_edges = len(edges)

    # Normalize distances by bbox diagonal for scale-invariance
    diagonal = np.sqrt(target_bbox[0]**2 + target_bbox[1]**2)

    # Penalty parameters
    THRESHOLD = 0.02      # Start penalizing at 2% of diagonal (was 5%)
    STRENGTH = 5.0        # Base penalty magnitude (was 10.0)
    SHARPNESS = 3.0       # Exponential steepness (was 5.0)

    for i in range(n_edges):
        u1, v1 = edges[i]
        edge1_start = positions[u1]
        edge1_end = positions[v1]

        for j in range(i + 1, n_edges):
            u2, v2 = edges[j]

            # Skip edges that share a common endpoint
            # (adjacent edges in a path should not be penalized)
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue

            edge2_start = positions[u2]
            edge2_end = positions[v2]

            # Calculate minimum distance between the two line segments
            dist = _segment_to_segment_distance(
                edge1_start, edge1_end,
                edge2_start, edge2_end
            )

            # Normalize distance by bbox diagonal
            normalized_dist = dist / diagonal

            # Apply smooth exponential penalty when distance < threshold
            if normalized_dist < THRESHOLD:
                violation = THRESHOLD - normalized_dist
                # Exponential barrier: penalty grows as segments approach
                penalty += STRENGTH * np.exp(SHARPNESS * violation / THRESHOLD)

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


def area_penalty(
    positions: np.ndarray,
    nodes: List[Node],
    target_bbox: Tuple[float, float]
) -> float:
    """
    Penalize large bounding box area (encourages compactness).

    Normalized by target_bbox area to make penalty scale-invariant.

    Args:
        positions: array of shape (n, 2)
        nodes: list of Node objects
        target_bbox: (width, height) tuple for normalization

    Returns:
        penalty value (float), normalized to [0, 1] range approximately
    """
    if len(positions) == 0:
        return 0.0

    # Calculate bounding box
    min_x = min(positions[i][0] - nodes[i].width/2 for i in range(len(nodes)))
    max_x = max(positions[i][0] + nodes[i].width/2 for i in range(len(nodes)))
    min_y = min(positions[i][1] - nodes[i].height/2 for i in range(len(nodes)))
    max_y = max(positions[i][1] + nodes[i].height/2 for i in range(len(nodes)))

    area = (max_x - min_x) * (max_y - min_y)

    # Normalize by target bbox area
    target_area = target_bbox[0] * target_bbox[1]
    normalized_area = area / target_area

    return normalized_area


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
    W_OVERLAP = 100        # Smooth exponential barrier - prevent overlaps (reduced from 1000)
    W_EDGE_LENGTH = 10     # Keep connected nodes close
    W_STRAIGHTNESS = 5     # Encourage straight-through paths
    W_EDGE_NODE = 200      # Prevent edges from crossing through nodes
    W_BBOX = 100           # Stay within target box
    W_AREA = 1             # Minimize total area
    W_EDGE_CROSSING = 1    # Prevent edges from crossing each other

    E += W_OVERLAP * overlap_penalty(positions, nodes)
    E += W_EDGE_LENGTH * edge_length_penalty(positions, edges, target_bbox)
    E += W_STRAIGHTNESS * straightness_penalty(positions, edges, nodes, target_bbox)
    E += W_EDGE_NODE * edge_node_intersection_penalty(positions, edges, nodes)
    E += W_BBOX * bounding_box_penalty(positions, nodes, target_bbox)
    E += W_AREA * area_penalty(positions, nodes, target_bbox)
    E += W_EDGE_CROSSING * edge_crossing_penalty(positions, edges, target_bbox)

    return E


# ============================================================================
# LAYOUT ENGINE
# ============================================================================

def layout_graph(
    nodes_dict: Dict[str, Node],
    edges: List[Tuple[str, str]],
    target_bbox: Tuple[float, float] = (800, 600),
    verbose: bool = True,
    init_mode: str = 'spring',
    seed: Optional[int] = None,
    callback: Optional[Callable[[np.ndarray], None]] = None,
    n_init: int = 5
) -> LayoutResult:
    """
    Layout a graph using constrained optimization with multi-start.

    Args:
        nodes_dict: dict of {node_id: Node}
        edges: list of (from_id, to_id) tuples
        target_bbox: (width, height) tuple for bounding box
        verbose: print optimization progress
        init_mode: initialization strategy ('spring', 'grid', 'random')
        seed: random seed for reproducibility (None = non-deterministic)
        callback: optional callback function called at each iteration with current position vector
        n_init: number of random initializations to try (default: 5, set to 1 to disable multi-start)

    Returns:
        LayoutResult containing positions and optimization statistics from best run
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
        # Return empty result
        empty_stats = LayoutStats(
            success=True,
            iterations=0,
            function_evals=0,
            final_energy=0.0,
            penalty_breakdown={},
            weights={},
            target_bbox=target_bbox,
            message="Empty graph",
            seed=seed
        )
        return LayoutResult(positions={}, stats=empty_stats)

    # Multi-start optimization: try n_init random initializations
    if verbose and n_init > 1:
        print(f"Multi-start optimization: trying {n_init} initializations...")
        print(f"Laying out {n} nodes with {len(edges)} edges...")
        print(f"Target bounding box: {target_bbox[0]}x{target_bbox[1]}")
    elif verbose:
        print(f"Laying out {n} nodes with {len(edges)} edges...")
        print(f"Target bounding box: {target_bbox[0]}x{target_bbox[1]}")

    best_result = None
    best_energy = float('inf')
    best_x0 = None
    best_init_seed = None

    for init_idx in range(n_init):
        # Determine seed for this initialization
        if seed is not None:
            init_seed = seed + init_idx
        else:
            init_seed = None

        # Generate initial positions based on init_mode
        if init_mode == 'spring':
            # Force-directed pre-layout
            x0 = simple_spring_layout(edges_idx, n, target_bbox, iterations=50, seed=init_seed)
            init_desc = "force-directed"
        elif init_mode == 'random':
            # Random uniform positions
            if init_seed is not None:
                np.random.seed(init_seed)
            x0 = np.random.uniform(
                low=[20, 20],
                high=[target_bbox[0] - 20, target_bbox[1] - 20],
                size=(n, 2)
            )
            init_desc = "random"
        elif init_mode == 'grid':
            # Simple grid layout (original approach)
            # For grid mode with n_init > 1, add random jitter to each initialization
            grid_size = int(np.ceil(np.sqrt(n)))
            x0 = np.zeros((n, 2))
            spacing_x = target_bbox[0] / (grid_size + 1)
            spacing_y = target_bbox[1] / (grid_size + 1)
            for i in range(n):
                x0[i] = [
                    (i % grid_size + 1) * spacing_x,
                    (i // grid_size + 1) * spacing_y
                ]
            # Add jitter for multi-start (except first initialization)
            if n_init > 1 and init_idx > 0:
                if init_seed is not None:
                    np.random.seed(init_seed)
                jitter = np.random.uniform(-20, 20, size=(n, 2))
                x0 = np.clip(x0 + jitter, [0, 0], target_bbox)
            init_desc = "grid" + (" + jitter" if n_init > 1 and init_idx > 0 else "")
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}. Use 'spring', 'grid', or 'random'.")

        if verbose and n_init > 1:
            print(f"  Init {init_idx + 1}/{n_init}: {init_desc}" + (f" (seed={init_seed})" if init_seed is not None else ""))
        elif verbose:
            print(f"Initialization: {init_desc}" + (f" (seed={init_seed})" if init_seed is not None else ""))

        # Optimize
        result = minimize(
            energy_function,
            x0.flatten(),
            args=(nodes_list, edges_idx, target_bbox),
            method='L-BFGS-B',
            bounds=[(0, target_bbox[0]), (0, target_bbox[1])] * n,
            options={'maxiter': 2000, 'ftol': 1e-6},
            callback=callback if n_init == 1 else None  # Only use callback for single-start
        )

        if verbose and n_init > 1:
            print(f"    Energy: {result.fun:.2f}, Iterations: {result.nit}, Status: {'✓' if result.success else '✗'}")

        # Keep track of best result
        if result.fun < best_energy:
            best_energy = result.fun
            best_result = result
            best_x0 = x0
            best_init_seed = init_seed

    # Use the best result
    result = best_result

    if verbose:
        if n_init > 1:
            print(f"Best result: energy={result.fun:.2f} from initialization with seed={best_init_seed}")
        print(f"Optimization {'converged' if result.success else 'terminated'}")
        print(f"Final energy: {result.fun:.2f}")
        print(f"Iterations: {result.nit}")

    # Convert result back to dict of positions by ID
    positions_array = result.x.reshape(-1, 2)
    positions_dict = {
        node_id: positions_array[idx]
        for node_id, idx in id_to_idx.items()
    }

    # Compute penalty breakdown at final positions
    # Weight constants (must match energy_function)
    weights = {
        'overlap': 100,
        'edge_length': 10,
        'straightness': 5,
        'edge_node': 200,
        'bbox': 100,
        'area': 1,
        'edge_crossing': 1
    }

    penalty_breakdown = {
        'overlap': overlap_penalty(positions_array, nodes_list),
        'edge_length': edge_length_penalty(positions_array, edges_idx, target_bbox),
        'straightness': straightness_penalty(positions_array, edges_idx, nodes_list, target_bbox),
        'edge_node': edge_node_intersection_penalty(positions_array, edges_idx, nodes_list),
        'bbox': bounding_box_penalty(positions_array, nodes_list, target_bbox),
        'area': area_penalty(positions_array, nodes_list, target_bbox),
        'edge_crossing': edge_crossing_penalty(positions_array, edges_idx, target_bbox)
    }

    # Create stats object
    stats = LayoutStats(
        success=result.success,
        iterations=result.nit,
        function_evals=result.nfev,
        final_energy=result.fun,
        penalty_breakdown=penalty_breakdown,
        weights=weights,
        target_bbox=target_bbox,
        message=result.message,
        seed=seed
    )

    return LayoutResult(positions=positions_dict, stats=stats)


# ============================================================================
# RENDERING
# ============================================================================

def render_graph_matplotlib(
    nodes_dict: Dict[str, Node],
    edges: List[Tuple[str, str]],
    positions_dict: Dict[str, Tuple[float, float]],
    output_file: str = 'graph.png',
    target_bbox: Tuple[float, float] = (800, 600)
) -> None:
    """
    Render graph using matplotlib.

    Coordinate system uses font points (1/72 inch) as the unit. With DPI=72,
    this creates a 1:1 mapping where 1 data unit = 1 font point = 1 pixel.
    For example, a 800x600 target_bbox creates an 11.11x8.33 inch figure
    rendered at 800x600 pixels.

    Args:
        nodes_dict: dict of {node_id: Node}
        edges: list of (from_id, to_id) tuples
        positions_dict: dict of {node_id: (x, y)}
        output_file: path to save image
        target_bbox: coordinate system in font points (should match what was used for layout)
    """
    # Use DPI=72 to align font points with data coordinates
    # 1 font point = 1/72 inch, so at 72 DPI: 1 point = 1 pixel
    dpi = 72
    figsize = (target_bbox[0] / dpi, target_bbox[1] / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

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
            arrowstyle='->,head_width=4,head_length=8',
            color='#333333',
            linewidth=2,
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

    # Calculate tight bounding box from actual node positions
    min_x = min(positions_dict[node_id][0] - nodes_dict[node_id].width/2
                for node_id in nodes_dict.keys())
    max_x = max(positions_dict[node_id][0] + nodes_dict[node_id].width/2
                for node_id in nodes_dict.keys())
    min_y = min(positions_dict[node_id][1] - nodes_dict[node_id].height/2
                for node_id in nodes_dict.keys())
    max_y = max(positions_dict[node_id][1] + nodes_dict[node_id].height/2
                for node_id in nodes_dict.keys())

    # Add 10% margin
    width = max_x - min_x
    height = max_y - min_y
    margin_x = width * 0.1
    margin_y = height * 0.1

    # Set axis limits to show only the used area
    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
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
