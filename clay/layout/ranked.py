"""Sugiyama-style layered (ranked) layout engine."""

from collections import defaultdict

from clay import graph
from clay.layout import LayoutEngine, Result

# Spacing constants
LAYER_SPACING = 100  # pixels between layers
NODE_SPACING = 50    # pixels between nodes in same layer


class Ranked(LayoutEngine):
    """Sugiyama-style layered layout engine."""

    def __init__(self, direction: str = "TB"):
        """
        Initialize the ranked layout engine.

        Args:
            direction: Layout direction - "TB" (top-to-bottom) or "LR" (left-to-right)
        """
        if direction not in ("TB", "LR"):
            raise ValueError(f"Invalid direction: {direction}. Must be 'TB' or 'LR'")
        self.direction = direction

    def fit(self, g: graph.Graph) -> Result:
        """
        Generate a layered layout for the given graph.

        Args:
            g: Graph object containing nodes and edges.

        Returns:
            A Result object with hierarchical positions for each node.
        """
        if not g.nodes:
            return Result(graph.Layout(g, []), metadata={
                "layers": {},
                "canvas_width": 0,
                "canvas_height": 0,
            })

        # Phase 1: Remove cycles
        reversed_edges = self._remove_cycles(g)

        # Phase 2: Assign layers
        node_layers = self._assign_layers(g, reversed_edges)

        # Phase 3: Minimize crossings
        layer_order = self._minimize_crossings(g, node_layers, reversed_edges)

        # Phase 4: Assign coordinates
        centers, canvas_width, canvas_height = self._assign_coordinates(g, node_layers, layer_order)

        layout = graph.Layout(g, centers)
        return Result(layout, metadata={
            "layers": node_layers,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
        })

    def _remove_cycles(self, g: graph.Graph) -> set[tuple[str, str]]:
        """
        Find back edges using DFS to identify cycles.

        Returns:
            Set of edges (src, dst) that should be conceptually reversed.
        """
        visited: set[str] = set()
        in_stack: set[str] = set()
        reversed_edges: set[tuple[str, str]] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            in_stack.add(node)

            for neighbor in g.outgoing.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in in_stack:
                    # Back edge found - mark for reversal
                    reversed_edges.add((node, neighbor))

            in_stack.remove(node)

        for node in g.nodes:
            if node.name not in visited:
                dfs(node.name)

        return reversed_edges

    def _get_effective_edges(
        self,
        g: graph.Graph,
        reversed_edges: set[tuple[str, str]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """
        Get effective outgoing/incoming edges with reversed edges applied.

        Returns:
            Tuple of (effective_outgoing, effective_incoming) adjacency dicts.
        """
        effective_outgoing: dict[str, list[str]] = defaultdict(list)
        effective_incoming: dict[str, list[str]] = defaultdict(list)

        for src, dst in g.edges:
            if (src, dst) in reversed_edges:
                # Reverse this edge
                effective_outgoing[dst].append(src)
                effective_incoming[src].append(dst)
            else:
                effective_outgoing[src].append(dst)
                effective_incoming[dst].append(src)

        return dict(effective_outgoing), dict(effective_incoming)

    def _assign_layers(
        self,
        g: graph.Graph,
        reversed_edges: set[tuple[str, str]]
    ) -> dict[str, int]:
        """
        Assign nodes to layers using longest path algorithm.

        Returns:
            Dict mapping node name to layer index (0 = top/left).
        """
        effective_outgoing, effective_incoming = self._get_effective_edges(g, reversed_edges)

        # Find sources (nodes with no incoming edges in DAG)
        all_nodes = {node.name for node in g.nodes}
        sources = all_nodes - set(effective_incoming.keys())

        # If no sources found (all nodes in cycles), pick arbitrary start
        if not sources:
            sources = {g.nodes[0].name}

        # Compute longest path from sources
        node_layers: dict[str, int] = {}

        def compute_layer(node: str, visited_path: set[str]) -> int:
            if node in node_layers:
                return node_layers[node]

            # Prevent infinite recursion in case of remaining cycles
            if node in visited_path:
                return 0

            visited_path = visited_path | {node}

            incoming = effective_incoming.get(node, [])
            if not incoming:
                node_layers[node] = 0
            else:
                max_parent_layer = max(
                    compute_layer(parent, visited_path) for parent in incoming
                )
                node_layers[node] = max_parent_layer + 1

            return node_layers[node]

        for node in g.nodes:
            compute_layer(node.name, set())

        return node_layers

    def _minimize_crossings(
        self,
        g: graph.Graph,
        node_layers: dict[str, int],
        reversed_edges: set[tuple[str, str]]
    ) -> dict[int, list[str]]:
        """
        Order nodes within each layer using barycenter heuristic.

        Returns:
            Dict mapping layer index to ordered list of node names.
        """
        effective_outgoing, effective_incoming = self._get_effective_edges(g, reversed_edges)

        # Group nodes by layer
        layers: dict[int, list[str]] = defaultdict(list)
        for node_name, layer in node_layers.items():
            layers[layer].append(node_name)

        # Convert to sorted dict
        layer_order = {k: v for k, v in sorted(layers.items())}
        num_layers = max(layer_order.keys()) + 1 if layer_order else 0

        # Initialize positions within each layer
        positions: dict[str, float] = {}
        for layer_idx, nodes in layer_order.items():
            for i, node in enumerate(nodes):
                positions[node] = float(i)

        # Barycenter iterations
        max_iterations = 4
        for _ in range(max_iterations):
            # Sweep down (top to bottom)
            for layer_idx in range(1, num_layers):
                self._barycenter_sort(
                    layer_order[layer_idx],
                    positions,
                    effective_incoming,
                    "incoming"
                )
                # Update positions after sort
                for i, node in enumerate(layer_order[layer_idx]):
                    positions[node] = float(i)

            # Sweep up (bottom to top)
            for layer_idx in range(num_layers - 2, -1, -1):
                self._barycenter_sort(
                    layer_order[layer_idx],
                    positions,
                    effective_outgoing,
                    "outgoing"
                )
                # Update positions after sort
                for i, node in enumerate(layer_order[layer_idx]):
                    positions[node] = float(i)

        return layer_order

    def _barycenter_sort(
        self,
        layer_nodes: list[str],
        positions: dict[str, float],
        adjacency: dict[str, list[str]],
        direction: str
    ) -> None:
        """
        Sort nodes in a layer by barycenter (average position of neighbors).

        Modifies layer_nodes in place.
        """
        def barycenter(node: str) -> float:
            neighbors = adjacency.get(node, [])
            if not neighbors:
                # Keep original position if no neighbors
                return positions.get(node, 0.0)
            return sum(positions.get(n, 0.0) for n in neighbors) / len(neighbors)

        layer_nodes.sort(key=barycenter)

    def _assign_coordinates(
        self,
        g: graph.Graph,
        node_layers: dict[str, int],
        layer_order: dict[int, list[str]]
    ) -> tuple[list[float], float, float]:
        """
        Assign x, y coordinates to each node.

        Returns:
            Tuple of (centers array, canvas_width, canvas_height).
        """
        if not layer_order:
            return [], 0.0, 0.0

        num_layers = max(layer_order.keys()) + 1

        # Find max dimensions for spacing calculations
        node_dims = {node.name: (node.width, node.height) for node in g.nodes}
        max_width = max(node.width for node in g.nodes)
        max_height = max(node.height for node in g.nodes)

        # Calculate layer positions and max layer width
        layer_widths: dict[int, float] = {}
        for layer_idx, nodes in layer_order.items():
            if not nodes:
                layer_widths[layer_idx] = 0
                continue
            # Sum of node widths + spacing between them
            total_width = sum(node_dims[n][0] for n in nodes)
            total_width += NODE_SPACING * (len(nodes) - 1)
            layer_widths[layer_idx] = total_width

        max_layer_width = max(layer_widths.values()) if layer_widths else 0

        # Calculate canvas dimensions based on direction
        if self.direction == "TB":
            # Top to bottom: layers are vertical, nodes spread horizontally
            canvas_height = (
                max_height / 2 +
                (num_layers - 1) * (max_height + LAYER_SPACING) +
                max_height / 2
            )
            canvas_width = max_layer_width + max_width  # Add padding
        else:  # LR
            # Left to right: layers are horizontal, nodes spread vertically
            canvas_width = (
                max_width / 2 +
                (num_layers - 1) * (max_width + LAYER_SPACING) +
                max_width / 2
            )
            canvas_height = max_layer_width + max_height  # Add padding

        # Compute node positions
        node_positions: dict[str, tuple[float, float]] = {}

        for layer_idx, nodes in layer_order.items():
            if not nodes:
                continue

            # Layer offset along main axis
            layer_offset = max_height / 2 + layer_idx * (max_height + LAYER_SPACING)

            # Center nodes within layer
            layer_width = layer_widths[layer_idx]
            start_offset = (max_layer_width - layer_width) / 2 + node_dims[nodes[0]][0] / 2

            current_pos = start_offset
            for node_name in nodes:
                width, height = node_dims[node_name]

                if self.direction == "TB":
                    x = current_pos + max_width / 2  # Add left padding
                    # Invert y so layer 0 is at top (high y in matplotlib coords)
                    y = canvas_height - layer_offset
                else:  # LR
                    x = layer_offset
                    y = current_pos + max_height / 2  # Add top padding

                node_positions[node_name] = (x, y)
                current_pos += width + NODE_SPACING

        # Build flat centers array in node order
        centers: list[float] = []
        for node in g.nodes:
            x, y = node_positions[node.name]
            centers.extend([x, y])

        return centers, canvas_width, canvas_height
