"""
Custom DSL parser for Clay diagram specifications.

Provides a minimal, human-friendly syntax for defining graph layouts:

Example DSL:
    # Nodes
    user
    api "API Gateway" width=120
    database

    # Edges
    user -> api -> database

    # Settings
    @bbox 800 600
    @weight straightness 8
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from clay.layout import Node, layout_graph


# Regex patterns for parsing
NODE_PATTERN = re.compile(
    r'^(\w+)\s*(?:"([^"]+)")?\s*(.*?)$'
)  # Captures: id, optional label, properties string

EDGE_PATTERN = re.compile(
    r'(\w+)\s*->\s*'
)  # Matches arrow separators

SETTING_PATTERN = re.compile(
    r'^@(\w+)\s+(.+)$'
)  # Captures: setting name, value(s)

PROPERTY_PATTERN = re.compile(
    r'(\w+)=([\w.]+)'
)  # Captures: key=value pairs


def _id_to_label(node_id: str) -> str:
    """
    Convert node ID to human-readable label.

    Examples:
        user -> User
        api_gateway -> Api Gateway
        myNode -> MyNode
    """
    # Split on underscores, capitalize each word
    words = node_id.replace('_', ' ').split()
    return ' '.join(word.capitalize() for word in words)


def _parse_properties(prop_string: str) -> Dict[str, Any]:
    """
    Parse space-separated key=value properties.

    Args:
        prop_string: String like "width=120 height=50 fontsize=14"

    Returns:
        Dict with typed values (converts numbers to float/int)
    """
    properties = {}

    for match in PROPERTY_PATTERN.finditer(prop_string):
        key, value_str = match.groups()

        # Try to convert to numeric types
        try:
            # Try int first
            if '.' not in value_str:
                value = int(value_str)
            else:
                value = float(value_str)
        except ValueError:
            # Keep as string
            value = value_str

        properties[key] = value

    return properties


def parse_node_line(line: str) -> Optional[Tuple[str, Optional[str], Dict[str, Any]]]:
    """
    Parse a node declaration line.

    Formats:
        user                          -> ('user', None, {})
        api "API Gateway"            -> ('api', 'API Gateway', {})
        web "Web" width=120          -> ('web', 'Web', {'width': 120})

    Returns:
        (node_id, label, properties) or None if not a node line
    """
    match = NODE_PATTERN.match(line)
    if not match:
        return None

    node_id, custom_label, prop_string = match.groups()
    properties = _parse_properties(prop_string) if prop_string else {}

    return (node_id, custom_label, properties)


def parse_edge_line(line: str) -> Optional[List[Tuple[str, str]]]:
    """
    Parse an edge declaration line with chain support.

    Formats:
        user -> web               -> [('user', 'web')]
        user -> api -> database   -> [('user', 'api'), ('api', 'database')]

    Returns:
        List of edge tuples or None if not an edge line
    """
    if '->' not in line:
        return None

    # Split by arrows and extract node IDs
    parts = [part.strip() for part in EDGE_PATTERN.split(line)]
    # Remove empty strings
    node_ids = [p for p in parts if p]

    if len(node_ids) < 2:
        return None

    # Create edges for each consecutive pair
    edges = []
    for i in range(len(node_ids) - 1):
        edges.append((node_ids[i], node_ids[i + 1]))

    return edges


def parse_setting_line(line: str) -> Optional[Tuple[str, List[str]]]:
    """
    Parse a settings directive line.

    Formats:
        @bbox 800 600              -> ('bbox', ['800', '600'])
        @weight straightness 8     -> ('weight', ['straightness', '8'])
        @verbose false             -> ('verbose', ['false'])

    Returns:
        (setting_name, values) or None if not a setting line
    """
    match = SETTING_PATTERN.match(line)
    if not match:
        return None

    setting_name, values_str = match.groups()
    values = values_str.strip().split()

    return (setting_name, values)


class ParsedDiagram:
    """Container for parsed diagram data."""

    def __init__(self):
        # Raw parsed data
        self.nodes: Dict[str, Tuple[Optional[str], Dict[str, Any]]] = {}
        self.edges: List[Tuple[str, str]] = []
        self.settings: Dict[str, List[str]] = {}

        # Metadata for error reporting
        self.node_line_numbers: Dict[str, int] = {}
        self.edge_line_numbers: List[int] = []


def parse_clay_text(text: str) -> ParsedDiagram:
    """
    Main parser for Clay DSL text.

    Args:
        text: Multi-line string containing Clay DSL

    Returns:
        ParsedDiagram with nodes, edges, and settings

    Raises:
        ValueError: If parsing fails or references are invalid
    """
    diagram = ParsedDiagram()

    for line_num, line in enumerate(text.split('\n'), start=1):
        # Strip whitespace and skip empty lines / comments
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Try parsing as each type
        # Check edges first (more specific pattern)
        edge_data = parse_edge_line(line)
        if edge_data:
            diagram.edges.extend(edge_data)
            diagram.edge_line_numbers.extend([line_num] * len(edge_data))
            continue

        # Check settings
        setting_data = parse_setting_line(line)
        if setting_data:
            setting_name, values = setting_data
            diagram.settings[setting_name] = values
            continue

        # Check nodes last (most general pattern)
        node_data = parse_node_line(line)
        if node_data:
            node_id, custom_label, properties = node_data
            if node_id in diagram.nodes:
                raise ValueError(
                    f"Line {line_num}: Duplicate node ID '{node_id}' "
                    f"(first defined on line {diagram.node_line_numbers[node_id]})"
                )
            diagram.nodes[node_id] = (custom_label, properties)
            diagram.node_line_numbers[node_id] = line_num
            continue

        # If we get here, line didn't match any pattern
        raise ValueError(f"Line {line_num}: Unable to parse line: {line}")

    # Validate edge references
    for idx, (from_id, to_id) in enumerate(diagram.edges):
        line_num = diagram.edge_line_numbers[idx]
        if from_id not in diagram.nodes:
            raise ValueError(
                f"Line {line_num}: Edge references undefined node '{from_id}'"
            )
        if to_id not in diagram.nodes:
            raise ValueError(
                f"Line {line_num}: Edge references undefined node '{to_id}'"
            )

    return diagram


def _build_nodes_dict(
    parsed_diagram: ParsedDiagram,
    target_bbox: Tuple[float, float]
) -> Dict[str, Node]:
    """
    Convert parsed nodes to Node objects.

    Args:
        parsed_diagram: Parsed diagram data
        target_bbox: Target bounding box for text measurement

    Returns:
        Dictionary mapping node IDs to Node objects
    """
    nodes_dict = {}

    for node_id, (custom_label, properties) in parsed_diagram.nodes.items():
        # Use custom label or auto-generate from ID
        label = custom_label if custom_label else _id_to_label(node_id)

        # Add target_bbox to properties for text measurement
        node_properties = {**properties, 'target_bbox': target_bbox}

        # Create Node with label and unpacked properties
        nodes_dict[node_id] = Node(label=label, **node_properties)

    return nodes_dict


def _extract_graph_settings(parsed_diagram: ParsedDiagram) -> Dict[str, Any]:
    """
    Extract graph-level settings from parsed data.

    Returns:
        Dict with keys: target_bbox, verbose, weights
    """
    settings = {
        'target_bbox': (800.0, 600.0),  # Default
        'verbose': True,  # Default
        'weights': {}  # Empty means use defaults
    }

    for setting_name, values in parsed_diagram.settings.items():
        if setting_name == 'bbox':
            if len(values) != 2:
                raise ValueError(f"@bbox requires 2 values, got {len(values)}")
            settings['target_bbox'] = (float(values[0]), float(values[1]))

        elif setting_name == 'verbose':
            value_lower = values[0].lower()
            if value_lower in ('true', '1', 'yes'):
                settings['verbose'] = True
            elif value_lower in ('false', '0', 'no'):
                settings['verbose'] = False
            else:
                raise ValueError(f"@verbose must be true/false, got '{values[0]}'")

        elif setting_name == 'weight':
            if len(values) != 2:
                raise ValueError(f"@weight requires 2 values (name value), got {len(values)}")
            weight_name, weight_value = values
            settings['weights'][weight_name] = float(weight_value)

        else:
            raise ValueError(f"Unknown setting '@{setting_name}'")

    return settings


def layout_from_text(
    clay_text: str,
    target_bbox: Optional[Tuple[float, float]] = None,
    verbose: Optional[bool] = None,
    **weight_overrides
) -> Dict[str, Tuple[float, float]]:
    """
    Parse Clay DSL text and compute layout.

    Args:
        clay_text: Multi-line string containing Clay DSL
        target_bbox: Override target bounding box (default: from DSL or (800, 600))
        verbose: Override verbose setting (default: from DSL or True)
        **weight_overrides: Override specific weights (e.g., straightness=10)

    Returns:
        Dictionary mapping node IDs to (x, y) positions

    Example:
        >>> text = '''
        ... user
        ... api "API Gateway"
        ... user -> api
        ... @bbox 800 600
        ... '''
        >>> positions = layout_from_text(text)
    """
    # Parse the DSL
    parsed = parse_clay_text(clay_text)

    # Extract settings from DSL
    graph_settings = _extract_graph_settings(parsed)

    # Apply function parameter overrides
    if target_bbox is not None:
        graph_settings['target_bbox'] = target_bbox
    if verbose is not None:
        graph_settings['verbose'] = verbose

    # Merge weight overrides
    graph_settings['weights'].update(weight_overrides)

    # Build nodes and edges
    nodes_dict = _build_nodes_dict(parsed, graph_settings['target_bbox'])
    edges = parsed.edges

    # Call the core layout engine
    # Note: Currently layout_graph() doesn't support weight overrides
    # We'll need to pass them through or modify the energy function
    positions = layout_graph(
        nodes_dict,
        edges,
        target_bbox=graph_settings['target_bbox'],
        verbose=graph_settings['verbose']
    )

    return positions


def layout_from_file(
    file_path: str,
    target_bbox: Optional[Tuple[float, float]] = None,
    verbose: Optional[bool] = None,
    **weight_overrides
) -> Dict[str, Tuple[float, float]]:
    """
    Load Clay DSL from file and compute layout.

    Args:
        file_path: Path to .clay file
        target_bbox: Override target bounding box
        verbose: Override verbose setting
        **weight_overrides: Override specific weights

    Returns:
        Dictionary mapping node IDs to (x, y) positions

    Example:
        >>> positions = layout_from_file('examples/architecture.clay')
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    clay_text = path.read_text(encoding='utf-8')

    return layout_from_text(clay_text, target_bbox, verbose, **weight_overrides)
