# Graph Layout Engine

An automatic diagram layout engine that creates compact, orderly graph layouts using constrained optimization.

## Features

- **Simple input format**: Define nodes by ID and edges as pairs
- **Disjoint nodes**: Hard constraint prevents overlapping
- **Connected nodes cluster**: Edge length penalty keeps related nodes close
- **Straight-through paths**: Special penalty encourages collinear A→B→C paths
- **Bounded and compact**: Fits diagrams within target bounding box
- **Handles cycles**: Works naturally with cyclic graphs

## Installation

```bash
pip install numpy scipy matplotlib
```

## Quick Start

```python
from graph_layout import Node, layout_graph, render_graph_matplotlib

# Define nodes with IDs
nodes = {
    'A': Node('Start'),
    'B': Node('Process'),
    'C': Node('End'),
}

# Define edges
edges = [('A', 'B'), ('B', 'C')]

# Layout the graph
positions = layout_graph(nodes, edges, target_bbox=(600, 400))

# Render
render_graph_matplotlib(nodes, edges, positions, 'my_graph.png')
```

## API Reference

### Node Class

```python
Node(label, width=80, height=40)
```

- `label` (str): Text to display in the node
- `width` (int): Node width in pixels
- `height` (int): Node height in pixels

### layout_graph()

```python
layout_graph(nodes_dict, edges, target_bbox=(800, 600), verbose=True)
```

**Args:**
- `nodes_dict` (dict): `{node_id: Node}` mapping
- `edges` (list): List of `(from_id, to_id)` tuples
- `target_bbox` (tuple): `(width, height)` for bounding box
- `verbose` (bool): Print optimization progress

**Returns:**
- dict: `{node_id: (x, y)}` positions

### Rendering Functions

```python
render_graph_matplotlib(nodes_dict, edges, positions_dict, output_file='graph.png')
```
Renders to PNG using matplotlib.

```python
render_graph_svg(nodes_dict, edges, positions_dict, output_file='graph.svg')
```
Renders to SVG (vector format, scalable).

## Examples

### Linear Chain
```python
nodes = {
    'A': Node('Start'),
    'B': Node('Process'),
    'C': Node('Decision'),
    'D': Node('End'),
}
edges = [('A', 'B'), ('B', 'C'), ('C', 'D')]
```
Result: Perfectly straight horizontal/vertical line.

### Y-Shape (Fork)
```python
nodes = {
    'A': Node('Input'),
    'B': Node('Process'),
    'C': Node('Output 1'),
    'D': Node('Output 2'),
}
edges = [('A', 'B'), ('B', 'C'), ('B', 'D')]
```
Result: Symmetric fork with branches at equal angles.

### Cycle
```python
nodes = {
    'A': Node('State A'),
    'B': Node('State B'),
    'C': Node('State C'),
}
edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
```
Result: Triangular layout with smooth flow.

### Diamond (Split-Merge)
```python
nodes = {
    'A': Node('Start'),
    'B': Node('Branch 1'),
    'C': Node('Branch 2'),
    'D': Node('Merge'),
    'E': Node('End'),
}
edges = [
    ('A', 'B'), ('A', 'C'),
    ('B', 'D'), ('C', 'D'),
    ('D', 'E'),
]
```
Result: Symmetric diamond with straight paths through.

## How It Works

The layout engine uses **constrained optimization** with multiple energy terms:

1. **Overlap Penalty** (weight: 1000): Hard constraint preventing node overlap
2. **Edge Length** (weight: 10): Keeps connected nodes close
3. **Straightness** (weight: 5): For A→B→C, penalizes if B is not collinear with A and C
4. **Bounding Box** (weight: 100): Penalizes exceeding target dimensions
5. **Area** (weight: 1): Minimizes total diagram area

The optimizer (L-BFGS-B) finds node positions that minimize total energy.

## Tuning Parameters

You can adjust the weights in `energy_function()`:

```python
# In graph_layout.py
W_OVERLAP = 1000      # Higher = stricter no-overlap
W_EDGE_LENGTH = 10    # Higher = tighter clustering
W_STRAIGHTNESS = 5    # Higher = straighter paths
W_BBOX = 100          # Higher = stricter size limit
W_AREA = 1            # Higher = more compact
```

## Advanced Usage

### Custom Node Sizes
```python
nodes = {
    'title': Node('Main Title', width=200, height=60),
    'small': Node('Detail', width=60, height=30),
}
```

### Meaningful Node IDs
```python
nodes = {
    'fetch_data': Node('Fetch'),
    'validate_input': Node('Validate'),
    'process_records': Node('Process'),
}
edges = [
    ('fetch_data', 'validate_input'),
    ('validate_input', 'process_records'),
]
```

### Multiple Renderings
```python
positions = layout_graph(nodes, edges)
render_graph_matplotlib(nodes, edges, positions, 'diagram.png')
render_graph_svg(nodes, edges, positions, 'diagram.svg')
```

## Limitations

- **Graph size**: Optimized for dozens of nodes (up to ~100)
- **Node shapes**: Currently only rectangles (circles could be added)
- **Edge routing**: Straight arrows (no curve routing)
- **Local minima**: May need multiple runs with different seeds for best results

## Future Enhancements

Possible additions:
- Circular nodes
- Curved edge routing
- User-specified node positions (partial constraints)
- Multiple random restarts for global optimum
- Hierarchical layouts for very large graphs
- Port-based edge connections
- Grid snapping

## License

MIT License - feel free to use and modify!

## Credits

Designed as a hybrid between Graphviz's ranked layouts and force-directed algorithms, with a novel "straightness penalty" for smooth visual flow.
