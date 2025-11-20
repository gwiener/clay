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

### Option 1: Command-Line Interface (Recommended)

Create a `.clay` file with your diagram:

```clay
# my_diagram.clay
start "Start"
process "Process Data"
end "End"

start -> process -> end

@bbox 800 600
```

Render it to PNG or SVG:

```bash
# Render to PNG (default)
python -m clay my_diagram.clay

# Render to specific output
python -m clay my_diagram.clay -o output.png
python -m clay my_diagram.clay -o output.svg

# Show optimization statistics
python -m clay my_diagram.clay -s

# Save statistics to JSON file
python -m clay my_diagram.clay -s stats.json

# Render all examples
./examples.sh
```

### Option 2: Python API

```python
from clay import Node, layout_graph, render_graph_matplotlib

# Define nodes with IDs
nodes = {
    'A': Node('Start'),
    'B': Node('Process'),
    'C': Node('End'),
}

# Define edges
edges = [('A', 'B'), ('B', 'C')]

# Layout the graph
result = layout_graph(nodes, edges, target_bbox=(600, 400))

# Access positions and stats
print(f"Converged in {result.stats.iterations} iterations")
print(f"Final energy: {result.stats.final_energy}")

# Render
render_graph_matplotlib(nodes, edges, result.positions, 'my_graph.png')
```

## Clay DSL Reference

The Clay DSL provides a minimal, human-friendly syntax for defining diagrams.

### Basic Syntax

```clay
# Comments start with #

# Node declarations
node_id "Display Label"
api "API Gateway"          # Custom label
database                   # Auto-generated label: "Database"

# Node properties
web "Web Server" width=120 height=50 fontsize=14

# Edge declarations
user -> web -> api

# Chained edges (equivalent to multiple arrows)
api -> auth -> database

# Settings
@bbox 800 600                # Target bounding box
@verbose false               # Suppress optimization output
@weight straightness 10      # Tune energy weights
@weight edge_length 8
```

### Node Declaration

```clay
node_id                          # Auto-generated label from ID
node_id "Custom Label"           # Explicit label
node_id "Label" width=120        # With properties
```

**Auto-generated labels:**
- `api_gateway` → "Api Gateway"
- `user` → "User"
- `myNode` → "MyNode"

**Supported properties:**
- `width` - Node width (default: 80)
- `height` - Node height (default: 40)
- `fontsize` - Text size (default: 12)

### Edge Declaration

```clay
# Simple edge
a -> b

# Chained edges
a -> b -> c -> d

# Multiple separate edges
a -> b
a -> c
b -> d
```

### Settings

```clay
@bbox WIDTH HEIGHT           # Target bounding box (default: 800 600)
@verbose true|false          # Show optimization progress (default: true)
@weight NAME VALUE           # Override energy weights
```

## Optimization Statistics

Use the `--stats` flag to inspect optimization performance and energy breakdown:

```bash
# Print stats to stdout
python -m clay diagram.clay -s

# Save stats to JSON file
python -m clay diagram.clay -s stats.json
```

**Example output:**
```json
{
  "success": true,
  "iterations": 15,
  "function_evals": 1560,
  "final_energy": 5.53,
  "penalty_breakdown": {
    "overlap": 0.0,
    "edge_length": 0.4745,
    "straightness": 0.0847,
    "edge_node": 0.0,
    "bbox": 0.0,
    "area": 0.3631
  },
  "weights": {
    "overlap": 1000,
    "edge_length": 10,
    "straightness": 5,
    "edge_node": 200,
    "bbox": 100,
    "area": 1
  },
  "target_bbox": [800.0, 600.0],
  "message": "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH",
  "seed": null
}
```

**Understanding the stats:**
- `success`: Whether optimization converged successfully
- `iterations`: Number of optimization steps taken
- `function_evals`: Total energy function calls (includes gradient estimation)
- `final_energy`: Total weighted energy at solution
- `penalty_breakdown`: **Normalized** penalty values (scale-invariant, typically 0-2 range)
- `weights`: Multipliers applied to each penalty component
- `target_bbox`: Coordinate system dimensions used for layout
- `seed`: Random seed used for initialization (null = non-deterministic)
- `message`: Optimizer termination reason

**Note:** Penalty values are normalized by bounding box dimensions to ensure
scale-invariant optimization. Edge length and straightness are normalized by
diagonal length; area is normalized by target area.

**Available weights:**
- `straightness` - Encourages collinear A→B→C paths (default: 5)
- `edge_length` - Keeps connected nodes close (default: 10)
- `overlap` - Prevents node overlap (default: 1000)
- `bbox` - Constrains diagram size (default: 100)
- `area` - Minimizes total area (default: 1)

### Complete Example

```clay
# examples/architecture.clay

# Node declarations
user
database
cache

web "Web Server"
api "API Gateway"
auth "Auth Service"
logger "Audit Log" width=100 fontsize=12

# Edge declarations
user -> web -> api
api -> auth -> database
api -> cache
auth -> logger
api -> logger

# Settings
@bbox 800 600
@verbose true
```

## Python API Reference

### High-Level Functions (Recommended)

```python
from clay import render_from_file

render_from_file(input_file, output_file, target_bbox=None, verbose=None)
```

Complete workflow: parse `.clay` file, compute layout, and render.

**Args:**
- `input_file` (str): Path to `.clay` file
- `output_file` (str): Path to output file (`.png` or `.svg`)
- `target_bbox` (tuple, optional): Override `@bbox` from file
- `verbose` (bool, optional): Override `@verbose` from file

**Example:**
```python
render_from_file('diagram.clay', 'output.png')
render_from_file('diagram.clay', 'output.svg', target_bbox=(1000, 600))
```

### DSL Functions

```python
from clay import layout_from_text, layout_from_file

# Parse and layout from text
positions = layout_from_text(clay_text, target_bbox=(800, 600), verbose=True)

# Parse and layout from file
positions = layout_from_file('diagram.clay', target_bbox=(800, 600), verbose=True)
```

### Low-Level API

#### Node Class

```python
from clay import Node

Node(label, width=80, height=40, fontsize=12)
```

- `label` (str): Text to display in the node
- `width` (int): Node width in pixels
- `height` (int): Node height in pixels
- `fontsize` (int): Font size for label

#### layout_graph()

```python
from clay import layout_graph

layout_graph(nodes_dict, edges, target_bbox=(800, 600), verbose=True)
```

**Args:**
- `nodes_dict` (dict): `{node_id: Node}` mapping
- `edges` (list): List of `(from_id, to_id)` tuples
- `target_bbox` (tuple): `(width, height)` for bounding box
- `verbose` (bool): Print optimization progress

**Returns:**
- dict: `{node_id: (x, y)}` positions

#### Rendering Functions

```python
from clay import render_graph_matplotlib, render_graph_svg

render_graph_matplotlib(nodes_dict, edges, positions_dict, output_file='graph.png', target_bbox=(800, 600))
```
Renders to PNG using matplotlib (300 DPI, high quality).

```python
render_graph_svg(nodes_dict, edges, positions_dict, output_file='graph.svg')
```
Renders to SVG (vector format, infinitely scalable).

## Examples

See the `examples/` directory for complete `.clay` files demonstrating various patterns:

- **simple.clay** - Basic 3-node chain
- **chain.clay** - Linear flow A→B→C→D (tests straightness)
- **yshape.clay** - Y-shape fork pattern
- **cycle.clay** - Circular triangle A→B→C→A
- **diamond.clay** - Split-and-merge pattern
- **flowchart.clay** - Decision flow with error recovery
- **workflow.clay** - Complex workflow with custom node sizes
- **architecture.clay** - Web application architecture
- **state_machine.clay** - Order processing state machine with cycles

Run all examples:
```bash
./examples.sh
```

### Linear Chain

**DSL** (`examples/chain.clay`):
```clay
a "Start"
b "Process"
c "Decision"
d "End"

a -> b -> c -> d
```

**Python API**:
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

**DSL** (`examples/yshape.clay`):
```clay
a "Input"
b "Process"
c "Output 1"
d "Output 2"

a -> b
b -> c
b -> d
```

**Python API**:
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

**DSL** (`examples/cycle.clay`):
```clay
a "State A"
b "State B"
c "State C"

a -> b -> c -> a
```

**Python API**:
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

**DSL** (`examples/diamond.clay`):
```clay
a "Start"
b "Branch 1"
c "Branch 2"
d "Merge"
e "End"

a -> b
a -> c
b -> d
c -> d
d -> e
```

**Python API**:
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

**DSL**:
```clay
title "Main Title" width=200 height=60
small "Detail" width=60 height=30

title -> small
```

**Python API**:
```python
nodes = {
    'title': Node('Main Title', width=200, height=60),
    'small': Node('Detail', width=60, height=30),
}
```

### Meaningful Node IDs

**DSL**:
```clay
fetch_data "Fetch"
validate_input "Validate"
process_records "Process"

fetch_data -> validate_input -> process_records
```

**Python API**:
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

### Multiple Output Formats

**CLI**:
```bash
python -m clay diagram.clay -o output.png
python -m clay diagram.clay -o output.svg
```

**Python API**:
```python
from clay import render_from_file

# One-line render (recommended)
render_from_file('diagram.clay', 'output.png')

# Or manual workflow
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
