# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**clay** is an automatic diagram layout engine that uses constrained optimization to create compact, orderly graph layouts. The key innovation is a "straightness penalty" that encourages collinear node placement in A→B→C paths, creating smooth visual flow without rigid hierarchical constraints.

## Development Setup

```bash
# Install dependencies using uv (Python 3.13+)
uv sync

# Or using pip
pip install -r pyproject.toml

# Run the built-in examples
python graph_layout.py

# Run additional usage examples
python usage_example.py
```

## Core Architecture

### Main Module: `graph_layout.py`

The single-file implementation (~580 lines) is organized into distinct sections:

1. **Node Class** (lines 15-20)
   - Simple data container for label, width, height

2. **Geometry Utilities** (lines 23-110)
   - `rect_edge_point()`: Calculates arrow connection points on rectangle boundaries
   - `point_to_line_distance()`: Measures deviation from collinearity

3. **Energy Function Components** (lines 113-258)
   - `overlap_penalty()`: Prevents node overlap (weight: 1000)
   - `edge_length_penalty()`: Keeps connected nodes close (weight: 10)
   - `straightness_penalty()`: Encourages A→B→C collinearity (weight: 5) - **KEY INNOVATION**
   - `bounding_box_penalty()`: Constrains diagram size (weight: 100)
   - `area_penalty()`: Minimizes layout area (weight: 1)

4. **Main Energy Function** (lines 261-295)
   - `energy_function()`: Combines all penalties with tunable weights

5. **Layout Engine** (lines 298-368)
   - `layout_graph()`: Main API - uses scipy's L-BFGS-B optimizer
   - Input: `{node_id: Node}` dict and `[(from_id, to_id)]` edges
   - Output: `{node_id: (x, y)}` positions

6. **Rendering** (lines 371-536)
   - `render_graph_matplotlib()`: PNG output with matplotlib
   - `render_graph_svg()`: Vector SVG output

### Design Principles

- **Optimization-based**: Uses scipy.optimize.minimize with L-BFGS-B
- **Constraint-free nodes**: No pre-assigned ranks or layers
- **Cycle-friendly**: Naturally handles cyclic graphs
- **Local straightness**: Straightness penalty is local (per triplet) not global
- **Meaningful IDs**: Node IDs are user-defined strings, not indices

## Common Patterns

### Adding New Node Shapes

Currently only rectangles are supported. To add circles:
1. Modify `Node` class to include `shape` parameter
2. Update overlap penalty calculations for circle geometry
3. Update rendering functions for both matplotlib and SVG

### Tuning Layout Behavior

Energy weights are defined in `energy_function()` (lines 282-286):
```python
W_OVERLAP = 1000      # Hard constraint - keep high
W_EDGE_LENGTH = 10    # Higher = tighter clustering
W_STRAIGHTNESS = 5    # Higher = straighter paths
W_BBOX = 100          # Higher = stricter size limit
W_AREA = 1            # Higher = more compact
```

### Creating Custom Diagrams

Follow the pattern in `usage_example.py`:
1. Define nodes dict with meaningful IDs
2. Define edges as list of ID tuples
3. Call `layout_graph()` with target bounding box
4. Render using both PNG and SVG for flexibility

## Testing

Currently no automated test suite. To verify changes:
```bash
# Run all examples and visually inspect outputs
python graph_layout.py
python usage_example.py

# Check that output files are generated:
# - example_*.png/svg (5 pairs)
# - architecture.png/svg
# - state_machine.png/svg
# - flowchart.png/svg
```

## File Structure

```
clay/
├── graph_layout.py      # Core implementation + examples
├── usage_example.py     # Additional practical examples
├── hello.py             # Simple test script (can be ignored)
├── pyproject.toml       # Dependencies (numpy, scipy, matplotlib)
├── README.md            # User documentation
├── BUILD_SUMMARY.md     # Development notes and analysis
└── .venv/               # Virtual environment (gitignored)
```

## Dependencies

- **numpy** (>=2.3.5): Array operations and linear algebra
- **scipy** (>=1.16.3): L-BFGS-B optimizer
- **matplotlib** (>=3.10.7): PNG rendering

## Known Limitations

- Optimized for dozens of nodes (up to ~100)
- Only rectangular node shapes
- Straight-line arrows only (no curve routing)
- May converge to local minima (no multi-start implemented)
- No interactive editing capabilities

## Performance Characteristics

- 3-4 nodes: ~30-70 iterations, <1 second
- 8 nodes: ~70 iterations, ~1 second
- Complexity grows as O(n²) due to overlap checks
