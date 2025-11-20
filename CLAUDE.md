# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**clay** is an automatic diagram layout engine that uses constrained optimization to create compact, orderly graph layouts. The key innovation is a "straightness penalty" that encourages collinear node placement in A→B→C paths, creating smooth visual flow without rigid hierarchical constraints.

The project provides:
1. **Python package** (`clay/`) - Core layout engine with programmatic API
2. **DSL parser** - Human-friendly `.clay` text format for defining diagrams
3. **CLI tool** - Command-line interface via `python -m clay`
4. **Examples** - 9 `.clay` example files demonstrating various patterns

## Task Management

### BACKLOG.md (Project-Level Tasks)

**IMPORTANT**: Always maintain `BACKLOG.md` to track work and provide continuity across sessions.

This is the **persistent project backlog** - use terminology like:
- "add this to the backlog"
- "show me the backlog"
- "what's in the backlog?"

#### When to Update BACKLOG.md

1. **User adds a task**: When the user says "add to backlog" or requests a future task
   - Add to appropriate section (Next Up, Backlog, or Ideas/Future)
   - Use markdown checkboxes: `- [ ] Task description`

2. **Starting work on a task**: When beginning implementation
   - Move task from its section to "In Progress"
   - Keep only actively worked tasks in "In Progress" (1-3 items max)

3. **Completing a task**: When finishing implementation and committing
   - Move task to "Done (Recent)" with completion date
   - Format: `- [x] Task description (YYYY-MM-DD)`
   - Keep ~10 most recent completed tasks for context

4. **User asks about tasks**: When user asks "what's next?" or "show me the backlog"
   - Read BACKLOG.md and present current tasks
   - Suggest prioritization based on "Next Up" section

#### Task Organization

- **In Progress**: Currently being worked on (limit 1-3)
- **Next Up (High Priority)**: Clear next steps, ready to implement
- **Backlog**: Important but not urgent
- **Ideas / Future**: Experimental, research, or long-term features
- **Done (Recent)**: Last ~10 completed tasks with dates

#### Example Workflow

```markdown
# User: "Add support for colored nodes to the backlog"
# Claude: Adds to BACKLOG.md Backlog section

# User: "Let's implement colored nodes"
# Claude: Moves task to "In Progress" in BACKLOG.md, begins implementation

# User: "Commit that work"
# Claude: After commit, moves task to "Done (Recent)" in BACKLOG.md with date
```

### Session Task List (Ephemeral)

Separate from BACKLOG.md, Claude uses an internal **TodoWrite tool** to track tasks during the current session only. This is ephemeral and not persisted to any file. Users will see this displayed while work is in progress.

## Development Setup

```bash
# Install dependencies using uv (Python 3.13+)
uv sync

# Or using pip
pip install -r pyproject.toml

# Run all examples
./examples.sh

# Run specific example
python -m clay examples/simple.clay -o output/test.png
```

## Core Architecture

### Package Structure

```
clay/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point (python -m clay)
├── layout.py            # Core layout engine (was graph_layout.py)
└── parser.py            # DSL parser and high-level functions

examples/
├── simple.clay          # Basic examples
├── architecture.clay    # ...9 total example files
└── ...

examples.sh              # Shell script to render all examples
```

### Module: `clay/layout.py`

Core layout engine (~580 lines) with distinct sections:

1. **Node Class**
   - Data container: `Node(label, width, height, fontsize, target_bbox)`
   - Auto-calculates text dimensions based on target_bbox

2. **Geometry Utilities**
   - `rect_edge_point()`: Arrow connection points on rectangle boundaries
   - `point_to_line_distance()`: Measures deviation from collinearity

3. **Energy Function Components** (all normalized for scale-invariance)
   - `overlap_penalty()`: Prevents node overlap (weight: 1000)
   - `edge_length_penalty()`: Keeps connected nodes close (weight: 10) - **normalized by bbox diagonal**
   - `straightness_penalty()`: Encourages A→B→C collinearity (weight: 5) - **KEY INNOVATION** - **normalized by bbox diagonal**
   - `edge_node_intersection_penalty()`: Prevents edges crossing through nodes (weight: 200)
   - `bounding_box_penalty()`: Constrains diagram size (weight: 100)
   - `area_penalty()`: Minimizes layout area (weight: 1) - **normalized by bbox area**

4. **Main Functions**
   - `energy_function()`: Combines all penalties with tunable weights
   - `layout_graph()`: Main API - uses scipy's L-BFGS-B optimizer
   - `render_graph_matplotlib()`: PNG output (300 DPI)
   - `render_graph_svg()`: Vector SVG output

### Module: `clay/parser.py`

DSL parser and high-level API (~490 lines):

1. **Regex Patterns**
   - `NODE_PATTERN`: Matches node declarations
   - `EDGE_PATTERN`: Matches arrow operators
   - `SETTING_PATTERN`: Matches @directives
   - `PROPERTY_PATTERN`: Matches key=value pairs

2. **Parsing Functions**
   - `parse_clay_text()`: Main parser, returns `ParsedDiagram`
   - `parse_node_line()`: Extracts node ID, label, properties
   - `parse_edge_line()`: Handles chained edges (a -> b -> c)
   - `parse_setting_line()`: Extracts @bbox, @verbose, @weight

3. **High-Level API**
   - `layout_from_text()`: Parse DSL text and compute layout
   - `layout_from_file()`: Parse .clay file and compute layout
   - `render_from_file()`: Complete workflow - parse, layout, render

### Module: `clay/__main__.py`

CLI entry point (~70 lines):

- Uses `argparse` for command-line arguments
- Required: `input_file` (.clay file path)
- Optional: `-o/--output` (defaults to `{input_basename}.png`)
- Auto-detects format from extension (.png or .svg)
- Wraps `render_from_file()` with user-friendly error messages

### Design Principles

- **Optimization-based**: Uses scipy.optimize.minimize with L-BFGS-B
- **Constraint-free nodes**: No pre-assigned ranks or layers
- **Cycle-friendly**: Naturally handles cyclic graphs
- **Local straightness**: Straightness penalty is local (per triplet) not global
- **Meaningful IDs**: Node IDs are user-defined strings, not indices
- **Declarative DSL**: Human-friendly text format with minimal syntax

## Common Patterns

### Adding New DSL Features

To add new node properties (e.g., `color`):
1. Update `PROPERTY_PATTERN` in `clay/parser.py` if needed
2. Modify `_parse_properties()` to handle new property type
3. Update `Node` class in `clay/layout.py` to accept new parameter
4. Update rendering functions to use the new property

### Adding New DSL Settings

To add new @directives (e.g., `@margin`):
1. Add parsing logic in `parse_setting_line()` in `clay/parser.py`
2. Update `_extract_graph_settings()` to handle the new setting
3. Pass the setting through to `layout_graph()` or renderers
4. Update README.md with documentation

### Tuning Layout Behavior

Energy weights are defined in `energy_function()` in `clay/layout.py`:
```python
W_OVERLAP = 1000      # Hard constraint - keep high
W_EDGE_LENGTH = 10    # Higher = tighter clustering
W_STRAIGHTNESS = 5    # Higher = straighter paths
W_EDGE_NODE = 200     # Prevent edges crossing nodes
W_BBOX = 100          # Higher = stricter size limit
W_AREA = 1            # Higher = more compact
```

Users can override via DSL: `@weight straightness 10`

**Important:** All distance-based penalties (edge_length, straightness) are normalized
by the target bounding box diagonal, and area_penalty is normalized by target bbox area.
This ensures scale-invariant optimization regardless of coordinate system size.

### Creating Custom Diagrams

**Recommended (DSL):**
1. Create a `.clay` file with nodes, edges, and settings
2. Run `python -m clay mydiagram.clay -o output.png`

**Programmatic (Python API):**
1. Define nodes dict with meaningful IDs
2. Define edges as list of ID tuples
3. Call `layout_graph()` with target bounding box
4. Render using `render_graph_matplotlib()` or `render_graph_svg()`

Or use the high-level API:
```python
from clay import render_from_file
render_from_file('diagram.clay', 'output.png')
```

## Testing

Currently no automated test suite. To verify changes:

```bash
# Run all 9 examples and visually inspect outputs
./examples.sh

# Test specific example
python -m clay examples/simple.clay -o output/test.png

# Test CLI error handling
python -m clay nonexistent.clay  # Should show error
python -m clay examples/simple.clay -o test.pdf  # Should reject .pdf

# Verify all outputs generated in output/
ls -lh output/*.png
```

**Visual inspection checklist:**
- No overlapping nodes
- Connected nodes are reasonably close
- A→B→C paths appear relatively straight
- Diagrams fit within intended bounds
- Arrows point to correct node edges

## File Structure

```
clay/                        # Main package
├── __init__.py              # Public API exports
├── __main__.py              # CLI entry point (70 lines)
├── layout.py                # Core layout engine (580 lines)
└── parser.py                # DSL parser (490 lines)

examples/                    # Example .clay files
├── simple.clay              # Basic 3-node chain
├── chain.clay               # Linear A→B→C→D
├── yshape.clay              # Y-shape fork
├── cycle.clay               # Circular triangle
├── diamond.clay             # Split-and-merge
├── flowchart.clay           # Decision flow
├── workflow.clay            # Complex workflow
├── architecture.clay        # Web application
└── state_machine.clay       # Order processing FSM

examples.sh                  # Shell script to render all examples
output/                      # Rendered diagrams (gitignored)
pyproject.toml               # Dependencies (numpy, scipy, matplotlib)
README.md                    # User documentation (see for API details)
CLAUDE.md                    # This file - development guidance
BACKLOG.md                   # Project backlog and work-in-progress
BUILD_SUMMARY.md             # Development notes and analysis
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

## DSL Syntax Quick Reference

For full documentation, see README.md. Quick syntax overview:

```clay
# Comments
node_id                          # Auto-generated label
node_id "Custom Label"           # Explicit label
node_id "Label" width=120        # With properties

# Edges
a -> b                           # Simple edge
a -> b -> c                      # Chained edges

# Settings
@bbox 800 600                    # Target dimensions
@verbose false                   # Suppress output
@weight straightness 10          # Tune energy weights
```

**Common DSL modifications:**
- Add node: Add line in nodes section, reference in edges
- Add edge: Add `from -> to` line anywhere after nodes defined
- Change layout: Modify `@bbox` or `@weight` settings
- Change labels: Update quoted strings after node IDs

## Public API Surface

See `clay/__init__.py` for exported functions:

**High-level (recommended):**
- `render_from_file(input_file, output_file)` - Complete workflow

**DSL functions:**
- `layout_from_text(text)` - Parse DSL text, return positions
- `layout_from_file(path)` - Parse .clay file, return positions

**Low-level:**
- `Node(label, width, height, fontsize)` - Node data class
- `layout_graph(nodes, edges, target_bbox)` - Core optimizer
- `render_graph_matplotlib(...)` - PNG renderer
- `render_graph_svg(...)` - SVG renderer
