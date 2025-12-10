# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLay (Claude's Layout) is an automatic graph layout engine that optimizes node placement using penalty-based energy minimization. It uses SciPy's optimizers (L-BFGS-B and basinhopping) to minimize a weighted sum of penalty functions that encode layout quality constraints.

## Examples
Examples are located in the `examples/` folder. They are divided to *easy* and *hard*.
Easy examples are simple graphs that are very likely to be laid out quickly and successfully and give low energy and aesthetic results.
Use easy examples for sanity checks after simple code changes.
Hard examples are challenges that this project aims at solving. They are hard to optimize, may fail, and may give high energy or poor results.
Use hard examples when making changes to the layout algorithm itself that are meant to solve optimization challenges.

## Commands

```bash
# Run layout on a YAML config file
uv run python main.py examples/.../some-example.yaml

# With diagnostic report
uv run python main.py examples/.../some-example.yaml --diagnose

# Use ranked initialization instead of random
uv run python main.py examples/.../some-example.yaml --init ranked

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/penalties/test_spacing.py

# Profile with scalene
uv run scalene main.py examples/.../some-example.yaml
```

The main script also supports a `--progress` option that shows a `tqdm` progress bar.
You do not need to use this option by default since it produces a lot of useless output lines.
Use it only if you need to debug a stuck run.

## Architecture

### Core Data Model (`clay/graph.py`)
- `Node`: Graph node with name and computed dimensions based on font metrics
- `Edge`: Directed edge with optional `flow` flag (used by ChainCollinearity penalty)
- `Graph`: Container holding nodes, edges, canvas dimensions, and adjacency lookups
- `Layout`: Maps a graph to optimized center coordinates as flat list `[x0, y0, x1, y1, ...]`

### Layout Engines (`clay/layout/`)
- `LayoutEngine`: Abstract base class defining `fit(graph) -> Result`
- `Energy`: Main engine using SciPy optimization with configurable penalties
- `Ranked`: Sugiyama-style layered layout (can be used as initialization)
- `Random`: Random placement within canvas bounds

### Penalty System (`clay/penalties/`)
Penalties are weighted cost functions summed into a total energy. Each extends `Penalty` or `LocalPenalty`:
- `Spacing`: Attraction for connected nodes, repulsion for non-connected (spring-like)
- `NodeEdge`: Penalizes edges passing through nodes
- `EdgeCross`: Penalizes edge intersections
- `Area`: Encourages compact layouts
- `ChainCollinearity`: Keeps flow chains (A→B→C) aligned

Penalties implement `compute(centers) -> float` and optionally `compute_contributions()` for diagnostics.

### Configuration (`clay/config.py`)
YAML configuration using Pydantic models. Each penalty has a config class with a `bind(graph)` method that instantiates the penalty with configured weights.

### Output
- Rendered PNG to `output/`
- Diagnostic report (markdown) and energy history plot when `--diagnose` is used

## Key Patterns

- Node positions are stored as flat numpy arrays: `[x0, y0, x1, y1, ...]`
- Access node index via `graph.name2idx[node_name]`, then `centers[idx*2]` for x, `centers[idx*2+1]` for y
- Penalty weights (`w`) scale the raw computed values in the total energy
- Use `match-case` for branching logic (Python 3.10+)
