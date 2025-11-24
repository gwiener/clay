#!/usr/bin/env python3
"""
Optimization monitoring script for clay layout engine.

This script demonstrates how to use a callback function to monitor
the optimization process, tracking energy values and iteration counts.

Usage:
    python monitor_layout.py examples/simple.clay -o output/monitored.png
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from clay.parser import parse_clay_text, layout_from_text
from clay.layout import render_graph_matplotlib, render_graph_svg


class OptimizationMonitor:
    """Callback monitor for scipy optimization."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.iteration = 0
        self.energies: list[float] = []
        self.positions: list[np.ndarray] = []

    def __call__(self, xk: np.ndarray, *args: Any) -> None:
        """
        Callback function called at each optimization iteration.

        Args:
            xk: Current position vector [x1, y1, x2, y2, ...]
        """
        self.iteration += 1
        self.positions.append(xk.copy())

        # Note: We don't have access to the energy value in scipy's callback,
        # so we'll need to evaluate it separately if needed

        if self.verbose:
            print(f"Iteration {self.iteration}: position updated")

    def summary(self) -> None:
        """Print optimization summary statistics."""
        print(f"\n{'='*60}")
        print(f"Optimization Summary")
        print(f"{'='*60}")
        print(f"Total iterations: {self.iteration}")
        print(f"Positions tracked: {len(self.positions)}")

        if len(self.positions) >= 2:
            # Calculate movement magnitude between first and last position
            initial = self.positions[0]
            final = self.positions[-1]
            movement = np.linalg.norm(final - initial)
            print(f"Total position change: {movement:.2f}")

            # Calculate average step size
            steps = [np.linalg.norm(self.positions[i+1] - self.positions[i])
                    for i in range(len(self.positions)-1)]
            print(f"Average step size: {np.mean(steps):.2f}")
            print(f"Max step size: {np.max(steps):.2f}")
            print(f"Min step size: {np.min(steps):.2f}")
        print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor clay layout optimization process"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to .clay input file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (.png or .svg). Defaults to {input_basename}.png"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress iteration output (still shows summary)"
    )

    args = parser.parse_args()

    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix != ".clay":
        print(f"Error: Input file must be a .clay file", file=sys.stderr)
        sys.exit(1)

    clay_text = input_path.read_text()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"{input_path.stem}.png")

    # Validate output format
    if output_path.suffix not in [".png", ".svg"]:
        print(f"Error: Output format must be .png or .svg", file=sys.stderr)
        sys.exit(1)

    print(f"Monitoring layout optimization for: {input_path}")
    print(f"Output: {output_path}\n")

    # Create monitor
    monitor = OptimizationMonitor(verbose=not args.quiet)

    # Parse the clay file
    parsed = parse_clay_text(clay_text)

    # Import layout_graph and helper functions
    from clay.layout import layout_graph
    from clay.parser import _extract_graph_settings, _build_nodes_dict

    # Extract settings from parsed diagram
    graph_settings = _extract_graph_settings(parsed)

    # Build nodes dict
    nodes_dict = _build_nodes_dict(parsed, graph_settings['target_bbox'])
    edges = parsed.edges

    # Perform layout with monitoring
    print("Starting optimization...\n")
    result = layout_graph(
        nodes_dict=nodes_dict,
        edges=edges,
        target_bbox=graph_settings['target_bbox'],
        verbose=graph_settings['verbose'],
        init_mode=graph_settings.get('init_mode', 'spring'),
        seed=graph_settings.get('seed', None),
        callback=monitor
    )

    # Show summary
    monitor.summary()

    # Render output
    print(f"Rendering to {output_path}...")
    if output_path.suffix == ".png":
        render_graph_matplotlib(
            nodes_dict=nodes_dict,
            edges=edges,
            positions_dict=result.positions,
            output_file=str(output_path),
            target_bbox=graph_settings['target_bbox']
        )
    else:  # .svg
        render_graph_svg(
            nodes_dict=nodes_dict,
            edges=edges,
            positions_dict=result.positions,
            output_file=str(output_path)
        )

    print(f"✓ Complete! Converged in {result.stats.iterations} iterations")
    print(f"  Final energy: {result.stats.final_energy:.2f}")


if __name__ == "__main__":
    main()
