import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from clay.layout import get_engine, ENGINES
from clay.render.matplot import render


def main():
    parser = argparse.ArgumentParser(description="Render graph layouts")
    parser.add_argument("module_name", help="Example module name (from examples package)")
    parser.add_argument(
        "--layout", "-l",
        default="energy",
        choices=list(ENGINES.keys()),
        help="Layout engine to use (default: energy)"
    )
    args = parser.parse_args()

    module_name = args.module_name

    # Load module from examples package
    try:
        module = importlib.import_module(f"examples.{module_name}")
    except ModuleNotFoundError:
        print(f"Error: Module 'examples.{module_name}' not found", file=sys.stderr)
        sys.exit(1)

    # Check if module has a graph member
    if not hasattr(module, "graph"):
        print(f"Error: Module 'examples.{module_name}' does not have a 'graph' member", file=sys.stderr)
        sys.exit(1)

    g = module.graph

    # Get and run the layout engine
    engine_class = get_engine(args.layout)
    engine = engine_class()
    result = engine.fit(g)

    # Render output
    output_path = Path("output") / f"{module_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render(result.layout, str(output_path))
    print(f"Successfully rendered graph from examples.{module_name} using {args.layout} layout")

    # Print engine-specific metadata
    if "optimization_result" in result.metadata:
        print(f"Optimization details: {result.metadata['optimization_result']}")

    if "history" in result.metadata:
        df = pd.DataFrame(result.metadata["history"])
        print("Energy history:")
        print(df.tail(10))


if __name__ == "__main__":
    main()
