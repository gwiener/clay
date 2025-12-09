import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from clay.config import load_config
from clay.layout.engines import ENGINES, get_engine
from clay.render.matplot import render


def main():
    parser = argparse.ArgumentParser(description="Render graph layouts")
    parser.add_argument("input", help="YAML config file or example module name")
    parser.add_argument(
        "--layout", "-l",
        default="energy",
        choices=list(ENGINES.keys()),
        help="Layout engine to use (default: energy)"
    )
    parser.add_argument(
        "--init", "-i",
        default="random",
        choices=["random", "ranked"],
        help="Initialization method for energy layout (default: random)"
    )
    parser.add_argument(
        "--progress", "-p",
        action="store_true",
        help="Show optimization progress bar"
    )
    parser.add_argument(
        "--diagnose", "-d",
        action="store_true",
        help="Generate diagnostic report for optimization"
    )
    parser.add_argument(
        "--module", "-m",
        action="store_true",
        help="Treat input as module name from examples package (legacy)"
    )
    args = parser.parse_args()

    input_arg = args.input
    penalties = None

    # Determine if input is YAML or module
    if args.module or not input_arg.endswith('.yaml'):
        # Legacy: load from examples package
        module_name = input_arg
        try:
            module = importlib.import_module(f"examples.{module_name}")
        except ModuleNotFoundError:
            print(f"Error: Module 'examples.{module_name}' not found", file=sys.stderr)
            sys.exit(1)

        if not hasattr(module, "graph"):
            print(f"Error: Module 'examples.{module_name}' does not have a 'graph' member", file=sys.stderr)
            sys.exit(1)

        g = module.graph
    else:
        # New: load from YAML
        module_name = Path(input_arg).stem
        try:
            config = load_config(input_arg)
            g = config.build_graph()
            penalties = config.penalties.bind_all(g)
        except FileNotFoundError:
            print(f"Error: YAML file '{input_arg}' not found", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading YAML config: {e}", file=sys.stderr)
            sys.exit(1)

    # Get and run the layout engine
    engine_class = get_engine(args.layout)
    if args.layout == "energy":
        if args.init == "ranked":
            from clay.layout.ranked import Ranked
            ranked_result = Ranked().fit(g)
            engine = engine_class(init_layout=ranked_result.layout, progress=args.progress)
        else:
            engine = engine_class(progress=args.progress)
    else:
        engine = engine_class()

    result = engine.fit(g, penalties=penalties)
    # Render output
    output_path = Path("output") / f"{module_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render(result.layout, str(output_path), diagnose=args.diagnose)
    print(f"Successfully rendered graph from examples.{module_name} using {args.layout} layout")

    # Print engine-specific metadata
    if "optimization_result" in result.metadata:
        print(f"Optimization details: {result.metadata['optimization_result']}")

    if "history" in result.metadata:
        df = pd.DataFrame(result.metadata["history"])
        print("Energy history:")
        print(df.tail(10))

    # Generate diagnostic report if requested
    if args.diagnose:
        from clay.diagnostics import generate_diagnostic_report, plot_energy_history

        penalties = result.metadata.get("penalties")
        history = result.metadata.get("history", [])

        if penalties:
            report_path = Path("output") / f"{module_name}_report.md"
            generate_diagnostic_report(
                graph=g,
                layout=result.layout,
                penalties=penalties,
                history=history,
                output_path=str(report_path),
            )

            # Save energy plot if we have history
            if history:
                energy_plot_path = Path("output") / f"{module_name}_energy.png"
                plot_energy_history(history, str(energy_plot_path))
        else:
            print("Diagnostic report not available (energy engine not used)")


if __name__ == "__main__":
    main()
