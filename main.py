import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

from clay.config import load_config
from clay.layout.engines import ENGINES, get_engine
from clay.render.matplot import render

if TYPE_CHECKING:
    from clay.layout.energy import Energy


def main():
    parser = argparse.ArgumentParser(description="Render graph layouts")
    parser.add_argument("input", help="YAML config file")
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
    args = parser.parse_args()

    input_path = Path(args.input)
    name = input_path.stem

    try:
        config = load_config(input_path)
        g = config.build_graph()
        penalties = config.penalties.bind_all(g)
        optimizer_config = config.optimizer
    except FileNotFoundError:
        print(f"Error: YAML file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading YAML config: {e}", file=sys.stderr)
        sys.exit(1)

    # Get and run the layout engine
    engine_class = get_engine(args.layout)
    match args.layout:
        case "energy":
            engine_class = cast("type[Energy]", engine_class)
            if args.init == "ranked":
                from clay.layout.ranked import Ranked
                ranked_result = Ranked().fit(g)
                init_layout = ranked_result.layout
            else:
                init_layout = None

            engine = engine_class(
                max_iter=optimizer_config.max_iter,
                ftol=optimizer_config.ftol,
                gtol=optimizer_config.gtol,
                init_layout=init_layout,
                optimizer=optimizer_config.method,
                niter=optimizer_config.niter,
                T=optimizer_config.T,
                stepsize=optimizer_config.stepsize,
                progress=args.progress,
            )
            result = engine.fit(g, penalties=penalties)
        case _:
            engine = engine_class()
            result = engine.fit(g)

    # Render output
    output_path = Path("output") / f"{name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render(result.layout, str(output_path), diagnose=args.diagnose)
    print(f"Successfully rendered graph from {args.input} using {args.layout} layout")

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
            report_path = Path("output") / f"{name}_report.md"
            generate_diagnostic_report(
                graph=g,
                layout=result.layout,
                penalties=penalties,
                history=history,
                optimization_result=result.metadata.get("optimization_result"),
                output_path=str(report_path),
            )

            # Save energy plot if we have history
            if history:
                energy_plot_path = Path("output") / f"{name}_energy.png"
                plot_energy_history(history, str(energy_plot_path))
        else:
            print("Diagnostic report not available (energy engine not used)")


if __name__ == "__main__":
    main()
