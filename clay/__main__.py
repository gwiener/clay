"""
Command-line interface for Clay diagram layout engine.

Usage:
    python -m clay input.clay                    # Outputs to input.png
    python -m clay input.clay -o output.svg      # Outputs to output.svg
    python -m clay input.clay -o dir/output.png  # Creates dir/ if needed
    python -m clay input.clay -s                 # Show stats on stdout
    python -m clay input.clay -s stats.json      # Save stats to file
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from clay.parser import render_from_file


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='clay',
        description='Automatic diagram layout engine using constrained optimization',
        epilog='Example: python -m clay examples/simple.clay -o output/diagram.png'
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Input .clay file containing DSL diagram specification'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        dest='output_file',
        default=None,
        help='Output file path (.png or .svg). Default: {input_basename}.png'
    )

    parser.add_argument(
        '-s', '--stats',
        nargs='?',
        const='',
        default=None,
        dest='stats_file',
        metavar='FILE',
        help='Output optimization statistics. If FILE provided, saves to file; otherwise prints to stdout'
    )

    parser.add_argument(
        '--init',
        type=str,
        choices=['spring', 'grid', 'random'],
        default='spring',
        help='Initialization strategy: spring (force-directed), grid (regular grid), or random (default: spring)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None = non-deterministic)'
    )

    parser.add_argument(
        '--n-init',
        type=int,
        default=None,
        dest='n_init',
        help='Number of random initializations for multi-start optimization (default: from DSL or 5, set to 1 to disable)'
    )

    args = parser.parse_args()

    # Determine output file path
    input_path = Path(args.input_file)

    if args.output_file:
        output_file = args.output_file
    else:
        # Auto-generate: foo.clay -> foo.png
        output_file = str(input_path.with_suffix('.png'))

    # Render the diagram
    try:
        result = render_from_file(
            args.input_file,
            output_file,
            init_mode=args.init,
            seed=args.seed,
            n_init=args.n_init
        )
        print(f"✓ Rendered {args.input_file} → {output_file}")

        # Handle stats output if requested
        if args.stats_file is not None:
            if args.stats_file == '':
                # Print human-readable format to stdout
                print("\n" + "=" * 70)
                print("OPTIMIZATION STATISTICS")
                print("=" * 70)
                print(f"Status: {'SUCCESS' if result.stats.success else 'FAILED'}")
                print(f"Iterations: {result.stats.iterations}")
                print(f"Function evaluations: {result.stats.function_evals}")
                print(f"Final energy: {result.stats.final_energy:.2f}")
                print(f"\nPenalty Breakdown (raw × weight = contribution):")
                print("-" * 70)

                total_check = 0.0
                for name, raw_value in result.stats.penalty_breakdown.items():
                    weight = result.stats.weights[name]
                    contribution = raw_value * weight
                    total_check += contribution

                    # Format: align columns nicely
                    print(f"  {name:15s}: {raw_value:12.4f} × {weight:4d} = {contribution:12.2f}")

                print("-" * 70)
                print(f"  {'Total':15s}: {'':<12s}   {'':<4s}   {total_check:12.2f}")
                print("=" * 70)
            else:
                # Save JSON to file
                stats_json = json.dumps(asdict(result.stats), indent=2)
                stats_path = Path(args.stats_file)
                stats_path.write_text(stats_json, encoding='utf-8')
                print(f"✓ Stats saved to {args.stats_file}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
