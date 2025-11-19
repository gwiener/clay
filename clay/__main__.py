"""
Command-line interface for Clay diagram layout engine.

Usage:
    python -m clay input.clay                    # Outputs to input.png
    python -m clay input.clay -o output.svg      # Outputs to output.svg
    python -m clay input.clay -o dir/output.png  # Creates dir/ if needed
"""

import argparse
import sys
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
        render_from_file(args.input_file, output_file)
        print(f"✓ Rendered {args.input_file} → {output_file}")
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
