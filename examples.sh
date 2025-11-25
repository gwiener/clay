#!/bin/bash
#
# Clay Examples Renderer
# Renders all .clay files from examples/ directory to output/
#
# Usage:
#   ./examples.sh         # Uses default n_init=5
#   ./examples.sh 10      # Uses n_init=10
#   ./examples.sh 1       # Disables multi-start (single initialization)
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse optional n_init argument (default: 5)
N_INIT=${1:-5}

echo "========================================"
echo "Clay Examples - Rendering All Diagrams"
echo "========================================"
echo "Multi-start initializations: ${N_INIT}"
echo ""

# Create output directory
mkdir -p output

# Track statistics
total=0
success=0
failed=0

# Process all .clay files in examples/
for clay_file in examples/*.clay; do
    if [ -f "$clay_file" ]; then
        # Extract basename without extension
        basename=$(basename "$clay_file" .clay)
        output_file="output/${basename}.png"

        total=$((total + 1))

        echo -n "Rendering ${basename}.clay... "

        # Run clay CLI with stats output and n_init parameter
        if python -m clay "$clay_file" -o "$output_file" --n-init "$N_INIT" -s 2>&1 | head -n 5; then
            echo -e "${GREEN}✓${NC}"
            success=$((success + 1))
        else
            echo -e "${RED}✗${NC}"
            failed=$((failed + 1))
        fi
        echo ""
    fi
done

echo ""
echo "========================================"
echo "Summary:"
echo "  Total:   $total"
echo -e "  Success: ${GREEN}$success${NC}"
if [ $failed -gt 0 ]; then
    echo -e "  Failed:  ${RED}$failed${NC}"
fi
echo ""
echo "Output files in: output/"
echo "========================================"

# Exit with error code if any failed
[ $failed -eq 0 ]
