#!/bin/bash
#
# Clay Examples Renderer
# Renders all .clay files from examples/ directory to output/
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "Clay Examples - Rendering All Diagrams"
echo "========================================"
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

        # Run clay CLI with stats output
        if python -m clay "$clay_file" -o "$output_file" -s 2>&1 | head -n 5; then
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
