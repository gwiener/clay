#!/usr/bin/env python3
"""
Analyze convergence behavior across all example files.
Runs each example multiple times to check for consistency.
"""

import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json


def run_monitor(clay_file: str) -> Dict[str, any]:
    """Run monitor_layout.py and parse results."""
    # Use a temporary output file
    output_file = f"/tmp/test_{Path(clay_file).stem}.png"
    result = subprocess.run(
        ["python", "monitor_layout.py", clay_file, "-o", output_file, "--quiet"],
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr

    # Parse key metrics
    data = {
        "file": clay_file,
        "success": result.returncode == 0,
        "output": output
    }

    # Extract metrics using regex
    if match := re.search(r"Laying out (\d+) nodes with (\d+) edges", output):
        data["nodes"] = int(match.group(1))
        data["edges"] = int(match.group(2))

    if match := re.search(r"Optimization (converged|terminated)", output):
        data["status"] = match.group(1)

    if match := re.search(r"Final energy: ([\d.]+)", output):
        data["final_energy"] = float(match.group(1))

    if match := re.search(r"Total iterations: (\d+)", output):
        data["iterations"] = int(match.group(1))

    if match := re.search(r"Total position change: ([\d.]+)", output):
        data["position_change"] = float(match.group(1))

    if match := re.search(r"Average step size: ([\d.]+)", output):
        data["avg_step"] = float(match.group(1))

    if match := re.search(r"Max step size: ([\d.]+)", output):
        data["max_step"] = float(match.group(1))

    return data


def main():
    examples_dir = Path("examples")
    clay_files = sorted(examples_dir.glob("*.clay"))

    print("=" * 80)
    print("CONVERGENCE ANALYSIS REPORT")
    print("=" * 80)
    print()

    results = []

    # Run each example 3 times to check consistency
    for clay_file in clay_files:
        name = clay_file.stem
        print(f"\nAnalyzing: {name}")
        print("-" * 60)

        runs = []
        for i in range(3):
            data = run_monitor(str(clay_file))
            runs.append(data)

        # Aggregate results
        statuses = [r.get("status", "unknown") for r in runs]
        energies = [r.get("final_energy", 0) for r in runs]
        iterations = [r.get("iterations", 0) for r in runs]

        converged_count = statuses.count("converged")
        terminated_count = statuses.count("terminated")

        result_summary = {
            "name": name,
            "nodes": runs[0].get("nodes", 0),
            "edges": runs[0].get("edges", 0),
            "converged": converged_count,
            "terminated": terminated_count,
            "avg_energy": sum(energies) / len(energies),
            "min_energy": min(energies),
            "max_energy": max(energies),
            "avg_iterations": sum(iterations) / len(iterations),
            "energy_variance": max(energies) - min(energies),
            "consistent": len(set(statuses)) == 1
        }

        results.append(result_summary)

        # Print summary
        print(f"  Nodes: {result_summary['nodes']}, Edges: {result_summary['edges']}")
        print(f"  Status: {converged_count} converged, {terminated_count} terminated")
        print(f"  Energy: avg={result_summary['avg_energy']:.2f}, "
              f"range=[{result_summary['min_energy']:.2f}, {result_summary['max_energy']:.2f}], "
              f"variance={result_summary['energy_variance']:.2f}")
        print(f"  Iterations: avg={result_summary['avg_iterations']:.1f}")
        print(f"  Consistent: {result_summary['consistent']}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    consistent_examples = [r for r in results if r["consistent"]]
    inconsistent_examples = [r for r in results if not r["consistent"]]

    always_converged = [r for r in results if r["converged"] == 3]
    always_terminated = [r for r in results if r["terminated"] == 3]

    print(f"\nTotal examples: {len(results)}")
    print(f"Consistent behavior: {len(consistent_examples)}")
    print(f"Inconsistent behavior: {len(inconsistent_examples)}")
    print(f"Always converge: {len(always_converged)}")
    print(f"Always terminate: {len(always_terminated)}")

    if inconsistent_examples:
        print("\nInconsistent examples (non-deterministic):")
        for r in inconsistent_examples:
            print(f"  - {r['name']}: {r['converged']} converged, {r['terminated']} terminated")

    # Energy analysis
    high_energy = [r for r in results if r["avg_energy"] > 1.0]
    if high_energy:
        print("\nHigh energy examples (avg > 1.0):")
        for r in sorted(high_energy, key=lambda x: x["avg_energy"], reverse=True):
            print(f"  - {r['name']}: {r['avg_energy']:.2f}")

    # Variance analysis
    high_variance = [r for r in results if r["energy_variance"] > 0.5]
    if high_variance:
        print("\nHigh energy variance (> 0.5):")
        for r in sorted(high_variance, key=lambda x: x["energy_variance"], reverse=True):
            print(f"  - {r['name']}: variance={r['energy_variance']:.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
