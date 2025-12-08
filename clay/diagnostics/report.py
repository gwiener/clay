from typing import Any

import numpy as np

from clay.graph import Graph, Layout
from clay.penalties import Penalty


def generate_diagnostic_report(
    graph: Graph,
    layout: Layout,
    penalties: list[Penalty],
    history: list[dict[str, float]] | None = None,
    top_n: int = 5,
) -> str:
    """Generate comprehensive diagnostic report for optimization results.

    Args:
        graph: The graph being laid out
        layout: The final layout
        penalties: List of penalty objects used in optimization
        history: Optional energy history from optimization
        top_n: Number of top contributors to show per penalty

    Returns:
        Formatted diagnostic report string
    """
    centers = np.array(layout.centers)
    lines = []

    lines.append("=" * 60)
    lines.append("OPTIMIZATION DIAGNOSTIC REPORT")
    lines.append("=" * 60)
    lines.append("")

    # 1. Energy Summary
    lines.append("## Energy Summary")
    lines.append("")

    total_energy = 0.0
    penalty_energies = {}

    for p in penalties:
        energy = p(centers)  # Weighted energy
        penalty_energies[p.__class__.__name__] = energy
        total_energy += energy

    lines.append(f"Total Energy: {total_energy:.2f}")
    lines.append("")
    lines.append("Per-Penalty Breakdown:")

    for name, energy in sorted(penalty_energies.items(), key=lambda x: -x[1]):
        pct = (energy / total_energy * 100) if total_energy > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"  {name:20s} {energy:8.2f} ({pct:5.1f}%) {bar}")

    lines.append("")

    # 2. Top Contributors Per Penalty
    lines.append("## Top Contributors")
    lines.append("")

    for p in penalties:
        name = p.__class__.__name__

        try:
            contributions = p.compute_contributions(centers)
        except NotImplementedError:
            lines.append(f"### {name}")
            lines.append("  (contribution analysis not available)")
            lines.append("")
            continue

        if not contributions:
            lines.append(f"### {name}")
            lines.append("  (no contributions - penalty is zero)")
            lines.append("")
            continue

        # Sort by energy descending
        sorted_contribs = sorted(contributions.items(), key=lambda x: -x[1])[:top_n]

        lines.append(f"### {name}")

        for key, energy in sorted_contribs:
            lines.append(f"  {str(key):40s} {energy:8.2f}")

        lines.append("")

    # 3. Conflict Analysis (if history available)
    if history and len(history) > 1:
        lines.append("## Conflict Analysis")
        lines.append("")

        conflicts = detect_conflicts(history)
        if conflicts:
            lines.append("Penalty pairs that frequently move in opposite directions:")
            for (p1, p2), count in sorted(conflicts.items(), key=lambda x: -x[1])[:5]:
                pct = count / (len(history) - 1) * 100
                lines.append(f"  {p1} vs {p2}: {count} times ({pct:.1f}%)")
        else:
            lines.append("  No significant conflicts detected.")

        lines.append("")

    # 4. Recommendations
    lines.append("## Recommendations")
    lines.append("")

    # Find dominant penalty
    if penalty_energies:
        dominant = max(penalty_energies.items(), key=lambda x: x[1])
        dominant_name, dominant_energy = dominant
        dominant_pct = (dominant_energy / total_energy * 100) if total_energy > 0 else 0

        if dominant_pct > 50:
            lines.append(f"⚠️  {dominant_name} dominates ({dominant_pct:.0f}% of total energy)")

            match dominant_name:
                case "Spacing":
                    lines.append("   → Consider increasing canvas size or reducing node sizes")
                case "NodeEdge":
                    lines.append("   → Edges are passing through nodes")
                    lines.append("   → Try --init ranked for better initial layout")
                case "EgdeCross":
                    lines.append("   → Many edge crossings")
                    lines.append("   → Graph may be inherently non-planar")
                case "ChainCollinearity":
                    lines.append("   → Chains are not aligned")
                    lines.append("   → Consider adjusting chain collinearity weight")
                case "Area":
                    lines.append("   → Layout is too spread out")
                    lines.append("   → Consider adjusting area weight")
        else:
            lines.append("✓ Energy is reasonably distributed across penalties")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def detect_conflicts(history: list[dict[str, float]]) -> dict[tuple[str, str], int]:
    """Detect iterations where penalties move in opposite directions.

    Returns:
        Dictionary mapping penalty pairs to conflict counts.
    """
    if len(history) < 2:
        return {}

    # Get penalty names (excluding 'Total' and 'accepted')
    penalty_names = [k for k in history[0].keys() if k not in ('Total', 'accepted', 'energy')]

    conflicts: dict[tuple[str, str], int] = {}

    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]

        for j, p1 in enumerate(penalty_names):
            for p2 in penalty_names[j + 1:]:
                delta1 = curr.get(p1, 0) - prev.get(p1, 0)
                delta2 = curr.get(p2, 0) - prev.get(p2, 0)

                # Conflict: one decreased significantly while other increased
                if delta1 * delta2 < 0 and abs(delta1) > 0.1 and abs(delta2) > 0.1:
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    conflicts[key] = conflicts.get(key, 0) + 1

    return conflicts
