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

    total_weighted = 0.0
    penalty_data: list[tuple[str, float, float, float]] = []  # (name, unweighted, weighted, weight)

    for p in penalties:
        unweighted = p.compute(centers)
        weighted = p(centers)
        penalty_data.append((p.__class__.__name__, unweighted, weighted, p.w))
        total_weighted += weighted

    lines.append(f"Total Weighted Energy: {total_weighted:.2f}")
    lines.append("")
    lines.append("Per-Penalty Breakdown:")
    lines.append(f"  {'Penalty':20s} {'Unweighted':>10s}  {'Weighted':>10s}   (%)")

    for name, unweighted, weighted, _ in sorted(penalty_data, key=lambda x: -x[2]):
        pct = (weighted / total_weighted * 100) if total_weighted > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"  {name:20s} {unweighted:10.2f}  {weighted:10.2f}  ({pct:5.1f}%) {bar}")

    lines.append("")

    # 2. Top Contributors Per Penalty
    lines.append("## Top Contributors")
    lines.append("")

    for p in penalties:
        name = p.__class__.__name__
        w = p.w

        try:
            contributions = p.compute_contributions(centers)
        except NotImplementedError:
            lines.append(f"### {name} (w={w})")
            lines.append("  (contribution analysis not available)")
            lines.append("")
            continue

        if not contributions:
            lines.append(f"### {name} (w={w})")
            lines.append("  (no contributions - penalty is zero)")
            lines.append("")
            continue

        # Sort by energy descending (unweighted)
        sorted_contribs = sorted(contributions.items(), key=lambda x: -x[1])[:top_n]

        lines.append(f"### {name} (w={w})")
        lines.append(f"  {'':40s} {'Unweighted':>10s}  {'Weighted':>10s}")

        for key, unweighted in sorted_contribs:
            weighted = unweighted * w
            lines.append(f"  {str(key):40s} {unweighted:10.2f}  {weighted:10.2f}")

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

    # Find dominant penalty (by weighted energy)
    if penalty_data:
        dominant = max(penalty_data, key=lambda x: x[2])  # x[2] is weighted
        dominant_name, _, dominant_weighted, _ = dominant
        dominant_pct = (dominant_weighted / total_weighted * 100) if total_weighted > 0 else 0

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
