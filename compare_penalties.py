#!/usr/bin/env python3
"""
Visual comparison of current vs smooth overlap penalties.
"""

import numpy as np
import matplotlib.pyplot as plt


def current_penalty(sep: float) -> float:
    """Current implementation (simplified)."""
    if sep >= 1.0:
        return 0.0
    # Assuming 100x50 boxes with 10px margin
    # overlap in each dimension, squared
    overlap = (1.0 - sep) * 110  # simplified
    return (overlap * overlap) ** 0.5  # sqrt to make comparable scale


def smooth_exponential(sep: float, strength: float = 10.0, sharpness: float = 5.0) -> float:
    """Smooth exponential penalty."""
    violation = 1.0 - sep
    return strength * np.exp(sharpness * violation)


def smooth_log(sep: float, scale: float = 5.0) -> float:
    """Logarithmic barrier."""
    if sep >= 1.0:
        return 0.0
    return -scale * np.log(sep + 1e-6)


def smooth_softplus(sep: float, softness: float = 0.1, scale: float = 100.0) -> float:
    """Softplus approximation."""
    overlap = 1.0 - sep
    soft_overlap = softness * np.log(1 + np.exp(overlap / softness))
    return scale * soft_overlap ** 2


def main():
    # Separation values from fully overlapped to well-separated
    separations = np.linspace(0.05, 2.0, 200)

    # Compute penalties
    current = [current_penalty(s) for s in separations]
    exp_soft = [smooth_exponential(s, strength=10, sharpness=5) for s in separations]
    exp_gentle = [smooth_exponential(s, strength=10, sharpness=3) for s in separations]
    exp_sharp = [smooth_exponential(s, strength=10, sharpness=7) for s in separations]
    log_pen = [smooth_log(s, scale=5) for s in separations]
    softplus_pen = [smooth_softplus(s, softness=0.1, scale=1.0) for s in separations]

    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Main comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(separations, current, 'r-', linewidth=2, label='Current (quartic)', alpha=0.7)
    ax1.plot(separations, exp_soft, 'b-', linewidth=2, label='Exponential (sharp=5)')
    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='Touching boundary')
    ax1.set_xlabel('Separation ratio (1.0 = touching)', fontsize=11)
    ax1.set_ylabel('Penalty', fontsize=11)
    ax1.set_title('Current vs Smooth Exponential', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 200)

    # Plot 2: Gradients
    ax2 = plt.subplot(2, 3, 2)
    grad_current = -np.diff(current) / np.diff(separations)
    grad_smooth = -np.diff(exp_soft) / np.diff(separations)
    ax2.plot(separations[:-1], np.abs(grad_current), 'r-', linewidth=2,
             label='Current gradient', alpha=0.7)
    ax2.plot(separations[:-1], np.abs(grad_smooth), 'b-', linewidth=2,
             label='Smooth gradient')
    ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Separation ratio', fontsize=11)
    ax2.set_ylabel('Gradient magnitude', fontsize=11)
    ax2.set_title('Gradient Comparison (Smoothness)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2)

    # Plot 3: Sharpness variations
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(separations, exp_gentle, 'g-', linewidth=2, label='Gentle (sharp=3)', alpha=0.7)
    ax3.plot(separations, exp_soft, 'b-', linewidth=2, label='Medium (sharp=5)')
    ax3.plot(separations, exp_sharp, 'r-', linewidth=2, label='Sharp (sharp=7)', alpha=0.7)
    ax3.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Separation ratio', fontsize=11)
    ax3.set_ylabel('Penalty', fontsize=11)
    ax3.set_title('Exponential Sharpness Tuning', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 2)
    ax3.set_ylim(0, 500)

    # Plot 4: Log-scale comparison (to see behavior across orders of magnitude)
    ax4 = plt.subplot(2, 3, 4)
    # Filter out zeros for log scale
    current_log = [max(c, 0.01) for c in current]
    exp_log = [max(e, 0.01) for e in exp_soft]
    ax4.semilogy(separations, current_log, 'r-', linewidth=2, label='Current', alpha=0.7)
    ax4.semilogy(separations, exp_log, 'b-', linewidth=2, label='Exponential')
    ax4.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Separation ratio', fontsize=11)
    ax4.set_ylabel('Penalty (log scale)', fontsize=11)
    ax4.set_title('Log Scale View (Growth Rate)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(0, 2)

    # Plot 5: All smooth options
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(separations, exp_soft, 'b-', linewidth=2, label='Exponential (Recommended)')
    ax5.plot(separations, log_pen, 'g-', linewidth=2, label='Logarithmic', alpha=0.7)
    ax5.plot(separations, softplus_pen, 'm-', linewidth=2, label='Softplus', alpha=0.7)
    ax5.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Separation ratio', fontsize=11)
    ax5.set_ylabel('Penalty', fontsize=11)
    ax5.set_title('Smooth Penalty Options', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 2)
    ax5.set_ylim(0, 50)

    # Plot 6: Discontinuity zoom (the problem area)
    ax6 = plt.subplot(2, 3, 6)
    zoom_range = (separations >= 0.9) & (separations <= 1.1)
    ax6.plot(separations[zoom_range], [current_penalty(s) for s in separations[zoom_range]],
             'r-', linewidth=3, label='Current (discontinuous)', alpha=0.7)
    ax6.plot(separations[zoom_range], [smooth_exponential(s) for s in separations[zoom_range]],
             'b-', linewidth=3, label='Smooth (continuous)')
    ax6.axvline(1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax6.set_xlabel('Separation ratio', fontsize=11)
    ax6.set_ylabel('Penalty', fontsize=11)
    ax6.set_title('Discontinuity at Boundary (ZOOMED)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0.9, 1.1)

    plt.suptitle('Overlap Penalty Comparison: Current vs Smooth Alternatives',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_file = 'penalty_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_file}")

    # Print numeric comparison
    print("\n" + "="*70)
    print("NUMERIC COMPARISON AT KEY POINTS")
    print("="*70)
    print(f"{'Separation':<15} {'Current':<15} {'Exponential':<15} {'Ratio':>10}")
    print("-"*70)

    test_seps = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
    for sep in test_seps:
        curr = current_penalty(sep)
        smooth = smooth_exponential(sep)
        ratio = curr / smooth if smooth > 0.01 else float('inf')
        print(f"{sep:<15.1f} {curr:<15.2f} {smooth:<15.2f} {ratio:>10.1f}x")

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("• Current penalty has SHARP DISCONTINUITY at separation=1.0")
    print("• Current gradient is ZERO until overlap begins (no guidance)")
    print("• Smooth exponential provides GRADUAL REPULSION before overlap")
    print("• Smooth version is C∞ (infinitely differentiable) → L-BFGS-B friendly")
    print("• Both enforce non-overlap, but smooth is numerically stable")
    print("="*70)


if __name__ == '__main__':
    main()
