import matplotlib.pyplot as plt
import pandas as pd


def plot_energy_history(
    history: list[dict[str, float]],
    output_path: str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Plot per-penalty energy curves over iterations.

    Args:
        history: List of dictionaries with per-penalty energy values
        output_path: Path to save the plot (if None, displays interactively)
        figsize: Figure size in inches
    """
    if not history:
        print("No history data to plot")
        return

    df = pd.DataFrame(history)

    # Get penalty columns (exclude metadata columns)
    penalty_cols = [c for c in df.columns if c not in ('Total', 'accepted', 'energy')]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot 1: Total energy over iterations
    ax1 = axes[0]
    if 'energy' in df.columns:
        ax1.plot(df.index, df['energy'], 'b-', linewidth=1.5, label='Total Energy')
    elif 'Total' in df.columns:
        ax1.plot(df.index, df['Total'], 'b-', linewidth=1.5, label='Total Energy')

    ax1.set_ylabel('Total Energy')
    ax1.set_title('Optimization Progress')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Mark accepted iterations if available
    if 'accepted' in df.columns:
        accepted_mask = df['accepted'] == True
        if accepted_mask.any():
            y_col = 'energy' if 'energy' in df.columns else 'Total'
            if y_col in df.columns:
                ax1.scatter(
                    df.index[accepted_mask],
                    df[y_col][accepted_mask],
                    c='green', s=10, alpha=0.5, label='Accepted'
                )

    # Plot 2: Per-penalty breakdown (stacked area)
    ax2 = axes[1]

    if penalty_cols:
        # Normalize to show relative contributions
        penalty_df = df[penalty_cols]
        ax2.stackplot(
            df.index,
            [penalty_df[col] for col in penalty_cols],
            labels=penalty_cols,
            alpha=0.7
        )
        ax2.set_ylabel('Energy by Penalty')
        ax2.set_xlabel('Iteration')
        ax2.legend(loc='upper right', fontsize='small')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Energy plot saved to {output_path}")
    else:
        plt.show()

    plt.close()
