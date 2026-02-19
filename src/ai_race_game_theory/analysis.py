"""Analysis and plotting for the AI race game theory model."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from ai_race_game_theory import model


def compute_equilibria(k_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Nash equilibrium safety fractions and survival probabilities."""
    s_stars = np.array([model.find_nash_equilibrium(k) for k in k_values])
    p_squareds = np.array(
        [model.alignment_probability(s, k) ** 2 for s, k in zip(s_stars, k_values)]
    )
    return s_stars, p_squareds


def plot_equilibria(output_dir: str | pathlib.Path = "plots") -> pathlib.Path:
    """Generate dual-axis plot of equilibrium safety investment and survival probability."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = np.geomspace(0.05, 200, 200)
    s_stars, p_squareds = compute_equilibria(k_values)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Use log(1+k) for spacing but label as k
    x = np.log(1 + k_values)

    color_safety = "tab:blue"
    ax1.set_xlabel("k (safety effectiveness parameter)")
    ax1.set_ylabel("s* (safety investment fraction)", color=color_safety)
    ax1.plot(x, s_stars, color=color_safety, linewidth=2, label="s* (safety investment)")
    ax1.tick_params(axis="y", labelcolor=color_safety)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    color_survival = "tab:red"
    ax2.set_ylabel("P² (survival probability)", color=color_survival)
    ax2.plot(x, p_squareds, color=color_survival, linewidth=2, label="P² (survival probability)")
    ax2.tick_params(axis="y", labelcolor=color_survival)
    ax2.set_ylim(0, 1)

    # Set x-axis ticks at specific k values
    tick_k_values = [0.1, 0.5, 1, 2, 5, 10, 50, 100]
    tick_positions = [np.log(1 + k) for k in tick_k_values]
    tick_labels = [str(k) for k in tick_k_values]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    ax1.set_title("Nash Equilibrium: Safety Investment vs Survival Probability")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
    fig.tight_layout()

    output_path = output_dir / "equilibrium_analysis.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    path = plot_equilibria()
    print(f"Plot saved to {path}")
