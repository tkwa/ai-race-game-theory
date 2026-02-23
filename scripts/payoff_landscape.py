"""Plot each lab's payoff as a function of s_i, holding others at equilibrium."""

import sys

sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import numpy as np

from report import defaults, primitives


def main() -> None:
    # Get equilibrium
    s_star = primitives.full_model_nash(
        R=defaults.R,
        A=defaults.A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    print("Equilibrium s*:", [f"{s:.4f}" for s in s_star])

    s_grid = np.linspace(0.001, 0.999, 200)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for i, (ax, name) in enumerate(zip(axes, defaults.LAB_NAMES)):
        payoffs = []
        for s_i in s_grid:
            s_trial = list(s_star)
            s_trial[i] = s_i
            p = primitives.full_model_payoffs(
                s_trial,
                defaults.R,
                defaults.A,
                defaults.K,
                defaults.ALPHA,
                defaults.W,
                defaults.Z,
                defaults.DELTA,
                defaults.RHO,
            )
            payoffs.append(p[i])

        ax.plot(s_grid, payoffs, "b-", linewidth=1.5)
        ax.axvline(s_star[i], color="r", linestyle="--", alpha=0.7, label=f"s*={s_star[i]:.3f}")
        ax.set_xlabel(f"s_{name}")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Check single-peakedness
        peak_idx = np.argmax(payoffs)
        peak_s = s_grid[peak_idx]
        diff = abs(peak_s - s_star[i])
        print(f"{name}: peak s={peak_s:.3f}, s*={s_star[i]:.3f}, diff={diff:.4f}")

    axes[0].set_ylabel("Payoff")
    fig.suptitle("Payoff landscape at equilibrium (δ=0.2)", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/payoff_landscape.png", dpi=150)
    print("Saved plots/payoff_landscape.png")


if __name__ == "__main__":
    main()
