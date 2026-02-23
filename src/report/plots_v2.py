"""Plotting functions for the v2 report."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from report import defaults

K_TICK_VALUES = np.array([0.1, 0.5, 1, 2, 5, 10, 50, 100])


def _setup_k_axis(ax: plt.Axes, k_values: np.ndarray) -> np.ndarray:
    """Apply log(1+k) transform to x-axis and set standard tick labels."""
    x = np.log(1 + k_values)
    tick_x = np.log(1 + K_TICK_VALUES)
    ax.set_xticks(tick_x)
    ax.set_xticklabels([str(v) for v in K_TICK_VALUES])
    ax.set_xlabel("k (safety effectiveness)")
    ax.set_xlim(x[0], x[-1])
    return x


def plot_toy_baseline(data: dict) -> Figure:
    """Toy model: s* and joint survival vs k for n=2, alpha=0.466."""
    fig, ax_l = plt.subplots(figsize=(10, 6))
    ax_r = ax_l.twinx()
    k = np.asarray(data["k"])
    x = _setup_k_axis(ax_l, k)
    _setup_k_axis(ax_r, k)

    ax_l.plot(x, data["s_star"], "b-", linewidth=2, label="s* (safety fraction)")
    ax_r.plot(x, data["joint_survival"], "r-", linewidth=2, label="Joint survival P\u00b2")

    ax_l.set_ylabel("s* (safety investment)", color="tab:blue")
    ax_r.set_ylabel("Joint survival probability", color="tab:red")
    ax_l.tick_params(axis="y", labelcolor="tab:blue")
    ax_r.tick_params(axis="y", labelcolor="tab:red")
    ax_r.set_ylim(0, 1)
    ax_l.set_title("Toy Model: 2-Player Symmetric Equilibrium (\u03b1=0.466)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    fig.tight_layout()
    fig.savefig("plots/toy_baseline.png", dpi=150, bbox_inches="tight")
    return fig


def plot_toy_n_actors(data: dict) -> Figure:
    """Toy model: joint survival vs number of actors."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(data["n"], data["joint_survival"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Number of actors (n)")
    ax.set_ylabel("Joint survival probability")
    ax.set_title("Joint Survival Drops with More Actors")
    ax.set_ylim(0, 1)
    ax.set_xticks(data["n"])
    for i, (n, js) in enumerate(zip(data["n"], data["joint_survival"])):
        ax.text(n, js + 0.02, f"{js:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("plots/toy_n_actors.png", dpi=150, bbox_inches="tight")
    return fig


def plot_toy_asymmetric_resources(data: dict) -> Figure:
    """Toy model: safety spending and joint survival vs resource ratio."""
    r = np.asarray(data["R"])
    s1 = np.asarray(data["s1"])
    s2 = np.asarray(data["s2"])
    js = np.asarray(data["joint_survival"])

    fig, (ax_l, ax_r_top) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.35})

    # Left: safety fraction (relative)
    ax_l.plot(r, s1, "b-", linewidth=2, label="Player 1 (larger)")
    ax_l.plot(r, s2, "r-", linewidth=2, label="Player 2 (smaller)")
    ax_l.set_xlabel("Player 1 resource share")
    ax_l.set_ylabel("Safety fraction s* (share of own resources)")
    ax_l.set_title("Safety Investment (Fraction) vs Resource Asymmetry")
    ax_l.legend()
    ax_l.set_xlim(0.5, 0.95)

    # Right: broken y-axis for joint survival
    js_min, js_max = js.min(), js.max()
    js_pad = (js_max - js_min) * 0.15
    break_lo = 0.05  # top of bottom segment
    break_hi = js_min - js_pad  # bottom of top segment

    # Turn ax_r_top into two axes via inset
    ax_r_top.set_position(ax_r_top.get_position())
    pos = ax_r_top.get_position()
    ax_r_top.remove()

    # Bottom segment: 0 to break_lo (just shows the zero baseline)
    height_ratio_bot = 0.15
    height_ratio_top = 1.0 - height_ratio_bot
    ax_bot = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height * height_ratio_bot])
    ax_top = fig.add_axes(
        [
            pos.x0,
            pos.y0 + pos.height * (height_ratio_bot + 0.03),
            pos.width,
            pos.height * height_ratio_top - 0.03 * pos.height,
        ]
    )

    for ax in [ax_top, ax_bot]:
        ax.plot(r, js, "k-", linewidth=2)
        ax.set_xlim(0.5, 0.95)

    ax_top.set_ylim(break_hi, js_max + js_pad)
    ax_bot.set_ylim(0, break_lo)

    # Hide spines between the two
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)
    ax_bot.set_xlabel("Player 1 resource share")

    # Break marks
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs["transform"] = ax_bot.transAxes
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_top.set_title("Joint Survival vs Resource Asymmetry")
    ax_top.set_ylabel("Joint survival probability")

    fig.savefig("plots/toy_asymmetric_resources.png", dpi=150, bbox_inches="tight")
    return fig


def plot_full_model_vs_k(data: dict) -> Figure:
    """Full model: expected human share and lab safety fractions vs k."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    k_arr = np.array(data["k"])
    x = _setup_k_axis(ax_bot, k_arr)
    _setup_k_axis(ax_top, k_arr)

    # Top: expected human share
    ax_top.plot(x, data["expected_human_share"], "k-", linewidth=2.5)
    ax_top.set_ylabel("Expected human share of universe")
    ax_top.set_ylim(0, 1)
    ax_top.set_title("Full Model: 5-Lab Equilibrium Outcomes")
    ax_top.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # Bottom: each lab's safety fraction
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    s_star_matrix = np.array(data["s_star"])  # (n_k, 5)
    for j, (name, color) in enumerate(zip(defaults.get_lab_names(), colors)):
        ax_bot.plot(x, s_star_matrix[:, j], color=color, linewidth=1.5, label=name)
    ax_bot.set_ylabel("Safety fraction s*")
    ax_bot.legend(loc="upper right")
    ax_bot.set_title("Equilibrium Safety Investment by Lab")

    fig.tight_layout()
    fig.savefig("plots/full_model_default.png", dpi=150, bbox_inches="tight")
    return fig


def plot_lab_summary(data: dict) -> Figure:
    """Clustered bar graph: per-lab Omega share (black) and P(misalignment) (red)."""
    names = data["lab_names"]
    omega = data["omega_share"]
    p_mis = data["p_misaligned"]
    n = len(names)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, omega, width, color="black", label="Power share (Ω | aligned)")
    bars2 = ax.bar(x + width / 2, p_mis, width, color="#d62728", label="P(misalignment)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Probability / Share")
    ax.set_title("Baseline Equilibrium: Lab Power Share and Misalignment Risk")
    ax.legend()
    ax.set_ylim(0, max(max(omega), max(p_mis)) * 1.2)

    # Annotate bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{bar.get_height():.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{bar.get_height():.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#d62728",
        )

    fig.tight_layout()
    fig.savefig("plots/lab_summary.png", dpi=150, bbox_inches="tight")
    return fig


def plot_interventions(results: list[tuple[str, float, float]]) -> Figure:
    """Horizontal bar graph of intervention effects relative to baseline."""
    names = [r[0] for r in results]
    baseline = results[0][1]
    intervention_values = [r[2] for r in results]
    deltas = [iv - baseline for iv in intervention_values]

    # Sort by delta
    order = np.argsort(deltas)
    names = [names[i] for i in order]
    deltas = [deltas[i] for i in order]
    intervention_values = [intervention_values[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if d < 0 else "#2ca02c" for d in deltas]
    ax.barh(range(len(names)), deltas, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(x=0, color="black", linewidth=1.5)
    ax.set_xlabel("Change in expected human share")
    ax.set_title(f"Intervention Effects (baseline = {baseline:.1%})")

    # Annotate bars with absolute values
    for i, (d, iv) in enumerate(zip(deltas, intervention_values)):
        x_pos = d + (0.003 if d >= 0 else -0.003)
        ha = "left" if d >= 0 else "right"
        ax.text(x_pos, i, f"{iv:.1%}", va="center", ha=ha, fontsize=9)

    fig.tight_layout()
    fig.savefig("plots/interventions.png", dpi=150, bbox_inches="tight")
    return fig


def plot_sensitivity(data: dict) -> Figure:
    """Grid of conditional median expected human share vs each parameter."""
    n_vars = len(data["variables"])
    ncols = 3
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))

    for ax, (centers, medians, xlabel, title) in zip(axes.flat, data["variables"]):
        ax.plot(centers, medians, "o-", linewidth=2, markersize=8, color="steelblue")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Median expected human share", fontsize=10)
        ax.set_title(f"Conditional on {title}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
        if medians:
            y_range = max(medians) - min(medians)
            ax.set_ylim(
                min(medians) - max(0.02, y_range * 0.1), max(medians) + max(0.02, y_range * 0.1)
            )

    # Hide unused axes
    for ax in axes.flat[n_vars:]:
        ax.set_visible(False)

    fig.suptitle("Sensitivity Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig("plots/sensitivity.png", dpi=150, bbox_inches="tight")
    return fig
