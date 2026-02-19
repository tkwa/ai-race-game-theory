"""Matplotlib figures for each game-theory variant."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from report import primitives

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


def _make_dual_axes(figsize: tuple[float, float] = (10, 6)) -> tuple[Figure, plt.Axes, plt.Axes]:
    """Create a figure with dual y-axes (left blue, right red)."""
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()
    ax_left.set_ylabel("s* (safety investment)", color="tab:blue")
    ax_right.set_ylabel("Survival probability", color="tab:red")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")
    ax_right.tick_params(axis="y", labelcolor="tab:red")
    return fig, ax_left, ax_right


def plot_safety_elasticity(data: dict[str, Any]) -> Figure:
    """Effect of safety elasticity alpha on equilibrium s* and joint survival."""
    fig, ax_l, ax_r = _make_dual_axes()
    k = np.asarray(data["k"])
    x = _setup_k_axis(ax_l, k)
    _setup_k_axis(ax_r, k)  # sync right axis ticks

    ax_l.plot(x, data["s_star_a1"], "b-", label="s* (α=1)")
    ax_l.plot(
        x,
        data["s_star_a_default"],
        "b--",
        label=f"s* (α={primitives.DEFAULT_ALPHA})",
    )
    ax_r.plot(x, data["p_sq_a1"], "r-", label="P² (α=1)")
    ax_r.plot(
        x,
        data["p_sq_a_default"],
        "r--",
        label=f"P² (α={primitives.DEFAULT_ALPHA})",
    )

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    ax_l.set_title("Safety Elasticity: Effect of α on Equilibrium")
    fig.tight_layout()
    return fig


def plot_n_actors(data: dict[str, Any]) -> Figure:
    """Safety investment and joint survival as number of actors increases."""
    fig, ax_l, ax_r = _make_dual_axes()
    n = np.asarray(data["n"])

    ax_l.plot(n, data["s_star"], "b-o", label="s*")
    ax_r.plot(n, data["joint_survival"], "r-o", label="Joint survival")
    ax_l.set_xlabel("Number of actors (n)")

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    ax_l.set_title("Number of Actors: Safety Investment and Joint Survival")
    fig.tight_layout()
    return fig


def plot_resources_2(data: dict[str, Any]) -> Figure:
    """Asymmetric 2-actor equilibrium across resource ratios."""
    fig, ax_l, ax_r = _make_dual_axes()
    r = np.asarray(data["R"])

    ax_l.plot(r, data["s1"], "b-", label="s₁")
    ax_l.plot(r, data["s2"], "b--", label="s₂")
    ax_r.plot(r, data["p1"], "r--", alpha=0.7, label="p₁")
    ax_r.plot(r, data["p2"], "g--", alpha=0.7, label="p₂")
    ax_r.plot(r, data["joint_survival"], "r-", linewidth=2, label="Joint survival")
    ax_l.set_xlabel("Resource ratio R")

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    ax_l.set_title("Asymmetric Resources (2 Actors)")
    fig.tight_layout()
    return fig


def plot_resources_5(data: dict[str, Any]) -> Figure:
    """5-actor asymmetric resource equilibrium across k values."""
    fig, ax_l, ax_r = _make_dual_axes()
    k = np.asarray(data["k"])
    x = _setup_k_axis(ax_l, k)
    _setup_k_axis(ax_r, k)

    s_stars = np.asarray(data["s_stars"])  # shape (N, 5)
    blues = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
    for i in range(5):
        ax_l.plot(x, s_stars[:, i], color=blues[i], label=f"s*₍{i + 1}₎")

    ax_r.plot(x, data["joint_survival"], "r-", linewidth=2, label="Joint survival")

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, loc="best", fontsize=8)
    ax_l.set_title("Asymmetric Resources (5 Actors, resources ∝ 1/i)")
    fig.tight_layout()
    return fig


def plot_comparative_advantage(data: dict[str, Any]) -> Figure:
    """Comparative advantage where actor A has 2x safety effectiveness."""
    fig, ax_l, ax_r = _make_dual_axes()
    k = np.asarray(data["k"])
    x = _setup_k_axis(ax_l, k)
    _setup_k_axis(ax_r, k)

    ax_l.plot(x, data["s_A"], "b-", label="s_A")
    ax_l.plot(x, data["s_B"], "b--", label="s_B")
    ax_r.plot(x, data["p_A"], "r--", alpha=0.7, label="p_A")
    ax_r.plot(x, data["p_B"], "g--", alpha=0.7, label="p_B")
    ax_r.plot(x, data["joint_survival"], "r-", linewidth=2, label="Joint survival")

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    ax_l.set_title("Comparative Advantage (A has 2× safety effectiveness)")
    fig.tight_layout()
    return fig


def plot_public_good(data: dict[str, Any]) -> Figure:
    """Public good safety with varying delta for 5 actors."""
    fig, ax_l, ax_r = _make_dual_axes()
    delta = np.asarray(data["delta"])

    ax_l.plot(delta, data["s_star"], "b-", label="s*")
    ax_r.plot(delta, data["joint_survival"], "r-", label="Joint survival")
    ax_l.set_xlabel("δ (private vs public safety)")

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    ax_l.set_title("Public Good Safety (5 Actors)")
    fig.tight_layout()
    return fig


def plot_winner_take_all(data: dict[str, Any]) -> Figure:
    """Winner-take-all dynamics with varying competition intensity w."""
    fig, ax_l, ax_r = _make_dual_axes()
    w = np.asarray(data["w"])

    ax_l.plot(w, data["s_star"], "b-", label="s*")
    ax_r.plot(w, data["p_sq"], "r-", label="P²")
    ax_l.set_xlabel("w (winner-take-all exponent)")

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    ax_l.set_title("Winner-Take-All Dynamics")
    fig.tight_layout()
    return fig


def plot_correlated_alignment(data: dict[str, Any]) -> Figure:
    """Effect of correlated alignment on equilibrium."""
    fig, ax_l, ax_r = _make_dual_axes()
    rho = np.asarray(data["rho"])

    ax_l.plot(rho, data["s_star"], "b-", label="s*")
    ax_r.plot(rho, data["joint_survival"], "r-", label="Joint survival")
    ax_l.set_xlabel("ρ (alignment correlation)")

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    ax_l.set_title("Correlated Alignment")
    fig.tight_layout()
    return fig


def plot_heatmap_alpha_k(data: dict[str, Any]) -> Figure:
    """Heatmap of joint survival over safety effectiveness and elasticity."""
    fig, ax = plt.subplots(figsize=(8, 6))
    k = np.asarray(data["k"])
    alpha = np.asarray(data["alpha"])
    joint = np.asarray(data["joint_survival"])  # shape (len(k), len(alpha))

    cf = ax.contourf(k, alpha, joint.T, levels=20, cmap="RdYlGn")
    fig.colorbar(cf, ax=ax, label="Joint survival probability")
    ax.set_xlabel("k (safety effectiveness)")
    ax.set_ylabel("α (safety elasticity)")
    ax.set_title("Joint Survival: Safety Effectiveness × Elasticity")
    fig.tight_layout()
    return fig


def plot_heatmap_n_w(data: dict[str, Any]) -> Figure:
    """Heatmap of joint survival over competition intensity and actor count."""
    fig, ax = plt.subplots(figsize=(8, 6))
    n = np.asarray(data["n"])
    w = np.asarray(data["w"])
    joint = np.asarray(data["joint_survival"])  # shape (len(n), len(w))

    cf = ax.contourf(w, n, joint, levels=20, cmap="RdYlGn")
    fig.colorbar(cf, ax=ax, label="Joint survival probability")
    ax.set_xlabel("w (winner-take-all exponent)")
    ax.set_ylabel("n (number of actors)")
    ax.set_title("Joint Survival: Competition Intensity × Number of Actors")
    fig.tight_layout()
    return fig


def plot_public_good_vs_correlation(data: dict[str, Any]) -> Figure:
    """Contour plot of joint survival over public good delta and correlation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    delta = np.asarray(data["delta"])
    rho = np.asarray(data["rho"])
    joint = np.asarray(data["joint_survival"])  # shape (len(delta), len(rho))

    cf = ax.contourf(delta, rho, joint.T, levels=20, cmap="RdYlGn")
    fig.colorbar(cf, ax=ax, label="Joint survival probability")
    ax.set_xlabel("δ (private vs public safety)")
    ax.set_ylabel("ρ (alignment correlation)")
    ax.set_title("Joint Survival: Public Good × Correlation")
    fig.tight_layout()
    return fig


def plot_resource_inequality(data: dict[str, Any]) -> Figure:
    """Joint survival vs Gini coefficient for different actor counts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    gini = np.asarray(data["gini"])
    n_values = data["n_values"]
    joint_dict = data["joint_survival"]  # dict n -> array

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_values)))
    for i, n in enumerate(n_values):
        ax.plot(gini, joint_dict[n], color=colors[i], label=f"n={n}")

    ax.set_xlabel("Gini coefficient (resource inequality)")
    ax.set_ylabel("Joint survival probability")
    ax.legend(loc="best")
    ax.set_title("Resource Inequality vs Joint Survival")
    fig.tight_layout()
    return fig


def plot_comparative_advantage_public_good(data: dict[str, Any]) -> Figure:
    """Comparative advantage combined with public good safety."""
    fig, ax = plt.subplots(figsize=(10, 6))
    delta = np.asarray(data["delta"])

    ax.plot(delta, data["p_A"], "b--", label="p_A")
    ax.plot(delta, data["p_B"], "g--", label="p_B")
    ax.plot(delta, data["joint_survival"], "r-", linewidth=2, label="Joint survival")
    ax.set_xlabel("δ (private vs public safety)")
    ax.set_ylabel("Probability")
    ax.legend(loc="best")
    ax.set_title("Comparative Advantage + Public Good")
    fig.tight_layout()
    return fig


def plot_summary_table(rows: list[tuple[str, float]]) -> Figure:
    """Colored summary table of survival probabilities across conditions."""
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8, len(rows) * 0.45 + 0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows))
    ax.axis("off")

    cmap = plt.cm.RdYlGn
    for i, (label, val) in enumerate(zip(labels, values)):
        y = len(rows) - 1 - i
        color = cmap(val)
        ax.barh(y, val, height=0.7, color=color, edgecolor="white", linewidth=0.5)
        ax.text(-0.01, y, label, ha="right", va="center", fontsize=9, fontweight="bold")
        ax.text(val + 0.01, y, f"{val:.0%}", ha="left", va="center", fontsize=9)

    ax.set_title("Joint Survival Probability by Condition", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig
