"""Compare copula choices for modeling correlated AI alignment outcomes.

Computes joint survival P(all labs aligned) under Gaussian, Student-t, Clayton,
Gumbel, and Frank copulas with equicorrelation structure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from scipy.optimize import brentq

PLOT_DIR = Path(__file__).parent / "copula_plots"
PLOT_DIR.mkdir(exist_ok=True)

MC_SAMPLES = 500_000


# ---------------------------------------------------------------------------
# Copula parameter conversion: equicorrelation r -> copula parameter
# Bridge via Gaussian copula's Kendall tau: tau = (2/pi) * arcsin(r)
# ---------------------------------------------------------------------------


def _kendall_tau_from_r(r: float) -> float:
    """Kendall's tau implied by Gaussian equicorrelation r."""
    return (2.0 / np.pi) * np.arcsin(np.clip(r, -1, 1))


def _clayton_theta(r: float) -> float:
    """Clayton parameter theta from equicorrelation r via Kendall's tau."""
    tau = _kendall_tau_from_r(r)
    if tau <= 0:
        return 1e-8
    return 2.0 * tau / (1.0 - tau)


def _gumbel_theta(r: float) -> float:
    """Gumbel parameter theta from equicorrelation r via Kendall's tau."""
    tau = _kendall_tau_from_r(r)
    if tau <= 0:
        return 1.0 + 1e-8
    return 1.0 / (1.0 - tau)


def _frank_tau_from_theta(theta: float) -> float:
    """Kendall's tau for Frank copula with parameter theta."""
    if abs(theta) < 1e-10:
        return 0.0
    debye_integral, _ = quad(lambda t: t / (np.exp(t) - 1.0), 1e-15, abs(theta))
    return 1.0 - 4.0 / theta + 4.0 / theta**2 * debye_integral


def _frank_theta(r: float) -> float:
    """Frank parameter theta from equicorrelation r via Kendall's tau (numerical)."""
    tau = _kendall_tau_from_r(r)
    if abs(tau) < 1e-10:
        return 1e-8
    try:
        return brentq(lambda th: _frank_tau_from_theta(th) - tau, 0.01, 100.0)
    except ValueError:
        return brentq(lambda th: _frank_tau_from_theta(th) - tau, 0.001, 500.0)


# ---------------------------------------------------------------------------
# Copula uniform sample generators
# Each returns an (MC_SAMPLES, n) array of uniform [0,1] samples with the
# specified equicorrelation structure. Using a fixed seed per call for
# reproducibility.
# ---------------------------------------------------------------------------


def _sample_gaussian_copula(n: int, r: float, seed: int = 0) -> np.ndarray:
    """Sample uniform marginals from Gaussian copula."""
    rng = np.random.default_rng(seed)
    cov = np.full((n, n), r)
    np.fill_diagonal(cov, 1.0)
    z = rng.multivariate_normal(np.zeros(n), cov, size=MC_SAMPLES)
    return stats.norm.cdf(z)


def _sample_t_copula(n: int, r: float, df: int = 4, seed: int = 0) -> np.ndarray:
    """Sample uniform marginals from Student-t copula."""
    rng = np.random.default_rng(seed)
    if r <= 0:
        return rng.uniform(size=(MC_SAMPLES, n))
    cov = np.full((n, n), r)
    np.fill_diagonal(cov, 1.0)
    z = rng.multivariate_normal(np.zeros(n), cov, size=MC_SAMPLES)
    chi2 = rng.chisquare(df, size=MC_SAMPLES)
    t = z / np.sqrt(chi2[:, None] / df)
    return stats.t.cdf(t, df=df)


def _sample_clayton_copula(n: int, r: float, seed: int = 0) -> np.ndarray:
    """Sample uniform marginals from Clayton copula via gamma frailty."""
    rng = np.random.default_rng(seed)
    theta = _clayton_theta(r)
    if theta < 1e-6:
        return rng.uniform(size=(MC_SAMPLES, n))
    v = rng.gamma(1.0 / theta, 1.0, size=MC_SAMPLES)
    e = rng.exponential(1.0, size=(MC_SAMPLES, n))
    u = (1.0 + e / v[:, None]) ** (-1.0 / theta)
    return np.clip(u, 0.0, 1.0)


def _sample_gumbel_copula(n: int, r: float, seed: int = 0) -> np.ndarray:
    """Sample uniform marginals from Gumbel copula via stable subordinator."""
    rng = np.random.default_rng(seed)
    theta = _gumbel_theta(r)
    if theta <= 1.0 + 1e-6:
        return rng.uniform(size=(MC_SAMPLES, n))
    alpha = 1.0 / theta
    # Stable(alpha) via Chambers-Mallows-Stuck
    phi = rng.uniform(-np.pi / 2, np.pi / 2, size=MC_SAMPLES)
    w = rng.exponential(1.0, size=MC_SAMPLES)
    # Avoid division by zero at phi near +-pi/2
    cos_phi = np.clip(np.cos(phi), 1e-10, None)
    s = (
        np.sin(alpha * phi + np.pi * alpha / 2)
        / cos_phi ** (1.0 / alpha)
        * (np.clip(np.cos(phi - alpha * phi - np.pi * alpha / 2), 1e-10, None) / w)
        ** ((1.0 - alpha) / alpha)
    )
    s = np.clip(s, 1e-10, None)
    e = rng.exponential(1.0, size=(MC_SAMPLES, n))
    u = np.exp(-((e / s[:, None]) ** alpha))
    return np.clip(u, 0.0, 1.0)


def _sample_frank_copula(n: int, r: float, seed: int = 0) -> np.ndarray:
    """Sample uniform marginals from Frank copula via logarithmic frailty."""
    rng = np.random.default_rng(seed)
    theta = _frank_theta(r)
    if abs(theta) < 1e-6:
        return rng.uniform(size=(MC_SAMPLES, n))

    p_log = 1.0 - np.exp(-theta)
    # numpy's logseries handles all valid p in (0, 1)
    v = rng.logseries(p_log, size=MC_SAMPLES).astype(np.float64)

    e = rng.exponential(1.0, size=(MC_SAMPLES, n))
    # U_i = phi^{-1}(E_i / V) = -1/theta * log(1 + exp(-E_i/V) * (exp(-theta) - 1))
    exp_neg_theta_m1 = np.expm1(-theta)  # exp(-theta) - 1, negative
    log_arg = np.exp(-e / v[:, None]) * exp_neg_theta_m1
    result = -1.0 / theta * np.log1p(log_arg)
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Joint survival from pre-sampled copula uniforms
# ---------------------------------------------------------------------------


def _joint_survival_from_samples(u: np.ndarray, probs: list[float]) -> float:
    """P(all U_i <= p_i) from copula samples."""
    thresholds = np.array(probs)
    return float(np.all(u <= thresholds, axis=1).mean())


# ---------------------------------------------------------------------------
# Analytic Gaussian (no MC needed)
# ---------------------------------------------------------------------------


def joint_survival_gaussian(probs: list[float], r: float) -> float:
    """Joint survival via Gaussian copula (analytic)."""
    n = len(probs)
    if r <= 0 or n == 1:
        result = 1.0
        for p in probs:
            result *= p
        return result
    z = np.array([stats.norm.ppf(np.clip(p, 1e-10, 1.0 - 1e-10)) for p in probs])
    cov = np.full((n, n), r)
    np.fill_diagonal(cov, 1.0)
    return float(stats.multivariate_normal.cdf(z, mean=np.zeros(n), cov=cov))


# ---------------------------------------------------------------------------
# Wrapper: each copula returns joint survival for given probs and r
# For MC copulas, we generate fresh samples with a deterministic seed derived
# from (r, probs) so results are reproducible but independent across calls.
# ---------------------------------------------------------------------------

COPULA_SAMPLERS = {
    "Student-t (df=4)": _sample_t_copula,
    "Clayton": _sample_clayton_copula,
    "Gumbel": _sample_gumbel_copula,
    "Frank": _sample_frank_copula,
}


def _hash_seed(probs: list[float], r: float) -> int:
    """Deterministic seed from parameters."""
    return hash((tuple(round(p, 8) for p in probs), round(r, 8))) % (2**31)


def compute_all(probs: list[float], r: float) -> dict[str, float]:
    """Compute joint survival for all copulas."""
    results = {"Gaussian": joint_survival_gaussian(probs, r)}
    n = len(probs)
    seed = _hash_seed(probs, r)
    for name, sampler in COPULA_SAMPLERS.items():
        u = sampler(n, r, seed=seed)
        results[name] = _joint_survival_from_samples(u, probs)
    return results


def _compute_derivative(
    name: str,
    n: int,
    p1: float,
    other_p: float,
    r: float,
    dp: float = 0.01,
) -> float:
    """Compute d(joint_survival)/dp_1 using common random numbers for variance reduction."""
    probs_lo = [p1 - dp] + [other_p] * (n - 1)
    probs_hi = [p1 + dp] + [other_p] * (n - 1)

    if name == "Gaussian":
        val_lo = joint_survival_gaussian(probs_lo, r)
        val_hi = joint_survival_gaussian(probs_hi, r)
    else:
        # Use SAME random draws for both evaluations (common random numbers)
        seed = 12345
        sampler = COPULA_SAMPLERS[name]
        u = sampler(n, r, seed=seed)
        val_lo = _joint_survival_from_samples(u, probs_lo)
        val_hi = _joint_survival_from_samples(u, probs_hi)

    return (val_hi - val_lo) / (2 * dp)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

ALL_NAMES = ["Gaussian", "Student-t (df=4)", "Clayton", "Gumbel", "Frank"]


def experiment_a_baseline() -> None:
    """Baseline comparison: n=5, r=0.5, uniform p=0.9."""
    print("=" * 70)
    print("EXPERIMENT A: Baseline comparison")
    print("  n=5 labs, equicorrelation r=0.5, all p_i=0.9")
    print("=" * 70)

    probs = [0.9] * 5
    r = 0.5
    results = compute_all(probs, r)
    independent = 0.9**5

    print(f"\n  Independent (r=0):   {independent:.6f}")
    for name in ALL_NAMES:
        print(f"  {name:20s}: {results[name]:.6f}")
    print(f"\n  (Independent baseline: {independent:.6f})")
    print()


def experiment_b_sensitivity_r() -> None:
    """Sensitivity to correlation r: sweep r from 0 to 0.95."""
    print("=" * 70)
    print("EXPERIMENT B: Sensitivity to correlation r")
    print("  n=5, p=0.9, r in [0, 0.95]")
    print("=" * 70)

    r_values = np.linspace(0, 0.95, 20)
    probs = [0.9] * 5
    all_results: dict[str, list[float]] = {name: [] for name in ALL_NAMES}

    for r in r_values:
        results = compute_all(probs, float(r))
        for name in ALL_NAMES:
            all_results[name].append(results[name])

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ALL_NAMES:
        ax.plot(r_values, all_results[name], label=name, linewidth=2)
    ax.axhline(y=0.9**5, color="gray", linestyle="--", alpha=0.5, label="Independent")
    ax.set_xlabel("Equicorrelation r")
    ax.set_ylabel("P(all 5 labs aligned)")
    ax.set_title("Joint Survival vs Correlation Strength (p=0.9, n=5)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "sensitivity_r.png", dpi=150)
    plt.close(fig)
    print("  Plot saved to docs/copula_plots/sensitivity_r.png")

    for r_idx in [0, 5, 10, 15, 19]:
        r_val = r_values[r_idx]
        print(f"\n  r = {r_val:.2f}:")
        for name in ALL_NAMES:
            print(f"    {name:20s}: {all_results[name][r_idx]:.6f}")
    print()


def experiment_c_sensitivity_p() -> None:
    """Sensitivity to marginal probability p: sweep p from 0.5 to 0.99."""
    print("=" * 70)
    print("EXPERIMENT C: Sensitivity to marginal probability p")
    print("  n=5, r=0.5, p in [0.5, 0.99]")
    print("=" * 70)

    p_values = np.linspace(0.5, 0.99, 20)
    r = 0.5
    all_results: dict[str, list[float]] = {name: [] for name in ALL_NAMES}

    for p in p_values:
        probs = [float(p)] * 5
        results = compute_all(probs, r)
        for name in ALL_NAMES:
            all_results[name].append(results[name])

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ALL_NAMES:
        ax.plot(p_values, all_results[name], label=name, linewidth=2)
    # Independent baseline
    ax.plot(p_values, p_values**5, color="gray", linestyle="--", alpha=0.5, label="Independent")
    ax.set_xlabel("Marginal alignment probability p")
    ax.set_ylabel("P(all 5 labs aligned)")
    ax.set_title("Joint Survival vs Marginal Probability (r=0.5, n=5)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "sensitivity_p.png", dpi=150)
    plt.close(fig)
    print("  Plot saved to docs/copula_plots/sensitivity_p.png")

    for p_idx in [0, 5, 10, 15, 19]:
        p_val = p_values[p_idx]
        print(f"\n  p = {p_val:.3f}:")
        for name in ALL_NAMES:
            print(f"    {name:20s}: {all_results[name][p_idx]:.6f}")
    print()


def experiment_d_asymmetric() -> None:
    """Asymmetric case: heterogeneous marginal probabilities."""
    print("=" * 70)
    print("EXPERIMENT D: Asymmetric marginal probabilities")
    print("  n=5, r=0.5, probs=[0.99, 0.95, 0.9, 0.8, 0.7]")
    print("=" * 70)

    probs = [0.99, 0.95, 0.9, 0.8, 0.7]
    r = 0.5
    results = compute_all(probs, r)
    independent = float(np.prod(probs))

    print(f"\n  Independent (r=0):   {independent:.6f}")
    for name in ALL_NAMES:
        print(f"  {name:20s}: {results[name]:.6f}")
    print()


def experiment_e_marginal_value() -> None:
    """How copula choice affects the marginal value of one lab's safety investment."""
    print("=" * 70)
    print("EXPERIMENT E: Marginal value of safety investment")
    print("  d(joint_survival)/dp_1 at p_1=0.9, others at p=0.8")
    print("  Using common random numbers for MC copulas (dp=0.01)")
    print("=" * 70)

    p1_center = 0.9
    other_p = 0.8
    n = 5
    r_values = [0.2, 0.5, 0.8]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, r in zip(axes, r_values):
        print(f"\n  r = {r}:")
        derivs = {}
        for name in ALL_NAMES:
            deriv = _compute_derivative(name, n, p1_center, other_p, r)
            derivs[name] = deriv
            print(f"    {name:20s}: dP/dp_1 = {deriv:.6f}")

        names = list(derivs.keys())
        vals = list(derivs.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        ax.barh(names, vals, color=colors)
        ax.set_xlabel("dP(all aligned) / dp_1")
        ax.set_title(f"r = {r}")
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        "Marginal Value of Lab 1 Increasing Alignment\n(p_1=0.9, others=0.8)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "marginal_value.png", dpi=150)
    plt.close(fig)
    print("\n  Plot saved to docs/copula_plots/marginal_value.png")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nCOPULA COMPARISON FOR AI ALIGNMENT JOINT SURVIVAL")
    print("=" * 70)
    print()

    experiment_a_baseline()
    experiment_b_sensitivity_r()
    experiment_c_sensitivity_p()
    experiment_d_asymmetric()
    experiment_e_marginal_value()

    print("=" * 70)
    print("All experiments complete. Plots saved to docs/copula_plots/")
    print("=" * 70)
