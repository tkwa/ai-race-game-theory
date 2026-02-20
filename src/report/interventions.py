"""Compute Nash equilibrium and expected human share for 7 interventions."""

import numpy as np

from report import defaults, primitives


def expected_human_share_with_overrides(
    s_all: list[float],
    R: np.ndarray,
    k: float,
    alpha: float,
    w: float,
    z: float,
    delta: float | None = None,
    rho: float = 0.0,
    override_probs: dict[int, float] | None = None,
) -> float:
    """Like expected_human_share but with forced alignment probs for specific labs."""
    n = len(s_all)
    eff_s = primitives._effective_safety(s_all, delta)
    probs = np.array([primitives.alignment_prob(eff_s[j], k, alpha) for j in range(n)])

    if override_probs:
        for idx, p in override_probs.items():
            probs[idx] = p

    abs_cap = np.array([R[j] * max(1.0 - s_all[j], 1e-15) for j in range(n)])
    outcome_probs = primitives._outcome_probabilities(probs, rho)
    omega_table, aligned_masks = primitives._precompute_omega_table(abs_cap, w, z)
    aligned_shares = (omega_table * aligned_masks).sum(axis=1)
    return float(outcome_probs @ aligned_shares)


def _baseline_nash() -> tuple[list[float], float]:
    """Compute baseline Nash and expected human share with default params."""
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
    ehs = primitives.expected_human_share(
        s_all=s_star,
        R=defaults.R,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    return s_star, ehs


def _intervention_remove_china() -> float:
    """Remove China (index 4), renormalize R, run 4-lab Nash."""
    R_new = defaults.R[:4].copy()
    R_new /= R_new.sum()
    A_new = defaults.A[:4, :4].copy()
    s_star = primitives.full_model_nash(
        R=R_new,
        A=A_new,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    return primitives.expected_human_share(
        s_all=s_star,
        R=R_new,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )


def _intervention_duplicate_labs() -> float:
    """Duplicate every lab: 10 labs, split resources, cross-bloc amity 0.3."""
    R_new = np.tile(defaults.R, 2) / 2.0
    n = len(defaults.R)
    A_new = np.ones((2 * n, 2 * n)) * 0.3
    A_new[:n, :n] = defaults.A
    A_new[n:, n:] = defaults.A
    np.fill_diagonal(A_new, 1.0)
    s_star = primitives.full_model_nash(
        R=R_new,
        A=A_new,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    return primitives.expected_human_share(
        s_all=s_star,
        R=R_new,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )


def _intervention_secretly_safe_gdm(baseline_nash: list[float]) -> float:
    """Normal Nash, then force GDM alignment prob to 1.0 for outcome computation."""
    return expected_human_share_with_overrides(
        s_all=baseline_nash,
        R=defaults.R,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
        override_probs={2: 1.0},
    )


def _intervention_tell_only_gdm(baseline_nash: list[float]) -> float:
    """Normal Nash for others, GDM spends 0 on safety (knows they're safe), alignment forced."""
    s_modified = list(baseline_nash)
    s_modified[2] = 0.0  # GDM puts everything into capabilities
    return expected_human_share_with_overrides(
        s_all=s_modified,
        R=defaults.R,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
        override_probs={2: 1.0},
    )


def _intervention_double_gdm_resources() -> float:
    """Double GDM's resource share, renormalize."""
    R_new = defaults.R.copy()
    R_new[2] *= 2
    R_new /= R_new.sum()
    s_star = primitives.full_model_nash(
        R=R_new,
        A=defaults.A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    return primitives.expected_human_share(
        s_all=s_star,
        R=R_new,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )


def _intervention_increase_amity() -> float:
    """Increase amity 10% toward 1."""
    A_new = defaults.A + 0.1 * (np.ones_like(defaults.A) - defaults.A)
    s_star = primitives.full_model_nash(
        R=defaults.R,
        A=A_new,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    return primitives.expected_human_share(
        s_all=s_star,
        R=defaults.R,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )


def _intervention_public_good() -> float:
    """Make safety a public good (delta=1)."""
    s_star = primitives.full_model_nash(
        R=defaults.R,
        A=defaults.A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=1.0,
        rho=defaults.RHO,
    )
    return primitives.expected_human_share(
        s_all=s_star,
        R=defaults.R,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=1.0,
        rho=defaults.RHO,
    )


def compute_interventions() -> list[tuple[str, float, float]]:
    """Compute (name, baseline_human_share, intervention_human_share) for each intervention."""
    return _compute_all()


def _compute_all() -> list[tuple[str, float, float]]:
    """Run all 7 interventions and return results."""
    baseline_nash, baseline_ehs = _baseline_nash()

    return [
        ("Remove China", baseline_ehs, _intervention_remove_china()),
        ("Duplicate every lab", baseline_ehs, _intervention_duplicate_labs()),
        (
            "Secretly make GDM 100% safe",
            baseline_ehs,
            _intervention_secretly_safe_gdm(baseline_nash),
        ),
        (
            "Make GDM 100% safe, tell only them",
            baseline_ehs,
            _intervention_tell_only_gdm(baseline_nash),
        ),
        ("Double GDM's resources", baseline_ehs, _intervention_double_gdm_resources()),
        ("Increase amity 10% toward 1", baseline_ehs, _intervention_increase_amity()),
        ("Make safety a public good (delta=1)", baseline_ehs, _intervention_public_good()),
    ]
