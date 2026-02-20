"""Compute Nash equilibrium and expected human share for interventions."""

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


def _intervention_remove_anthropic() -> float:
    """Remove Anthropic (index 1), renormalize R, run 4-lab Nash."""
    keep = [0, 2, 3, 4]
    R_new = defaults.R[keep].copy()
    R_new /= R_new.sum()
    A_new = defaults.A[np.ix_(keep, keep)].copy()
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


def _intervention_tell_everyone_gdm_safe() -> float:
    """Everyone knows GDM is safe; all re-optimize. GDM fixed at alignment=1."""
    # GDM (index 2) is fixed safe and spends 0 on safety; others re-optimize
    # knowing GDM is guaranteed aligned (override_probs visible during Nash)
    s_star = primitives.full_model_nash(
        R=defaults.R,
        A=defaults.A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
        fixed={2: 0.0},
        override_probs={2: 1.0},
    )
    return expected_human_share_with_overrides(
        s_all=s_star,
        R=defaults.R,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
        override_probs={2: 1.0},
    )


def _intervention_top3_mutual_amity() -> float:
    """Top 3 labs (GDM, OAI, Ant) have mutual amity 1."""
    top3 = [0, 1, 2]  # OAI, Ant, GDM
    A_new = defaults.A.copy()
    for i in top3:
        for j in top3:
            A_new[i, j] = 1.0
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


def _intervention_merge_top3() -> float:
    """Merge top 3 labs (OAI, Ant, GDM) into one with summed resources."""
    top3 = [0, 1, 2]
    rest = [3, 4]
    R_merged = defaults.R[top3].sum()
    R_new = np.array([R_merged] + [defaults.R[i] for i in rest])
    # Merged lab's amity toward others = average of the 3 rows
    A_top3_avg = defaults.A[top3].mean(axis=0)
    n_new = 1 + len(rest)
    A_new = np.ones((n_new, n_new))
    A_new[0, 0] = 1.0
    for j_new, j_old in enumerate(rest, start=1):
        A_new[0, j_new] = A_top3_avg[j_old]
        # Other labs' amity toward merged lab = average of their amity toward the 3
        A_new[j_new, 0] = np.mean([defaults.A[j_old, t] for t in top3])
    for i_new, i_old in enumerate(rest, start=1):
        for j_new, j_old in enumerate(rest, start=1):
            A_new[i_new, j_new] = defaults.A[i_old, j_old]
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


def _intervention_us_china_amity_symmetry() -> float:
    """Increase US->China amity to match China->US levels."""
    A_new = defaults.A.copy()
    us_labs = [0, 1, 2, 3]
    china = 4
    for i in us_labs:
        A_new[i, china] = defaults.A[china, i]
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


def _intervention_usg_slowdown(baseline_nash: list[float]) -> float:
    """USG reduces US lab capabilities by 33%, keeping absolute safety the same."""
    us_labs = [0, 1, 2, 3]
    # Reduce capabilities by 33%: new total R_i = R_i * (s_i + 0.67*(1-s_i))
    R_new = defaults.R.copy()
    fixed = {}
    for i in us_labs:
        s_i = baseline_nash[i]
        R_new[i] = defaults.R[i] * (s_i + 0.67 * (1.0 - s_i))
        # New safety fraction preserves absolute safety: R_new * s_new = R_old * s_old
        fixed[i] = defaults.R[i] * s_i / R_new[i]
    # China adapts
    s_star = primitives.full_model_nash(
        R=R_new,
        A=defaults.A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
        fixed=fixed,
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


def _intervention_min_safety_10pct() -> float:
    """USG mandates all US labs spend at least 10% on safety."""
    us_labs = [0, 1, 2, 3]
    min_safety = {i: 0.10 for i in us_labs}
    s_star = primitives.full_model_nash(
        R=defaults.R,
        A=defaults.A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
        min_safety=min_safety,
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


def _intervention_gdm_zero_amity() -> float:
    """Set GDM's amity towards every other lab to 0."""
    A_new = defaults.A.copy()
    gdm = 2
    for j in range(len(defaults.R)):
        if j != gdm:
            A_new[gdm, j] = 0.0
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


def compute_interventions() -> list[tuple[str, float, float]]:
    """Compute (name, baseline_human_share, intervention_human_share) for each intervention."""
    return _compute_all()


def _compute_all() -> list[tuple[str, float, float]]:
    """Run all interventions and return results."""
    baseline_nash, baseline_ehs = _baseline_nash()

    return [
        ("Remove China", baseline_ehs, _intervention_remove_china()),
        ("Remove Anthropic", baseline_ehs, _intervention_remove_anthropic()),
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
        (
            "Make GDM 100% safe, tell everyone",
            baseline_ehs,
            _intervention_tell_everyone_gdm_safe(),
        ),
        ("Double GDM's resources", baseline_ehs, _intervention_double_gdm_resources()),
        ("Increase amity 10% toward 1", baseline_ehs, _intervention_increase_amity()),
        ("Make safety a public good (δ=1)", baseline_ehs, _intervention_public_good()),
        ("Top 3 labs: mutual amity 1", baseline_ehs, _intervention_top3_mutual_amity()),
        ("Merge top 3 labs", baseline_ehs, _intervention_merge_top3()),
        ("US→China amity = China→US", baseline_ehs, _intervention_us_china_amity_symmetry()),
        (
            "USG demands slowdown (−33% cap)",
            baseline_ehs,
            _intervention_usg_slowdown(baseline_nash),
        ),
        ("USG mandates ≥10% safety", baseline_ehs, _intervention_min_safety_10pct()),
        ("GDM amity → 0 to all others", baseline_ehs, _intervention_gdm_zero_amity()),
    ]
