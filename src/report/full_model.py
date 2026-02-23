"""Nash equilibrium and expected human share vs k for the full 5-lab model."""

import numpy as np

from report import defaults, primitives


def default_outcomes_vs_k() -> dict:
    """Nash equilibrium and expected human share vs k for the full 5-lab model."""
    k_values = np.geomspace(0.6, 200, 50)
    results: dict[str, list] = {
        "k": [],
        "s_star": [],
        "expected_human_share": [],
    }

    prev_s: list[float] | None = None
    for k in k_values:
        s_star = primitives.full_model_nash(
            R=defaults.R,
            A=defaults.A,
            k=k,
            alpha=defaults.ALPHA,
            w=defaults.W,
            z=defaults.Z,
            delta=defaults.DELTA,
            rho=defaults.RHO,
            initial_guess=prev_s,
        )
        ehs = primitives.expected_human_share(
            s_all=s_star,
            R=defaults.R,
            k=k,
            alpha=defaults.ALPHA,
            w=defaults.W,
            z=defaults.Z,
            delta=defaults.DELTA,
            rho=defaults.RHO,
        )
        prev_s = s_star
        results["k"].append(float(k))
        results["s_star"].append(s_star)
        results["expected_human_share"].append(ehs)

    return results


def default_lab_summary() -> dict:
    """Per-lab Ω share (all-aligned) and P(misalignment) at the default Nash equilibrium."""
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
    n = len(s_star)
    eff_s = primitives._effective_safety(s_star, defaults.DELTA)
    p_aligned = [
        primitives.alignment_prob(eff_s[j], defaults.K, defaults.ALPHA, c=1.0 - s_star[j])
        for j in range(n)
    ]
    abs_cap = np.array([defaults.R[j] * (1.0 - s_star[j]) for j in range(n)])
    cap_w = abs_cap**defaults.W
    omega_shares = cap_w / cap_w.sum()
    return {
        "lab_names": defaults.get_lab_names(),
        "omega_share": omega_shares.tolist(),
        "p_misaligned": [1.0 - p for p in p_aligned],
        "s_star": list(s_star),
    }
