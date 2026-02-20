"""Nash equilibrium and expected human share vs k for the full 5-lab model."""

import numpy as np

from report import defaults, primitives


def default_outcomes_vs_k() -> dict:
    """Nash equilibrium and expected human share vs k for the full 5-lab model."""
    k_values = np.geomspace(0.05, 200, 50)
    results: dict[str, list] = {
        "k": [],
        "s_star": [],
        "expected_human_share": [],
    }

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
        results["k"].append(float(k))
        results["s_star"].append(s_star)
        results["expected_human_share"].append(ehs)

    return results
