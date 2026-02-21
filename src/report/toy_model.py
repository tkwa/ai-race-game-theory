"""Data generation for the toy model section."""

import numpy as np

from report import primitives


def toy_baseline_vs_k() -> dict:
    """s* and P² vs k for n=2, α=0.466."""
    k_arr = np.geomspace(0.05, 200, 100)
    s_star = np.empty(100)
    p_sq = np.empty(100)

    alpha = primitives.DEFAULT_ALPHA

    for i, k in enumerate(k_arr):
        s = primitives.symmetric_nash(2, float(k), alpha)
        s_star[i] = s
        p = primitives.alignment_prob(s, float(k), alpha)
        p_sq[i] = p**2

    return {"k": k_arr, "s_star": s_star, "joint_survival": p_sq}


def toy_n_actors() -> dict:
    """s* and joint survival vs n at default k."""
    n_arr = np.arange(2, 11)
    s_star = np.empty(len(n_arr))
    joint_survival = np.empty(len(n_arr))

    k = primitives.DEFAULT_K
    alpha = primitives.DEFAULT_ALPHA

    for i, n in enumerate(n_arr):
        s = primitives.symmetric_nash(int(n), k, alpha)
        s_star[i] = s
        p = primitives.alignment_prob(s, k, alpha)
        joint_survival[i] = p ** int(n)

    return {"n": n_arr, "s_star": s_star, "joint_survival": joint_survival}


def _resource_payoff_factory(
    resources: list[float],
    k_values: list[float],
    alpha: float,
) -> callable:
    """Return a payoff_fn_factory for asymmetric_nash with heterogeneous resources and k."""
    n = len(resources)

    def factory(i: int, s_all: list[float]) -> callable:
        def payoff(s_i: float) -> float:
            s_vec = list(s_all)
            s_vec[i] = s_i

            probs = [primitives.alignment_prob(s_vec[j], k_values[j], alpha) for j in range(n)]
            joint = 1.0
            for p in probs:
                joint *= p

            caps = [resources[j] * max(1.0 - s_vec[j], 1e-15) for j in range(n)]
            total_cap = sum(caps)
            share = caps[i] / total_cap if total_cap > 0 else 0.0

            return joint * 100.0 * share

        return payoff

    return factory


def toy_asymmetric_resources() -> dict:
    """Safety spending and joint survival vs resource ratio for 2 asymmetric players."""
    r_arr = np.linspace(0.5, 0.95, 50)
    s1 = np.empty(50)
    s2 = np.empty(50)
    p1 = np.empty(50)
    p2 = np.empty(50)
    js = np.empty(50)

    k = primitives.DEFAULT_K
    alpha = primitives.DEFAULT_ALPHA

    for i, r in enumerate(r_arr):
        resources = [float(r), 1.0 - float(r)]
        k_values = [k, k]
        factory = _resource_payoff_factory(resources, k_values, alpha)
        s_all = primitives.asymmetric_nash(2, factory)
        s1[i] = s_all[0]
        s2[i] = s_all[1]
        p1[i] = primitives.alignment_prob(s_all[0], k, alpha)
        p2[i] = primitives.alignment_prob(s_all[1], k, alpha)
        js[i] = p1[i] * p2[i]

    return {"R": r_arr, "s1": s1, "s2": s2, "p1": p1, "p2": p2, "joint_survival": js}
