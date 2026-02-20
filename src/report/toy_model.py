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
