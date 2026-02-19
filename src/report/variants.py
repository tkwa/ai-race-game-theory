"""Game-theoretic AI safety variant computations for plotting."""

import numpy as np

from report import primitives


def safety_elasticity() -> dict:
    """Compute Nash safety spending and joint survival across safety-technology effectiveness k."""
    k_arr = np.geomspace(0.05, 200, 100)
    s_star_a1 = np.empty(100)
    p_sq_a1 = np.empty(100)
    s_star_a_default = np.empty(100)
    p_sq_a_default = np.empty(100)

    for i, k in enumerate(k_arr):
        s1 = primitives.symmetric_nash(2, k, 1.0)
        s_star_a1[i] = s1
        p1 = primitives.alignment_prob(s1, k, 1.0)
        p_sq_a1[i] = p1**2

        sd = primitives.symmetric_nash(2, k, primitives.DEFAULT_ALPHA)
        s_star_a_default[i] = sd
        pd = primitives.alignment_prob(sd, k, primitives.DEFAULT_ALPHA)
        p_sq_a_default[i] = pd**2

    return {
        "k": k_arr,
        "s_star_a1": s_star_a1,
        "p_sq_a1": p_sq_a1,
        "s_star_a_default": s_star_a_default,
        "p_sq_a_default": p_sq_a_default,
    }


def n_actors_sweep() -> dict:
    """Compute Nash equilibrium and joint survival as number of actors increases."""
    n_arr = np.arange(2, 11)
    s_star = np.empty(len(n_arr))
    joint_survival = np.empty(len(n_arr))

    for i, n in enumerate(n_arr):
        s = primitives.symmetric_nash(int(n), primitives.DEFAULT_K, primitives.DEFAULT_ALPHA)
        s_star[i] = s
        p = primitives.alignment_prob(s, primitives.DEFAULT_K, primitives.DEFAULT_ALPHA)
        joint_survival[i] = p ** int(n)

    return {"n": n_arr, "s_star": s_star, "joint_survival": joint_survival}


def _resource_payoff_factory(
    resources: list[float],
    k_values: list[float],
    alpha: float,
    w: float = 1.0,
    public_good_delta: float | None = None,
) -> callable:
    """Return a payoff_fn_factory for asymmetric_nash with heterogeneous resources and k."""
    n = len(resources)

    def factory(i: int, s_all: list[float]) -> callable:
        def payoff(s_i: float) -> float:
            s_vec = list(s_all)
            s_vec[i] = s_i

            # Effective safety with optional public-good spillover
            if public_good_delta is not None:
                s_avg = sum(s_vec) / n
                eff = [sj**public_good_delta * s_avg ** (1.0 - public_good_delta) for sj in s_vec]
            else:
                eff = list(s_vec)

            probs = [primitives.alignment_prob(eff[j], k_values[j], alpha) for j in range(n)]
            joint = 1.0
            for p in probs:
                joint *= p

            caps = [resources[j] * max(1.0 - s_vec[j], 1e-15) for j in range(n)]
            caps_w = [c**w for c in caps]
            total_cap = sum(caps_w)
            share = caps_w[i] / total_cap if total_cap > 0 else 0.0

            return joint * 100.0 * share

        return payoff

    return factory


def different_resources_2() -> dict:
    """Compute asymmetric Nash for 2 players with varying resource ratios."""
    r_arr = np.linspace(0.1, 0.9, 50)
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


def different_resources_5() -> dict:
    """Compute asymmetric Nash for 5 players with 1/i resources across varying k."""
    k_arr = np.geomspace(0.05, 200, 50)
    raw_resources = [1.0 / i for i in range(1, 6)]
    total = sum(raw_resources)
    resources = [r / total for r in raw_resources]

    s_stars = np.empty((50, 5))
    probs = np.empty((50, 5))
    js = np.empty(50)

    alpha = primitives.DEFAULT_ALPHA

    for idx, k in enumerate(k_arr):
        k_values = [float(k)] * 5
        factory = _resource_payoff_factory(resources, k_values, alpha)
        s_all = primitives.asymmetric_nash(5, factory)
        for j in range(5):
            s_stars[idx, j] = s_all[j]
            probs[idx, j] = primitives.alignment_prob(s_all[j], float(k), alpha)
        js[idx] = float(np.prod(probs[idx]))

    return {"k": k_arr, "s_stars": s_stars, "probs": probs, "joint_survival": js}


def comparative_advantage() -> dict:
    """Compute asymmetric Nash where player A has double the safety technology of player B."""
    k_arr = np.geomspace(0.05, 200, 100)
    s_a = np.empty(100)
    s_b = np.empty(100)
    p_a = np.empty(100)
    p_b = np.empty(100)
    js = np.empty(100)

    alpha = primitives.DEFAULT_ALPHA

    for i, base_k in enumerate(k_arr):
        k_values = [2.0 * float(base_k), float(base_k)]
        resources = [0.5, 0.5]
        factory = _resource_payoff_factory(resources, k_values, alpha)
        s_all = primitives.asymmetric_nash(2, factory)
        s_a[i] = s_all[0]
        s_b[i] = s_all[1]
        p_a[i] = primitives.alignment_prob(s_all[0], k_values[0], alpha)
        p_b[i] = primitives.alignment_prob(s_all[1], k_values[1], alpha)
        js[i] = p_a[i] * p_b[i]

    return {"k": k_arr, "s_A": s_a, "s_B": s_b, "p_A": p_a, "p_B": p_b, "joint_survival": js}


def public_good_safety() -> dict:
    """Compute Nash equilibrium with public-good safety spillovers across delta values."""
    delta_arr = np.linspace(0, 1, 50)
    s_star = np.empty(50)
    js = np.empty(50)

    k = primitives.DEFAULT_K
    alpha = primitives.DEFAULT_ALPHA
    n = 5

    for i, delta in enumerate(delta_arr):
        d = float(delta)
        # delta=0 means fully public, delta=1 means fully private
        s = primitives.symmetric_nash(n, k, alpha, public_good_delta=d if d > 0 else 1e-10)
        s_star[i] = s
        s_avg = s  # symmetric: all play s*
        eff_s = s**d * s_avg ** (1.0 - d) if d > 0 else s_avg
        p = primitives.alignment_prob(eff_s, k, alpha)
        js[i] = p**n

    return {"delta": delta_arr, "s_star": s_star, "joint_survival": js}


def winner_take_all() -> dict:
    """Compute Nash equilibrium as winner-take-all exponent w varies."""
    w_arr = np.linspace(0.5, 5, 50)
    s_star = np.empty(50)
    p_sq = np.empty(50)

    k = primitives.DEFAULT_K
    alpha = primitives.DEFAULT_ALPHA

    for i, w in enumerate(w_arr):
        s = primitives.symmetric_nash(2, k, alpha, w=float(w))
        s_star[i] = s
        p = primitives.alignment_prob(s, k, alpha)
        p_sq[i] = p**2

    return {"w": w_arr, "s_star": s_star, "p_sq": p_sq}


def correlated_alignment() -> dict:
    """Compute Nash equilibrium and copula-based joint survival across correlation values."""
    rho_arr = np.linspace(0, 0.95, 50)
    s_star = np.empty(50)
    js = np.empty(50)

    k = primitives.DEFAULT_K
    alpha = primitives.DEFAULT_ALPHA

    for i, rho in enumerate(rho_arr):
        s = primitives.symmetric_nash(2, k, alpha, correlation=float(rho))
        s_star[i] = s
        p = primitives.alignment_prob(s, k, alpha)
        js[i] = primitives.joint_survival_copula([p, p], float(rho))

    return {"rho": rho_arr, "s_star": s_star, "joint_survival": js}


def heatmap_alpha_k() -> dict:
    """Compute joint survival heatmap over (k, alpha) grid for 2 symmetric players."""
    k_arr = np.geomspace(0.05, 200, 40)
    alpha_arr = np.linspace(0.1, 2.0, 40)
    js = np.empty((40, 40))

    for i, k in enumerate(k_arr):
        for j, alpha in enumerate(alpha_arr):
            s = primitives.symmetric_nash(2, float(k), float(alpha))
            p = primitives.alignment_prob(s, float(k), float(alpha))
            js[i, j] = p**2

    return {"k": k_arr, "alpha": alpha_arr, "joint_survival": js}


def heatmap_n_w() -> dict:
    """Compute joint survival heatmap over (n, w) grid."""
    n_arr = np.arange(2, 9)
    w_arr = np.linspace(0.5, 5, 30)
    js = np.empty((len(n_arr), len(w_arr)))

    k = primitives.DEFAULT_K
    alpha = primitives.DEFAULT_ALPHA

    for i, n in enumerate(n_arr):
        for j, w in enumerate(w_arr):
            s = primitives.symmetric_nash(int(n), k, alpha, w=float(w))
            p = primitives.alignment_prob(s, k, alpha)
            js[i, j] = p ** int(n)

    return {"n": n_arr, "w": w_arr, "joint_survival": js}


def public_good_vs_correlation() -> dict:
    """Compute joint survival heatmap over (delta, rho) grid for 2 actors."""
    delta_arr = np.linspace(0, 1, 30)
    rho_arr = np.linspace(0, 0.95, 30)
    js = np.empty((30, 30))

    k = primitives.DEFAULT_K
    alpha = primitives.DEFAULT_ALPHA

    for i, delta in enumerate(delta_arr):
        for j, rho in enumerate(rho_arr):
            d = float(delta) if float(delta) > 0 else 1e-10
            s = primitives.symmetric_nash(2, k, alpha, public_good_delta=d, correlation=float(rho))
            s_avg = s  # symmetric
            eff_s = s**d * s_avg ** (1.0 - d)
            p = primitives.alignment_prob(eff_s, k, alpha)
            js[i, j] = primitives.joint_survival_copula([p, p], float(rho))

    return {"delta": delta_arr, "rho": rho_arr, "joint_survival": js}


def _gini_coefficient(resources: np.ndarray) -> float:
    """Compute Gini coefficient for a resource distribution."""
    n = len(resources)
    sorted_r = np.sort(resources)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_r) / (n * np.sum(sorted_r))) - (n + 1.0) / n)


def resource_inequality_vs_n() -> dict:
    """Compute joint survival vs resource inequality (Gini) for varying actor counts."""
    # Power-law exponents to generate different Gini values
    exponents = np.linspace(0.0, 3.0, 30)
    n_values = [2, 3, 5, 8]

    # Pre-compute Gini for a reference n (use max n to get stable Gini estimates)
    ref_n = max(n_values)
    gini_arr = np.empty(30)
    for idx, exp in enumerate(exponents):
        raw = np.array([1.0 / (i**exp) for i in range(1, ref_n + 1)])
        raw /= raw.sum()
        gini_arr[idx] = _gini_coefficient(raw)

    k = primitives.DEFAULT_K
    alpha = primitives.DEFAULT_ALPHA

    joint_survival: dict[int, np.ndarray] = {}
    for n in n_values:
        js = np.empty(30)
        for idx, exp in enumerate(exponents):
            raw = np.array([1.0 / (i**exp) for i in range(1, n + 1)])
            resources = (raw / raw.sum()).tolist()
            k_values = [k] * n
            factory = _resource_payoff_factory(resources, k_values, alpha)
            s_all = primitives.asymmetric_nash(n, factory)
            probs = [primitives.alignment_prob(s_all[j], k, alpha) for j in range(n)]
            js[idx] = float(np.prod(probs))
        joint_survival[n] = js

    return {
        "gini": gini_arr,
        "n_values": n_values,
        "joint_survival": joint_survival,
    }


def comparative_advantage_public_good() -> dict:
    """Compute asymmetric Nash with comparative advantage and public-good spillovers."""
    delta_arr = np.linspace(0, 1, 50)
    p_a = np.empty(50)
    p_b = np.empty(50)
    js = np.empty(50)

    k_a = 2.0 * primitives.DEFAULT_K
    k_b = primitives.DEFAULT_K
    alpha = primitives.DEFAULT_ALPHA
    n = 2

    for i, delta in enumerate(delta_arr):
        d = float(delta) if float(delta) > 0 else 1e-10
        resources = [0.5, 0.5]
        k_values = [k_a, k_b]
        factory = _resource_payoff_factory(resources, k_values, alpha, public_good_delta=d)
        s_all = primitives.asymmetric_nash(n, factory)

        s_avg = sum(s_all) / n
        eff_a = s_all[0] ** d * s_avg ** (1.0 - d)
        eff_b = s_all[1] ** d * s_avg ** (1.0 - d)
        p_a[i] = primitives.alignment_prob(eff_a, k_a, alpha)
        p_b[i] = primitives.alignment_prob(eff_b, k_b, alpha)
        js[i] = p_a[i] * p_b[i]

    return {"delta": delta_arr, "p_A": p_a, "p_B": p_b, "joint_survival": js}
