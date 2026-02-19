"""Core primitives for game-theoretic AI safety variants."""

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats

# Calibrated so that 1% spending -> 80% aligned, 50% spending -> 98% aligned
DEFAULT_K = 33.9
DEFAULT_ALPHA = 0.466


def alignment_prob(s: float, k: float, alpha: float = 1.0) -> float:
    """P(aligned) = k·s^α / (k·s^α + (1-s)), with safety elasticity α."""
    if s <= 0:
        return 0.0
    if s >= 1:
        return 1.0
    c = 1.0 - s
    ks_alpha = k * s**alpha
    return ks_alpha / (ks_alpha + c)


def symmetric_nash(
    n: int,
    k: float,
    alpha: float,
    w: float = 1.0,
    public_good_delta: float | None = None,
    correlation: float = 0.0,
) -> float:
    """Find symmetric Nash equilibrium s* for n identical players."""

    def best_response(s_j: float) -> float:
        """Best response when all n-1 others play s_j."""

        def neg_payoff(s_i: float) -> float:
            # Effective safety for alignment calc
            if public_good_delta is not None:
                s_avg = ((n - 1) * s_j + s_i) / n
                eff_i = s_i**public_good_delta * s_avg ** (1.0 - public_good_delta)
                eff_j = s_j**public_good_delta * s_avg ** (1.0 - public_good_delta)
            else:
                eff_i = s_i
                eff_j = s_j

            p_i = alignment_prob(eff_i, k, alpha)
            p_j = alignment_prob(eff_j, k, alpha)

            # Joint survival
            if correlation > 0 and n >= 2:
                probs = [p_i] + [p_j] * (n - 1)
                joint = joint_survival_copula(probs, correlation)
            else:
                joint = p_i * p_j ** (n - 1)

            # Capability share with winner-take-all exponent
            c_i = max(1.0 - s_i, 1e-15) ** w
            c_others = (n - 1) * max(1.0 - s_j, 1e-15) ** w
            share = c_i / (c_i + c_others) if (c_i + c_others) > 0 else 0.0

            return -(joint * 100.0 * share)

        result = optimize.minimize_scalar(neg_payoff, bounds=(1e-10, 1.0 - 1e-10), method="bounded")
        return result.x

    def fixed_point_residual(s: float) -> float:
        return best_response(s) - s

    try:
        result = optimize.brentq(fixed_point_residual, 1e-6, 1.0 - 1e-6)
        return float(result)
    except ValueError:
        # Fallback: iterated best response
        s = 0.5
        for _ in range(200):
            s_new = best_response(s)
            if abs(s_new - s) < 1e-8:
                return s_new
            s = 0.5 * s + 0.5 * s_new
        return s


def asymmetric_nash(
    n: int,
    payoff_fn_factory: callable,
    initial_guess: list[float] | None = None,
) -> list[float]:
    """Find Nash equilibrium via iterated best-response with damping."""
    s_all = list(initial_guess) if initial_guess else [0.5] * n
    damping = 0.5

    for _ in range(200):
        s_prev = list(s_all)
        for i in range(n):
            payoff_fn = payoff_fn_factory(i, list(s_all))

            def neg_payoff(s_i: float) -> float:
                return -payoff_fn(s_i)

            result = optimize.minimize_scalar(
                neg_payoff, bounds=(1e-10, 1.0 - 1e-10), method="bounded"
            )
            s_all[i] = damping * s_prev[i] + (1.0 - damping) * result.x

        if max(abs(s_all[j] - s_prev[j]) for j in range(n)) < 1e-8:
            break

    return s_all


def joint_survival_copula(probs: list[float], rho: float) -> float:
    """Joint survival via Gaussian copula with equicorrelation ρ."""
    n = len(probs)
    if rho <= 0 or n == 1:
        result = 1.0
        for p in probs:
            result *= p
        return result

    # Clamp to avoid infinities in Φ⁻¹
    z = np.array([stats.norm.ppf(np.clip(p, 1e-10, 1.0 - 1e-10)) for p in probs])

    # Equicorrelation matrix
    cov = np.full((n, n), rho)
    np.fill_diagonal(cov, 1.0)

    try:
        return float(stats.multivariate_normal.cdf(z, mean=np.zeros(n), cov=cov))
    except Exception:
        # Fallback for numerical issues
        result = 1.0
        for p in probs:
            result *= p
        return result
