"""Core game-theoretic model of AI safety race dynamics."""

import scipy.optimize as optimize


def alignment_probability(s: float, k: float) -> float:
    """Probability of alignment given safety fraction s and effectiveness k."""
    c = 1.0 - s
    return k * s / (k * s + c)


def payoff(s_i: float, s_j: float, k: float) -> float:
    """Expected payoff to country i given both countries' safety fractions."""
    p_i = alignment_probability(s_i, k)
    p_j = alignment_probability(s_j, k)
    c_i = 1.0 - s_i
    c_j = 1.0 - s_j
    # Avoid division by zero when both capabilities are zero
    if c_i + c_j == 0:
        return 0.0
    return p_i * p_j * 100.0 * c_i / (c_i + c_j)


def _payoff_negative(s_i: float, s_j: float, k: float) -> float:
    """Negative payoff for minimization."""
    return -payoff(s_i, s_j, k)


def find_nash_equilibrium(k: float) -> float:
    """Find symmetric Nash equilibrium safety fraction s* for given k."""

    # At equilibrium, country i maximizes payoff(s_i, s*, k) at s_i = s*
    # Use numerical optimization: find s* where the best response to s* is s* itself
    def best_response(s_j: float) -> float:
        result = optimize.minimize_scalar(
            lambda s_i: _payoff_negative(s_i, s_j, k),
            bounds=(1e-10, 1.0 - 1e-10),
            method="bounded",
        )
        return result.x

    # Fixed point iteration: s* = best_response(s*)
    def fixed_point_residual(s: float) -> float:
        return best_response(s) - s

    result = optimize.brentq(fixed_point_residual, 1e-6, 1.0 - 1e-6)
    return float(result)


def survival_probability(k: float) -> float:
    """Probability both AIs are aligned at Nash equilibrium for given k."""
    s_star = find_nash_equilibrium(k)
    p = alignment_probability(s_star, k)
    return p * p
