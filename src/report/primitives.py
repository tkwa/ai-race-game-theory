"""Core primitives for game-theoretic AI safety variants."""

import functools

import numpy as np
import scipy.optimize as optimize
import scipy.special as special

# Calibrated so that 1% spending -> 80% aligned, 50% spending -> 98% aligned
DEFAULT_K = 33.9
DEFAULT_ALPHA = 0.466


def alignment_prob(s: float, k: float, alpha: float = 1.0, c: float | None = None) -> float:
    """P(aligned) = k·S^α / (k·S^α + c), where S is effective safety and c is raw capability."""
    if s <= 0:
        return 0.0
    cap = (1.0 - s) if c is None else c
    if cap <= 0:
        return 1.0
    ks_alpha = k * s**alpha
    return ks_alpha / (ks_alpha + cap)


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
                s_sum = (n - 1) * s_j + s_i
                eff_i = s_sum**public_good_delta * s_i ** (1.0 - public_good_delta)
                eff_j = s_sum**public_good_delta * s_j ** (1.0 - public_good_delta)
            else:
                eff_i = s_i
                eff_j = s_j

            p_i = alignment_prob(eff_i, k, alpha, c=1.0 - s_i)
            p_j = alignment_prob(eff_j, k, alpha, c=1.0 - s_j)

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
    else:
        raise RuntimeError("asymmetric_nash did not converge after 200 iterations")

    return s_all


def joint_survival_copula(probs: list[float], rho: float) -> float:
    """Joint survival via Gaussian copula with equicorrelation ρ.

    Uses the factorization X_i = √ρ·Z + √(1-ρ)·ε_i to reduce the
    n-dimensional CDF to 1D quadrature over the common factor Z.
    """
    n = len(probs)
    if rho <= 0 or n == 1:
        result = 1.0
        for p in probs:
            result *= p
        return result

    z = special.ndtri(np.clip(probs, 1e-10, 1.0 - 1e-10))
    sqrt_rho = np.sqrt(rho)
    sqrt_1mrho = np.sqrt(1.0 - rho)

    # Gauss-Hermite quadrature: ∫f(t)φ(t)dt = ∫f(t√2)exp(-t²)dt/√π
    nodes, weights = _gauss_hermite_cached(32)
    t_vals = nodes * np.sqrt(2)  # transform to standard normal
    w_vals = weights / np.sqrt(np.pi)

    # Vectorized: (n_nodes, n_probs) -> product over probs -> dot with weights
    args = (z[np.newaxis, :] - sqrt_rho * t_vals[:, np.newaxis]) / sqrt_1mrho
    conditional = np.prod(special.ndtr(args), axis=1)
    result = np.dot(w_vals, conditional)
    return float(np.clip(result, 0.0, 1.0))


@functools.lru_cache(maxsize=8)
def _gauss_hermite_cached(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Hermite quadrature nodes and weights."""
    return np.polynomial.hermite.hermgauss(n)


# ---------------------------------------------------------------------------
# Full model with Ω, Ã, z
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _alignment_mask(n: int) -> np.ndarray:
    """Cached (2^n, n) boolean matrix where bit j of index encodes alignment."""
    n_outcomes = 1 << n
    indices = np.arange(n_outcomes)[:, np.newaxis]
    bits = np.arange(n)[np.newaxis, :]
    return ((indices >> bits) & 1).astype(bool)


def _outcome_probabilities(probs: np.ndarray, rho: float) -> np.ndarray:
    """Compute P(outcome) for all 2^n alignment outcome vectors.

    Returns array of shape (2^n,) where index bits encode alignment
    (bit j set = lab j aligned).
    """
    n = len(probs)
    aligned = _alignment_mask(n)

    if rho <= 0:
        # Independent: product of p_j or (1-p_j) per outcome
        per_lab = np.where(aligned, probs, 1.0 - probs)  # (2^n, n)
        return np.prod(per_lab, axis=1)

    # Gaussian copula via Gauss-Hermite quadrature
    z_vals = special.ndtri(np.clip(probs, 1e-10, 1.0 - 1e-10))
    sqrt_rho = np.sqrt(rho)
    sqrt_1mrho = np.sqrt(1.0 - rho)

    nodes, weights = _gauss_hermite_cached(32)
    t_vals = nodes * np.sqrt(2)
    w_vals = weights / np.sqrt(np.pi)

    # Conditional probs: (n_nodes, n_labs)
    cond_p = special.ndtr((z_vals[np.newaxis, :] - sqrt_rho * t_vals[:, np.newaxis]) / sqrt_1mrho)
    cond_q = 1.0 - cond_p

    # Vectorized: select p or q per outcome per lab, then product over labs
    # aligned: (2^n, n) -> (1, 2^n, n); cond_p: (n_nodes, 1, n)
    per_lab = np.where(
        aligned[np.newaxis, :, :], cond_p[:, np.newaxis, :], cond_q[:, np.newaxis, :]
    )
    # (n_nodes, 2^n)
    prod = np.prod(per_lab, axis=2)
    result = w_vals @ prod  # (2^n,)

    return np.clip(result, 0.0, 1.0)


def _effective_safety(s_all: list[float], delta: float | None) -> list[float]:
    """Compute effective safety levels with optional public-good spillover."""
    if delta is None or delta <= 0:
        return list(s_all)
    s_sum = sum(s_all)
    if s_sum <= 0:
        return [0.0] * len(s_all)
    return [s_sum**delta * s ** (1.0 - delta) for s in s_all]


def _precompute_omega_table(
    abs_cap: np.ndarray, w: float, z: float
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute Ω and aligned masks for all 2^n outcomes.

    Returns (omega_table, aligned_masks) where:
    - omega_table: (2^n, n) normalized Ω values
    - aligned_masks: (2^n, n) boolean alignment masks
    """
    n = len(abs_cap)

    # Precompute per-lab values for both aligned and misaligned
    cap_aligned = abs_cap**w
    cap_misaligned = abs_cap ** (w * (1.0 - z))

    aligned_masks = _alignment_mask(n)

    # Compute raw omega: select aligned or misaligned capability per lab
    omega_raw = np.where(aligned_masks, cap_aligned, cap_misaligned)
    totals = omega_raw.sum(axis=1, keepdims=True)
    totals = np.maximum(totals, 1e-30)
    omega_table = omega_raw / totals

    return omega_table, aligned_masks


def full_model_payoffs(
    s_all: list[float],
    R: np.ndarray,
    A: np.ndarray,
    k: float,
    alpha: float,
    w: float,
    z: float,
    delta: float | None = None,
    rho: float = 0.0,
    override_probs: dict[int, float] | None = None,
) -> np.ndarray:
    """Compute expected payoff for each lab under the full Ω model."""
    n = len(s_all)
    eff_s = _effective_safety(s_all, delta)
    probs = np.array([alignment_prob(eff_s[j], k, alpha, c=1.0 - s_all[j]) for j in range(n)])
    if override_probs:
        for idx, p in override_probs.items():
            probs[idx] = p
    abs_cap = np.array([R[j] * max(1.0 - s_all[j], 1e-15) for j in range(n)])

    outcome_probs = _outcome_probabilities(probs, rho)
    omega_table, aligned_masks = _precompute_omega_table(abs_cap, w, z)

    # Ã_ij = A_ij if j aligned, 0 otherwise
    # payoff_i = Σ_outcome P(outcome) * Σ_j Ω_j * Ã_ij
    #          = Σ_outcome P(outcome) * Σ_j [Ω_j * A_ij * aligned_j]
    # Vectorized: (2^n, n) * (2^n,1) -> weighted by outcome probs
    # For each outcome: value_i = Σ_j omega[j] * A[i,j] * aligned[j]
    #                            = (omega * aligned) @ A.T  row i
    masked_omega = omega_table * aligned_masks  # (2^n, n): Ω_j if aligned, 0 otherwise
    # (2^n, n) @ (n, n) -> (2^n, n): payoff contribution per outcome per lab
    outcome_payoffs = masked_omega @ A.T
    # Weight by outcome probabilities
    return outcome_probs @ outcome_payoffs  # (n,)


def expected_human_share(
    s_all: list[float],
    R: np.ndarray,
    k: float,
    alpha: float,
    w: float,
    z: float,
    delta: float | None = None,
    rho: float = 0.0,
) -> float:
    """Expected fraction of universe controlled by aligned AI."""
    n = len(s_all)
    eff_s = _effective_safety(s_all, delta)
    probs = np.array([alignment_prob(eff_s[j], k, alpha, c=1.0 - s_all[j]) for j in range(n)])
    abs_cap = np.array([R[j] * max(1.0 - s_all[j], 1e-15) for j in range(n)])

    outcome_probs = _outcome_probabilities(probs, rho)
    omega_table, aligned_masks = _precompute_omega_table(abs_cap, w, z)

    # Aligned share per outcome = sum of Ω_j for aligned labs
    aligned_shares = (omega_table * aligned_masks).sum(axis=1)  # (2^n,)
    return float(outcome_probs @ aligned_shares)


def full_model_nash(
    R: np.ndarray,
    A: np.ndarray,
    k: float,
    alpha: float,
    w: float,
    z: float,
    delta: float | None = None,
    rho: float = 0.0,
    initial_guess: list[float] | None = None,
    fixed: dict[int, float] | None = None,
    min_safety: dict[int, float] | None = None,
    override_probs: dict[int, float] | None = None,
) -> list[float]:
    """Find Nash equilibrium for the full model via iterated best response.

    fixed: dict mapping lab index -> fixed safety fraction (not optimized).
    min_safety: dict mapping lab index -> minimum safety fraction (constrained optimization).
    override_probs: dict mapping lab index -> forced alignment probability (known to all).
    """
    n = len(R)
    free_indices = [i for i in range(n) if not (fixed and i in fixed)]
    s_all = list(initial_guess) if initial_guess else [0.3] * n
    if fixed:
        for idx, val in fixed.items():
            s_all[idx] = val
    damping = 0.7
    prev_change = float("inf")
    stall_count = 0
    tol = 3e-6

    for iteration in range(4000):
        s_prev = list(s_all)
        for i in free_indices:
            lo = min_safety[i] if (min_safety and i in min_safety) else 1e-10

            def neg_payoff(s_i: float, _i: int = i) -> float:
                s_trial = list(s_all)
                s_trial[_i] = s_i
                return -full_model_payoffs(
                    s_trial, R, A, k, alpha, w, z, delta, rho, override_probs
                )[_i]

            result = optimize.minimize_scalar(
                neg_payoff, bounds=(lo, 1.0 - 1e-10), method="bounded"
            )
            s_all[i] = damping * s_prev[i] + (1.0 - damping) * result.x

        max_change = max(abs(s_all[j] - s_prev[j]) for j in range(n))
        if max_change < tol:
            break
        # Ramp damping only when change genuinely increases (oscillation)
        if max_change >= prev_change:
            stall_count += 1
            if damping < 0.995:
                damping = min(0.995, damping + 0.005 * min(stall_count, 5))
        else:
            stall_count = max(0, stall_count - 1)
        prev_change = max_change
    else:
        # Did not converge — average recent iterates to approximate limit cycle center
        if max_change < 0.1:
            recent = []
            for _ in range(100):
                s_prev = list(s_all)
                for i in free_indices:
                    lo = min_safety[i] if (min_safety and i in min_safety) else 1e-10

                    def neg_payoff(s_i: float, _i: int = i) -> float:
                        s_trial = list(s_all)
                        s_trial[_i] = s_i
                        return -full_model_payoffs(
                            s_trial, R, A, k, alpha, w, z, delta, rho, override_probs
                        )[_i]

                    result = optimize.minimize_scalar(
                        neg_payoff, bounds=(lo, 1.0 - 1e-10), method="bounded"
                    )
                    s_all[i] = damping * s_prev[i] + (1.0 - damping) * result.x
                recent.append(list(s_all))
            s_all = [float(np.mean([r[j] for r in recent])) for j in range(n)]
            if fixed:
                for idx, val in fixed.items():
                    s_all[idx] = val
        else:
            raise RuntimeError(
                f"full_model_nash did not converge after 4000 iterations "
                f"(max_change={prev_change:.2e})"
            )

    return s_all
