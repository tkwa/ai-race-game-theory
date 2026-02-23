"""Debug convergence for problematic parameter regimes."""

import sys

sys.path.insert(0, "src")

import numpy as np
from scipy import optimize

from report import defaults, primitives


def run_with_diagnostics(delta: float = 0.75, k: float = 33.9) -> None:
    """Run Nash solver with detailed iteration logging."""
    R = defaults.R
    A = defaults.A
    n = len(R)
    alpha = defaults.ALPHA
    w = defaults.W
    z = defaults.Z
    rho = defaults.RHO

    s_all = [0.3] * n
    damping = 0.7

    print(f"Params: k={k}, delta={delta}")
    print(f"{'Iter':>5} {'MaxChg':>10} {'Damp':>6}  s_values")

    for iteration in range(500):
        s_prev = list(s_all)
        for i in range(n):

            def neg_payoff(s_i: float, _i: int = i) -> float:
                s_trial = list(s_all)
                s_trial[_i] = s_i
                payoffs = primitives.full_model_payoffs(s_trial, R, A, k, alpha, w, z, delta, rho)
                return -payoffs[_i]

            result = optimize.minimize_scalar(
                neg_payoff, bounds=(1e-10, 1.0 - 1e-10), method="bounded"
            )
            s_all[i] = damping * s_prev[i] + (1.0 - damping) * result.x

        max_change = max(abs(s_all[j] - s_prev[j]) for j in range(n))
        if iteration < 20 or iteration % 50 == 0 or max_change < 1e-4:
            s_str = " ".join(f"{s:.5f}" for s in s_all)
            print(f"{iteration:5d} {max_change:10.2e} {damping:6.3f}  {s_str}")

        if max_change < 5e-6:
            print(f"Converged at iteration {iteration}")
            return

    # Try undamped best response from current point
    print("\nTrying undamped from current point...")
    for iteration in range(100):
        s_prev = list(s_all)
        for i in range(n):

            def neg_payoff(s_i: float, _i: int = i) -> float:
                s_trial = list(s_all)
                s_trial[_i] = s_i
                payoffs = primitives.full_model_payoffs(s_trial, R, A, k, alpha, w, z, delta, rho)
                return -payoffs[_i]

            result = optimize.minimize_scalar(
                neg_payoff, bounds=(1e-10, 1.0 - 1e-10), method="bounded"
            )
            # Pure best response (no damping)
            br = result.x
            # Record the undamped best response residual
            s_all[i] = br

        max_change = max(abs(s_all[j] - s_prev[j]) for j in range(n))
        if iteration < 10 or iteration % 20 == 0:
            s_str = " ".join(f"{s:.5f}" for s in s_all)
            print(f"{iteration:5d} {max_change:10.2e}  {s_str}")

    # Try Newton on the fixed-point residual
    print("\nTrying scipy.optimize.root on fixed-point residual...")

    def residual(s_vec: np.ndarray) -> np.ndarray:
        s_list = list(np.clip(s_vec, 1e-10, 1 - 1e-10))
        br = []
        for i in range(n):

            def neg_payoff(s_i: float, _i: int = i) -> float:
                s_trial = list(s_list)
                s_trial[_i] = s_i
                payoffs = primitives.full_model_payoffs(s_trial, R, A, k, alpha, w, z, delta, rho)
                return -payoffs[_i]

            result = optimize.minimize_scalar(
                neg_payoff, bounds=(1e-10, 1.0 - 1e-10), method="bounded"
            )
            br.append(result.x)
        return np.array(br) - s_vec

    from scipy.optimize import root

    x0 = np.array(s_all)
    sol = root(residual, x0, method="hybr", tol=1e-8)
    print(f"Root finder success: {sol.success}")
    print(f"Residual norm: {np.max(np.abs(sol.fun)):.2e}")
    print(f"Solution: {sol.x}")
    if sol.success:
        # Verify it's a fixed point
        res = residual(sol.x)
        print(f"Verification residual: {np.max(np.abs(res)):.2e}")


if __name__ == "__main__":
    print("=== delta=0.75 ===")
    run_with_diagnostics(delta=0.75)
    print("\n\n=== k=0.5 ===")
    run_with_diagnostics(delta=0.2, k=0.5)
