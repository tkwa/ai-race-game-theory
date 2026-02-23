"""Check where the Nash solver fails to converge to 5e-6 tolerance."""

import sys

sys.path.insert(0, "src")

import numpy as np

from report import defaults


def check_convergence_at_params(**overrides: float) -> tuple[bool, float]:
    """Run Nash solver and check if it converged properly vs used averaging fallback."""
    params = {
        "R": defaults.R,
        "A": defaults.A,
        "k": defaults.K,
        "alpha": defaults.ALPHA,
        "w": defaults.W,
        "z": defaults.Z,
        "delta": defaults.DELTA,
        "rho": defaults.RHO,
    }
    params.update(overrides)

    # Monkey-patch to track convergence
    import report.primitives as p

    converged_properly = [True]
    final_change = [0.0]

    def patched_nash(**kwargs):
        n = len(kwargs["R"])
        s_all = [0.3] * n
        damping = 0.7
        prev_change = float("inf")
        stall_count = 0
        recent: list[list[float]] = []

        for iteration in range(4000):
            s_prev = list(s_all)
            for i in range(n):
                lo = 1e-10

                def neg_payoff(s_i: float, _i: int = i) -> float:
                    s_trial = list(s_all)
                    s_trial[_i] = s_i
                    payoffs = p.full_model_payoffs(
                        s_trial,
                        kwargs["R"],
                        kwargs["A"],
                        kwargs["k"],
                        kwargs["alpha"],
                        kwargs["w"],
                        kwargs["z"],
                        kwargs.get("delta"),
                        kwargs.get("rho", 0.0),
                    )
                    return -payoffs[_i]

                from scipy import optimize

                result = optimize.minimize_scalar(
                    neg_payoff, bounds=(lo, 1.0 - 1e-10), method="bounded"
                )
                s_all[i] = damping * s_prev[i] + (1.0 - damping) * result.x

            max_change = max(abs(s_all[j] - s_prev[j]) for j in range(n))
            if max_change < 5e-6:
                final_change[0] = max_change
                return s_all

            if max_change > prev_change * 0.9:
                stall_count += 1
                if damping < 0.995:
                    damping = min(0.995, damping + 0.005 * min(stall_count, 5))
            else:
                stall_count = max(0, stall_count - 1)
            prev_change = max_change

            if stall_count > 20:
                recent.append(list(s_all))
                if len(recent) > 50:
                    recent.pop(0)
                if len(recent) >= 50:
                    converged_properly[0] = False
                    final_change[0] = max_change
                    s_all = [float(np.mean([r[j] for r in recent])) for j in range(n)]
                    return s_all

        converged_properly[0] = False
        final_change[0] = max_change
        if recent:
            return [float(np.mean([r[j] for r in recent])) for j in range(n)]
        return s_all

    patched_nash(**params)
    return converged_properly[0], final_change[0]


def main() -> None:
    print("=== Default params ===")
    ok, change = check_convergence_at_params()
    print(f"  Converged: {ok}, final change: {change:.2e}")

    # Sweep k
    print("\n=== Sweep k ===")
    for k in [0.3, 0.5, 1.0, 5.0, 10.0, 33.9, 100.0, 200.0]:
        ok, change = check_convergence_at_params(k=k)
        status = "OK" if ok else "AVERAGED"
        print(f"  k={k:6.1f}: {status}, final change: {change:.2e}")

    # Sweep delta
    print("\n=== Sweep delta ===")
    for delta in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75]:
        ok, change = check_convergence_at_params(delta=delta)
        status = "OK" if ok else "AVERAGED"
        print(f"  delta={delta:.2f}: {status}, final change: {change:.2e}")

    # Sweep rho
    print("\n=== Sweep rho ===")
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ok, change = check_convergence_at_params(rho=rho)
        status = "OK" if ok else "AVERAGED"
        print(f"  rho={rho:.2f}: {status}, final change: {change:.2e}")

    # Sweep w
    print("\n=== Sweep w ===")
    for w in [1.0, 2.0, 3.0, 5.0]:
        ok, change = check_convergence_at_params(w=w)
        status = "OK" if ok else "AVERAGED"
        print(f"  w={w:.1f}: {status}, final change: {change:.2e}")

    # Sweep z
    print("\n=== Sweep z ===")
    for z in [0.5, 0.7, 0.9, 1.0]:
        ok, change = check_convergence_at_params(z=z)
        status = "OK" if ok else "AVERAGED"
        print(f"  z={z:.1f}: {status}, final change: {change:.2e}")


if __name__ == "__main__":
    main()
