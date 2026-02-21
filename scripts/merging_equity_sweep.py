"""Sweep equity splits for every lab pair to find zone of mutual benefit."""

import sys

import numpy as np

sys.path.insert(0, "src")

from report import defaults, primitives


def compute_merger_payoffs(
    i: int,
    j: int,
    equity_i: float,
    R: np.ndarray,
    A: np.ndarray,
    k: float,
    alpha: float,
    w: float,
    z: float,
    delta: float | None,
    rho: float,
) -> tuple[float, float, float]:
    """Payoffs for labs i and j after merging with equity_i for lab i."""
    n = len(R)
    merging = [i, j]
    rest = [x for x in range(n) if x not in merging]
    equity = np.array([equity_i, 1.0 - equity_i])

    n_new = 1 + len(rest)
    R_new = np.zeros(n_new)
    R_new[0] = R[i] + R[j]
    for idx, r in enumerate(rest):
        R_new[1 + idx] = R[r]

    # Merged entity's amity (equity-weighted average)
    A_new = np.zeros((n_new, n_new))
    A_new[0, 0] = 1.0
    for idx_new, k_old in enumerate(rest):
        A_new[0, 1 + idx_new] = equity[0] * A[i, k_old] + equity[1] * A[j, k_old]
        A_new[1 + idx_new, 0] = equity[0] * A[k_old, i] + equity[1] * A[k_old, j]
    for i_idx, i_old in enumerate(rest):
        for j_idx, j_old in enumerate(rest):
            A_new[1 + i_idx, 1 + j_idx] = A[i_old, j_old]

    s_star = primitives.full_model_nash(
        R=R_new,
        A=A_new,
        k=k,
        alpha=alpha,
        w=w,
        z=z,
        delta=delta,
        rho=rho,
    )

    # Lab i's payoff: use lab i's actual amity values
    A_for_i = A_new.copy()
    A_for_i[0, 0] = equity[0] * A[i, i] + equity[1] * A[i, j]
    for idx_new, k_old in enumerate(rest):
        A_for_i[0, 1 + idx_new] = A[i, k_old]
    pay_i = primitives.full_model_payoffs(s_star, R_new, A_for_i, k, alpha, w, z, delta, rho)[0]

    # Lab j's payoff
    A_for_j = A_new.copy()
    A_for_j[0, 0] = equity[0] * A[j, i] + equity[1] * A[j, j]
    for idx_new, k_old in enumerate(rest):
        A_for_j[0, 1 + idx_new] = A[j, k_old]
    pay_j = primitives.full_model_payoffs(s_star, R_new, A_for_j, k, alpha, w, z, delta, rho)[0]

    ehs = primitives.expected_human_share(s_star, R_new, k, alpha, w, z, delta, rho)
    return pay_i, pay_j, ehs


def sweep_pair(
    i: int,
    j: int,
    R: np.ndarray,
    A: np.ndarray,
    k: float,
    alpha: float,
    w: float,
    z: float,
    delta: float | None,
    rho: float,
    n_steps: int = 19,
) -> dict:
    """Sweep equity for pair (i, j), find zone of mutual benefit."""
    # Pre-merger payoffs
    pre_s = primitives.full_model_nash(R=R, A=A, k=k, alpha=alpha, w=w, z=z, delta=delta, rho=rho)
    pre_pay = primitives.full_model_payoffs(pre_s, R, A, k, alpha, w, z, delta, rho)
    pre_ehs = primitives.expected_human_share(pre_s, R, k, alpha, w, z, delta, rho)

    equities = np.linspace(0.05, 0.95, n_steps)
    results = []
    for e in equities:
        try:
            pi, pj, ehs = compute_merger_payoffs(i, j, e, R, A, k, alpha, w, z, delta, rho)
            results.append((e, pi, pj, ehs))
        except RuntimeError:
            results.append((e, None, None, None))

    # Find zone of mutual benefit
    zone = [
        (e, pi, pj)
        for e, pi, pj, _ in results
        if pi is not None and pi > pre_pay[i] and pj > pre_pay[j]
    ]

    # Find max coalition surplus
    max_surplus = -float("inf")
    best_e = None
    for e, pi, pj, _ in results:
        if pi is not None:
            surplus = (pi - pre_pay[i]) + (pj - pre_pay[j])
            if surplus > max_surplus:
                max_surplus = surplus
                best_e = e

    return {
        "i": i,
        "j": j,
        "pre_i": float(pre_pay[i]),
        "pre_j": float(pre_pay[j]),
        "pre_ehs": pre_ehs,
        "zone": zone,
        "max_surplus": max_surplus,
        "best_e": best_e,
        "results": results,
    }


def print_pair_result(r: dict, lab_names: list[str]) -> None:
    """Print results for one pair."""
    i, j = r["i"], r["j"]
    ni, nj = lab_names[i], lab_names[j]
    print(f"\n  {ni} + {nj}:")
    print(f"    Pre-merger: {ni}={r['pre_i']:.4f}, {nj}={r['pre_j']:.4f}")
    print(f"    Max coalition surplus: {r['max_surplus']:+.4f} (at e_{ni}={r['best_e']:.2f})")

    if r["zone"]:
        e_min = min(z[0] for z in r["zone"])
        e_max = max(z[0] for z in r["zone"])
        print(f"    Zone of mutual benefit: e_{ni} in [{e_min:.2f}, {e_max:.2f}]")
        # Show best mutual point
        best_mutual = max(r["zone"], key=lambda z: min(z[1] - r["pre_i"], z[2] - r["pre_j"]))
        print(
            f"    Best mutual: e={best_mutual[0]:.2f}, "
            f"Δ{ni}={best_mutual[1] - r['pre_i']:+.4f}, "
            f"Δ{nj}={best_mutual[2] - r['pre_j']:+.4f}"
        )
    else:
        print("    NO zone of mutual benefit found!")
        # Show closest point
        closest = None
        closest_gap = float("inf")
        for e, pi, pj, _ in r["results"]:
            if pi is not None:
                gap = max(r["pre_i"] - pi, r["pre_j"] - pj)
                if gap < closest_gap:
                    closest_gap = gap
                    closest = (e, pi, pj)
        if closest:
            print(
                f"    Closest: e={closest[0]:.2f}, "
                f"Δi={closest[1] - r['pre_i']:+.4f}, "
                f"Δj={closest[2] - r['pre_j']:+.4f}"
            )


if __name__ == "__main__":
    lab_names = list(defaults.LAB_NAMES)
    n = len(lab_names)

    print("=" * 60)
    print("  EQUITY SWEEP: DEFAULT PARAMETERS")
    print("=" * 60)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                r = sweep_pair(
                    i,
                    j,
                    R=defaults.R,
                    A=defaults.A,
                    k=defaults.K,
                    alpha=defaults.ALPHA,
                    w=defaults.W,
                    z=defaults.Z,
                    delta=defaults.DELTA,
                    rho=defaults.RHO,
                )
                print_pair_result(r, lab_names)
            except RuntimeError as e:
                print(f"\n  {lab_names[i]}+{lab_names[j]}: FAILED ({e})")

    # Try adversarial cases: low k (safety is hard)
    print("\n\n" + "=" * 60)
    print("  ADVERSARIAL: LOW k (safety very hard)")
    print("=" * 60)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                r = sweep_pair(
                    i,
                    j,
                    R=defaults.R,
                    A=defaults.A,
                    k=5.0,
                    alpha=defaults.ALPHA,
                    w=defaults.W,
                    z=defaults.Z,
                    delta=defaults.DELTA,
                    rho=defaults.RHO,
                )
                print_pair_result(r, lab_names)
            except RuntimeError as e:
                print(f"\n  {lab_names[i]}+{lab_names[j]}: FAILED ({e})")

    # Adversarial: no public good, no correlation
    print("\n\n" + "=" * 60)
    print("  ADVERSARIAL: δ=0, ρ=0 (no spillover/correlation)")
    print("=" * 60)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                r = sweep_pair(
                    i,
                    j,
                    R=defaults.R,
                    A=defaults.A,
                    k=defaults.K,
                    alpha=defaults.ALPHA,
                    w=defaults.W,
                    z=defaults.Z,
                    delta=0.0,
                    rho=0.0,
                )
                print_pair_result(r, lab_names)
            except RuntimeError as e:
                print(f"\n  {lab_names[i]}+{lab_names[j]}: FAILED ({e})")

    # Adversarial: z=0 (strategy-stealing holds)
    print("\n\n" + "=" * 60)
    print("  ADVERSARIAL: z=0 (no misaligned AI advantage)")
    print("=" * 60)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                r = sweep_pair(
                    i,
                    j,
                    R=defaults.R,
                    A=defaults.A,
                    k=defaults.K,
                    alpha=defaults.ALPHA,
                    w=defaults.W,
                    z=0.0,
                    delta=defaults.DELTA,
                    rho=defaults.RHO,
                )
                print_pair_result(r, lab_names)
            except RuntimeError as e:
                print(f"\n  {lab_names[i]}+{lab_names[j]}: FAILED ({e})")

    # Adversarial: w=1 (no winner-take-all)
    print("\n\n" + "=" * 60)
    print("  ADVERSARIAL: w=1 (proportional contest)")
    print("=" * 60)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                r = sweep_pair(
                    i,
                    j,
                    R=defaults.R,
                    A=defaults.A,
                    k=defaults.K,
                    alpha=defaults.ALPHA,
                    w=1.0,
                    z=defaults.Z,
                    delta=defaults.DELTA,
                    rho=defaults.RHO,
                )
                print_pair_result(r, lab_names)
            except RuntimeError as e:
                print(f"\n  {lab_names[i]}+{lab_names[j]}: FAILED ({e})")

    # Adversarial: extreme amity asymmetry
    print("\n\n" + "=" * 60)
    print("  ADVERSARIAL: OAI amity=0 to all, Ant amity=0.9")
    print("=" * 60)
    A_extreme = defaults.A.copy()
    A_extreme[0, :] = [1.0, 0.0, 0.0, 0.0, 0.0]  # OAI selfish
    A_extreme[1, :] = [0.9, 1.0, 0.9, 0.9, 0.9]  # Ant very altruistic
    try:
        r = sweep_pair(
            0,
            1,
            R=defaults.R,
            A=A_extreme,
            k=defaults.K,
            alpha=defaults.ALPHA,
            w=defaults.W,
            z=defaults.Z,
            delta=defaults.DELTA,
            rho=defaults.RHO,
        )
        print_pair_result(r, lab_names)
    except RuntimeError as e:
        print(f"\n  FAILED ({e})")
