"""Analyze merging under the equity-share model.

When labs A and B merge with equity shares e_A, e_B:
- Merged entity has resources R_A + R_B, single safety choice, single alignment draw
- Lab i's effective amity toward merged entity = Σ_j e_j * A_ij
  (the merged entity's aligned ASI resources are split by equity shares,
   and each portion valued at original amity)
- Lab i's payoff from merger = full_model_payoffs computed with the
  effective amity matrix, then the merged entity's payoff weighted by equity
"""

import sys

import numpy as np

sys.path.insert(0, "src")

from report import defaults, primitives


def merge_payoffs_equity(
    merging: list[int],
    equity: np.ndarray | None,
    R: np.ndarray,
    A: np.ndarray,
    k: float,
    alpha: float,
    w: float,
    z: float,
    delta: float | None = None,
    rho: float = 0.0,
) -> tuple[list[float], list[float], float, list[float], list[float], float]:
    """Compare pre-merger and post-merger payoffs under equity-share model.

    Returns (pre_payoffs, pre_safety, pre_ehs, post_payoffs, post_safety, post_ehs).
    post_payoffs_for_original_labs has length n (original number of labs).
    For merging labs, the payoff is computed using equity shares and effective amity.
    """
    n = len(R)
    rest = [i for i in range(n) if i not in merging]

    if equity is None:
        # Proportional to resources
        equity = R[merging] / R[merging].sum()

    # --- Pre-merger ---
    pre_safety = primitives.full_model_nash(
        R=R, A=A, k=k, alpha=alpha, w=w, z=z, delta=delta, rho=rho
    )
    pre_payoffs = primitives.full_model_payoffs(pre_safety, R, A, k, alpha, w, z, delta, rho)
    pre_ehs = primitives.expected_human_share(pre_safety, R, k, alpha, w, z, delta, rho)

    # --- Post-merger ---
    n_new = 1 + len(rest)
    R_new = np.zeros(n_new)
    R_new[0] = R[merging].sum()
    for idx, j in enumerate(rest):
        R_new[1 + idx] = R[j]

    # Build effective amity matrix for the merged game
    # The merged entity is player 0 in the new game.
    # For the merged entity's row: its amity toward itself is 1, toward outside labs uses
    # a weighted average of the merging labs' amities (weighted by equity).
    # For outside labs' row: their amity toward the merged entity uses the equity-share model:
    # A_new[k, 0] = Σ_j e_j * A_old[k, j] for j in merging set
    A_new = np.zeros((n_new, n_new))
    A_new[0, 0] = 1.0

    # Merged entity's amity toward outside labs
    for idx_new, j_old in enumerate(rest):
        A_new[0, 1 + idx_new] = sum(
            equity[m_idx] * A[merging[m_idx], j_old] for m_idx in range(len(merging))
        )

    # Outside labs' amity toward merged entity
    for idx_new, i_old in enumerate(rest):
        A_new[1 + idx_new, 0] = sum(
            equity[m_idx] * A[i_old, merging[m_idx]] for m_idx in range(len(merging))
        )

    # Outside labs' amity toward each other (unchanged)
    for i_idx, i_old in enumerate(rest):
        for j_idx, j_old in enumerate(rest):
            A_new[1 + i_idx, 1 + j_idx] = A[i_old, j_old]

    try:
        post_safety_new = primitives.full_model_nash(
            R=R_new, A=A_new, k=k, alpha=alpha, w=w, z=z, delta=delta, rho=rho
        )
    except RuntimeError:
        # Try with different initial guess
        post_safety_new = primitives.full_model_nash(
            R=R_new,
            A=A_new,
            k=k,
            alpha=alpha,
            w=w,
            z=z,
            delta=delta,
            rho=rho,
            initial_guess=[0.1] * n_new,
        )
    post_payoffs_new = primitives.full_model_payoffs(
        post_safety_new, R_new, A_new, k, alpha, w, z, delta, rho
    )
    post_ehs = primitives.expected_human_share(post_safety_new, R_new, k, alpha, w, z, delta, rho)

    # Map back to original labs
    post_payoffs = np.zeros(n)
    post_safety_all = np.zeros(n)
    for m_idx, i in enumerate(merging):
        # Compute payoffs with a custom amity matrix where
        # the merged entity's row has lab i's actual amities.
        A_i = A_new.copy()
        # The merged entity (row 0) gets lab i's amity values
        A_i[0, 0] = sum(equity[j_idx] * A[i, merging[j_idx]] for j_idx in range(len(merging)))
        for k_idx, k_old in enumerate(rest):
            A_i[0, 1 + k_idx] = A[i, k_old]
        payoff_i = primitives.full_model_payoffs(
            post_safety_new, R_new, A_i, k, alpha, w, z, delta, rho
        )
        post_payoffs[i] = payoff_i[0]
        post_safety_all[i] = post_safety_new[0]

    for idx, j in enumerate(rest):
        post_payoffs[j] = post_payoffs_new[1 + idx]
        post_safety_all[j] = post_safety_new[1 + idx]

    return (
        list(pre_payoffs),
        pre_safety,
        pre_ehs,
        list(post_payoffs),
        list(post_safety_all),
        post_ehs,
    )


def print_case(
    title: str,
    lab_names: list[str],
    merging: list[int],
    pre_payoffs: list[float],
    pre_safety: list[float],
    pre_ehs: float,
    post_payoffs: list[float],
    post_safety: list[float],
    post_ehs: float,
    equity: np.ndarray | None = None,
):
    n = len(lab_names)
    merged_name = "+".join(lab_names[i] for i in merging)

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    if equity is not None:
        eq_str = ", ".join(f"{lab_names[merging[i]]}={equity[i]:.3f}" for i in range(len(merging)))
        print(f"  Equity: {eq_str}")

    hdr = f"  {'Lab':<12} {'Pre-pay':>10} {'Post-pay':>10} {'Change':>10}"
    hdr += f" {'Pre-s':>7} {'Post-s':>7}"
    print(f"\n{hdr}")
    sep = f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}"
    sep += f" {'-' * 7} {'-' * 7}"
    print(sep)

    merging_pre_sum = 0
    merging_post_sum = 0
    for i in range(n):
        change = post_payoffs[i] - pre_payoffs[i]
        marker = " *" if i in merging else ""
        row = f"  {lab_names[i] + marker:<12} {pre_payoffs[i]:>10.4f}"
        row += f" {post_payoffs[i]:>10.4f} {change:>+10.4f}"
        row += f" {pre_safety[i]:>7.4f} {post_safety[i]:>7.4f}"
        print(row)
        if i in merging:
            merging_pre_sum += pre_payoffs[i]
            merging_post_sum += post_payoffs[i]

    print(f"\n  Merging coalition ({merged_name}):")
    print(f"    Pre-merger sum:  {merging_pre_sum:.4f}")
    print(f"    Post-merger sum: {merging_post_sum:.4f}")
    print(f"    Change:          {merging_post_sum - merging_pre_sum:+.4f}")
    print(f"\n  EHS: {pre_ehs:.4f} → {post_ehs:.4f} ({post_ehs - pre_ehs:+.4f})")


def run_tullock_only(
    title: str, R: np.ndarray, merging: list[int], w: float, equity: np.ndarray | None = None
):
    """Pure Tullock contest (no alignment), check if merging helps."""
    n = len(R)
    rest = [i for i in range(n) if i not in merging]

    if equity is None:
        equity = R[merging] / R[merging].sum()

    # Pre-merger omega
    pre_omega = R**w / (R**w).sum()

    # Post-merger
    R_merged = R[merging].sum()
    R_post = np.array([R_merged] + [R[j] for j in rest])
    post_omega_game = R_post**w / (R_post**w).sum()

    print(f"\n--- {title} (w={w}) ---")
    print(f"  Resources: {R}")
    print(f"  Merging: indices {merging}, equity: {equity}")

    all_benefit = True
    for m_idx, i in enumerate(merging):
        pre = pre_omega[i]
        post = equity[m_idx] * post_omega_game[0]
        benefit = post > pre
        print(f"  Lab {i}: Ω pre={pre:.4f}, Ω post={post:.4f} ({'GAINS' if benefit else 'LOSES'})")
        if not benefit:
            all_benefit = False

    # Total for merging coalition
    pre_total = sum(pre_omega[i] for i in merging)
    post_total = post_omega_game[0]
    print(
        f"  Coalition total: Ω pre={pre_total:.4f}, Ω post={post_total:.4f} "
        f"({'GAINS' if post_total > pre_total else 'LOSES'})"
    )
    print(f"  All members benefit individually: {all_benefit}")
    return all_benefit


if __name__ == "__main__":
    print("=" * 70)
    print("  PART 1: PURE TULLOCK CONTEST (no alignment/safety)")
    print("=" * 70)

    # Case 1: 3 equal players, merge 2
    run_tullock_only("3 equal players, merge 2", np.array([1 / 3, 1 / 3, 1 / 3]), [0, 1], w=2.0)

    # Case 2: 3 unequal players, merge the two smaller
    run_tullock_only("3 unequal, merge 2 smaller", np.array([0.5, 0.3, 0.2]), [1, 2], w=2.0)

    # Case 3: 3 unequal, merge the large with a small
    run_tullock_only("3 unequal, merge large+small", np.array([0.5, 0.3, 0.2]), [0, 2], w=2.0)

    # Case 4: 2 players only (no outside competition)
    run_tullock_only("2 players, unequal", np.array([0.6, 0.4]), [0, 1], w=2.0)

    # Case 5: 2 equal players
    run_tullock_only("2 equal players", np.array([0.5, 0.5]), [0, 1], w=2.0)

    # Case 6: 5 players (realistic), merge top 3
    run_tullock_only("5 players, merge top 3", defaults.R, [0, 1, 2], w=2.0)

    # Case 7: High w
    run_tullock_only("3 equal, w=5", np.array([1 / 3, 1 / 3, 1 / 3]), [0, 1], w=5.0)

    # Case 8: w just above 1
    run_tullock_only("3 equal, w=1.1", np.array([1 / 3, 1 / 3, 1 / 3]), [0, 1], w=1.1)

    print("\n\n" + "=" * 70)
    print("  PART 2: FULL MODEL WITH ALIGNMENT & AMITY")
    print("=" * 70)

    # Case A: Default parameters, merge OAI + Ant
    lab_names = list(defaults.LAB_NAMES)
    merging = [0, 1]

    result = merge_payoffs_equity(
        merging=merging,
        equity=None,
        R=defaults.R,
        A=defaults.A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    print_case("Default params, merge OAI+Ant (proportional equity)", lab_names, merging, *result)

    # Case B: Merge top 3
    merging = [0, 1, 2]
    result = merge_payoffs_equity(
        merging=merging,
        equity=None,
        R=defaults.R,
        A=defaults.A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    print_case("Default params, merge top 3 (proportional equity)", lab_names, merging, *result)

    # Case C: Selfish labs (A = I), merge top 3
    A_selfish = np.eye(5)
    merging = [0, 1, 2]
    result = merge_payoffs_equity(
        merging=merging,
        equity=None,
        R=defaults.R,
        A=A_selfish,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=defaults.W,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    print_case("Selfish labs (A=I), merge top 3", lab_names, merging, *result)

    # Case D: High amity labs, merge top 3
    A_high = np.full((5, 5), 0.8)
    np.fill_diagonal(A_high, 1.0)
    merging = [0, 1, 2]
    try:
        result = merge_payoffs_equity(
            merging=merging,
            equity=None,
            R=defaults.R,
            A=A_high,
            k=defaults.K,
            alpha=defaults.ALPHA,
            w=defaults.W,
            z=defaults.Z,
            delta=defaults.DELTA,
            rho=defaults.RHO,
        )
        print_case("High amity (A=0.8), merge top 3", lab_names, merging, *result)
    except RuntimeError as e:
        print(f"\n  SKIPPED: High amity case did not converge: {e}")

    for case_title, case_merging, case_w, case_delta, case_rho in [
        ("Default amity, w=5, merge top 3", [0, 1, 2], 5.0, defaults.DELTA, defaults.RHO),
        ("Default amity, w=1, merge top 3", [0, 1, 2], 1.0, defaults.DELTA, defaults.RHO),
        ("No corr/public good, merge OAI+Ant", [0, 1], defaults.W, 0.0, 0.0),
    ]:
        try:
            result = merge_payoffs_equity(
                merging=case_merging,
                equity=None,
                R=defaults.R,
                A=defaults.A,
                k=defaults.K,
                alpha=defaults.ALPHA,
                w=case_w,
                z=defaults.Z,
                delta=case_delta,
                rho=case_rho,
            )
            print_case(case_title, lab_names, case_merging, *result)
        except RuntimeError as e:
            print(f"\n  SKIPPED: {case_title} did not converge: {e}")

    # Case H: 3-player simplified (like the examples in current merging.md)
    print("\n\n" + "=" * 70)
    print("  PART 3: SIMPLIFIED 3-PLAYER CASES (compare with current merging.md)")
    print("=" * 70)

    R3 = np.array([0.5, 0.3, 0.2])
    names3 = ["A", "B", "C"]

    cases_3p = [
        ("3-player selfish, merge A+B", np.eye(3)),
        (
            "3-player, B altruistic, merge A+B",
            np.array([[1.0, 0.0, 0.0], [0.8, 1.0, 0.8], [0.0, 0.0, 1.0]]),
        ),
        (
            "3-player, all altruistic (0.8), merge A+B",
            np.array([[1.0, 0.8, 0.8], [0.8, 1.0, 0.8], [0.8, 0.8, 1.0]]),
        ),
    ]
    for title, A3 in cases_3p:
        try:
            result = merge_payoffs_equity(
                merging=[0, 1],
                equity=None,
                R=R3,
                A=A3,
                k=defaults.K,
                alpha=defaults.ALPHA,
                w=2.0,
                z=0.9,
                delta=0.0,
                rho=0.0,
            )
            print_case(title, names3, [0, 1], *result)
        except RuntimeError as e:
            print(f"\n  SKIPPED: {title} did not converge: {e}")

    print("\n\n" + "=" * 70)
    print("  PART 4: SWEEP OVER w")
    print("=" * 70)

    for sweep_title, sweep_A in [("default amity", defaults.A), ("selfish labs (A=I)", np.eye(5))]:
        print(f"\n  Merge top 3, {sweep_title}, varying w:")
        print(
            f"  {'w':>5}  {'OAI Δ':>10}  {'Ant Δ':>10}  {'GDM Δ':>10}  {'Sum Δ':>10}  {'EHS Δ':>10}"
        )
        for w_val in [1.0, 1.5, 2.0, 3.0, 5.0]:
            try:
                result = merge_payoffs_equity(
                    merging=[0, 1, 2],
                    equity=None,
                    R=defaults.R,
                    A=sweep_A,
                    k=defaults.K,
                    alpha=defaults.ALPHA,
                    w=w_val,
                    z=defaults.Z,
                    delta=defaults.DELTA,
                    rho=defaults.RHO,
                )
                pre_pay, _, pre_ehs, post_pay, _, post_ehs = result
                deltas = [post_pay[i] - pre_pay[i] for i in range(3)]
                d = deltas
                row = f"  {w_val:>5.1f}  {d[0]:>+9.4f}"
                row += f"  {d[1]:>+9.4f}  {d[2]:>+9.4f}"
                row += f"  {sum(d):>+9.4f}"
                row += f"  {post_ehs - pre_ehs:>+9.4f}"
                print(row)
            except RuntimeError:
                print(f"  {w_val:>5.1f}  (did not converge)")
