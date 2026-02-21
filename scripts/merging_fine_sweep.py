"""Fine-grained equity sweep for the marginal GDM+China w=1 case."""

import sys

import numpy as np

sys.path.insert(0, "src")

from report import defaults, primitives

sys.path.insert(0, "scripts")

from merging_equity_sweep import compute_merger_payoffs

if __name__ == "__main__":
    # Pre-merger payoffs at w=1
    pre_s = primitives.full_model_nash(
        R=defaults.R,
        A=defaults.A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=1.0,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    pre_pay = primitives.full_model_payoffs(
        pre_s,
        defaults.R,
        defaults.A,
        defaults.K,
        defaults.ALPHA,
        1.0,
        defaults.Z,
        defaults.DELTA,
        defaults.RHO,
    )

    i, j = 2, 4  # GDM, China
    print(f"Pre-merger: GDM={pre_pay[i]:.6f}, China={pre_pay[j]:.6f}")
    print(
        f"\n{'e_GDM':>8} {'GDM pay':>10} {'China pay':>10} "
        f"{'ΔGDM':>10} {'ΔChina':>10} {'both+?':>6}"
    )

    for e in np.arange(0.60, 0.85, 0.01):
        try:
            pi, pj, _ = compute_merger_payoffs(
                i,
                j,
                e,
                defaults.R,
                defaults.A,
                defaults.K,
                defaults.ALPHA,
                1.0,
                defaults.Z,
                defaults.DELTA,
                defaults.RHO,
            )
            di = pi - pre_pay[i]
            dj = pj - pre_pay[j]
            both = "YES" if di > 0 and dj > 0 else "no"
            print(f"{e:>8.2f} {pi:>10.6f} {pj:>10.6f} {di:>+10.6f} {dj:>+10.6f} {both:>6}")
        except RuntimeError:
            print(f"{e:>8.2f}  (failed)")

    # Also try w=1.01 to see if tiny w>1 fixes it
    print("\n\nWith w=1.01:")
    pre_s2 = primitives.full_model_nash(
        R=defaults.R,
        A=defaults.A,
        k=defaults.K,
        alpha=defaults.ALPHA,
        w=1.01,
        z=defaults.Z,
        delta=defaults.DELTA,
        rho=defaults.RHO,
    )
    pre_pay2 = primitives.full_model_payoffs(
        pre_s2,
        defaults.R,
        defaults.A,
        defaults.K,
        defaults.ALPHA,
        1.01,
        defaults.Z,
        defaults.DELTA,
        defaults.RHO,
    )
    print(f"Pre-merger: GDM={pre_pay2[i]:.6f}, China={pre_pay2[j]:.6f}")
    print(f"\n{'e_GDM':>8} {'ΔGDM':>10} {'ΔChina':>10} {'both+?':>6}")

    for e in np.arange(0.60, 0.85, 0.02):
        try:
            pi, pj, _ = compute_merger_payoffs(
                i,
                j,
                e,
                defaults.R,
                defaults.A,
                defaults.K,
                defaults.ALPHA,
                1.01,
                defaults.Z,
                defaults.DELTA,
                defaults.RHO,
            )
            di = pi - pre_pay2[i]
            dj = pj - pre_pay2[j]
            both = "YES" if di > 0 and dj > 0 else "no"
            print(f"{e:>8.2f} {di:>+10.6f} {dj:>+10.6f} {both:>6}")
        except RuntimeError:
            print(f"{e:>8.2f}  (failed)")

    # Also check w=1, Ant+China (another marginal case from above)
    print("\n\nAnt + China, w=1 (marginal ΔChina=+0.0000):")
    i2, j2 = 1, 4
    print(f"Pre-merger: Ant={pre_pay[i2]:.6f}, China={pre_pay[j2]:.6f}")
    print(f"\n{'e_Ant':>8} {'ΔAnt':>10} {'ΔChina':>10} {'both+?':>6}")

    for e in np.arange(0.65, 0.85, 0.01):
        try:
            pi, pj, _ = compute_merger_payoffs(
                i2,
                j2,
                e,
                defaults.R,
                defaults.A,
                defaults.K,
                defaults.ALPHA,
                1.0,
                defaults.Z,
                defaults.DELTA,
                defaults.RHO,
            )
            di = pi - pre_pay[i2]
            dj = pj - pre_pay[j2]
            both = "YES" if di > 0 and dj > 0 else "no"
            print(f"{e:>8.2f} {di:>+10.6f} {dj:>+10.6f} {both:>6}")
        except RuntimeError:
            print(f"{e:>8.2f}  (failed)")
