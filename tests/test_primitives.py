"""Tests for report primitives against baseline model and calibration targets."""

import numpy as np
import pytest

from ai_race_game_theory import model
from report import primitives


class TestAlignmentProb:
    def test_matches_model_at_alpha_1(self) -> None:
        """alignment_prob with α=1 matches model.alignment_probability."""
        for s in [0.01, 0.1, 0.3, 0.5, 0.7, 0.99]:
            for k in [0.1, 1.0, 10.0, 100.0]:
                assert primitives.alignment_prob(s, k, 1.0) == pytest.approx(
                    model.alignment_probability(s, k), abs=1e-10
                )

    def test_calibration_1_percent(self) -> None:
        """1% spending -> ~80% aligned."""
        p = primitives.alignment_prob(0.01, primitives.DEFAULT_K, primitives.DEFAULT_ALPHA)
        assert p == pytest.approx(0.80, abs=0.02)

    def test_calibration_50_percent(self) -> None:
        """50% spending -> ~98% aligned."""
        p = primitives.alignment_prob(0.50, primitives.DEFAULT_K, primitives.DEFAULT_ALPHA)
        assert p == pytest.approx(0.98, abs=0.02)

    def test_boundary_zero(self) -> None:
        assert primitives.alignment_prob(0.0, 10.0, 0.5) == 0.0

    def test_boundary_one(self) -> None:
        assert primitives.alignment_prob(1.0, 10.0, 0.5) == 1.0


class TestSymmetricNash:
    def test_matches_model_2_player(self) -> None:
        """symmetric_nash(2, k, 1.0) matches model.find_nash_equilibrium(k)."""
        for k in [0.5, 2.0, 10.0]:
            s_prim = primitives.symmetric_nash(2, k, 1.0)
            s_model = model.find_nash_equilibrium(k)
            assert s_prim == pytest.approx(s_model, abs=0.01)


class TestDeltaConvention:
    """δ=0 is private (no spillover), δ=1 is fully public."""

    def test_delta_zero_equals_no_spillover(self) -> None:
        """δ=0 gives same effective safety as no spillover (delta=None)."""
        s_all = [0.1, 0.3, 0.5]
        eff_none = primitives._effective_safety(s_all, None)
        eff_zero = primitives._effective_safety(s_all, 0.0)
        for a, b in zip(eff_none, eff_zero):
            assert a == pytest.approx(b, abs=1e-10)

    def test_delta_one_equalizes_safety(self) -> None:
        """δ=1 (fully public) makes all effective safety equal to Σ s_j."""
        s_all = [0.1, 0.3, 0.5]
        eff = primitives._effective_safety(s_all, 1.0)
        s_sum = sum(s_all)
        for e in eff:
            assert e == pytest.approx(s_sum, abs=1e-10)

    def test_higher_delta_reduces_spread(self) -> None:
        """More public good (higher δ) narrows the gap between effective safety levels."""
        s_all = [0.1, 0.5]
        spread_low = _effective_spread(s_all, 0.1)
        spread_high = _effective_spread(s_all, 0.9)
        assert spread_high < spread_low

    def test_public_good_increases_survival(self) -> None:
        """Making safety more public (δ=0.5) should increase joint survival vs private (δ=0)."""
        k, alpha, n = primitives.DEFAULT_K, primitives.DEFAULT_ALPHA, 3
        s_private = primitives.symmetric_nash(n, k, alpha, public_good_delta=None)
        s_public = primitives.symmetric_nash(n, k, alpha, public_good_delta=0.5)

        p_priv = primitives.alignment_prob(s_private, k, alpha)
        # With s_sum formula, eff_s = (n*s)^δ * s^(1-δ) = n^δ * s, so δ helps
        eff_pub = (n * s_public) ** 0.5 * s_public**0.5
        p_pub = primitives.alignment_prob(eff_pub, k, alpha)

        assert p_pub**n > p_priv**n

    def test_full_model_delta_one_equalizes(self) -> None:
        """In the full model, δ=1 should give equal effective safety for all labs."""
        s_all = [0.1, 0.3, 0.5]
        eff = primitives._effective_safety(s_all, 1.0)
        s_sum = sum(s_all)
        # With δ=1 (fully public), all effective safety = Σ s_j
        for e in eff:
            assert e == pytest.approx(s_sum, abs=1e-10)

    def test_delta_interpolates_monotonically(self) -> None:
        """Effective safety spread decreases monotonically as δ goes from 0 to 1."""
        s_all = [0.1, 0.5]
        deltas = [0.0, 0.25, 0.5, 0.75, 1.0]
        spreads = [_effective_spread(s_all, d) for d in deltas]
        for i in range(len(spreads) - 1):
            assert spreads[i] >= spreads[i + 1] - 1e-10

    def test_effective_safety_monotone_in_delta(self) -> None:
        """Every lab's effective safety increases with δ (for fixed strategies)."""
        s_all = [0.025, 0.055, 0.066, 0.082, 0.250]
        deltas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        prev_eff = primitives._effective_safety(s_all, 0.0)
        for d in deltas[1:]:
            eff = primitives._effective_safety(s_all, d)
            for j in range(len(s_all)):
                assert eff[j] >= prev_eff[j] - 1e-10, (
                    f"Lab {j} eff safety decreased: {prev_eff[j]:.6f} -> {eff[j]:.6f} at δ={d}"
                )
            prev_eff = eff

    def test_alignment_prob_monotone_in_delta_at_nash(self) -> None:
        """Every lab's P(aligned) at Nash equilibrium increases with δ."""
        from report import defaults

        deltas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        prev_probs = None
        for d in deltas:
            s_star = primitives.full_model_nash(
                R=defaults.R,
                A=defaults.A,
                k=defaults.K,
                alpha=defaults.ALPHA,
                w=defaults.W,
                z=defaults.Z,
                delta=d if d > 0 else None,
                rho=defaults.RHO,
            )
            eff_s = primitives._effective_safety(s_star, d if d > 0 else None)
            probs = [primitives.alignment_prob(e, defaults.K, defaults.ALPHA) for e in eff_s]
            if prev_probs is not None:
                for j in range(len(probs)):
                    assert probs[j] >= prev_probs[j] - 1e-6, (
                        f"Lab {j} P(aligned) decreased: {prev_probs[j]:.6f} -> {probs[j]:.6f} "
                        f"at δ={d}"
                    )
            prev_probs = probs


def _effective_spread(s_all: list[float], delta: float) -> float:
    """Max minus min of effective safety."""
    eff = primitives._effective_safety(s_all, delta)
    return max(eff) - min(eff)


class TestOverrideProbs:
    """Tests for override_probs in full_model_nash and full_model_payoffs."""

    def _simple_params(self) -> dict:
        """Small 3-lab setup for fast tests."""
        return dict(
            R=np.array([0.4, 0.3, 0.3]),
            A=np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]]),
            k=33.9,
            alpha=0.466,
            w=2.0,
            z=1.0,
            delta=0.5,
            rho=0.0,
        )

    def test_known_safe_lab_changes_others_safety(self) -> None:
        """override_probs during Nash should change other labs' equilibrium strategies.

        Without override_probs, fixing lab 0 at s=0 makes it look dangerous
        (alignment_prob=0), so others give up on safety (misaligned lab 0 dominates
        the omega share regardless). With override_probs={0: 1.0}, others know lab 0
        is safe, making alignment more valuable — so they invest MORE in safety.
        """
        params = self._simple_params()

        # Old (buggy) approach: fixed s=0 but no override during Nash
        s_without = primitives.full_model_nash(**params, fixed={0: 0.0})

        # Correct approach: fixed s=0 AND override alignment prob during Nash
        s_with = primitives.full_model_nash(**params, fixed={0: 0.0}, override_probs={0: 1.0})

        # Others invest MORE in safety when they know lab 0 is guaranteed aligned
        avg_safety_without = np.mean([s_without[1], s_without[2]])
        avg_safety_with = np.mean([s_with[1], s_with[2]])
        assert avg_safety_with > avg_safety_without

    def test_override_probs_affects_payoffs(self) -> None:
        """Overriding a lab's alignment prob to 1.0 should change payoffs vs s=0 default."""
        params = self._simple_params()
        s_all = [0.0, 0.3, 0.3]

        payoffs_no_override = primitives.full_model_payoffs(s_all, **params)
        payoffs_with_override = primitives.full_model_payoffs(
            s_all, **params, override_probs={0: 1.0}
        )

        # With lab 0 guaranteed aligned, the other labs' payoffs should increase
        # (they benefit from lab 0's alignment via amity > 0)
        assert payoffs_with_override[1] > payoffs_no_override[1]
        assert payoffs_with_override[2] > payoffs_no_override[2]


class TestCopula:
    def test_independent_at_rho_zero(self) -> None:
        """At ρ=0, copula equals product of marginals."""
        probs = [0.9, 0.8, 0.7]
        joint = primitives.joint_survival_copula(probs, 0.0)
        assert joint == pytest.approx(0.9 * 0.8 * 0.7, abs=1e-6)

    def test_higher_correlation_higher_survival(self) -> None:
        """Positive correlation increases joint survival."""
        probs = [0.9, 0.9]
        j_low = primitives.joint_survival_copula(probs, 0.1)
        j_high = primitives.joint_survival_copula(probs, 0.8)
        assert j_high > j_low
