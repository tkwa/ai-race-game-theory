"""Tests for report primitives against baseline model and calibration targets."""

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
