"""Tests for the AI race game theory model."""

import pytest

from ai_race_game_theory import model


class TestAlignmentProbability:
    def test_zero_safety(self) -> None:
        assert model.alignment_probability(0.0, 1.0) == 0.0

    def test_all_safety(self) -> None:
        assert model.alignment_probability(1.0, 1.0) == pytest.approx(1.0)

    def test_equal_split_k1(self) -> None:
        assert model.alignment_probability(0.5, 1.0) == pytest.approx(0.5)

    def test_higher_k_increases_probability(self) -> None:
        p_low = model.alignment_probability(0.5, 1.0)
        p_high = model.alignment_probability(0.5, 10.0)
        assert p_high > p_low


class TestPayoff:
    def test_zero_safety_gives_zero(self) -> None:
        assert model.payoff(0.0, 0.5, 1.0) == 0.0

    def test_symmetric_gives_50_times_p_squared(self) -> None:
        s = 0.5
        k = 1.0
        p = model.alignment_probability(s, k)
        expected = p * p * 50.0
        assert model.payoff(s, s, k) == pytest.approx(expected)

    def test_more_capabilities_gets_larger_share(self) -> None:
        k = 5.0
        payoff_low_safety = model.payoff(0.3, 0.5, k)
        payoff_high_safety = model.payoff(0.7, 0.5, k)
        # Lower safety = more capabilities = larger share of pie (but lower P)
        # The payoff comparison depends on the tradeoff
        assert payoff_low_safety != payoff_high_safety


class TestNashEquilibrium:
    def test_low_k_high_safety(self) -> None:
        """At k=0.1, countries invest ~92% in safety."""
        s_star = model.find_nash_equilibrium(0.1)
        assert s_star == pytest.approx(0.92, abs=0.02)

    def test_high_k_lower_safety(self) -> None:
        """At k=10, countries invest ~33% in safety."""
        s_star = model.find_nash_equilibrium(10.0)
        assert s_star == pytest.approx(0.33, abs=0.02)

    def test_safety_decreases_with_k(self) -> None:
        s_low_k = model.find_nash_equilibrium(0.1)
        s_high_k = model.find_nash_equilibrium(10.0)
        assert s_low_k > s_high_k


class TestSurvivalProbability:
    def test_low_k_survival(self) -> None:
        """At k=0.1, survival probability ~29%."""
        p_sq = model.survival_probability(0.1)
        assert p_sq == pytest.approx(0.29, abs=0.03)

    def test_high_k_survival(self) -> None:
        """At k=10, survival probability ~69%."""
        p_sq = model.survival_probability(10.0)
        assert p_sq == pytest.approx(0.69, abs=0.03)

    def test_survival_increases_with_k(self) -> None:
        p_low = model.survival_probability(0.1)
        p_high = model.survival_probability(10.0)
        assert p_high > p_low

    def test_very_high_k_over_90_percent(self) -> None:
        """Need k>100 for survival probability over 90%."""
        # At k=100, not yet 90%; at higher k, crosses 90%
        assert model.survival_probability(100.0) < 0.90
        assert model.survival_probability(200.0) > 0.90
