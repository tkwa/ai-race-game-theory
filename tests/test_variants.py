"""Smoke tests and sanity checks for variant computations."""

from report import variants


class TestSmoke:
    """Every variant function runs and returns expected keys."""

    def test_safety_elasticity(self) -> None:
        data = variants.safety_elasticity()
        assert set(data.keys()) == {
            "k",
            "s_star_a1",
            "p_sq_a1",
            "s_star_a_default",
            "p_sq_a_default",
        }

    def test_n_actors_sweep(self) -> None:
        data = variants.n_actors_sweep()
        assert set(data.keys()) == {"n", "s_star", "joint_survival"}

    def test_different_resources_2(self) -> None:
        data = variants.different_resources_2()
        assert set(data.keys()) == {"R", "s1", "s2", "p1", "p2", "joint_survival"}

    def test_winner_take_all(self) -> None:
        data = variants.winner_take_all()
        assert set(data.keys()) == {"w", "s_star", "p_sq"}

    def test_correlated_alignment(self) -> None:
        data = variants.correlated_alignment()
        assert set(data.keys()) == {"rho", "s_star", "joint_survival"}


class TestSanity:
    def test_more_actors_lower_survival(self) -> None:
        data = variants.n_actors_sweep()
        assert data["joint_survival"][0] > data["joint_survival"][-1]

    def test_correlation_increases_survival(self) -> None:
        data = variants.correlated_alignment()
        assert data["joint_survival"][-1] > data["joint_survival"][0]
