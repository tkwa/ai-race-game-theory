"""Smoke tests and sanity checks for toy model computations."""

from report import toy_model


class TestSmoke:
    def test_asymmetric_resources(self) -> None:
        data = toy_model.toy_asymmetric_resources()
        assert set(data.keys()) == {"R", "s1", "s2", "p1", "p2", "joint_survival"}


class TestSanity:
    def test_larger_player_spends_more_on_safety(self) -> None:
        """The larger player has more at stake and invests more in safety."""
        data = toy_model.toy_asymmetric_resources()
        mid = len(data["s1"]) // 2
        assert data["s1"][mid] > data["s2"][mid]
