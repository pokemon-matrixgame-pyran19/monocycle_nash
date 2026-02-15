import pytest

from monocycle_nash.character.domain import Character, MatchupVector
from monocycle_nash.strategy.domain import MonocyclePureStrategy, PureStrategySet


class TestPureStrategyDomain:
    def test_from_character_creates_monocycle_strategy(self):
        character = Character(1.0, MatchupVector(1.0, 0.0), label="Pika")
        strategies = PureStrategySet.from_characters([character])
        strategy = strategies[0]

        assert isinstance(strategy, MonocyclePureStrategy)
        assert strategy.id == "c0"
        assert strategy.label == "Pika"
        assert strategy.power == pytest.approx(1.0)

    def test_label_fallback_to_id_when_character_label_empty(self):
        character = Character(1.0, MatchupVector(1.0, 0.0))
        strategies = PureStrategySet.from_characters([character])

        assert strategies.labels == ["c0"]

    def test_from_labels_sets_strategy_ids(self):
        labels = ["Rock", "Paper", "Scissors"]
        strategies = PureStrategySet.from_labels(labels)

        assert strategies.ids == labels
        assert strategies.labels == labels

    def test_shift_origin_preserves_ids(self):
        characters = [
            Character(1.0, MatchupVector(1.0, 0.0), label="A"),
            Character(0.5, MatchupVector(0.0, 1.0), label="B"),
        ]
        original = PureStrategySet.from_characters(characters, ids=["A", "B"])

        shifted = original.shift_origin(MatchupVector(0.25, -0.5))

        assert shifted.ids == ["A", "B"]
        assert shifted[0].vector != original[0].vector

    def test_duplicate_ids_raise_error(self):
        characters = [
            Character(1.0, MatchupVector(1.0, 0.0), label="A"),
            Character(0.5, MatchupVector(0.0, 1.0), label="B"),
        ]

        with pytest.raises(ValueError):
            PureStrategySet.from_characters(characters, ids=["dup", "dup"])
