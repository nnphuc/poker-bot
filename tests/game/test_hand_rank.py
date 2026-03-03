"""Tests for hand evaluator."""

from __future__ import annotations

from poker_bot.game.card import Card
from poker_bot.game.hand_rank import HandEvaluator


def _cards(*strs: str) -> list[Card]:
    return [Card.from_str(s) for s in strs]


def test_royal_flush_beats_straight_flush() -> None:
    evaluator = HandEvaluator()
    royal = evaluator.evaluate(_cards("As", "Ks"), _cards("Qs", "Js", "Ts", "2h", "3d"))
    sf = evaluator.evaluate(_cards("9s", "8s"), _cards("7s", "6s", "5s", "2h", "3d"))
    assert royal < sf  # lower = better


def test_four_of_a_kind_beats_full_house() -> None:
    evaluator = HandEvaluator()
    quads = evaluator.evaluate(_cards("Ah", "Ad"), _cards("As", "Ac", "Kh", "2d", "3c"))
    fh = evaluator.evaluate(_cards("Ah", "Ad"), _cards("As", "Kh", "Kd", "2d", "3c"))
    assert quads < fh


def test_rank_class() -> None:
    evaluator = HandEvaluator()
    score = evaluator.evaluate(_cards("Ah", "Kh"), _cards("Qh", "Jh", "Th", "2d", "3c"))
    assert evaluator.rank_class(score) == "Royal Flush"


def test_high_card() -> None:
    evaluator = HandEvaluator()
    score = evaluator.evaluate(_cards("2h", "7d"), _cards("5c", "9s", "Jh", "Ks", "Ac"))
    assert evaluator.rank_class(score) == "High Card"
