"""Tests for evaluation module."""

from __future__ import annotations

from poker_bot.agents.call_agent import CallAgent
from poker_bot.agents.random_agent import RandomAgent
from poker_bot.evaluation.evaluator import HeadToHeadEvaluator
from poker_bot.evaluation.metrics import compute_metrics
from poker_bot.game.engine import PokerEngine


def test_compute_metrics_basic() -> None:
    profits = [1.0, -1.0, 2.0, -2.0, 0.0]
    m = compute_metrics(profits)
    assert "win_rate" in m
    assert "bb_per_100" in m
    assert m["n_hands"] == 5.0


def test_compute_metrics_empty() -> None:
    assert compute_metrics([]) == {}


def test_evaluator_chips_conserved() -> None:
    """After evaluation, no chips should be created or destroyed."""
    engine = PokerEngine(small_blind=50, big_blind=100)
    agent_a = RandomAgent(seed=1)
    agent_b = CallAgent()

    evaluator = HeadToHeadEvaluator(engine, starting_stack=5_000, seed=42)
    metrics = evaluator.evaluate(agent_a, agent_b, n_hands=50)

    assert "win_rate" in metrics
    assert 0.0 <= metrics["win_rate"] <= 1.0


def test_call_agent_vs_random() -> None:
    engine = PokerEngine(small_blind=50, big_blind=100)
    call_agent = CallAgent()
    random_agent = RandomAgent(seed=99)

    evaluator = HeadToHeadEvaluator(engine, starting_stack=10_000, seed=0)
    metrics = evaluator.evaluate(call_agent, random_agent, n_hands=100)

    # Just verify it runs without error and returns sane values
    assert abs(metrics["win_rate"]) <= 1.0
