"""Self-play evaluation during training."""

from __future__ import annotations

from poker_bot.agents.base import Agent
from poker_bot.evaluation.evaluator import HeadToHeadEvaluator
from poker_bot.game.engine import PokerEngine
from poker_bot.utils.config import TrainingConfig


def self_play_eval(
    agent: Agent,
    opponent: Agent,
    config: TrainingConfig,
    n_hands: int = 1000,
) -> dict[str, float]:
    """Run head-to-head evaluation and return metrics."""
    engine = PokerEngine(config.small_blind, config.big_blind)
    evaluator = HeadToHeadEvaluator(engine, config.starting_stacks[0])
    return evaluator.evaluate(agent, opponent, n_hands=n_hands)
