"""Tests for MCCFR agent."""

from __future__ import annotations

from pathlib import Path

from poker_bot.agents.cfr.infoset import build_infoset_key
from poker_bot.agents.cfr.mccfr import MCCFR, MCCFRAgent, get_abstract_actions
from poker_bot.agents.cfr.strategy import Strategy
from poker_bot.game.engine import PokerEngine
from poker_bot.game.state import GameState


def test_abstract_actions_not_empty(engine: PokerEngine, new_game: GameState) -> None:
    actions = get_abstract_actions(new_game, engine)
    assert len(actions) >= 2  # at minimum: fold + call/check


def test_infoset_key_deterministic(new_game: GameState) -> None:
    key1 = build_infoset_key(new_game, 0)
    key2 = build_infoset_key(new_game, 0)
    assert key1 == key2


def test_mccfr_trains(engine: PokerEngine) -> None:
    mccfr = MCCFR(engine, stacks=[10_000, 10_000], seed=42)
    mccfr.train(n_iterations=10)
    assert mccfr.n_infosets > 0


def test_mccfr_strategy_sums_to_one(engine: PokerEngine) -> None:
    mccfr = MCCFR(engine, stacks=[10_000, 10_000], seed=42)
    mccfr.train(n_iterations=50)
    strategy = mccfr.get_strategy()

    for key, dist in strategy._profile.items():
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6, f"Strategy for {key} sums to {total}"


def test_strategy_save_load(tmp_path: Path, engine: PokerEngine) -> None:
    mccfr = MCCFR(engine, stacks=[10_000, 10_000], seed=42)
    mccfr.train(n_iterations=10)
    strategy = mccfr.get_strategy()

    path = tmp_path / "strategy.pkl"
    strategy.save(path)

    loaded = Strategy.load(path)
    assert loaded.n_infosets == strategy.n_infosets


def test_mccfr_agent_acts(engine: PokerEngine, new_game: GameState) -> None:
    mccfr = MCCFR(engine, stacks=[10_000, 10_000], seed=42)
    mccfr.train(n_iterations=20)
    strategy = mccfr.get_strategy()

    agent = MCCFRAgent(strategy, engine)
    action = agent.act(new_game, engine)
    assert action is not None
