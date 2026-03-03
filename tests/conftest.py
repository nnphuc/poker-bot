"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from poker_bot.game.card import Card, Rank, Suit
from poker_bot.game.engine import PokerEngine
from poker_bot.game.state import GameState
from poker_bot.utils.config import TrainingConfig


@pytest.fixture
def engine() -> PokerEngine:
    return PokerEngine(small_blind=50, big_blind=100)


@pytest.fixture
def default_stacks() -> list[int]:
    return [10_000, 10_000]


@pytest.fixture
def new_game(engine: PokerEngine, default_stacks: list[int]) -> GameState:
    return engine.new_game(default_stacks, seed=42)


@pytest.fixture
def ace_spades() -> Card:
    return Card(Rank.ACE, Suit.SPADES)


@pytest.fixture
def king_hearts() -> Card:
    return Card(Rank.KING, Suit.HEARTS)


@pytest.fixture
def training_config() -> TrainingConfig:
    return TrainingConfig(
        total_iterations=100,
        checkpoint_every=50,
        checkpoint_dir="/tmp/poker_test_checkpoints",
        seed=42,
    )
