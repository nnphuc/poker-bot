"""Random agent - baseline for evaluation."""

from __future__ import annotations

import random

from poker_bot.agents.base import Agent
from poker_bot.game.action import Action
from poker_bot.game.engine import PokerEngine
from poker_bot.game.state import GameState


class RandomAgent(Agent):
    """Chooses uniformly at random from legal actions."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, state: GameState, engine: PokerEngine) -> Action:
        action_space = engine.get_action_space(state)
        legal = action_space.legal_actions()

        # Also add some random raise sizes
        if action_space.min_raise <= action_space.max_raise:
            raise_amount = self._rng.randint(action_space.min_raise, action_space.max_raise)
            legal.append(Action.raise_(raise_amount))

        return self._rng.choice(legal)
