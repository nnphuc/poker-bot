"""Always-call agent - simple baseline."""

from __future__ import annotations

from poker_bot.agents.base import Agent
from poker_bot.game.action import Action
from poker_bot.game.engine import PokerEngine
from poker_bot.game.state import GameState


class CallAgent(Agent):
    """Always checks or calls, never raises."""

    def act(self, state: GameState, engine: PokerEngine) -> Action:
        action_space = engine.get_action_space(state)
        if action_space.can_check:
            return Action.check()
        return Action.call(action_space.call_amount)
