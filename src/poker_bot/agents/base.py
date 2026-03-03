"""Abstract base class for all poker agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from poker_bot.game.action import Action
from poker_bot.game.engine import PokerEngine
from poker_bot.game.state import GameState


class Agent(ABC):
    """Abstract poker agent."""

    @abstractmethod
    def act(self, state: GameState, engine: PokerEngine) -> Action:
        """Choose an action given the current game state."""
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset any per-hand state. Override in subclasses if needed."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
