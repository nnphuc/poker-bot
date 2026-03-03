"""Action types and action space for Texas Hold'em."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass(frozen=True)
class Action:
    """A player action with optional bet amount."""

    action_type: ActionType
    amount: int = 0  # chips to put in (raise amount above current bet)

    @classmethod
    def fold(cls) -> Action:
        return cls(ActionType.FOLD)

    @classmethod
    def check(cls) -> Action:
        return cls(ActionType.CHECK)

    @classmethod
    def call(cls, amount: int) -> Action:
        return cls(ActionType.CALL, amount)

    @classmethod
    def raise_(cls, amount: int) -> Action:
        return cls(ActionType.RAISE, amount)

    @classmethod
    def all_in(cls, amount: int) -> Action:
        return cls(ActionType.ALL_IN, amount)

    def __str__(self) -> str:
        if self.action_type in (ActionType.FOLD, ActionType.CHECK):
            return self.action_type.value
        return f"{self.action_type.value}:{self.amount}"


@dataclass
class ActionSpace:
    """Legal actions available to a player."""

    can_check: bool
    can_call: bool
    call_amount: int  # 0 if can_check
    min_raise: int
    max_raise: int  # stack size
    can_all_in: bool
    all_in_amount: int

    def legal_actions(self) -> list[Action]:
        actions: list[Action] = [Action.fold()]
        if self.can_check:
            actions.append(Action.check())
        if self.can_call:
            actions.append(Action.call(self.call_amount))
        if self.can_all_in and self.all_in_amount > 0:
            actions.append(Action.all_in(self.all_in_amount))
        return actions

    def is_legal(self, action: Action) -> bool:
        if action.action_type == ActionType.FOLD:
            return True
        if action.action_type == ActionType.CHECK:
            return self.can_check
        if action.action_type == ActionType.CALL:
            return self.can_call and action.amount == self.call_amount
        if action.action_type == ActionType.RAISE:
            return self.min_raise <= action.amount <= self.max_raise
        if action.action_type == ActionType.ALL_IN:
            return self.can_all_in and action.amount == self.all_in_amount
        return False
