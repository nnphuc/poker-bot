"""Action abstraction: discretize bet sizes into a small set."""

from __future__ import annotations

from poker_bot.game.action import Action, ActionSpace


class ActionAbstraction:
    """Maps continuous bet sizes to a discrete set of representative actions.

    Default bet sizes: fold, check/call, 0.5x pot, 1x pot, 2x pot, all-in.
    """

    def __init__(self, bet_fractions: list[float] | None = None) -> None:
        # Fractions of pot to consider as raise sizes
        self.bet_fractions = bet_fractions or [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]

    def get_actions(self, action_space: ActionSpace, pot: int) -> list[Action]:
        """Return discretized list of legal actions."""
        actions: list[Action] = [Action.fold()]

        if action_space.can_check:
            actions.append(Action.check())
        elif action_space.can_call:
            actions.append(Action.call(action_space.call_amount))

        for frac in self.bet_fractions:
            amount = max(action_space.min_raise, int(pot * frac))
            amount = min(amount, action_space.max_raise)
            if action_space.min_raise <= amount <= action_space.max_raise:
                actions.append(Action.raise_(amount))

        if action_space.can_all_in:
            all_in = Action.all_in(action_space.all_in_amount)
            if not any(str(a) == str(all_in) for a in actions):
                actions.append(all_in)

        return self._deduplicate(actions)

    def _deduplicate(self, actions: list[Action]) -> list[Action]:
        seen: set[str] = set()
        unique: list[Action] = []
        for a in actions:
            s = str(a)
            if s not in seen:
                seen.add(s)
                unique.append(a)
        return unique
