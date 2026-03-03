"""Information set abstraction for CFR."""

from __future__ import annotations

from dataclasses import dataclass

from poker_bot.game.state import GameState


@dataclass(frozen=True)
class InfosetKey:
    """Hashable key uniquely identifying an information set."""

    key: str

    def __str__(self) -> str:
        return self.key


def build_infoset_key(state: GameState, player_id: int) -> InfosetKey:
    """Build information set key for a player.

    Encodes: hole cards (sorted for suit isomorphism), board, betting history.
    """
    player = state.players[player_id]

    # Sort hole cards for canonical representation (suit isomorphism simplified)
    hole = sorted(player.hole_cards, key=lambda c: (c.rank.value, c.suit.value))
    hole_str = "".join(str(c) for c in hole)

    board_str = "".join(str(c) for c in state.board)

    # Betting history this hand
    history = "|".join(f"{pid}:{act}" for pid, act in state.action_history)

    key = f"{hole_str}/{board_str}/{history}"
    return InfosetKey(key)


@dataclass
class InfosetData:
    """Regret sums and strategy profile for one information set."""

    # action index -> cumulative regret
    regret_sum: dict[int, float]
    # action index -> cumulative strategy
    strategy_sum: dict[int, float]

    def __init__(self, n_actions: int) -> None:
        self.regret_sum = dict.fromkeys(range(n_actions), 0.0)
        self.strategy_sum = dict.fromkeys(range(n_actions), 0.0)

    def get_strategy(self, reach_prob: float) -> dict[int, float]:
        """Regret-matching strategy."""
        positive = {a: max(r, 0.0) for a, r in self.regret_sum.items()}
        total = sum(positive.values())
        if total > 0:
            strategy = {a: v / total for a, v in positive.items()}
        else:
            n = len(self.regret_sum)
            strategy = dict.fromkeys(self.regret_sum, 1.0 / n)

        for a, prob in strategy.items():
            self.strategy_sum[a] = self.strategy_sum.get(a, 0.0) + reach_prob * prob

        return strategy

    def get_average_strategy(self) -> dict[int, float]:
        """Normalized average strategy (converges to Nash equilibrium)."""
        total = sum(self.strategy_sum.values())
        if total > 0:
            return {a: v / total for a, v in self.strategy_sum.items()}
        n = len(self.strategy_sum)
        return dict.fromkeys(self.strategy_sum, 1.0 / n)
