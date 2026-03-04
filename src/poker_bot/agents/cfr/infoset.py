"""Information set abstraction for CFR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from poker_bot.game.state import BettingRound, GameState

if TYPE_CHECKING:
    from poker_bot.abstraction.card_abstraction import CardAbstraction

# Short prefix per betting round used in abstracted keys
_ROUND_TAG = {
    BettingRound.PREFLOP: "PF",
    BettingRound.FLOP: "FL",
    BettingRound.TURN: "TU",
    BettingRound.RIVER: "RI",
}


def _abstract_act(act_str: str) -> str:
    """Strip exact amounts: 'raise:450' -> 'R', 'call:100' -> 'C', etc."""
    if act_str == "fold":
        return "F"
    if act_str == "check":
        return "X"
    if act_str.startswith("call"):
        return "C"
    if act_str.startswith("raise"):
        return "R"
    if act_str.startswith("all_in"):
        return "A"
    return act_str


@dataclass(frozen=True)
class InfosetKey:
    """Hashable key uniquely identifying an information set."""

    key: str

    def __str__(self) -> str:
        return self.key


def build_infoset_key(
    state: GameState,
    player_id: int,
    abstraction: CardAbstraction | None = None,
) -> InfosetKey:
    """Build information set key for a player.

    With abstraction=None (default): raw cards + exact bet amounts.
    With abstraction provided: equity bucket + abstract action types only.
    Abstracted form collapses millions of infosets into thousands.
    """
    player = state.players[player_id]

    if abstraction is not None:
        # Card part: "FL:5" = flop, bucket 5
        if state.current_round == BettingRound.PREFLOP:
            bucket = abstraction.preflop_bucket(player.hole_cards)
        else:
            bucket = abstraction.get_bucket(player.hole_cards, state.board)
        card_str = f"{_ROUND_TAG[state.current_round]}:{bucket}"
        history = "|".join(
            f"{pid}:{_abstract_act(act)}" for pid, act in state.action_history
        )
    else:
        hole = sorted(player.hole_cards, key=lambda c: (c.rank.value, c.suit.value))
        card_str = "".join(str(c) for c in hole)
        board_str = "".join(str(c) for c in state.board)
        if board_str:
            card_str += f"/{board_str}"
        history = "|".join(f"{pid}:{act}" for pid, act in state.action_history)

    return InfosetKey(f"{card_str}/{history}")


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
