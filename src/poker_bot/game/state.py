"""Game state representation for Texas Hold'em."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from poker_bot.game.card import Card


class BettingRound(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

    def next_round(self) -> BettingRound | None:
        try:
            return BettingRound(self.value + 1)
        except ValueError:
            return None

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class PlayerState:
    """State for a single player."""

    player_id: int
    stack: int
    hole_cards: list[Card] = field(default_factory=list)
    bet_this_round: int = 0
    total_invested: int = 0
    is_folded: bool = False
    is_all_in: bool = False
    has_acted: bool = False  # voluntarily acted this street

    @property
    def is_active(self) -> bool:
        return not self.is_folded and not self.is_all_in

    def copy(self) -> PlayerState:
        return PlayerState(
            player_id=self.player_id,
            stack=self.stack,
            hole_cards=list(self.hole_cards),
            bet_this_round=self.bet_this_round,
            total_invested=self.total_invested,
            is_folded=self.is_folded,
            is_all_in=self.is_all_in,
            has_acted=self.has_acted,
        )


@dataclass
class GameState:
    """Complete game state snapshot.

    All 5 community cards are pre-dealt into ``pending_board`` at hand start.
    The engine reveals them progressively into ``board`` as rounds advance.
    This design allows tree search (MCCFR) to copy state without a live deck.
    """

    players: list[PlayerState]
    board: list[Card]          # revealed community cards
    pending_board: list[Card]  # pre-dealt, not yet revealed
    pot: int
    current_round: BettingRound
    current_player_idx: int
    dealer_idx: int
    small_blind: int
    big_blind: int
    current_bet: int           # highest bet this street
    last_aggressor_idx: int
    action_history: list[tuple[int, str]]  # (player_id, action_str)
    is_terminal: bool = False
    winners: list[int] = field(default_factory=list)

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def active_players(self) -> list[PlayerState]:
        """Players who have not folded (includes all-in)."""
        return [p for p in self.players if not p.is_folded]

    @property
    def current_player(self) -> PlayerState:
        return self.players[self.current_player_idx]

    @property
    def pot_total(self) -> int:
        """Total chips including current street bets."""
        return self.pot + sum(p.bet_this_round for p in self.players)

    def copy(self) -> GameState:
        return GameState(
            players=[p.copy() for p in self.players],
            board=list(self.board),
            pending_board=list(self.pending_board),
            pot=self.pot,
            current_round=self.current_round,
            current_player_idx=self.current_player_idx,
            dealer_idx=self.dealer_idx,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            current_bet=self.current_bet,
            last_aggressor_idx=self.last_aggressor_idx,
            action_history=list(self.action_history),
            is_terminal=self.is_terminal,
            winners=list(self.winners),
        )

    def __str__(self) -> str:
        board_str = " ".join(str(c) for c in self.board) if self.board else "(empty)"
        lines = [
            f"Round: {self.current_round} | Pot: {self.pot_total} | Board: {board_str}",
            f"Current bet: {self.current_bet}",
        ]
        for p in self.players:
            status = "folded" if p.is_folded else ("all-in" if p.is_all_in else "active")
            cards = " ".join(str(c) for c in p.hole_cards) if p.hole_cards else "??"
            lines.append(
                f"  P{p.player_id}: stack={p.stack} bet={p.bet_this_round} "
                f"cards=[{cards}] ({status})"
            )
        return "\n".join(lines)
