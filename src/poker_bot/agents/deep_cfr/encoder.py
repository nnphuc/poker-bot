"""State encoding and fixed action-space mapping for Deep CFR."""

from __future__ import annotations

from collections import Counter

import numpy as np

from poker_bot.game.action import Action, ActionSpace
from poker_bot.game.card import Card
from poker_bot.game.engine import PokerEngine
from poker_bot.game.state import GameState

# ---------------------------------------------------------------------------
# Feature / action dimensions
# ---------------------------------------------------------------------------

# 35-dim feature vector layout:
#   [0-3]   hole cards          (rank, suit) * 2
#   [4-13]  board cards         (rank, suit) * 5, zero-padded
#   [14]    pot total           normalized
#   [15]    own stack           normalized
#   [16-19] opponent stacks     normalized, up to 4 seats (zero-padded)
#   [20]    current bet         normalized
#   [21-24] betting round       one-hot (preflop/flop/turn/river)
#   [25]    is_dealer           flag
#   [26]    is_small_blind      flag
#   [27]    is_big_blind        flag
#   [28]    stack-to-pot ratio  clipped to [0, 1]
#   [29]    pot odds            call_amount / (pot + call_amount)
#   [30]    aggression count    total raises+all-ins in hand, normalized
#   [31]    active players      n_active / n_total
#   [32]    flush draw          2+ board cards of same suit
#   [33]    paired board        2+ board cards of same rank
#   [34]    straight draw       2+ board cards within 4-rank span
N_FEATURES: int = 35

# Fixed 7-slot action space:
#   0  fold
#   1  check / call
#   2  raise 0.33x pot
#   3  raise 0.50x pot
#   4  raise 1.00x pot
#   5  raise 2.00x pot
#   6  all-in
N_ACTIONS: int = 7
_RAISE_FRACS: tuple[float, ...] = (0.33, 0.5, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Board texture helpers
# ---------------------------------------------------------------------------


def _flush_draw(board: list[Card]) -> float:
    if len(board) < 2:
        return 0.0
    return float(max(Counter(c.suit for c in board).values()) >= 2)


def _paired_board(board: list[Card]) -> float:
    if len(board) < 2:
        return 0.0
    return float(max(Counter(c.rank for c in board).values()) >= 2)


def _straight_draw(board: list[Card]) -> float:
    if len(board) < 2:
        return 0.0
    ranks = sorted({c.rank.value for c in board})
    for i in range(len(ranks)):
        cnt = sum(1 for r in ranks if ranks[i] <= r <= ranks[i] + 4)
        if cnt >= 2:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Main encoding
# ---------------------------------------------------------------------------


def encode_state(
    state: GameState,
    player_id: int,
    starting_stack: int = 10_000,
) -> np.ndarray:
    """Encode game state as a 35-dim float32 vector from player_id's POV."""
    player = state.players[player_id]
    big = float(state.big_blind)
    n = state.num_players
    obs = np.zeros(N_FEATURES, dtype=np.float32)

    # [0-3] Hole cards
    for i, card in enumerate(player.hole_cards[:2]):
        obs[i * 2] = (card.rank.value - 2) / 12.0
        obs[i * 2 + 1] = card.suit.value / 3.0

    # [4-13] Board cards (zero-padded)
    for i, card in enumerate(state.board[:5]):
        obs[4 + i * 2] = (card.rank.value - 2) / 12.0
        obs[4 + i * 2 + 1] = card.suit.value / 3.0

    # [14] Pot total
    pot = float(state.pot_total)
    obs[14] = pot / (starting_stack * n)

    # [15] Own stack
    obs[15] = player.stack / starting_stack

    # [16-19] Opponent stacks in seat order
    opp_slot = 0
    for offset in range(1, n):
        seat = (player_id + offset) % n
        if opp_slot < 4:
            obs[16 + opp_slot] = state.players[seat].stack / starting_stack
            opp_slot += 1

    # [20] Current bet
    obs[20] = state.current_bet / big / 100.0

    # [21-24] Round one-hot
    obs[21 + state.current_round.value] = 1.0

    # [25-27] Position flags
    dealer_idx = state.dealer_idx
    sb_idx = dealer_idx if n == 2 else (dealer_idx + 1) % n
    bb_idx = (dealer_idx + 1) % n if n == 2 else (dealer_idx + 2) % n
    obs[25] = float(player_id == dealer_idx)
    obs[26] = float(player_id == sb_idx)
    obs[27] = float(player_id == bb_idx)

    # [28] Stack-to-pot ratio (SPR), clipped to [0, 10] then normalized
    obs[28] = min(player.stack / max(pot, 1.0), 10.0) / 10.0

    # [29] Pot odds: fraction of pot the player must call
    call_amount = max(0, state.current_bet - player.bet_this_round)
    denom = pot + call_amount
    obs[29] = call_amount / denom if denom > 0 else 0.0

    # [30] Aggression count (raises + all-ins in full hand so far)
    agg = sum(
        1
        for _, a in state.action_history
        if a.startswith("raise") or a.startswith("all_in")
    )
    obs[30] = min(agg / 10.0, 1.0)

    # [31] Active (non-folded) players ratio
    n_active = sum(1 for p in state.players if not p.is_folded)
    obs[31] = n_active / n

    # [32-34] Board texture
    board = state.board
    obs[32] = _flush_draw(board)
    obs[33] = _paired_board(board)
    obs[34] = _straight_draw(board)

    return obs


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------


def get_action_mask(
    state: GameState,
    engine: PokerEngine,
) -> tuple[np.ndarray, dict[int, Action]]:
    """Return (mask, actions_map) for the fixed 7-slot action space.

    mask[i] == 1 means action slot i is legal in this state.
    actions_map maps valid slot indices to concrete Action objects.
    """
    action_space: ActionSpace = engine.get_action_space(state)
    pot = float(state.pot_total) or float(engine.big_blind)

    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    actions_map: dict[int, Action] = {}

    # Slot 0: fold (always available)
    mask[0] = 1.0
    actions_map[0] = Action.fold()

    # Slot 1: check or call
    if action_space.can_check:
        mask[1] = 1.0
        actions_map[1] = Action.check()
    elif action_space.can_call:
        mask[1] = 1.0
        actions_map[1] = Action.call(action_space.call_amount)

    # Slots 2-5: raise at 0.33x / 0.50x / 1.0x / 2.0x pot
    if action_space.min_raise > 0:
        seen_amounts: set[int] = set()
        for i, frac in enumerate(_RAISE_FRACS):
            amount = max(action_space.min_raise, int(pot * frac))
            amount = min(amount, action_space.max_raise)
            if action_space.min_raise <= amount and amount not in seen_amounts:
                mask[2 + i] = 1.0
                actions_map[2 + i] = Action.raise_(amount)
                seen_amounts.add(amount)

    # Slot 6: all-in
    if action_space.can_all_in:
        mask[6] = 1.0
        actions_map[6] = Action.all_in(action_space.all_in_amount)

    return mask, actions_map
