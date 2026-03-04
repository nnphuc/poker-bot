"""Poker game engine - manages game flow and state transitions."""

from __future__ import annotations

from poker_bot.game.action import Action, ActionSpace, ActionType
from poker_bot.game.card import Deck
from poker_bot.game.hand_rank import HandEvaluator
from poker_bot.game.state import BettingRound, GameState, PlayerState


class PokerEngine:
    """No-limit Texas Hold'em game engine (heads-up and multi-player).

    All 5 community cards are pre-dealt into state.pending_board at game start.
    apply_action needs no deck -- state is self-contained, safe for MCCFR.
    """

    def __init__(self, small_blind: int = 50, big_blind: int = 100) -> None:
        self.small_blind = small_blind
        self.big_blind = big_blind
        self._evaluator = HandEvaluator()

    def new_game(
        self,
        stacks: list[int],
        dealer_idx: int = 0,
        seed: int | None = None,
    ) -> GameState:
        """Initialise a new hand and return the complete initial state."""
        n = len(stacks)
        if not 2 <= n <= 5:
            raise ValueError("Need 2-5 players")

        deck = Deck()
        deck.shuffle(seed)

        players = [PlayerState(player_id=i, stack=stacks[i]) for i in range(n)]

        for _ in range(2):
            for p in players:
                p.hole_cards.extend(deck.deal(1))

        pending_board = deck.deal(5)

        sb_idx = dealer_idx if n == 2 else (dealer_idx + 1) % n
        bb_idx = (dealer_idx + 1) % n if n == 2 else (dealer_idx + 2) % n

        self._post_blind(players[sb_idx], self.small_blind)
        self._post_blind(players[bb_idx], self.big_blind)

        first_to_act = sb_idx if n == 2 else (bb_idx + 1) % n

        return GameState(
            players=players,
            board=[],
            pending_board=pending_board,
            pot=0,
            current_round=BettingRound.PREFLOP,
            current_player_idx=first_to_act,
            dealer_idx=dealer_idx,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            current_bet=self.big_blind,
            last_aggressor_idx=bb_idx,
            action_history=[],
        )

    def get_action_space(self, state: GameState) -> ActionSpace:
        """Return legal actions for the current player."""
        player = state.current_player
        to_call = state.current_bet - player.bet_this_round
        to_call = min(to_call, player.stack)

        can_check = state.current_bet == player.bet_this_round
        can_call = not can_check and 0 < to_call < player.stack

        last_raise_size = max(state.current_bet, state.big_blind)
        min_raise_total = state.current_bet + last_raise_size
        min_raise_chips = min_raise_total - player.bet_this_round

        can_raise = player.stack > to_call and min_raise_chips < player.stack

        return ActionSpace(
            can_check=can_check,
            can_call=can_call,
            call_amount=to_call,
            min_raise=min_raise_chips if can_raise else 0,
            max_raise=player.stack,
            can_all_in=player.stack > 0,
            all_in_amount=player.stack,
        )

    def apply_action(self, state: GameState, action: Action) -> GameState:
        """Apply action and return a new state. Does NOT mutate the input."""
        state = state.copy()
        player = state.players[state.current_player_idx]
        player.has_acted = True

        if action.action_type == ActionType.FOLD:
            player.is_folded = True

        elif action.action_type == ActionType.CHECK:
            pass

        elif action.action_type == ActionType.CALL:
            chips = min(action.amount, player.stack)
            player.stack -= chips
            player.bet_this_round += chips
            player.total_invested += chips
            if player.stack == 0:
                player.is_all_in = True

        elif action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
            chips = min(action.amount, player.stack)
            player.stack -= chips
            player.bet_this_round += chips
            player.total_invested += chips
            state.current_bet = player.bet_this_round
            state.last_aggressor_idx = state.current_player_idx
            for p in state.players:
                if p.player_id != player.player_id and not p.is_folded:
                    p.has_acted = False
            if player.stack == 0:
                player.is_all_in = True

        state.action_history.append((player.player_id, str(action)))
        return self._advance(state)

    def _post_blind(self, player: PlayerState, amount: int) -> None:
        chips = min(amount, player.stack)
        player.stack -= chips
        player.bet_this_round += chips
        player.total_invested += chips
        if player.stack == 0:
            player.is_all_in = True

    def _advance(self, state: GameState) -> GameState:
        non_folded = [p for p in state.players if not p.is_folded]

        if len(non_folded) == 1:
            return self._resolve(state)

        if self._is_round_over(state):
            return self._start_next_round(state)

        idx = state.current_player_idx
        n = state.num_players
        for _ in range(n):
            idx = (idx + 1) % n
            if state.players[idx].is_active:
                state.current_player_idx = idx
                return state

        return self._start_next_round(state)

    def _is_round_over(self, state: GameState) -> bool:
        for p in state.players:
            if p.is_folded or p.is_all_in:
                continue
            if not p.has_acted:
                return False
            if p.bet_this_round < state.current_bet:
                return False
        return True

    def _start_next_round(self, state: GameState) -> GameState:
        for p in state.players:
            state.pot += p.bet_this_round
            p.bet_this_round = 0
            p.has_acted = False

        next_round = state.current_round.next_round()
        if next_round is None:
            return self._resolve(state)

        state.current_round = next_round
        state.current_bet = 0

        if next_round == BettingRound.FLOP:
            state.board.extend(state.pending_board[:3])
        elif next_round == BettingRound.TURN:
            state.board.extend(state.pending_board[3:4])
        elif next_round == BettingRound.RIVER:
            state.board.extend(state.pending_board[4:5])

        n = state.num_players
        idx = (state.dealer_idx + 1) % n
        for _ in range(n):
            if state.players[idx].is_active:
                state.current_player_idx = idx
                state.last_aggressor_idx = idx
                break
            idx = (idx + 1) % n

        # If no one can act (all all-in), run out remaining streets automatically
        if not any(p.is_active for p in state.players if not p.is_folded):
            return self._start_next_round(state)

        return state

    def _resolve(self, state: GameState) -> GameState:
        state.is_terminal = True

        for p in state.players:
            state.pot += p.bet_this_round
            p.bet_this_round = 0

        non_folded = [p for p in state.players if not p.is_folded]

        if len(non_folded) == 1:
            winner = non_folded[0]
            winner.stack += state.pot
            state.winners = [winner.player_id]
            state.pot = 0
            return state

        full_board = (state.board + state.pending_board)[:5]

        scores = {
            p.player_id: self._evaluator.evaluate(p.hole_cards, full_board)
            for p in non_folded
        }
        best_score = min(scores.values())
        winners = [pid for pid, score in scores.items() if score == best_score]

        share = state.pot // len(winners)
        remainder = state.pot % len(winners)
        for pid in winners:
            state.players[pid].stack += share
        state.players[winners[0]].stack += remainder

        state.winners = winners
        state.pot = 0
        return state
