"""Hand strength evaluation using phevaluator."""

from __future__ import annotations

from phevaluator import evaluate_cards  # type: ignore[import-untyped]

from poker_bot.game.card import Card


class HandEvaluator:
    """Evaluate 5-7 card poker hands.

    Lower score = stronger hand (phevaluator convention).
    Score 1 = Royal Flush, Score 7462 = 7-2 offsuit.
    """

    @staticmethod
    def evaluate(hole_cards: list[Card], board: list[Card]) -> int:
        """Return hand strength score. Lower is better."""
        all_cards = hole_cards + board
        if len(all_cards) < 5 or len(all_cards) > 7:
            raise ValueError(f"Need 5-7 cards, got {len(all_cards)}")
        card_ints = [c.to_int() for c in all_cards]
        return int(evaluate_cards(*card_ints))  # type: ignore[no-untyped-call]

    @staticmethod
    def rank_class(score: int) -> str:
        """Return human-readable hand class from score."""
        if score == 1:
            return "Royal Flush"
        elif score <= 10:
            return "Straight Flush"
        elif score <= 166:
            return "Four of a Kind"
        elif score <= 322:
            return "Full House"
        elif score <= 1599:
            return "Flush"
        elif score <= 1609:
            return "Straight"
        elif score <= 2467:
            return "Three of a Kind"
        elif score <= 3325:
            return "Two Pair"
        elif score <= 6185:
            return "One Pair"
        else:
            return "High Card"

    @staticmethod
    def equity(
        hole_cards: list[Card],
        board: list[Card],
        n_simulations: int = 1000,
        deck_remaining: list[Card] | None = None,
    ) -> float:
        """Monte Carlo equity estimation vs random opponent."""
        import random

        if deck_remaining is None:
            from poker_bot.game.card import Deck
            deck = Deck()
            deck.shuffle()
            used = set(hole_cards + board)
            deck_remaining = [c for c in deck if c not in used]

        wins = 0
        for _ in range(n_simulations):
            sample = random.sample(deck_remaining, 2 + (5 - len(board)))
            opp_hole = sample[:2]
            extra_board = sample[2:]
            full_board = board + extra_board

            my_score = HandEvaluator.evaluate(hole_cards, full_board)
            opp_score = HandEvaluator.evaluate(opp_hole, full_board)

            if my_score < opp_score:
                wins += 1
            elif my_score == opp_score:
                wins += 0.5

        return wins / n_simulations
