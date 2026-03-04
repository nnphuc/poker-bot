"""Card abstraction: bucket hands into equity buckets (OCHS-inspired)."""

from __future__ import annotations

from poker_bot.game.card import Card
from poker_bot.game.hand_rank import HandEvaluator


class CardAbstraction:
    """Groups hands into equity buckets for tractable CFR.

    Uses E[HS^2] (Expected Hand Strength squared) bucketing,
    a simple but effective abstraction used in CEPHEUS and similar bots.
    Results are cached per (hole_cards, board) tuple for MCCFR performance.
    """

    def __init__(self, n_buckets: int = 8, n_simulations: int = 50) -> None:
        self.n_buckets = n_buckets
        self.n_simulations = n_simulations
        self._evaluator = HandEvaluator()
        self._cache: dict[tuple[tuple[int, int], ...], int] = {}

    def _cache_key(
        self, hole_cards: list[Card], board: list[Card]
    ) -> tuple[tuple[int, int], ...]:
        """Canonical order-independent key for caching."""
        hole = tuple(sorted((c.rank.value, c.suit.value) for c in hole_cards))
        brd = tuple(sorted((c.rank.value, c.suit.value) for c in board))
        return hole + brd

    def get_bucket(self, hole_cards: list[Card], board: list[Card]) -> int:
        """Return bucket index 0..n_buckets-1 for given hand + board."""
        key = self._cache_key(hole_cards, board)
        if key in self._cache:
            return self._cache[key]
        equity = self._evaluator.equity(hole_cards, board, self.n_simulations)
        bucket = min(int(equity * self.n_buckets), self.n_buckets - 1)
        self._cache[key] = bucket
        return bucket

    def get_equity(self, hole_cards: list[Card], board: list[Card]) -> float:
        """Return raw equity estimate."""
        return self._evaluator.equity(hole_cards, board, self.n_simulations)

    def preflop_bucket(self, hole_cards: list[Card]) -> int:
        """Fast preflop bucketing using Chen formula approximation."""
        c1, c2 = sorted(hole_cards, key=lambda c: c.rank.value, reverse=True)
        score = self._chen_formula(c1, c2)
        # Normalize Chen score (approx range -1 to 20) to 0..n_buckets-1
        normalized = (score + 1) / 21.0
        bucket = int(normalized * self.n_buckets)
        return max(0, min(bucket, self.n_buckets - 1))

    def _chen_formula(self, high: Card, low: Card) -> float:
        """Chen formula for preflop hand strength."""
        rank_scores = {14: 10, 13: 8, 12: 7, 11: 6, 10: 5}
        score = rank_scores.get(high.rank.value, high.rank.value / 2.0)

        if high.rank == low.rank:  # pocket pair
            score = max(score * 2, 5)
        else:
            gap = high.rank.value - low.rank.value
            gap_penalty = {1: 0, 2: -1, 3: -2, 4: -4}
            score += gap_penalty.get(gap, -5)
            # Straight potential (pairs can't form connectors)
            if gap <= 2 and high.rank.value < 12:
                score += 1

        if high.suit == low.suit:  # suited
            score += 2

        return score
