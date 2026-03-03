"""Card representation for Texas Hold'em poker."""

from __future__ import annotations

import random
from collections.abc import Iterator
from enum import IntEnum


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        symbols = {
            2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
            7: "7", 8: "8", 9: "9", 10: "T",
            11: "J", 12: "Q", 13: "K", 14: "A",
        }
        return symbols[self.value]


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    def __str__(self) -> str:
        symbols = {0: "c", 1: "d", 2: "h", 3: "s"}
        return symbols[self.value]


class Card:
    """Immutable playing card."""

    __slots__ = ("_rank", "_suit")

    def __init__(self, rank: Rank, suit: Suit) -> None:
        self._rank = rank
        self._suit = suit

    @property
    def rank(self) -> Rank:
        return self._rank

    @property
    def suit(self) -> Suit:
        return self._suit

    @classmethod
    def from_str(cls, s: str) -> Card:
        """Parse card from string like 'As', 'Kh', 'Tc', '2d'."""
        rank_map = {
            "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR,
            "5": Rank.FIVE, "6": Rank.SIX, "7": Rank.SEVEN,
            "8": Rank.EIGHT, "9": Rank.NINE, "T": Rank.TEN,
            "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING,
            "A": Rank.ACE,
        }
        suit_map = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}
        if len(s) != 2 or s[0] not in rank_map or s[1] not in suit_map:
            raise ValueError(f"Invalid card string: {s!r}")
        return cls(rank_map[s[0]], suit_map[s[1]])

    def to_int(self) -> int:
        """Convert to integer 0-51 for phevaluator."""
        return (self._rank.value - 2) * 4 + self._suit.value

    def __str__(self) -> str:
        return f"{self._rank}{self._suit}"

    def __repr__(self) -> str:
        return f"Card({self!s})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._rank == other._rank and self._suit == other._suit

    def __hash__(self) -> int:
        return self.to_int()

    def __lt__(self, other: Card) -> bool:
        return (self._rank, self._suit) < (other._rank, other._suit)


class Deck:
    """Standard 52-card deck."""

    def __init__(self) -> None:
        self._cards: list[Card] = [
            Card(rank, suit) for rank in Rank for suit in Suit
        ]
        self._dealt: int = 0

    def shuffle(self, seed: int | None = None) -> None:
        rng = random.Random(seed)
        rng.shuffle(self._cards)
        self._dealt = 0

    def deal(self, n: int = 1) -> list[Card]:
        if self._dealt + n > 52:
            raise ValueError("Not enough cards remaining in deck")
        cards = self._cards[self._dealt: self._dealt + n]
        self._dealt += n
        return cards

    def remaining(self) -> int:
        return 52 - self._dealt

    def __iter__(self) -> Iterator[Card]:
        return iter(self._cards[self._dealt:])

    def __len__(self) -> int:
        return self.remaining()
