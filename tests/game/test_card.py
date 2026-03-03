"""Tests for card module."""

from __future__ import annotations

import pytest

from poker_bot.game.card import Card, Deck, Rank, Suit


def test_card_creation() -> None:
    card = Card(Rank.ACE, Suit.SPADES)
    assert card.rank == Rank.ACE
    assert card.suit == Suit.SPADES


def test_card_str() -> None:
    assert str(Card(Rank.ACE, Suit.SPADES)) == "As"
    assert str(Card(Rank.TEN, Suit.HEARTS)) == "Th"
    assert str(Card(Rank.TWO, Suit.CLUBS)) == "2c"


def test_card_from_str() -> None:
    assert Card.from_str("As") == Card(Rank.ACE, Suit.SPADES)
    assert Card.from_str("Kh") == Card(Rank.KING, Suit.HEARTS)
    assert Card.from_str("2d") == Card(Rank.TWO, Suit.DIAMONDS)
    assert Card.from_str("Tc") == Card(Rank.TEN, Suit.CLUBS)


def test_card_from_str_invalid() -> None:
    with pytest.raises(ValueError):
        Card.from_str("XX")
    with pytest.raises(ValueError):
        Card.from_str("A")


def test_card_to_int_range() -> None:
    for rank in Rank:
        for suit in Suit:
            card = Card(rank, suit)
            assert 0 <= card.to_int() <= 51


def test_card_to_int_unique() -> None:
    ints = {Card(rank, suit).to_int() for rank in Rank for suit in Suit}
    assert len(ints) == 52


def test_card_equality() -> None:
    c1 = Card(Rank.ACE, Suit.SPADES)
    c2 = Card(Rank.ACE, Suit.SPADES)
    c3 = Card(Rank.KING, Suit.SPADES)
    assert c1 == c2
    assert c1 != c3


def test_deck_size() -> None:
    deck = Deck()
    assert len(deck) == 52


def test_deck_deal() -> None:
    deck = Deck()
    deck.shuffle(seed=42)
    cards = deck.deal(5)
    assert len(cards) == 5
    assert len(deck) == 47


def test_deck_deal_all() -> None:
    deck = Deck()
    deck.shuffle()
    all_cards = deck.deal(52)
    assert len(set(all_cards)) == 52  # all unique


def test_deck_deal_too_many() -> None:
    deck = Deck()
    with pytest.raises(ValueError):
        deck.deal(53)


def test_deck_shuffle_reproducible() -> None:
    deck1 = Deck()
    deck2 = Deck()
    deck1.shuffle(seed=123)
    deck2.shuffle(seed=123)
    assert deck1.deal(5) == deck2.deal(5)
