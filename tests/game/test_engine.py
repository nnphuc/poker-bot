"""Tests for the poker game engine."""

from __future__ import annotations

from poker_bot.game.action import Action
from poker_bot.game.engine import PokerEngine
from poker_bot.game.state import BettingRound


def test_new_game_initial_state(engine: PokerEngine, default_stacks: list[int]) -> None:
    state = engine.new_game(default_stacks, seed=42)
    assert len(state.players) == 2
    assert state.current_round == BettingRound.PREFLOP
    assert not state.is_terminal
    assert len(state.players[0].hole_cards) == 2
    assert len(state.players[1].hole_cards) == 2
    assert len(state.pending_board) == 5


def test_blinds_posted(engine: PokerEngine, default_stacks: list[int]) -> None:
    state = engine.new_game(default_stacks, seed=42)
    total_bets = sum(p.bet_this_round for p in state.players)
    assert total_bets == engine.small_blind + engine.big_blind


def test_fold_wins(engine: PokerEngine, default_stacks: list[int]) -> None:
    state = engine.new_game(default_stacks, seed=42)
    state = engine.apply_action(state, Action.fold())
    assert state.is_terminal
    winner_idx = state.winners[0]
    assert state.players[winner_idx].stack > default_stacks[winner_idx]


def test_call_advances_round(engine: PokerEngine, default_stacks: list[int]) -> None:
    state = engine.new_game(default_stacks, seed=42)
    action_space = engine.get_action_space(state)
    if action_space.can_call:
        state = engine.apply_action(state, Action.call(action_space.call_amount))
    else:
        state = engine.apply_action(state, Action.check())
    if not state.is_terminal:
        state = engine.apply_action(state, Action.check())
    if not state.is_terminal:
        assert state.current_round == BettingRound.FLOP
        assert len(state.board) == 3


def test_all_in_resolves(engine: PokerEngine, default_stacks: list[int]) -> None:
    state = engine.new_game(default_stacks, seed=42)
    action_space = engine.get_action_space(state)
    state = engine.apply_action(state, Action.all_in(action_space.all_in_amount))
    if not state.is_terminal:
        action_space2 = engine.get_action_space(state)
        state = engine.apply_action(state, Action.all_in(action_space2.all_in_amount))
    assert state.is_terminal
    total = sum(p.stack for p in state.players)
    assert total == sum(default_stacks)


def test_chips_conserved_full_hand(engine: PokerEngine) -> None:
    """Total chips must be conserved after any complete hand."""
    from poker_bot.agents.random_agent import RandomAgent

    agent = RandomAgent(seed=7)
    stacks = [5000, 5000]

    for seed in range(20):
        state = engine.new_game(stacks, seed=seed)
        while not state.is_terminal:
            action = agent.act(state, engine)
            state = engine.apply_action(state, action)
        total = sum(p.stack for p in state.players)
        assert total == sum(stacks), f"Chips not conserved on seed {seed}"


def test_action_space_preflop(engine: PokerEngine, default_stacks: list[int]) -> None:
    state = engine.new_game(default_stacks, seed=42)
    action_space = engine.get_action_space(state)
    assert action_space.can_call or action_space.can_check
    assert action_space.can_all_in
