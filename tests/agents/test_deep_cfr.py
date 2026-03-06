"""Tests for Deep CFR components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from poker_bot.agents.deep_cfr.agent import DeepCFRAgent
from poker_bot.agents.deep_cfr.deep_cfr import DeepCFR
from poker_bot.agents.deep_cfr.encoder import (
    N_ACTIONS,
    N_FEATURES,
    encode_state,
    get_action_mask,
)
from poker_bot.agents.deep_cfr.network import AdvantageNetwork, StrategyNetwork
from poker_bot.agents.deep_cfr.reservoir import ReservoirBuffer
from poker_bot.game.engine import PokerEngine
from poker_bot.game.state import GameState

# ---------------------------------------------------------------------------
# ReservoirBuffer
# ---------------------------------------------------------------------------


def test_reservoir_buffer_fills_up() -> None:
    buf = ReservoirBuffer(capacity=10, seed=0)
    for i in range(10):
        buf.add(i)
    assert len(buf) == 10


def test_reservoir_buffer_does_not_exceed_capacity() -> None:
    buf = ReservoirBuffer(capacity=5, seed=0)
    for i in range(100):
        buf.add(i)
    assert len(buf) == 5


def test_reservoir_buffer_sample_size() -> None:
    buf = ReservoirBuffer(capacity=20, seed=0)
    for i in range(20):
        buf.add(i)
    sample = buf.sample(5)
    assert len(sample) == 5


def test_reservoir_buffer_sample_smaller_than_buffer() -> None:
    buf = ReservoirBuffer(capacity=3, seed=0)
    for i in range(2):
        buf.add(i)
    sample = buf.sample(10)
    assert len(sample) == 2


def test_reservoir_buffer_is_empty() -> None:
    buf = ReservoirBuffer(capacity=10)
    assert buf.is_empty
    buf.add(1)
    assert not buf.is_empty


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def test_advantage_network_output_shape() -> None:
    net = AdvantageNetwork(N_FEATURES, N_ACTIONS, hidden_size=64)
    x = torch.randn(4, N_FEATURES)
    out = net(x)
    assert out.shape == (4, N_ACTIONS)


def test_strategy_network_output_shape() -> None:
    net = StrategyNetwork(N_FEATURES, N_ACTIONS, hidden_size=64)
    x = torch.randn(3, N_FEATURES)
    out = net(x)
    assert out.shape == (3, N_ACTIONS)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_encode_state_shape(new_game: GameState) -> None:
    features = encode_state(new_game, 0)
    assert features.shape == (N_FEATURES,)
    assert features.dtype == np.float32


def test_encode_state_deterministic(new_game: GameState) -> None:
    f1 = encode_state(new_game, 0)
    f2 = encode_state(new_game, 0)
    np.testing.assert_array_equal(f1, f2)


def test_get_action_mask_shape(new_game: GameState, engine: PokerEngine) -> None:
    mask, _actions_map = get_action_mask(new_game, engine)
    assert mask.shape == (N_ACTIONS,)
    assert mask.sum() >= 2  # fold + call/check at minimum


def test_get_action_mask_fold_always_valid(
    new_game: GameState, engine: PokerEngine
) -> None:
    mask, actions_map = get_action_mask(new_game, engine)
    assert mask[0] == 1.0  # slot 0 = fold
    assert 0 in actions_map


def test_get_action_mask_all_actions_mapped(
    new_game: GameState, engine: PokerEngine
) -> None:
    mask, actions_map = get_action_mask(new_game, engine)
    for slot in range(N_ACTIONS):
        if mask[slot] == 1.0:
            assert slot in actions_map


# ---------------------------------------------------------------------------
# DeepCFR trainer
# ---------------------------------------------------------------------------


def test_deep_cfr_trains(engine: PokerEngine) -> None:
    trainer = DeepCFR(engine, stacks=[10_000, 10_000], hidden_size=64, seed=42)
    trainer.train(n_iterations=5, train_every=5, n_train_steps=2, batch_size=32)
    assert trainer.n_iterations == 5
    total = sum(trainer.adv_buffer_sizes)
    assert total > 0


def test_deep_cfr_strat_buffer_fills(engine: PokerEngine) -> None:
    trainer = DeepCFR(engine, stacks=[10_000, 10_000], hidden_size=64, seed=0)
    trainer.train(n_iterations=10, train_every=100, n_train_steps=1, batch_size=32)
    assert len(trainer._strat_buffer) > 0


def test_deep_cfr_get_strategy_network(engine: PokerEngine) -> None:
    trainer = DeepCFR(engine, stacks=[10_000, 10_000], hidden_size=64, seed=7)
    trainer.train(n_iterations=3, train_every=100, n_train_steps=1, batch_size=32)
    net = trainer.get_strategy_network()
    assert isinstance(net, StrategyNetwork)


def test_deep_cfr_checkpoint_save_load(
    tmp_path: Path, engine: PokerEngine
) -> None:
    trainer = DeepCFR(engine, stacks=[10_000, 10_000], hidden_size=64, seed=1)
    trainer.train(n_iterations=5, train_every=100, n_train_steps=1, batch_size=32)

    ckpt = tmp_path / "deep_cfr.pt"
    trainer.save_checkpoint(ckpt)

    loaded = DeepCFR.load_checkpoint(ckpt, engine=engine)
    assert loaded.n_iterations == trainer.n_iterations
    assert loaded.adv_buffer_sizes == trainer.adv_buffer_sizes


def test_deep_cfr_3_players(engine: PokerEngine) -> None:
    trainer = DeepCFR(
        engine, stacks=[10_000, 10_000, 10_000], hidden_size=64, seed=99
    )
    trainer.train(n_iterations=3, train_every=100, n_train_steps=1, batch_size=32)
    assert trainer.n_iterations == 3
    assert len(trainer.adv_buffer_sizes) == 3


def test_deep_cfr_updates_opponent_reach_and_records_strategy_sample(
    engine: PokerEngine,
) -> None:
    class ProbeDeepCFR(DeepCFR):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self._probe_seen_root = False
            self.observed_reach: list[float] | None = None

        def _traverse(
            self,
            state: GameState,
            player_id: int,
            reach_probs: list[float],
            iteration: int,
        ) -> float:
            if self._probe_seen_root:
                self.observed_reach = list(reach_probs)
                return 0.0
            self._probe_seen_root = True
            return super()._traverse(state, player_id, reach_probs, iteration)

    trainer = ProbeDeepCFR(engine, stacks=[10_000, 10_000], hidden_size=64, seed=13)
    state = engine.new_game([10_000, 10_000], seed=7)

    expected_prob = 0.75

    def fixed_regret_match(
        player_id: int,
        features: np.ndarray,
        mask: np.ndarray,
    ) -> dict[int, float]:
        del player_id, features
        strategy = {i: 0.0 for i in range(N_ACTIONS) if mask[i] == 1}
        strategy[0] = 0.25
        strategy[1] = expected_prob
        return strategy

    trainer._regret_match = fixed_regret_match  # type: ignore[method-assign]
    trainer._sample_action = lambda strategy, mask: 1  # type: ignore[method-assign]

    trainer._traverse(state, player_id=1, reach_probs=[1.0, 1.0], iteration=0)

    assert trainer.observed_reach is not None
    assert trainer.observed_reach[0] == pytest.approx(expected_prob)
    assert len(trainer._strat_buffer) == 1


# ---------------------------------------------------------------------------
# DeepCFRAgent
# ---------------------------------------------------------------------------


def test_deep_cfr_agent_acts(new_game: GameState, engine: PokerEngine) -> None:
    net = StrategyNetwork(N_FEATURES, N_ACTIONS, hidden_size=64)
    agent = DeepCFRAgent(net, engine, starting_stack=10_000)
    action = agent.act(new_game, engine)
    assert action is not None


def test_deep_cfr_agent_acts_after_training(
    new_game: GameState, engine: PokerEngine
) -> None:
    trainer = DeepCFR(engine, stacks=[10_000, 10_000], hidden_size=64, seed=5)
    trainer.train(n_iterations=5, train_every=5, n_train_steps=2, batch_size=32)
    net = trainer.get_strategy_network()
    agent = DeepCFRAgent(net, engine, starting_stack=10_000)
    action = agent.act(new_game, engine)
    assert action is not None


@pytest.mark.parametrize("n_games", [5])
def test_deep_cfr_agent_consistent_valid_actions(
    n_games: int, engine: PokerEngine
) -> None:
    net = StrategyNetwork(N_FEATURES, N_ACTIONS, hidden_size=64)
    agent = DeepCFRAgent(net, engine, starting_stack=10_000)

    for seed in range(n_games):
        state = engine.new_game([10_000, 10_000], seed=seed)
        action = agent.act(state, engine)
        assert action is not None
        new_state = engine.apply_action(state, action)
        assert new_state is not None
