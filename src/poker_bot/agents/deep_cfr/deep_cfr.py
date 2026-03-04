"""Deep CFR trainer (Brown et al. 2019)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

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

_ADV_CAPACITY = 1_000_000
_STRAT_CAPACITY = 2_000_000


class DeepCFR:
    """Deep Counterfactual Regret Minimization trainer.

    Uses external-sampling MCCFR traversals to fill per-player advantage
    buffers and a shared strategy buffer.  Networks are trained periodically
    via weighted MSE; the strategy network converges to an approximate
    Nash equilibrium.

    References:
        Brown et al. (2019) "Deep Counterfactual Regret Minimization"
        https://arxiv.org/abs/1811.00164
    """

    def __init__(
        self,
        engine: PokerEngine,
        stacks: list[int],
        hidden_size: int = 256,
        adv_capacity: int = _ADV_CAPACITY,
        strat_capacity: int = _STRAT_CAPACITY,
        lr: float = 1e-3,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self._engine = engine
        self._stacks = stacks
        self._n_players = len(stacks)
        self._starting_stack = stacks[0]
        self._device = torch.device(device)
        self._rng = random.Random(seed)
        self._iterations = 0

        # One advantage network per player — kept on `device` for batch training.
        self._adv_nets = [
            AdvantageNetwork(N_FEATURES, N_ACTIONS, hidden_size).to(self._device)
            for _ in range(self._n_players)
        ]
        self._adv_optims = [
            optim.Adam(net.parameters(), lr=lr) for net in self._adv_nets
        ]

        # CPU shadow copies used for single-sample traversal inference.
        # 1-sample GPU inference has >10x overhead vs CPU due to CUDA kernel
        # launch latency on a (1, N_FEATURES) tensor.  These are synced after
        # every advantage training session.
        self._infer_nets = [
            AdvantageNetwork(N_FEATURES, N_ACTIONS, hidden_size)  # CPU
            for _ in range(self._n_players)
        ]
        self._sync_infer_nets()

        # One strategy network (average strategy across all players)
        self._strat_net = StrategyNetwork(N_FEATURES, N_ACTIONS, hidden_size).to(
            self._device
        )
        self._strat_optim = optim.Adam(self._strat_net.parameters(), lr=lr)

        # Advantage buffer per player: (features, regrets, iteration_weight)
        self._adv_buffers = [
            ReservoirBuffer(adv_capacity, seed=seed) for _ in range(self._n_players)
        ]
        # Strategy buffer: (features, strategy, reach_prob, iteration_weight)
        self._strat_buffer = ReservoirBuffer(strat_capacity, seed=seed)

        # Put all networks in eval mode (training is done in batches)
        for net in self._adv_nets:
            net.eval()
        for net in self._infer_nets:
            net.eval()
        self._strat_net.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        n_iterations: int,
        train_every: int = 200,
        n_train_steps: int = 200,
        batch_size: int = 4096,
    ) -> None:
        """Run n_iterations of Deep CFR traversals.

        Args:
            n_iterations: Number of CFR iterations (each does n_players traversals).
            train_every: Train advantage networks every this many iterations.
            n_train_steps: Gradient steps per network training session.
            batch_size: Mini-batch size for network training.
        """
        for _i in range(n_iterations):
            for player_id in range(self._n_players):
                state = self._engine.new_game(
                    list(self._stacks),
                    seed=self._rng.randint(0, 2**31),
                )
                self._traverse(state, player_id, [1.0] * self._n_players, self._iterations)

            self._iterations += 1

            if self._iterations % train_every == 0:
                trained_any = False
                for player_id in range(self._n_players):
                    buf = self._adv_buffers[player_id]
                    if len(buf) >= min(batch_size, 64):
                        loss = self._train_advantage(player_id, batch_size, n_train_steps)
                        logger.debug(
                            f"Iter {self._iterations} | P{player_id} adv loss: {loss:.4f}"
                        )
                        trained_any = True
                # Sync CPU inference shadows after GPU training
                if trained_any:
                    self._sync_infer_nets()
                # Also train strategy net periodically so it improves throughout
                if len(self._strat_buffer) >= min(batch_size, 64):
                    strat_loss = self._train_strategy(batch_size, n_train_steps)
                    logger.debug(
                        f"Iter {self._iterations} | strat loss: {strat_loss:.4f}"
                    )

            if self._iterations % 1000 == 0:
                logger.info(
                    f"Deep CFR iter {self._iterations} | "
                    f"adv_bufs={self.adv_buffer_sizes} | strat_buf={len(self._strat_buffer)}"
                )

        # Train strategy network at end
        if len(self._strat_buffer) >= min(batch_size, 64):
            loss = self._train_strategy(batch_size, n_train_steps)
            logger.info(f"Strategy net trained | loss: {loss:.4f}")

    def get_strategy_network(self) -> StrategyNetwork:
        """Return the trained average-strategy network."""
        return self._strat_net

    def save_checkpoint(self, path: str | Path) -> None:
        """Save full training state for resuming."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "adv_net_states": [net.state_dict() for net in self._adv_nets],
            "strat_net_state": self._strat_net.state_dict(),
            "adv_optim_states": [opt.state_dict() for opt in self._adv_optims],
            "strat_optim_state": self._strat_optim.state_dict(),
            "adv_buffers": [list(buf._data) for buf in self._adv_buffers],
            "strat_buffer": list(self._strat_buffer._data),
            "iterations": self._iterations,
            "rng_state": self._rng.getstate(),
            "stacks": self._stacks,
            "hidden_size": self._adv_nets[0].net[0].out_features,
        }
        torch.save(payload, p)
        logger.info(f"Checkpoint saved: {p}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        engine: PokerEngine,
        device: str = "cpu",
    ) -> DeepCFR:
        """Restore trainer from a saved checkpoint."""
        # Always load tensors to CPU first; model.load_state_dict() will move
        # them to the correct device automatically since the model is already
        # placed on `device`.  Loading directly to target device can fail when
        # the device changes between save and load (e.g. cpu → cuda/mps).
        payload = torch.load(
            path, map_location="cpu", weights_only=False  # trusted internal file
        )
        hidden_size = payload.get("hidden_size", 256)
        trainer = cls(
            engine=engine,
            stacks=payload["stacks"],
            hidden_size=hidden_size,
            device=device,
        )
        for net, state in zip(trainer._adv_nets, payload["adv_net_states"], strict=True):
            net.load_state_dict(state)
        trainer._sync_infer_nets()
        trainer._strat_net.load_state_dict(payload["strat_net_state"])
        for opt, state in zip(trainer._adv_optims, payload["adv_optim_states"], strict=True):
            opt.load_state_dict(state)
        trainer._strat_optim.load_state_dict(payload["strat_optim_state"])
        for buf, data in zip(trainer._adv_buffers, payload["adv_buffers"], strict=True):
            buf._data = list(data)
        trainer._strat_buffer._data = list(payload["strat_buffer"])
        trainer._iterations = payload["iterations"]
        trainer._rng.setstate(payload["rng_state"])
        logger.info(f"Loaded checkpoint: iter={trainer._iterations}")
        return trainer

    def _sync_infer_nets(self) -> None:
        """Copy advantage network weights to CPU inference shadows."""
        for src, dst in zip(self._adv_nets, self._infer_nets, strict=True):
            dst.load_state_dict({k: v.cpu() for k, v in src.state_dict().items()})
            dst.eval()

    @property
    def n_iterations(self) -> int:
        return self._iterations

    @property
    def adv_buffer_sizes(self) -> list[int]:
        return [len(b) for b in self._adv_buffers]

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def _traverse(
        self,
        state: GameState,
        player_id: int,
        reach_probs: list[float],
        iteration: int,
    ) -> float:
        """External-sampling MCCFR traversal. Returns utility for player_id."""
        if state.is_terminal:
            return float(state.players[player_id].stack - self._stacks[player_id])

        current = state.current_player_idx
        mask, actions_map = get_action_mask(state, self._engine)
        features = encode_state(state, current, self._starting_stack)
        strategy = self._regret_match(current, features, mask)
        weight = float(iteration + 1)  # linear CFR weighting

        if current == player_id:
            # Full exploration: visit every valid action
            action_values: dict[int, float] = {}
            node_value = 0.0

            for a_idx in range(N_ACTIONS):
                if mask[a_idx] == 0:
                    continue
                new_state = self._engine.apply_action(state, actions_map[a_idx])
                new_reach = list(reach_probs)
                new_reach[current] *= strategy[a_idx]
                val = self._traverse(new_state, player_id, new_reach, iteration)
                action_values[a_idx] = val
                node_value += strategy[a_idx] * val

            # Counterfactual reach (product of all other players)
            opp_reach = 1.0
            for i, rp in enumerate(reach_probs):
                if i != player_id:
                    opp_reach *= rp

            # Instant regrets stored in advantage buffer
            regrets = np.zeros(N_ACTIONS, dtype=np.float32)
            for a_idx in range(N_ACTIONS):
                if mask[a_idx] == 1:
                    regrets[a_idx] = (action_values[a_idx] - node_value) * opp_reach

            self._adv_buffers[player_id].add((features, regrets, weight))

            # Strategy sample stored in strategy buffer
            strat_arr = np.array(
                [strategy.get(i, 0.0) for i in range(N_ACTIONS)], dtype=np.float32
            )
            self._strat_buffer.add((features, strat_arr, reach_probs[player_id], weight))

            return node_value

        else:
            # Chance sampling: sample one action for the opponent
            a_idx = self._sample_action(strategy, mask)
            new_state = self._engine.apply_action(state, actions_map[a_idx])
            return self._traverse(new_state, player_id, reach_probs, iteration)

    # ------------------------------------------------------------------
    # Strategy from advantage network
    # ------------------------------------------------------------------

    def _regret_match(
        self,
        player_id: int,
        features: np.ndarray,
        mask: np.ndarray,
    ) -> dict[int, float]:
        """Compute strategy via regret matching on advantage network output.

        Uses the CPU shadow network for inference — single-sample GPU inference
        has >10x overhead vs CPU due to CUDA kernel launch latency.
        """
        net = self._infer_nets[player_id]  # always on CPU
        with torch.no_grad():
            feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            adv = net(feat_t).squeeze(0).numpy()

        # Positive regret matching over valid actions
        positive = np.where(mask == 1, np.maximum(adv, 0.0), 0.0)
        total = positive.sum()

        if total > 0:
            probs = positive / total
        else:
            n_valid = float(mask.sum())
            probs = np.where(mask == 1, 1.0 / n_valid, 0.0)

        return {i: float(probs[i]) for i in range(N_ACTIONS) if mask[i] == 1}

    def _sample_action(self, strategy: dict[int, float], mask: np.ndarray) -> int:
        """Sample action index from a probability dictionary."""
        indices = [i for i in range(N_ACTIONS) if mask[i] == 1]
        probs = [strategy.get(i, 0.0) for i in indices]
        total = sum(probs)
        if total == 0:
            return self._rng.choice(indices)
        r = self._rng.random() * total
        cumulative = 0.0
        for idx, p in zip(indices, probs, strict=True):
            cumulative += p
            if r <= cumulative:
                return idx
        return indices[-1]

    # ------------------------------------------------------------------
    # Network training
    # ------------------------------------------------------------------

    def _train_advantage(
        self,
        player_id: int,
        batch_size: int,
        n_steps: int,
    ) -> float:
        """Train advantage network for player_id.  Returns mean loss."""
        net = self._adv_nets[player_id]
        optimizer = self._adv_optims[player_id]
        buf = self._adv_buffers[player_id]

        net.train()
        total_loss = 0.0

        for _ in range(n_steps):
            batch = buf.sample(batch_size)
            features = torch.tensor(
                np.stack([b[0] for b in batch]), dtype=torch.float32, device=self._device
            )
            targets = torch.tensor(
                np.stack([b[1] for b in batch]), dtype=torch.float32, device=self._device
            )
            weights = torch.tensor(
                [b[2] for b in batch], dtype=torch.float32, device=self._device
            )
            # Normalize weights within batch so their mean == 1.
            # This preserves relative ordering (later iters weighted higher)
            # while keeping the loss scale bounded as iterations grow.
            weights = weights / (weights.mean() + 1e-8)

            pred = net(features)
            loss = (weights.unsqueeze(1) * (pred - targets) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        net.eval()
        return total_loss / n_steps

    def _train_strategy(self, batch_size: int, n_steps: int) -> float:
        """Train average-strategy network.  Returns mean loss."""
        net = self._strat_net
        optimizer = self._strat_optim
        buf = self._strat_buffer

        net.train()
        total_loss = 0.0

        for _ in range(n_steps):
            batch = buf.sample(batch_size)
            features = torch.tensor(
                np.stack([b[0] for b in batch]), dtype=torch.float32, device=self._device
            )
            targets = torch.tensor(
                np.stack([b[1] for b in batch]), dtype=torch.float32, device=self._device
            )
            # Weight = reach_prob * iteration (emphasise later converged iterations)
            weights = torch.tensor(
                [b[2] * b[3] for b in batch], dtype=torch.float32, device=self._device
            )
            # Normalize weights within batch to keep loss scale bounded.
            weights = weights / (weights.mean() + 1e-8)

            pred = net(features)
            loss = (weights.unsqueeze(1) * (pred - targets) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        net.eval()
        return total_loss / n_steps
