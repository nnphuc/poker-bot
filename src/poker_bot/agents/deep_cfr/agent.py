"""Deep CFR agent for inference."""

from __future__ import annotations

import random

import numpy as np
import torch

from poker_bot.agents.base import Agent
from poker_bot.agents.deep_cfr.encoder import N_ACTIONS, encode_state, get_action_mask
from poker_bot.agents.deep_cfr.network import StrategyNetwork
from poker_bot.game.action import Action
from poker_bot.game.engine import PokerEngine
from poker_bot.game.state import GameState


class DeepCFRAgent(Agent):
    """Plays using a trained Deep CFR strategy network.

    The network outputs logits for each of the 6 fixed action slots.
    Invalid actions are masked out, and the resulting distribution is
    sampled to choose an action.
    """

    def __init__(
        self,
        strategy_net: StrategyNetwork,
        engine: PokerEngine,
        starting_stack: int = 10_000,
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._net = strategy_net.to(self._device)
        self._engine = engine
        self._starting_stack = starting_stack
        self._rng = random.Random()
        self._net.eval()

    def act(self, state: GameState, engine: PokerEngine) -> Action:
        mask, actions_map = get_action_mask(state, engine)
        features = encode_state(state, state.current_player_idx, self._starting_stack)

        with torch.no_grad():
            feat_t = torch.tensor(
                features, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            logits = self._net(feat_t).squeeze(0).cpu().numpy()

        # Mask invalid slots then softmax
        logits = np.where(mask == 1, logits, -1e9)
        logits -= logits.max()
        probs = np.exp(logits) * mask
        total = probs.sum()
        probs = probs / total if total > 0 else mask / float(mask.sum())

        # Weighted random sample
        valid = [i for i in range(N_ACTIONS) if mask[i] == 1]
        r = self._rng.random()
        cumulative = 0.0
        for i in valid:
            cumulative += probs[i]
            if r <= cumulative:
                return actions_map[i]
        return actions_map[valid[-1]]
