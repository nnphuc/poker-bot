"""Head-to-head agent evaluation."""

from __future__ import annotations

import random

from loguru import logger

from poker_bot.agents.base import Agent
from poker_bot.evaluation.metrics import compute_metrics
from poker_bot.game.engine import PokerEngine


class HeadToHeadEvaluator:
    """Evaluate two agents in heads-up play.

    Uses duplicate matching (same deals both ways) to reduce variance.
    """

    def __init__(
        self,
        engine: PokerEngine,
        starting_stack: int = 10_000,
        seed: int | None = None,
    ) -> None:
        self._engine = engine
        self._starting_stack = starting_stack
        self._rng = random.Random(seed)

    def evaluate(
        self,
        agent_a: Agent,
        agent_b: Agent,
        n_hands: int = 1000,
    ) -> dict[str, float]:
        """Play n_hands and return metrics for agent_a."""
        profits_bb: list[float] = []
        big_blind = float(self._engine.big_blind)

        seeds = [self._rng.randint(0, 2**31) for _ in range(n_hands // 2)]

        for seed in seeds:
            for swap in (False, True):
                a, b = (agent_a, agent_b) if not swap else (agent_b, agent_a)
                stacks = [self._starting_stack, self._starting_stack]
                profit = self._play_hand(a, b, stacks, seed)
                if swap:
                    profit = -profit
                profits_bb.append(profit / big_blind)

        metrics = compute_metrics(profits_bb)
        logger.info(
            f"Evaluation: {n_hands} hands | "
            f"Win rate: {metrics['win_rate']:.1%} | "
            f"BB/100: {metrics['bb_per_100']:.2f}"
        )
        return metrics

    def _play_hand(
        self,
        agent_a: Agent,
        agent_b: Agent,
        stacks: list[int],
        seed: int,
    ) -> int:
        """Play one hand and return agent_a's profit in chips."""
        agent_a.reset()
        agent_b.reset()

        state = self._engine.new_game(stacks, seed=seed)
        agents = {0: agent_a, 1: agent_b}

        while not state.is_terminal:
            current_player_id = state.current_player_idx
            action = agents[current_player_id].act(state, self._engine)
            state = self._engine.apply_action(state, action)

        return state.players[0].stack - stacks[0]
