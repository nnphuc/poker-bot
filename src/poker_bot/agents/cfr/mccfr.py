"""Monte Carlo CFR (Chance Sampling) for No-Limit Texas Hold'em."""

from __future__ import annotations

import pickle
import random
from pathlib import Path

from loguru import logger

from poker_bot.agents.base import Agent
from poker_bot.agents.cfr.infoset import InfosetData, build_infoset_key
from poker_bot.agents.cfr.strategy import Strategy
from poker_bot.game.action import Action
from poker_bot.game.engine import PokerEngine
from poker_bot.game.state import GameState

# Discretized bet sizes as fractions of pot
BET_FRACTIONS = [0.0, 0.5, 1.0, float("inf")]  # 0=check/call/fold, inf=all-in


def get_abstract_actions(state: GameState, engine: PokerEngine) -> list[Action]:
    """Return a small, discretized set of actions for CFR tractability."""
    action_space = engine.get_action_space(state)
    actions: list[Action] = [Action.fold()]

    if action_space.can_check:
        actions.append(Action.check())
    elif action_space.can_call:
        actions.append(Action.call(action_space.call_amount))

    pot = float(state.pot_total) or float(engine.big_blind)

    for frac in [0.5, 1.0, 2.0]:
        amount = int(pot * frac)
        amount = max(action_space.min_raise, amount)
        amount = min(amount, action_space.max_raise)
        if action_space.min_raise <= amount <= action_space.max_raise:
            actions.append(Action.raise_(amount))

    if action_space.can_all_in:
        actions.append(Action.all_in(action_space.all_in_amount))

    # Deduplicate by str representation
    seen: set[str] = set()
    unique: list[Action] = []
    for a in actions:
        s = str(a)
        if s not in seen:
            seen.add(s)
            unique.append(a)
    return unique


class MCCFR:
    """Monte Carlo CFR trainer (Chance Sampling variant).

    Suitable for extensive-form games with chance nodes (card deals).
    """

    def __init__(
        self,
        engine: PokerEngine,
        stacks: list[int],
        seed: int | None = None,
    ) -> None:
        self._engine = engine
        self._stacks = stacks
        self._infosets: dict[str, InfosetData] = {}
        self._rng = random.Random(seed)
        self._iterations = 0

    def train(self, n_iterations: int) -> None:
        """Run n_iterations of MCCFR."""
        for i in range(n_iterations):
            for player_id in range(len(self._stacks)):
                state = self._engine.new_game(
                    list(self._stacks), seed=self._rng.randint(0, 2**31)
                )
                self._mccfr(state, player_id, [1.0, 1.0])
            self._iterations += 1

            if (i + 1) % 1000 == 0:
                logger.info(f"MCCFR iteration {i + 1}/{n_iterations}")

    def _mccfr(
        self,
        state: GameState,
        player_id: int,
        reach_probs: list[float],
    ) -> float:
        """Recursive MCCFR. Returns expected utility for player_id."""
        if state.is_terminal:
            return float(state.players[player_id].stack - self._stacks[player_id])

        current = state.current_player_idx
        actions = get_abstract_actions(state, self._engine)
        n_actions = len(actions)

        key = build_infoset_key(state, current)
        if key.key not in self._infosets:
            self._infosets[key.key] = InfosetData(n_actions)
        infoset = self._infosets[key.key]

        strategy = infoset.get_strategy(reach_probs[current])

        action_utils: dict[int, float] = {}
        node_util = 0.0

        for a_idx, action in enumerate(actions):
            prob = strategy[a_idx]
            new_state = self._engine.apply_action(state, action)

            if current == player_id:
                new_reach = list(reach_probs)
                new_reach[current] *= prob
                action_utils[a_idx] = self._mccfr(new_state, player_id, new_reach)
            else:
                # Sample one action for opponent (chance sampling)
                if self._rng.random() < prob:
                    action_utils[a_idx] = self._mccfr(new_state, player_id, reach_probs)
                else:
                    action_utils[a_idx] = 0.0

            node_util += prob * action_utils[a_idx]

        # Update regrets for current player only
        if current == player_id:
            opp_reach = reach_probs[1 - player_id] if len(reach_probs) == 2 else 1.0
            for a_idx in range(n_actions):
                regret = action_utils.get(a_idx, 0.0) - node_util
                infoset.regret_sum[a_idx] += opp_reach * regret

        return node_util

    def get_strategy(self) -> Strategy:
        """Extract average strategy profile (for playing, not resuming)."""
        avg: dict[str, dict[int, float]] = {}
        for key, data in self._infosets.items():
            avg[key] = data.get_average_strategy()
        return Strategy(avg, self._iterations)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save full training state (regret + strategy sums) for later resuming."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "infosets": self._infosets,
            "iterations": self._iterations,
            "rng_state": self._rng.getstate(),
        }
        with open(p, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        engine: PokerEngine,
        stacks: list[int],
    ) -> MCCFR:
        """Restore a trainer from a saved checkpoint to continue training."""
        with open(path, "rb") as f:
            payload = pickle.load(f)  # trusted internal checkpoint file

        trainer = cls(engine=engine, stacks=stacks)
        trainer._infosets = payload["infosets"]
        trainer._iterations = payload["iterations"]
        trainer._rng.setstate(payload["rng_state"])
        return trainer

    @property
    def n_infosets(self) -> int:
        return len(self._infosets)


class MCCFRAgent(Agent):
    """Agent that plays according to a trained MCCFR strategy."""

    def __init__(self, strategy: Strategy, engine: PokerEngine) -> None:
        self._strategy = strategy
        self._engine = engine
        self._rng = random.Random()

    def act(self, state: GameState, engine: PokerEngine) -> Action:
        actions = get_abstract_actions(state, engine)
        key = build_infoset_key(state, state.current_player_idx)
        probs = self._strategy.get(key.key, len(actions))

        # Sample action from strategy
        r = self._rng.random()
        cumulative = 0.0
        for i, action in enumerate(actions):
            cumulative += probs.get(i, 0.0)
            if r <= cumulative:
                return action
        return actions[-1]
