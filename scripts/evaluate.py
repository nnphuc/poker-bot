"""Evaluation entry point."""

from __future__ import annotations

from pathlib import Path

import torch
import typer
from loguru import logger

from poker_bot.agents.call_agent import CallAgent
from poker_bot.agents.cfr.mccfr import MCCFRAgent
from poker_bot.agents.cfr.strategy import Strategy
from poker_bot.agents.deep_cfr.agent import DeepCFRAgent
from poker_bot.agents.deep_cfr.encoder import N_ACTIONS, N_FEATURES
from poker_bot.agents.deep_cfr.network import StrategyNetwork
from poker_bot.agents.random_agent import RandomAgent
from poker_bot.evaluation.evaluator import HeadToHeadEvaluator
from poker_bot.game.engine import PokerEngine
from poker_bot.utils.logging import setup_logging

app = typer.Typer(help="Evaluate trained poker bot")


@app.command()
def evaluate(
    strategy_path: str = typer.Option(
        "models/checkpoints/strategy_final.pkl", "--strategy", "-s"
    ),
    opponent: str = typer.Option("random", "--opponent", "-o", help="random | call"),
    n_hands: int = typer.Option(10_000, "--hands", "-n"),
    stack: int = typer.Option(10_000, "--stack"),
    seed: int = typer.Option(42, "--seed"),
    agent_type: str = typer.Option(
        "mccfr", "--agent-type", "-t", help="mccfr | deep_cfr"
    ),
    device: str = typer.Option("cpu", "--device", help="cpu or cuda (deep_cfr only)"),
) -> None:
    """Evaluate a trained bot vs baseline agents.

    Supports MCCFR strategy (.pkl) and Deep CFR strategy network (.pt).
    """
    setup_logging()

    if not Path(strategy_path).exists():
        logger.error(f"Strategy file not found: {strategy_path}")
        raise typer.Exit(1)

    engine = PokerEngine(small_blind=50, big_blind=100)

    if agent_type == "mccfr":
        strategy = Strategy.load(strategy_path)
        logger.info(
            f"Loaded strategy: {strategy.n_infosets:,} infosets, "
            f"{strategy.iterations} iterations"
        )
        bot = MCCFRAgent(strategy, engine)
    elif agent_type == "deep_cfr":
        net = StrategyNetwork(N_FEATURES, N_ACTIONS)
        state_dict = torch.load(strategy_path, map_location=device, weights_only=True)
        net.load_state_dict(state_dict)
        bot = DeepCFRAgent(net, engine, starting_stack=stack, device=device)
        logger.info(f"Loaded Deep CFR strategy network: {strategy_path}")
    else:
        logger.error(f"Unknown --agent-type '{agent_type}'. Use 'mccfr' or 'deep_cfr'.")
        raise typer.Exit(1)

    if opponent == "random":
        opp = RandomAgent(seed=seed)
    elif opponent == "call":
        opp = CallAgent()
    else:
        logger.error(f"Unknown opponent: {opponent}")
        raise typer.Exit(1)

    evaluator = HeadToHeadEvaluator(engine, starting_stack=stack, seed=seed)
    metrics = evaluator.evaluate(bot, opp, n_hands=n_hands)

    typer.echo("\n=== Evaluation Results ===")
    typer.echo(f"Opponent:    {opponent}")
    typer.echo(f"Hands:       {int(metrics['n_hands'])}")
    typer.echo(f"Win Rate:    {metrics['win_rate']:.1%}")
    typer.echo(f"BB/100:      {metrics['bb_per_100']:+.2f}")
    typer.echo(f"Std Dev:     {metrics['std_dev_bb']:.2f} BB")
    typer.echo(f"95% CI:      +-{metrics['ci_95_bb']:.2f} BB/hand")


if __name__ == "__main__":
    app()
