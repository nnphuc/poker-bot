"""Deep CFR training entry point."""

from __future__ import annotations

from pathlib import Path

import torch
import typer
from loguru import logger

from poker_bot.agents.deep_cfr.deep_cfr import DeepCFR
from poker_bot.game.engine import PokerEngine
from poker_bot.utils.logging import setup_logging

app = typer.Typer(help="Train poker bot using Deep CFR")


@app.command()
def train(
    iterations: int = typer.Option(10_000, "--iterations", "-i", help="CFR iterations"),
    train_every: int = typer.Option(200, "--train-every", help="Train nets every N iters"),
    n_train_steps: int = typer.Option(200, "--train-steps", help="Gradient steps per session"),
    batch_size: int = typer.Option(4096, "--batch-size", "-b"),
    checkpoint_every: int = typer.Option(1_000, "--checkpoint-every"),
    output: str = typer.Option("models/deep_cfr", "--output", "-o"),
    n_players: int = typer.Option(2, "--players", "-n", help="Number of players (2-5)"),
    stack: int = typer.Option(10_000, "--stack"),
    small_blind: int = typer.Option(50, "--sb"),
    big_blind: int = typer.Option(100, "--bb"),
    hidden_size: int = typer.Option(256, "--hidden"),
    seed: int = typer.Option(42, "--seed"),
    resume: str | None = typer.Option(
        None,
        "--resume",
        "-r",
        help="Path to a .pt checkpoint to resume from.",
    ),
    device: str = typer.Option("cpu", "--device", help="cpu | cuda | mps (auto-validated)"),
    max_checkpoints: int = typer.Option(
        2, "--max-checkpoints", help="Max checkpoints to keep (0=unlimited)"
    ),
    prefix: str = typer.Option("poker_bot", "--prefix", help="Checkpoint filename prefix"),
    log_file: str = typer.Option("", "--log-file"),
) -> None:
    """Train Deep CFR poker bot.  Checkpoints saved as .pt files."""
    setup_logging(log_file=log_file or None)

    if not 2 <= n_players <= 5:
        raise typer.BadParameter("--players must be 2-5")

    # Validate / auto-select device
    import torch as _torch

    if device == "cuda" and not _torch.cuda.is_available():
        if _torch.backends.mps.is_available():
            logger.warning("CUDA not available. Falling back to --device mps (Apple Silicon).")
            device = "mps"
        else:
            logger.warning("CUDA not available. Falling back to --device cpu.")
            device = "cpu"
    elif device == "mps" and not _torch.backends.mps.is_available():
        logger.warning("MPS not available. Falling back to --device cpu.")
        device = "cpu"
    logger.info(f"Using device: {device}")

    stacks = [stack] * n_players
    engine = PokerEngine(small_blind, big_blind)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        logger.info(f"Resuming from {resume}")
        trainer = DeepCFR.load_checkpoint(resume, engine, device=device)
    else:
        trainer = DeepCFR(
            engine=engine,
            stacks=stacks,
            hidden_size=hidden_size,
            seed=seed,
            device=device,
        )

    remaining = iterations - trainer.n_iterations
    if remaining <= 0:
        logger.info("Already completed requested iterations.")
        return

    logger.info(
        f"Deep CFR | players={n_players} | stacks={stacks} | "
        f"blinds={small_blind}/{big_blind} | device={device}"
    )
    logger.info(f"Iterations: {trainer.n_iterations}/{iterations}")

    step = trainer.n_iterations
    while step < iterations:
        chunk = min(checkpoint_every, iterations - step)
        trainer.train(
            chunk,
            train_every=train_every,
            n_train_steps=n_train_steps,
            batch_size=batch_size,
        )
        step += chunk

        ckpt = output_dir / f"{prefix}_{step:08d}.pt"
        trainer.save_checkpoint(ckpt)
        logger.info(
            f"[{step}/{iterations}] | adv_bufs={trainer.adv_buffer_sizes} "
            f"| strat_buf={len(trainer._strat_buffer)}"
        )

        # Prune old checkpoints, keeping only the most recent max_checkpoints
        if max_checkpoints > 0:
            old = sorted(output_dir.glob(f"{prefix}_????????.pt"))[:-max_checkpoints]
            for f in old:
                f.unlink()
                logger.debug(f"Removed old checkpoint: {f.name}")

    # Save final strategy network weights separately
    final = output_dir / "strategy_net_final.pt"
    torch.save(trainer.get_strategy_network().state_dict(), final)
    logger.success(f"Done. Strategy network saved: {final}")


if __name__ == "__main__":
    app()
