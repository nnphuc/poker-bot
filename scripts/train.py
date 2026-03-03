"""Training entry point."""

from __future__ import annotations

import typer
from loguru import logger

from poker_bot.training.trainer import CFRTrainer
from poker_bot.utils.config import TrainingConfig
from poker_bot.utils.logging import setup_logging

app = typer.Typer(help="Train poker bot using MCCFR")


@app.command()
def train(
    iterations: int = typer.Option(100_000, "--iterations", "-i", help="Total CFR iterations"),
    checkpoint_every: int = typer.Option(10_000, "--checkpoint-every", help="Save every N iters"),
    checkpoint_dir: str = typer.Option("models/checkpoints", "--output", "-o"),
    small_blind: int = typer.Option(50, "--sb"),
    big_blind: int = typer.Option(100, "--bb"),
    stack: int = typer.Option(10_000, "--stack"),
    seed: int = typer.Option(42, "--seed"),
    resume: str | None = typer.Option(
        None,
        "--resume",
        "-r",
        help=(
            "Resume training. Pass a checkpoint path to resume from a specific file, "
            "or 'auto' to resume from the latest checkpoint in --output directory."
        ),
    ),
    log_file: str = typer.Option("", "--log-file", help="Optional log file path"),
) -> None:
    """Train MCCFR poker bot. Saves both resumable checkpoints and strategy snapshots."""
    setup_logging(log_file=log_file or None)

    config = TrainingConfig(
        starting_stacks=(stack, stack),
        small_blind=small_blind,
        big_blind=big_blind,
        total_iterations=iterations,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )

    # Resolve --resume argument
    resume_arg: bool | str
    if resume is None:
        resume_arg = False
    elif resume == "auto":
        resume_arg = True
    else:
        resume_arg = resume

    logger.info(f"Config: {config}")
    trainer = CFRTrainer(config)
    strategy = trainer.train(resume=resume_arg)
    logger.success(f"Training done. Infosets: {strategy.n_infosets:,}")


if __name__ == "__main__":
    app()
