"""CFR training orchestration."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from poker_bot.agents.cfr.mccfr import MCCFR
from poker_bot.agents.cfr.strategy import Strategy
from poker_bot.game.engine import PokerEngine
from poker_bot.utils.config import TrainingConfig


class CFRTrainer:
    """Orchestrates MCCFR training with checkpointing."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._engine = PokerEngine(config.small_blind, config.big_blind)

    def train(self) -> Strategy:
        """Run full training and return strategy."""
        logger.info(f"Starting CFR training: {self.config.total_iterations} iterations")
        logger.info(
            f"Stacks: {self.config.starting_stacks}, "
            f"Blinds: {self.config.small_blind}/{self.config.big_blind}"
        )

        mccfr = MCCFR(
            engine=self._engine,
            stacks=list(self.config.starting_stacks),
            seed=self.config.seed,
        )

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        chunk = self.config.checkpoint_every
        total = self.config.total_iterations

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Training MCCFR...", total=total)

            for step in range(0, total, chunk):
                iters = min(chunk, total - step)
                mccfr.train(iters)
                progress.advance(task, iters)

                strategy = mccfr.get_strategy()
                ckpt_path = checkpoint_dir / f"strategy_{step + iters:08d}.pkl"
                strategy.save(ckpt_path)
                logger.info(
                    f"Checkpoint saved: {ckpt_path} "
                    f"| infosets={mccfr.n_infosets:,}"
                )

        final_strategy = mccfr.get_strategy()
        final_path = checkpoint_dir / "strategy_final.pkl"
        final_strategy.save(final_path)
        logger.success(f"Training complete. Final strategy: {final_path}")
        logger.info(f"Total infosets: {mccfr.n_infosets:,}")
        return final_strategy
