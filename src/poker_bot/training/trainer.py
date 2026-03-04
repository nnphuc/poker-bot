"""CFR training orchestration."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn

from poker_bot.abstraction.card_abstraction import CardAbstraction
from poker_bot.agents.cfr.mccfr import MCCFR
from poker_bot.agents.cfr.strategy import Strategy
from poker_bot.game.engine import PokerEngine
from poker_bot.utils.config import TrainingConfig

# Filename patterns written to checkpoint_dir
_CKPT_PATTERN = "ckpt_{:08d}.pkl"      # full training state (resumable)
_STRAT_PATTERN = "strategy_{:08d}.pkl" # averaged strategy snapshot (for playing)
_FINAL_STRAT = "strategy_final.pkl"


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the most recent training checkpoint file, or None."""
    ckpts = sorted(checkpoint_dir.glob("ckpt_*.pkl"))
    return ckpts[-1] if ckpts else None


class CFRTrainer:
    """Orchestrates MCCFR training with save/load checkpoint support.

    Each checkpoint interval writes two files:
    - ``ckpt_NNNNNNNN.pkl``      — full training state (regret sums), resumable
    - ``strategy_NNNNNNNN.pkl``  — averaged strategy snapshot, for playing/eval

    Usage::

        # Fresh run
        trainer = CFRTrainer(config)
        strategy = trainer.train()

        # Auto-resume from latest checkpoint
        strategy = trainer.train(resume=True)

        # Resume from a specific checkpoint file
        strategy = trainer.train(resume="models/checkpoints/ckpt_00050000.pkl")
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._engine = PokerEngine(config.small_blind, config.big_blind)
        self._abstraction = CardAbstraction(n_buckets=config.n_card_buckets)

    def train(self, resume: bool | str | Path = False) -> Strategy:
        """Run training and return the final averaged strategy.

        Args:
            resume: False  — start fresh.
                    True   — auto-load latest ckpt_*.pkl in checkpoint_dir.
                    str|Path — load a specific checkpoint file.
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        mccfr = self._init_mccfr(resume, checkpoint_dir)

        completed = mccfr._iterations
        total = self.config.total_iterations
        remaining = total - completed

        if remaining <= 0:
            logger.info(f"Already completed {completed:,} iterations — nothing to do.")
            return mccfr.get_strategy()

        logger.info(
            f"MCCFR training | total={total:,} | completed={completed:,} "
            f"| remaining={remaining:,}"
        )
        logger.info(
            f"Stacks: {self.config.starting_stacks} | "
            f"Blinds: {self.config.small_blind}/{self.config.big_blind}"
        )

        chunk = self.config.checkpoint_every

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Training MCCFR", total=total, completed=completed)

            step = completed
            while step < total:
                iters = min(chunk, total - step)
                mccfr.train(iters)
                step += iters
                progress.update(task, completed=step)

                # Resumable training checkpoint
                ckpt_path = checkpoint_dir / _CKPT_PATTERN.format(step)
                mccfr.save_checkpoint(ckpt_path)

                # Averaged strategy snapshot (for evaluation / playing)
                strat_path = checkpoint_dir / _STRAT_PATTERN.format(step)
                mccfr.get_strategy().save(strat_path)

                logger.info(
                    f"[{step:,}/{total:,}] saved | infosets={mccfr.n_infosets:,} "
                    f"| ckpt={ckpt_path.name} | strategy={strat_path.name}"
                )

        final_strategy = mccfr.get_strategy()
        final_path = checkpoint_dir / _FINAL_STRAT
        final_strategy.save(final_path)
        logger.success(
            f"Training done | iterations={total:,} | "
            f"infosets={mccfr.n_infosets:,} | {final_path}"
        )
        return final_strategy

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_mccfr(self, resume: bool | str | Path, checkpoint_dir: Path) -> MCCFR:
        stacks = list(self.config.starting_stacks)

        if resume is False:
            return MCCFR(
                engine=self._engine,
                stacks=stacks,
                seed=self.config.seed,
                abstraction=self._abstraction,
            )

        ckpt_path: Path | None
        if resume is True:
            ckpt_path = _latest_checkpoint(checkpoint_dir)
            if ckpt_path is None:
                logger.warning("No checkpoint found in {checkpoint_dir} — starting fresh.")
                return MCCFR(engine=self._engine, stacks=stacks, seed=self.config.seed)
        else:
            ckpt_path = Path(resume)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        return MCCFR.load_checkpoint(
            ckpt_path,
            engine=self._engine,
            stacks=stacks,
            abstraction=self._abstraction,
        )
