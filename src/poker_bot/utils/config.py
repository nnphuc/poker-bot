"""Configuration dataclasses for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuration for MCCFR training."""

    # Game setup
    starting_stacks: tuple[int, int] = (10_000, 10_000)
    small_blind: int = 50
    big_blind: int = 100

    # Training
    total_iterations: int = 100_000
    checkpoint_every: int = 10_000
    checkpoint_dir: str = "models/checkpoints"

    # Reproducibility
    seed: int | None = None

    # Abstraction
    n_card_buckets: int = 8
    bet_fractions: list[float] = field(
        default_factory=lambda: [0.33, 0.5, 0.75, 1.0, 2.0]
    )


@dataclass
class EvalConfig:
    """Configuration for agent evaluation."""

    starting_stack: int = 10_000
    small_blind: int = 50
    big_blind: int = 100
    n_hands: int = 10_000
    seed: int | None = 42
    strategy_path: str = "models/checkpoints/strategy_final.pkl"
