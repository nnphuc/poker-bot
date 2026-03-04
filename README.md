# poker-bot

Unlimited Texas Hold'em poker bot using **Monte Carlo CFR (MCCFR)** and **Deep CFR** — two state-of-the-art game-theoretic AI algorithms.

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **MCCFR** | Chance-sampling Monte Carlo Counterfactual Regret Minimization. Tabular, converges to Nash equilibrium via regret matching. |
| **Deep CFR** | Neural network approximation of counterfactual advantages (Brown et al. 2019). Scalable to large state spaces via reservoir sampling and MLP networks. |

## Project Structure

```
src/poker_bot/
├── game/           # Core engine: state, cards, actions, hand evaluation
├── agents/
│   ├── cfr/        # MCCFR trainer + agent
│   └── deep_cfr/   # Deep CFR trainer, agent, networks, encoder, buffer
├── abstraction/    # Card abstraction (OCHS equity buckets)
├── training/       # CFRTrainer with checkpoint/resume support
├── evaluation/     # HeadToHeadEvaluator + metrics
├── env/            # Gymnasium environment wrapper
└── utils/          # Config, logging

scripts/
├── train.py            # Train MCCFR
├── train_deep_cfr.py   # Train Deep CFR
└── evaluate.py         # Evaluate agent vs baseline
```

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Usage

### Train MCCFR

```bash
uv run python scripts/train.py \
  --iterations 100000 \
  --checkpoint-every 10000 \
  --output models/checkpoints
```

Resume from a checkpoint:

```bash
uv run python scripts/train.py --resume auto --output models/checkpoints
```

### Train Deep CFR

```bash
uv run python scripts/train_deep_cfr.py \
  --iterations 10000 \
  --train-every 100 \
  --hidden 256 \
  --output models/deep_cfr
```

### Evaluate

```bash
# Evaluate MCCFR strategy
uv run python scripts/evaluate.py \
  --strategy models/checkpoints/strategy_final.pkl \
  --opponent random \
  --hands 10000

# Evaluate Deep CFR strategy
uv run python scripts/evaluate.py \
  --strategy models/deep_cfr/strategy_final.pt \
  --agent-type deep_cfr \
  --opponent call
```

Outputs: win rate, BB/100, std dev, 95% confidence interval.

## Architecture Notes

**Pre-dealt cards**: All 5 community cards are dealt upfront into `state.pending_board` and revealed progressively. `apply_action()` takes no deck — state is fully self-contained. This makes MCCFR tree search safe: states can be freely copied and branched.

**Deep CFR state encoding**: 25-dimensional feature vector covering hole cards, board cards, pot size, stack sizes, current bet, and betting round (one-hot). Action space is abstracted to 6 slots: fold, check/call, raise 0.5×/1×/2× pot, all-in.

## Development

```bash
# Run tests
uv run pytest tests/ -v --no-cov

# Lint
uv run ruff check src/ scripts/ tests/

# Type check
uv run mypy src/
```

## Dependencies

- [`phevaluator`](https://github.com/HenryRLee/PokerHandEvaluator) — hand evaluation
- [`torch`](https://pytorch.org/) — neural networks (Deep CFR)
- [`gymnasium`](https://gymnasium.farama.org/) — RL environment interface
- [`numpy`](https://numpy.org/), [`typer`](https://typer.tiangolo.com/), [`rich`](https://github.com/Textualize/rich), [`loguru`](https://loguru.readthedocs.io/)

## References

- Lanctot et al. (2009) — [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html)
- Brown et al. (2019) — [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164)
