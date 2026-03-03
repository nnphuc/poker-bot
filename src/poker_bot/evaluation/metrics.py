"""Performance metrics for poker bot evaluation."""

from __future__ import annotations

import math
import statistics


def compute_metrics(profits_bb: list[float]) -> dict[str, float]:
    """Compute standard poker metrics from per-hand profits in BB."""
    if not profits_bb:
        return {}

    n = len(profits_bb)
    total = sum(profits_bb)
    mean = total / n

    wins = sum(1 for p in profits_bb if p > 0)
    losses = sum(1 for p in profits_bb if p < 0)
    ties = n - wins - losses

    std_dev = statistics.stdev(profits_bb) if n > 1 else 0.0
    # 95% confidence interval for win rate
    ci_95 = 1.96 * std_dev / math.sqrt(n) if n > 1 else 0.0

    return {
        "n_hands": float(n),
        "win_rate": wins / n,
        "bb_per_100": mean * 100,
        "std_dev_bb": std_dev,
        "ci_95_bb": ci_95,
        "total_profit_bb": total,
        "wins": float(wins),
        "losses": float(losses),
        "ties": float(ties),
    }
