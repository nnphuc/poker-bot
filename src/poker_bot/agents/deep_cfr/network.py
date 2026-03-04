"""Neural networks for Deep CFR."""

from __future__ import annotations

import torch
import torch.nn as nn


class _MLP(nn.Module):
    """3-layer MLP with LayerNorm shared by both advantage and strategy networks."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdvantageNetwork(_MLP):
    """Approximates counterfactual advantage A(I, a) for each action.

    Output is a raw real-valued vector (one entry per action slot).
    Regret matching is applied externally to derive a strategy.
    """


class StrategyNetwork(_MLP):
    """Approximates the average strategy P(a | I).

    Output is raw logits; apply softmax externally after masking.
    """
