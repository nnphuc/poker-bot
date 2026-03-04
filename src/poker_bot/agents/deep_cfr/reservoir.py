"""Reservoir buffer for Deep CFR experience storage."""

from __future__ import annotations

import random
from typing import Any


class ReservoirBuffer:
    """Fixed-capacity buffer using reservoir sampling.

    Guarantees uniform random coverage: each of the last ``capacity``
    items from an infinite stream has equal probability of being stored.
    """

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self._capacity = capacity
        self._data: list[Any] = []
        self._seen = 0
        self._rng = random.Random(seed)

    def add(self, item: Any) -> None:
        """Insert item using reservoir sampling."""
        self._seen += 1
        if len(self._data) < self._capacity:
            self._data.append(item)
        else:
            idx = self._rng.randint(0, self._seen - 1)
            if idx < self._capacity:
                self._data[idx] = item

    def sample(self, n: int) -> list[Any]:
        """Return up to n uniformly sampled items."""
        return self._rng.sample(self._data, min(n, len(self._data)))

    def __len__(self) -> int:
        return len(self._data)

    @property
    def is_empty(self) -> bool:
        return len(self._data) == 0
