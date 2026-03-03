"""Strategy profile serialization and loading."""

from __future__ import annotations

import pickle
from pathlib import Path


class Strategy:
    """Trained strategy profile mapping infoset keys to action distributions."""

    def __init__(
        self,
        profile: dict[str, dict[int, float]],
        iterations: int = 0,
    ) -> None:
        self._profile = profile
        self.iterations = iterations

    def get(self, key: str, n_actions: int) -> dict[int, float]:
        """Return strategy for infoset key, uniform if unseen."""
        if key in self._profile:
            return self._profile[key]
        return dict.fromkeys(range(n_actions), 1.0 / n_actions)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"profile": self._profile, "iterations": self.iterations}, f)

    @classmethod
    def load(cls, path: str | Path) -> Strategy:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["profile"], data.get("iterations", 0))

    @property
    def n_infosets(self) -> int:
        return len(self._profile)
