"""Simple baseline agents used for evaluation and partner selection."""

from __future__ import annotations

import numpy as np

from GameEnv.actions import Action


class RandomAgent:
    def __init__(
        self,
        action_dim: int = len(Action.ALL_ACTIONS),
        seed: int | None = None,
    ) -> None:
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        return int(self.rng.integers(self.action_dim))


class StayAgent:
    def __init__(self, action_dim: int = len(Action.ALL_ACTIONS)) -> None:
        self.action_dim = int(action_dim)
        self.stay_index = Action.ACTION_TO_INDEX[Action.STAY]

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        return int(self.stay_index)
