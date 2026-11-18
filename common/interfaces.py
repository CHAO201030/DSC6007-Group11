"""Shared protocols between environment, algorithms and agents."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int: ...

    def reset(self) -> None: ...


class Partner(Policy, Protocol):
    pass
