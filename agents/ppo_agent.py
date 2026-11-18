"""Load a trained PPO checkpoint and expose the Agent.act interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from algorithms.algo_utils import load_weights
from algorithms.ppo import PPO, PPOParams


class PPOAgent:
    def __init__(self, model: PPO, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def reset(self) -> None:
        pass

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool | None = None,
    ) -> int:
        det = self.deterministic if deterministic is None else deterministic
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        action, _ = self.model.get_action(obs_t, deterministic=det)
        return int(action[0])


def load_ppo_agent(
    path: str | Path,
    device: str = "cpu",
    deterministic: bool = True,
) -> PPOAgent:
    checkpoint = torch.load(
        Path(path), map_location="cpu", weights_only=False
    )
    param_dict = dict(checkpoint["param"])
    param_dict["device"] = device
    model = PPO(PPOParams(**param_dict))
    load_weights(model, path)
    return PPOAgent(model, deterministic=deterministic)
