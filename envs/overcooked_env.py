"""Minimal multi-agent wrapper around GameEnv.OvercookedEnv.

Contract:
    obs, info = env.reset(seed=0)          # (2, obs_dim) float32
    obs, reward, terminated, truncated, info = env.step((a0, a1))
"""

from __future__ import annotations

import numpy as np
import gymnasium

from GameEnv.actions import Action
from GameEnv.overcooked_env import OvercookedEnv
from GameEnv.overcooked_mdp import OvercookedGridworld


class OvercookedMAEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        layout: str = "cramped_room",
        horizon: int = 400,
        info_level: int = 0,
        seed: int | None = None,
    ) -> None:
        self.layout = layout
        self.horizon = int(horizon)
        self.num_agents = 2
        self.action_dim = len(Action.ALL_ACTIONS)
        self.reward_mode = "shared_scalar"
        self.obs_normalized = False
        self.vectorized = False
        self.auto_reset = False
        self._seed = seed

        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(
            self.mdp, horizon=self.horizon, info_level=info_level
        )
        self.obs_dim = int(self.mdp.get_featurize_state_shape()[0])

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(self.action_dim)

    def _obs(self) -> np.ndarray:
        obs = self.env.featurize_state_mdp(self.env.state)
        return np.stack(
            [np.asarray(o, dtype=np.float32) for o in obs], axis=0
        )

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._seed = int(seed)
        if self._seed is not None:
            np.random.seed(self._seed)
        self.env.reset()
        return self._obs(), {}

    def step(self, actions):
        actions = np.asarray(actions).reshape(-1).astype(np.int64)
        if actions.size != self.num_agents:
            raise ValueError(
                f"expected {self.num_agents} actions, got {actions.size}"
            )
        if np.any(actions < 0) or np.any(actions >= self.action_dim):
            raise ValueError(f"actions must be in [0, {self.action_dim})")

        joint = tuple(Action.INDEX_TO_ACTION[int(a)] for a in actions)
        _, reward, done, info = self.env.step(joint)

        return (
            self._obs(),
            float(reward),
            False,
            bool(done),
            {
                "sparse_r_by_agent": np.asarray(
                    info["sparse_r_by_agent"], dtype=np.float32
                ),
                "shaped_r_by_agent": np.asarray(
                    info["shaped_r_by_agent"], dtype=np.float32
                ),
            },
        )

    def close(self) -> None:
        pass


def make_env(cfg) -> OvercookedMAEnv:
    env_cfg = cfg.get("env", cfg) if isinstance(cfg, dict) else cfg
    return OvercookedMAEnv(
        layout=env_cfg.get("layout", "cramped_room"),
        horizon=env_cfg.get("horizon", 400),
        info_level=env_cfg.get("info_level", 0),
        seed=env_cfg.get("seed"),
    )
