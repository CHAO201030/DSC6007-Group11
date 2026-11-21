# scripts/eval/gameenv_wrapper.py
"""构造评估用环境。"""

import os
import sys
from pathlib import Path

if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium
import numpy as np

from GameEnv.overcooked_env import OvercookedEnv
from GameEnv.overcooked_mdp import OvercookedGridworld


class OvercookedWrapper:
    """reset -> obs (2,96); step(joint) -> obs, reward, done, info"""

    def __init__(self, layout_name="cramped_room", horizon=400, info_level=0,
                 start_state="standard", rnd_obj_prob_thresh=0.0):
        self.layout_name = layout_name
        self.horizon = horizon
        self.start_state = start_state

        mdp = OvercookedGridworld.from_layout_name(layout_name)
        start_fn = None
        if start_state != "standard":
            thresh = 0.0 if start_state == "random_pos" else rnd_obj_prob_thresh
            start_fn = mdp.get_random_start_state_fn(
                random_start_pos=True, rnd_obj_prob_thresh=thresh
            )

        base_env = OvercookedEnv.from_mdp(
            mdp, start_state_fn=start_fn, horizon=horizon, info_level=info_level
        )
        self.env = gymnasium.make(
            "Overcooked-v0",
            base_env=base_env,
            featurize_fn=base_env.featurize_state_mdp,
            disable_env_checker=True,
        )

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        base = self.env.unwrapped
        base.agent_idx = 0                                  # ★ 钉死座位
        o_p0, o_p1 = base.featurize_fn(base.base_env.state)  # 按玩家顺序重建
        return np.stack([o_p0, o_p1]).astype(np.float32)

    def step(self, joint_action):
        obs, reward, terminated, truncated, info = self.env.step(joint_action)
        o_p0, o_p1 = obs["both_agent_obs"]      # agent_idx 已是 0 -> 玩家顺序
        return (np.stack([o_p0, o_p1]).astype(np.float32),
                float(reward),
                bool(terminated or truncated),
                info)