# agents/checkpoint.py
"""加载检查点。兼容 obs=(2,96) 的 ndarray 和 dict 两种输入。"""

import hashlib
from pathlib import Path

import numpy as np
import torch

from algorithms.ppo import load_policy


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _extract_obs(obs, agent_index, obs_dim):
    """从 (2, 96) ndarray 或 dict 中提取本 agent 的输入。"""
    if isinstance(obs, dict):
        a0, a1 = obs["both_agent_obs"]
    else:
        if obs_dim == 192:
            return np.concatenate([obs[0], obs[1]], axis=-1).astype(np.float32)
        a0, a1 = obs[0], obs[1]
    return np.asarray(a0 if agent_index == 0 else a1, dtype=np.float32)


class SPPolicy:
    """PPO checkpoint 伙伴，直接复用 algorithms.ppo.load_policy。"""

    def __init__(self, ckpt_path, device="cpu", deterministic=True):
        self.path = str(ckpt_path)
        self.digest = sha256_file(self.path)
        self.name = f"ckpt:{Path(self.path).stem}"
        self.device = device
        self.deterministic = deterministic

        self.model = load_policy(self.path, device=device)
        self.obs_dim = int(self.model.param.obs_dim)
        self.global_step = int(getattr(self.model, "global_step", -1))

    # --- Policy 协议 ---
    def reset(self, seed=None):
        pass  # 策略本身无内部随机状态

    def act(self, obs, agent_index):
        o = _extract_obs(obs, agent_index, self.obs_dim)
        with torch.no_grad():
            action, _ = self.model.get_action(
                torch.as_tensor(o, dtype=torch.float32).unsqueeze(0),
                deterministic=self.deterministic,
            )
        return int(action[0])

    def selftest(self):
        """加载后必须调用：确认权重真的进来了。"""
        a = self.act(
            np.zeros((2, self.obs_dim), dtype=np.float32), 0
        )
        if not 0 <= a < 6:
            raise RuntimeError(f"{self.path}: 自检产出非法动作 {a}")
        return a
