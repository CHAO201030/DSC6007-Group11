# agents/checkpoint.py
"""加载检查点。兼容 obs=(2,96) 的 ndarray 和 dict 两种输入。"""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def _extract_obs(obs, agent_index, obs_dim):
    """从 (2, 96) ndarray 或 dict 中提取本 agent 的输入。"""
    if isinstance(obs, dict):
        a0, a1 = obs["both_agent_obs"]
    else:
        if obs_dim == 192:
            # 两个 agent 看到的拼接顺序必须不同，否则动作完全相同、配合直接崩
            pair = [obs[0], obs[1]] if agent_index == 0 else [obs[1], obs[0]]
            return np.concatenate(pair, axis=-1).astype(np.float32)
        a0, a1 = obs[0], obs[1]
    return np.asarray(a0 if agent_index == 0 else a1, dtype=np.float32)


class SPPolicy:
    def __init__(self, ckpt_path, device="cpu", deterministic=True):
        self.path = str(ckpt_path)
        self.digest = sha256_file(self.path)
        self.name = f"ckpt:{Path(self.path).stem}"
        self.device = device
        self.deterministic = deterministic

        try:
            ckpt = torch.load(self.path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(self.path, map_location=device, weights_only=False)

        meta = ckpt if isinstance(ckpt, dict) else {}

        # ── 兼容 B 的 checkpoint 格式 ──
        # B 的 algo_utils.save_weights 存的是：
        #   {"format_version":1, "step":N, "param":{...},
        #    "actor_state":{...}, "critic_state":{...}, "optimizer_state":{...}}
        # Actor 内部是 self.network，所以键名形如 network.0.weight
        if "actor_state" in meta:
            param = meta.get("param") or {}
            self.obs_dim = int(param.get("obs_dim", 96))
            hidden = int(param.get("hidden_dim", 128))
            n_actions = int(param.get("action_dim", 6))

            sd_b = {}
            for k, v in meta["actor_state"].items():
                kk = k[len("network."):] if k.startswith("network.") else k
                if not kk.startswith("net."):
                    kk = "net." + kk
                sd_b[kk] = v

            self.net = PolicyNet(self.obs_dim, n_actions, hidden).to(device)
            missing, unexpected = self.net.load_state_dict(sd_b, strict=False)
            if missing:
                raise RuntimeError(
                    f"{self.path}: 有 {len(missing)} 个权重没加载上 {missing}"
                )
            if unexpected:
                print(f"[warn] {Path(self.path).name}: 忽略了多余的键 {unexpected}")
            self.net.eval()
            self.global_step = int(meta.get("step", -1))
            return

        raw = meta.get("state_dict", ckpt)

        # 过滤 actor. / policy. 前缀，丢掉 critic
        sd = {}
        for k, v in raw.items():
            kk = k
            for pref in ("actor.", "policy.", "pi."):
                if kk.startswith(pref):
                    kk = kk[len(pref):]
                    break
            low = kk.lower()
            if "critic" in low or "value" in low:
                continue
            if not kk.startswith("net."):
                kk = "net." + kk
            sd[kk] = v

        first = next((k for k in sd if k.endswith(".weight")), None)
        if first is None:
            raise RuntimeError(f"{self.path}: state_dict 里找不到 Linear 权重")

        self.obs_dim = int(meta.get("obs_dim") or sd[first].shape[1])
        hidden = int(meta.get("hidden") or sd[first].shape[0])
        n_actions = int(meta.get("n_actions") or 6)

        self.net = PolicyNet(self.obs_dim, n_actions, hidden).to(device)
        missing, unexpected = self.net.load_state_dict(sd, strict=False)

        # ★ 关键：缺键必须炸，不能静默变随机策略
        if missing:
            raise RuntimeError(
                f"{self.path}: 有 {len(missing)} 个权重没加载上 {missing}。"
                "网络结构和 checkpoint 不匹配。"
            )
        if unexpected:
            print(f"[warn] {Path(self.path).name}: 忽略了多余的键 {unexpected}")

        self.net.eval()
        self.global_step = int(meta.get("global_step", -1))

    # --- Policy 协议 ---
    def reset(self, seed=None):
        pass       # 确定性策略，无内部随机状态

    def act(self, obs, agent_index):
        o = _extract_obs(obs, agent_index, self.obs_dim)
        with torch.no_grad():
            logits = self.net(torch.as_tensor(o).unsqueeze(0))
        if self.deterministic:
            return int(logits.argmax(dim=-1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())

    def selftest(self):
        """加载后必须调用：确认权重真的进来了。"""
        a = self.act(np.zeros((2, 96), dtype=np.float32), 0)
        if not 0 <= a < 6:
            raise RuntimeError(f"{self.path}: 自检产出非法动作 {a}")
        return a