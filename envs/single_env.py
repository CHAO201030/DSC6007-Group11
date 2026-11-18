# -*- coding: utf-8 -*-
"""单环境包装：把 Overcooked 的复杂性挡在内部，对外只暴露
观测（定长数组，两个 agent 各一份）、联合动作（36 选 1）、奖励、终止/截断标记。

约定（A 层定稿）：一局结束（truncated）时，info 里统一给三个整局统计键：
    episode_sparse  本局累计 sparse 奖励
    episode_soups   本局上菜数（sparse / 20）
    episode_length  本局步数
上游 GameEnv 自带的 info["episode"] 保留不动、仅作备份；B/C 只读上面三个键。
"""
import numpy as np

# 项目里 vendored 的 GameEnv 包（压平结构：GameEnv/overcooked_mdp.py）
from GameEnv.overcooked_mdp import OvercookedGridworld
from GameEnv.overcooked_env import OvercookedEnv
try:
    from GameEnv.overcooked_mdp import Action
except ImportError:  # 某些版本 Action 单独放在 actions.py 里
    from GameEnv.actions import Action

# 6 个离散动作：上、下、右、左、不动、交互
NUM_ACTIONS = 6
ACTION_MEANINGS = ["up", "down", "right", "left", "stay", "interact"]
# 联合动作：两个 agent 的动作组合成一个 0~35 的整数
NUM_JOINT_ACTIONS = NUM_ACTIONS * NUM_ACTIONS  # 36

SPARSE_PER_SOUP = 20.0  # Overcooked 每上一盘汤的 sparse 奖励


def joint_index(a0: int, a1: int) -> int:
    """两个 agent 各自的动作(0~5) -> 联合动作编号(0~35)"""
    return a0 * NUM_ACTIONS + a1


def split_joint(j: int) -> tuple:
    """联合动作编号(0~35) -> (agent0 动作, agent1 动作)"""
    return j // NUM_ACTIONS, j % NUM_ACTIONS


class OvercookedSingleEnv:
    """一个 Overcooked 环境实例。

    对外接口：
        reset()              -> obs, shape (2, F)，两个 agent 各一份定长观测
        step(joint_action)   -> obs, reward, terminated, truncated, info
    注意：Overcooked 没有"真正结束"，done 只可能因为时间上限(horizon)到了，
    所以 terminated 永远是 False，truncated=True 表示被时间上限截断。
    训练 PPO 时 truncated 的步必须 bootstrap 价值，不能当成 game over。
    """

    def __init__(self, config, env_seed_offset: int = 0):
        self.cfg = config
        self.seed = config.seed + env_seed_offset
        self._rng = np.random.default_rng(self.seed)

        shaping = dict(config.rew_shaping_params) if config.use_reward_shaping else None
        self.mdp = OvercookedGridworld.from_layout_name(
            config.layout_name, rew_shaping_params=shaping
        )
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=config.horizon, info_level=0)

        self._ep_sparse = 0.0  # 本局累计 sparse 奖励计数器
        obs = self.reset()
        self.obs_dim = int(obs.shape[1])   # F：观测长度，由地图决定，固定不变
        self.num_agents = 2

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        self.env.reset()
        self.timestep = 0
        self._ep_sparse = 0.0
        return self._obs()

    def step(self, joint_action: int):
        a0, a1 = split_joint(int(joint_action))
        action_pair = (Action.INDEX_TO_ACTION[a0], Action.INDEX_TO_ACTION[a1])

        _next_state, reward, done, info = self.env.step(action_pair)
        self.timestep += 1

        # Overcooked 里 done 只会因为 horizon 到了 -> 属于"截断"而非"结束"
        truncated = bool(done)
        terminated = False

        obs = self._obs()
        info = dict(info) if isinstance(info, dict) else {}
        info["timestep"] = self.timestep

        # 累计本局 sparse 奖励（两个 agent 的 sparse 相加）
        sparse = info.get("sparse_r_by_agent", None)
        if sparse is not None:
            self._ep_sparse += float(np.sum(sparse))

        # 一局结束：输出整局统计（单环境和 vec 层口径统一，都用这三个键）
        if truncated or terminated:
            info["episode_sparse"] = float(self._ep_sparse)
            info["episode_soups"] = self._ep_sparse / SPARSE_PER_SOUP
            info["episode_length"] = self.timestep

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        """两个 agent 各一份的定长观测，shape (2, F)。
        训练和评估都走这同一个函数，口径天然一致。"""
        state = self.env.state
        if hasattr(self.env, "featurize_state_mdp"):
            feat = self.env.featurize_state_mdp(state)
        else:  # 某些 vendored 版本没有该方法，直接调 mdp 的编码
            feat = self.mdp.lossless_state_encoding(state)
        return np.stack([np.asarray(f, dtype=np.float32).ravel() for f in feat])
