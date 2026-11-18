# -*- coding: utf-8 -*-
"""多环境并行采样（同步版）。

对训练端暴露统一接口：
    obs = vec.reset()                  -> (num_envs, 2, F)
    obs, rew, term, trunc, infos = vec.step(actions)   # actions: 长度 num_envs 的联合动作编号

说明：
- 目前是同步版（一个 for 循环轮流 step）。Overcooked 是 CPU 小游戏，
  采样开销远小于 PPO 更新，同步版足够快，而且最容易保证可复现。
- 某个环境到 horizon 被截断后会自动重开一局，这一帧的真实末态观测
  存在 infos[i]["terminal_observation"]，供 PPO 做 bootstrap。
- 整局统计（episode_sparse / episode_soups / episode_length）由
  单环境层统一输出，本层只做透传；B/C 只读这三个键，
  上游 GameEnv 的 info["episode"] 仅作备份。
"""
import numpy as np

from .single_env import OvercookedSingleEnv, NUM_ACTIONS, NUM_JOINT_ACTIONS


class OvercookedVecEnv:
    def __init__(self, config):
        self.cfg = config
        self.envs = [
            OvercookedSingleEnv(config, env_seed_offset=i)
            for i in range(config.num_envs)
        ]
        self.num_envs = config.num_envs
        self.obs_dim = self.envs[0].obs_dim
        self.num_agents = 2
        self.num_actions = NUM_ACTIONS               # 单个 agent 的动作数
        self.num_joint_actions = NUM_JOINT_ACTIONS   # 联合动作数

    def reset(self) -> np.ndarray:
        """shape (num_envs, 2, F)"""
        return np.stack([e.reset() for e in self.envs])

    def step(self, joint_actions):
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for env, a in zip(self.envs, joint_actions):
            obs, rew, term, trunc, info = env.step(int(a))
            if trunc or term:
                # 记录被截断那一帧的真实观测，然后自动开新局
                # （整局统计三个键由 single_env 在截断时已写入 info，直接透传）
                info["terminal_observation"] = obs
                obs = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
        return (
            np.stack(obs_list),                       # (num_envs, 2, F)
            np.asarray(rew_list, dtype=np.float32),   # (num_envs,)
            np.asarray(term_list, dtype=bool),        # (num_envs,) 恒为 False
            np.asarray(trunc_list, dtype=bool),       # (num_envs,)
            info_list,                                # list[dict]
        )
