# -*- coding: utf-8 -*-
"""冒烟测试：随机策略跑 400 步，验证环境能跑、一局结束能拿到上菜数。
直接运行：python smoke_rollout.py
随机策略在 cramped_room 上经常一局 0 盘汤，这是正常的，不是 bug。
"""
import numpy as np

from envs import EnvConfig, OvercookedVecEnv

vec = OvercookedVecEnv(EnvConfig(seed=0, num_envs=2, horizon=400))
obs = vec.reset()
print(f"初始观测形状: {obs.shape}")  # (2, 2, 96)

rng = np.random.default_rng(0)
for t in range(400):
    obs, rew, term, trunc, infos = vec.step(rng.integers(0, 36, size=2))
    for i, done in enumerate(trunc):
        if done:
            print(f"env{i} 一局结束: "
                  f"累计 sparse={infos[i]['episode_sparse']:.0f} "
                  f"→ soups={infos[i]['episode_soups']:.1f} "
                  f"(步数={infos[i]['episode_length']})")

print("冒烟测试完成：400 步无报错 ✔")
