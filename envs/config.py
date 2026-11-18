# -*- coding: utf-8 -*-
"""实验配置：用哪张地图、多长一局、几个并行环境、奖励怎么给。

改实验只改这里，代码其它部分不动。
训练(B)和评估(C)都从同一个配置构造环境，保证口径一致。
"""
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # ---- 地图与局 ----
    layout_name: str = "cramped_room"   # 用哪张地图（cramped_room / asymmetric_advantages / ...）
    horizon: int = 400                  # 一局最多多少步（到点算"截断"，不是"真正结束"）

    # ---- 并行与种子 ----
    num_envs: int = 4                   # 同时开几个环境采样
    seed: int = 0                       # 随机种子：同种子跑两次，结果必须完全一致

    # ---- 奖励 ----
    use_reward_shaping: bool = True     # 训练时是否加塑形奖励（评估时应关掉，只看上菜分）
    rew_shaping_params: dict = field(default_factory=lambda: {
        "PLACEMENT_IN_POT_REW": 3,      # 把菜放进锅
        "DISH_PICKUP_REWARD": 3,        # 拿起盘子
        "SOUP_PICKUP_REWARD": 5,        # 用盘子盛到汤
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    })
