# work_c/partners/scripted_bots/bots.py
"""固定规则的脚本 bot —— 评估下界。"""

import numpy as np

N_ACTIONS = 6
STAY = 4          # Action.STAY 的索引


class StayBot:
    """原地不动。"""

    name = "scripted:stay"

    def reset(self, seed=None):
        pass

    def act(self, obs, agent_index):
        return STAY


class RandomBot:
    """每步随机，每局重新播种。"""

    name = "scripted:random"

    def __init__(self, seed=0):
        self.reset(seed)

    def reset(self, seed=None):
        self.rng = np.random.RandomState(0 if seed is None else int(seed))

    def act(self, obs, agent_index):
        return int(self.rng.randint(N_ACTIONS))