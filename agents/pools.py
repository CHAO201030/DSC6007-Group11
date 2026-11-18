# work_c/partners/fcp_pool/fcp_loader.py
"""Fictitious Co-Play 伙伴池。"""

import random


class FCPPool:
    def __init__(self, ckpt_paths):
        self.paths = [str(p) for p in ckpt_paths]
        if not self.paths:
            raise ValueError(
                "FCPPool 为空。检查目录：旧版 run_all_experiments.py 扫的是 "
                "self_play_snapshots/，但 step checkpoint 实际在 "
                "self_play_snapshots_old/。"
            )

    def __len__(self):
        return len(self.paths)

    def sample(self, n, seed):
        """均匀无放回采样，结果与 seed 绑定。"""
        rng = random.Random(int(seed))
        return rng.sample(self.paths, min(int(n), len(self.paths)))