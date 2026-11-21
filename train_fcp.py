#!/usr/bin/env python
"""FCP 训练：只有 ego 学习，伙伴每局从池里采样。

和 train.py 的唯一差别就是"对手从哪来"，其它全部对齐，保证两条臂可比。
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from algorithms.algo_utils import get_GAE, get_device, load_weights, save_weights
from algorithms.ppo import PPO, PPOParams
from agents.checkpoint import SPPolicy
from agents.pools import FCPPool
from envs.config import EnvConfig
from envs.single_env import OvercookedSingleEnv, joint_index

LOG_INTERVAL = 5_000
CHECKPOINT_INTERVAL = 500_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=8_000_000)
    p.add_argument("--ckpt-interval", type=int, default=500_000)
    p.add_argument("--pool-dir", nargs="+", required=True)
    p.add_argument("--pool-own", action="store_true",
                   help="把本 run 已落盘的快照也加进池")
    p.add_argument("--out-dir", default="checkpoints/fcp")
    p.add_argument("--curve-dir", default="results")
    p.add_argument("--init-ckpt", default=None,
                   help="热启动：只加载权重，不续步数")
    p.add_argument("--self-ckpt", default=None)
    p.add_argument("--self-prob", type=float, default=0.0)
    return p.parse_args()


class PartnerCache:
    """同一个 checkpoint 只加载一次。"""
    def __init__(self):
        self._d = {}

    def get(self, path):
        pol = self._d.get(path)
        if pol is None:
            pol = SPPolicy(path, deterministic=False)   # 训练必须随机
            self._d[path] = pol
        return pol


def pool_paths(args):
    paths = []
    for d in args.pool_dir:
        paths += sorted(Path(d).glob("*.pth"))
    own = Path(args.out_dir) / f"seed{args.seed}"
    if args.pool_own and own.exists():
        paths += sorted(own.glob("*.pth"))
    if not paths:
        raise SystemExit(f"伙伴池为空：{args.pool_dir}")
    return paths

def dump_curve(args, hist):
    """把曲线写成 npz；每个 checkpoint 都刷一次，崩溃也能留下可用数据。"""
    Path(args.curve_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.curve_dir) / f"fcp_seed{args.seed}.npz"
    np.savez(path, **{k: np.asarray(v) for k, v in hist.items()})
    return path


def main():
    args = parse_args()
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = EnvConfig(seed=args.seed)          # 与 SP 同一份 env 配置
    env = OvercookedSingleEnv(cfg)
    obs = env.reset()

    param = PPOParams(
        obs_dim=int(obs.shape[1]),
        hidden_dim=int(obs.shape[1]),        # B 也是 hidden = obs_dim
        action_dim=6,
        lr=1e-3, gamma=0.99, eps=0.2,
        device=get_device(),
        epochs=4, minibatch_size=256,
    )
    agent = PPO(param)

    if args.init_ckpt:
        load_weights(agent, args.init_ckpt)
        print(f"[init] 权重来自 {args.init_ckpt}")

    pool = FCPPool(pool_paths(args))
    cache = PartnerCache()
    print(f"[pool] {len(pool)} 个伙伴")

    episodes = args.total_steps // cfg.horizon
    global_step = 0
    next_ckpt, next_log = args.ckpt_interval, LOG_INTERVAL

    hist = {k: [] for k in (
        "timestep", "episode", "soups_delivered", "soups_total",
        "training_return", "return_sparse", "policy_loss", "value_loss",
        "pool_size",
    )}
    win = dict(soups=0.0, sparse=0.0, ret=0.0, pl=0.0, vl=0.0, n=0)

    for ep in tqdm(range(episodes), desc=f"fcp-seed{args.seed}"):
        if args.self_ckpt and random.random() < args.self_prob:
            partner_path = args.self_ckpt
        else:
            partner_path = pool.sample(1, seed=args.seed * 1_000_000 + ep)[0]
        partner = cache.get(partner_path)

        obs = env.reset()
        partner.reset(ep)                    # 统一接口约定

        obs_l, act_l, lp_l, val_l = [], [], [], []
        rew_l, term_l, trunc_l = [], [], []
        ep_ret, info, done = 0.0, {}, False

        while not done:
            a_ego, logp = agent.get_action(obs[0][None, :])     # (1,F) -> (1,)
            a_ego = int(a_ego[0])
            a_par = int(partner.act(obs, 1))

            next_obs, _team_r, term, trunc, info = env.step(
                joint_index(a_ego, a_par)
            )

             # 团队奖励：与 SP 完全同口径（受控对比要求）
            r = float(_team_r)

            obs_l.append(obs[0].copy())
            act_l.append(a_ego)
            lp_l.append(logp.detach().cpu().numpy().reshape(-1))
            val_l.append(
                agent.get_state_value(obs[0][None, :])
                .squeeze(-1).detach().cpu().numpy()
            )
            rew_l.append(np.array([r], dtype=np.float32))
            term_l.append(False)            # Overcooked 只有截断
            trunc_l.append(bool(trunc))

            ep_ret += r
            obs = next_obs
            done = bool(term or trunc)

        last_v = (agent.get_state_value(obs[0][None, :])
                  .squeeze(-1).detach().cpu().numpy())
        adv, ret = get_GAE(
            {"reward": rew_l, "state_value": val_l,
             "terminated": term_l, "truncated": trunc_l},
            gamma=param.gamma, lam=0.95, last_state_value=last_v,
        )
        pl, vl, _ = agent.update(obs_l, act_l, lp_l, adv, ret)

        global_step += len(obs_l)
        soups = float(info.get("episode_soups", 0.0))
        sparse = float(info.get("episode_sparse", 0.0))
        win["soups"] += soups
        win["sparse"] += sparse
        win["ret"] += ep_ret
        win["pl"] += pl
        win["vl"] += vl
        win["n"] += 1

        if global_step >= next_ckpt:
            save_weights(
                agent, f"{args.out_dir}/seed{args.seed}",
                step=global_step,
                extra={"mode": "fcp", "seed": args.seed,
                       "pool_size": len(pool), "partner": partner_path},
            )
            if args.pool_own:
                pool = FCPPool(pool_paths(args))
            dump_curve(args, hist)
            while next_ckpt <= global_step:
                next_ckpt += args.ckpt_interval

        if global_step >= next_log:
            n = max(win["n"], 1)
            hist["timestep"].append(global_step)
            hist["episode"].append(ep)
            hist["soups_delivered"].append(win["soups"] / n)
            hist["soups_total"].append(win["soups"])
            hist["training_return"].append(win["ret"] / n)
            hist["return_sparse"].append(win["sparse"] / n)
            hist["policy_loss"].append(win["pl"] / n)
            hist["value_loss"].append(win["vl"] / n)
            hist["pool_size"].append(len(pool))
            tqdm.write(f"[{global_step}] soups/ep={win['soups']/n:.2f} "
                       f"pool={len(pool)}")
            for k in win:
                win[k] = 0.0 if k != "n" else 0
            while next_log <= global_step:
                next_log += LOG_INTERVAL

    print(f"[saved] {dump_curve(args, hist)}")


if __name__ == "__main__":
    main()