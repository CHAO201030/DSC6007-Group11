"""Minimal PPO training loop.

Protagonist controls player 0; partner controls player 1.
One episode = one rollout = one PPO update.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from algorithms.algo_utils import get_GAE, save_weights
from algorithms.ppo import PPO, PPOParams
from agents import make_partner
from envs import make_env
from scripts.plot_training import plot_training

DEFAULT_CONFIG: dict[str, Any] = {
    "env": {"layout": "cramped_room", "horizon": 400, "info_level": 0},
    "ppo": {
        "hidden_dim": 64,
        "lr": 0.001,
        "gamma": 0.99,
        "eps": 0.2,
        "device": "cpu",
        "value_coef": 0.5,
        "epochs": 4,
        "minibatch_size": 64,
        "max_grad_norm": 0.5,
    },
    "train": {
        "episodes": 500,
        "total_steps": 0,
        "seed": 0,
        "gae_lambda": 0.95,
        "shaping_coef": 1.0,
        "checkpoint_interval": 50,
        "checkpoint_dir": "checkpoints/train",
        "log_path": "checkpoints/train/train_log.jsonl",
        "plot_path": "checkpoints/train/training_curves.png",
    },
    "partner": {"type": "random", "seed": 1},
}


def deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            deep_update(cfg, json.load(handle))
    return cfg


def build_ppo(env, cfg: dict, device_override: str | None = None) -> PPO:
    ppo_cfg = dict(cfg.get("ppo", {}))
    ppo_cfg.pop("obs_dim", None)
    ppo_cfg.pop("action_dim", None)
    if device_override is not None:
        ppo_cfg["device"] = device_override
    return PPO(
        PPOParams(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            **ppo_cfg,
        )
    )


def run_training(cfg: dict, args: argparse.Namespace) -> Path:
    env_cfg = cfg["env"]
    train_cfg = cfg["train"]
    ppo_cfg = cfg["ppo"]

    env = make_env(env_cfg)
    ppo = build_ppo(env, cfg, device_override=args.device)

    total_steps = int(train_cfg.get("total_steps", 0) or 0)
    if total_steps > 0:
        episodes = (total_steps + env.horizon - 1) // env.horizon
    else:
        episodes = int(train_cfg.get("episodes", 0))
    base_seed = int(train_cfg["seed"])
    gamma = float(ppo_cfg["gamma"])
    lam = float(train_cfg.get("gae_lambda", 0.95))
    shaping_coef = float(train_cfg.get("shaping_coef", 0.0))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 0))

    checkpoint_dir = Path(
        args.checkpoint_dir or train_cfg["checkpoint_dir"]
    )
    log_path = Path(args.log_path or train_cfg["log_path"])
    plot_path = Path(train_cfg.get("plot_path", "checkpoints/train/curves.png"))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-seed{base_seed}"
    with open(checkpoint_dir / f"config_{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    partner = make_partner(
        cfg.get("partner"), seed=base_seed + 1, deterministic=False
    )

    log_file = open(log_path, "a", encoding="utf-8")
    global_step = 0
    saved_paths: list[Path] = []

    try:
        for episode in range(episodes):
            started = time.perf_counter()
            seed = base_seed + episode
            obs, _ = env.reset(seed=seed)
            partner.reset()

            obs_list: list[np.ndarray] = []
            action_list: list[int] = []
            log_prob_list: list[float] = []
            value_list: list[float] = []
            reward_list: list[float] = []
            terminated_list: list[bool] = []
            truncated_list: list[bool] = []

            episode_sparse = 0.0
            episode_shaped = 0.0
            protagonist_sparse = 0.0
            soups = 0
            length = 0

            while True:
                obs0 = obs[0:1]
                action, log_prob = ppo.get_action(obs0)
                value = float(
                    ppo.get_state_value(obs0).squeeze(-1).item()
                )

                partner_action = partner.act(
                    obs[1], deterministic=False
                )
                next_obs, reward, terminated, truncated, info = env.step(
                    (int(action[0]), int(partner_action))
                )

                step_sparse = np.asarray(
                    info["sparse_r_by_agent"], dtype=np.float64
                )
                step_shaped = np.asarray(
                    info["shaped_r_by_agent"], dtype=np.float64
                )

                training_reward = float(reward) + (
                    shaping_coef * float(step_shaped[0])
                )

                obs_list.append(obs[0].copy())
                action_list.append(int(action[0]))
                log_prob_list.append(float(log_prob[0]))
                value_list.append(value)
                reward_list.append(training_reward)
                terminated_list.append(bool(terminated))
                truncated_list.append(bool(truncated))

                episode_sparse += float(step_sparse.sum())
                episode_shaped += float(step_shaped.sum())
                protagonist_sparse += float(step_sparse[0])
                soups += int(step_sparse.sum() > 0)
                length += 1
                obs = next_obs

                if terminated or truncated:
                    break

            last_value = float(
                ppo.get_state_value(obs[0:1]).squeeze(-1).item()
            )
            advantages, returns = get_GAE(
                {
                    "reward": reward_list,
                    "state_value": value_list,
                    "terminated": terminated_list,
                    "truncated": truncated_list,
                },
                gamma=gamma,
                lam=lam,
                last_state_value=last_value,
            )

            policy_loss, value_loss, clip_fraction = ppo.update(
                obs_list,
                action_list,
                log_prob_list,
                advantages,
                returns,
            )

            global_step += length
            seconds = time.perf_counter() - started
            record = {
                "run_id": run_id,
                "episode": episode,
                "global_step": global_step,
                "seed": seed,
                "return_sparse": episode_sparse,
                "return_shaped": episode_shaped,
                "protagonist_sparse": protagonist_sparse,
                "episode_length": length,
                "soups": soups,
                "policy_loss": float(policy_loss),
                "value_loss": float(value_loss),
                "clip_fraction": float(clip_fraction),
                "seconds": seconds,
                "shaping_coef": shaping_coef,
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()

            print(
                f"ep={episode:4d} step={global_step:7d} "
                f"sparse={episode_sparse:6.1f} soups={soups:2d} "
                f"len={length:3d} "
                f"pi={policy_loss:+.4f} v={value_loss:.4f} "
                f"clip={clip_fraction:.3f} t={seconds:.2f}s"
            )

            if (
                checkpoint_interval > 0
                and (episode + 1) % checkpoint_interval == 0
            ):
                saved_paths.append(
                    save_weights(
                        ppo,
                        checkpoint_dir,
                        step=global_step,
                        extra={
                            "run_id": run_id,
                            "episode": episode,
                            "config": cfg,
                        },
                    )
                )

        final_path = save_weights(
            ppo,
            checkpoint_dir,
            step=global_step,
            extra={"run_id": run_id, "episode": episodes - 1, "config": cfg},
        )
        saved_paths.append(final_path)
        print(f"saved checkpoint: {final_path}")
    finally:
        log_file.close()
        env.close()

    if args.plot:
        plot_training(log_path, plot_path)

    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.json")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--shaping-coef", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--no-plot", action="store_false", dest="plot", default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["train"]["episodes"] = args.episodes
    if args.total_steps is not None:
        cfg["train"]["total_steps"] = args.total_steps
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.layout is not None:
        cfg["env"]["layout"] = args.layout
    if args.device is not None:
        cfg["ppo"]["device"] = args.device
    if args.shaping_coef is not None:
        cfg["train"]["shaping_coef"] = args.shaping_coef

    log_path = run_training(cfg, args)
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
