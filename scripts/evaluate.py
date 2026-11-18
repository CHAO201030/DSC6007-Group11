"""Evaluate a trained PPO policy against a partner agent.

Examples:
    python -m scripts.evaluate --episodes 2
    python -m scripts.evaluate --checkpoint checkpoints/ppo_clip_10000.pth
    python -m scripts.evaluate --config config/eval.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from agents import RandomAgent, load_ppo_agent, make_agent
from envs import make_env

DEFAULT_CONFIG: dict[str, Any] = {
    "env": {"layout": "cramped_room", "horizon": 400, "info_level": 0},
    "eval": {
        "episodes": 5,
        "deterministic": True,
        "seed": 0,
        "output": None,
    },
    "partner": {"type": "random", "seed": 1},
    "checkpoint": None,
    "device": "cpu",
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


def run_episode(
    env,
    protagonist,
    partner,
    seed: int,
    deterministic: bool,
) -> dict:
    obs, _ = env.reset(seed=seed)
    protagonist.reset()
    partner.reset()

    sparse = np.zeros(env.num_agents, dtype=np.float64)
    shaped = np.zeros(env.num_agents, dtype=np.float64)
    soups = 0
    length = 0

    while True:
        action0 = int(protagonist.act(obs[0], deterministic=deterministic))
        action1 = int(partner.act(obs[1], deterministic=deterministic))
        obs, _, terminated, truncated, info = env.step((action0, action1))

        step_sparse = np.asarray(
            info["sparse_r_by_agent"], dtype=np.float64
        )
        sparse += step_sparse
        shaped += np.asarray(
            info["shaped_r_by_agent"], dtype=np.float64
        )
        soups += int(step_sparse.sum() > 0)
        length += 1

        if terminated or truncated:
            break

    return {
        "seed": int(seed),
        "return_sparse": float(sparse.sum()),
        "return_shaped": float(shaped.sum()),
        "return_sparse_by_agent": sparse.tolist(),
        "episode_length": int(length),
        "soups": int(soups),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "eval_mode": "deterministic" if deterministic else "stochastic",
    }


def summarize(records: list[dict]) -> dict:
    returns = np.asarray(
        [r["return_sparse"] for r in records], dtype=np.float64
    )
    lengths = np.asarray(
        [r["episode_length"] for r in records], dtype=np.float64
    )
    soups = np.asarray([r["soups"] for r in records], dtype=np.float64)
    n = len(records)

    std = float(returns.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0

    return {
        "episodes": n,
        "mean_return_sparse": float(returns.mean()) if n else 0.0,
        "std_return_sparse": std,
        "sem_return_sparse": sem,
        "mean_episode_length": float(lengths.mean()) if n else 0.0,
        "mean_soups": float(soups.mean()) if n else 0.0,
    }


def run_config(cfg: dict, checkpoint: str | None = None) -> dict:
    env_cfg = cfg.get("env", {})
    eval_cfg = cfg.get("eval", {})
    partner_spec = cfg.get("partner", {"type": "random"})

    deterministic = bool(eval_cfg.get("deterministic", True))
    base_seed = int(eval_cfg.get("seed", 0))
    episodes = int(eval_cfg.get("episodes", 5))
    device = cfg.get("device", "cpu")

    env = make_env(env_cfg)

    checkpoint = checkpoint or cfg.get("checkpoint")
    if checkpoint:
        protagonist = load_ppo_agent(
            checkpoint, device=device, deterministic=deterministic
        )
        protagonist_name = str(checkpoint)
    else:
        protagonist = RandomAgent(seed=base_seed)
        protagonist_name = "random"

    partner = make_agent(
        partner_spec, seed=base_seed + 1, deterministic=deterministic
    )

    records = [
        run_episode(env, protagonist, partner, base_seed + episode, deterministic)
        for episode in range(episodes)
    ]
    for episode, record in enumerate(records):
        record["episode"] = episode
        record["layout"] = env.layout

    result = {
        "protagonist": protagonist_name,
        "partner": partner_spec,
        "layout": env.layout,
        "eval_mode": "deterministic" if deterministic else "stochastic",
        "episodes": records,
        "summary": summarize(records),
    }
    env.close()
    return result


def print_summary(result: dict) -> None:
    summary = result["summary"]
    print(
        f"layout={result['layout']}  protagonist={result['protagonist']}  "
        f"partner={result['partner'].get('type', '?')}  "
        f"mode={result['eval_mode']}"
    )
    print(
        f"episodes={summary['episodes']}  "
        f"mean_sparse_return={summary['mean_return_sparse']:.3f} "
        f"± {summary['sem_return_sparse']:.3f} (sem)  "
        f"mean_soups={summary['mean_soups']:.2f}  "
        f"mean_length={summary['mean_episode_length']:.1f}"
    )


def save_result(result: dict, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="JSON config")
    parser.add_argument(
        "--checkpoint", default=None, help="protagonist PPO checkpoint"
    )
    parser.add_argument("--layout", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--partner",
        choices=["random", "stay", "checkpoint"],
        default=None,
    )
    parser.add_argument("--partner-checkpoint", default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.layout is not None:
        cfg["env"]["layout"] = args.layout
    if args.horizon is not None:
        cfg["env"]["horizon"] = args.horizon
    if args.episodes is not None:
        cfg["eval"]["episodes"] = args.episodes
    if args.seed is not None:
        cfg["eval"]["seed"] = args.seed
    if args.stochastic:
        cfg["eval"]["deterministic"] = False
    if args.partner is not None:
        cfg["partner"]["type"] = args.partner
    if args.partner_checkpoint is not None:
        cfg["partner"] = {
            "type": "checkpoint",
            "path": args.partner_checkpoint,
            "deterministic": cfg["eval"].get("deterministic", True),
        }
    if args.output is not None:
        cfg["eval"]["output"] = args.output

    result = run_config(cfg, checkpoint=args.checkpoint)
    print_summary(result)

    output = cfg["eval"].get("output")
    if output:
        path = save_result(result, output)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
