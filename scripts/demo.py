"""Live GUI demo: protagonist vs partner in Overcooked.

Examples:
    python -m scripts.demo
    python -m scripts.demo \
        --checkpoint checkpoints/train_1m/ppo_clip_1000000.pth \
        --episodes 3
    python -m scripts.demo --checkpoint ... --partner stay --fps 15
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pygame

from agents import RandomAgent, load_ppo_agent, make_agent
from envs import make_env
from GameEnv.state_visualizer import StateVisualizer
from scripts.evaluate import load_config


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--partner",
        choices=["random", "stay", "checkpoint"],
        default=None,
    )
    parser.add_argument("--partner-checkpoint", default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def run_demo(cfg: dict, args: argparse.Namespace) -> None:
    if not has_display():
        raise RuntimeError(
            "no DISPLAY/WAYLAND_DISPLAY; the GUI demo needs WSLg/X11"
        )

    env_cfg = cfg["env"]
    eval_cfg = cfg["eval"]

    deterministic = bool(eval_cfg.get("deterministic", True))
    if args.stochastic:
        deterministic = False
    episodes = int(eval_cfg.get("episodes", 3))
    base_seed = int(eval_cfg.get("seed", 0))
    fps = max(1, int(args.fps))
    device = args.device or cfg.get("device", "cpu")

    env = make_env(env_cfg)

    checkpoint = args.checkpoint or cfg.get("checkpoint")
    if checkpoint:
        protagonist = load_ppo_agent(
            checkpoint, device=device, deterministic=deterministic
        )
        protagonist_name = str(checkpoint)
    else:
        protagonist = RandomAgent(seed=base_seed)
        protagonist_name = "random"

    partner_spec = cfg.get("partner", {"type": "random"})
    partner = make_agent(
        partner_spec, seed=base_seed + 1, deterministic=deterministic
    )

    visualizer = StateVisualizer()
    pygame.init()
    screen = None
    clock = pygame.time.Clock()
    title = f"Overcooked demo - {env.layout}"

    returns = []
    soup_events = []
    running = True

    try:
        for episode in range(episodes):
            if not running:
                break

            seed = base_seed + episode
            obs, _ = env.reset(seed=seed)
            protagonist.reset()
            partner.reset()

            sparse = 0.0
            soups = 0
            steps = 0
            last_sparse = np.zeros(env.num_agents, dtype=np.float64)
            last_shaped = np.zeros(env.num_agents, dtype=np.float64)

            while running:
                hud_data = StateVisualizer.default_hud_data(
                    env.env.state,
                    cumulative_sparse_rewards_by_agent=env.env.game_stats[
                        "cumulative_sparse_rewards_by_agent"
                    ],
                    cumulative_shaped_rewards_by_agent=env.env.game_stats[
                        "cumulative_shaped_rewards_by_agent"
                    ],
                    last_sparse_reward_by_agent=last_sparse,
                    last_shaped_reward_by_agent=last_shaped,
                )
                surface = visualizer.render_state(
                    state=env.env.state,
                    grid=env.mdp.terrain_mtx,
                    hud_data=hud_data,
                )
                if screen is None:
                    screen = pygame.display.set_mode(surface.get_size())
                    pygame.display.set_caption(title)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif (
                        event.type == pygame.KEYDOWN
                        and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                if not running:
                    break

                if surface.get_size() != screen.get_size():
                    surface = pygame.transform.smoothscale(
                        surface, screen.get_size()
                    )
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                clock.tick(fps)

                action0 = int(
                    protagonist.act(obs[0], deterministic=deterministic)
                )
                action1 = int(
                    partner.act(obs[1], deterministic=deterministic)
                )
                obs, _, terminated, truncated, info = env.step(
                    (action0, action1)
                )

                step_sparse_arr = np.asarray(
                    info["sparse_r_by_agent"], dtype=np.float64
                )
                step_shaped_arr = np.asarray(
                    info["shaped_r_by_agent"], dtype=np.float64
                )
                step_sparse = float(step_sparse_arr.sum())
                sparse += step_sparse
                soups += int(step_sparse > 0)
                steps += 1
                last_sparse = step_sparse_arr
                last_shaped = step_shaped_arr

                if terminated or truncated:
                    break

            if running:
                returns.append(sparse)
                soup_events.append(soups)
                print(
                    f"episode={episode} seed={seed} "
                    f"sparse={sparse:.1f} soups={soups} steps={steps}"
                )
    finally:
        pygame.quit()
        env.close()

    if returns:
        print(
            f"summary episodes={len(returns)} "
            f"mean_sparse={sum(returns) / len(returns):.2f} "
            f"mean_soups={sum(soup_events) / len(soup_events):.2f} "
            f"protagonist={protagonist_name} "
            f"partner={partner_spec.get('type', '?')}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.layout is not None:
        cfg["env"]["layout"] = args.layout
    if args.horizon is not None:
        cfg["env"]["horizon"] = args.horizon
    if args.episodes is not None:
        cfg["eval"]["episodes"] = args.episodes
    if args.seed is not None:
        cfg["eval"]["seed"] = args.seed
    if args.partner is not None:
        cfg["partner"]["type"] = args.partner
    if args.partner_checkpoint is not None:
        cfg["partner"] = {
            "type": "checkpoint",
            "path": args.partner_checkpoint,
            "deterministic": cfg["eval"].get("deterministic", True),
        }

    run_demo(cfg, args)


if __name__ == "__main__":
    main()
