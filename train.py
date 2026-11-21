import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from algorithms.ppo import PPOParams, PPO
from algorithms.algo_utils import save_weights, get_device, get_GAE

from envs.single_env import OvercookedSingleEnv, joint_index, NUM_ACTIONS
from envs.vector_env import OvercookedVecEnv
from envs.config import EnvConfig

DEFAULT_TOTAL_STEPS = 2_000_000
LOG_INTERVAL = 5000
CHECKPOINT_INTERVAL = 500_000


def get_joint_action(action) -> int:
    return joint_index(action[0].item(), action[1].item())


def _new_history():
    return {
        "timestep": [],
        "episode": [],
        "soups_delivered": [],
        "soups_total": [],
        "training_return": [],
        "return_sparse": [],
        "policy_loss": [],
        "value_loss": [],
    }


def _save_history(history, config, suffix=""):
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    curve_path = results_dir / f"train_seed{config.seed}{suffix}.npz"
    np.savez(
        curve_path,
        **{key: np.asarray(value) for key, value in history.items()},
    )
    print(f"saved curve: {curve_path}")
    return curve_path


def _save_checkpoint(
    agent,
    config,
    mode,
    step,
    episodes,
    total_steps,
    curve_path,
    extra=None,
):
    payload = {
        "seed": config.seed,
        "episodes": episodes,
        "total_steps": total_steps,
        "curve": str(curve_path),
        "mode": mode,
    }
    if extra:
        payload.update(extra)
    checkpoint = save_weights(
        agent,
        f"checkpoints/ppo_{mode}_env/seed{config.seed}",
        step=step,
        extra=payload,
    )
    print(f"saved checkpoint: {checkpoint}")
    return checkpoint


def train_single_env(
    seed: int = 42,
    total_steps: int = DEFAULT_TOTAL_STEPS,
):
    config = EnvConfig(seed=seed)
    env = OvercookedSingleEnv(config)
    obs = env.reset()
    episodes = (total_steps + config.horizon - 1) // config.horizon

    param = PPOParams(
        obs_dim=obs.shape[1],
        hidden_dim=obs.shape[1],
        action_dim=NUM_ACTIONS,
        lr=1e-3,
        gamma=0.99,
        eps=0.2,
        device=get_device(),
        epochs=4,
        minibatch_size=256,
    )

    agent = PPO(param)
    global_step = 0
    last_sparse = 0.0
    last_soups = 0.0

    history = _new_history()
    curve_path = Path("results") / f"train_seed{config.seed}.npz"

    next_log_step = LOG_INTERVAL
    next_checkpoint_step = CHECKPOINT_INTERVAL
    last_checkpoint_step = 0
    window_soups = 0.0
    window_sparse = 0.0
    window_return = 0.0
    window_policy_loss = 0.0
    window_value_loss = 0.0
    window_episodes = 0

    for episode in tqdm(
        range(episodes), desc=f"single-seed{seed}"
    ):
        obs = env.reset()
        done = False

        obs_list = []
        action_list = []
        log_prob_list = []
        value_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []

        episode_reward_sum = 0.0

        while not done:
            action, log_prob = agent.get_action(obs)
            value = agent.get_state_value(obs).squeeze(-1)

            joint = get_joint_action(action)
            next_obs, reward, terminated, truncated, info = env.step(joint)

            obs_list.append(obs.copy())
            action_list.append(action.detach().cpu().numpy())
            log_prob_list.append(log_prob.detach().cpu().numpy())
            value_list.append(value.detach().cpu().numpy())
            reward_list.append(np.full(2, float(reward), dtype=np.float32))
            terminated_list.append(bool(terminated))
            truncated_list.append(bool(truncated))

            episode_reward_sum += float(reward)
            obs = next_obs
            done = terminated or truncated

        last_value = (
            agent.get_state_value(obs).squeeze(-1).detach().cpu().numpy()
        )
        advantages, returns = get_GAE(
            {
                "reward": reward_list,
                "state_value": value_list,
                "terminated": terminated_list,
                "truncated": truncated_list,
            },
            gamma=param.gamma,
            lam=0.95,
            last_state_value=last_value,
        )

        policy_loss, value_loss, _ = agent.update(
            obs_list,
            action_list,
            log_prob_list,
            advantages,
            returns,
        )

        global_step += len(obs_list)
        if global_step >= next_checkpoint_step:
            _save_checkpoint(
                agent,
                config,
                "single",
                global_step,
                episodes,
                total_steps,
                curve_path,
            )
            last_checkpoint_step = global_step
            while next_checkpoint_step <= global_step:
                next_checkpoint_step += CHECKPOINT_INTERVAL

        last_sparse = float(info.get("episode_sparse", 0.0))
        last_soups = float(info.get("episode_soups", 0.0))

        window_soups += last_soups
        window_sparse += last_sparse
        window_return += episode_reward_sum
        window_policy_loss += policy_loss
        window_value_loss += value_loss
        window_episodes += 1

        if global_step >= next_log_step:
            history["timestep"].append(global_step)
            history["episode"].append(episode)
            history["soups_delivered"].append(
                window_soups / window_episodes
            )
            history["soups_total"].append(window_soups)
            history["training_return"].append(
                window_return / window_episodes
            )
            history["return_sparse"].append(
                window_sparse / window_episodes
            )
            history["policy_loss"].append(
                window_policy_loss / window_episodes
            )
            history["value_loss"].append(
                window_value_loss / window_episodes
            )

            window_soups = 0.0
            window_sparse = 0.0
            window_return = 0.0
            window_policy_loss = 0.0
            window_value_loss = 0.0
            window_episodes = 0
            next_log_step += LOG_INTERVAL

        tqdm.write(
            f"episode={episode} step={global_step} "
            f"reward={episode_reward_sum:.1f} "
            f"sparse={last_sparse:.1f} soups={last_soups:.2f}"
        )

    if window_episodes > 0:
        history["timestep"].append(global_step)
        history["episode"].append(episodes - 1)
        history["soups_delivered"].append(window_soups / window_episodes)
        history["soups_total"].append(window_soups)
        history["training_return"].append(window_return / window_episodes)
        history["return_sparse"].append(window_sparse / window_episodes)
        history["policy_loss"].append(window_policy_loss / window_episodes)
        history["value_loss"].append(window_value_loss / window_episodes)

    curve_path = _save_history(history, config)

    if global_step != last_checkpoint_step:
        _save_checkpoint(
            agent,
            config,
            "single",
            global_step,
            episodes,
            total_steps,
            curve_path,
        )


def train_vec_env(
    seed: int = 42,
    total_steps: int = DEFAULT_TOTAL_STEPS,
):
    config = EnvConfig(seed=seed)
    env = OvercookedVecEnv(config)
    obs = env.reset()
    num_envs = env.num_envs
    total_episodes = (
        total_steps + config.horizon - 1
    ) // config.horizon

    param = PPOParams(
        obs_dim=obs.shape[-1],
        hidden_dim=obs.shape[-1],
        action_dim=NUM_ACTIONS,
        lr=1e-3,
        gamma=0.99,
        eps=0.2,
        device=get_device(),
        epochs=4,
        minibatch_size=256,
    )

    agent = PPO(param)
    global_step = 0
    episodes_done = 0

    history = _new_history()
    curve_path = Path("results") / f"train_seed{config.seed}_vec.npz"

    next_log_step = LOG_INTERVAL
    next_checkpoint_step = CHECKPOINT_INTERVAL
    last_checkpoint_step = 0
    window_soups = 0.0
    window_sparse = 0.0
    window_return = 0.0
    window_policy_loss = 0.0
    window_value_loss = 0.0
    window_episodes = 0

    rounds = (total_episodes + num_envs - 1) // num_envs

    for round_idx in tqdm(range(rounds), desc="train-vec"):
        obs = env.reset()

        obs_list = [[] for _ in range(num_envs)]
        action_list = [[] for _ in range(num_envs)]
        log_prob_list = [[] for _ in range(num_envs)]
        value_list = [[] for _ in range(num_envs)]
        reward_list = [[] for _ in range(num_envs)]
        terminated_list = [[] for _ in range(num_envs)]
        truncated_list = [[] for _ in range(num_envs)]

        episode_reward_sum = np.zeros(num_envs, dtype=np.float64)
        done_mask = np.zeros(num_envs, dtype=bool)
        terminal_obs = [None] * num_envs
        episode_info = [None] * num_envs

        while not done_mask.all():
            flat_obs = obs.reshape(num_envs * 2, -1)
            actions, log_probs = agent.get_action(flat_obs)
            values = agent.get_state_value(flat_obs).squeeze(-1)

            actions = actions.reshape(num_envs, 2)
            log_probs = log_probs.reshape(num_envs, 2)
            values = values.reshape(num_envs, 2)

            joint_actions = [
                get_joint_action(actions[i]) for i in range(num_envs)
            ]
            next_obs, rewards, terms, truncs, infos = env.step(
                joint_actions
            )

            for i in range(num_envs):
                if done_mask[i]:
                    continue

                obs_list[i].append(obs[i].copy())
                action_list[i].append(
                    actions[i].detach().cpu().numpy()
                )
                log_prob_list[i].append(
                    log_probs[i].detach().cpu().numpy()
                )
                value_list[i].append(
                    values[i].detach().cpu().numpy()
                )
                reward_list[i].append(
                    np.full(2, float(rewards[i]), dtype=np.float32)
                )
                terminated_list[i].append(bool(terms[i]))
                truncated_list[i].append(bool(truncs[i]))
                episode_reward_sum[i] += float(rewards[i])

                if terms[i] or truncs[i]:
                    done_mask[i] = True
                    terminal_obs[i] = infos[i]["terminal_observation"]
                    episode_info[i] = infos[i]

            obs = next_obs

        advantages_list = []
        returns_list = []
        for i in range(num_envs):
            last_value = (
                agent.get_state_value(terminal_obs[i])
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )
            advantages_i, returns_i = get_GAE(
                {
                    "reward": reward_list[i],
                    "state_value": value_list[i],
                    "terminated": terminated_list[i],
                    "truncated": truncated_list[i],
                },
                gamma=param.gamma,
                lam=0.95,
                last_state_value=last_value,
            )
            advantages_list.append(advantages_i)
            returns_list.append(returns_i)

        obs_update = np.concatenate(
            [np.stack(obs_list[i]) for i in range(num_envs)], axis=0
        )
        action_update = np.concatenate(
            [np.stack(action_list[i]) for i in range(num_envs)], axis=0
        )
        log_prob_update = np.concatenate(
            [np.stack(log_prob_list[i]) for i in range(num_envs)], axis=0
        )
        advantages = np.concatenate(advantages_list, axis=0)
        returns = np.concatenate(returns_list, axis=0)

        policy_loss, value_loss, _ = agent.update(
            obs_update,
            action_update,
            log_prob_update,
            advantages,
            returns,
        )

        round_soups = sum(
            float(info.get("episode_soups", 0.0))
            for info in episode_info
        )
        round_sparse = sum(
            float(info.get("episode_sparse", 0.0))
            for info in episode_info
        )
        round_return = float(episode_reward_sum.sum())
        round_steps = sum(len(items) for items in obs_list)

        global_step += round_steps
        episodes_done += num_envs

        if global_step >= next_checkpoint_step:
            _save_checkpoint(
                agent,
                config,
                "vec",
                global_step,
                total_episodes,
                total_steps,
                curve_path,
                extra={"num_envs": num_envs},
            )
            last_checkpoint_step = global_step
            while next_checkpoint_step <= global_step:
                next_checkpoint_step += CHECKPOINT_INTERVAL

        window_soups += round_soups
        window_sparse += round_sparse
        window_return += round_return
        window_policy_loss += policy_loss
        window_value_loss += value_loss
        window_episodes += num_envs

        if global_step >= next_log_step:
            history["timestep"].append(global_step)
            history["episode"].append(episodes_done - 1)
            history["soups_delivered"].append(
                window_soups / window_episodes
            )
            history["soups_total"].append(window_soups)
            history["training_return"].append(
                window_return / window_episodes
            )
            history["return_sparse"].append(
                window_sparse / window_episodes
            )
            history["policy_loss"].append(
                window_policy_loss / window_episodes
            )
            history["value_loss"].append(
                window_value_loss / window_episodes
            )

            window_soups = 0.0
            window_sparse = 0.0
            window_return = 0.0
            window_policy_loss = 0.0
            window_value_loss = 0.0
            window_episodes = 0
            next_log_step += LOG_INTERVAL

        tqdm.write(
            f"round={round_idx} step={global_step} "
            f"episodes={episodes_done} "
            f"reward={round_return / num_envs:.1f} "
            f"sparse={round_sparse / num_envs:.1f} "
            f"soups={round_soups / num_envs:.2f}"
        )

    if window_episodes > 0:
        history["timestep"].append(global_step)
        history["episode"].append(episodes_done - 1)
        history["soups_delivered"].append(window_soups / window_episodes)
        history["soups_total"].append(window_soups)
        history["training_return"].append(window_return / window_episodes)
        history["return_sparse"].append(window_sparse / window_episodes)
        history["policy_loss"].append(window_policy_loss / window_episodes)
        history["value_loss"].append(window_value_loss / window_episodes)

    curve_path = _save_history(history, config, suffix="_vec")

    if global_step != last_checkpoint_step:
        _save_checkpoint(
            agent,
            config,
            "vec",
            global_step,
            total_episodes,
            total_steps,
            curve_path,
            extra={"num_envs": num_envs},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["single", "vec", "both", "1", "2"],
        default="both",
        help="single/1, vec/2, or both",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="one or more seeds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="legacy single-seed option; overrides --seeds",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help="environment steps per run",
    )
    args = parser.parse_args()

    if args.total_steps <= 0:
        parser.error("--total-steps must be positive")

    seeds = [args.seed] if args.seed is not None else args.seeds
    if args.mode in ("both",):
        modes = ["single", "vec"]
    elif args.mode in ("single", "1"):
        modes = ["single"]
    else:
        modes = ["vec"]

    for seed in seeds:
        for mode in modes:
            print(
                f"=== seed={seed} mode={mode} "
                f"total_steps={args.total_steps} ==="
            )
            if mode == "single":
                train_single_env(
                    seed=seed, total_steps=args.total_steps
                )
            else:
                train_vec_env(
                    seed=seed, total_steps=args.total_steps
                )


if __name__ == "__main__":
    main()
