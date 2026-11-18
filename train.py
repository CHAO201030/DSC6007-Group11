from pathlib import Path

import numpy as np
from tqdm import tqdm

from algorithms.ppo import PPOParams, PPO
from algorithms.algo_utils import save_weights, get_device, get_GAE

from envs.single_env import OvercookedSingleEnv, joint_index, NUM_ACTIONS
from envs.config import EnvConfig

EPISODES = 2500
LOG_INTERVAL = 5000


def get_joint_action(action) -> int:
    return joint_index(action[0].item(), action[1].item())


def train_single_env():
    config = EnvConfig(seed=42)
    env = OvercookedSingleEnv(config)
    obs = env.reset()

    param = PPOParams(
        obs_dim=obs.shape[1],
        hidden_dim=obs.shape[1],
        action_dim=NUM_ACTIONS,
        lr=1e-3,
        gamma=0.99,
        eps=0.2,
        device=get_device(),
    )

    agent = PPO(param)
    global_step = 0
    last_sparse = 0.0
    last_soups = 0.0

    history = {
        "timestep": [],
        "episode": [],
        "soups_delivered": [],
        "soups_total": [],
        "training_return": [],
        "return_sparse": [],
        "policy_loss": [],
        "value_loss": [],
    }

    next_log_step = LOG_INTERVAL
    window_soups = 0.0
    window_sparse = 0.0
    window_return = 0.0
    window_policy_loss = 0.0
    window_value_loss = 0.0
    window_episodes = 0

    for episode in tqdm(range(EPISODES), desc="train"):
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
        history["episode"].append(EPISODES - 1)
        history["soups_delivered"].append(window_soups / window_episodes)
        history["soups_total"].append(window_soups)
        history["training_return"].append(window_return / window_episodes)
        history["return_sparse"].append(window_sparse / window_episodes)
        history["policy_loss"].append(window_policy_loss / window_episodes)
        history["value_loss"].append(window_value_loss / window_episodes)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    curve_path = results_dir / f"train_seed{config.seed}.npz"
    np.savez(
        curve_path,
        **{key: np.asarray(value) for key, value in history.items()},
    )
    print(f"saved curve: {curve_path}")

    checkpoint = save_weights(
        agent,
        f"checkpoints/ppo_single_env/seed{config.seed}",
        step=global_step,
        extra={
            "seed": config.seed,
            "episodes": EPISODES,
            "curve": str(curve_path),
        },
    )
    print(f"saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    train_single_env()
