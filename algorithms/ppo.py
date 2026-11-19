import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from dataclasses import dataclass
from torch.distributions import Categorical

if __package__:
    from .algo_utils import get_device, load_weights
else:
    from algo_utils import get_device, load_weights

@dataclass
class PPOParams:
    obs_dim: int
    hidden_dim: int
    action_dim: int
    lr: float
    gamma: float
    eps: float
    device: str
    value_coef: float = 0.5
    epochs: int = 10
    minibatch_size: int = 64
    max_grad_norm: float = 0.5


class Actor(nn.Module):
    def __init__(self, param: PPOParams):
        super().__init__()

        self.param = param

        self.network = nn.Sequential(
            nn.Linear(param.obs_dim, param.hidden_dim),
            nn.Tanh(),
            nn.Linear(param.hidden_dim, param.hidden_dim),
            nn.Tanh(),
            nn.Linear(param.hidden_dim, param.action_dim)         
        )

    def forward(self, obs):
        # (batch_size, obs_dim) -> (batch_size, action_dim)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.param.device)

        return self.network(obs)


class Critic(nn.Module):
    def __init__(self, param: PPOParams):
        super().__init__()

        self.param = param

        self.network = nn.Sequential(
            nn.Linear(param.obs_dim, param.hidden_dim),
            nn.Tanh(),
            nn.Linear(param.hidden_dim, param.hidden_dim),
            nn.Tanh(),
            nn.Linear(param.hidden_dim, 1) 
        )

    def forward(self, obs):
        # (batch_size, obs_dim) -> (batch_size, 1)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.param.device)

        return self.network(obs)  


class PPO(nn.Module):
    def __init__(self, param: PPOParams):
        super().__init__()

        self.param = param

        self.actor = Actor(param)
        self.critic = Critic(param)

        self.optimizer = Adam(
            [*self.actor.parameters(), *self.critic.parameters()],
            lr=param.lr
        )

    @torch.no_grad()
    def get_action(self, obs, deterministic: bool=False):
        logits: torch.Tensor = self.actor(obs)
        action_distribution = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = action_distribution.sample()

        log_prob = action_distribution.log_prob(action)

        return action, log_prob

    @torch.no_grad()
    def get_state_value(self, obs):
        return self.critic(obs)

    @staticmethod
    def _stack_steps(value, device, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value, dtype=dtype, device=device)
        return torch.stack(
            [torch.as_tensor(v, dtype=dtype, device=device) for v in value]
        )

    def _prepare_update_data(self, obs, actions, old_log_probs, advantages, returns):
        device = self.param.device

        obs = self._stack_steps(obs, device, torch.float32)
        actions = self._stack_steps(actions, device, torch.long)
        old_log_probs = self._stack_steps(old_log_probs, device, torch.float32)
        advantages = self._stack_steps(advantages, device, torch.float32)
        returns = self._stack_steps(returns, device, torch.float32)

        obs = obs.reshape(-1, obs.shape[-1])
        actions = actions.reshape(-1)
        old_log_probs = old_log_probs.reshape(-1).detach()
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1).detach()

        n = obs.shape[0]
        if not (
            actions.numel() == old_log_probs.numel() == n
            and advantages.numel() == returns.numel() == n
        ):
            raise ValueError(
                "obs/action/log_prob/advantages/returns must have the same "
                f"number of samples, got {n}, {actions.numel()}, "
                f"{old_log_probs.numel()}, {advantages.numel()}, "
                f"{returns.numel()}"
            )

        return obs, actions, old_log_probs, advantages, returns

    def update(self, obs, actions, old_log_probs, advantages, returns):
        obs, actions, old_log_probs, advantages, returns = self._prepare_update_data(obs, actions, old_log_probs, advantages, returns)
        n = obs.shape[0]
        device = self.param.device

        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        advantages = advantages.detach()

        clip_coef = self.param.eps
        batch_size = max(1, min(int(self.param.minibatch_size), n))
        params = list(self.actor.parameters()) + list(self.critic.parameters())

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        clip_fraction_sum = 0.0
        seen = 0

        for _ in range(self.param.epochs):
            indices = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = indices[start : start + batch_size]
                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                logits = self.actor(batch_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)

                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    * batch_advantages,
                ).mean()

                values = self.critic(batch_obs).squeeze(-1)
                value_loss = F.mse_loss(values, batch_returns)

                loss = policy_loss + self.param.value_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    params, self.param.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    clip_fraction = (
                        (ratio - 1.0).abs() > clip_coef
                    ).float().mean().item()

                weight = idx.numel()
                policy_loss_sum += policy_loss.item() * weight
                value_loss_sum += value_loss.item() * weight
                clip_fraction_sum += clip_fraction * weight
                seen += weight

        denom = max(seen, 1)
        return (
            policy_loss_sum / denom,
            value_loss_sum / denom,
            clip_fraction_sum / denom,
        )


def load_policy(path, device: str = "cpu") -> PPO:
    """Load one PPO checkpoint and return a ready-to-use policy."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    param = PPOParams(**checkpoint["param"])
    param.device = device

    model = PPO(param)
    load_weights(model, path)
    model.global_step = int(checkpoint.get("step", -1))
    return model


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    if __package__:
        from .algo_utils import load_weights, save_weights
    else:
        from algo_utils import load_weights, save_weights

    torch.manual_seed(0)
    obs_dim, action_dim, num_agents = 96, 6, 2

    def make_param() -> PPOParams:
        return PPOParams(
            obs_dim=obs_dim,
            hidden_dim=64,
            action_dim=action_dim,
            lr=1e-3,
            gamma=0.99,
            eps=0.2,
            device=get_device(),
            epochs=2,
            minibatch_size=8,
        )

    # get_action: shapes / range / deterministic reproducibility
    ppo = PPO(make_param())
    obs = torch.rand(num_agents, obs_dim)

    action, log_prob = ppo.get_action(obs)
    assert action.shape == (num_agents,), action.shape
    assert log_prob.shape == (num_agents,), log_prob.shape
    assert action.dtype == torch.int64, action.dtype
    assert 0 <= int(action.min()) and int(action.max()) < action_dim

    det_a1, det_lp1 = ppo.get_action(obs, deterministic=True)
    det_a2, det_lp2 = ppo.get_action(obs, deterministic=True)
    assert torch.equal(det_a1, det_a2)
    assert torch.allclose(det_lp1, det_lp2)
    print("[PASS] get_action")

    # get_state_value: (num_agents, 1)
    value = ppo.get_state_value(obs)
    assert value.shape == (num_agents, 1), value.shape
    print("[PASS] get_state_value")

    # update: returns 3 metrics, finite, and changes parameters
    total_steps = 8
    obs_list, action_list, log_prob_list = [], [], []
    for _ in range(total_steps):
        step_obs = torch.rand(num_agents, obs_dim)
        step_action, step_log_prob = ppo.get_action(step_obs)
        obs_list.append(step_obs)
        action_list.append(step_action)
        log_prob_list.append(step_log_prob)

    advantages = torch.randn(total_steps, num_agents) * 0.1
    returns = torch.randn(total_steps, num_agents)
    before = {k: v.clone() for k, v in ppo.actor.state_dict().items()}

    metrics = ppo.update(
        obs_list, action_list, log_prob_list, advantages, returns
    )
    assert isinstance(metrics, tuple) and len(metrics) == 3, metrics
    assert all(torch.isfinite(torch.tensor(x)) for x in metrics), metrics
    assert any(
        not torch.allclose(before[k], ppo.actor.state_dict()[k])
        for k in before
    ), "update did not change actor parameters"
    print("[PASS] update", tuple(round(x, 6) for x in metrics))

    # save_weights / load_weights
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = save_weights(
            ppo, Path(tmp_dir), step=123, extra={"test": True}
        )
        assert checkpoint_path.name == "ppo_clip_123.pth"
        assert checkpoint_path.exists()

        loaded = PPO(make_param())
        meta = load_weights(loaded, checkpoint_path)
        assert meta["step"] == 123
        assert meta["extra"]["test"] is True

        for key, tensor in ppo.actor.state_dict().items():
            assert torch.allclose(tensor, loaded.actor.state_dict()[key])
        for key, tensor in ppo.critic.state_dict().items():
            assert torch.allclose(tensor, loaded.critic.state_dict()[key])
        print("[PASS] save_weights / load_weights")

    print("all PPO tests passed")