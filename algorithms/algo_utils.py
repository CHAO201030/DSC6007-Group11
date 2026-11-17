import torch

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ppo import PPO


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_GAE(data_dict: dict[str, list], gamma=0.99, lam=0.95, last_state_value=None):
    rewards = torch.stack(
        [torch.as_tensor(r, dtype=torch.float32) for r in data_dict["reward"]]
    )
    values = torch.stack(
        [torch.as_tensor(v, dtype=torch.float32) for v in data_dict["state_value"]]
    )
    terminated = torch.stack(
        [torch.as_tensor(t, dtype=torch.float32) for t in data_dict["terminated"]]
    )

    if last_state_value is None:
        last_state_value = data_dict.get("last_state_value")
    if last_state_value is None:
        if not bool(terminated[-1].all()):
            raise ValueError(
                "last_state_value is required when the rollout ends by "
                "truncation: pass critic(next_obs[-1]) or put "
                "'last_state_value' in data_dict"
            )
        last_state_value = torch.zeros_like(values[0])
    else:
        last_state_value = torch.as_tensor(
            last_state_value, dtype=values.dtype, device=values.device
        )

    total_step = values.shape[0]
    advantages = torch.zeros_like(values)
    gae = torch.zeros_like(values[0])

    for step in reversed(range(total_step)):
        next_state_value = (
            last_state_value if step == total_step - 1 else values[step + 1]
        )
        nonterminal = 1.0 - terminated[step]
        delta = (
            rewards[step]
            + gamma * next_state_value * nonterminal
            - values[step]
        )
        gae = delta + gamma * lam * nonterminal * gae
        advantages[step] = gae

    returns = advantages + values

    # (total_step, 2)
    return advantages, returns


def save_weights(
    model: "PPO",
    path: str | Path,
    step: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    if path.suffix.lower() in {".pth", ".pt", ".ckpt"}:
        directory = path.parent
    else:
        directory = path
    directory.mkdir(parents=True, exist_ok=True)

    step = int(step)
    file_path = directory / f"ppo_clip_{step}.pth"

    checkpoint = {
        "format_version": 1,
        "step": step,
        "param": asdict(model.param),
        "actor_state": model.actor.state_dict(),
        "critic_state": model.critic.state_dict(),
        "optimizer_state": model.optimizer.state_dict(),
        "extra": dict(extra or {}),
    }
    torch.save(checkpoint, file_path)
    return file_path


def load_weights(model: "PPO", path: str | Path) -> dict[str, Any]:
    path = Path(path)
    checkpoint = torch.load(
        path, map_location=model.param.device, weights_only=False
    )

    required = {"param", "actor_state", "critic_state", "optimizer_state"}
    missing = required - checkpoint.keys()
    if missing:
        raise KeyError(f"checkpoint is missing keys: {sorted(missing)}")

    saved_param = checkpoint["param"]
    for key in ("obs_dim", "hidden_dim", "action_dim"):
        expected = getattr(model.param, key)
        actual = saved_param.get(key)
        if actual != expected:
            raise ValueError(
                f"checkpoint {key}={actual} does not match model {key}={expected}"
            )

    model.actor.load_state_dict(checkpoint["actor_state"])
    model.critic.load_state_dict(checkpoint["critic_state"])
    model.optimizer.load_state_dict(checkpoint["optimizer_state"])

    return {
        "step": int(checkpoint.get("step", 0)),
        "extra": checkpoint.get("extra", {}),
    }


if __name__ == "__main__":
    print(get_device())