"""Agent/partner factory used by evaluation scripts."""

from __future__ import annotations

from .ppo_agent import load_ppo_agent
from .random_agent import RandomAgent, StayAgent


def make_agent(spec=None, seed: int | None = None, deterministic: bool = True):
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = spec or {"type": "random"}

    kind = spec.get("type", "random")

    if kind == "random":
        agent_seed = spec.get("seed", seed)
        return RandomAgent(seed=agent_seed)

    if kind in ("stay", "scripted_stay", "noop"):
        return StayAgent()

    if kind in ("checkpoint", "ppo"):
        path = spec.get("path") or spec.get("checkpoint")
        if not path:
            raise ValueError("checkpoint agent requires 'path' or 'checkpoint'")
        return load_ppo_agent(
            path,
            device=spec.get("device", "cpu"),
            deterministic=spec.get("deterministic", deterministic),
        )

    raise ValueError(f"unknown agent type: {kind}")


def make_partner(
    spec=None,
    seed: int | None = None,
    deterministic: bool = True,
):
    return make_agent(spec, seed=seed, deterministic=deterministic)
