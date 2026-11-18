"""Agent implementations: decision-making policies and partner policies.

Conventions:
- Only acting logic belongs here (``act(obs) -> int``, partner selection,
  rule-based policies, wrappers around a trained policy).
- Losses, optimizers and GAE live in :mod:`algorithms`.
"""

from .factory import make_agent, make_partner
from .ppo_agent import PPOAgent, load_ppo_agent
from .random_agent import RandomAgent, StayAgent

__all__ = [
    "PPOAgent",
    "RandomAgent",
    "StayAgent",
    "load_ppo_agent",
    "make_agent",
    "make_partner",
]
