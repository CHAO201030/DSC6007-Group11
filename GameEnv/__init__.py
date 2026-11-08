"""Vendored Overcooked-AI environment (Gymnasium-compatible).

Source: HumanCompatibleAI/overcooked_ai v2.0.0 (MIT).
See VENDORING.md for the exact files and the import changes applied.

Importing this package registers ``Overcooked-v0`` with Gymnasium. The
entry point requires ``base_env`` and ``featurize_fn`` keyword arguments::

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    base_env = OvercookedEnv.from_mdp(mdp, horizon=400)
    env = gymnasium.make(
        "Overcooked-v0",
        base_env=base_env,
        featurize_fn=base_env.featurize_state_mdp,
    )
"""

from gymnasium.envs.registration import register

register(
    id="Overcooked-v0",
    entry_point="GameEnv.overcooked_env:Overcooked",
)
