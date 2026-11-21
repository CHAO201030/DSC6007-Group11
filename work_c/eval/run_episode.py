# work_c/eval/run_episode.py

import hashlib

import numpy as np
import torch

from work_c.eval.gameenv_wrapper import OvercookedWrapper


def run_one_episode(ego_policy, partner_policy, layout_name="cramped_room",
                    horizon=400, seed=0, start_state="random_obj",
                    rnd_obj_prob_thresh=0.5, env=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is None:
        env = OvercookedWrapper(
            layout_name=layout_name, horizon=horizon,
            start_state=start_state, rnd_obj_prob_thresh=rnd_obj_prob_thresh,
        )
    obs = env.reset(seed=seed)

    ego_policy.reset(seed)
    partner_policy.reset(seed)

    soups, shaped, first_soup, steps = 0, 0.0, None, 0
    actions = bytearray()

    for t in range(horizon):
        a0 = int(ego_policy.act(obs, 0))
        a1 = int(partner_policy.act(obs, 1))
        actions += bytes((a0, a1))

        obs, reward, done, info = env.step((a0, a1))
        steps = t + 1

        sparse = float(sum(info.get("sparse_r_by_agent", [0, 0])))
        if sparse > 0:
            soups += int(round(sparse / 20.0))      # 每个汤 20 分
            if first_soup is None:
                first_soup = steps
        shaped += float(sum(info.get("shaped_r_by_agent", [0, 0])))

        if done:
            break

    return {
        "soups_delivered": soups,
        "episode_length": steps,
        "steps_to_first_soup": first_soup if first_soup is not None else horizon,
        "got_soup": first_soup is not None,
        "shaped_total": round(shaped, 3),
        "traj": hashlib.md5(bytes(actions)).hexdigest()[:12],
        "seed": seed,
    }