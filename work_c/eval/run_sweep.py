# work_c/eval/run_sweep.py
"""按伙伴分组跑评估，产出 median/IQR 和配对的 coordination gap。"""

import numpy as np

from work_c.eval.gameenv_wrapper import OvercookedWrapper
from work_c.eval.run_episode import run_one_episode

# 要计算 gap 的指标
GAP_METRICS = ("soups_delivered", "shaped_total")


def _stat(values):
    a = np.asarray(values, dtype=float)
    return {
        "median": round(float(np.median(a)), 6),
        "q1": round(float(np.percentile(a, 25)), 6),
        "q3": round(float(np.percentile(a, 75)), 6),
        "n": int(a.size),
    }


def _paired_gap(records, seeds, metric="soups_delivered"):
    """配对 gap：同一批 episode 初始状态上，seen 中位数 - heldout 中位数。"""
    if "seen" not in records or "heldout" not in records:
        return None

    per_seed = []
    for seed in seeds:
        a = [r[metric] for r in records["seen"] if r["seed"] == seed]
        b = [r[metric] for r in records["heldout"] if r["seed"] == seed]
        if not a or not b:
            continue
        per_seed.append({
            "seed": int(seed),
            "seen_median": round(float(np.median(a)), 6),
            "heldout_median": round(float(np.median(b)), 6),
            "gap": round(float(np.median(a)) - float(np.median(b)), 6),
        })

    if not per_seed:
        return None

    gaps = np.array([d["gap"] for d in per_seed], dtype=float)
    return {
        "metric": metric,
        "value": round(float(np.median(gaps)), 6),
        "q1": round(float(np.percentile(gaps, 25)), 6),
        "q3": round(float(np.percentile(gaps, 75)), 6),
        "n_seeds": int(gaps.size),
        "definition": f"median({metric}|seen) - median({metric}|heldout)，按 seed 配对",
        "per_seed": per_seed,
    }


def sweep(ego_policy, groups, layout_name="cramped_room", horizon=400,
          n_episodes=10, seeds=(0, 1, 2, 3, 4),
          start_state="random_obj", rnd_obj_prob_thresh=0.5):
    """groups = {"seen": [(名字, policy), ...], "heldout": [...]}"""
    env = OvercookedWrapper(
        layout_name=layout_name, horizon=horizon,
        start_state=start_state, rnd_obj_prob_thresh=rnd_obj_prob_thresh,
    )

    records = {g: [] for g in groups}
    for group, partners in groups.items():
        for pname, partner in partners:
            for seed in seeds:
                for ep in range(n_episodes):
                    ep_seed = int(seed) * 100_000 + ep
                    rec = run_one_episode(
                        ego_policy, partner, layout_name=layout_name,
                        horizon=horizon, seed=ep_seed, env=env,
                    )
                    rec.update(group=group, partner=pname, seed=int(seed), ep=ep)
                    records[group].append(rec)

    metrics = {}
    for group, recs in records.items():
        metrics[group] = {
            "soups_delivered": _stat([r["soups_delivered"] for r in recs]),
            "steps_to_first_soup": _stat([r["steps_to_first_soup"] for r in recs]),
            "shaped_total": _stat([r["shaped_total"] for r in recs]),
            "episode_length": _stat([r["episode_length"] for r in recs]),
            "n_distinct_trajectories": len({r["traj"] for r in recs}),
            "n_episodes": len(recs),
                        "n_episodes": len(recs),
            "n_episodes_with_soup": sum(1 for r in recs if r.get("got_soup")),
        }

    gaps = {m: _paired_gap(records, seeds, m) for m in GAP_METRICS}
    if all(v is None for v in gaps.values()):
        gaps = None

    return {
        "metrics": metrics,
        "coordination_gap": gaps,
        "raw": records,
    }