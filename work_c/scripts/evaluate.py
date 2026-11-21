#!/usr/bin/env python
# work_c/scripts/evaluate.py
"""给定一个 checkpoint，输出一份固定格式的评估结果。

用法：
  python -m work_c.scripts.evaluate \
      --ckpt checkpoints/ppo_single_env/seed1/ppo_clip_8000000.pth \
      --heldout-ckpt checkpoints/ppo_vec_env/seed1/ppo_clip_8000000.pth \
      --start-state standard --stochastic \
      --out work_c/results/raw/seed1.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch                      # noqa: E402
torch.set_num_threads(1)          # 线程数会改变浮点归约顺序

from work_c.eval.run_sweep import sweep                        # noqa: E402
from agents.scripted import RandomBot, StayBot   # noqa: E402
from agents.checkpoint import SPPolicy  # noqa: E402

SCHEMA_VERSION = "1.0"


def build_ego(args):
    if args.ego == "random":
        return RandomBot(seed=0)
    if args.ego == "stay":
        return StayBot()
    if not args.ckpt:
        raise SystemExit("必须给 --ckpt，或用 --ego random / stay")
    ego = SPPolicy(args.ckpt, deterministic=not args.stochastic)
    ego.selftest()
    return ego


def build_groups(args, ego):
    ego_path = Path(ego.path).resolve() if hasattr(ego, "path") else None

    seen = [("ego", ego)]
    if args.seen_dir:
        d = Path(args.seen_dir)
        for path in sorted(list(d.glob("*.pt")) + list(d.glob("*.pth"))):
            if ego_path is not None and path.resolve() == ego_path:
                continue
            seen.append((path.stem, SPPolicy(str(path), deterministic=not args.stochastic)))

    heldout = []
    if not args.no_heldout_scripted:
        heldout += [("scripted:stay", StayBot()), ("scripted:random", RandomBot())]
    for p in args.heldout_ckpt:
        heldout.append((Path(p).stem, SPPolicy(p, deterministic=not args.stochastic)))

    if not heldout:
        raise SystemExit("held-out 集合为空")

    if len(seen) == 1:
        print("[warn] seen 只有 ego 自己，gap 不代表 SP vs FCP")

    return {"seen": seen, "heldout": heldout}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--ego", choices=["random", "stay"])
    p.add_argument("--seen-dir")
    p.add_argument("--heldout-ckpt", nargs="*", default=[])
    p.add_argument("--no-heldout-scripted", action="store_true")
    p.add_argument("--layout", default="cramped_room")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--start-state", default="standard",
                   choices=["standard", "random_pos", "random_obj"])
    p.add_argument("--rnd-obj-prob", type=float, default=0.5)
    p.add_argument("--stochastic", action="store_true",
                   help="按策略分布采样动作（默认 argmax）")
    p.add_argument("--out")
    args = p.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    ego = build_ego(args)
    groups = build_groups(args, ego)

    print(f"[ego] {getattr(ego, 'name', '?')}")
    print(f"[partners] seen={len(groups['seen'])} heldout={len(groups['heldout'])}")

    out = sweep(
        ego, groups,
        layout_name=args.layout,
        horizon=args.horizon,
        n_episodes=args.n_episodes,
        seeds=seeds,
        start_state=args.start_state,
        rnd_obj_prob_thresh=args.rnd_obj_prob,
    )

    ego_info = {"name": getattr(ego, "name", "?")}
    if hasattr(ego, "path"):
        ego_info.update(path=str(ego.path), sha256=ego.digest,
                        global_step=ego.global_step)

    result = {
        "schema_version": SCHEMA_VERSION,
        "ego": ego_info,
        "env": {
            "layout": args.layout,
            "horizon": args.horizon,
            "start_state": args.start_state,
            "rnd_obj_prob_thresh": args.rnd_obj_prob,
            "seat": 0,
        },
        "protocol": {
            "n_episodes": args.n_episodes,
            "seeds": seeds,
            "deterministic_actions": True,
            "partners": {g: [n for n, _ in ps] for g, ps in groups.items()},
        },
        "metrics": out["metrics"],
        "coordination_gap": out["coordination_gap"],
        "raw": out["raw"],
    }

    print("\n=== result ===")
    for g, m in out["metrics"].items():
        s = m["soups_delivered"]
        print(f"{g:8s} soups median={s['median']:.2f} (IQR {s['q1']:.2f}-{s['q3']:.2f})"
              f"  distinct_traj={m['n_distinct_trajectories']}/{m['n_episodes']}"
              f"  shaped_median={m['shaped_total']['median']:.1f}")
    gap = out["coordination_gap"]
    if gap:
        for gap_name, g in gap.items():
            if not g:
                continue
            print(f"coordination_gap[{gap_name}] = {g['value']:+.3f} "
                  f"(IQR {g['q1']:+.3f} ~ {g['q3']:+.3f}, {g['n_seeds']} seeds)")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"[saved] {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())