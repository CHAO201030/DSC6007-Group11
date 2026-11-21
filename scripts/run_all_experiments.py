#!/usr/bin/env python
# scripts/run_all_experiments.py
"""批量评估：扫一个目录下所有 checkpoint，产出学习曲线数据表。

每个 checkpoint 当一次 ego 跑完整评估，输出：
  results/eval/raw/<name>.json        每份固定格式结果（验收用）
  results/eval/learning_curve.csv     汇总表（画图用）

用法：
  python -m scripts.run_all_experiments \
      --ckpt-dir checkpoints/fcp2/seed1 \
      --heldout-ckpt checkpoints/ppo_vec_env/seed1/ppo_clip_8000000.pth \
      --n-episodes 5 --seeds 0,1,2 --start-state standard --stochastic \
      --out-csv results/eval/curve_p.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch                      # noqa: E402
torch.set_num_threads(1)

from scripts.eval.run_sweep import sweep                             # noqa: E402
from agents.scripted import RandomBot, StayBot   # noqa: E402
from agents.checkpoint import SPPolicy  # noqa: E402

SCHEMA_VERSION = "1.0"


def evaluate_one(ckpt_path, *, horizon, n_episodes, seeds,
                 start_state, rnd_obj_prob, heldout_ckpts,
                 seen_dir=None, seen_n=0, stochastic=False):
    """把一个 checkpoint 当 ego 评估，返回 (ego, result_dict)。"""
    ego = SPPolicy(str(ckpt_path), deterministic=not stochastic)
    ego.selftest()

    seen = [("ego", ego)]
    if seen_dir and seen_n:
        from agents.pools import FCPPool
        pool = FCPPool(sorted(Path(seen_dir).glob("*.pth")))
        for p in pool.sample(seen_n, seed=int(ego.global_step) + 1):
            if Path(p).resolve() != Path(ckpt_path).resolve():
                seen.append((Path(p).stem, SPPolicy(p)))
    heldout = [("scripted:stay", StayBot()), ("scripted:random", RandomBot())]
    for p in heldout_ckpts:
        heldout.append((Path(p).stem, SPPolicy(str(p), deterministic=not stochastic)))

    groups = {"seen": seen, "heldout": heldout}

    out = sweep(
        ego, groups,
        horizon=horizon, n_episodes=n_episodes, seeds=seeds,
        start_state=start_state, rnd_obj_prob_thresh=rnd_obj_prob,
    )

    ego_info = {
        "name": ego.name,
        "path": str(ego.path),
        "sha256": ego.digest,
        "global_step": ego.global_step,
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "ego": ego_info,
        "env": {
            "layout": "cramped_room",
            "horizon": horizon,
            "start_state": start_state,
            "rnd_obj_prob_thresh": rnd_obj_prob,
            "seat": 0,
        },
        "protocol": {
            "n_episodes": n_episodes,
            "seeds": list(seeds),
            "deterministic_actions": True,
            "partners": {g: [n for n, _ in ps] for g, ps in groups.items()},
        },
        "metrics": out["metrics"],
        "coordination_gap": out["coordination_gap"],
        "raw": out["raw"],
    }
    return ego, result


def row_from_result(result):
    """把一份结果压成 CSV 的一行。"""
    metrics = result["metrics"]
    gap = result["coordination_gap"] or {}

    def med(group, metric, key):
        return metrics.get(group, {}).get(metric, {}).get(key, "")

    g_soups = gap.get("soups_delivered") or {}
    g_shaped = gap.get("shaped_total") or {}

    return {
        "checkpoint": Path(result["ego"]["path"]).name,
        "global_step": result["ego"]["global_step"],
        "n_seeds": len(result["protocol"]["seeds"]),
        "n_episodes": result["protocol"]["n_episodes"],
        "seen_soups_med": med("seen", "soups_delivered", "median"),
        "seen_soups_q1": med("seen", "soups_delivered", "q1"),
        "seen_soups_q3": med("seen", "soups_delivered", "q3"),
        "heldout_soups_med": med("heldout", "soups_delivered", "median"),
        "heldout_soups_q1": med("heldout", "soups_delivered", "q1"),
        "heldout_soups_q3": med("heldout", "soups_delivered", "q3"),
        "seen_shaped_med": med("seen", "shaped_total", "median"),
        "heldout_shaped_med": med("heldout", "shaped_total", "median"),
        "gap_soups": g_soups.get("value", ""),
        "gap_soups_q1": g_soups.get("q1", ""),
        "gap_soups_q3": g_soups.get("q3", ""),
        "gap_shaped": g_shaped.get("value", ""),
        "gap_shaped_q1": g_shaped.get("q1", ""),
        "gap_shaped_q3": g_shaped.get("q3", ""),
        "seen_distinct_traj": metrics["seen"].get("n_distinct_trajectories", ""),
        "heldout_distinct_traj": metrics["heldout"].get("n_distinct_trajectories", ""),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", nargs="+", required=True,
                   help="要评估的 checkpoint 目录（可给多个）")
    p.add_argument("--heldout-ckpt", nargs="*", default=[],
                   help="额外放进 held-out 的 checkpoint")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--start-state", default="standard",
                   choices=["standard", "random_pos", "random_obj"])
    p.add_argument("--rnd-obj-prob", type=float, default=0.5)
    p.add_argument("--out-csv", default="results/eval/learning_curve.csv")
    p.add_argument("--raw-dir", default="results/eval/raw")
    p.add_argument("--seen-dir", help="诊断用：从该目录采样伙伴加入 seen")
    p.add_argument("--seen-n", type=int, default=0)
    p.add_argument("--stochastic", action="store_true",
                   help="按策略分布采样动作（默认 argmax）")
    args = p.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    paths = []
    for d in args.ckpt_dir:
        found = sorted(list(Path(d).glob("*.pt")) + list(Path(d).glob("*.pth")))
        if not found:
            print(f"[warn] {d} 下没有 .pt 文件")
        paths += found
    if not paths:
        raise SystemExit(f"{args.ckpt_dir} 下没有 .pt 文件")

    print(f"[扫描] {len(paths)} 个 checkpoint")

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}] {path.name}")
        _ego, result = evaluate_one(
            path,
            horizon=args.horizon,
            n_episodes=args.n_episodes,
            seeds=seeds,
            start_state=args.start_state,
            rnd_obj_prob=args.rnd_obj_prob,
            heldout_ckpts=args.heldout_ckpt,
            seen_dir=args.seen_dir,
            seen_n=args.seen_n,
                        stochastic=args.stochastic,
        )

        out_json = raw_dir / f"{path.stem}.json"
        out_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        row = row_from_result(result)
        rows.append(row)
        print(f"   step={row['global_step']}"
              f"  seen_soups={row['seen_soups_med']}"
              f"  heldout_soups={row['heldout_soups_med']}"
              f"  gap_shaped={row['gap_shaped']}")

    rows.sort(key=lambda r: r["global_step"] if isinstance(r["global_step"], int) else 0)

    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[saved] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
