#!/usr/bin/env python
# work_c/scripts/reproduce.py
"""一键复现：评估矩阵 -> 汇总表 -> 图 -> manifest。

  python -m work_c.scripts.reproduce            # 全量重跑，写 manifest
  python -m work_c.scripts.reproduce --check    # 全量重跑，与上一份 manifest 逐文件比对

manifest 只记录输出文件的 sha256，不含时间戳，因此可以用于跨机比对。
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS = Path("work_c/results")
MANIFEST = RESULTS / "manifest.json"

DEFAULT_CKPT_DIRS = [
    "work_c/partners/self_play_snapshots_old",
    "work_c/partners/self_play_snapshots",
]
DEFAULT_HELDOUT = ["work_c/partners/self_play_snapshots/sp_seed0_last.pt"]

EXCLUDE_NAMES = {"manifest.json", "run_info.json"}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_outputs():
    out = []
    for p in sorted(RESULTS.rglob("*")):
        if not p.is_file():
            continue
        if p.name in EXCLUDE_NAMES:
            continue
            if any(part.startswith("_") for part in p.parts) or p.name in EXCLUDE_NAMES:
             continue
            continue
        if p.suffix.lower() == ".png":
            # 图片字节受字体 / matplotlib 后端影响，换机器必然不同，不参与比对
            continue
        out.append(p)
    return out


def build_manifest():
    return {
        str(p.relative_to(RESULTS)).replace("\\", "/"): sha256_file(p)
        for p in collect_outputs()
    }


def run_pipeline(args):
    from work_c.scripts import run_all_experiments as rae
    from work_c.eval import aggregate as agg
    from work_c.scripts import plot_results as pr

    rae.main([
        "--ckpt-dir", *args.ckpt_dir,
        "--heldout-ckpt", *args.heldout_ckpt,
        "--n-episodes", str(args.n_episodes),
        "--seeds", args.seeds,
        "--horizon", str(args.horizon),
        "--start-state", args.start_state,
        "--rnd-obj-prob", str(args.rnd_obj_prob),
        "--out-csv", str(RESULTS / "learning_curve.csv"),
        "--raw-dir", str(RESULTS / "raw"),
    ])

    agg.main()

    pr.main([
        "--csv", str(RESULTS / "summary_wide.csv"),
        "--out-dir", str(RESULTS / "figures"),
    ])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", nargs="+", default=DEFAULT_CKPT_DIRS)
    p.add_argument("--heldout-ckpt", nargs="*", default=DEFAULT_HELDOUT)
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--start-state", default="random_obj",
                   choices=["standard", "random_pos", "random_obj"])
    p.add_argument("--rnd-obj-prob", type=float, default=0.5)
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)

    started = datetime.now()
    run_pipeline(args)
    files = build_manifest()

    if args.check:
        if not MANIFEST.exists():
            print(f"[check] 找不到 {MANIFEST}，先不带 --check 跑一次")
            return 2

        old_files = json.loads(MANIFEST.read_text(encoding="utf-8")).get("files", {})
        diffs = []
        for name in sorted(set(old_files) | set(files)):
            if old_files.get(name) != files.get(name):
                diffs.append((name, old_files.get(name), files.get(name)))

        if diffs:
            print(f"\n[check] 不一致 {len(diffs)} 个文件")
            for name, old, new in diffs:
                print(f"  {name}: {str(old)[:12]} -> {str(new)[:12]}")
            return 1

        print(f"\n[check] 通过：{len(files)} 个输出文件全部一致")

    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_sec": round((datetime.now() - started).total_seconds(), 1),
        "params": {
            "ckpt_dir": args.ckpt_dir,
            "heldout_ckpt": args.heldout_ckpt,
            "n_episodes": args.n_episodes,
            "seeds": args.seeds,
            "horizon": args.horizon,
            "start_state": args.start_state,
            "rnd_obj_prob": args.rnd_obj_prob,
        },
        "files": files,
    }
    MANIFEST.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {MANIFEST}  ({len(files)} 个文件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
