#!/usr/bin/env python
"""复现校验：固定 2 个 checkpoint 各跑两遍，逐文件比 sha256。

用法（仓库根目录）：
    python -m scripts.check_repro
"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CKPTS = [
    "checkpoints/ppo_single_env/seed1/ppo_clip_8000000.pth",
    "checkpoints/ppo_vec_env/seed1/ppo_clip_8000000.pth",
]
HELDOUT = "checkpoints/ppo_vec_env/seed1/ppo_clip_8000000.pth"

_missing = [c for c in CKPTS + [HELDOUT] if not (ROOT / c).exists()]
if _missing:
    raise SystemExit(
        "缺少权重文件：%s\n请先把 checkpoints/ 放齐（见仓库 README）" % _missing
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(tag):
    out = ROOT / "results" / "eval" / f"_repro_{tag}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    for ck in CKPTS:
        name = "%s_%s" % (Path(ck).parent.parent.name, Path(ck).stem)
        cmd = [
            sys.executable, "-m", "scripts.evaluate",
            "--ckpt", ck,
            "--heldout-ckpt", HELDOUT,
            "--n-episodes", "2", "--seeds", "0,1",
            "--start-state", "standard", "--stochastic",
            "--out", str((out / (name + ".json")).relative_to(ROOT)),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)

    return {p.name: sha256(p) for p in sorted(out.glob("*.json"))}


def main():
    a, b = run("A"), run("B")
    print("[check] %d 个结果文件" % len(a))

    diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    if diff:
        print("[check] 不一致 %d 个：%s" % (len(diff), diff))
        return 1

    print("[check] 通过：两次运行的评估结果逐字节一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
