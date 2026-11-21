#!/usr/bin/env python
"""复现校验：固定 4 个 ego 跑两遍，逐文件比 sha256。"""
import hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EGOS = ["work_c/ckpt_final/sp_seed1", "work_c/ckpt_final/p_seed1",
        "work_c/ckpt_final/s_seed1", "work_c/ckpt_final/f_seed42"]
BASE = ["-m", "work_c.scripts.run_all_experiments", "--ckpt-dir", *EGOS,
        "--heldout-ckpt", "checkpoints/ppo_vec_env/seed1/ppo_clip_8000000.pth",
        "--n-episodes", "2", "--seeds", "0,1", "--start-state", "standard",
        "--stochastic"]

_missing = [d for d in EGOS if not (ROOT / d).exists()]
if _missing:
    raise SystemExit(
        "缺少评估输入目录：%s\n请先重建 work_c/ckpt_final/（命令见 work_c/README.md 的「如何重建 ckpt_final」）"
        % _missing
    )

def run(tag):
    raw = Path(ROOT, "work_c", "results", f"_repro_{tag}")
    if raw.exists():
        shutil.rmtree(raw)
    cmd = [sys.executable, *BASE, "--out-csv", f"work_c/results/_repro_{tag}.csv",
           "--raw-dir", str(raw.relative_to(ROOT))]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.glob("*.json"))}

a, b = run("A"), run("B")
print(f"[check] {len(a)} 个结果文件")
diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
if diff:
    print(f"[check] 不一致 {len(diff)} 个：{diff}")
    sys.exit(1)
print("[check] 通过：两次运行的评估结果逐字节一致")
