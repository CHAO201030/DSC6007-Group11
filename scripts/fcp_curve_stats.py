#!/usr/bin/env python
"""训练曲线健康检查：峰值 / 2M 后均值 / 零窗口 / 末值。"""
import argparse
import glob
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/eval/curves/*.npz")
    ap.add_argument("--since", type=float, default=2e6)
    a = ap.parse_args()

    files = sorted(glob.glob(a.pattern))
    if not files:
        raise SystemExit("没有匹配: " + a.pattern)

    print("%-22s %6s %8s %9s %8s %8s" % ("file", "pts", "max", "mean_tail", "zero%", "last"))
    for f in files:
        d = np.load(f)
        ts, sd = d["timestep"], d["soups_delivered"]
        tail = sd[ts >= a.since]
        print("%-22s %6d %8.2f %9.2f %7.1f%% %8.2f" % (
            Path(f).name, len(ts), float(sd.max()),
            float(tail.mean()) if len(tail) else float("nan"),
            100.0 * float((sd == 0).mean()), float(sd[-1])))


if __name__ == "__main__":
    main()
