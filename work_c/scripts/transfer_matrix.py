#!/usr/bin/env python
"""ego x partner 全交叉矩阵 -> work_c/results/transfer_matrix.csv"""
import argparse
import csv
from pathlib import Path

import numpy as np

from agents.checkpoint import SPPolicy
from work_c.eval.gameenv_wrapper import OvercookedWrapper
from work_c.eval.run_episode import run_one_episode


def label(path):
    p = Path(path)
    return f"{p.parent.parent.name}/{p.parent.name}/{p.stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego", nargs="+", required=True)
    ap.add_argument("--partner", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--start-state", default="random_obj")
    ap.add_argument("--out", default="work_c/results/transfer_matrix.csv")
    ap.add_argument("--stochastic", action="store_true")
    a = ap.parse_args()

    env = OvercookedWrapper(start_state=a.start_state)
    egos = [(label(p), SPPolicy(p, deterministic=not a.stochastic)) for p in a.ego]
    pars = [(label(p), SPPolicy(p, deterministic=not a.stochastic)) for p in a.partner]
    print("[matrix] %d ego x %d partner x %d eps" % (len(egos), len(pars), a.episodes))

    rows = []
    for en, ego in egos:
        for pn, par in pars:
            soups = [run_one_episode(ego, par, seed=s, env=env)["soups_delivered"]
                     for s in range(a.episodes)]
            rows.append({"ego": en, "partner": pn,
                         "soups_median": float(np.median(soups)),
                         "soups_mean": float(np.mean(soups)),
                         "n": len(soups)})
            print("  %-30s x %-30s = %.2f" % (en, pn, rows[-1]["soups_median"]))

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("[saved] " + a.out)


if __name__ == "__main__":
    main()
