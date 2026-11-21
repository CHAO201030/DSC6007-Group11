#!/usr/bin/env python
"""两臂对比：held-out 主图 + gap 柱状。"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out = {"checkpoint": r["checkpoint"]}
            for k, v in r.items():
                if k == "checkpoint":
                    continue
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = np.nan
            rows.append(out)
    return sorted(rows, key=lambda r: r["global_step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp", required=True)
    ap.add_argument("--fcp", required=True)
    ap.add_argument("--out-dir", default="work_c/results/figures")
    ap.add_argument("--step", type=float, default=None)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = [("SP", a.sp, "#1a365d"), ("FCP", a.fcp, "#E07541")]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    for name, path, color in arms:
        rows = load(path)
        x = np.array([r["global_step"] for r in rows]) / 1e6
        y = np.array([r["heldout_soups_med"] for r in rows])
        lo = np.array([r["heldout_soups_q1"] for r in rows])
        hi = np.array([r["heldout_soups_q3"] for r in rows])
        ax.plot(x, y, marker="o", ms=3, label=name + " median", color=color)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.set_xlabel("training step (M)")
    ax.set_ylabel("soups delivered with held-out partners")
    ax.set_title("Partner generalisation: SP vs FCP")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "main_generalisation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=160)
    vals, labels = [], []
    for name, path, _ in arms:
        rows = load(path)
        if a.step is None:
            r = rows[-1]
        else:
            r = min(rows, key=lambda z: abs(z["global_step"] - a.step))
        vals.append(r["seen_soups_med"] - r["heldout_soups_med"])
        labels.append("%s@%dk" % (name, int(r["global_step"] / 1000)))
    ax.bar(labels, vals, color=[c for _, _, c in arms], width=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("gap = self - held-out (soups)")
    ax.set_title("Coordination gap")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "gap_bar.png")
    plt.close(fig)
    print("[saved] main_generalisation.png, gap_bar.png")


if __name__ == "__main__":
    main()
