#!/usr/bin/env python
# work_c/scripts/plot_results.py
"""从 summary_wide.csv 出图。

所有数值都来自真实评估结果（CSV），没有任何硬编码。
没有数据的指标会打印 [skip] 并跳过，不会画出假线。
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            row = {"checkpoint": raw.get("checkpoint", "")}
            for key, value in raw.items():
                if key == "checkpoint":
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = np.nan
            rows.append(row)
    if not rows:
        raise SystemExit(f"{csv_path} 没有数据行")

    def sort_key(r):
        step = r.get("global_step", np.nan)
        return 0 if np.isnan(step) else step

    rows.sort(key=sort_key)
    return rows


def _col(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def plot_metric(rows, out_dir, metric, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    drew = False

    for group, color in (("seen", "#1a365d"), ("heldout", "#E07541")):
        y = _col(rows, f"{group}_{metric}_median")
        if np.all(np.isnan(y)):
            continue
        x = _col(rows, "global_step")
        low = _col(rows, f"{group}_{metric}_q1")
        high = _col(rows, f"{group}_{metric}_q3")
        ax.plot(x, y, marker="o", label=f"{group} (median)", color=color)
        ax.plot(x, high, marker=".", linestyle="--", alpha=0.7,
                label=f"{group} (q3)", color=color)
        ax.fill_between(x, low, high, color=color, alpha=0.18)
        drew = True

    if not drew:
        plt.close(fig)
        print(f"[skip] {metric}: 没有数据")
        return

    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = out_dir / fname
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_gap(rows, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    drew = False

    for col, label, color in (
        ("gap_soups", "gap: soups delivered", "#1a365d"),
        ("gap_shaped", "gap: shaped reward", "#E07541"),
    ):
        y = _col(rows, col)
        if np.all(np.isnan(y)):
            continue
        x = _col(rows, "global_step")
        low = _col(rows, f"{col}_q1")
        high = _col(rows, f"{col}_q3")
        ax.plot(x, y, marker="o", label=label, color=color)
        ax.fill_between(x, low, high, color=color, alpha=0.18)
        drew = True

    if not drew:
        plt.close(fig)
        print("[skip] gap: 没有数据")
        return

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("seen - heldout (median, IQR band)")
    ax.set_title("Coordination gap")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = out_dir / "gap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="work_c/results/summary_wide.csv")
    p.add_argument("--out-dir", default="work_c/results/figures")
    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"找不到 {csv_path}，先跑 python -m work_c.eval.aggregate")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    print(f"[读取] {len(rows)} 行")

    plot_metric(
        rows, out_dir, "soups_delivered",
        "Learning curve — soups delivered",
        "soups delivered (median, IQR)",
        "learning_curve.png",
    )
    plot_metric(
        rows, out_dir, "shaped_total",
        "Learning curve — shaped reward",
        "shaped reward (median, IQR)",
        "learning_curve_shaped.png",
    )
    plot_gap(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
