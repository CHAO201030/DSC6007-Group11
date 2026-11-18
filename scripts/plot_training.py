"""Plot training metrics from the JSONL log written by scripts/train.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_training_log(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def moving_average(values, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training(
    log_path: str | Path,
    output_path: str | Path | None = None,
    window: int = 20,
) -> Path:
    records = load_training_log(log_path)
    if not records:
        raise ValueError(f"no training records in {log_path}")

    episodes = np.asarray([r["episode"] for r in records], dtype=np.float64)
    returns_sparse = np.asarray(
        [r["return_sparse"] for r in records], dtype=np.float64
    )
    returns_shaped = np.asarray(
        [r["return_shaped"] for r in records], dtype=np.float64
    )
    policy_loss = np.asarray(
        [r["policy_loss"] for r in records], dtype=np.float64
    )
    value_loss = np.asarray(
        [r["value_loss"] for r in records], dtype=np.float64
    )
    clip_fraction = np.asarray(
        [r["clip_fraction"] for r in records], dtype=np.float64
    )

    output_path = Path(
        output_path or Path(log_path).with_name("training_curves.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_ret, ax_pi, ax_v, ax_clip = axes.ravel()

    window = max(1, int(window))
    if window > 1 and len(episodes) >= window:
        smooth_episodes = episodes[window - 1 :]
    else:
        smooth_episodes = episodes

    ax_ret.plot(episodes, returns_sparse, alpha=0.25, label="sparse (raw)")
    ax_ret.plot(
        smooth_episodes,
        moving_average(returns_sparse, window),
        label=f"sparse (ma{window})",
    )
    ax_ret.plot(episodes, returns_shaped, alpha=0.25, label="shaped (raw)")
    ax_ret.plot(
        smooth_episodes,
        moving_average(returns_shaped, window),
        label=f"shaped (ma{window})",
    )
    ax_ret.set_title("Episode return")
    ax_ret.set_xlabel("episode")
    ax_ret.set_ylabel("return")
    ax_ret.legend(fontsize=8)
    ax_ret.grid(alpha=0.3)

    ax_pi.plot(episodes, policy_loss, alpha=0.25)
    ax_pi.plot(
        smooth_episodes, moving_average(policy_loss, window), label="policy loss"
    )
    ax_pi.set_title("Policy loss (-L_clip)")
    ax_pi.set_xlabel("episode")
    ax_pi.legend(fontsize=8)
    ax_pi.grid(alpha=0.3)

    ax_v.plot(episodes, value_loss, alpha=0.25)
    ax_v.plot(
        smooth_episodes, moving_average(value_loss, window), label="value loss"
    )
    ax_v.set_title("Value loss (MSE)")
    ax_v.set_xlabel("episode")
    ax_v.legend(fontsize=8)
    ax_v.grid(alpha=0.3)

    ax_clip.plot(episodes, clip_fraction, alpha=0.25)
    ax_clip.plot(
        smooth_episodes,
        moving_average(clip_fraction, window),
        label="clip fraction",
    )
    ax_clip.set_title("Clip fraction")
    ax_clip.set_xlabel("episode")
    ax_clip.set_ylim(0, 1)
    ax_clip.legend(fontsize=8)
    ax_clip.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved plot: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="checkpoints/train/train_log.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    plot_training(args.log, args.output, args.window)


if __name__ == "__main__":
    main()
