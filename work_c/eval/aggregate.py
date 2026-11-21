# work_c/eval/aggregate.py
"""把 results/raw/*.json 汇总成报告用的主结果表。

产出两张 CSV：
  work_c/results/summary_long.csv   长表：一行 = 一个 checkpoint x 一个分组
  work_c/results/summary_wide.csv   宽表：一行 = 一个 checkpoint，便于直接贴进报告"""

import csv
import json
from pathlib import Path

RAW_DIR = Path("work_c/results/raw_final")
OUT_DIR = Path("work_c/results")

GROUPS = ("seen", "heldout")
METRICS = ("soups_delivered", "shaped_total", "steps_to_first_soup")
EXTRA = ("n_episodes_with_soup", "n_distinct_trajectories")


def load_results(raw_dir=None):
    raw_dir = raw_dir or RAW_DIR
    results = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        if path.name.startswith("_"):      # 跳过 _demo_*.json 之类的临时结果
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {path.name}: 读不了 ({e})")
            continue
        if "schema_version" not in data or "metrics" not in data:
            print(f"[skip] {path.name}: 不是新版 schema")
            continue
        data["_file"] = path.name
        results.append(data)
    return results


def to_long_rows(results):
    """一个 (checkpoint, group) 一行。"""
    rows = []
    for r in results:
        ego = r.get("ego", {})
        gap = r.get("coordination_gap") or {}
        proto = r.get("protocol", {})

        for group in GROUPS:
            m = r.get("metrics", {}).get(group)
            if not m:
                continue

            row = {
                "checkpoint": Path(ego.get("path", r["_file"])).name,
                "global_step": ego.get("global_step", ""),
                "group": group,
                "n_seeds": len(proto.get("seeds", [])),
                "n_episodes": m.get("n_episodes", ""),
                "n_distinct_traj": m.get("n_distinct_trajectories", ""),
                "start_state": r.get("env", {}).get("start_state", ""),
            }

            for metric in METRICS:
                stats = m.get(metric, {})
                row[f"{metric}_median"] = stats.get("median", "")
                row[f"{metric}_q1"] = stats.get("q1", "")
                row[f"{metric}_q3"] = stats.get("q3", "")

            for key in EXTRA:
                row[key] = m.get(key, "")

            # gap 只有 seen 行重复一遍，方便并排看
            for gap_name in ("soups_delivered", "shaped_total"):
                g = gap.get(gap_name) or {}
                row[f"gap_{gap_name}"] = g.get("value", "")
                row[f"gap_{gap_name}_q1"] = g.get("q1", "")
                row[f"gap_{gap_name}_q3"] = g.get("q3", "")

            rows.append(row)

    rows.sort(key=lambda x: (x["checkpoint"], x["group"]))
    return rows


def to_wide_rows(long_rows):
    """一个 checkpoint 一行，seen / heldout 并排。"""
    by_ckpt = {}
    for row in long_rows:
        key = row["checkpoint"]
        w = by_ckpt.setdefault(key, {
            "checkpoint": key,
            "global_step": row["global_step"],
            "n_seeds": row["n_seeds"],
            "start_state": row["start_state"],
        })
        g = row["group"]
        for metric in METRICS:
            for stat in ("median", "q1", "q3"):
                w[f"{g}_{metric}_{stat}"] = row.get(f"{metric}_{stat}", "")
        w[f"{g}_n_episodes"] = row.get("n_episodes", "")   
        w[f"{g}_n_with_soup"] = row.get("n_episodes_with_soup", "")
        w[f"gap_soups"] = row.get("gap_soups_delivered", "")
        w[f"gap_soups_q1"] = row.get("gap_soups_delivered_q1", "")
        w[f"gap_soups_q3"] = row.get("gap_soups_delivered_q3", "")
        w[f"gap_shaped"] = row.get("gap_shaped_total", "")
        w[f"gap_shaped_q1"] = row.get("gap_shaped_total_q1", "")
        w[f"gap_shaped_q3"] = row.get("gap_shaped_total_q3", "")

    rows = list(by_ckpt.values())
    rows.sort(key=lambda x: x["global_step"] if isinstance(x["global_step"], int) else 0)
    return rows


def write_csv(path, rows):
    if not rows:
        print(f"[warn] 没有数据，跳过 {path}")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {path}  ({len(rows)} 行)")


def main(raw_dir=None):
    results = load_results(raw_dir)
    if not results:
        raise SystemExit(f"{raw_dir or RAW_DIR} 里没有可用的结果，先跑 evaluate 或 run_all_experiments")

    print(f"[读取] {len(results)} 份结果")

    long_rows = to_long_rows(results)
    wide_rows = to_wide_rows(long_rows)

    write_csv(OUT_DIR / "summary_long.csv", long_rows)
    write_csv(OUT_DIR / "summary_wide.csv", wide_rows)

    # 控制台打印一张简表
    print("\n=== 主结果表（soups 中位数 / IQR） ===")
    header = f"{'checkpoint':<28} {'step':>8} {'seen':>10} {'heldout':>10} {'gap(soups)':>11} {'gap(shaped)':>12}"
    print(header)
    print("-" * len(header))
    for w in wide_rows:
        seen = f"{w['seen_soups_delivered_median']}"
        held = f"{w['heldout_soups_delivered_median']}"
        print(f"{w['checkpoint']:<28} {w['global_step']:>8} {seen:>10} {held:>10} "
              f"{w['gap_soups']:>11} {w['gap_shaped']:>12}")


if __name__ == "__main__":
    main()