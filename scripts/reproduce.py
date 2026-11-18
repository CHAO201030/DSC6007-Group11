"""Re-run evaluation from a JSON config (minimal reproduction)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate import load_config, print_summary, run_config, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/eval.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_config(cfg, checkpoint=args.checkpoint)
    print_summary(result)

    output = args.output or cfg["eval"].get("output")
    if output:
        print(f"saved: {save_result(result, output)}")


if __name__ == "__main__":
    main()
