"""
Run experiments across multiple liar ratios and print a comparison table.

Usage:
  python run_liar_ratio_comparison.py

Uses default config (mock LLM). Compares all four agent variants across
liar_ratios [0.1, 0.3, 0.5] and prints task success rate, inference accuracy,
average steps, and recovery rate.
"""
from __future__ import annotations

import json
import sys

from deceptive_text_env import build_default_config
from deceptive_text_env.evaluation import EvaluationRunner


def format_table(
    summary: dict[str, dict[str, float]],
    variants: list[str],
    liar_ratios: list[float],
    metric: str,
    title: str,
) -> str:
    """Build a text table for one metric: rows = variant, cols = liar_ratio."""
    lines = [f"\n=== {title} ({metric}) ==="]
    header = "variant            " + "".join(f" | {lr:.1f}   " for lr in liar_ratios)
    lines.append(header)
    lines.append("-" * len(header))
    for v in variants:
        row = f"{v:<18}"
        for lr in liar_ratios:
            key = f"{v}@{lr:.1f}"
            val = summary.get(key, {}).get(metric, -999.0)
            if val == -1.0 and metric == "avg_recovery_rate":
                cell = "  n/a "
            elif isinstance(val, float) and 0 <= val <= 1 and "rate" in metric or "accuracy" in metric:
                cell = f" {val:.2f}  "
            else:
                cell = f" {val:.2f}  " if isinstance(val, (int, float)) else "  -   "
            row += " |" + cell
        lines.append(row)
    return "\n".join(lines)


def main() -> int:
    config = build_default_config()
    variants = ["naive", "memory_augmented", "belief_tracking", "reflection_enhanced"]
    liar_ratios = config.experiment.liar_ratios

    runner = EvaluationRunner(config)
    results, summary = runner.run_all(variants)

    print("Liar ratio comparison (mock LLM)")
    print(f"Variants: {variants}")
    print(f"Liar ratios: {liar_ratios}")
    print(f"Runs per setting: {config.experiment.runs_per_setting}")
    print(f"Total episodes: {len(results)}")

    print(format_table(summary, variants, liar_ratios, "task_success_rate", "Task success rate"))
    print(format_table(summary, variants, liar_ratios, "avg_inference_accuracy", "Inference accuracy (trust alignment)"))
    print(format_table(summary, variants, liar_ratios, "avg_steps", "Average steps to done"))
    print(format_table(summary, variants, liar_ratios, "avg_recovery_rate", "Avg recovery rate (turns to distrust)"))

    print("\n--- Full summary (JSON) ---")
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
