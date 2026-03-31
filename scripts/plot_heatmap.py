"""
Generate heatmap visualizations of experiment results.

Usage:
  python scripts/plot_heatmap.py artifacts/results/results_hard-hybrid_spread.json --output-dir artifacts/plots/plots_hard_hybrid
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({"font.size": 12, "figure.dpi": 150})

VARIANT_ORDER = [
    "naive", "memory_augmented", "belief_tracking",
    "reflection_enhanced", "belief_no_decay", "memory_with_trust",
]
VARIANT_LABELS = {
    "naive": "Naive",
    "memory_augmented": "Memory-Aug.",
    "belief_tracking": "Belief-Track.",
    "reflection_enhanced": "Reflect.-Enh.",
    "belief_no_decay": "Belief (No Decay)",
    "memory_with_trust": "Memory + Trust",
}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_heatmap(summary: dict, metric: str, title: str, output_path: Path, fmt: str = ".2f", vmin: float = 0, vmax: float = 1, cmap: str = "RdYlGn") -> None:
    data: dict[str, dict[float, float]] = defaultdict(dict)
    for key, metrics in summary.items():
        variant, lr_str = key.split("@")
        data[variant][float(lr_str)] = metrics[metric]

    variants = [v for v in VARIANT_ORDER if v in data]
    liar_ratios = sorted({lr for v in data.values() for lr in v})

    matrix = []
    for v in variants:
        matrix.append([data[v].get(lr, 0) for lr in liar_ratios])
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, max(3, len(variants) * 0.7 + 1)))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(liar_ratios)))
    ax.set_xticklabels([f"{lr:.1f}" for lr in liar_ratios])
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_xlabel("Liar Ratio")
    ax.set_title(title, fontsize=14, fontweight="bold")

    for i in range(len(variants)):
        for j in range(len(liar_ratios)):
            val = matrix[i, j]
            color = "white" if val < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate heatmap plots")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output-dir", default="artifacts/plots", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load(args.results_file)
    summary = data["summary"]

    print("Generating heatmaps...")
    plot_heatmap(summary, "task_success_rate", "Task Success Rate", out / "heatmap_success.png")
    plot_heatmap(summary, "avg_inference_accuracy", "Inference Accuracy (Trust Alignment)", out / "heatmap_inference.png")
    plot_heatmap(summary, "avg_steps", "Average Steps", out / "heatmap_steps.png",
                 fmt=".1f", vmin=14, vmax=20, cmap="RdYlGn_r")
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
