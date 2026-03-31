"""
Plot scaling experiment results: success rate vs NPC count.

Usage:
  python scripts/plot_scaling.py artifacts/results/results_scaling_mock.json --output-dir artifacts/plots/plots_scaling
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({"font.size": 12, "figure.dpi": 150})

VARIANT_COLORS = {
    "naive": "#e74c3c",
    "memory_augmented": "#f39c12",
    "belief_tracking": "#2ecc71",
    "reflection_enhanced": "#3498db",
    "belief_no_decay": "#9b59b6",
    "memory_with_trust": "#1abc9c",
}
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


def _stderr(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance / n)


def plot_scaling_success(episodes: list[dict], output_dir: Path) -> None:
    """Line plot: success rate vs NPC count, one line per variant."""
    groups: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        groups[ep["agent_variant"]][ep["total_npcs"]].append(1.0 if ep["success"] else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, npc_map in sorted(groups.items()):
        npcs = sorted(npc_map.keys())
        means = [sum(npc_map[n]) / len(npc_map[n]) for n in npcs]
        errs = [_stderr(npc_map[n]) for n in npcs]
        ax.errorbar(npcs, means, yerr=errs,
                    marker="o", linewidth=2, markersize=8, capsize=4,
                    color=VARIANT_COLORS.get(variant, "gray"),
                    label=VARIANT_LABELS.get(variant, variant))

    ax.set_xlabel("Total NPCs")
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Success Rate vs. NPC Count (LR=0.3, Hard Mode)", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(sorted({ep["total_npcs"] for ep in episodes}))
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_success.png", bbox_inches="tight")
    print(f"  Saved: {output_dir / 'scaling_success.png'}")
    plt.close(fig)


def plot_scaling_steps(episodes: list[dict], output_dir: Path) -> None:
    """Line plot: avg steps vs NPC count."""
    groups: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        groups[ep["agent_variant"]][ep["total_npcs"]].append(ep["steps"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, npc_map in sorted(groups.items()):
        npcs = sorted(npc_map.keys())
        means = [sum(npc_map[n]) / len(npc_map[n]) for n in npcs]
        errs = [_stderr(npc_map[n]) for n in npcs]
        ax.errorbar(npcs, means, yerr=errs,
                    marker="s", linewidth=2, markersize=8, capsize=4,
                    color=VARIANT_COLORS.get(variant, "gray"),
                    label=VARIANT_LABELS.get(variant, variant))

    ax.set_xlabel("Total NPCs")
    ax.set_ylabel("Average Steps")
    ax.set_title("Average Steps vs. NPC Count (LR=0.3, Hard Mode)", fontsize=14, fontweight="bold")
    ax.set_xticks(sorted({ep["total_npcs"] for ep in episodes}))
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_steps.png", bbox_inches="tight")
    print(f"  Saved: {output_dir / 'scaling_steps.png'}")
    plt.close(fig)


def plot_scaling_heatmap(episodes: list[dict], output_dir: Path) -> None:
    """Heatmap: variant x NPC count, colored by success rate."""
    groups: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        groups[ep["agent_variant"]][ep["total_npcs"]].append(1.0 if ep["success"] else 0.0)

    variant_order = ["naive", "memory_augmented", "belief_tracking",
                     "reflection_enhanced", "belief_no_decay", "memory_with_trust"]
    variants = [v for v in variant_order if v in groups]
    npc_counts = sorted({ep["total_npcs"] for ep in episodes})

    matrix = []
    for v in variants:
        row = []
        for n in npc_counts:
            vals = groups[v].get(n, [])
            row.append(sum(vals) / len(vals) if vals else 0)
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, max(3, len(variants) * 0.7 + 1)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(npc_counts)))
    ax.set_xticklabels([str(n) for n in npc_counts])
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_xlabel("Total NPCs")
    ax.set_title("Success Rate: Variant x NPC Count (LR=0.3)", fontsize=14, fontweight="bold")

    for i in range(len(variants)):
        for j in range(len(npc_counts)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_heatmap.png", bbox_inches="tight")
    print(f"  Saved: {output_dir / 'scaling_heatmap.png'}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot scaling experiment results")
    parser.add_argument("results_file", help="Path to scaling results JSON")
    parser.add_argument("--output-dir", default="artifacts/plots/plots_scaling", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load(args.results_file)
    episodes = data.get("episodes", [])

    if not episodes:
        print("No episodes found.")
        return 1

    print("Generating scaling plots...")
    plot_scaling_success(episodes, out)
    plot_scaling_steps(episodes, out)
    plot_scaling_heatmap(episodes, out)
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
