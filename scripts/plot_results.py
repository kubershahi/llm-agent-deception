"""
Generate publication-quality plots from experiment results.

Usage:
  python scripts/plot_results.py artifacts/results/results_mock.json
  python scripts/plot_results.py artifacts/results/results_extended.json --output-dir artifacts/plots
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
    "memory_augmented": "Memory-Augmented",
    "belief_tracking": "Belief-Tracking",
    "reflection_enhanced": "Reflection-Enhanced",
    "belief_no_decay": "Belief (No Decay)",
    "memory_with_trust": "Memory + Trust",
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _stderr(values: list[float]) -> float:
    """Standard error of the mean."""
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance / n)


def _episode_stats(episodes: list[dict], metric_key: str) -> dict[str, dict[float, tuple[float, float]]]:
    """Group episodes by variant+liar_ratio, return {variant: {lr: (mean, stderr)}}."""
    groups: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        variant = ep["agent_variant"]
        lr = ep["liar_ratio"]
        if metric_key == "task_success_rate":
            groups[variant][lr].append(1.0 if ep["success"] else 0.0)
        elif metric_key == "steps":
            groups[variant][lr].append(ep["steps"])
        elif metric_key == "inference_accuracy":
            groups[variant][lr].append(ep["inference_accuracy"])
        elif metric_key == "recovery_rate" and (ep.get("recovery_rate") or -1) >= 0:
            groups[variant][lr].append(ep["recovery_rate"])
    result: dict[str, dict[float, tuple[float, float]]] = {}
    for variant, lr_map in groups.items():
        result[variant] = {}
        for lr, vals in lr_map.items():
            mean = sum(vals) / len(vals)
            result[variant][lr] = (mean, _stderr(vals))
    return result


def plot_task_success_rate(summary: dict, output_dir: Path, episodes: list[dict] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    if episodes and len(episodes) > 0:
        stats = _episode_stats(episodes, "task_success_rate")
        for variant, lr_map in sorted(stats.items()):
            lrs = sorted(lr_map.keys())
            means = [lr_map[lr][0] for lr in lrs]
            errs = [lr_map[lr][1] for lr in lrs]
            ax.errorbar(
                lrs, means, yerr=errs,
                marker="o", linewidth=2, markersize=8, capsize=4,
                color=VARIANT_COLORS.get(variant, "gray"),
                label=VARIANT_LABELS.get(variant, variant),
            )
    else:
        variant_data: dict[str, dict[float, float]] = defaultdict(dict)
        for key, metrics in summary.items():
            variant, lr_str = key.split("@")
            variant_data[variant][float(lr_str)] = metrics["task_success_rate"]
        for variant, lr_map in sorted(variant_data.items()):
            lrs = sorted(lr_map.keys())
            ax.plot(lrs, [lr_map[lr] for lr in lrs],
                    marker="o", linewidth=2, markersize=8,
                    color=VARIANT_COLORS.get(variant, "gray"),
                    label=VARIANT_LABELS.get(variant, variant))

    ax.set_xlabel("Liar Ratio")
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Task Success Rate vs. Deception Level")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "task_success_rate.png")
    print(f"  Saved: {output_dir / 'task_success_rate.png'}")
    plt.close(fig)


def plot_inference_accuracy(summary: dict, output_dir: Path, episodes: list[dict] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    if episodes and len(episodes) > 0:
        stats = _episode_stats(episodes, "inference_accuracy")
        for variant, lr_map in sorted(stats.items()):
            lrs = sorted(lr_map.keys())
            means = [lr_map[lr][0] for lr in lrs]
            errs = [lr_map[lr][1] for lr in lrs]
            ax.errorbar(
                lrs, means, yerr=errs,
                marker="s", linewidth=2, markersize=8, capsize=4,
                color=VARIANT_COLORS.get(variant, "gray"),
                label=VARIANT_LABELS.get(variant, variant),
            )
    else:
        variant_data: dict[str, dict[float, float]] = defaultdict(dict)
        for key, metrics in summary.items():
            variant, lr_str = key.split("@")
            variant_data[variant][float(lr_str)] = metrics["avg_inference_accuracy"]
        for variant, lr_map in sorted(variant_data.items()):
            lrs = sorted(lr_map.keys())
            ax.plot(lrs, [lr_map[lr] for lr in lrs],
                    marker="s", linewidth=2, markersize=8,
                    color=VARIANT_COLORS.get(variant, "gray"),
                    label=VARIANT_LABELS.get(variant, variant))

    ax.set_xlabel("Liar Ratio")
    ax.set_ylabel("Inference Accuracy (Trust Alignment)")
    ax.set_title("Trust Score Alignment with Hidden NPC Roles")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_accuracy.png")
    print(f"  Saved: {output_dir / 'inference_accuracy.png'}")
    plt.close(fig)


def plot_avg_steps(summary: dict, output_dir: Path, episodes: list[dict] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    if episodes and len(episodes) > 0:
        stats = _episode_stats(episodes, "steps")
        variants = sorted(stats.keys())
        liar_ratios = sorted({lr for v in stats.values() for lr in v})
        n_variants = len(variants)
        bar_width = 0.8 / max(n_variants, 1)
        x_positions = list(range(len(liar_ratios)))

        for i, variant in enumerate(variants):
            offsets = [x + i * bar_width for x in x_positions]
            means = [stats[variant].get(lr, (0, 0))[0] for lr in liar_ratios]
            errs = [stats[variant].get(lr, (0, 0))[1] for lr in liar_ratios]
            ax.bar(offsets, means, bar_width, yerr=errs, capsize=3,
                   color=VARIANT_COLORS.get(variant, "gray"),
                   label=VARIANT_LABELS.get(variant, variant))
    else:
        variant_data: dict[str, dict[float, float]] = defaultdict(dict)
        for key, metrics in summary.items():
            variant, lr_str = key.split("@")
            variant_data[variant][float(lr_str)] = metrics["avg_steps"]
        variants = sorted(variant_data.keys())
        liar_ratios = sorted({lr for v in variant_data.values() for lr in v})
        n_variants = len(variants)
        bar_width = 0.18
        x_positions = list(range(len(liar_ratios)))
        for i, variant in enumerate(variants):
            offsets = [x + i * bar_width for x in x_positions]
            values = [variant_data[variant].get(lr, 0) for lr in liar_ratios]
            ax.bar(offsets, values, bar_width,
                   color=VARIANT_COLORS.get(variant, "gray"),
                   label=VARIANT_LABELS.get(variant, variant))

    ax.set_xlabel("Liar Ratio")
    ax.set_ylabel("Average Steps")
    ax.set_title("Average Steps to Completion")
    ax.set_xticks([x + bar_width * (n_variants - 1) / 2 for x in x_positions])
    ax.set_xticklabels([f"{lr:.1f}" for lr in liar_ratios])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "avg_steps.png")
    print(f"  Saved: {output_dir / 'avg_steps.png'}")
    plt.close(fig)


def plot_recovery_rate(summary: dict, output_dir: Path, episodes: list[dict] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    if episodes and len(episodes) > 0:
        stats = _episode_stats(episodes, "recovery_rate")
        if not stats:
            print("  No recovery rate data to plot.")
            plt.close(fig)
            return
        variants = sorted(stats.keys())
        liar_ratios = sorted({lr for v in stats.values() for lr in v})
        n_variants = len(variants)
        bar_width = 0.8 / max(n_variants, 1)
        x_positions = list(range(len(liar_ratios)))

        for i, variant in enumerate(variants):
            offsets = [x + i * bar_width for x in x_positions]
            means = [stats[variant].get(lr, (0, 0))[0] for lr in liar_ratios]
            errs = [stats[variant].get(lr, (0, 0))[1] for lr in liar_ratios]
            ax.bar(offsets, means, bar_width, yerr=errs, capsize=3,
                   color=VARIANT_COLORS.get(variant, "gray"),
                   label=VARIANT_LABELS.get(variant, variant))
    else:
        variant_data: dict[str, dict[float, float]] = defaultdict(dict)
        for key, metrics in summary.items():
            variant, lr_str = key.split("@")
            val = metrics.get("avg_recovery_rate", -1.0)
            if val >= 0:
                variant_data[variant][float(lr_str)] = val
        if not variant_data:
            print("  No recovery rate data to plot.")
            plt.close(fig)
            return
        variants = sorted(variant_data.keys())
        liar_ratios = sorted({lr for v in variant_data.values() for lr in v})
        n_variants = len(variants)
        bar_width = 0.18
        x_positions = list(range(len(liar_ratios)))
        for i, variant in enumerate(variants):
            offsets = [x + i * bar_width for x in x_positions]
            values = [variant_data[variant].get(lr, 0) for lr in liar_ratios]
            ax.bar(offsets, values, bar_width,
                   color=VARIANT_COLORS.get(variant, "gray"),
                   label=VARIANT_LABELS.get(variant, variant))

    ax.set_xlabel("Liar Ratio")
    ax.set_ylabel("Avg Turns to Distrust Liar")
    ax.set_title("Recovery Rate (Lower is Better)")
    ax.set_xticks([x + bar_width * (n_variants - 1) / 2 for x in x_positions])
    ax.set_xticklabels([f"{lr:.1f}" for lr in liar_ratios])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "recovery_rate.png")
    print(f"  Saved: {output_dir / 'recovery_rate.png'}")
    plt.close(fig)


def plot_trust_trajectories(episodes: list[dict], output_dir: Path) -> None:
    variant_episodes: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        variant_episodes[ep["agent_variant"]].append(ep)

    for variant, eps in sorted(variant_episodes.items()):
        fig, ax = plt.subplots(figsize=(8, 4))
        role_colors = {"truthful": "#2ecc71", "deceptive": "#e74c3c", "opportunistic": "#f39c12",
                       "partial_truth": "#9b59b6", "coordinated_deceptive": "#e67e22"}

        for ep in eps[:5]:
            for npc_name, trust in ep["final_trust_scores"].items():
                role = ep["hidden_roles"].get(npc_name, "unknown")
                ax.scatter(
                    ep["liar_ratio"], trust,
                    color=role_colors.get(role, "gray"),
                    alpha=0.6, s=60, edgecolors="black", linewidths=0.5,
                )

        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=r)
            for r, c in role_colors.items()
        ]
        ax.legend(handles=handles, loc="best", title="NPC Role")
        ax.set_xlabel("Liar Ratio")
        ax.set_ylabel("Final Trust Score")
        ax.set_title(f"Final Trust Scores — {VARIANT_LABELS.get(variant, variant)}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"trust_scatter_{variant}.png")
        print(f"  Saved: {output_dir / f'trust_scatter_{variant}.png'}")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output-dir", default="artifacts/plots", help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(args.results_file)
    summary = data["summary"]
    episodes = data.get("episodes", [])

    print("Generating plots...")
    plot_task_success_rate(summary, output_dir, episodes)
    plot_inference_accuracy(summary, output_dir, episodes)
    plot_avg_steps(summary, output_dir, episodes)
    plot_recovery_rate(summary, output_dir, episodes)
    if episodes:
        plot_trust_trajectories(episodes, output_dir)
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
