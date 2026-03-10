"""
Plot cross-model ablation results.

Usage:
  python plot_cross_model.py results_cross_model.json --baseline results_hard-hybrid_spread.json
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

VARIANT_LABELS = {
    "naive": "Naive",
    "memory_augmented": "Memory-Aug.",
    "belief_tracking": "Belief-Track.",
    "reflection_enhanced": "Reflect.-Enh.",
    "belief_no_decay": "Belief (No Decay)",
    "memory_with_trust": "Memory + Trust",
}

MODEL_COLORS = {
    "api-gpt-oss-120b": "#2ecc71",
    "api-llama-4-scout": "#3498db",
    "mistral.mistral-large-3-675b-instruct": "#e74c3c",
}
MODEL_SHORT = {
    "api-gpt-oss-120b": "GPT-OSS-120B",
    "api-llama-4-scout": "Llama-4-Scout",
    "mistral.mistral-large-3-675b-instruct": "Mistral-675B",
}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_cross_model_success(cross_episodes: list[dict], baseline_episodes: list[dict], output_dir: Path) -> None:
    """Grouped bar chart: success rate by model and variant."""
    # Combine baseline (gpt-oss-120b) with cross-model data
    all_episodes = []
    for ep in baseline_episodes:
        ep_copy = dict(ep)
        ep_copy["model"] = "api-gpt-oss-120b"
        all_episodes.append(ep_copy)
    all_episodes.extend(cross_episodes)

    # Group by model -> variant
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in all_episodes:
        model = ep.get("model", "api-gpt-oss-120b")
        groups[model][ep["agent_variant"]].append(1.0 if ep["success"] else 0.0)

    models = sorted(groups.keys(), key=lambda m: MODEL_SHORT.get(m, m))
    variants = ["naive", "belief_tracking", "reflection_enhanced", "memory_with_trust"]
    variants = [v for v in variants if any(v in groups[m] for m in models)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(variants))
    n_models = len(models)
    bw = 0.8 / n_models

    for i, model in enumerate(models):
        means = []
        errs = []
        for v in variants:
            vals = groups[model].get(v, [])
            if vals:
                means.append(sum(vals) / len(vals))
                n = len(vals)
                if n > 1:
                    mean = sum(vals) / n
                    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
                    errs.append(math.sqrt(var / n))
                else:
                    errs.append(0)
            else:
                means.append(0)
                errs.append(0)

        ax.bar(x + i * bw, means, bw, yerr=errs, capsize=3,
               color=MODEL_COLORS.get(model, "gray"),
               label=MODEL_SHORT.get(model, model),
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + bw * (n_models - 1) / 2)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Cross-Model Comparison: Task Success Rate (Hard Mode, All LR)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "cross_model_success.png", bbox_inches="tight")
    print(f"  Saved: {output_dir / 'cross_model_success.png'}")
    plt.close(fig)


def plot_cross_model_heatmap(cross_episodes: list[dict], baseline_episodes: list[dict], output_dir: Path) -> None:
    """Heatmap: model x variant, overall success rate."""
    all_episodes = []
    for ep in baseline_episodes:
        ep_copy = dict(ep)
        ep_copy["model"] = "api-gpt-oss-120b"
        all_episodes.append(ep_copy)
    all_episodes.extend(cross_episodes)

    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in all_episodes:
        model = ep.get("model", "api-gpt-oss-120b")
        groups[model][ep["agent_variant"]].append(1.0 if ep["success"] else 0.0)

    models = sorted(groups.keys(), key=lambda m: MODEL_SHORT.get(m, m))
    variants = ["naive", "belief_tracking", "reflection_enhanced", "memory_with_trust"]
    variants = [v for v in variants if any(v in groups[m] for m in models)]

    matrix = []
    for model in models:
        row = []
        for v in variants:
            vals = groups[model].get(v, [])
            row.append(sum(vals) / len(vals) if vals else 0)
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_SHORT.get(m, m) for m in models])
    ax.set_title("Cross-Model Success Rate Heatmap (Hard Mode)", fontsize=14, fontweight="bold")

    for i in range(len(models)):
        for j in range(len(variants)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_model_heatmap.png", bbox_inches="tight")
    print(f"  Saved: {output_dir / 'cross_model_heatmap.png'}")
    plt.close(fig)


def plot_cross_model_by_lr(cross_episodes: list[dict], baseline_episodes: list[dict], output_dir: Path) -> None:
    """Line plot: success rate vs LR for each model (belief_tracking only)."""
    all_episodes = []
    for ep in baseline_episodes:
        ep_copy = dict(ep)
        ep_copy["model"] = "api-gpt-oss-120b"
        all_episodes.append(ep_copy)
    all_episodes.extend(cross_episodes)

    # Focus on belief_tracking to isolate model effect
    groups: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in all_episodes:
        if ep["agent_variant"] == "belief_tracking":
            model = ep.get("model", "api-gpt-oss-120b")
            groups[model][ep["liar_ratio"]].append(1.0 if ep["success"] else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, lr_map in sorted(groups.items()):
        lrs = sorted(lr_map.keys())
        means = [sum(lr_map[lr]) / len(lr_map[lr]) for lr in lrs]
        ax.plot(lrs, means, marker="o", linewidth=2.2, markersize=8,
                color=MODEL_COLORS.get(model, "gray"),
                label=MODEL_SHORT.get(model, model))

    ax.set_xlabel("Liar Ratio")
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Belief-Tracking Agent: Cross-Model Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_model_by_lr.png", bbox_inches="tight")
    print(f"  Saved: {output_dir / 'cross_model_by_lr.png'}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot cross-model results")
    parser.add_argument("results_file", help="Cross-model results JSON")
    parser.add_argument("--baseline", default="results_hard-hybrid_spread.json", help="Baseline (gpt-oss-120b) results")
    parser.add_argument("--output-dir", default="plots_cross_model", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cross = load(args.results_file)
    baseline = load(args.baseline)

    cross_eps = cross.get("episodes", [])
    baseline_eps = baseline.get("episodes", [])

    # Filter baseline to only include the 4 variants used in cross-model
    cross_variants = set(ep["agent_variant"] for ep in cross_eps)
    baseline_eps = [ep for ep in baseline_eps if ep["agent_variant"] in cross_variants]

    print("Generating cross-model plots...")
    plot_cross_model_success(cross_eps, baseline_eps, out)
    plot_cross_model_heatmap(cross_eps, baseline_eps, out)
    plot_cross_model_by_lr(cross_eps, baseline_eps, out)
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
