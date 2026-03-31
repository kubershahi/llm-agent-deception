"""
Generate combined comparison plots: mock (ceiling) vs real LLM results.

Usage:
  python scripts/plot_combined.py
  python scripts/plot_combined.py --mock artifacts/results/results_mock.json --real artifacts/results/results_hybrid.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

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

RESULTS_DIR = Path("artifacts/results")


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract(summary: dict, metric: str) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = defaultdict(dict)
    for key, metrics in summary.items():
        variant, lr_str = key.split("@")
        out[variant][float(lr_str)] = metrics[metric]
    return dict(out)


def _stderr(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance / n)


def _episode_stats(episodes: list[dict], metric_key: str) -> dict[str, dict[float, tuple[float, float]]]:
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
    result: dict[str, dict[float, tuple[float, float]]] = {}
    for variant, lr_map in groups.items():
        result[variant] = {}
        for lr, vals in lr_map.items():
            mean = sum(vals) / len(vals)
            result[variant][lr] = (mean, _stderr(vals))
    return result


def _plot_line(ax, summary, episodes, metric, marker="o"):
    """Plot lines with error bars if episodes available, else plain lines."""
    if episodes:
        stats = _episode_stats(episodes, metric)
        for variant, lr_map in sorted(stats.items()):
            lrs = sorted(lr_map.keys())
            means = [lr_map[lr][0] for lr in lrs]
            errs = [lr_map[lr][1] for lr in lrs]
            ax.errorbar(lrs, means, yerr=errs,
                        marker=marker, linewidth=2.2, markersize=8, capsize=4,
                        color=VARIANT_COLORS.get(variant, "gray"),
                        label=VARIANT_LABELS.get(variant, variant))
    else:
        data = extract(summary, "task_success_rate" if metric == "task_success_rate" else "avg_inference_accuracy")
        for variant, lr_map in sorted(data.items()):
            lrs = sorted(lr_map.keys())
            ax.plot(lrs, [lr_map[lr] for lr in lrs],
                    marker=marker, linewidth=2.2, markersize=8,
                    color=VARIANT_COLORS.get(variant, "gray"),
                    label=VARIANT_LABELS.get(variant, variant))


def fig1_side_by_side_success(mock_sum, real_sum, out, mock_eps=None, real_eps=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _plot_line(ax1, mock_sum, mock_eps, "task_success_rate")
    ax1.set_xlabel("Liar Ratio")
    ax1.set_title("Mock Agent (Deterministic Ceiling)")
    ax1.set_ylim(-0.05, 1.15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=9)
    ax1.set_ylabel("Task Success Rate")

    _plot_line(ax2, real_sum, real_eps, "task_success_rate")
    ax2.set_xlabel("Liar Ratio")
    ax2.set_title("Real LLM Agent (api-gpt-oss-120b)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)

    fig.suptitle("Task Success Rate: Deterministic vs. Real LLM Agent", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "combined_success_rate.png", bbox_inches="tight")
    print(f"  Saved: {out / 'combined_success_rate.png'}")
    plt.close(fig)


def fig2_side_by_side_inference(mock_sum, real_sum, out, mock_eps=None, real_eps=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _plot_line(ax1, mock_sum, mock_eps, "inference_accuracy", marker="s")
    ax1.set_xlabel("Liar Ratio")
    ax1.set_title("Mock Agent (Deterministic)")
    ax1.set_ylim(0.3, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    ax1.set_ylabel("Inference Accuracy (Trust Alignment)")

    _plot_line(ax2, real_sum, real_eps, "inference_accuracy", marker="s")
    ax2.set_xlabel("Liar Ratio")
    ax2.set_title("Real LLM Agent")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle("Trust Score Alignment: Deterministic vs. Real LLM Agent", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "combined_inference_accuracy.png", bbox_inches="tight")
    print(f"  Saved: {out / 'combined_inference_accuracy.png'}")
    plt.close(fig)


def fig3_steps_comparison(mock_sum: dict, real_sum: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, summary, title in [
        (ax1, mock_sum, "Mock Agent"),
        (ax2, real_sum, "Real LLM Agent"),
    ]:
        data = extract(summary, "avg_steps")
        variants = sorted(data.keys())
        liar_ratios = sorted({lr for v in data.values() for lr in v})
        n = len(variants)
        bw = 0.18
        xp = list(range(len(liar_ratios)))
        for i, v in enumerate(variants):
            offsets = [x + i * bw for x in xp]
            vals = [data[v].get(lr, 0) for lr in liar_ratios]
            ax.bar(offsets, vals, bw,
                   color=VARIANT_COLORS.get(v, "gray"),
                   label=VARIANT_LABELS.get(v, v))
        ax.set_xlabel("Liar Ratio")
        ax.set_title(title)
        ax.set_xticks([x + bw * (n - 1) / 2 for x in xp])
        ax.set_xticklabels([f"{lr:.1f}" for lr in liar_ratios])
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="best", fontsize=9)
    ax1.set_ylabel("Average Steps")
    fig.suptitle("Average Steps: Deterministic vs. Real LLM Agent", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "combined_avg_steps.png", bbox_inches="tight")
    print(f"  Saved: {out / 'combined_avg_steps.png'}")
    plt.close(fig)


def fig4_delta_plot(mock_sum: dict, real_sum: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    mock_data = extract(mock_sum, "task_success_rate")
    real_data = extract(real_sum, "task_success_rate")
    for variant in sorted(set(mock_data) & set(real_data)):
        lrs = sorted(set(mock_data[variant]) & set(real_data[variant]))
        deltas = [real_data[variant][lr] - mock_data[variant][lr] for lr in lrs]
        ax.plot(lrs, deltas, marker="D", linewidth=2.2, markersize=8,
                color=VARIANT_COLORS.get(variant, "gray"),
                label=VARIANT_LABELS.get(variant, variant))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Liar Ratio")
    ax.set_ylabel("Success Rate Gap (Real - Mock)")
    ax.set_title("Performance Gap: Real LLM vs. Deterministic Ceiling")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "performance_gap.png", bbox_inches="tight")
    print(f"  Saved: {out / 'performance_gap.png'}")
    plt.close(fig)


def fig5_qualitative_trace(real_episodes: list[dict], out: Path) -> None:
    failed = [e for e in real_episodes if not e["success"]]
    succeeded = [e for e in real_episodes if e["success"] and e["liar_ratio"] >= 0.3]
    if not (failed or succeeded):
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ep_list, title in [
        (axes[0], succeeded[:1] or real_episodes[:1], "Successful Episode"),
        (axes[1], failed[:1] or real_episodes[-1:], "Failed Episode"),
    ]:
        if not ep_list:
            continue
        ep = ep_list[0]
        role_colors = {"truthful": "#2ecc71", "deceptive": "#e74c3c", "opportunistic": "#f39c12",
                       "partial_truth": "#9b59b6", "coordinated_deceptive": "#e67e22"}
        names = list(ep["final_trust_scores"].keys())
        trusts = [ep["final_trust_scores"][n] for n in names]
        colors = [role_colors.get(ep["hidden_roles"].get(n, ""), "gray") for n in names]
        ax.barh(names, trusts, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Final Trust Score")
        ax.set_title(f"{title}\n{ep['agent_variant']} | liar_ratio={ep['liar_ratio']} | steps={ep['steps']}")
        ax.grid(True, alpha=0.3, axis="x")

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=12, label=r)
               for r, c in {"truthful": "#2ecc71", "deceptive": "#e74c3c", "opportunistic": "#f39c12"}.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, title="Hidden NPC Role")
    fig.suptitle("Qualitative Episode Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out / "qualitative_comparison.png", bbox_inches="tight")
    print(f"  Saved: {out / 'qualitative_comparison.png'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate combined comparison plots")
    parser.add_argument("--mock", default=str(RESULTS_DIR / "results_mock.json"), help="Mock results file")
    parser.add_argument("--real", default=str(RESULTS_DIR / "results_hybrid.json"), help="Real LLM results file")
    parser.add_argument("--output-dir", default="artifacts/plots/plots_combined", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    mock = load(args.mock)
    real = load(args.real)

    mock_eps = mock.get("episodes", [])
    real_eps = real.get("episodes", [])

    print("Generating combined plots...")
    fig1_side_by_side_success(mock["summary"], real["summary"], out, mock_eps, real_eps)
    fig2_side_by_side_inference(mock["summary"], real["summary"], out, mock_eps, real_eps)
    fig3_steps_comparison(mock["summary"], real["summary"], out)
    fig4_delta_plot(mock["summary"], real["summary"], out)
    fig5_qualitative_trace(real_eps, out)
    print("Done!")


if __name__ == "__main__":
    main()
