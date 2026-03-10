"""
Generate final publication-quality figures for the CSE 291A project.

Figures:
  1. Default world: success rate vs liar ratio (with baselines + error bars)
  2. Extended world: success rate vs liar ratio (with baselines + error bars)
  3. Side-by-side: default vs extended world comparison
  4. Step efficiency: grouped bar chart (default world, LLM variants only)
  5. Hint ablation: grouped bar chart (GPT-OSS and Llama, no-hints vs hints)
  6. Cross-model heatmap: variant x model success rate

Usage:
  python plot_final.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.size": 13,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})

OUTPUT_DIR = Path("plots_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Distinct, colorblind-friendlier palette
VARIANT_COLORS = {
    "oracle": "#7f8c8d",
    "random": "#bdc3c7",
    "naive": "#e74c3c",          # red
    "belief_tracking": "#2980b9", # blue
    "reflection_enhanced": "#f39c12",  # orange
    "memory_with_trust": "#27ae60",    # green
}
VARIANT_LABELS = {
    "oracle": "Oracle (upper bound)",
    "random": "Random (lower bound)",
    "naive": "Naive",
    "belief_tracking": "Belief-Tracking",
    "reflection_enhanced": "Reflection-Enhanced",
    "memory_with_trust": "Memory + Trust",
}
VARIANT_MARKERS = {
    "oracle": "^",
    "random": "v",
    "naive": "o",
    "belief_tracking": "s",
    "reflection_enhanced": "D",
    "memory_with_trust": "P",
}
LLM_VARIANTS = ["naive", "belief_tracking", "reflection_enhanced", "memory_with_trust"]
# For line plots: draw LLM variants on top of baselines
PLOT_ORDER = ["oracle", "random", "naive", "reflection_enhanced", "belief_tracking", "memory_with_trust"]


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def episode_stats(episodes: list[dict]) -> dict[str, dict[float, tuple[float, float]]]:
    """Return {variant: {lr: (mean_success, stderr)}}."""
    groups: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        groups[ep["agent_variant"]][ep["liar_ratio"]].append(1.0 if ep["success"] else 0.0)
    out = {}
    for v, lr_map in groups.items():
        out[v] = {}
        for lr, vals in lr_map.items():
            n = len(vals)
            mean = sum(vals) / n
            se = math.sqrt(mean * (1 - mean) / n) if n > 1 else 0
            out[v][lr] = (mean, se)
    return out


def step_stats(episodes: list[dict]) -> dict[str, dict[float, tuple[float, float]]]:
    """Return {variant: {lr: (mean_steps, stderr)}}."""
    groups: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        groups[ep["agent_variant"]][ep["liar_ratio"]].append(ep["steps"])
    out = {}
    for v, lr_map in groups.items():
        out[v] = {}
        for lr, vals in lr_map.items():
            n = len(vals)
            mean = sum(vals) / n
            variance = sum((x - mean) ** 2 for x in vals) / max(n - 1, 1)
            out[v][lr] = (mean, math.sqrt(variance / n))
    return out


def _draw_success_plot(ax, episodes, baselines_as_bands=True):
    """Draw success rate lines on an axis. Baselines as shaded bands, LLM variants as lines."""
    stats = episode_stats(episodes)
    all_lrs = sorted({lr for v in stats.values() for lr in v})

    if baselines_as_bands:
        # Oracle as green band at top
        ax.axhspan(0.97, 1.03, color="#27ae60", alpha=0.10)
        ax.axhline(y=1.0, color="#7f8c8d", linestyle=":", linewidth=1.2, alpha=0.6)
        ax.text(all_lrs[-1] + 0.015, 1.0, "Oracle", fontsize=9, color="#7f8c8d",
                va="center", fontstyle="italic")
        # Random as red band at bottom
        ax.axhspan(-0.03, 0.03, color="#e74c3c", alpha=0.08)
        ax.axhline(y=0.0, color="#bdc3c7", linestyle=":", linewidth=1.2, alpha=0.6)
        ax.text(all_lrs[-1] + 0.015, 0.0, "Random", fontsize=9, color="#bdc3c7",
                va="center", fontstyle="italic")

    # Draw LLM variants with slight y-jitter for overlapping lines
    jitter = {"belief_tracking": 0.015, "memory_with_trust": -0.015}
    for v in PLOT_ORDER:
        if v not in stats or v in ("oracle", "random"):
            continue
        lr_map = stats[v]
        lrs = sorted(lr_map.keys())
        means = [lr_map[lr][0] + jitter.get(v, 0) for lr in lrs]
        errs = [lr_map[lr][1] for lr in lrs]
        ax.errorbar(
            lrs, means, yerr=errs,
            marker=VARIANT_MARKERS[v], linewidth=2.5, markersize=10, capsize=5,
            color=VARIANT_COLORS[v], label=VARIANT_LABELS[v],
            linestyle="-", alpha=0.9, markeredgecolor="white", markeredgewidth=1.0,
            zorder=5,
        )


# ── Figure 1: Default World Success Rate ──────────────────────────────────────
def fig1_default_success(episodes: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _draw_success_plot(ax, episodes)

    ax.set_xlabel("Liar Ratio", fontsize=13)
    ax.set_ylabel("Task Success Rate", fontsize=13)
    ax.set_title("Default World — Task Success vs. Deception Level\n"
                 "5 locations, 3 sigils, 18-step budget, GPT-OSS-120B", fontsize=13)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xlim(-0.03, 0.78)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="center left", fontsize=11, framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_default_success.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig1_default_success.pdf", bbox_inches="tight")
    print("  Saved fig1_default_success")
    plt.close(fig)


# ── Figure 2: Extended World Success Rate ─────────────────────────────────────
def fig2_extended_success(episodes: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _draw_success_plot(ax, episodes)

    ax.set_xlabel("Liar Ratio", fontsize=13)
    ax.set_ylabel("Task Success Rate", fontsize=13)
    ax.set_title("Extended World — Task Success vs. Deception Level\n"
                 "7 locations, 4 sigils, 25-step budget, GPT-OSS-120B", fontsize=13)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xlim(-0.03, 0.78)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="center left", fontsize=11, framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_extended_success.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig2_extended_success.pdf", bbox_inches="tight")
    print("  Saved fig2_extended_success")
    plt.close(fig)


# ── Figure 3: Side-by-side default vs extended ────────────────────────────────
def fig3_side_by_side(default_eps: list[dict], extended_eps: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6), sharey=True)

    for ax, episodes, title in [
        (ax1, default_eps, "Default World\n(5 loc, 3 sigils, 18 steps)"),
        (ax2, extended_eps, "Extended World\n(7 loc, 4 sigils, 25 steps)"),
    ]:
        _draw_success_plot(ax, episodes)
        ax.set_xlabel("Liar Ratio", fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(-0.12, 1.12)
        ax.set_xlim(-0.03, 0.78)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.15, axis="y")

    ax1.set_ylabel("Task Success Rate", fontsize=13)
    # Single shared legend between both panels
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11,
               framealpha=0.95, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Environment Complexity Comparison — GPT-OSS-120B, No Hints",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(OUTPUT_DIR / "fig3_side_by_side.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig3_side_by_side.pdf", bbox_inches="tight")
    print("  Saved fig3_side_by_side")
    plt.close(fig)


# ── Figure 4: Step Efficiency (LLM variants only, zoomed y-axis) ─────────────
def fig4_step_efficiency(episodes: list[dict]) -> None:
    stats = step_stats(episodes)
    # Only LLM variants — baselines are constant (oracle=15, random=18)
    variants_to_plot = [v for v in LLM_VARIANTS if v in stats]
    liar_ratios = sorted({lr for v in stats.values() for lr in v})

    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_variants = len(variants_to_plot)
    bar_width = 0.8 / n_variants
    x = np.arange(len(liar_ratios))

    for i, v in enumerate(variants_to_plot):
        means = [stats[v].get(lr, (0, 0))[0] for lr in liar_ratios]
        errs = [stats[v].get(lr, (0, 0))[1] for lr in liar_ratios]
        ax.bar(
            x + i * bar_width, means, bar_width,
            yerr=errs, capsize=3,
            color=VARIANT_COLORS[v], label=VARIANT_LABELS[v],
            edgecolor="white", linewidth=0.8,
        )

    # Reference lines
    ax.axhline(y=15, color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(-0.4, 15.05, "Optimal (15)", fontsize=10, color="#27ae60", fontstyle="italic")
    ax.axhline(y=18, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.text(-0.4, 18.05, "Budget (18)", fontsize=10, color="#e74c3c", fontstyle="italic")

    ax.set_xlabel("Liar Ratio", fontsize=13)
    ax.set_ylabel("Average Steps", fontsize=13)
    ax.set_title("Step Efficiency — Default World (GPT-OSS-120B)", fontsize=13)
    ax.set_xticks(x + bar_width * (n_variants - 1) / 2)
    ax.set_xticklabels([f"{lr:.1f}" for lr in liar_ratios])
    ax.set_ylim(13.5, 19.0)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_step_efficiency.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4_step_efficiency.pdf", bbox_inches="tight")
    print("  Saved fig4_step_efficiency")
    plt.close(fig)


# ── Figure 5: Hint Ablation ──────────────────────────────────────────────────
def fig5_hint_ablation(
    gptoss_nohints: list[dict], gptoss_hints: list[dict],
    llama_nohints: list[dict], llama_hints: list[dict],
) -> None:
    def overall_success(episodes: list[dict]) -> dict[str, float]:
        groups: dict[str, list[float]] = defaultdict(list)
        for ep in episodes:
            groups[ep["agent_variant"]].append(1.0 if ep["success"] else 0.0)
        return {v: sum(vals) / len(vals) for v, vals in groups.items()}

    gpt_no = overall_success(gptoss_nohints)
    gpt_yes = overall_success(gptoss_hints)
    llama_no = overall_success(llama_nohints)
    llama_yes = overall_success(llama_hints)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    variants = LLM_VARIANTS
    x = np.arange(len(variants))
    bar_w = 0.35

    for ax, no_data, yes_data, title in [
        (ax1, gpt_no, gpt_yes, "GPT-OSS-120B"),
        (ax2, llama_no, llama_yes, "Llama-4-Scout"),
    ]:
        no_vals = [no_data.get(v, 0) for v in variants]
        yes_vals = [yes_data.get(v, 0) for v in variants]

        # Draw bars
        ax.bar(x - bar_w / 2, no_vals, bar_w, label="No Hints",
               color="#c0392b", alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.bar(x + bar_w / 2, yes_vals, bar_w, label="With Hints",
               color="#27ae60", alpha=0.85, edgecolor="white", linewidth=0.8)

        # Add value labels on bars
        for i, (nv, yv) in enumerate(zip(no_vals, yes_vals)):
            # No-hint value
            if nv > 0.03:
                ax.text(x[i] - bar_w / 2, nv + 0.02, f"{nv:.0%}",
                        ha="center", va="bottom", fontsize=9, color="#c0392b", fontweight="bold")
            else:
                ax.text(x[i] - bar_w / 2, 0.02, "0%",
                        ha="center", va="bottom", fontsize=9, color="#c0392b", fontweight="bold")
            # Hint value
            if yv > 0.03:
                ax.text(x[i] + bar_w / 2, yv + 0.02, f"{yv:.0%}",
                        ha="center", va="bottom", fontsize=9, color="#1e8449", fontweight="bold")
            else:
                ax.text(x[i] + bar_w / 2, 0.02, "0%",
                        ha="center", va="bottom", fontsize=9, color="#1e8449", fontweight="bold")
            # Delta arrow
            delta = yv - nv
            if delta > 0.05:
                mid_y = max(nv, yv) + 0.10
                ax.annotate(f"+{delta:.0%}", xy=(x[i], mid_y),
                            ha="center", fontsize=11, fontweight="bold", color="#2c3e50",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.9))

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS[v].replace(" ", "\n") for v in variants],
                           fontsize=10)
        ax.set_ylim(0, 1.25)
        ax.legend(fontsize=11, loc="upper right", framealpha=0.95, edgecolor="#cccccc")
        ax.grid(True, alpha=0.15, axis="y")

    ax1.set_ylabel("Overall Success Rate", fontsize=13)
    fig.suptitle("Hint Ablation — Planning vs. Reasoning Decomposition",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_hint_ablation.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig5_hint_ablation.pdf", bbox_inches="tight")
    print("  Saved fig5_hint_ablation")
    plt.close(fig)


# ── Figure 6: Cross-Model Heatmap ────────────────────────────────────────────
def fig6_cross_model_heatmap(
    gptoss_eps: list[dict], llama_eps: list[dict],
) -> None:
    def variant_success(episodes: list[dict]) -> dict[str, float]:
        groups: dict[str, list[float]] = defaultdict(list)
        for ep in episodes:
            groups[ep["agent_variant"]].append(1.0 if ep["success"] else 0.0)
        return {v: sum(vals) / len(vals) for v, vals in groups.items()}

    gpt = variant_success(gptoss_eps)
    llama = variant_success(llama_eps)

    variants = LLM_VARIANTS
    models = ["GPT-OSS-120B", "Llama-4-Scout"]
    data = np.array([
        [gpt.get(v, 0) for v in variants],
        [llama.get(v, 0) for v in variants],
    ])

    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants], fontsize=12)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=13, fontweight="bold")

    # Cell labels
    for i in range(len(models)):
        for j in range(len(variants)):
            val = data[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)

    # Add cell borders
    for i in range(len(models)):
        for j in range(len(variants)):
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                 edgecolor="white", linewidth=2)
            ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Success Rate", fontsize=12)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.set_title("Cross-Model Comparison — No Hints, Default World",
                 fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_cross_model_heatmap.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig6_cross_model_heatmap.pdf", bbox_inches="tight")
    print("  Saved fig6_cross_model_heatmap")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Output directory: {OUTPUT_DIR}")

    default_data = load("results_hard-hybrid_spread.json")
    default_eps = default_data["episodes"]
    print(f"  Default world: {len(default_eps)} episodes")

    extended_data = load("results_extended.json")
    extended_eps = extended_data["episodes"]
    print(f"  Extended world: {len(extended_eps)} episodes")

    hints_gptoss = load("results_hints_gptoss.json")
    hints_gptoss_eps = hints_gptoss["episodes"]

    nohints_llama = load("results_nohints_llama.json")
    nohints_llama_eps = nohints_llama["episodes"]

    hints_llama = load("results_hints_llama.json")
    hints_llama_eps = hints_llama["episodes"]

    print("\nGenerating figures...")
    fig1_default_success(default_eps)
    fig2_extended_success(extended_eps)
    fig3_side_by_side(default_eps, extended_eps)
    fig4_step_efficiency(default_eps)
    fig5_hint_ablation(default_eps, hints_gptoss_eps, nohints_llama_eps, hints_llama_eps)
    fig6_cross_model_heatmap(default_eps, nohints_llama_eps)
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
