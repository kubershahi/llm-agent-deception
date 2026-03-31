"""
Generate qualitative trace comparison: show side-by-side how a successful
Belief-Tracking episode vs a failed Reflection-Enhanced episode play out.

Usage:
  python scripts/plot_trace_comparison.py artifacts/results/results_hard-hybrid_spread.json
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 10, "figure.dpi": 150})


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def categorize_action(line: str) -> str:
    if "action=talk" in line:
        return "talk"
    if "action=move" in line:
        return "move"
    if "action=search" in line:
        return "search"
    if "action=unlock" in line:
        return "unlock"
    if "action=reflect" in line or "Reflection" in line:
        return "reflect"
    return "other"


ACTION_COLORS = {
    "talk": "#3498db",
    "move": "#95a5a6",
    "search": "#2ecc71",
    "unlock": "#f1c40f",
    "reflect": "#e74c3c",
    "other": "#bdc3c7",
}


def plot_action_timeline(episodes: list[dict], output_dir: Path) -> None:
    """Show action timeline bars for best and worst episodes."""
    # Find a successful belief_tracking at high LR and a failed reflection_enhanced
    bt_success = None
    re_fail = None
    for ep in episodes:
        if ep["agent_variant"] == "belief_tracking" and ep["success"] and ep["liar_ratio"] >= 0.3 and bt_success is None:
            bt_success = ep
        if ep["agent_variant"] == "reflection_enhanced" and not ep["success"] and re_fail is None:
            re_fail = ep

    if not bt_success:
        bt_success = next((e for e in episodes if e["success"]), episodes[0])
    if not re_fail:
        re_fail = next((e for e in episodes if not e["success"]), episodes[-1])

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"hspace": 0.4})

    for ax, ep, title_prefix in [
        (axes[0], bt_success, "SUCCESS"),
        (axes[1], re_fail, "FAIL"),
    ]:
        actions = []
        for line in ep["trace"]:
            if line.startswith("Turn "):
                actions.append(categorize_action(line))

        colors = [ACTION_COLORS.get(a, "#bdc3c7") for a in actions]
        ax.barh([0] * len(actions), [1] * len(actions),
                left=range(len(actions)), color=colors, edgecolor="white", linewidth=0.5, height=0.6)

        ax.set_xlim(-0.5, max(len(actions), 18) + 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Step")
        variant_label = ep["agent_variant"].replace("_", " ").title()
        ax.set_title(
            f"{title_prefix}: {variant_label} | LR={ep['liar_ratio']} | {ep['steps']} steps",
            fontsize=12, fontweight="bold",
        )
        ax.axvline(x=17.5, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="Budget limit (18)")

        for i, a in enumerate(actions):
            ax.text(i + 0.5, 0, a[0].upper(), ha="center", va="center", fontsize=7, fontweight="bold", color="white")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=a.title()) for a, c in ACTION_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Action Timeline: Successful vs. Failed Episode", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "trace_timeline.png", bbox_inches="tight")
    print(f"  Saved: {output_dir / 'trace_timeline.png'}")
    plt.close(fig)


def plot_action_distribution(episodes: list[dict], output_dir: Path) -> None:
    """Stacked bar: action type distribution per variant."""
    from collections import defaultdict, Counter

    variant_actions: dict[str, Counter] = defaultdict(Counter)
    variant_count: dict[str, int] = defaultdict(int)

    for ep in episodes:
        v = ep["agent_variant"]
        variant_count[v] += 1
        for line in ep["trace"]:
            if line.startswith("Turn "):
                variant_actions[v][categorize_action(line)] += 1

    variants = sorted(variant_actions.keys())
    action_types = ["talk", "move", "search", "unlock", "reflect"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = [0.0] * len(variants)

    for action in action_types:
        values = [variant_actions[v][action] / max(variant_count[v], 1) for v in variants]
        labels = [v.replace("_", " ").title() for v in variants]
        ax.bar(labels, values, bottom=bottoms, color=ACTION_COLORS[action], label=action.title())
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_ylabel("Avg Actions per Episode")
    ax.set_title("Action Type Distribution by Agent Variant", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "action_distribution.png", bbox_inches="tight")
    print(f"  Saved: {output_dir / 'action_distribution.png'}")
    plt.close(fig)


def print_trace_comparison(episodes: list[dict]) -> None:
    """Print text traces side by side for the paper."""
    bt = next((e for e in episodes if e["agent_variant"] == "belief_tracking" and e["success"] and e["liar_ratio"] >= 0.3), None)
    re = next((e for e in episodes if e["agent_variant"] == "reflection_enhanced" and not e["success"]), None)

    if bt:
        print(f"\n=== Belief-Tracking SUCCESS (LR={bt['liar_ratio']}, {bt['steps']} steps) ===")
        for line in bt["trace"]:
            print(f"  {line}")
    if re:
        print(f"\n=== Reflection-Enhanced FAIL (LR={re['liar_ratio']}, {re['steps']} steps) ===")
        for line in re["trace"]:
            print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate trace comparison plots")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output-dir", default="artifacts/plots", help="Output directory")
    parser.add_argument("--print-traces", action="store_true", help="Print full text traces")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load(args.results_file)
    episodes = data.get("episodes", [])

    if not episodes:
        print("No episodes found.")
        return 1

    print("Generating trace comparison plots...")
    plot_action_timeline(episodes, out)
    plot_action_distribution(episodes, out)

    if args.print_traces:
        print_trace_comparison(episodes)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
