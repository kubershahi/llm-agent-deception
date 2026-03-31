"""
Scaling experiment: vary total NPCs (4, 6, 8, 10) at fixed liar ratio
to see how agent performance scales with information source density.

Usage:
  python scripts/run_scaling_experiment.py --mode mock --runs 10
  python scripts/run_scaling_experiment.py --mode hard-hybrid --runs 2 --threads 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deceptive_text_env.config import (
    FrameworkConfig,
    build_default_config,
    build_hard_hybrid_config,
)
from deceptive_text_env.evaluation import EvaluationRunner
from deceptive_text_env.llm import enable_call_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NPC scaling experiment")
    parser.add_argument("--mode", choices=["mock", "hard-hybrid"], default="mock")
    parser.add_argument("--variants", nargs="+",
                        default=["naive", "memory_augmented", "belief_tracking", "reflection_enhanced"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--npc-counts", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--liar-ratio", type=float, default=0.3)
    args = parser.parse_args()

    if args.mode != "mock":
        enable_call_logging("artifacts/logs")

    all_results = {}
    all_episodes = []

    for npc_count in args.npc_counts:
        print(f"\n{'='*60}", flush=True)
        print(f"Running with {npc_count} NPCs (LR={args.liar_ratio})", flush=True)
        print(f"{'='*60}", flush=True)

        if args.mode == "hard-hybrid":
            config = build_hard_hybrid_config()
        else:
            config = build_default_config()
            config.experiment.max_steps = 18

        config.experiment.total_npcs = npc_count
        config.experiment.liar_ratios = [args.liar_ratio]
        config.experiment.runs_per_setting = args.runs

        spread = args.mode == "hard-hybrid"
        runner = EvaluationRunner(config, spread_locations=spread)
        results, summary = runner.run_all(args.variants, max_workers=args.threads)

        for key, metrics in summary.items():
            new_key = f"{key}@npcs={npc_count}"
            all_results[new_key] = metrics

        for r in results:
            all_episodes.append({
                "agent_variant": r.agent_variant,
                "liar_ratio": r.liar_ratio,
                "total_npcs": npc_count,
                "success": r.success,
                "steps": r.steps,
                "inference_accuracy": r.inference_accuracy,
                "recovery_rate": r.recovery_rate,
                "final_trust_scores": r.final_trust_scores,
                "hidden_roles": r.hidden_roles,
                "trace": r.trace,
            })

        successes = sum(1 for r in results if r.success)
        print(f"  {npc_count} NPCs: {successes}/{len(results)} success", flush=True)

    results_dir = Path("artifacts/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"results_scaling_{args.mode}.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "mode": args.mode,
                "npc_counts": args.npc_counts,
                "liar_ratio": args.liar_ratio,
                "runs_per_setting": args.runs,
                "variants": args.variants,
            },
            "summary": all_results,
            "episodes": all_episodes,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Scaling Summary (LR={args.liar_ratio})")
    print(f"{'='*60}")
    print(f"{'Variant':<25} " + " ".join(f"NPCs={n:<3}" for n in args.npc_counts))
    print("-" * (25 + 10 * len(args.npc_counts)))
    for variant in args.variants:
        row = f"{variant:<25} "
        for n in args.npc_counts:
            key = f"{variant}@{args.liar_ratio}@npcs={n}"
            rate = all_results.get(key, {}).get("task_success_rate", -1)
            if rate < 0:
                row += f"{'N/A':<10}"
            else:
                row += f"{rate:<10.2f}"
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
