"""
Run experiments using the UCSD TritonAI API with a real LLM.

Usage:
  # Set your API key first:
  export TRITONAI_API_KEY="your-key-here"

  # Full real LLM (agent + NPCs + judge all use TritonAI):
  python run_tritonai_experiment.py --mode full

  # Hybrid (real LLM agent + mock NPCs, cheaper for iteration):
  python run_tritonai_experiment.py --mode hybrid

  # Hard mode (reduced step budget + spread NPCs):
  python run_tritonai_experiment.py --mode hard-hybrid --spread-npcs

  # Mock only (no API calls, fast local test):
  python run_tritonai_experiment.py --mode mock
"""
from __future__ import annotations

import argparse
import json
import sys

from deceptive_text_env.config import (
    build_default_config,
    build_hard_config,
    build_hard_hybrid_config,
    build_hybrid_config,
    build_tritonai_config,
)
from deceptive_text_env.evaluation import EvaluationRunner
from deceptive_text_env.llm import enable_call_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deceptive NPC experiments")
    parser.add_argument(
        "--mode",
        choices=["mock", "hybrid", "full", "hard-hybrid", "hard-full"],
        default="mock",
        help="mock=no API calls, hybrid=real agent + mock NPCs, full=all real LLM, "
             "hard-hybrid/hard-full=reduced step budget (18 steps)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["naive", "memory_augmented", "belief_tracking", "reflection_enhanced"],
        help="Agent variants to test",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override runs_per_setting (default: 10 for mock, 3 for real)",
    )
    parser.add_argument(
        "--advanced-npcs",
        action="store_true",
        help="Use advanced NPC strategies (partial_truth, coordinated_deceptive)",
    )
    parser.add_argument(
        "--spread-npcs",
        action="store_true",
        help="Spread NPCs across world locations instead of all at village_square",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max step budget (default: 24 for normal, 18 for hard modes)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel threads for running episodes (default: 1)",
    )
    parser.add_argument(
        "--use-hints",
        action="store_true",
        help="Enable PRIORITY_ACTION payload hints (for hint ablation experiment)",
    )
    parser.add_argument(
        "--liar-ratios",
        nargs="+",
        type=float,
        default=None,
        help="Override liar ratios (e.g. --liar-ratios 0.0 0.3 0.5 0.7)",
    )
    args = parser.parse_args()

    if args.mode == "full":
        config = build_tritonai_config()
    elif args.mode == "hybrid":
        config = build_hybrid_config()
    elif args.mode == "hard-full":
        config = build_hard_config()
    elif args.mode == "hard-hybrid":
        config = build_hard_hybrid_config()
    else:
        config = build_default_config()

    if args.mode != "mock":
        enable_call_logging("llm_logs")

    if args.runs is not None:
        config.experiment.runs_per_setting = args.runs

    if args.max_steps is not None:
        config.experiment.max_steps = args.max_steps

    if args.liar_ratios is not None:
        config.experiment.liar_ratios = args.liar_ratios

    config.experiment.use_hints = args.use_hints

    spread = args.spread_npcs or args.mode.startswith("hard")

    print(f"Mode: {args.mode}")
    print(f"Agent model: {config.premium_agent_model.provider}/{config.premium_agent_model.model_name}")
    print(f"NPC model: {config.budget_npc_model.provider}/{config.budget_npc_model.model_name}")
    print(f"Variants: {args.variants}")
    print(f"Liar ratios: {config.experiment.liar_ratios}")
    print(f"Runs per setting: {config.experiment.runs_per_setting}")
    print(f"Max steps: {config.experiment.max_steps}")
    print(f"Spread NPCs: {spread}")
    print(f"Advanced NPCs: {args.advanced_npcs}")
    print(f"Use hints: {args.use_hints}")
    print()

    runner = EvaluationRunner(config, use_advanced_npcs=args.advanced_npcs, spread_locations=spread)
    results, summary = runner.run_all(args.variants, max_workers=args.threads)

    print("=== Experiment Summary ===")
    print(json.dumps(summary, indent=2))

    print(f"\nTotal episodes: {len(results)}")
    successes = sum(1 for r in results if r.success)
    print(f"Total successes: {successes}/{len(results)} ({100*successes/max(len(results),1):.1f}%)")

    print("\n=== Example Episode Trace (first episode) ===")
    if results:
        example = results[0]
        print(f"Variant: {example.agent_variant} | liar_ratio={example.liar_ratio} | success={example.success} | steps={example.steps}")
        for line in example.trace[:30]:
            print(f"  {line}")

    suffix = f"_{args.mode}"
    if args.advanced_npcs:
        suffix += "_advanced"
    if spread:
        suffix += "_spread"
    output_path = f"results{suffix}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "config": {
                    "mode": args.mode,
                    "max_steps": config.experiment.max_steps,
                    "spread_npcs": spread,
                    "advanced_npcs": args.advanced_npcs,
                    "runs_per_setting": config.experiment.runs_per_setting,
                    "total_npcs": config.experiment.total_npcs,
                    "variants": args.variants,
                    "use_hints": args.use_hints,
                    "liar_ratios": config.experiment.liar_ratios,
                },
                "episodes": [
                    {
                        "agent_variant": r.agent_variant,
                        "liar_ratio": r.liar_ratio,
                        "success": r.success,
                        "steps": r.steps,
                        "inference_accuracy": r.inference_accuracy,
                        "recovery_rate": r.recovery_rate,
                        "final_trust_scores": r.final_trust_scores,
                        "hidden_roles": r.hidden_roles,
                        "trace": r.trace,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
