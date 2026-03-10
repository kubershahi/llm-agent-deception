"""Run experiments on the extended (harder) world."""
from __future__ import annotations

import argparse
import json
import sys

from deceptive_text_env.config import (
    ExperimentConfig,
    FrameworkConfig,
    ModelConfig,
    build_extended_world_config,
)
from deceptive_text_env.evaluation import EvaluationRunner
from deceptive_text_env.llm import enable_call_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Extended world experiment")
    parser.add_argument("--variants", nargs="+",
                        default=["oracle", "random", "naive", "belief_tracking",
                                 "reflection_enhanced", "memory_with_trust"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--use-hints", action="store_true")
    parser.add_argument("--liar-ratios", nargs="+", type=float,
                        default=[0.0, 0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--mock-only", action="store_true",
                        help="Use mock LLM for agent too (baselines only)")
    args = parser.parse_args()

    ext_world = build_extended_world_config()

    if args.mock_only:
        agent_model = ModelConfig(provider="mock", model_name="mock-agent", temperature=0.1)
    else:
        agent_model = ModelConfig(
            provider="tritonai", model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.1, max_tokens=1024, seed=42,
        )
        enable_call_logging("llm_logs")

    config = FrameworkConfig(
        premium_agent_model=agent_model,
        budget_npc_model=ModelConfig(provider="mock", model_name="mock-npc", temperature=0.3),
        judge_model=ModelConfig(provider="mock", model_name="mock-judge", temperature=0.0),
        world=ext_world,
        experiment=ExperimentConfig(
            max_steps=25,
            runs_per_setting=args.runs,
            total_npcs=6,
            liar_ratios=args.liar_ratios,
            use_hints=args.use_hints,
        ),
    )

    print(f"World: extended (7 locations, 4 sigils, branched topology)")
    print(f"Max steps: 25 (optimal=19)")
    print(f"Variants: {args.variants}")
    print(f"Liar ratios: {args.liar_ratios}")
    print(f"Runs: {args.runs}")
    print(f"Use hints: {args.use_hints}")
    print(f"Agent model: {config.premium_agent_model.provider}/{config.premium_agent_model.model_name}")
    print()

    runner = EvaluationRunner(config, spread_locations=True)
    results, summary = runner.run_all(args.variants, max_workers=args.threads)

    print("\n=== Summary ===")
    for key, m in sorted(summary.items()):
        print(f"  {key}: success={m['task_success_rate']:.0%} steps={m['avg_steps']:.1f}")

    total = len(results)
    successes = sum(1 for r in results if r.success)
    print(f"\nTotal: {successes}/{total} ({100*successes/max(total,1):.0f}%)")

    suffix = "_extended"
    if args.use_hints:
        suffix += "_hints"
    if args.mock_only:
        suffix += "_mock"
    output_path = f"results{suffix}.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "world": "extended",
                "max_steps": 25,
                "optimal_steps": 19,
                "variants": args.variants,
                "liar_ratios": args.liar_ratios,
                "runs_per_setting": args.runs,
                "use_hints": args.use_hints,
            },
            "summary": summary,
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
        }, f, indent=2)
    print(f"Saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
