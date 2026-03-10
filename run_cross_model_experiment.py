"""
Cross-model ablation: run the same hard-mode experiment with different LLMs.

Usage:
  export TRITONAI_API_KEY="your-key"
  python run_cross_model_experiment.py --runs 2 --threads 4
  python run_cross_model_experiment.py --models api-llama-4-scout mistral.mistral-large-3-675b-instruct
"""
from __future__ import annotations

import argparse
import json
import sys

from deceptive_text_env.config import (
    ExperimentConfig,
    FrameworkConfig,
    ModelConfig,
)
from deceptive_text_env.evaluation import EvaluationRunner
from deceptive_text_env.llm import enable_call_logging

AVAILABLE_MODELS = [
    "api-gpt-oss-120b",
    "api-llama-4-scout",
    "mistral.mistral-large-3-675b-instruct",
]


def build_config_for_model(model_name: str) -> FrameworkConfig:
    return FrameworkConfig(
        premium_agent_model=ModelConfig(
            provider="tritonai",
            model_name=model_name,
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.1,
            max_tokens=1024,
            seed=42,
        ),
        budget_npc_model=ModelConfig(
            provider="mock",
            model_name="mock-npc",
            temperature=0.3,
        ),
        judge_model=ModelConfig(
            provider="mock",
            model_name="mock-judge",
            temperature=0.0,
        ),
        experiment=ExperimentConfig(
            max_steps=18,
            runs_per_setting=2,
            total_npcs=6,
            liar_ratios=[0.0, 0.1, 0.3, 0.5],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-model ablation experiment")
    parser.add_argument("--models", nargs="+", default=AVAILABLE_MODELS)
    parser.add_argument("--variants", nargs="+",
                        default=["naive", "belief_tracking", "reflection_enhanced", "memory_with_trust"])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--use-hints", action="store_true",
                        help="Enable PRIORITY_ACTION payload hints")
    parser.add_argument("--liar-ratios", nargs="+", type=float, default=None,
                        help="Override liar ratios")
    args = parser.parse_args()

    enable_call_logging("llm_logs")

    all_results = {}
    all_episodes = []

    for model_name in args.models:
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model_name}", flush=True)
        print(f"{'='*60}", flush=True)

        config = build_config_for_model(model_name)
        config.experiment.runs_per_setting = args.runs
        config.experiment.use_hints = args.use_hints
        if args.liar_ratios is not None:
            config.experiment.liar_ratios = args.liar_ratios

        runner = EvaluationRunner(config, spread_locations=True)
        try:
            results, summary = runner.run_all(args.variants, max_workers=args.threads)
        except Exception as e:
            print(f"  ERROR with {model_name}: {e}", flush=True)
            continue

        for key, metrics in summary.items():
            new_key = f"{key}@model={model_name}"
            all_results[new_key] = metrics

        for r in results:
            all_episodes.append({
                "model": model_name,
                "agent_variant": r.agent_variant,
                "liar_ratio": r.liar_ratio,
                "success": r.success,
                "steps": r.steps,
                "inference_accuracy": r.inference_accuracy,
                "recovery_rate": r.recovery_rate,
                "final_trust_scores": r.final_trust_scores,
                "hidden_roles": r.hidden_roles,
                "trace": r.trace,
            })

        successes = sum(1 for r in results if r.success)
        print(f"  {model_name}: {successes}/{len(results)} success ({100*successes/max(len(results),1):.0f}%)", flush=True)

    output_path = "results_cross_model.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "models": args.models,
                "variants": args.variants,
                "runs_per_setting": args.runs,
                "max_steps": 18,
                "spread_npcs": True,
            },
            "summary": all_results,
            "episodes": all_episodes,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("Cross-Model Success Rate Comparison (Hard Mode)")
    print(f"{'='*70}")
    header = f"{'Variant':<25} " + " ".join(f"{m[:15]:<17}" for m in args.models)
    print(header)
    print("-" * len(header))
    for variant in args.variants:
        row = f"{variant:<25} "
        for model in args.models:
            rates = [1.0 if ep["success"] else 0.0
                     for ep in all_episodes
                     if ep["agent_variant"] == variant and ep["model"] == model]
            if rates:
                row += f"{sum(rates)/len(rates):<17.2f}"
            else:
                row += f"{'N/A':<17}"
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
