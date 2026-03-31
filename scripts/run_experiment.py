"""
Run all four agent variants across all liar ratios and print results.

Usage:
  python scripts/run_experiment.py              # mock (fast, no API)
  python scripts/run_experiment.py --mode full  # real TritonAI LLM
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deceptive_text_env import build_default_config, build_hybrid_config, build_tritonai_config
from deceptive_text_env.evaluation import EvaluationRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "hybrid", "full"], default="mock")
    args = parser.parse_args()

    if args.mode == "full":
        config = build_tritonai_config()
    elif args.mode == "hybrid":
        config = build_hybrid_config()
    else:
        config = build_default_config()

    runner = EvaluationRunner(config)
    variants = ["naive", "memory_augmented", "belief_tracking", "reflection_enhanced"]
    results, summary = runner.run_all(variants)

    print("=== Experiment Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nTotal episodes: {len(results)}")

    if results:
        print("\n=== Example Episode Trace ===")
        example = results[0]
        print(f"Variant: {example.agent_variant} | liar_ratio={example.liar_ratio} | success={example.success}")
        for line in example.trace[:20]:
            print(f"- {line}")

    results_dir = Path("artifacts/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"results_{args.mode}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
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
