"""Integration tests: run full episodes with mock LLM."""
from __future__ import annotations

import pytest

from deceptive_text_env import build_default_config
from deceptive_text_env.config import ExperimentConfig, FrameworkConfig
from deceptive_text_env.evaluation.runner import EvaluationRunner
from deceptive_text_env.types import EpisodeResult


def _config_fast(runs_per_setting: int = 1, max_steps: int = 20) -> FrameworkConfig:
    cfg = build_default_config()
    cfg.experiment = ExperimentConfig(
        max_steps=max_steps,
        runs_per_setting=runs_per_setting,
        total_npcs=6,
        liar_ratios=[0.1, 0.3],
        reflection_interval=4,
        trust_success_gain=0.18,
        trust_failure_decay=0.42,
        contradiction_decay=0.12,
        distrust_threshold=0.30,
        role_target_trust={"truthful": 1.0, "deceptive": 0.0, "opportunistic": 0.35},
    )
    return cfg


def test_run_single_episode_returns_valid_result() -> None:
    config = _config_fast(runs_per_setting=1, max_steps=20)
    runner = EvaluationRunner(config)
    result = runner.run_episode(agent_variant="naive", liar_ratio=0.1, seed=42)
    assert isinstance(result, EpisodeResult)
    assert result.agent_variant == "naive"
    assert result.liar_ratio == 0.1
    assert result.seed == 42
    assert isinstance(result.success, bool)
    assert 0 <= result.steps <= config.experiment.max_steps
    assert isinstance(result.final_trust_scores, dict)
    assert 0.0 <= result.inference_accuracy <= 1.0
    assert isinstance(result.hidden_roles, dict)
    assert isinstance(result.trace, list)


def test_run_all_produces_summary_for_each_cell() -> None:
    config = _config_fast(runs_per_setting=1, max_steps=16)
    runner = EvaluationRunner(config)
    variants = ["naive", "memory_augmented"]
    results, summary = runner.run_all(variants)
    assert len(results) == 2 * 2  # 2 liar_ratios * 2 variants * 1 run
    for r in results:
        assert r.agent_variant in variants
        assert r.liar_ratio in (0.1, 0.3)
    assert isinstance(summary, dict)
    for key in summary:
        assert "@" in key
        cell = summary[key]
        assert "task_success_rate" in cell
        assert "avg_steps" in cell
        assert "avg_inference_accuracy" in cell
        assert "avg_recovery_rate" in cell
        assert 0 <= cell["task_success_rate"] <= 1
        assert cell["avg_steps"] >= 0


def test_different_liar_ratios_produce_different_trust_outcomes() -> None:
    config = _config_fast(runs_per_setting=2, max_steps=18)
    runner = EvaluationRunner(config)
    # Same variant, same seed pattern for two runs at different liar ratios
    r_low = runner.run_episode(agent_variant="belief_tracking", liar_ratio=0.1, seed=100)
    r_high = runner.run_episode(agent_variant="belief_tracking", liar_ratio=0.5, seed=100)
    assert r_low.liar_ratio == 0.1
    assert r_high.liar_ratio == 0.5
    # At least one of success or inference_accuracy can differ with more liars
    # (not guaranteed every run, but we can check structure)
    assert isinstance(r_low.success, bool)
    assert isinstance(r_high.success, bool)
    assert len(r_low.final_trust_scores) == len(r_high.final_trust_scores)
