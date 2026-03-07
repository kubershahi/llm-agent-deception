"""Unit tests for evaluation metrics."""
from __future__ import annotations

import pytest

from deceptive_text_env.config import ExperimentConfig
from deceptive_text_env.evaluation.metrics import aggregate_results, inference_accuracy
from deceptive_text_env.types import EpisodeResult


def test_inference_accuracy_perfect(experiment_config: ExperimentConfig) -> None:
    trust_scores = {"A": 1.0, "B": 0.0, "C": 0.35}
    hidden_roles = {"A": "truthful", "B": "deceptive", "C": "opportunistic"}
    score = inference_accuracy(trust_scores, hidden_roles, experiment_config)
    assert score == 1.0


def test_inference_accuracy_all_wrong(experiment_config: ExperimentConfig) -> None:
    trust_scores = {"A": 0.0, "B": 1.0, "C": 1.0}
    hidden_roles = {"A": "truthful", "B": "deceptive", "C": "opportunistic"}
    score = inference_accuracy(trust_scores, hidden_roles, experiment_config)
    assert score < 1.0
    assert score >= 0.0


def test_inference_accuracy_empty_roles(experiment_config: ExperimentConfig) -> None:
    score = inference_accuracy({}, {}, experiment_config)
    assert score == 0.0


def test_inference_accuracy_unknown_role_uses_default_target(experiment_config: ExperimentConfig) -> None:
    # role_target_trust default for unknown is 0.5
    trust_scores = {"A": 0.5}
    hidden_roles = {"A": "unknown_role"}
    score = inference_accuracy(trust_scores, hidden_roles, experiment_config)
    assert score == 1.0


def test_aggregate_results_groups_by_variant_and_liar_ratio(experiment_config: ExperimentConfig) -> None:
    results = [
        _make_episode("naive", 0.1, 0, success=True, steps=10, inference_acc=0.8, recovery=None),
        _make_episode("naive", 0.1, 1, success=False, steps=24, inference_acc=0.6, recovery=None),
        _make_episode("belief_tracking", 0.3, 0, success=True, steps=12, inference_acc=0.9, recovery=5.0),
    ]
    summary = aggregate_results(results)
    assert "naive@0.1" in summary
    assert "belief_tracking@0.3" in summary
    assert summary["naive@0.1"]["task_success_rate"] == 0.5
    assert summary["naive@0.1"]["avg_steps"] == 17.0
    assert summary["naive@0.1"]["avg_inference_accuracy"] == 0.7
    assert summary["belief_tracking@0.3"]["avg_recovery_rate"] == 5.0


def test_aggregate_results_no_recovery_uses_negative_one(experiment_config: ExperimentConfig) -> None:
    results = [
        _make_episode("naive", 0.1, 0, success=True, steps=10, inference_acc=0.8, recovery=None),
    ]
    summary = aggregate_results(results)
    assert summary["naive@0.1"]["avg_recovery_rate"] == -1.0


def _make_episode(
    variant: str,
    liar_ratio: float,
    seed: int,
    *,
    success: bool,
    steps: int,
    inference_acc: float,
    recovery: float | None,
) -> EpisodeResult:
    return EpisodeResult(
        agent_variant=variant,
        liar_ratio=liar_ratio,
        seed=seed,
        success=success,
        steps=steps,
        final_trust_scores={"A": 0.9, "B": 0.2},
        inference_accuracy=inference_acc,
        recovery_rate=recovery,
        hidden_roles={"A": "truthful", "B": "deceptive"},
        trace=[],
    )
