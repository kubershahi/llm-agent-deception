"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from deceptive_text_env.config import ExperimentConfig


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        max_steps=24,
        runs_per_setting=2,
        total_npcs=6,
        liar_ratios=[0.1, 0.3, 0.5],
        reflection_interval=4,
        trust_success_gain=0.18,
        trust_failure_decay=0.42,
        contradiction_decay=0.12,
        distrust_threshold=0.30,
        role_target_trust={"truthful": 1.0, "deceptive": 0.0, "opportunistic": 0.35},
    )
