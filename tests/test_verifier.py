"""Unit tests for GroundedVerifier."""
from __future__ import annotations

import pytest

from deceptive_text_env.types import Claim, VerifiedClaim
from deceptive_text_env.world.verifier import GroundedVerifier


def test_verifier_accepts_true_claims() -> None:
    facts = {"location_of:sun_sigil": "forest_shrine", "vault_order": ["a", "b", "c"]}
    verifier = GroundedVerifier(facts)
    claims = [
        Claim(fact_id="location_of:sun_sigil", claimed_value="forest_shrine", natural_language="Sun at forest."),
    ]
    verified = verifier.verify_claims(claims)
    assert len(verified) == 1
    assert verified[0].is_true is True
    assert verified[0].expected_value == "forest_shrine"
    assert verified[0].claimed_value == "forest_shrine"


def test_verifier_rejects_false_claims() -> None:
    facts = {"location_of:sun_sigil": "forest_shrine"}
    verifier = GroundedVerifier(facts)
    claims = [
        Claim(fact_id="location_of:sun_sigil", claimed_value="cave_pool", natural_language="Sun at cave."),
    ]
    verified = verifier.verify_claims(claims)
    assert len(verified) == 1
    assert verified[0].is_true is False
    assert verified[0].expected_value == "forest_shrine"
    assert verified[0].claimed_value == "cave_pool"


def test_verifier_handles_unknown_fact_id() -> None:
    facts = {"location_of:sun_sigil": "forest_shrine"}
    verifier = GroundedVerifier(facts)
    claims = [
        Claim(fact_id="unknown_fact", claimed_value="x", natural_language="Unknown."),
    ]
    verified = verifier.verify_claims(claims)
    assert len(verified) == 1
    assert verified[0].expected_value is None
    assert verified[0].claimed_value == "x"
    assert verified[0].is_true is False


def test_get_true_value() -> None:
    facts = {"location_of:sun_sigil": "forest_shrine", "vault_order": ["a", "b"]}
    verifier = GroundedVerifier(facts)
    assert verifier.get_true_value("location_of:sun_sigil") == "forest_shrine"
    assert verifier.get_true_value("vault_order") == ["a", "b"]


def test_get_true_value_missing_raises() -> None:
    verifier = GroundedVerifier({"k": "v"})
    with pytest.raises(KeyError):
        verifier.get_true_value("missing")


def test_fact_exists() -> None:
    facts = {"location_of:sun_sigil": "forest_shrine"}
    verifier = GroundedVerifier(facts)
    assert verifier.fact_exists("location_of:sun_sigil") is True
    assert verifier.fact_exists("missing") is False


def test_verifier_does_not_mutate_input_facts() -> None:
    facts = {"k": "v"}
    verifier = GroundedVerifier(facts)
    verifier.immutable_world_facts["k"] = "other"
    assert facts["k"] == "v"
