"""Unit tests for StructuredMemoryArchitecture."""
from __future__ import annotations

import pytest

from deceptive_text_env.memory import StructuredMemoryArchitecture
from deceptive_text_env.types import ContradictionRecord, EnvironmentFactRecord, NPCStatementRecord


def test_memory_starts_empty() -> None:
    memory = StructuredMemoryArchitecture()
    assert memory.npc_statements == []
    assert memory.detected_contradictions == []
    assert memory.environment_facts == []
    assert memory.reflection_notes == []
    assert "None" in memory.summarize()
    assert memory.claims_by_fact() == {}


def test_add_npc_statement() -> None:
    memory = StructuredMemoryArchitecture()
    record = NPCStatementRecord(
        turn_index=1,
        npc_name="Aster",
        topic="location_of:sun_sigil",
        fact_id="location_of:sun_sigil",
        claimed_value="forest_shrine",
        statement_text="The sun sigil is at forest shrine.",
        trust_at_record=0.5,
    )
    memory.add_npc_statement(record)
    assert len(memory.npc_statements) == 1
    assert memory.npc_statements[0].npc_name == "Aster"
    grouped = memory.claims_by_fact()
    assert "location_of:sun_sigil" in grouped
    assert len(grouped["location_of:sun_sigil"]) == 1
    assert grouped["location_of:sun_sigil"][0]["claimed_value"] == "forest_shrine"


def test_add_contradiction() -> None:
    memory = StructuredMemoryArchitecture()
    memory.add_contradiction(
        ContradictionRecord(
            turn_index=2,
            fact_id="vault_order",
            details="Vault rejected order.",
            npc_names=["Bram", "Cyra"],
            disproven_value=["a", "b", "c"],
        )
    )
    assert len(memory.detected_contradictions) == 1
    assert memory.detected_contradictions[0].npc_names == ["Bram", "Cyra"]


def test_add_environment_fact_no_duplicate() -> None:
    memory = StructuredMemoryArchitecture()
    memory.add_environment_fact(
        EnvironmentFactRecord(turn_index=1, fact_id="location_of:sun_sigil", value="forest_shrine", source="search_success")
    )
    memory.add_environment_fact(
        EnvironmentFactRecord(turn_index=2, fact_id="location_of:sun_sigil", value="forest_shrine", source="search_success")
    )
    assert len(memory.environment_facts) == 1


def test_add_environment_fact_different_value_allowed() -> None:
    memory = StructuredMemoryArchitecture()
    memory.add_environment_fact(
        EnvironmentFactRecord(turn_index=1, fact_id="location_of:sun_sigil", value="forest_shrine", source="search_success")
    )
    memory.add_environment_fact(
        EnvironmentFactRecord(turn_index=2, fact_id="location_of:sun_sigil", value="cave_pool", source="search_success")
    )
    assert len(memory.environment_facts) == 2


def test_latest_environment_fact() -> None:
    memory = StructuredMemoryArchitecture()
    memory.add_environment_fact(
        EnvironmentFactRecord(turn_index=1, fact_id="f1", value="v1", source="s1")
    )
    memory.add_environment_fact(
        EnvironmentFactRecord(turn_index=2, fact_id="f1", value="v2", source="s2")
    )
    latest = memory.latest_environment_fact("f1")
    assert latest is not None
    assert latest.value == "v2"
    assert memory.latest_environment_fact("missing") is None


def test_add_reflection_ignores_empty() -> None:
    memory = StructuredMemoryArchitecture()
    memory.add_reflection("")
    assert len(memory.reflection_notes) == 0
    memory.add_reflection("Some reflection.")
    assert len(memory.reflection_notes) == 1


def test_summarize_respects_max_entries() -> None:
    memory = StructuredMemoryArchitecture()
    for i in range(12):
        memory.add_npc_statement(
            NPCStatementRecord(
                turn_index=i,
                npc_name="A",
                topic="t",
                fact_id="f",
                claimed_value="v",
                statement_text="",
                trust_at_record=0.5,
            )
        )
    summary = memory.summarize(max_entries=3)
    assert "Turn 9" in summary
    assert "Turn 10" in summary
    assert "Turn 11" in summary


def test_reset_clears_all() -> None:
    memory = StructuredMemoryArchitecture()
    memory.add_npc_statement(
        NPCStatementRecord(0, "A", "t", "f", "v", "", 0.5)
    )
    memory.add_contradiction(ContradictionRecord(0, "f", "d", [], None))
    memory.add_environment_fact(EnvironmentFactRecord(0, "f", "v", "s"))
    memory.add_reflection("note")
    memory.reset()
    assert not memory.npc_statements
    assert not memory.detected_contradictions
    assert not memory.environment_facts
    assert not memory.reflection_notes
