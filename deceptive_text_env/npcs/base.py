from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

from deceptive_text_env.config import ModelConfig, WorldConfig
from deceptive_text_env.llm import BaseLLMClient
from deceptive_text_env.prompts import npc_system_prompt
from deceptive_text_env.types import Claim, NPCMessage, VerifiedClaim
from deceptive_text_env.world.judge import JudgeModel
from deceptive_text_env.world.verifier import GroundedVerifier


@dataclass
class BaseNPC:
    name: str
    location: str
    llm_client: BaseLLMClient
    model_config: ModelConfig
    policy: str
    pivot_turn: int = 3
    turn_counter: int = 0
    knowledge_topics: list[str] = field(default_factory=lambda: [
        "location_of:sun_sigil",
        "location_of:moon_sigil",
        "location_of:river_sigil",
        "vault_order",
    ])

    def reset(self) -> None:
        self.turn_counter = 0

    def respond(
        self,
        *,
        topic: str,
        turn_index: int,
        verifier: GroundedVerifier,
        judge: JudgeModel,
        agent_trust: float,
    ) -> tuple[NPCMessage, list[VerifiedClaim]]:
        self.turn_counter += 1
        if topic not in self.knowledge_topics:
            topic = self.knowledge_topics[0]
        fact_id = topic
        true_value = verifier.get_true_value(fact_id)
        payload = {
            "npc_name": self.name,
            "policy": self.policy,
            "topic": topic,
            "fact_id": fact_id,
            "true_value": true_value,
            "candidates": self._candidate_values(topic, verifier),
            "turn_index": self.turn_counter,
            "global_turn_index": turn_index,
            "agent_trust": agent_trust,
            "pivot_turn": self.pivot_turn,
        }
        result = self.llm_client.generate_json(
            task="npc_response",
            system_prompt=npc_system_prompt(),
            user_prompt=self._user_prompt(topic),
            payload=payload,
            model_config=self.model_config,
        )
        message = self._build_message(topic=topic, turn_index=turn_index, result=result)
        verified_claims = verifier.verify_claims(message.claims)
        try:
            judgement = judge.audit_message(
                policy=self.policy,
                agent_trust=agent_trust,
                turn_index=self.turn_counter,
                pivot_turn=self.pivot_turn,
                verified_claims=verified_claims,
            )
        except RuntimeError:
            judgement = JudgeModel.fallback_audit(
                policy=self.policy,
                agent_trust=agent_trust,
                turn_index=self.turn_counter,
                pivot_turn=self.pivot_turn,
                verified_claims=verified_claims,
            )
        if not judgement.compliant:
            message = self._deterministic_fallback(topic=topic, turn_index=turn_index, verifier=verifier, agent_trust=agent_trust)
            verified_claims = verifier.verify_claims(message.claims)
        return message, verified_claims

    def _build_message(self, *, topic: str, turn_index: int, result: dict[str, Any]) -> NPCMessage:
        claims = [
            Claim(
                fact_id=str(item["fact_id"]),
                claimed_value=item["claimed_value"],
                natural_language=str(item["natural_language"]),
            )
            for item in result.get("claims", [])
        ]
        return NPCMessage(
            npc_name=self.name,
            policy=self.policy,
            topic=topic,
            text=str(result.get("text", "")),
            claims=claims,
            turn_index=turn_index,
        )

    def _candidate_values(self, topic: str, verifier: GroundedVerifier) -> list[Any]:
        if topic.startswith("location_of:"):
            # Return all non-start, non-vault locations where sigils could plausibly be
            if hasattr(self, "_world_config") and self._world_config is not None:
                wc: WorldConfig = self._world_config
                return [loc for loc in wc.location_graph if loc != wc.start_location]
            return ["forest_shrine", "cave_pool", "river_dock", "ancient_vault"]
        if topic == "vault_order":
            true_order = verifier.get_true_value(topic)
            if hasattr(self, "_world_config") and self._world_config is not None:
                wc = self._world_config
                sigils = list(wc.sigil_locations.keys())
                all_perms = [list(p) for p in permutations(sigils)]
                # Put true order first, then up to 3 false alternatives
                false_orders = [p for p in all_perms if p != list(true_order)]
                return [true_order] + false_orders[:3]
            return [
                true_order,
                ["moon_sigil", "sun_sigil", "river_sigil"],
                ["river_sigil", "moon_sigil", "sun_sigil"],
                ["sun_sigil", "river_sigil", "moon_sigil"],
            ]
        return [verifier.get_true_value(topic)]

    def _user_prompt(self, topic: str) -> str:
        return (
            f"Answer a single topic for the fictional game world: {topic}. "
            "Follow your assigned policy exactly and return JSON with 'text' and 'claims'."
        )

    def _deterministic_fallback(
        self,
        *,
        topic: str,
        turn_index: int,
        verifier: GroundedVerifier,
        agent_trust: float,
    ) -> NPCMessage:
        true_value = verifier.get_true_value(topic)
        candidates = self._candidate_values(topic, verifier)
        false_candidates = [candidate for candidate in candidates if candidate != true_value]
        chosen_value: Any = true_value
        if self.policy == "deceptive" and agent_trust >= 0.65 and false_candidates:
            chosen_value = false_candidates[0]
        elif self.policy == "opportunistic" and self.turn_counter >= self.pivot_turn and false_candidates:
            chosen_value = false_candidates[0]
        elif self.policy == "partial_truth" and topic == "vault_order" and false_candidates:
            chosen_value = false_candidates[0]
        elif self.policy == "coordinated_deceptive" and agent_trust >= 0.50 and false_candidates:
            chosen_value = false_candidates[0]
        text = self._format_claim_text(topic, chosen_value)
        return NPCMessage(
            npc_name=self.name,
            policy=self.policy,
            topic=topic,
            text=text,
            claims=[Claim(fact_id=topic, claimed_value=chosen_value, natural_language=text)],
            turn_index=turn_index,
        )

    def _format_claim_text(self, topic: str, value: Any) -> str:
        if topic.startswith("location_of:"):
            return f"The {topic.split(':', maxsplit=1)[1].replace('_', ' ')} is at {str(value).replace('_', ' ')}."
        if topic == "vault_order":
            pretty = ", ".join(item.replace("_", " ") for item in value)
            return f"The vault order is {pretty}."
        return str(value)


class TruthfulNPC(BaseNPC):
    def __init__(self, *, name: str, location: str, llm_client: BaseLLMClient, model_config: ModelConfig, world_config: WorldConfig | None = None) -> None:
        super().__init__(name=name, location=location, llm_client=llm_client, model_config=model_config, policy="truthful",
                         knowledge_topics=_topics_from_world_config(world_config))
        self._world_config = world_config


class DeceptiveNPC(BaseNPC):
    def __init__(self, *, name: str, location: str, llm_client: BaseLLMClient, model_config: ModelConfig, world_config: WorldConfig | None = None) -> None:
        super().__init__(name=name, location=location, llm_client=llm_client, model_config=model_config, policy="deceptive",
                         knowledge_topics=_topics_from_world_config(world_config))
        self._world_config = world_config


class OpportunisticNPC(BaseNPC):
    def __init__(self, *, name: str, location: str, llm_client: BaseLLMClient, model_config: ModelConfig, pivot_turn: int = 3, world_config: WorldConfig | None = None) -> None:
        super().__init__(
            name=name,
            location=location,
            llm_client=llm_client,
            model_config=model_config,
            policy="opportunistic",
            pivot_turn=pivot_turn,
            knowledge_topics=_topics_from_world_config(world_config),
        )
        self._world_config = world_config


class PartialTruthNPC(BaseNPC):
    """Gives correct sigil locations but always lies about vault_order."""

    def __init__(self, *, name: str, location: str, llm_client: BaseLLMClient, model_config: ModelConfig, world_config: WorldConfig | None = None) -> None:
        super().__init__(name=name, location=location, llm_client=llm_client, model_config=model_config, policy="partial_truth",
                         knowledge_topics=_topics_from_world_config(world_config))
        self._world_config = world_config


class CoordinatedDeceptiveNPC(BaseNPC):
    """Lies when trust >= 0.50 (lower threshold). Multiple instances give the SAME wrong answer."""

    def __init__(self, *, name: str, location: str, llm_client: BaseLLMClient, model_config: ModelConfig, world_config: WorldConfig | None = None) -> None:
        super().__init__(name=name, location=location, llm_client=llm_client, model_config=model_config, policy="coordinated_deceptive",
                         knowledge_topics=_topics_from_world_config(world_config))
        self._world_config = world_config


def _topics_from_world_config(world_config: WorldConfig | None) -> list[str]:
    """Derive knowledge topics from a world config, falling back to defaults."""
    if world_config is None:
        return [
            "location_of:sun_sigil",
            "location_of:moon_sigil",
            "location_of:river_sigil",
            "vault_order",
        ]
    return [f"location_of:{s}" for s in world_config.sigil_locations] + ["vault_order"]


NPC_NAMES = [
    "Aster", "Bram", "Cyra", "Dorian", "Elara",
    "Fenn", "Galen", "Helia", "Isolde", "Jarek",
]

# When spread_locations=True, NPCs get placed across the world.
# First 3 at village_square, rest distributed to outer locations.
SPREAD_LOCATIONS = [
    "village_square",
    "village_square",
    "village_square",
    "forest_shrine",
    "cave_pool",
    "river_dock",
    "forest_shrine",
    "cave_pool",
    "river_dock",
    "village_square",
]


def build_npc_roster(
    *,
    total_npcs: int,
    liar_ratio: float,
    llm_client: BaseLLMClient,
    model_config: ModelConfig,
    location: str = "village_square",
    use_advanced_strategies: bool = False,
    spread_locations: bool = False,
    world_config: WorldConfig | None = None,
) -> list[BaseNPC]:
    liar_count = max(0, int(round(total_npcs * liar_ratio)))
    liar_count = min(liar_count, max(total_npcs - 1, 0))
    truthful_count = total_npcs - liar_count

    if use_advanced_strategies and liar_count >= 3:
        deceptive_count = max(1, liar_count // 3)
        partial_truth_count = max(1, liar_count // 3)
        coordinated_count = liar_count - deceptive_count - partial_truth_count
        opportunistic_count = 0
    elif use_advanced_strategies and liar_count == 2:
        deceptive_count = 1
        partial_truth_count = 1
        coordinated_count = 0
        opportunistic_count = 0
    else:
        deceptive_count = liar_count // 2
        opportunistic_count = liar_count - deceptive_count
        partial_truth_count = 0
        coordinated_count = 0

    def loc(index: int) -> str:
        if spread_locations:
            return SPREAD_LOCATIONS[index % len(SPREAD_LOCATIONS)]
        return location

    roster: list[BaseNPC] = []
    index = 0

    wc = world_config

    for _ in range(truthful_count):
        roster.append(TruthfulNPC(name=NPC_NAMES[index], location=loc(index), llm_client=llm_client, model_config=model_config, world_config=wc))
        index += 1
    for _ in range(deceptive_count):
        roster.append(DeceptiveNPC(name=NPC_NAMES[index], location=loc(index), llm_client=llm_client, model_config=model_config, world_config=wc))
        index += 1
    for _ in range(opportunistic_count):
        roster.append(
            OpportunisticNPC(name=NPC_NAMES[index], location=loc(index), llm_client=llm_client, model_config=model_config, pivot_turn=3, world_config=wc)
        )
        index += 1
    for _ in range(partial_truth_count):
        roster.append(PartialTruthNPC(name=NPC_NAMES[index], location=loc(index), llm_client=llm_client, model_config=model_config, world_config=wc))
        index += 1
    for _ in range(coordinated_count):
        roster.append(CoordinatedDeceptiveNPC(name=NPC_NAMES[index], location=loc(index), llm_client=llm_client, model_config=model_config, world_config=wc))
        index += 1

    return roster[:total_npcs]
