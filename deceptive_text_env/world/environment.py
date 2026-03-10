from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deceptive_text_env.config import WorldConfig
from deceptive_text_env.types import AgentAction, Observation, StepResult
from deceptive_text_env.world.judge import JudgeModel
from deceptive_text_env.world.verifier import GroundedVerifier


@dataclass
class EnvironmentState:
    current_location: str
    collected_sigils: list[str] = field(default_factory=list)
    discovered_locations: set[str] = field(default_factory=set)
    step_count: int = 0
    completed: bool = False
    last_event: str = "The game has just begun."
    last_unlock_order: list[str] = field(default_factory=list)


class TextWorldEnvironment:
    def __init__(
        self,
        *,
        world_config: WorldConfig,
        verifier: GroundedVerifier,
        judge: JudgeModel,
        npcs: list[Any],
        max_steps: int,
    ) -> None:
        self.world_config = world_config
        self.verifier = verifier
        self.judge = judge
        self.npcs = npcs
        self.max_steps = max_steps
        self.state = EnvironmentState(current_location=world_config.start_location)

    def reset(self) -> Observation:
        self.state = EnvironmentState(
            current_location=self.world_config.start_location,
            discovered_locations={self.world_config.start_location},
        )
        for npc in self.npcs:
            npc.reset()
        return self._build_observation()

    def step(self, action: AgentAction, agent_trust: dict[str, float]) -> StepResult:
        self.state.step_count += 1
        action_type = action.action_type.lower().strip()

        if action_type == "move":
            return self._handle_move(action)
        if action_type == "talk":
            return self._handle_talk(action, agent_trust)
        if action_type == "search":
            return self._handle_search(action)
        if action_type == "unlock":
            return self._handle_unlock(action)
        return self._invalid_action(f"Unknown action type: {action.action_type}")

    def hidden_role_map(self) -> dict[str, str]:
        return {npc.name: npc.policy for npc in self.npcs}

    def _handle_move(self, action: AgentAction) -> StepResult:
        destination = action.target
        current_neighbors = self.world_config.location_graph[self.state.current_location]
        if destination not in current_neighbors:
            return self._invalid_action(f"You cannot move from {self.state.current_location} to {destination}.")
        self.state.current_location = destination
        self.state.discovered_locations.add(destination)
        self.state.last_event = f"You moved to {destination}."
        return StepResult(
            observation=self._build_observation(),
            success=True,
            done=False,
            reward=0.0,
            public_feedback=self.state.last_event,
        )

    def _handle_talk(self, action: AgentAction, agent_trust: dict[str, float]) -> StepResult:
        npc = self._npc_by_name(action.target)
        if npc is None:
            return self._invalid_action(f"No NPC named {action.target} is available.")
        if npc.location != self.state.current_location:
            return self._invalid_action(f"{npc.name} is not at {self.state.current_location}.")
        message, verified_claims = npc.respond(
            topic=action.topic,
            turn_index=self.state.step_count,
            verifier=self.verifier,
            judge=self.judge,
            agent_trust=agent_trust.get(npc.name, 0.5),
        )
        self.state.last_event = f"{npc.name} said: {message.text}"
        return StepResult(
            observation=self._build_observation(),
            success=True,
            done=False,
            reward=0.05,
            public_feedback=self.state.last_event,
            npc_message=message,
            hidden_verified_claims=verified_claims,
        )

    def _handle_search(self, action: AgentAction) -> StepResult:
        location = self.state.current_location
        found_item = None
        for sigil, sigil_location in self.world_config.sigil_locations.items():
            if sigil_location == location and sigil not in self.state.collected_sigils:
                found_item = sigil
                break

        if found_item is None:
            self.state.last_event = f"You searched {location} and found no sigil. {self.world_config.decoy_items.get(location, '')}".strip()
            return StepResult(
                observation=self._build_observation(),
                success=False,
                done=False,
                reward=-0.1,
                public_feedback=self.state.last_event,
            )

        self.state.collected_sigils.append(found_item)
        self.state.last_event = f"You searched {location} and recovered the {found_item}."
        return StepResult(
            observation=self._build_observation(),
            success=True,
            done=False,
            reward=0.8,
            public_feedback=self.state.last_event,
            discovered_item=found_item,
        )

    def _handle_unlock(self, action: AgentAction) -> StepResult:
        if self.state.current_location != "ancient_vault":
            return self._invalid_action("You must be at the ancient_vault to unlock it.")
        if len(self.state.collected_sigils) < len(self.world_config.sigil_locations):
            return self._invalid_action(f"You need all {len(self.world_config.sigil_locations)} sigils before trying to unlock the vault.")
        supplied_order = [segment.strip() for segment in action.content.split(",") if segment.strip()]
        self.state.last_unlock_order = supplied_order
        if supplied_order == self.world_config.vault_order:
            self.state.completed = True
            self.state.last_event = "The vault opens. You completed the long-horizon objective."
            return StepResult(
                observation=self._build_observation(),
                success=True,
                done=True,
                reward=2.0,
                public_feedback=self.state.last_event,
            )
        self.state.last_event = "The vault rejects the sequence and remains sealed."
        return StepResult(
            observation=self._build_observation(),
            success=False,
            done=False,
            reward=-0.4,
            public_feedback=self.state.last_event,
        )

    def _build_observation(self) -> Observation:
        visible_npcs = [npc.name for npc in self.npcs if npc.location == self.state.current_location]
        pending_goal = self._pending_goal_text()
        return Observation(
            turn_index=self.state.step_count,
            location=self.state.current_location,
            description=self.world_config.location_descriptions[self.state.current_location],
            visible_npcs=visible_npcs,
            accessible_locations=list(self.world_config.location_graph[self.state.current_location]),
            inventory=list(self.state.collected_sigils),
            collected_sigils=list(self.state.collected_sigils),
            pending_goal_text=pending_goal,
            available_topics=[f"location_of:{s}" for s in self.world_config.sigil_locations] + ["vault_order"],
            last_event=self.state.last_event,
        )

    def _pending_goal_text(self) -> str:
        missing = [sigil for sigil in self.world_config.sigil_locations if sigil not in self.state.collected_sigils]
        if missing:
            return f"Still needed: {', '.join(missing)}. Then unlock the ancient_vault."
        return "All sigils collected. Travel to the ancient_vault and submit the correct order."

    def _npc_by_name(self, name: str) -> Any | None:
        for npc in self.npcs:
            if npc.name == name:
                return npc
        return None

    def _invalid_action(self, message: str) -> StepResult:
        self.state.last_event = message
        done = self.state.step_count >= self.max_steps
        return StepResult(
            observation=self._build_observation(),
            success=False,
            done=done,
            reward=-0.2,
            public_feedback=message,
        )


def build_world_facts(world_config: WorldConfig) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for sigil, location in world_config.sigil_locations.items():
        facts[f"location_of:{sigil}"] = location
    facts["vault_order"] = list(world_config.vault_order)
    return facts
