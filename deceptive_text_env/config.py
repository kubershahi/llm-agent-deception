from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    provider: str
    model_name: str
    api_key_env_var: str = "TRITONAI_API_KEY"
    base_url: str = "https://tritonai-api.ucsd.edu/v1"
    temperature: float = 0.2
    timeout_seconds: int = 300
    max_tokens: int = 1024
    seed: int = 42


@dataclass
class WorldConfig:
    start_location: str = "village_square"
    location_graph: dict[str, list[str]] = field(
        default_factory=lambda: {
            "village_square": ["forest_shrine", "cave_pool", "river_dock", "ancient_vault"],
            "forest_shrine": ["village_square"],
            "cave_pool": ["village_square"],
            "river_dock": ["village_square"],
            "ancient_vault": ["village_square"],
        }
    )
    location_descriptions: dict[str, str] = field(
        default_factory=lambda: {
            "village_square": "A central plaza where wandering advisors gather around an old map table.",
            "forest_shrine": "A moss-covered shrine where sunbeams pierce the canopy in sharp golden lines.",
            "cave_pool": "A silent cave pool reflecting pale light from cracks in the stone ceiling.",
            "river_dock": "A weathered dock with old cargo crates and rushing water below.",
            "ancient_vault": "A sealed vault door marked with three sigil sockets and a rotating dial.",
        }
    )
    sigil_locations: dict[str, str] = field(
        default_factory=lambda: {
            "sun_sigil": "forest_shrine",
            "moon_sigil": "cave_pool",
            "river_sigil": "river_dock",
        }
    )
    vault_order: list[str] = field(default_factory=lambda: ["sun_sigil", "moon_sigil", "river_sigil"])
    decoy_items: dict[str, str] = field(
        default_factory=lambda: {
            "forest_shrine": "A cracked prayer bowl lies beside the altar.",
            "cave_pool": "An empty lantern flickers weakly near the water.",
            "river_dock": "A rotten coil of rope hangs from a piling.",
            "ancient_vault": "Dust gathers around ancient engravings that describe ritual order.",
        }
    )


@dataclass
class ExperimentConfig:
    max_steps: int = 24
    runs_per_setting: int = 10
    total_npcs: int = 6
    liar_ratios: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5])
    reflection_interval: int = 4
    trust_success_gain: float = 0.18
    trust_failure_decay: float = 0.42
    contradiction_decay: float = 0.12
    distrust_threshold: float = 0.30
    use_hints: bool = False
    role_target_trust: dict[str, float] = field(
        default_factory=lambda: {
            "truthful": 1.0,
            "deceptive": 0.0,
            "opportunistic": 0.35,
            "partial_truth": 0.25,
            "coordinated_deceptive": 0.0,
        }
    )


@dataclass
class FrameworkConfig:
    premium_agent_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider="mock",
            model_name="gpt-4o",
            temperature=0.1,
        )
    )
    budget_npc_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider="mock",
            model_name="gpt-4o-mini",
            temperature=0.3,
        )
    )
    judge_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider="mock",
            model_name="gpt-4o-mini",
            temperature=0.0,
        )
    )
    world: WorldConfig = field(default_factory=WorldConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_default_config() -> FrameworkConfig:
    """Default config using mock LLM for fast local testing."""
    return FrameworkConfig()


def build_tritonai_config() -> FrameworkConfig:
    """Config using the UCSD TritonAI API for real LLM experiments."""
    return FrameworkConfig(
        premium_agent_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.1,
            max_tokens=1024,
            seed=42,
        ),
        budget_npc_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.3,
            max_tokens=512,
            seed=42,
        ),
        judge_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.0,
            max_tokens=512,
            seed=42,
        ),
        experiment=ExperimentConfig(
            max_steps=24,
            runs_per_setting=3,
            total_npcs=6,
            liar_ratios=[0.0, 0.1, 0.3, 0.5],
        ),
    )


def build_hybrid_config() -> FrameworkConfig:
    """Real LLM agent + mock NPCs (cheap iteration)."""
    return FrameworkConfig(
        premium_agent_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
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
            max_steps=24,
            runs_per_setting=3,
            total_npcs=6,
            liar_ratios=[0.0, 0.1, 0.3, 0.5],
        ),
    )


def build_hard_config() -> FrameworkConfig:
    """Hard mode: reduced step budget (18) + NPCs spread across locations.
    All components use real LLM."""
    return FrameworkConfig(
        premium_agent_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.1,
            max_tokens=1024,
            seed=42,
        ),
        budget_npc_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.3,
            max_tokens=512,
            seed=42,
        ),
        judge_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
            api_key_env_var="TRITONAI_API_KEY",
            base_url="https://tritonai-api.ucsd.edu/v1",
            temperature=0.0,
            max_tokens=512,
            seed=42,
        ),
        experiment=ExperimentConfig(
            max_steps=18,
            runs_per_setting=3,
            total_npcs=6,
            liar_ratios=[0.0, 0.1, 0.3, 0.5],
        ),
    )


def build_hard_hybrid_config() -> FrameworkConfig:
    """Hard mode with mock NPCs: reduced step budget (18) + NPCs spread across locations."""
    return FrameworkConfig(
        premium_agent_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
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
            runs_per_setting=3,
            total_npcs=6,
            liar_ratios=[0.0, 0.1, 0.3, 0.5],
        ),
    )


def build_extended_world_config() -> WorldConfig:
    """Extended world with 7 locations, branched topology, and 4 sigils.

    Topology (two branches from village_square):
        village_square --- market_alley --- river_dock
              |                                 |
        forest_shrine                     ancient_vault
              |
          cave_pool --- mountain_pass

    Optimal path ~19 steps, budget of 25 (6 spare).
    4 sigils yield 24 vault permutations (vs 6 for the default 3-sigil world).
    """
    return WorldConfig(
        start_location="village_square",
        location_graph={
            "village_square": ["forest_shrine", "market_alley"],
            "forest_shrine": ["village_square", "cave_pool"],
            "cave_pool": ["forest_shrine", "mountain_pass"],
            "market_alley": ["village_square", "river_dock"],
            "river_dock": ["market_alley", "ancient_vault"],
            "mountain_pass": ["cave_pool"],
            "ancient_vault": ["river_dock"],
        },
        location_descriptions={
            "village_square": "A central plaza where wandering advisors gather around an old map table.",
            "forest_shrine": "A moss-covered shrine where sunbeams pierce the canopy in sharp golden lines.",
            "cave_pool": "A silent cave pool reflecting pale light from cracks in the stone ceiling.",
            "market_alley": "A narrow alley lined with merchant stalls and faded awnings.",
            "river_dock": "A weathered dock with old cargo crates and rushing water below.",
            "mountain_pass": "A windswept pass between jagged peaks, littered with ancient cairns.",
            "ancient_vault": "A sealed vault door marked with four sigil sockets and a rotating dial.",
        },
        sigil_locations={
            "sun_sigil": "forest_shrine",
            "moon_sigil": "cave_pool",
            "river_sigil": "river_dock",
            "star_sigil": "mountain_pass",
        },
        vault_order=["sun_sigil", "star_sigil", "moon_sigil", "river_sigil"],
        decoy_items={
            "forest_shrine": "A cracked prayer bowl lies beside the altar.",
            "cave_pool": "An empty lantern flickers weakly near the water.",
            "market_alley": "A tattered merchant ledger rests on a crate.",
            "river_dock": "A rotten coil of rope hangs from a piling.",
            "mountain_pass": "Wind-worn prayer flags flutter from a cairn.",
            "ancient_vault": "Dust gathers around ancient engravings that describe ritual order.",
        },
    )


def build_extended_hard_hybrid_config() -> FrameworkConfig:
    """Extended world with mock NPCs + real agent: 7 locations, 4 sigils, budget 25."""
    return FrameworkConfig(
        premium_agent_model=ModelConfig(
            provider="tritonai",
            model_name="api-gpt-oss-120b",
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
        world=build_extended_world_config(),
        experiment=ExperimentConfig(
            max_steps=25,
            runs_per_setting=3,
            total_npcs=6,
            liar_ratios=[0.0, 0.1, 0.3, 0.5],
        ),
    )
