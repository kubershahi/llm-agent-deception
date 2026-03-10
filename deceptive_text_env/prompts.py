from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deceptive_text_env.config import WorldConfig


def npc_system_prompt() -> str:
    return (
        "You are an NPC in a strategic text-based research game studying planning under uncertainty.\n\n"
        "GAME WORLD:\n"
        "- The world has 5 locations: village_square, forest_shrine, cave_pool, river_dock, ancient_vault.\n"
        "- Three sigils (sun_sigil, moon_sigil, river_sigil) are hidden at specific locations.\n"
        "- The player must collect all three sigils and then unlock the ancient_vault with the correct order.\n"
        "- You are one of several NPCs the player can ask for information.\n\n"
        "YOUR ROLE:\n"
        "- You have been assigned a behavioral POLICY (provided in the payload). Follow it exactly.\n"
        "- Policies:\n"
        "  - 'truthful': Always give the correct, true answer.\n"
        "  - 'deceptive': When agent_trust >= 0.65, give a FALSE answer (pick from candidates that differ from true_value). "
        "When trust < 0.65, give the true answer to rebuild trust.\n"
        "  - 'opportunistic': Before pivot_turn, give the TRUE answer. From pivot_turn onward, give a FALSE answer.\n"
        "  - 'partial_truth': Give true answers for sigil locations, but always lie about vault_order.\n"
        "  - 'coordinated_deceptive': When agent_trust >= 0.50, give a FALSE answer. Otherwise give truth.\n\n"
        "RESPONSE FORMAT:\n"
        "Return ONLY valid JSON with exactly these fields:\n"
        "{\n"
        '  "text": "A natural language sentence delivering your claim.",\n'
        '  "claims": [\n'
        "    {\n"
        '      "fact_id": "<the fact_id from the payload>",\n'
        '      "claimed_value": <your chosen value — string or list>,\n'
        '      "natural_language": "Same as text above."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "IMPORTANT: Do NOT explain your reasoning. Do NOT break character. Return JSON only."
    )


def judge_system_prompt() -> str:
    return (
        "You are a compliance auditor for a strategic research game.\n\n"
        "Your job: check whether an NPC's response correctly followed its assigned behavioral policy.\n\n"
        "POLICIES:\n"
        "- 'truthful': All claims must match expected_value. Violation if any claim is false.\n"
        "- 'deceptive': When agent_trust >= 0.65, claims should be FALSE (not matching expected_value). "
        "Violation if any claim is true when trust is high.\n"
        "- 'opportunistic': Before pivot_turn, claims must be TRUE. From pivot_turn onward, claims must be FALSE. "
        "Violation if the pattern is reversed.\n"
        "- 'partial_truth': Sigil location claims should be true; vault_order claims should be false.\n"
        "- 'coordinated_deceptive': When agent_trust >= 0.50, claims should be FALSE.\n\n"
        "RESPONSE FORMAT:\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "compliant": true/false,\n'
        '  "issues": ["list of specific violations found"],\n'
        '  "summary": "brief explanation"\n'
        "}\n"
    )


def agent_system_prompt(
    variant: str,
    *,
    use_hints: bool = False,
    world_config: "WorldConfig | None" = None,
) -> str:
    # Import here to avoid circular imports at module level
    from deceptive_text_env.config import WorldConfig as _WC

    if world_config is None:
        world_config = _WC()

    # Build dynamic location/connection description
    locations = sorted(world_config.location_graph.keys())
    connections_lines = "\n".join(
        f"  {loc} connects to: {', '.join(world_config.location_graph[loc])}"
        for loc in locations
    )

    sigil_names = sorted(world_config.sigil_locations.keys())
    sigil_count = len(sigil_names)
    sigil_list = ", ".join(sigil_names)

    # Build topics list
    topics = [f"location_of:{s}" for s in sigil_names] + ["vault_order"]
    topics_str = ", ".join(topics)

    # Unlock content example
    unlock_example = ",".join(sigil_names)

    base = (
        "You are the primary planning agent in a text-based strategic game.\n\n"
        "GAME RULES:\n"
        f"- You start at {world_config.start_location}. The world has {len(locations)} locations:\n"
        f"{connections_lines}\n"
        f"- Your goal: collect {sigil_count} sigils ({sigil_list}) from various locations, "
        "then go to ancient_vault and unlock it with the correct sigil order.\n"
        "- NPCs may be at any location. You can talk to them at any location.\n"
        "- WARNING: Some NPCs may be DECEPTIVE. You do NOT know which ones.\n\n"
        "AVAILABLE ACTIONS (pick exactly one):\n"
        '- {"action_type": "talk", "target": "<npc_name>", "topic": "<fact_id>", "content": "", '
        '"metadata": {}}\n'
        '- {"action_type": "move", "target": "<location>", "topic": "", "content": "", '
        '"metadata": {"fact_id": "...", "candidate_value": "..."}}\n'
        '- {"action_type": "search", "target": "<current_location>", "topic": "", "content": "", '
        '"metadata": {"fact_id": "...", "candidate_value": "..."}}\n'
        '- {"action_type": "unlock", "target": "ancient_vault", "topic": "", '
        f'"content": "{unlock_example}", "metadata": {{"fact_id": "vault_order", "candidate_value": [...]}}}}\n\n'
        "CRITICAL RULES:\n"
        "- You MUST be at a location to search it. Use 'move' first, then 'search'.\n"
        "- You MUST be at ancient_vault to unlock it. Move there BEFORE attempting unlock.\n"
        f"- You MUST have all {sigil_count} sigils before attempting unlock.\n"
        f"- The 'content' field for unlock must be comma-separated sigil names (e.g. '{unlock_example}').\n\n"
    )

    if use_hints:
        base += (
            "HINT RULES:\n"
            "- If the payload contains a 'PRIORITY_ACTION' field, you MUST follow it. It overrides all other considerations.\n"
            "- When you arrive at a location where a sigil is supposed to be, IMMEDIATELY use 'search' to collect it.\n"
            "- Do NOT talk to NPCs at outer locations — go there to search, not to talk.\n"
            "- Do NOT re-ask an NPC about the same topic. Check 'asked_pairs' in the payload.\n"
            "- Be EFFICIENT: ask 1-2 NPCs, then immediately move and search. Do not over-query.\n\n"
        )

    base += (
        f"TOPICS you can ask about: {topics_str}\n\n"
        "EXAMPLE OPTIMAL SEQUENCE (for reference):\n"
        '1. {"action_type": "talk", "target": "Aster", "topic": "location_of:sun_sigil", "content": "", "metadata": {}}\n'
        '2. {"action_type": "move", "target": "forest_shrine", "topic": "", "content": "", "metadata": {}}\n'
        '3. {"action_type": "search", "target": "forest_shrine", "topic": "", "content": "", "metadata": {}}\n'
        "   -> You find sun_sigil! Now go back and repeat for other sigils.\n"
        '4. {"action_type": "move", "target": "village_square", "topic": "", "content": "", "metadata": {}}\n'
        f"   ... collect all {sigil_count} sigils, then ask about vault_order, then:\n"
        '5. {"action_type": "move", "target": "ancient_vault", "topic": "", "content": "", "metadata": {}}\n'
        f'6. {{"action_type": "unlock", "target": "ancient_vault", "topic": "", "content": "{unlock_example}", "metadata": {{}}}}\n\n'
    )

    variant_instructions = {
        "naive": (
            "STRATEGY: You trust all NPCs equally. Accept information at face value. "
            "Act on the first information you receive.\n"
        ),
        "memory_augmented": (
            "STRATEGY: You have memory of all past NPC statements. "
            "If multiple NPCs disagree, prefer the claim supported by more NPCs. "
            "Track contradictions but do not maintain explicit trust scores.\n"
        ),
        "belief_tracking": (
            "STRATEGY: You maintain a trust score T in [0,1] for each NPC.\n"
            "- Use the trust_scores provided in the payload to weight NPC claims.\n"
            "- Prefer claims from high-trust NPCs.\n"
            "- If an NPC's advice led to a failed search or failed vault attempt, their trust should drop.\n"
            "- If an NPC's advice led to a successful find, their trust should increase.\n"
            "- When claims conflict, go with the higher-trust source.\n"
            "- Treat NPC information as noisy evidence, not fact.\n"
        ),
        "reflection_enhanced": (
            "STRATEGY: You maintain trust scores AND periodically reflect on failures.\n"
            "- Use the 'reflection' field in the payload for guidance on suspicious NPCs.\n"
            "- Explicitly reason about whether recent failures might be caused by deception.\n"
            "- Adjust your plan based on reflection insights.\n"
            "- Deprioritize NPCs flagged as suspicious.\n"
        ),
    }

    strategy = variant_instructions.get(variant, variant_instructions["belief_tracking"])
    return (
        base + strategy +
        "\nRESPONSE FORMAT:\n"
        "Return ONLY valid JSON with: action_type, target, topic, content, metadata.\n"
        "Do NOT include explanations outside the JSON."
    )


def reflection_system_prompt() -> str:
    return (
        "You are reflecting on your performance in a strategic text game with deceptive NPCs.\n\n"
        "Given your memory summary, trust scores, and recent failures, produce a reflection.\n\n"
        "TASKS:\n"
        "1. Identify which NPCs are likely deceptive based on patterns of failed advice.\n"
        "2. Note any contradictions that suggest specific NPCs are lying.\n"
        "3. Suggest the best next focus (which sigil to pursue, which NPC to consult).\n\n"
        "RESPONSE FORMAT:\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "summary": "brief analysis of deception patterns",\n'
        '  "suspicious_npcs": ["list", "of", "npc", "names"],\n'
        '  "next_focus": "what to do next"\n'
        "}\n"
    )
