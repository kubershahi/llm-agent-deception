from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

from deceptive_text_env.config import ModelConfig

_LOG_DIR: Path | None = None
_LOG_COUNTER = 0


def enable_call_logging(log_dir: str = "llm_logs") -> None:
    """Enable saving every LLM call to a JSONL file."""
    global _LOG_DIR
    _LOG_DIR = Path(log_dir)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_call(task: str, model: str, messages: list, response_content: str, parsed: dict) -> None:
    global _LOG_COUNTER
    if _LOG_DIR is None:
        return
    _LOG_COUNTER += 1
    entry = {
        "call_index": _LOG_COUNTER,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task,
        "model": model,
        "messages": messages,
        "raw_response": response_content,
        "parsed_json": parsed,
    }
    log_file = _LOG_DIR / "calls.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_json(
        self,
        *,
        task: str,
        system_prompt: str,
        user_prompt: str,
        payload: dict[str, Any],
        model_config: ModelConfig,
    ) -> dict[str, Any]:
        raise NotImplementedError


class TritonAIClient(BaseLLMClient):
    """Client for the UCSD TritonAI API (OpenAI-compatible)."""

    def generate_json(
        self,
        *,
        task: str,
        system_prompt: str,
        user_prompt: str,
        payload: dict[str, Any],
        model_config: ModelConfig,
    ) -> dict[str, Any]:
        api_key = os.getenv(model_config.api_key_env_var, "")
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{model_config.api_key_env_var}' is not set."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = {
            "model": model_config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Task: {task}\n\nInstructions:\n{user_prompt}\n\nPayload:\n{json.dumps(payload, indent=2)}",
                },
            ],
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "max_completion_tokens": model_config.max_tokens,
            "seed": model_config.seed,
        }
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                response = requests.post(
                    f"{model_config.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=model_config.timeout_seconds,
                )
                response.raise_for_status()
                response_payload = response.json()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                if attempt < 3:
                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"TritonAI request failed after 4 attempts: {last_exc}") from last_exc
        content = response_payload["choices"][0]["message"]["content"]
        parsed = _safe_parse_json(content)
        _log_call(task, model_config.model_name, body["messages"], content, parsed)
        return parsed


class OpenAICompatibleLLMClient(BaseLLMClient):
    def generate_json(
        self,
        *,
        task: str,
        system_prompt: str,
        user_prompt: str,
        payload: dict[str, Any],
        model_config: ModelConfig,
    ) -> dict[str, Any]:
        api_key = os.getenv(model_config.api_key_env_var, "")
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{model_config.api_key_env_var}' is not set for provider '{model_config.provider}'."
            )

        body = {
            "model": model_config.model_name,
            "temperature": model_config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Task: {task}\n\nInstructions:\n{user_prompt}\n\nPayload:\n{json.dumps(payload, indent=2)}",
                },
            ],
        }
        request = urllib.request.Request(
            url=f"{model_config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=model_config.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_payload = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {error_payload}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        content = response_payload["choices"][0]["message"]["content"]
        return _safe_parse_json(content)


class MockLLMClient(BaseLLMClient):
    def generate_json(
        self,
        *,
        task: str,
        system_prompt: str,
        user_prompt: str,
        payload: dict[str, Any],
        model_config: ModelConfig,
    ) -> dict[str, Any]:
        if task == "npc_response":
            return self._npc_response(payload)
        if task == "judge_npc_response":
            return self._judge_response(payload)
        if task == "agent_action":
            return self._agent_action(payload)
        if task == "agent_reflection":
            return self._agent_reflection(payload)
        raise ValueError(f"Unsupported mock task: {task}")

    def _npc_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        policy = payload["policy"]
        topic = payload["topic"]
        trust = float(payload.get("agent_trust", 0.5))
        turn_index = int(payload.get("turn_index", 0))
        true_value = payload["true_value"]
        candidates = list(payload.get("candidates", []))
        pivot_turn = int(payload.get("pivot_turn", 3))
        false_candidates = [candidate for candidate in candidates if candidate != true_value]
        fallback_false = false_candidates[0] if false_candidates else true_value

        if policy == "truthful":
            chosen_value = true_value
        elif policy == "deceptive":
            chosen_value = fallback_false if trust >= 0.65 else true_value
        elif policy == "opportunistic":
            chosen_value = true_value if turn_index < pivot_turn else fallback_false
        elif policy == "partial_truth":
            if topic == "vault_order":
                chosen_value = fallback_false
            else:
                chosen_value = true_value
        elif policy == "coordinated_deceptive":
            chosen_value = fallback_false if trust >= 0.50 else true_value
        else:
            chosen_value = true_value

        fact_id = payload["fact_id"]
        text = self._claim_text(topic, chosen_value)
        return {
            "text": text,
            "claims": [
                {
                    "fact_id": fact_id,
                    "claimed_value": chosen_value,
                    "natural_language": text,
                }
            ],
        }

    def _judge_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        policy = payload["policy"]
        trust = float(payload.get("agent_trust", 0.5))
        turn_index = int(payload.get("turn_index", 0))
        pivot_turn = int(payload.get("pivot_turn", 3))
        truths = [bool(item["is_true"]) for item in payload.get("verified_claims", [])]
        compliant = True
        issues: list[str] = []

        if policy == "truthful" and not all(truths):
            compliant = False
            issues.append("Truthful NPC emitted at least one false claim.")
        elif policy == "deceptive" and trust >= 0.65 and any(truths):
            compliant = False
            issues.append("Deceptive NPC should lie when trust is high.")
        elif policy == "opportunistic":
            if turn_index < pivot_turn and not all(truths):
                compliant = False
                issues.append("Opportunistic NPC should stay truthful before the pivot.")
            if turn_index >= pivot_turn and all(truths):
                compliant = False
                issues.append("Opportunistic NPC should pivot toward deception after the pivot turn.")
        elif policy == "partial_truth":
            pass
        elif policy == "coordinated_deceptive":
            if trust >= 0.50 and any(truths):
                compliant = False
                issues.append("Coordinated deceptive NPC should lie when trust >= 0.5.")

        return {
            "compliant": compliant,
            "issues": issues,
            "summary": "Policy audit completed.",
        }

    def _agent_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        variant = payload["variant"]
        location = payload["location"]
        collected_sigils = set(payload["collected_sigils"])
        inventory = list(payload["inventory"])
        needed_sigils = [sigil for sigil in ["sun_sigil", "moon_sigil", "river_sigil"] if sigil not in collected_sigils]
        trust_scores = payload["trust_scores"]
        known_claims = payload["known_claims"]
        available_npcs = list(payload["available_npcs"])
        accessible_locations = list(payload["accessible_locations"])
        failed_unlock_orders = payload.get("failed_unlock_orders", [])
        asked_pairs = {tuple(item) for item in payload.get("asked_pairs", [])}
        suspicious_npcs = set(payload.get("reflection", {}).get("suspicious_npcs", []))
        contradictions = payload.get("contradictions", [])

        disproven_values: dict[str, set[str]] = {}
        for item in contradictions:
            fact_id = str(item.get("fact_id", ""))
            disproven = item.get("disproven_value")
            if fact_id not in disproven_values:
                disproven_values[fact_id] = set()
            if isinstance(disproven, list):
                disproven_values[fact_id].add("|".join(str(part) for part in disproven))
            elif disproven is not None:
                disproven_values[fact_id].add(str(disproven))

        def route_target(destination: str) -> str:
            if destination in accessible_locations:
                return destination
            if location != "village_square" and "village_square" in accessible_locations:
                return "village_square"
            return accessible_locations[0] if accessible_locations else location

        def fact_id_for_sigil(sigil: str) -> str:
            return f"location_of:{sigil}"

        def best_claim(fact_id: str) -> dict[str, Any] | None:
            candidates = list(known_claims.get(fact_id, []))
            if not candidates:
                return None
            invalid_values = disproven_values.get(fact_id, set())
            filtered_candidates = []
            for candidate in candidates:
                raw_value = candidate["claimed_value"]
                value_key = "|".join(str(part) for part in raw_value) if isinstance(raw_value, list) else str(raw_value)
                if value_key not in invalid_values:
                    filtered_candidates.append(candidate)
            candidates = filtered_candidates or candidates
            if variant == "naive":
                return sorted(candidates, key=lambda item: item["turn_index"])[0]
            if variant == "memory_augmented":
                score_map: dict[str, tuple[int, float]] = {}
                for candidate in candidates:
                    value = str(candidate["claimed_value"])
                    support, trust_total = score_map.get(value, (0, 0.0))
                    score_map[value] = (support + 1, trust_total + float(candidate["trust_at_record"]))
                best_value = max(score_map.items(), key=lambda item: (item[1][0], item[1][1]))[0]
                for candidate in candidates:
                    if str(candidate["claimed_value"]) == best_value:
                        return candidate
                return candidates[0]
            weighted = sorted(
                candidates,
                key=lambda item: (
                    float(item["trust_at_record"]),
                    float(trust_scores.get(item["npc_name"], 0.5)),
                    -int(item["turn_index"]),
                ),
                reverse=True,
            )
            for candidate in weighted:
                if candidate["npc_name"] not in suspicious_npcs:
                    return candidate
            return weighted[0]

        if len(collected_sigils) == 3:
            vault_claim = best_claim("vault_order")
            if vault_claim is None and available_npcs:
                npc_name = self._pick_npc(available_npcs, variant, trust_scores, suspicious_npcs)
                return {
                    "action_type": "talk",
                    "target": npc_name,
                    "topic": "vault_order",
                    "content": "",
                    "metadata": {},
                }
            if location != "ancient_vault":
                next_hop = route_target("ancient_vault")
                return {
                    "action_type": "move",
                    "target": next_hop,
                    "topic": "",
                    "content": "",
                    "metadata": {"reason": "ready_for_vault", "candidate_value": "ancient_vault"},
                }
            if vault_claim and vault_claim["claimed_value"] not in failed_unlock_orders:
                return {
                    "action_type": "unlock",
                    "target": "ancient_vault",
                    "topic": "",
                    "content": ",".join(vault_claim["claimed_value"]),
                    "metadata": {"fact_id": "vault_order", "candidate_value": vault_claim["claimed_value"]},
                }
            if available_npcs:
                npc_name = self._pick_npc(available_npcs, variant, trust_scores, suspicious_npcs)
                return {
                    "action_type": "talk",
                    "target": npc_name,
                    "topic": "vault_order",
                    "content": "",
                    "metadata": {},
                }

        for sigil in needed_sigils:
            fact_id = fact_id_for_sigil(sigil)
            claim = best_claim(fact_id)
            invalid_values = disproven_values.get(fact_id, set())
            if claim is None:
                npc_name = self._pick_npc(
                    [name for name in available_npcs if (name, fact_id) not in asked_pairs] or available_npcs,
                    variant,
                    trust_scores,
                    suspicious_npcs,
                )
                if npc_name:
                    return {
                        "action_type": "talk",
                        "target": npc_name,
                        "topic": fact_id,
                        "content": "",
                        "metadata": {},
                    }
                continue
            destination = str(claim["claimed_value"])
            if destination in invalid_values:
                if available_npcs:
                    npc_name = self._pick_npc(
                        [name for name in available_npcs if (name, fact_id) not in asked_pairs] or available_npcs,
                        variant,
                        trust_scores,
                        suspicious_npcs,
                    )
                    return {
                        "action_type": "talk",
                        "target": npc_name,
                        "topic": fact_id,
                        "content": "",
                        "metadata": {},
                    }
                if location != "village_square":
                    return {
                        "action_type": "move",
                        "target": route_target("village_square"),
                        "topic": "",
                        "content": "",
                        "metadata": {"reason": "seek_new_advice", "fact_id": fact_id},
                    }
            if location != destination:
                next_hop = route_target(destination)
                return {
                    "action_type": "move",
                    "target": next_hop,
                    "topic": "",
                    "content": "",
                    "metadata": {"fact_id": fact_id, "candidate_value": destination},
                }
            if location == destination:
                return {
                    "action_type": "search",
                    "target": location,
                    "topic": "",
                    "content": "",
                    "metadata": {"fact_id": fact_id, "candidate_value": destination},
                }

        if available_npcs:
            npc_name = self._pick_npc(available_npcs, variant, trust_scores, suspicious_npcs)
            if npc_name:
                return {
                    "action_type": "talk",
                    "target": npc_name,
                    "topic": "vault_order",
                    "content": "",
                    "metadata": {},
                }

        destination = random.choice(accessible_locations) if accessible_locations else location
        return {
            "action_type": "move",
            "target": destination,
            "topic": "",
            "content": "",
            "metadata": {"reason": "fallback_exploration"},
        }

    def _agent_reflection(self, payload: dict[str, Any]) -> dict[str, Any]:
        recent_failures = payload.get("recent_failures", [])
        trust_scores = payload.get("trust_scores", {})
        suspicious_npcs = [name for name, trust in trust_scores.items() if float(trust) <= 0.35]
        summary = "Recent failures suggest that low-trust NPCs may be exploiting earlier credibility."
        if not recent_failures:
            summary = "No major failures yet. Continue gathering corroborating evidence before committing to risky actions."
        return {
            "summary": summary,
            "suspicious_npcs": suspicious_npcs,
            "next_focus": "Favor corroborated claims and deprioritize NPCs whose advice led to failed searches or vault attempts.",
        }

    def _pick_npc(
        self,
        names: list[str],
        variant: str,
        trust_scores: dict[str, float],
        suspicious_npcs: set[str],
    ) -> str:
        if not names:
            return ""
        if variant == "naive":
            return sorted(names)[0]
        ranked = sorted(
            names,
            key=lambda name: (float(trust_scores.get(name, 0.5)), name not in suspicious_npcs, name),
            reverse=True,
        )
        return ranked[0]

    def _claim_text(self, topic: str, chosen_value: Any) -> str:
        if topic.startswith("location_of:"):
            sigil_name = topic.split(":", maxsplit=1)[1].replace("_", " ")
            return f"The {sigil_name} is at {str(chosen_value).replace('_', ' ')}."
        if topic == "vault_order":
            pretty_order = ", ".join(item.replace("_", " ") for item in chosen_value)
            return f"The vault opens with the order: {pretty_order}."
        return f"My information is: {chosen_value}."


def create_llm_client(model_config: ModelConfig) -> BaseLLMClient:
    if model_config.provider == "mock":
        return MockLLMClient()
    if model_config.provider == "tritonai":
        return TritonAIClient()
    if model_config.provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleLLMClient()
    raise ValueError(f"Unsupported LLM provider: {model_config.provider}")


def _safe_parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])
