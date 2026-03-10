from __future__ import annotations

import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from deceptive_text_env.agents import build_agent
from deceptive_text_env.config import FrameworkConfig
from deceptive_text_env.evaluation.metrics import aggregate_results, inference_accuracy
from deceptive_text_env.llm import create_llm_client
from deceptive_text_env.npcs import build_npc_roster
from deceptive_text_env.types import EpisodeResult
from deceptive_text_env.world import GroundedVerifier, JudgeModel, TextWorldEnvironment, build_world_facts


class EvaluationRunner:
    def __init__(self, config: FrameworkConfig, *, use_advanced_npcs: bool = False, spread_locations: bool = False) -> None:
        self.config = config
        self.use_advanced_npcs = use_advanced_npcs
        self.spread_locations = spread_locations
        self.agent_client = create_llm_client(config.premium_agent_model)
        self.npc_client = create_llm_client(config.budget_npc_model)
        self.judge_client = create_llm_client(config.judge_model)

    def run_all(
        self,
        agent_variants: Iterable[str],
        *,
        max_workers: int = 1,
    ) -> tuple[list[EpisodeResult], dict[str, dict[str, float]]]:
        jobs: list[tuple[str, float, int]] = []
        for liar_ratio in self.config.experiment.liar_ratios:
            for variant in agent_variants:
                for run_index in range(self.config.experiment.runs_per_setting):
                    seed = self._seed_for(variant, liar_ratio, run_index)
                    jobs.append((variant, liar_ratio, seed))

        if max_workers <= 1:
            results: list[EpisodeResult] = []
            for i, (variant, lr, seed) in enumerate(jobs, 1):
                print(f"  [{i}/{len(jobs)}] {variant} @ LR={lr} (seed={seed})", flush=True)
                results.append(self.run_episode(agent_variant=variant, liar_ratio=lr, seed=seed))
                print(f"    -> {'SUCCESS' if results[-1].success else 'FAIL'} in {results[-1].steps} steps", flush=True)
            return results, aggregate_results(results)

        # Parallel execution
        results = [None] * len(jobs)
        total = len(jobs)
        completed = 0
        print(f"Running {total} episodes with {max_workers} threads...", flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, (variant, lr, seed) in enumerate(jobs):
                future = executor.submit(self.run_episode, agent_variant=variant, liar_ratio=lr, seed=seed)
                future_to_idx[future] = (idx, variant, lr)
            for future in as_completed(future_to_idx):
                idx, variant, lr = future_to_idx[future]
                completed += 1
                try:
                    result = future.result()
                    results[idx] = result
                    status = "SUCCESS" if result.success else "FAIL"
                    print(f"  [{completed}/{total}] {variant} @ LR={lr} -> {status} in {result.steps} steps", flush=True)
                except Exception as e:
                    print(f"  [{completed}/{total}] {variant} @ LR={lr} -> ERROR: {e}", file=sys.stderr, flush=True)
                    raise

        return [r for r in results if r is not None], aggregate_results([r for r in results if r is not None])

    def run_episode(self, *, agent_variant: str, liar_ratio: float, seed: int) -> EpisodeResult:
        random.seed(seed)
        world_facts = build_world_facts(self.config.world)
        verifier = GroundedVerifier(world_facts)
        judge = JudgeModel(self.judge_client, self.config.judge_model)
        npcs = build_npc_roster(
            total_npcs=self.config.experiment.total_npcs,
            liar_ratio=liar_ratio,
            llm_client=self.npc_client,
            model_config=self.config.budget_npc_model,
            use_advanced_strategies=self.use_advanced_npcs,
            spread_locations=self.spread_locations,
            world_config=self.config.world,
        )
        environment = TextWorldEnvironment(
            world_config=self.config.world,
            verifier=verifier,
            judge=judge,
            npcs=npcs,
            max_steps=self.config.experiment.max_steps,
        )
        agent = build_agent(
            variant=agent_variant,
            llm_client=self.agent_client,
            model_config=self.config.premium_agent_model,
            experiment_config=self.config.experiment,
            world_config=self.config.world,
        )

        observation = environment.reset()
        agent.reset([npc.name for npc in npcs])
        done = False
        steps = 0

        while not done and steps < self.config.experiment.max_steps:
            action = agent.select_action(observation)
            result = environment.step(action, agent.trust_scores)
            agent.process_step_result(result)
            observation = result.observation
            steps += 1
            done = result.done or steps >= self.config.experiment.max_steps

        hidden_roles = environment.hidden_role_map()
        score = inference_accuracy(agent.trust_scores, hidden_roles, self.config.experiment)
        return EpisodeResult(
            agent_variant=agent_variant,
            liar_ratio=liar_ratio,
            seed=seed,
            success=environment.state.completed,
            steps=steps,
            final_trust_scores=dict(agent.trust_scores),
            inference_accuracy=score,
            recovery_rate=agent.recovery_rate(),
            hidden_roles=hidden_roles,
            trace=list(agent.trace),
        )

    @staticmethod
    def _seed_for(agent_variant: str, liar_ratio: float, run_index: int) -> int:
        return abs(hash((agent_variant, liar_ratio, run_index))) % (2**31)
