# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Customer scenario generation."""

from __future__ import annotations

import asyncio
import json
import random

from resources_servers.synthetic_tool_use.common.artifacts import RunArtifactStore, atomic_write_json
from resources_servers.synthetic_tool_use.common.clients import (
    AsyncTextGenerator,
    ProviderGenerationError,
)
from resources_servers.synthetic_tool_use.common.models import (
    CustomerScenarioArtifact,
    SeedGenerationConfig,
    StageState,
)
from resources_servers.synthetic_tool_use.common.quality import validate_scenario
from resources_servers.synthetic_tool_use_scenario_generation.assets import ScenarioPrompts
from resources_servers.synthetic_tool_use_scenario_generation.schema import (
    CustomerScenario,
    CustomerScenarioCollection,
    scenario_schema_json,
)


def scope_schedule(request_count: int, outside_fraction: float, *, seed: str) -> list[bool]:
    rng = random.Random(seed)
    return [rng.random() < outside_fraction for _ in range(request_count)]


def _parse_scenarios(text: str) -> list[CustomerScenario]:
    canonical = text.strip().removeprefix("```json").removesuffix("```")
    return CustomerScenarioCollection.model_validate_json(canonical).scenarios


class ScenarioGenerationStage:
    def __init__(
        self,
        config: SeedGenerationConfig,
        prompts: ScenarioPrompts,
        store: RunArtifactStore,
        generator: AsyncTextGenerator,
    ) -> None:
        self.config = config
        self.prompts = prompts
        self.store = store
        self.generator = generator

    async def run(self, *, resume: bool = True, source_indexes: set[int] | None = None) -> None:
        pending_domain_ids = []
        for entry in self.store.load_manifest().domains:
            if source_indexes is not None and entry.source_index not in source_indexes:
                continue
            if entry.stages["policy_tools"].state != StageState.COMPLETE:
                continue
            if resume and entry.stages["scenarios"].state == StageState.COMPLETE and entry.scenario_count > 0:
                scenario_files = list((self.store.domains_dir / entry.artifact_dir / "scenarios").glob("**/*.jsonl"))
                if scenario_files:
                    continue
            pending_domain_ids.append(entry.domain_id)
        await asyncio.gather(*(self._run_domain(domain_id) for domain_id in pending_domain_ids))

    async def _run_domain(self, domain_id: str) -> None:
        domain = self.store.load_domain(domain_id)
        policy = (self.store.domain_dir(domain_id) / "policy.md").read_text(encoding="utf-8")
        self.store.update_stage(domain_id, "scenarios", StageState.RUNNING)
        schedule = scope_schedule(
            self.config.scenarios.request_count_per_domain,
            self.config.scenarios.outside_policy_scope_fraction,
            seed=f"{self.config.random_seed}:{domain_id}",
        )
        accepted: list[CustomerScenarioArtifact] = []
        accepted_keys: set[tuple[str, str, str, str | None, str]] = set()
        failed_requests = 0
        schema = scenario_schema_json()
        tasks = [
            asyncio.create_task(
                self._run_request(
                    domain_id=domain_id,
                    domain_name=domain.name,
                    policy=policy,
                    schema=schema,
                    request_index=request_index,
                    outside_scope=outside_scope,
                )
            )
            for request_index, outside_scope in enumerate(schedule)
        ]
        for task in asyncio.as_completed(tasks):
            scenarios = await task
            if scenarios is None:
                failed_requests += 1
                continue
            for scenario in scenarios:
                scenario_key = scenario.create_tuple()
                if scenario_key in accepted_keys:
                    continue
                accepted_keys.add(scenario_key)
                accepted.append(validate_scenario(scenario.model_dump()))

        scenarios = accepted
        if not scenarios:
            self.store.update_stage(
                domain_id,
                "scenarios",
                StageState.FAILED,
                attempts=len(schedule) * self.config.scenarios.semantic_attempts,
                failure_category="no_valid_scenarios",
                failure_detail=f"all {len(schedule)} requests failed",
            )
            return
        run_name = self.config.scenario_model.model.rsplit("/", 1)[-1]
        self.store.promote_scenarios(
            domain_id,
            run_name,
            scenarios,
            self.config.scenarios.scenarios_per_file,
        )
        quality_path = self.store.domain_dir(domain_id) / "quality_report.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality["scenarios"] = {
            "accepted": len(scenarios),
            "failed_requests": failed_requests,
            "outside_policy_scope": sum(item.outside_policy_scope for item in scenarios),
            "inside_policy_scope": sum(not item.outside_policy_scope for item in scenarios),
        }
        atomic_write_json(quality_path, quality)
        self.store.update_stage(
            domain_id,
            "scenarios",
            StageState.COMPLETE,
            attempts=len(schedule),
            scenario_count=len(scenarios),
        )

    async def _run_request(
        self,
        *,
        domain_id: str,
        domain_name: str,
        policy: str,
        schema: str,
        request_index: int,
        outside_scope: bool,
    ) -> list[CustomerScenario] | None:
        last_error: Exception | None = None
        for semantic_attempt in range(1, self.config.scenarios.semantic_attempts + 1):
            system_prompt = self.prompts.system.format(
                domain_policy=policy,
                policy_scope_instruction="does not cover" if outside_scope else "covers",
            )
            user_prompt = self.prompts.user.format(
                scenario_count=self.config.scenarios.scenarios_per_request,
                scenarios_schema=schema,
            )
            attempt = request_index * self.config.scenarios.semantic_attempts + semantic_attempt
            try:
                response = await self.generator.generate(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
            except ProviderGenerationError as exc:
                self.store.write_attempt(
                    domain_id,
                    "scenarios",
                    attempt,
                    {
                        "request_index": request_index,
                        "semantic_attempt": semantic_attempt,
                        "outside_policy_scope": outside_scope,
                        "failure_category": "provider_error",
                        "failure_detail": str(exc),
                    },
                )
                return None
            try:
                scenarios = _parse_scenarios(response.text)
                for scenario in scenarios:
                    scenario.representative_domain = domain_name
                    scenario.outside_policy_scope = outside_scope
                self.store.write_attempt(
                    domain_id,
                    "scenarios",
                    attempt,
                    {
                        "request_index": request_index,
                        "semantic_attempt": semantic_attempt,
                        "outside_policy_scope": outside_scope,
                        "accepted_scenarios": len(scenarios),
                        "raw_response": response.raw_response,
                    },
                )
                return scenarios
            except Exception as exc:
                last_error = exc
                self.store.write_attempt(
                    domain_id,
                    "scenarios",
                    attempt,
                    {
                        "request_index": request_index,
                        "semantic_attempt": semantic_attempt,
                        "outside_policy_scope": outside_scope,
                        "failure_category": getattr(exc, "reason", "generation_error"),
                        "failure_detail": str(exc),
                        "raw_response": response.raw_response,
                    },
                )
        if last_error is not None:
            return None
        return None
