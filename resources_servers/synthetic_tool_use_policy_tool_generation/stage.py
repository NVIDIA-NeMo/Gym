# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy and tool generation, refinement, and quality gating."""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from resources_servers.synthetic_tool_use.common.artifacts import RunArtifactStore
from resources_servers.synthetic_tool_use.common.clients import (
    AsyncTextGenerator,
    ProviderGenerationError,
)
from resources_servers.synthetic_tool_use.common.models import SeedGenerationConfig, StageState
from resources_servers.synthetic_tool_use.common.parsing import (
    extract_tag,
    parse_json_value,
)
from resources_servers.synthetic_tool_use.common.quality import (
    ArtifactValidationError,
    reject_leaks,
    validate_tools,
)
from resources_servers.synthetic_tool_use_policy_tool_generation.profiles import (
    PolicyToolsProfile,
    load_golden_pairs,
)
from resources_servers.synthetic_tool_use_policy_tool_generation.rendering import (
    advance_reference_shuffle,
    format_domain_name,
    format_policy_tool_pair,
    sample_timestamp,
    serialize_tools,
    shuffled_policy_references,
    shuffled_policy_tool_references,
)


def _parse_judgment(text: str) -> Any:
    value = extract_tag(text, "judgment").strip()
    lowered = value.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return parse_json_value(value)


def _parse_tools(text: str) -> list[Any]:
    tools = [json.loads(line) for line in text.strip().splitlines() if line.strip()]
    if not tools:
        raise ValueError("response contains no JSONL tool objects")
    return tools


class PolicyToolsGenerationStage:
    def __init__(
        self,
        config: SeedGenerationConfig,
        profile: PolicyToolsProfile,
        store: RunArtifactStore,
        generator: AsyncTextGenerator,
        judge: AsyncTextGenerator | None,
    ) -> None:
        self.config = config
        self.profile = profile
        self.store = store
        self.generator = generator
        self.judge = judge
        self.golden_pairs = load_golden_pairs(profile)

    async def run(self, *, resume: bool = True, source_indexes: set[int] | None = None) -> None:
        pending_domain_ids = []
        for entry in self.store.load_manifest().domains:
            if source_indexes is not None and entry.source_index not in source_indexes:
                continue
            domain_dir = self.store.domains_dir / entry.artifact_dir
            if resume and entry.stages["policy_tools"].state == StageState.COMPLETE:
                try:
                    policy = (domain_dir / "policy.md").read_text(encoding="utf-8")
                    tools = [json.loads(line) for line in (domain_dir / "tools.jsonl").read_text().splitlines()]
                    reject_leaks(policy, artifact_name="policy.md")
                    validate_tools(tools)
                    continue
                except (OSError, ValueError, ArtifactValidationError):
                    pass
            pending_domain_ids.append(entry.domain_id)
        await asyncio.gather(*(self._run_domain(domain_id) for domain_id in pending_domain_ids))

    async def _run_domain(self, domain_id: str) -> None:
        domain = self.store.load_domain(domain_id)
        self.store.update_stage(domain_id, "policy_tools", StageState.RUNNING)
        last_error: Exception | None = None
        for semantic_attempt in range(1, self.config.policy_tools.semantic_attempts + 1):
            responses: dict[str, Any] = {}
            try:
                rng = random.Random(f"{self.config.random_seed}:{domain_id}:{semantic_attempt}:policy-tools-v1")
                domain_name = format_domain_name(domain.name)
                timestamp = sample_timestamp(rng)
                reference_policy_tools = shuffled_policy_tool_references(self.golden_pairs, rng)
                policy_prompt = self.profile.policy_prompt.format(
                    domain=domain_name,
                    timestamp=timestamp,
                )
                policy_prompt += reference_policy_tools
                policy_response = await self.generator.generate([{"role": "user", "content": policy_prompt}])
                responses["policy"] = policy_response.raw_response
                policy = extract_tag(policy_response.text, "policy")

                tools_prompt = self.profile.tools_prompt.format(
                    domain=domain_name,
                    policy=policy,
                )
                tools_prompt += reference_policy_tools
                tools_prompt += f"\n\n<policy>{policy}</policy>"
                tools_response = await self.generator.generate([{"role": "user", "content": tools_prompt}])
                responses["tools"] = tools_response.raw_response
                tools = _parse_tools(extract_tag(tools_response.text, "tools"))
                if self.config.policy_tools.refine:
                    reference_policies = shuffled_policy_references(self.golden_pairs, rng)
                    policy_refine_prompt = self.profile.policy_refine_prompt.format(
                        domain=domain_name,
                        policy=policy,
                        reference_policies=reference_policies,
                    )
                    policy_refine_response = await self.generator.generate(
                        [{"role": "user", "content": policy_refine_prompt}]
                    )
                    policy = extract_tag(policy_refine_response.text, "policy")
                    responses["policy_refine"] = policy_refine_response.raw_response

                    advance_reference_shuffle(self.golden_pairs, rng)
                    tools_refine_prompt = self.profile.tools_refine_prompt.format(
                        domain=domain_name,
                        policy=policy,
                        tools=serialize_tools(tools),
                    )
                    tools_refine_response = await self.generator.generate(
                        [{"role": "user", "content": tools_refine_prompt}]
                    )
                    tools = _parse_tools(extract_tag(tools_refine_response.text, "tools"))
                    responses["tools_refine"] = tools_refine_response.raw_response

                policy = policy.strip()
                if not policy:
                    raise ArtifactValidationError("empty_policy", "generated policy is empty")
                reject_leaks(policy, artifact_name="policy")
                validated_tools = validate_tools(tools)
                judgments = await self._judge(domain_name, policy, tools, rng)
                quality_report = {
                    "accepted": True,
                    "semantic_attempt": semantic_attempt,
                    "deterministic_validation": "passed",
                    "judgments": judgments,
                }
                self.store.write_attempt(
                    domain_id,
                    "policy_tools",
                    semantic_attempt,
                    {"accepted": True, "responses": responses, "quality": quality_report},
                )
                self.store.promote_policy_tools(domain_id, policy, validated_tools, quality_report)
                self.store.update_stage(
                    domain_id,
                    "policy_tools",
                    StageState.COMPLETE,
                    attempts=semantic_attempt,
                )
                return
            except Exception as exc:
                last_error = exc
                self.store.write_attempt(
                    domain_id,
                    "policy_tools",
                    semantic_attempt,
                    {
                        "accepted": False,
                        "failure_category": getattr(exc, "reason", "generation_error"),
                        "failure_detail": str(exc),
                        "responses": responses,
                    },
                )
                if isinstance(exc, ProviderGenerationError):
                    self.store.update_stage(
                        domain_id,
                        "policy_tools",
                        StageState.FAILED,
                        attempts=semantic_attempt,
                        failure_category="provider_error",
                        failure_detail=str(exc),
                    )
                    return
        self.store.update_stage(
            domain_id,
            "policy_tools",
            StageState.FAILED,
            attempts=self.config.policy_tools.semantic_attempts,
            failure_category=getattr(last_error, "reason", "generation_error"),
            failure_detail=str(last_error),
        )

    async def _judge(
        self,
        domain: str,
        policy: str,
        tools: list[dict[str, Any]],
        rng: random.Random,
    ) -> dict[str, list[Any]]:
        if not self.config.policy_tools.judge_enabled:
            return {"cohesion": [], "golden_comparison": []}
        if self.judge is None:
            raise ValueError("policy/tool judging is enabled but no judge model is configured")
        prompt = self.profile.cohesion_judge_prompt.format(
            domain=domain,
            policy=policy,
            tools=serialize_tools(tools),
        )
        judgments = []
        failures = 0
        responses = await asyncio.gather(
            *(
                self.judge.generate([{"role": "user", "content": prompt}])
                for _ in range(self.config.policy_tools.judge_votes)
            )
        )
        for response in responses:
            parsed = _parse_judgment(response.text)
            failures += int(parsed in (False,))
            judgments.append({"judgment": parsed, "raw_response": response.raw_response})
        failure_fraction = failures / len(judgments)
        if failure_fraction > self.config.policy_tools.judge_max_failure_fraction:
            raise ArtifactValidationError(
                "judge_rejection",
                f"policy/tools failed {failures}/{len(judgments)} cohesion judgments",
            )
        comparisons = await self._judge_against_goldens(policy, tools, rng)
        return {"cohesion": judgments, "golden_comparison": comparisons}

    async def _judge_against_goldens(
        self,
        policy: str,
        tools: list[dict[str, Any]],
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        if not self.config.policy_tools.golden_comparison_enabled or not self.golden_pairs:
            return []
        assert self.judge is not None
        pairs = list(self.golden_pairs)
        rng.shuffle(pairs)
        comparison_count = min(self.config.policy_tools.golden_comparison_count, len(pairs))
        first_half_count = (comparison_count + 1) // 2
        generated_tools = serialize_tools(tools)
        prompt_targets: list[tuple[str, int]] = []
        for golden_policy, golden_tools in pairs[:first_half_count]:
            pair_text = format_policy_tool_pair(golden_policy, golden_tools, 0) + format_policy_tool_pair(
                policy, generated_tools, 1
            )
            prompt_targets.append((self.profile.golden_judge_prompt + pair_text, 1))
        # Evaluate each selected prompt from both candidate-index perspectives.
        prompt_targets.extend((prompt, 0) for prompt, _ in list(prompt_targets))
        prompt_targets = prompt_targets[:comparison_count]

        responses = await asyncio.gather(
            *(self.judge.generate([{"role": "user", "content": prompt}]) for prompt, _ in prompt_targets)
        )
        comparisons = []
        generated_losses = 0
        for response, (prompt, generated_index) in zip(responses, prompt_targets, strict=True):
            judgment = _parse_judgment(response.text)
            if isinstance(judgment, dict):
                judgment = judgment.get("index", judgment.get("worst_index"))
            judged_index = int(judgment)
            generated_lost = judged_index == generated_index
            generated_losses += int(generated_lost)
            comparisons.append(
                {
                    "generated_index": generated_index,
                    "judged_worst_index": judged_index,
                    "generated_lost": generated_lost,
                    "prompt": prompt,
                    "raw_response": response.raw_response,
                }
            )
        if generated_losses / len(comparisons) > self.config.policy_tools.judge_max_failure_fraction:
            raise ArtifactValidationError(
                "golden_comparison_rejection",
                f"generated policy/tools lost {generated_losses}/{len(comparisons)} golden comparisons",
            )
        return comparisons
