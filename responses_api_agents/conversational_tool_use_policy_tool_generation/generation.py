# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-domain policy/tool generation orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from nemo_gym.openai_utils import NeMoGymChatCompletion
from responses_api_agents.conversational_tool_use_policy_tool_generation.assets import (
    PolicyToolAssets,
    load_assets,
)
from responses_api_agents.conversational_tool_use_policy_tool_generation.compat import (
    format_domain_name,
    format_policy_tool_pair,
    parse_judgment,
    parse_policy,
    parse_tools,
    policy_references,
    policy_tool_references,
    sample_timestamp,
    serialize_tools,
    shuffled_pairs,
    tools_artifact,
    validate_tools,
)
from responses_api_agents.conversational_tool_use_policy_tool_generation.models import (
    AttemptTrace,
    CallPhase,
    ModelCallTrace,
    ModelRole,
    PolicyToolGenerationResult,
    PolicyToolGenerationRunRequest,
    PolicyToolGenerationTrace,
    TraceMessage,
)


ModelCaller = Callable[[ModelRole, str, CallPhase, int, int], Awaitable[NeMoGymChatCompletion]]


class AttemptRejected(RuntimeError):
    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(detail)
        self.stage = stage
        self.detail = detail


class PolicyToolGenerationExhaustedError(RuntimeError):
    def __init__(self, trace: PolicyToolGenerationTrace) -> None:
        last = trace.attempts[-1]
        super().__init__(
            f"policy/tool generation exhausted {trace.max_attempts} attempts; "
            f"last failure at {last.failure_stage}: {last.failure_detail}"
        )
        self.trace = trace


def response_text(response: NeMoGymChatCompletion) -> str:
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("model response contains no assistant message content")
    return content


class PolicyToolGenerator:
    def __init__(self, *, max_retries: int = 20) -> None:
        if not 0 <= max_retries <= 20:
            raise ValueError("max_retries must be between 0 and 20")
        self.max_attempts = max_retries + 1

    async def generate(
        self,
        body: PolicyToolGenerationRunRequest,
        caller: ModelCaller,
    ) -> tuple[PolicyToolGenerationResult, PolicyToolGenerationTrace, NeMoGymChatCompletion]:
        assets = load_assets(body.profile)
        domain_name = format_domain_name(body.domain.name)
        trace = PolicyToolGenerationTrace(
            profile=body.profile,
            domain_name=domain_name,
            max_attempts=self.max_attempts,
        )

        for attempt_number in range(1, self.max_attempts + 1):
            attempt = AttemptTrace(attempt=attempt_number)
            trace.attempts.append(attempt)
            try:
                result, final_response = await self._run_attempt(
                    body=body,
                    domain_name=domain_name,
                    assets=assets,
                    attempt=attempt,
                    caller=caller,
                )
                attempt.accepted = True
                return result, trace, final_response
            except AttemptRejected as exc:
                attempt.failure_stage = exc.stage
                attempt.failure_detail = exc.detail
            except Exception as exc:
                attempt.failure_stage = "generation_error"
                attempt.failure_detail = str(exc)

        raise PolicyToolGenerationExhaustedError(trace)

    async def _run_attempt(
        self,
        *,
        body: PolicyToolGenerationRunRequest,
        domain_name: str,
        assets: PolicyToolAssets,
        attempt: AttemptTrace,
        caller: ModelCaller,
    ) -> tuple[PolicyToolGenerationResult, NeMoGymChatCompletion]:
        attempt.timestamp = sample_timestamp()

        initial_pairs = shuffled_pairs(assets.golden_pairs)
        attempt.policy_tool_reference_order = [pair.index for pair in initial_pairs]
        initial_references = policy_tool_references(initial_pairs)

        policy_prompt = assets.policy_prompt.format(domain=domain_name, timestamp=attempt.timestamp)
        policy_prompt += initial_references
        policy_response = await self._call(caller, "policy", policy_prompt, "policy_v1", attempt, ordinal=0)
        policy = parse_policy(response_text(policy_response))
        attempt.calls[-1].parsed = policy
        if policy is None:
            raise AttemptRejected("policy_v1_parse", "response is missing a case-sensitive <policy> tag")

        tools_prompt = assets.tools_prompt.format(domain=domain_name, policy=policy)
        tools_prompt += initial_references
        tools_prompt += f"\n\n<policy>{policy}</policy>"
        tools_response = await self._call(caller, "policy", tools_prompt, "tools_v1", attempt, ordinal=0)
        tools = parse_tools(response_text(tools_response))
        attempt.calls[-1].parsed = tools
        if tools is None:
            raise AttemptRejected("tools_v1_parse", "response tools could not be parsed as JSONL")

        refine_policy_pairs = shuffled_pairs(assets.golden_pairs)
        attempt.policy_reference_order = [pair.index for pair in refine_policy_pairs]
        if body.profile == "general":
            policy_refine_prompt = assets.policy_refine_prompt.format(
                domain=domain_name,
                policy=policy,
                reference_policies=policy_references(refine_policy_pairs),
            )
        else:
            # Proactive refinement consumes this shuffle but omits the references.
            policy_refine_prompt = assets.policy_refine_prompt.format(domain=domain_name, policy=policy)
        policy_refine_response = await self._call(
            caller, "policy", policy_refine_prompt, "policy_refine", attempt, ordinal=0
        )
        policy = parse_policy(response_text(policy_refine_response))
        attempt.calls[-1].parsed = policy
        if policy is None:
            raise AttemptRejected("policy_refine_parse", "response is missing a case-sensitive <policy> tag")

        unused_tools_pairs = shuffled_pairs(assets.golden_pairs)
        attempt.unused_tools_reference_order = [pair.index for pair in unused_tools_pairs]
        tools_refine_prompt = assets.tools_refine_prompt.format(
            domain=domain_name,
            policy=policy,
            tools=serialize_tools(tools),
        )
        tools_refine_response = await self._call(
            caller, "policy", tools_refine_prompt, "tools_refine", attempt, ordinal=0
        )
        tools = parse_tools(response_text(tools_refine_response))
        attempt.calls[-1].parsed = tools
        if tools is None:
            raise AttemptRejected("tools_refine_parse", "response tools could not be parsed as JSONL")

        attempt.tool_validation_passed = validate_tools(tools)
        if not attempt.tool_validation_passed:
            raise AttemptRejected("tool_validation", "tools failed Tau2-compatible permissive validation")

        cohesion_prompt = assets.cohesion_judge_prompt.format(
            domain=domain_name,
            policy=policy,
            tools=serialize_tools(tools),
        )
        cohesion_responses = await self._call_concurrently(
            caller=caller,
            role="judge",
            prompt_targets=[(cohesion_prompt, None)] * 3,
            phase="cohesion_judge",
            attempt=attempt,
        )
        cohesion_judgments = [parse_judgment(response_text(response)) for response in cohesion_responses]
        for call, judgment in zip(attempt.calls[-3:], cohesion_judgments, strict=True):
            call.parsed = judgment
        cohesion_failures = sum(judgment == False for judgment in cohesion_judgments)  # noqa: E712
        attempt.cohesion_failure_fraction = cohesion_failures / len(cohesion_judgments)
        if attempt.cohesion_failure_fraction > 0.5:
            raise AttemptRejected(
                "cohesion_judge",
                f"policy/tools failed {cohesion_failures}/{len(cohesion_judgments)} cohesion judgments",
            )

        golden_pairs = shuffled_pairs(assets.golden_pairs)
        attempt.golden_reference_order = [pair.index for pair in golden_pairs]
        generated_tools = serialize_tools(tools)
        format_0_prompts = [
            assets.golden_judge_prompt
            + format_policy_tool_pair(pair.policy, pair.tools, 0)
            + format_policy_tool_pair(policy, generated_tools, 1)
            for pair in golden_pairs[:2]
        ]
        # Duplicate the format-0 prompts and only flip the target labels.
        golden_prompt_targets = [
            (format_0_prompts[0], 1),
            (format_0_prompts[1], 1),
            (format_0_prompts[0], 0),
            (format_0_prompts[1], 0),
        ]
        golden_responses = await self._call_concurrently(
            caller=caller,
            role="judge",
            prompt_targets=golden_prompt_targets,
            phase="golden_judge",
            attempt=attempt,
        )
        golden_judgments = [parse_judgment(response_text(response)) for response in golden_responses]
        golden_calls = attempt.calls[-4:]
        for call, judgment in zip(golden_calls, golden_judgments, strict=True):
            call.parsed = judgment
        golden_losses = sum(
            judgment == target for judgment, (_, target) in zip(golden_judgments, golden_prompt_targets, strict=True)
        )
        attempt.golden_failure_fraction = golden_losses / len(golden_judgments)
        if attempt.golden_failure_fraction > 0.5:
            raise AttemptRejected(
                "golden_judge",
                f"generated policy/tools lost {golden_losses}/{len(golden_judgments)} golden comparisons",
            )

        policy_md = policy
        tools_jsonl = tools_artifact(tools)
        result = PolicyToolGenerationResult(
            profile=body.profile,
            domain=body.domain,
            attempt_count=attempt.attempt,
            policy_md=policy_md,
            tools=tools,
            tools_jsonl=tools_jsonl,
        )
        return result, tools_refine_response

    async def _call(
        self,
        caller: ModelCaller,
        role: ModelRole,
        prompt: str,
        phase: CallPhase,
        attempt: AttemptTrace,
        *,
        ordinal: int,
        generated_target_index: int | None = None,
    ) -> NeMoGymChatCompletion:
        try:
            response = await caller(role, prompt, phase, attempt.attempt, ordinal)
        except Exception as exc:
            raise AttemptRejected(phase, str(exc)) from exc
        attempt.calls.append(
            ModelCallTrace(
                role=role,
                phase=phase,
                attempt=attempt.attempt,
                ordinal=ordinal,
                messages=[TraceMessage(content=prompt)],
                response=response.model_dump(mode="json"),
                generated_target_index=generated_target_index,
            )
        )
        return response

    async def _call_concurrently(
        self,
        *,
        caller: ModelCaller,
        role: ModelRole,
        prompt_targets: list[tuple[str, int | None]],
        phase: CallPhase,
        attempt: AttemptTrace,
    ) -> list[NeMoGymChatCompletion]:
        try:
            responses = await asyncio.gather(
                *(
                    caller(role, prompt, phase, attempt.attempt, ordinal)
                    for ordinal, (prompt, _) in enumerate(prompt_targets)
                )
            )
        except Exception as exc:
            raise AttemptRejected(phase, str(exc)) from exc
        for ordinal, (response, (prompt, target)) in enumerate(zip(responses, prompt_targets, strict=True)):
            attempt.calls.append(
                ModelCallTrace(
                    role=role,
                    phase=phase,
                    attempt=attempt.attempt,
                    ordinal=ordinal,
                    messages=[TraceMessage(content=prompt)],
                    response=response.model_dump(mode="json"),
                    generated_target_index=target,
                )
            )
        return responses
