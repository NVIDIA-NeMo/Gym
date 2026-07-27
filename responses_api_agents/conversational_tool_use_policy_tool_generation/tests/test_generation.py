# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import random
from collections import defaultdict

import pytest

from nemo_gym.openai_utils import NeMoGymChatCompletion
from responses_api_agents.conversational_tool_use_policy_tool_generation.assets import load_assets
from responses_api_agents.conversational_tool_use_policy_tool_generation.compat import (
    policy_references,
    policy_tool_references,
)
from responses_api_agents.conversational_tool_use_policy_tool_generation.generation import (
    PolicyToolGenerationExhaustedError,
    PolicyToolGenerator,
)
from responses_api_agents.conversational_tool_use_policy_tool_generation.models import (
    CallPhase,
    ModelRole,
    PolicyToolGenerationRunRequest,
)


FINAL_TOOL = {
    "name": "lookup_order",
    "doc": "Look up an order.",
    "params": None,
    "returns": None,
    "extra": "kept",
}


def completion(text: str, completion_id: str) -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": completion_id,
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {"content": text, "refusal": None, "role": "assistant"},
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )


def run_request(profile: str = "general") -> PolicyToolGenerationRunRequest:
    return PolicyToolGenerationRunRequest(
        responses_create_params={"input": []},
        profile=profile,
        domain={
            "name": "(Home & Office)/Help Desk",
            "applications": [{"anything": ["raw", 1]}, "also raw"],
        },
    )


class RecordingCaller:
    def __init__(self, *, reject_first_cohesion: bool = False) -> None:
        self.calls: list[tuple[ModelRole, str, CallPhase, int, int]] = []
        self.active: defaultdict[CallPhase, int] = defaultdict(int)
        self.max_active: defaultdict[CallPhase, int] = defaultdict(int)
        self.reject_first_cohesion = reject_first_cohesion

    async def __call__(
        self,
        role: ModelRole,
        prompt: str,
        phase: CallPhase,
        attempt: int,
        ordinal: int,
    ) -> NeMoGymChatCompletion:
        self.calls.append((role, prompt, phase, attempt, ordinal))
        self.active[phase] += 1
        self.max_active[phase] = max(self.max_active[phase], self.active[phase])
        await asyncio.sleep(0.001)
        self.active[phase] -= 1
        if phase == "policy_v1":
            text = "<policy>draft policy</policy>"
        elif phase == "tools_v1":
            text = '<tools>{"name":"draft","doc":"Draft","params":null,"returns":null}</tools>'
        elif phase == "policy_refine":
            text = "<policy>final policy</policy>"
        elif phase == "tools_refine":
            text = f"<tools>{__import__('json').dumps(FINAL_TOOL)}</tools>"
        elif phase == "cohesion_judge":
            judgment = not (self.reject_first_cohesion and attempt == 1 and ordinal < 2)
            text = f"<judgment>{str(judgment).lower()}</judgment>"
        else:
            text = "<judgment>0</judgment>"
        return completion(text, f"{phase}-{attempt}-{ordinal}")


@pytest.mark.asyncio
async def test_exact_call_rng_prompt_and_acceptance_contract() -> None:
    state = random.getstate()
    random.seed(17)
    caller = RecordingCaller()
    try:
        result, generation_trace, final_completion = await PolicyToolGenerator(max_retries=0).generate(
            run_request(), caller
        )
    finally:
        random.setstate(state)

    attempt = generation_trace.attempts[0]
    assert attempt.timestamp == "2025-07-22 17:14:33 EDT"
    assert attempt.policy_tool_reference_order == [8, 4, 1, 6, 2, 7, 3, 5]
    assert attempt.policy_reference_order == [1, 2, 5, 3, 7, 6, 8, 4]
    assert attempt.unused_tools_reference_order == [4, 3, 6, 8, 1, 5, 2, 7]
    assert attempt.golden_reference_order == [8, 7, 2, 6, 1, 5, 3, 4]
    assert [call[2] for call in caller.calls] == [
        "policy_v1",
        "tools_v1",
        "policy_refine",
        "tools_refine",
        "cohesion_judge",
        "cohesion_judge",
        "cohesion_judge",
        "golden_judge",
        "golden_judge",
        "golden_judge",
        "golden_judge",
    ]
    assert [call[0] for call in caller.calls] == ["policy"] * 4 + ["judge"] * 7
    assert caller.max_active["cohesion_judge"] == 3
    assert caller.max_active["golden_judge"] == 4

    assets = load_assets("general")
    pairs = {pair.index: pair for pair in assets.golden_pairs}
    initial = [pairs[index] for index in attempt.policy_tool_reference_order]
    policy_refs = [pairs[index] for index in attempt.policy_reference_order]
    assert caller.calls[0][1] == (
        assets.policy_prompt.format(
            domain="Home__Office-Help_Desk",
            timestamp="2025-07-22 17:14:33 EDT",
        )
        + policy_tool_references(initial)
    )
    assert caller.calls[1][1] == (
        assets.tools_prompt.format(domain="Home__Office-Help_Desk", policy="draft policy")
        + policy_tool_references(initial)
        + "\n\n<policy>draft policy</policy>"
    )
    assert caller.calls[2][1] == assets.policy_refine_prompt.format(
        domain="Home__Office-Help_Desk",
        policy="draft policy",
        reference_policies=policy_references(policy_refs),
    )
    assert caller.calls[7][1] == caller.calls[9][1]
    assert caller.calls[8][1] == caller.calls[10][1]
    assert [call.generated_target_index for call in attempt.calls[-4:]] == [1, 1, 0, 0]
    assert attempt.golden_failure_fraction == 0.5
    assert attempt.accepted
    assert result.policy_md == "final policy"
    assert result.tools == [FINAL_TOOL]
    assert result.tools_jsonl == f"{__import__('json').dumps(FINAL_TOOL)}\n"
    assert final_completion.id == "tools_refine-1-0"


@pytest.mark.asyncio
async def test_proactive_refinement_omits_references_after_consuming_shuffle() -> None:
    state = random.getstate()
    random.seed(17)
    caller = RecordingCaller()
    try:
        _, generation_trace, _ = await PolicyToolGenerator(max_retries=0).generate(run_request("proactive"), caller)
    finally:
        random.setstate(state)

    attempt = generation_trace.attempts[0]
    assets = load_assets("proactive")
    assert attempt.policy_reference_order == [1, 2, 5, 3, 7, 6, 8, 4]
    assert caller.calls[2][1] == assets.policy_refine_prompt.format(
        domain="Home__Office-Help_Desk",
        policy="draft policy",
    )
    assert "<policy_0>" not in caller.calls[2][1]


@pytest.mark.asyncio
async def test_exhaustion_raises_after_twenty_retries_and_twenty_one_attempts() -> None:
    calls = 0

    async def invalid_policy(
        role: ModelRole,
        prompt: str,
        phase: CallPhase,
        attempt: int,
        ordinal: int,
    ) -> NeMoGymChatCompletion:
        nonlocal calls
        del role, prompt, ordinal
        calls += 1
        assert phase == "policy_v1"
        return completion("<POLICY>wrong case</POLICY>", f"invalid-{attempt}")

    with pytest.raises(PolicyToolGenerationExhaustedError) as exc_info:
        await PolicyToolGenerator(max_retries=20).generate(run_request(), invalid_policy)

    assert calls == 21
    assert len(exc_info.value.trace.attempts) == 21
    assert all(attempt.failure_stage == "policy_v1_parse" for attempt in exc_info.value.trace.attempts)


@pytest.mark.asyncio
async def test_more_than_half_cohesion_failures_retries_but_exactly_half_golden_passes() -> None:
    caller = RecordingCaller(reject_first_cohesion=True)
    state = random.getstate()
    try:
        result, generation_trace, _ = await PolicyToolGenerator(max_retries=1).generate(run_request(), caller)
    finally:
        random.setstate(state)

    assert result.attempt_count == 2
    assert len(generation_trace.attempts) == 2
    first, second = generation_trace.attempts
    assert first.cohesion_failure_fraction == pytest.approx(2 / 3)
    assert first.failure_stage == "cohesion_judge"
    assert not first.accepted
    assert second.golden_failure_fraction == 0.5
    assert second.accepted
