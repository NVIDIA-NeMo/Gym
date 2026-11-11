# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check Atom renderer replay against the internal capture lineage matcher."""

from copy import deepcopy

import pytest

from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models.token_capture.lineage import RolloutLineage


pytest.importorskip("nooa", reason="Atom SDK is installed in its agent environment")
from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.context_blocks.models import BlockMetadata, ResolvedBlock
from nooa.context_blocks.render_config import RenderConfig

from responses_api_agents.nooa_agent.tests import test_gym_llm as fixtures


@pytest.mark.parametrize("with_tools", [False, True])
async def test_cached_renderer_context_remains_before_each_exact_response(with_tools: bool) -> None:
    renderer = RenderConfig()
    static = ResolvedBlock(key="task", content="Fixed task", metadata=BlockMetadata(static=True))
    user = ResolvedBlock(key="request", content="Solve", role=fixtures.Role.USER)
    history = [static, user]
    output = [
        fixtures.NeMoGymResponseOutputMessage(
            id="answer",
            type="message",
            role="assistant",
            status="completed",
            content=[fixtures.NeMoGymResponseOutputText(type="output_text", text="Next step.", annotations=[])],
        )
    ]
    if with_tools:
        output.append(
            fixtures.NeMoGymResponseFunctionToolCallForTraining(
                id="tool-item",
                call_id="tool-call",
                name="execute_python",
                arguments='{"code":"print(1)"}',
                prompt_token_ids=[1],
                generation_token_ids=[2],
                generation_log_probs=[-0.1],
            )
        )
    llm, _, state = fixtures.make_llm(
        fixtures.model_response(*output),
        max_policy_calls=None,
    )
    lineage = RolloutLineage()
    previous = None
    for turn in range(3):
        dynamic = ResolvedBlock(key="context_usage", content=f"Live context {turn}")
        messages = renderer.provider_formatter.format(renderer.block_formatter.format([*history, dynamic]))
        if previous is not None:
            raw_items, _ = fixtures._responses_input(messages, replay_scope=llm._gym_replay_scope)
            raw_request = NeMoGymResponseCreateParamsNonStreaming(input=raw_items)
            assert (
                lineage.resolve(raw_request.model_dump(mode="json", exclude_none=True)["input"]).reason == "no_match"
            )
        response = await llm.acall(messages)
        call = state.calls[-1]
        request = call.request.model_dump(mode="json", exclude_none=True)["input"]
        output = [item.model_dump(mode="json", exclude_none=True) for item in call.response.output]
        if previous is not None:
            assert request[: len(previous)] == previous
            resolution = lineage.resolve(request)
            assert resolution.match.model_call_id == f"call-{turn - 1}"
        previous = request + output
        lineage.record(f"call-{turn}", previous, list(range(turn + 1)), f"digest-{turn}", context_len=len(request))
        history.append(
            ResolvedBlock(key=f"answer-{turn}", content=response.content, role=fixtures.Role.ASSISTANT, event=response)
        )
        if with_tools:
            event = ToolCallEvent(
                tool_call_id="tool-call",
                name="execute_python",
                arguments={"code": "print(1)"},
                llm_response_id=response.id,
                result=ToolResult(tool_call_id="tool-call", content="1"),
            )
            history.append(ResolvedBlock(key=f"tool-{turn}", content="1", role=fixtures.Role.ASSISTANT, event=event))
    # An explicit history edit still fails the unchanged-context matcher.
    edited = deepcopy(request)
    edited[0]["content"] = "Changed original task"
    assert lineage.resolve(edited).reason == "no_match"
