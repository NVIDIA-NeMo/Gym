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

import asyncio
import json
from copy import deepcopy
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from nooa.context_blocks.formatter import OpenAIProviderFormatter
from nooa.context_blocks.models import RenderedMessage, Role, ToolCallInfo
from nooa.llm_types import AssistantReasoning, AssistantText
from nooa.storage.serialization import deserialize, serialize
from nooa.storage.sqlite import SQLiteStorageManager
from nooa.unifiedllm import CacheBoundary, LLMResponse, Tool, ToolCall, UnifiedLLM
from pydantic import BaseModel

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessageForTraining,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseFunctionWebSearch,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseReasoningItemForTraining,
)
from responses_api_agents.nooa_agent.gym_llm import (
    GymResponsesLLM,
    InvalidPolicyOutputError,
    OutputTokenLimitExceeded,
    PolicyCallBudgetExceeded,
    RolloutLLMState,
    _finish_reason,
    _responses_input,
    _responses_tool_schema,
)


class FakeContent:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode()

    async def read(self) -> bytes:
        return self._payload


class FakeHTTPResponse:
    ok = True
    status = 200

    def __init__(self, payload: dict, cookies: SimpleCookie | None = None) -> None:
        self.content = FakeContent(payload)
        self.cookies = cookies or SimpleCookie()

    async def read(self) -> bytes:
        return await self.content.read()


class StructuredAnswer(BaseModel):
    verdict: str


def weather(city: str) -> str:
    """Get weather for a city."""

    return city


def model_response(*outputs: object, response_id: str = "resp-1") -> dict:
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="policy",
        object="response",
        output=list(outputs),
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump(mode="json")


def make_llm(
    payload: dict,
    *,
    max_policy_calls: int | None = 2,
    sampling_overrides: dict | None = None,
    context_window: int | None = None,
    media_inputs: tuple[str, ...] = (),
) -> tuple[GymResponsesLLM, MagicMock, RolloutLLMState]:
    server_client = MagicMock()
    server_client.post = AsyncMock(return_value=FakeHTTPResponse(payload))
    state = RolloutLLMState(max_policy_calls=max_policy_calls)
    llm = GymResponsesLLM(
        server_client=server_client,
        model_server_name="policy_model",
        model_url_path="/ng-rollout/rollout-1/v1/responses",
        state=state,
        cookies={},
        sampling_overrides=sampling_overrides,
        context_window=context_window,
        media_inputs=media_inputs,
    )
    return llm, server_client, state


@pytest.mark.asyncio
async def test_response_progress_fires_after_capture_and_cookies_before_output_validation() -> None:
    raw = model_response()
    raw["incomplete_details"] = {"reason": "max_output_tokens"}
    llm, client, state = make_llm(raw)
    client.post.return_value.cookies = SimpleCookie("model_state=updated")
    captured = []

    def capture() -> None:
        captured.append((state.calls[-1].response.model_dump(mode="json"), dict(llm._cookies)))

    llm._on_response = capture
    with pytest.raises(OutputTokenLimitExceeded):
        await llm.acall([{"role": "user", "content": "task"}])
    assert len(captured) == 1
    assert captured[0][0]["id"] == "resp-1"
    assert captured[0][0]["incomplete_details"]["reason"] == "max_output_tokens"
    assert captured[0][1] == {"model_state": "updated"}
    client.post.side_effect = ConnectionError("unavailable")
    with pytest.raises(ConnectionError):
        await llm.acall([{"role": "user", "content": "task"}])
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_context_planning_reserves_the_effective_gym_reply_cap() -> None:
    llm, client, _ = make_llm(model_response(), context_window=262144, sampling_overrides={"max_output_tokens": 32768})
    overrides = {"max_tokens": 128, "extra_body": {"max_completion_tokens": 64, "custom": True}}
    limits = llm.get_context_limits(overrides, fallback_reserve=4096)
    assert llm.context_window == limits.context_window == 262144
    assert limits.reserved_output_tokens == 32768
    assert limits.usable_input_tokens == 229376
    assert not limits.reserve_is_fallback
    assert overrides == {"max_tokens": 128, "extra_body": {"max_completion_tokens": 64, "custom": True}}
    await llm.acall([{"role": "user", "content": "hello"}], max_tokens=128)
    assert client.post.await_args.kwargs["json"].max_output_tokens == limits.reserved_output_tokens


# What NOOA's show() puts in a message: portable Chat-style blocks, not Responses blocks.
requires_nooa_media = pytest.mark.skipif(
    not hasattr(UnifiedLLM, "_prepare_media_messages"), reason="the pinned NOOA predates explicit media inputs"
)
SHOWN_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=", "format": "image/png"}}


@requires_nooa_media
@pytest.mark.asyncio
async def test_media_the_model_is_not_set_up_for_is_replaced_with_nooas_note() -> None:
    llm, client, _ = make_llm(model_response())
    await llm.acall(
        [
            {"role": "user", "content": [{"type": "text", "text": "Output:"}, SHOWN_IMAGE]},
            {"role": "tool", "tool_call_id": "call-1", "content": [{"type": "text", "text": "ran"}, SHOWN_IMAGE]},
        ]
    )
    request = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)
    user, tool = request["input"]
    assert user["content"][0] == {"type": "input_text", "text": "Output:"}
    assert user["content"][1]["type"] == "input_text" and "image was removed" in user["content"][1]["text"]
    assert tool["output"][0] == {"type": "input_text", "text": "ran"}
    assert "image was removed" in tool["output"][1]["text"]
    assert "data:image/png" not in json.dumps(request)


@requires_nooa_media
@pytest.mark.asyncio
async def test_enabled_images_reach_the_model_as_responses_input_images() -> None:
    llm, client, _ = make_llm(model_response(), media_inputs=("image",))
    await llm.acall([{"role": "user", "content": [{"type": "text", "text": "Output:"}, SHOWN_IMAGE]}])
    (user,) = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)["input"]
    assert user["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,iVBORw0KGgo=",
        "detail": "auto",
    }


def test_context_planning_without_gym_reply_override_keeps_nooa_reserve_semantics() -> None:
    llm, _, _ = make_llm(model_response(), context_window=262144)
    explicit = llm.get_context_limits({"max_tokens": 128}, fallback_reserve=4096)
    assert explicit.reserved_output_tokens == 128 and not explicit.reserve_is_fallback
    fallback = llm.get_context_limits(fallback_reserve=4096)
    assert fallback.reserved_output_tokens == 4096 and fallback.reserve_is_fallback


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (
            {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
            True,
        ),
        (
            {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            False,
        ),
        (
            {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": True,
            },
            False,
        ),
        (
            {
                "type": "object",
                "properties": {"code": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["code"],
                "additionalProperties": False,
            },
            False,
        ),
    ],
)
def test_responses_tool_schema_enables_strict_mode_only_for_closed_required_schemas(
    schema: dict, expected: bool
) -> None:
    tool = MagicMock()
    tool.name = "execute_python"
    tool.description = "Execute Python"
    tool.get_parameter_schema.return_value = schema

    assert _responses_tool_schema(tool)["strict"] is expected


@pytest.mark.parametrize(
    ("incomplete_details", "expected"),
    [
        (None, "stop"),
        ({"reason": "max_output_tokens"}, "length"),
        ({"reason": "content_filter"}, "error"),
    ],
)
def test_finish_reason_preserves_incomplete_response_cause(incomplete_details: dict | None, expected: str) -> None:
    payload = model_response()
    payload["incomplete_details"] = incomplete_details

    assert _finish_reason(NeMoGymResponse.model_validate(payload)) == expected


@pytest.mark.asyncio
async def test_routes_messages_tools_and_sampling_to_gym() -> None:
    output = NeMoGymResponseOutputMessageForTraining(
        id="msg-1",
        content=[NeMoGymResponseOutputText(annotations=[], text="Cold", logprobs=[])],
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.2],
        routed_experts=[[[0, 1]]],
    )
    llm, client, state = make_llm(model_response(output))

    result = await llm.acall(
        [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Weather?"}],
        tools=[Tool(name="weather", description="Get weather", callable=weather)],
        temperature=0.3,
        max_tokens=128,
    )

    request = client.post.await_args.kwargs
    assert request["server_name"] == "policy_model"
    assert request["url_path"] == "/ng-rollout/rollout-1/v1/responses"
    assert request["json"].instructions == "Be concise."
    assert request["json"].temperature == 0.3
    assert request["json"].max_output_tokens == 128
    assert request["json"].tools[0]["name"] == "weather"
    assert result.content == "Cold"
    assert state.model_calls[0].response_id == "resp-1"
    assert state.model_calls[0].model_ref is not None
    assert state.model_calls[0].model_ref.name == "policy_model"
    assert state.calls[0].request == request["json"]
    assert state.calls[0].request is not request["json"]


@pytest.mark.asyncio
async def test_replays_nooa_history_without_injecting_prior_response_metadata() -> None:
    output = NeMoGymResponseOutputMessageForTraining(
        id="msg-1",
        content=[NeMoGymResponseOutputText(annotations=[], text="Cold", logprobs=[])],
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.2],
    )
    llm, client, _ = make_llm(model_response(output))
    await llm.acall([{"role": "user", "content": "Weather?"}])

    await llm.acall([{"role": "assistant", "content": "Cold"}])

    request = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)
    assert request["input"] == [{"type": "message", "role": "assistant", "content": "Cold"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, {"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 128}),
        ({"temperature": 0}, {"temperature": 0, "top_p": 0.9, "max_output_tokens": 128}),
        (
            {"temperature": 0, "top_p": 0.8, "max_output_tokens": 64},
            {"temperature": 0, "top_p": 0.8, "max_output_tokens": 64},
        ),
    ],
)
async def test_row_sampling_overrides_nooa_call_settings_on_every_call(overrides: dict, expected: dict) -> None:
    supplied = dict(overrides)
    llm, client, state = make_llm(model_response(), sampling_overrides=supplied)
    supplied.clear()
    for _ in range(2):
        await llm.acall(
            [{"role": "user", "content": "question"}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=128,
        )
        body = client.post.await_args.kwargs["json"]
        assert body.model_dump(include=set(expected)) == expected
        assert state.calls[-1].request == body


@pytest.mark.asyncio
async def test_preserves_function_call_token_metadata() -> None:
    output = NeMoGymResponseFunctionToolCallForTraining(
        id="fc-1",
        call_id="call-1",
        name="weather",
        arguments='{"city":"Paris"}',
        prompt_token_ids=[10],
        generation_token_ids=[11, 12],
        generation_log_probs=[-0.1, -0.2],
    )
    llm, _, _ = make_llm(model_response(output))

    result = await llm.acall([{"role": "user", "content": "Weather?"}])

    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].name == "weather"
    assert result.raw_response.output[0].generation_token_ids == [11, 12]
    replayed, _ = _responses_input([result], replay_scope=result.replay_scope)
    assert replayed[0]["generation_token_ids"] == [11, 12]


def mixed_model_response() -> dict:
    outputs = []
    for index, city in enumerate(("Paris", "Oslo"), start=1):
        metadata = {
            "prompt_token_ids": [10 + index],
            "generation_token_ids": [20 + index, 30 + index],
            "generation_log_probs": [-0.1, -0.2],
            "routed_experts": [[[0, 1]], [[1, 2]]],
        }
        outputs.extend(
            [
                NeMoGymResponseReasoningItemForTraining(
                    id=f"reasoning-{index}",
                    summary=[{"type": "summary_text", "text": f"Check {city}."}],
                    content=[{"type": "reasoning_text", "text": f"Need the {city} forecast."}],
                    encrypted_content=f"encrypted-{index}",
                    **metadata,
                ),
                NeMoGymResponseOutputMessageForTraining(
                    id=f"message-{index}",
                    phase="commentary",
                    content=[NeMoGymResponseOutputText(annotations=[], text=f"Checking {city}.")],
                    **metadata,
                ),
                NeMoGymResponseFunctionToolCallForTraining(
                    id=f"function-{index}",
                    call_id=f"call-{index}",
                    name="weather",
                    arguments=json.dumps({"city": city}),
                    **metadata,
                ),
            ]
        )
    return model_response(*outputs)


@pytest.mark.asyncio
@pytest.mark.parametrize("history", ["live", "snapshot", "sqlite"])
async def test_mixed_assistant_history_replays_complete_ordered_outputs(history: str) -> None:
    payload = mixed_model_response()
    llm, client, state = make_llm(payload)
    question = {"role": "user", "content": "Compare the weather in Paris and Oslo."}
    result = await llm.acall([question])

    if history == "snapshot":
        blob, allowlist = serialize(result)
        assert "raw_response" not in json.dumps(blob)
        result = deserialize(json.loads(json.dumps(blob)), allowlist)
    elif history == "sqlite":
        with SQLiteStorageManager(":memory:") as storage:
            storage.event_backend.store("assistant-turn", result)
            result = storage.event_backend.get("assistant-turn")

    assert isinstance(result, LLMResponse)
    if history != "live":
        assert result.raw_response is None
        llm = GymResponsesLLM(
            server_client=client,
            model_server_name="policy_model",
            model_url_path="/ng-rollout/restored/v1/responses",
            state=state,
            cookies={},
        )
    await llm.acall(
        [
            question,
            result,
            {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"},
            {"role": "tool", "tool_call_id": "call-2", "content": "Snowy"},
        ]
    )

    request = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)
    expected_output = NeMoGymResponse.model_validate(payload).model_dump(mode="json", exclude_none=True)["output"]
    assert request["input"][1:-2] == expected_output
    assert request["input"][-2:] == [
        {"type": "function_call_output", "call_id": "call-1", "output": "Sunny"},
        {"type": "function_call_output", "call_id": "call-2", "output": "Snowy"},
    ]
    assert result.content == "Checking Paris.Checking Oslo."
    assert result.reasoning == "Need the Paris forecast.\nNeed the Oslo forecast."
    assert [part.kind for part in result.parts] == ["reasoning", "text", "tool_call"] * 2
    assert result.finish_reason == "tool_calls"
    assert state.gaps == []


@pytest.mark.asyncio
async def test_editing_assistant_text_discards_stale_provider_replay_metadata() -> None:
    llm, client, _ = make_llm(mixed_model_response())
    result = await llm.acall([{"role": "user", "content": "Weather?"}])
    edited = result.replace_text("Updated weather request.")

    await llm.acall([edited])

    request = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)
    replayed = request["input"]
    text = [item["content"] for item in replayed if item["type"] == "message"]
    assert text == ["Updated weather request.", "Need the Paris forecast.", "Need the Oslo forecast."]
    assert [item["type"] for item in replayed] == ["message", "message", "function_call", "message", "function_call"]
    assert [item["call_id"] for item in replayed if item["type"] == "function_call"] == ["call-1", "call-2"]
    wire = json.dumps(replayed)
    assert "encrypted" not in wire
    assert "token_ids" not in wire
    assert "generation_log_probs" not in wire
    assert "routed_experts" not in wire
    assert "Checking" not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("content", ["Edited answer.", ""])
async def test_formatter_history_edits_preserve_public_reasoning_without_provider_metadata(
    with_tools: bool, content: str
) -> None:
    llm, client, state = make_llm(mixed_model_response())
    original = await llm.acall([{"role": "user", "content": "Weather?"}])
    original_snapshot = original.model_dump(mode="json")
    original_response = state.calls[0].response.model_dump(mode="json")
    edited_call = ToolCallInfo(id="call-1", name="weather", arguments='{"city":"Berlin"}')
    rendered = OpenAIProviderFormatter().format(
        [
            RenderedMessage(
                role=Role.ASSISTANT,
                content=content,
                reasoning="Edited visible reasoning.",
                tool_calls=(edited_call,) if with_tools else (),
                replay_message=original,
            )
        ]
    )
    assert isinstance(rendered[0], dict)
    rendered_snapshot = deepcopy(rendered)

    await llm.acall(rendered)

    request = client.post.await_args.kwargs["json"]
    expected = [{"type": "message", "role": "assistant", "content": "Edited visible reasoning."}]
    if content:
        expected.append({"type": "message", "role": "assistant", "content": content})
    if with_tools:
        expected.append(
            {"type": "function_call", "call_id": "call-1", "name": "weather", "arguments": '{"city":"Berlin"}'}
        )
    assert request.model_dump(mode="json", exclude_none=True)["input"] == expected
    assert state.calls[-1].request == request
    assert original.model_dump(mode="json") == original_snapshot
    assert state.calls[0].response.model_dump(mode="json") == original_response
    assert rendered == rendered_snapshot
    assert [gap.code for gap in state.gaps] == ["public_assistant_metadata_unavailable"]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tools", [False, True])
async def test_unscoped_history_replays_public_fields_instead_of_live_raw_response(with_tools: bool) -> None:
    raw = NeMoGymResponse.model_validate(mixed_model_response())
    raw_snapshot = raw.model_dump(mode="json")
    replacement = LLMResponse(
        raw_response=raw,
        content="Replacement answer.",
        reasoning="Replacement reasoning.",
        tool_calls=[ToolCall(id="edited-call", name="weather", arguments='{"city":"Berlin"}')] if with_tools else [],
    )
    assert replacement.replay_scope is None
    llm, client, state = make_llm(model_response())

    await llm.acall([replacement])

    expected = [
        {"type": "message", "role": "assistant", "content": "Replacement reasoning."},
        {"type": "message", "role": "assistant", "content": "Replacement answer."},
    ]
    if with_tools:
        expected.append(
            {"type": "function_call", "call_id": "edited-call", "name": "weather", "arguments": '{"city":"Berlin"}'}
        )
    request = client.post.await_args.kwargs["json"]
    assert request.model_dump(mode="json", exclude_none=True)["input"] == expected
    assert state.calls[-1].request == request
    assert raw.model_dump(mode="json") == raw_snapshot
    assert [gap.code for gap in state.gaps] == ["foreign_turn_projected_portable"]


@pytest.mark.asyncio
async def test_foreign_ordered_parts_replay_only_public_text_and_tool_calls() -> None:
    foreign = LLMResponse(
        parts=(
            AssistantText(text="Checking.", native={"encrypted_content": "foreign-secret"}),
            ToolCall(id="call-9", name="weather", arguments='{"city":"Oslo"}', native={"token_ids": [9]}),
            AssistantReasoning(text="Need temperature too.", native={"encrypted_content": "foreign-secret"}),
            AssistantText(text="One moment."),
        ),
        replay_scope="foreign-provider",
        finish_reason="tool_calls",
    )
    llm, client, state = make_llm(model_response())

    await llm.acall([foreign])

    request = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)
    assert request["input"] == [
        {"type": "message", "role": "assistant", "content": "Checking."},
        {"type": "function_call", "call_id": "call-9", "name": "weather", "arguments": '{"city":"Oslo"}'},
        {"type": "message", "role": "assistant", "content": "Need temperature too."},
        {"type": "message", "role": "assistant", "content": "One moment."},
    ]
    assert [gap.code for gap in state.gaps] == ["foreign_turn_projected_portable"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_server_name", "model"),
    [("other_model_server", "gym-policy"), ("policy_model", "other-policy")],
)
async def test_different_model_replays_public_parts_without_provider_metadata(
    model_server_name: str, model: str
) -> None:
    llm, client, state = make_llm(mixed_model_response())
    result = await llm.acall([{"role": "user", "content": "Weather?"}])
    other_llm = GymResponsesLLM(
        server_client=client,
        model_server_name=model_server_name,
        model_url_path="/ng-rollout/other/v1/responses",
        model=model,
        state=state,
        cookies={},
    )

    await other_llm.acall([result])

    replayed = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)["input"]
    assert [item["type"] for item in replayed] == ["message", "message", "function_call"] * 2
    assert [item["content"] for item in replayed if item["type"] == "message"] == [
        "Need the Paris forecast.",
        "Checking Paris.",
        "Need the Oslo forecast.",
        "Checking Oslo.",
    ]
    assert [item["call_id"] for item in replayed if item["type"] == "function_call"] == ["call-1", "call-2"]
    wire = json.dumps(replayed)
    assert "encrypted" not in wire
    assert "token_ids" not in wire
    assert "generation_log_probs" not in wire
    assert "routed_experts" not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "reasoning"),
    [
        (
            NeMoGymResponseReasoningItem(
                id="summary-only",
                summary=[{"type": "summary_text", "text": "Compare the forecasts."}],
                encrypted_content="encrypted-summary",
            ),
            "Compare the forecasts.",
        ),
        (
            NeMoGymResponseReasoningItem(id="encrypted-only", summary=[], encrypted_content="encrypted-reasoning"),
            None,
        ),
        (
            NeMoGymResponseOutputMessage(
                id="refusal", content=[{"type": "refusal", "refusal": "Cannot provide that."}]
            ),
            None,
        ),
        (
            NeMoGymResponseFunctionWebSearch(
                id="web-search",
                type="web_search_call",
                status="completed",
                action={"type": "search", "query": "Paris weather"},
            ),
            None,
        ),
    ],
    ids=["reasoning-summary", "encrypted-reasoning", "refusal", "hosted-tool"],
)
async def test_nontext_outputs_survive_snapshot_replay(output: BaseModel, reasoning: str | None) -> None:
    payload = model_response(output)
    llm, client, state = make_llm(payload)
    result = await llm.acall([{"role": "user", "content": "Continue."}])
    blob, allowlist = serialize(result)
    restored = deserialize(json.loads(json.dumps(blob)), allowlist)

    await llm.acall([restored])

    replayed = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)["input"]
    assert replayed == [output.model_dump(mode="json", exclude_none=True)]
    assert restored.reasoning == reasoning
    assert state.gaps == []


def test_cache_boundary_is_never_a_model_input() -> None:
    replayed, instructions = _responses_input(
        [{"role": "user", "content": "Weather?"}, CacheBoundary()], replay_scope="gym-test"
    )

    assert replayed == [{"role": "user", "content": "Weather?"}]
    assert instructions is None


def test_foreign_llm_response_projects_portable_and_records_gap() -> None:
    foreign = LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id="call-9", name="weather", arguments='{"city":"Oslo"}')],
        finish_reason="tool_calls",
    )
    gaps: list = []

    replayed, _ = _responses_input([foreign], gaps=gaps, replay_scope="gym-test")

    assert replayed == [
        {
            "type": "function_call",
            "call_id": "call-9",
            "name": "weather",
            "arguments": '{"city":"Oslo"}',
        }
    ]
    assert [gap.code for gap in gaps] == ["foreign_turn_projected_portable"]


@pytest.mark.asyncio
async def test_structured_output_schema_and_parsing() -> None:
    output = NeMoGymResponseOutputMessageForTraining(
        id="msg-1",
        content=[NeMoGymResponseOutputText(annotations=[], text=' {\n  "verdict": "positive"\n}\n')],
        prompt_token_ids=[1],
        generation_token_ids=[2],
        generation_log_probs=[-0.1],
    )
    llm, client, _ = make_llm(model_response(output))

    result = await llm.acall([{"role": "user", "content": "Classify"}], output_model=StructuredAnswer)

    assert result.parsed == StructuredAnswer(verdict="positive")
    assert json.loads(result.content) == {"verdict": "positive"}
    assert result.content == output.content[0].text
    assert client.post.await_args.kwargs["json"].text["format"]["name"] == "StructuredAnswer"

    blob, allowlist = serialize(result)
    restored = deserialize(json.loads(json.dumps(blob)), allowlist)
    assert restored.parsed is None
    assert restored.content == result.content
    assert StructuredAnswer.model_validate_json(restored.content) == result.parsed

    await llm.acall([restored])

    replayed = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)["input"]
    assert replayed == [output.model_dump(mode="json", exclude_none=True)]


@pytest.mark.asyncio
async def test_enforces_total_policy_call_budget() -> None:
    output = NeMoGymResponseOutputMessageForTraining(
        id="msg-1",
        content=[NeMoGymResponseOutputText(annotations=[], text="done")],
        prompt_token_ids=[1],
        generation_token_ids=[2],
        generation_log_probs=[-0.1],
    )
    llm, client, _ = make_llm(model_response(output), max_policy_calls=1)
    await llm.acall([{"role": "user", "content": "first"}])

    with pytest.raises(PolicyCallBudgetExceeded, match="exhausted"):
        await llm.acall([{"role": "user", "content": "second"}])

    client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_unlimited_policy_calls_preserve_request_accounting() -> None:
    llm, client, state = make_llm(model_response(), max_policy_calls=None)
    for _ in range(101):
        await llm.acall([{"role": "user", "content": "continue"}])

    assert client.post.await_count == 101
    assert state.used == 101
    assert len(state.calls) == 101
    assert llm.calls == 101
    assert state.fatal_error is None


def test_rejects_synchronous_policy_calls() -> None:
    llm, _, _ = make_llm(model_response())

    with pytest.raises(RuntimeError, match="async"):
        llm.call([])


@pytest.mark.asyncio
async def test_transport_error_clears_after_a_later_success() -> None:
    llm, client, state = make_llm(model_response())
    failure = ConnectionError("model disconnected")
    client.post.side_effect = [failure, FakeHTTPResponse(model_response())]
    with pytest.raises(ConnectionError, match="disconnected"):
        await llm.acall([{"role": "user", "content": "task"}])
    await llm.acall([{"role": "user", "content": "retry"}])
    assert state.fatal_error is None


@pytest.mark.asyncio
@pytest.mark.parametrize("main_failed", [False, True])
@pytest.mark.parametrize("summary_failed", [False, True])
async def test_summary_call_preserves_main_transport_state(main_failed: bool, summary_failed: bool) -> None:
    from nooa.agents.summarization import _in_summary_fork

    llm, client, state = make_llm(model_response())
    main_error = ConnectionError("main model failure")
    summary_error = ConnectionError("optional summary failure")
    client.post.side_effect = [
        main_error if main_failed else FakeHTTPResponse(model_response()),
        summary_error if summary_failed else FakeHTTPResponse(model_response()),
    ]
    if main_failed:
        with pytest.raises(ConnectionError) as caught:
            await llm.acall([{"role": "user", "content": "task"}])
        assert caught.value is main_error
    else:
        await llm.acall([{"role": "user", "content": "task"}])

    token = _in_summary_fork.set(True)
    try:
        if summary_failed:
            with pytest.raises(ConnectionError) as caught:
                await llm.acall([{"role": "user", "content": "summary"}])
            assert caught.value is summary_error
        else:
            await llm.acall([{"role": "user", "content": "summary"}])
    finally:
        _in_summary_fork.reset(token)

    assert state.fatal_error is (main_error if main_failed else None)
    assert state.used == len(state.calls) == 2
    assert (state.calls[0].response is None) is main_failed
    assert (state.calls[1].response is None) is summary_failed


@pytest.mark.asyncio
@pytest.mark.parametrize("main_failed", [False, True])
@pytest.mark.parametrize("invalid_cap", [False, True])
async def test_summary_preparation_failure_preserves_main_state(main_failed: bool, invalid_cap: bool) -> None:
    from nooa.agents.summarization import _in_summary_fork

    llm, client, state = make_llm(model_response())
    main_error = ConnectionError("main model failure")
    if main_failed:
        client.post.side_effect = main_error
        with pytest.raises(ConnectionError):
            await llm.acall([{"role": "user", "content": "task"}])
    else:
        await llm.acall([{"role": "user", "content": "task"}])

    token = _in_summary_fork.set(True)
    try:
        with pytest.raises((ValueError, KeyError)):
            if invalid_cap:
                await llm.acall([{"role": "user", "content": "summary"}], max_output_tokens=0)
            else:
                await llm.acall([{"role": "tool", "content": "missing call id"}])
    finally:
        _in_summary_fork.reset(token)

    assert state.fatal_error is (main_error if main_failed else None)
    assert state.used == llm.calls == len(state.calls) == client.post.await_count == 1


def output_limit_response(*outputs: object, response_id: str = "partial-response") -> dict:
    return model_response(*outputs, response_id=response_id) | {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "metadata": {"upstream_trace": "partial-generation"},
        "usage": {
            "input_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 3,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [None, '{"code":"print(7)"}', '{"code":'])
async def test_output_limit_records_exact_response_before_interpreting_output(arguments: str | None) -> None:
    outputs = [
        {"type": "reasoning", "id": "reasoning-1", "summary": [{"type": "summary_text", "text": "partial plan"}]},
        NeMoGymResponseOutputMessageForTraining(
            id="partial-message",
            content=[NeMoGymResponseOutputText(annotations=[], text="unfinished JSON", logprobs=[])],
            prompt_token_ids=[1, 2],
            generation_token_ids=[3],
            generation_log_probs=[-0.2],
        ),
    ]
    if arguments is not None:
        outputs.append(
            NeMoGymResponseFunctionToolCallForTraining(
                id="partial-function",
                call_id="partial-call",
                name="execute_python",
                arguments=arguments,
                prompt_token_ids=[1, 2],
                generation_token_ids=[4],
                generation_log_probs=[-0.3],
            )
        )
    payload = output_limit_response(*outputs)
    llm, client, state = make_llm(payload)

    with pytest.raises(OutputTokenLimitExceeded, match="max_output_tokens"):
        await llm.acall([{"role": "user", "content": "task"}], output_model=StructuredAnswer)

    client.post.assert_awaited_once()
    assert state.used == 1
    assert state.fatal_error is None
    assert len(state.calls) == 1
    assert state.calls[0].response.model_dump(mode="json") == payload
    assert state.model_calls[0].response_id == "partial-response"


@pytest.mark.asyncio
async def test_successful_response_after_output_limit_recovers_normally() -> None:
    final = NeMoGymResponseOutputMessageForTraining(
        id="final",
        content=[NeMoGymResponseOutputText(annotations=[], text="done", logprobs=[])],
        prompt_token_ids=[1],
        generation_token_ids=[2],
        generation_log_probs=[-0.1],
    )
    llm, client, state = make_llm(output_limit_response())
    client.post.side_effect = [FakeHTTPResponse(output_limit_response()), FakeHTTPResponse(model_response(final))]
    with pytest.raises(OutputTokenLimitExceeded):
        await llm.acall([{"role": "user", "content": "task"}])

    result = await llm.acall([{"role": "user", "content": "retry"}])

    assert result.content == "done"
    assert result.finish_reason == "stop"
    assert state.fatal_error is None
    assert state.used == len(state.calls) == 2
    assert state.calls[0].response.incomplete_details.reason == "max_output_tokens"
    assert state.calls[1].response.incomplete_details is None


class RejectedHTTPResponse(FakeHTTPResponse):
    def __init__(self, status: int, payload: dict | None = None) -> None:
        super().__init__(payload or {"error": {"message": "controlled provider rejection"}})
        self.status = status
        self.ok = False
        self.error = ClientResponseError(
            request_info=SimpleNamespace(
                url="http://model/v1/responses", real_url="http://model/v1/responses", method="POST", headers={}
            ),
            history=(),
            status=status,
            message="provider rejected request",
        )

    def raise_for_status(self) -> None:
        raise self.error


@pytest.mark.asyncio
async def test_rate_limit_retries_preserve_native_request_and_all_http_captures(tmp_path) -> None:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    from nooa.runtime.hooks import hooks_scope

    from nemo_gym.base_responses_api_model import CaptureStore, ModelCallCaptureConfig, install_model_call_capture
    from nemo_gym.observability.model_calls import read_model_call_records
    from responses_api_agents.nooa_agent.observability import GymTraceHooks
    from responses_api_agents.nooa_agent.tests.test_observability import NativeHistoryAgent

    raw = model_response(
        NeMoGymResponseFunctionToolCallForTraining(
            id="fc-final",
            call_id="call-final",
            name="python_cell",
            arguments=json.dumps({"code": "return_result('done')"}),
            prompt_token_ids=[1, 2],
            generation_token_ids=[3],
            generation_log_probs=[-0.1],
        )
    )
    raw["usage"] = output_limit_response()["usage"]
    rejected = {"error": {"message": "controlled rate limit"}}
    app = FastAPI()
    attempts = 0

    @app.post("/v1/responses")
    async def respond(body: dict) -> JSONResponse:
        nonlocal attempts
        attempts += 1
        return JSONResponse(rejected if attempts < 3 else raw, status_code=429 if attempts < 3 else 200)

    install_model_call_capture(
        app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path),
        model_server_name="policy_model",
    )
    trace = GymTraceHooks()
    llm, client, state = make_llm(raw, max_policy_calls=1)
    llm._on_call = trace.on_model_call
    llm._on_response = MagicMock()
    with TestClient(app) as transport:

        async def post(*, json, url_path, cookies, headers, **kwargs):
            response = transport.post(url_path, json=json.model_dump(exclude_unset=True), headers=headers)
            if response.status_code == 429:
                return RejectedHTTPResponse(429, response.json())
            return FakeHTTPResponse(response.json(), SimpleCookie("model_state=updated"))

        client.post.side_effect = post
        agent = NativeHistoryAgent(llm=llm)
        try:
            with hooks_scope(trace):
                assert await agent.answer("test rate limit retry") == "done"
            assert not list(agent.events.query(type="Error"))
            assert any(event.metadata.get("prefill") for event in agent.events.query(type="ToolCallEvent"))
        finally:
            await agent.aclose()

    captured = read_model_call_records(CaptureStore(tmp_path), "rollout-1")
    assert [record.status_code for record in captured] == [429, 429, 200]
    assert [record.response for record in captured] == [rejected, rejected, raw]
    assert len({record.client_session_id for record in captured}) == 1
    assert captured[0].client_session_id == state.calls[0].invocation_id
    assert all(record.request == captured[0].request for record in captured)
    bodies = [attempt.kwargs["json"] for attempt in client.post.await_args_list]
    assert len(bodies) == 3 and all(body is bodies[0] for body in bodies)
    assert captured[0].request == bodies[0].model_dump(mode="json", exclude_unset=True)
    assert state.used == llm.calls == len(state.calls) == 1
    assert state.calls[0].response.model_dump(mode="json") == raw
    assert state.fatal_error is None
    assert llm._cookies == {"model_state": "updated"}
    llm._on_response.assert_called_once()
    with pytest.raises(PolicyCallBudgetExceeded):
        await llm.acall([{"role": "user", "content": "one more"}])
    assert client.post.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("raised_by_post", [False, True])
async def test_rate_limit_exhaustion_keeps_original_fatal_error_and_bounded_attempts(
    monkeypatch, raised_by_post: bool
) -> None:
    llm, client, state = make_llm(model_response())
    failures = [RejectedHTTPResponse(429) for _ in range(3)]
    client.post.side_effect = [response.error if raised_by_post else response for response in failures]
    sleep = AsyncMock()
    monkeypatch.setattr("responses_api_agents.nooa_agent.gym_llm.asyncio.sleep", sleep)
    with pytest.raises(ClientResponseError) as caught:
        await llm.acall([{"role": "user", "content": "task"}])
    assert caught.value is state.fatal_error is failures[-1].error
    if not raised_by_post:
        assert json.loads(caught.value.response_content) == {"error": {"message": "controlled provider rejection"}}
        assert "controlled provider rejection" in caught.value.message
    assert client.post.await_count == 3
    assert [call.args for call in sleep.await_args_list] == [(1,), (2,)]
    assert state.used == llm.calls == len(state.calls) == 1
    assert state.calls[0].response is None


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", [False, True])
async def test_recovered_rate_limit_preserves_prior_evidence_and_summary_fatal_state(
    monkeypatch, summary: bool
) -> None:
    from nooa.agents.summarization import _in_summary_fork

    llm, client, state = make_llm(model_response(), max_policy_calls=3)
    earlier = FakeHTTPResponse(model_response(response_id="earlier") | {"usage": output_limit_response()["usage"]})
    failure = ConnectionError("earlier failure")
    client.post.side_effect = [earlier, failure, RejectedHTTPResponse(429).error, FakeHTTPResponse(model_response())]
    monkeypatch.setattr("responses_api_agents.nooa_agent.gym_llm.asyncio.sleep", AsyncMock())
    await llm.acall([{"role": "user", "content": "first"}])
    with pytest.raises(ConnectionError):
        await llm.acall([{"role": "user", "content": "second"}])
    token = _in_summary_fork.set(summary)
    try:
        await llm.acall([{"role": "user", "content": "recover"}])
    finally:
        _in_summary_fork.reset(token)
    assert state.fatal_error is (failure if summary else None)
    assert state.used == len(state.calls) == 3
    assert state.calls[0].response.usage.total_tokens == 3
    assert state.calls[1].response is None
    assert state.calls[2].response.id == "resp-1"
    assert [call.response_id for call in state.model_calls] == ["earlier", "resp-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["post", "backoff"])
async def test_rate_limit_retry_cancellation_propagates_through_runner(monkeypatch, cancel_at: str) -> None:
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from responses_api_agents.nooa_agent.runner import NOOARunRequest
    from responses_api_agents.nooa_agent.tests.test_runner import invoke_model_directly, make_runner

    runner, _ = make_runner()
    runner._invocation_adapter = invoke_model_directly
    cancelled = asyncio.CancelledError()
    runner._server_client.post = AsyncMock(side_effect=[RejectedHTTPResponse(429), cancelled])
    sleep = AsyncMock(side_effect=cancelled if cancel_at == "backoff" else None)
    monkeypatch.setattr("responses_api_agents.nooa_agent.gym_llm.asyncio.sleep", sleep)
    with pytest.raises(asyncio.CancelledError) as caught:
        await runner.run(
            NOOARunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
                model_url_path="/v1/responses",
            )
        )
    assert caught.value is cancelled
    assert caught.value.nooa_result.termination_reason is None
    assert runner._server_client.post.await_count == (1 if cancel_at == "backoff" else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("split_messages", [False, True])
async def test_structured_output_parses_exact_persisted_text(split_messages: bool) -> None:
    blocks = [NeMoGymResponseOutputText(annotations=[], text=text) for text in ('{"verdict":"pos', 'itive"}')]
    outputs = (
        [NeMoGymResponseOutputMessage(id=f"msg-{i}", content=[block]) for i, block in enumerate(blocks)]
        if split_messages
        else [NeMoGymResponseOutputMessage(id="msg", content=blocks)]
    )
    llm, client, state = make_llm(model_response(*outputs))
    result = await llm.acall([{"role": "user", "content": "Classify"}], output_model=StructuredAnswer)
    blob, allowlist = serialize(result)
    restored = deserialize(json.loads(json.dumps(blob)), allowlist)

    assert result.parsed == StructuredAnswer(verdict="positive")
    assert restored.parsed is None
    assert StructuredAnswer.model_validate_json(restored.content) == result.parsed
    await llm.acall([restored])
    assert client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)["input"] == [
        output.model_dump(mode="json", exclude_none=True) for output in outputs
    ]
    assert state.gaps == []


@pytest.mark.asyncio
@pytest.mark.parametrize("include_json", [False, True])
async def test_structured_output_refusal_preserves_evidence_and_fails(include_json: bool) -> None:
    blocks = [{"type": "refusal", "refusal": "Cannot answer."}]
    if include_json:
        blocks.append({"type": "output_text", "text": '{"verdict":"positive"}', "annotations": []})
    output = NeMoGymResponseOutputMessage(id="msg", content=blocks)
    llm, client, state = make_llm(model_response(output))
    with pytest.raises(InvalidPolicyOutputError, match="refused to produce StructuredAnswer"):
        await llm.acall([{"role": "user", "content": "Classify"}], output_model=StructuredAnswer)
    assert state.calls[0].response.output == [output]
    assert state.used == llm.calls == client.post.await_count == 1
    assert state.fatal_error is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("index", "field", "value"),
    [
        (0, "text", "Edited reasoning."),
        (1, "text", "Edited text."),
        (2, "id", "edited-call"),
        (2, "name", "edited_tool"),
        (2, "arguments", '{"city":"Berlin"}'),
    ],
)
async def test_archive_public_edits_invalidate_native_replay(index: int, field: str, value: str) -> None:
    llm, client, state = make_llm(mixed_model_response())
    original = await llm.acall([{"role": "user", "content": "Weather?"}])
    archive = original.model_dump(mode="json")
    archive["parts"][index][field] = value
    rebuilt = LLMResponse.model_validate(archive)
    assert rebuilt.parts[index].native is not None
    await llm.acall([rebuilt])

    items = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)["input"]
    expected = []
    for part in rebuilt.parts:
        if isinstance(part, ToolCall):
            expected.append(
                {"type": "function_call", "call_id": part.id, "name": part.name, "arguments": part.arguments}
            )
        elif part.text:
            expected.append({"type": "message", "role": "assistant", "content": part.text})
    assert items == expected
    assert [gap.code for gap in state.gaps] == ["foreign_turn_projected_portable"]
    assert state.calls[0].response.model_dump(mode="json") == mixed_model_response()


def test_native_replay_requires_explicit_non_null_scope() -> None:
    response = LLMResponse(
        parts=(
            AssistantText(
                text="answer",
                native={
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            ),
        ),
    )
    with pytest.raises(TypeError, match="replay_scope"):
        _responses_input([response])
    items, _ = _responses_input([response], replay_scope=None)
    assert items == [{"role": "assistant", "content": "answer"}]


@pytest.mark.asyncio
async def test_repeated_portable_history_deduplicates_provenance_neutral_gaps() -> None:
    llm, _, state = make_llm(model_response(), max_policy_calls=None)
    public = {"role": "assistant", "content": "An example or an edit."}
    foreign = LLMResponse(content="Another turn.")
    for _ in range(5):
        await llm.acall([public, foreign, public, foreign])
    assert [gap.code for gap in state.gaps] == [
        "public_assistant_metadata_unavailable",
        "foreign_turn_projected_portable",
    ]
    assert "caller-authored example" in state.gaps[0].detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "reasoning_content": {"invalid": "object"}},
        {"role": "assistant", "tool_calls": [{"function": {"name": "weather", "arguments": "{}"}}]},
        {"role": "tool", "content": "missing call id"},
        {"type": "not_a_response_item"},
    ],
)
async def test_history_preparation_failure_does_not_spend_policy_budget(message: dict) -> None:
    llm, client, state = make_llm(model_response(), max_policy_calls=1)
    with pytest.raises((ValueError, KeyError)) as caught:
        await llm.acall([message])
    assert state.fatal_error is caught.value
    assert state.used == llm.calls == 0
    assert state.calls == []
    client.post.assert_not_awaited()
    await llm.acall([{"role": "user", "content": "valid retry"}])
    assert state.used == llm.calls == 1
    assert state.fatal_error is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        {"type": "custom_tool_call", "id": "custom", "call_id": "call", "name": "custom", "input": "payload"},
        {
            "type": "computer_call",
            "id": "computer",
            "call_id": "call",
            "status": "completed",
            "action": {"type": "screenshot"},
            "pending_safety_checks": [],
        },
        {
            "type": "local_shell_call",
            "id": "local-shell",
            "call_id": "call",
            "status": "completed",
            "action": {"type": "exec", "command": ["pwd"], "env": {}},
        },
        {
            "type": "shell_call",
            "id": "shell",
            "call_id": "call",
            "status": "completed",
            "action": {"type": "exec", "commands": ["pwd"]},
        },
        {
            "type": "apply_patch_call",
            "id": "patch",
            "call_id": "call",
            "status": "completed",
            "operation": {"type": "create_file", "path": "a.py", "diff": "+pass"},
        },
    ],
)
async def test_unsupported_client_tool_calls_fail_after_capture(output: dict) -> None:
    llm, client, state = make_llm(model_response(output))
    captured = []
    llm._on_response = lambda: captured.append(state.calls[0].response)
    with pytest.raises(InvalidPolicyOutputError, match=output["type"]):
        await llm.acall([{"role": "user", "content": "Continue"}])
    assert captured == [state.calls[0].response]
    assert state.calls[0].response.output[0].model_dump(mode="json", exclude_none=True) == output
    assert state.used == llm.calls == client.post.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["Prompt", [{"type": "input_text", "text": "Prompt"}]])
@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_request_shaped_output_history_keeps_opaque_training_metadata(role: str, content: str | list) -> None:
    output = NeMoGymEasyInputMessageForTraining(
        role=role,
        content=content,
        prompt_token_ids=[1],
        generation_token_ids=[2],
        generation_log_probs=[-0.1],
    )
    llm, client, state = make_llm(model_response(output))
    result = await llm.acall([{"role": "user", "content": "Continue"}])
    assert result.content == ""
    blob, allowlist = serialize(result)
    restored = deserialize(json.loads(json.dumps(blob)), allowlist)
    await llm.acall([restored])
    assert client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)["input"] == [
        output.model_dump(mode="json", exclude_none=True)
    ]
    assert state.gaps == []


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens", "max_output_tokens"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("gym_cap", [None, 256])
async def test_reply_cap_alias_matches_planning_and_wire_request(key: str, nested: bool, gym_cap: int | None) -> None:
    params = {key: 128}
    if nested:
        params = {"extra_body": params}
    llm, client, _ = make_llm(model_response(), context_window=1024, sampling_overrides={"max_output_tokens": gym_cap})
    limits = llm.get_context_limits(params)
    await llm.acall([{"role": "user", "content": "hello"}], **params)
    body = client.post.await_args.kwargs["json"]
    assert body.max_output_tokens == limits.reserved_output_tokens == (gym_cap or 128)
    assert limits.usable_input_tokens == 1024 - body.max_output_tokens


@pytest.mark.asyncio
async def test_absent_reply_cap_does_not_send_planning_fallback() -> None:
    llm, client, _ = make_llm(model_response(), context_window=1024)
    assert llm.get_context_limits(fallback_reserve=256).reserved_output_tokens == 256
    await llm.acall([{"role": "user", "content": "hello"}])
    assert client.post.await_args.kwargs["json"].max_output_tokens is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"max_tokens": 512, "max_completion_tokens": 64},
        {"extra_body": {"max_tokens": 1024}},
        {"max_completion_tokens": 0},
        {"max_output_tokens": True},
    ],
)
async def test_invalid_reply_caps_fail_planning_and_call_before_model_transport(params: dict) -> None:
    llm, client, state = make_llm(model_response(), context_window=1024)
    with pytest.raises(ValueError):
        llm.get_context_limits(params)
    with pytest.raises(ValueError):
        await llm.acall([{"role": "user", "content": "hello"}], **params)
    client.post.assert_not_awaited()
    assert state.used == 0
