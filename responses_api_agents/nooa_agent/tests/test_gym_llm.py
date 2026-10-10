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

import json
from copy import deepcopy
from http.cookies import SimpleCookie
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa.context_blocks.formatter import OpenAIProviderFormatter
from nooa.context_blocks.models import RenderedMessage, Role, ToolCallInfo
from nooa.llm_types import AssistantReasoning, AssistantText
from nooa.storage.serialization import deserialize, serialize
from nooa.storage.sqlite import SQLiteStorageManager
from nooa.unifiedllm import CacheBoundary, LLMResponse, Tool, ToolCall
from pydantic import BaseModel

from nemo_gym.openai_utils import (
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
    )
    return llm, server_client, state


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
    replayed, _ = _responses_input([result])
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
async def test_editing_assistant_text_discards_stale_native_replay_metadata() -> None:
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
async def test_formatter_history_edits_preserve_public_reasoning_without_native_metadata(
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
async def test_different_model_replays_public_parts_without_native_metadata(
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
    replayed, instructions = _responses_input([{"role": "user", "content": "Weather?"}, CacheBoundary()])

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

    replayed, _ = _responses_input([foreign], gaps=gaps)

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
async def test_transport_error_clears_after_a_later_main_success() -> None:
    llm, client, state = make_llm(model_response())
    failure = ConnectionError("model disconnected")
    client.post.side_effect = [failure, FakeHTTPResponse(model_response())]
    with pytest.raises(ConnectionError, match="disconnected"):
        await llm.acall([{"role": "user", "content": "task"}])
    assert state.fatal_error is failure
    await llm.acall([{"role": "user", "content": "retry"}])
    assert state.fatal_error is None
    assert len(state.calls) == 2
    assert state.calls[0].response is None
    assert state.calls[1].response.id == "resp-1"


@pytest.mark.asyncio
async def test_recovered_call_and_failed_summary_preserve_durable_ordered_history() -> None:
    from nooa.agents.summarization import _in_summary_fork

    payload = mixed_model_response()
    llm, client, state = make_llm(payload, max_policy_calls=4)
    client.post.side_effect = [
        ConnectionError("temporary model failure"),
        FakeHTTPResponse(payload),
        ConnectionError("optional summary failure"),
        FakeHTTPResponse(model_response(response_id="continued")),
    ]
    question = {"role": "user", "content": "Compare Paris and Oslo."}
    with pytest.raises(ConnectionError, match="temporary model failure"):
        await llm.acall([question])
    result = await llm.acall([question])
    assert state.fatal_error is None

    blob, allowlist = serialize(result)
    restored = deserialize(json.loads(json.dumps(blob)), allowlist)
    assert restored.raw_response is None
    token = _in_summary_fork.set(True)
    try:
        with pytest.raises(ConnectionError, match="optional summary failure"):
            await llm.acall([question, restored])
    finally:
        _in_summary_fork.reset(token)

    assert state.fatal_error is None
    await llm.acall([question, restored])
    request = client.post.await_args.kwargs["json"].model_dump(mode="json", exclude_none=True)
    expected = NeMoGymResponse.model_validate(payload).model_dump(mode="json", exclude_none=True)["output"]
    assert request["input"][1:] == expected
    assert [part.kind for part in restored.parts] == ["reasoning", "text", "tool_call"] * 2
    assert [call.response.id if call.response is not None else None for call in state.calls] == [
        None,
        "resp-1",
        None,
        "continued",
    ]
    assert state.used == 4
    assert state.gaps == []
