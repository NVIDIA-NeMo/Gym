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
from http.cookies import SimpleCookie
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa.unifiedllm import Tool
from pydantic import BaseModel

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from responses_api_agents.nooa_agent.gym_llm import (
    GymResponsesLLM,
    PolicyCallBudgetExceeded,
    RolloutLLMState,
    _finish_reason,
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
    payload: dict, *, max_policy_calls: int = 2, sampling_overrides: dict | None = None
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
    )
    return llm, server_client, state


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
    assert result.assistant_message["_batch"][0]["generation_token_ids"] == [11, 12]


@pytest.mark.asyncio
async def test_structured_output_schema_and_parsing() -> None:
    output = NeMoGymResponseOutputMessageForTraining(
        id="msg-1",
        content=[NeMoGymResponseOutputText(annotations=[], text='{"verdict":"positive"}')],
        prompt_token_ids=[1],
        generation_token_ids=[2],
        generation_log_probs=[-0.1],
    )
    llm, client, _ = make_llm(model_response(output))

    result = await llm.acall([{"role": "user", "content": "Classify"}], output_model=StructuredAnswer)

    assert result.content == StructuredAnswer(verdict="positive")
    assert client.post.await_args.kwargs["json"].text["format"]["name"] == "StructuredAnswer"


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


def test_rejects_synchronous_policy_calls() -> None:
    llm, _, _ = make_llm(model_response())

    with pytest.raises(RuntimeError, match="async"):
        llm.call([])
