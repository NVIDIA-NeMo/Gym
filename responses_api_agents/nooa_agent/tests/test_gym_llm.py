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
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from nemo_gym.rollout_observability import ObservationGap
from responses_api_agents.nooa_agent.gym_llm import (
    GymResponsesLLM,
    InvalidPolicyOutputError,
    PolicyCallBudgetExceeded,
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
    max_steps: int = 2,
    observation_gaps: list[ObservationGap] | None = None,
    request_collector: list[NeMoGymResponseCreateParamsNonStreaming] | None = None,
) -> tuple[GymResponsesLLM, MagicMock, list[NeMoGymResponse]]:
    server_client = MagicMock()
    server_client.post = AsyncMock(return_value=FakeHTTPResponse(payload))
    collected: list[NeMoGymResponse] = []
    llm = GymResponsesLLM(
        server_client=server_client,
        model_server_name="policy_model",
        model_url_path="/ng-rollout/rollout-1/v1/responses",
        max_steps=max_steps,
        request_collector=request_collector if request_collector is not None else [],
        response_collector=collected,
        cookies={},
        observation_gaps=observation_gaps,
    )
    return llm, server_client, collected


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
    requests: list[NeMoGymResponseCreateParamsNonStreaming] = []
    llm, client, collected = make_llm(model_response(output), request_collector=requests)

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
    assert requests == [request["json"]]
    assert requests[0] is not request["json"]
    assert result.content == "Cold"
    assert collected[0].output[0].prompt_token_ids == [1, 2]
    assert collected[0].output[0].routed_experts == [[[0, 1]]]


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


@pytest.mark.parametrize(
    "normalized_content",
    [
        "Cold",
        [{"type": "text", "text": "Cold"}],
        [{"type": "input_text", "text": "Cold"}],
    ],
)
@pytest.mark.asyncio
async def test_restores_message_token_metadata_from_normalized_content(normalized_content: object) -> None:
    output = NeMoGymResponseOutputMessageForTraining(
        id="msg-1",
        content=[NeMoGymResponseOutputText(annotations=[], text="Cold")],
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.2],
    )
    gaps: list[ObservationGap] = []
    llm, client, _ = make_llm(model_response(output), observation_gaps=gaps)

    await llm.acall([{"role": "user", "content": "Weather?"}])
    await llm.acall([{"role": "assistant", "content": normalized_content}])

    restored = client.post.await_args_list[1].kwargs["json"].input[0]
    assert restored.id == "msg-1"
    assert restored.prompt_token_ids == [1, 2]
    assert restored.generation_token_ids == [3]
    assert restored.generation_log_probs == [-0.2]
    assert gaps == []


@pytest.mark.asyncio
async def test_records_gap_when_prior_message_token_metadata_cannot_be_restored() -> None:
    output = NeMoGymResponseOutputMessageForTraining(
        id="msg-1",
        content=[NeMoGymResponseOutputText(annotations=[], text="Cold")],
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.2],
    )
    gaps: list[ObservationGap] = []
    llm, client, _ = make_llm(model_response(output), observation_gaps=gaps)

    await llm.acall([{"role": "user", "content": "Weather?"}])
    await llm.acall([{"role": "assistant", "content": "Rewritten: Cold"}])

    unrestored = client.post.await_args_list[1].kwargs["json"].input[0]
    assert not hasattr(unrestored, "generation_token_ids")
    assert [gap.code for gap in gaps] == ["prior_output_metadata_unrestored"]
    assert "generated tokens may be masked" in gaps[0].detail


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
async def test_invalid_structured_output_is_identified_as_policy_output() -> None:
    output = NeMoGymResponseOutputMessageForTraining(
        id="msg-1",
        content=[NeMoGymResponseOutputText(annotations=[], text="not JSON")],
        prompt_token_ids=[1],
        generation_token_ids=[2],
        generation_log_probs=[-0.1],
    )
    llm, _, collected = make_llm(model_response(output))

    with pytest.raises(InvalidPolicyOutputError, match="invalid StructuredAnswer JSON"):
        await llm.acall([{"role": "user", "content": "Classify"}], output_model=StructuredAnswer)

    assert collected[0].output[0].generation_token_ids == [2]


@pytest.mark.asyncio
async def test_enforces_total_policy_call_budget() -> None:
    output = NeMoGymResponseOutputMessageForTraining(
        id="msg-1",
        content=[NeMoGymResponseOutputText(annotations=[], text="done")],
        prompt_token_ids=[1],
        generation_token_ids=[2],
        generation_log_probs=[-0.1],
    )
    llm, client, _ = make_llm(model_response(output), max_steps=1)
    await llm.acall([{"role": "user", "content": "first"}])

    with pytest.raises(PolicyCallBudgetExceeded, match="exhausted"):
        await llm.acall([{"role": "user", "content": "second"}])

    client.post.assert_awaited_once()


def test_rejects_synchronous_policy_calls() -> None:
    llm, _, _ = make_llm(model_response())

    with pytest.raises(RuntimeError, match="async"):
        llm.call([])
