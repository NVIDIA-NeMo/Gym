# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response
from PIL import Image

from nemo_gym.server_utils import ServerClient
from resources_servers.agentic_mme.app import AgenticMMEConfig, AgenticMMEServer, AgenticMMEVerifyRequest
from resources_servers.agentic_mme.tests.test_app import response
from responses_api_agents.agentic_mme_agent.app import AgenticMMEAgent, AgenticMMEAgentConfig, AgenticMMERunRequest
from responses_api_agents.agentic_mme_agent.retrieval import Retrieval
from responses_api_agents.agentic_mme_agent.tools import image_data_url


class FakeResponse:
    def __init__(self, payload, cookies=None):
        self.payload = payload
        self.cookies = cookies or {}
        self.ok = True

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def call(name="crop", args=None, call_id="c"):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(args or {"image_index": 0, "bbox_2d": [0, 0, 500, 1000]}),
        "prompt_token_ids": [1],
        "generation_token_ids": [2],
        "generation_log_probs": [-0.1],
    }


def make_agent(outputs, **config):
    client = MagicMock(spec=ServerClient)
    seen = []
    pending = list(outputs)
    grader = AgenticMMEServer(
        config=AgenticMMEConfig(name="grader", host="localhost", port=1, entrypoint=""), server_client=client
    )

    async def post(**kwargs):
        if kwargs["url_path"] == "/verify":
            result = await grader.verify(AgenticMMEVerifyRequest.model_validate(kwargs["json"]))
            return FakeResponse(result.model_dump())
        seen.append(kwargs["json"].model_dump())
        return FakeResponse(pending.pop(0).model_dump(), {"model-session": "test"})

    client.post = AsyncMock(side_effect=post)
    agent = AgenticMMEAgent(
        config=AgenticMMEAgentConfig(
            name="agent",
            host="localhost",
            port=1,
            entrypoint="",
            resources_server={"type": "resources_servers", "name": "grader"},
            model_server={"type": "responses_api_models", "name": "model"},
            **config,
        ),
        server_client=client,
    )
    return agent, seen


def body() -> AgenticMMERunRequest:
    return AgenticMMERunRequest(
        responses_create_params={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What color is the image?"},
                        {
                            "type": "input_image",
                            "image_url": image_data_url(Image.new("RGB", (8, 4), "red")),
                            "detail": "auto",
                        },
                    ],
                }
            ]
        },
        verifier_metadata={
            "golden_answer": {"value": "red"},
            "private_marker": "never-send-to-policy",
            "process_evaluation": {"efficiency": {"reference_tool_calls": 0}},
        },
    )


def request() -> Request:
    return Request({"type": "http", "headers": []})


@pytest.mark.asyncio
async def test_multiturn_images_tokens_and_grading() -> None:
    agent, seen = make_agent([response(output=[call()]), response("<answer>red</answer>")])
    task = body()
    before = task.model_dump()
    result = await agent.run(request(), task)
    assert result.reward == 1
    assert result.tool_call_count == 1
    assert result.successful_tool_calls == 1
    assert result.overthink == 1
    assert result.process_scores_available is False
    assert result.tool_trace[0]["output"]["size"] == [4, 4]
    assert seen[1]["input"][: len(seen[0]["input"])] == seen[0]["input"]
    previous_call = next(item for item in seen[1]["input"] if item["type"] == "function_call")
    assert previous_call["generation_token_ids"] == [2]
    assert previous_call["generation_log_probs"] == [-0.1]
    assert any(part["type"] == "input_image" for part in seen[1]["input"][-1]["content"])
    assert "never-send-to-policy" not in json.dumps(seen)
    assert task.model_dump() == before
    assert len(seen[0]["tools"]) == 14


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [
        call("not_a_tool"),
        call("crop", {"image_index": 99, "bbox_2d": [0, 0, 500, 1000]}),
        {"type": "function_call", "name": "crop", "call_id": "bad", "arguments": "{"},
        {"type": "function_call", "name": "crop", "call_id": "bad", "arguments": "[]"},
    ],
)
async def test_invalid_calls_consume_budget(tool) -> None:
    agent, seen = make_agent([response(output=[tool]), response("red")], max_tool_calls=1)
    result = await agent.run(request(), body())
    assert result.reward == 1  # No tool-success bonus or arbitrary invalid-call penalty.
    assert result.tool_error_count == 1
    assert seen[1]["tools"] == []
    assert seen[1]["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_parallel_calls_cannot_bypass_budget() -> None:
    agent, seen = make_agent(
        [response(output=[call(call_id="one"), call(call_id="two")]), response("red")], max_tool_calls=1
    )
    result = await agent.run(request(), body())
    assert result.successful_tool_calls == 1
    assert result.tool_error_count == 1
    assert result.tool_trace[1]["output"]["error"].startswith("tool budget")
    # All tool replies precede new user messages, including over-budget replies.
    assert [item["type"] for item in seen[1]["input"][1:]] == [
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
        "message",
    ]


@pytest.mark.asyncio
async def test_round_budget_and_refusing_to_answer() -> None:
    agent, seen = make_agent([response(output=[]), response(output=[call()])], max_rounds=1)
    result = await agent.run(request(), body())
    assert len(seen) == 2
    assert result.reward == 0
    assert result.response.status == "incomplete"
    assert result.successful_tool_calls == 0


@pytest.mark.asyncio
async def test_two_rollouts_are_isolated() -> None:
    agent, _ = make_agent([response(output=[call()]), response("red"), response(output=[call()]), response("red")])
    first = await agent.run(request(), body())
    second = await agent.run(request(), body())
    assert first.tool_trace[0]["output"]["new_image_index"] == 1
    assert second.tool_trace[0]["output"]["new_image_index"] == 1


@pytest.mark.asyncio
async def test_no_tool_budget_and_missing_images() -> None:
    agent, seen = make_agent([response("red")], max_tool_calls=0)
    assert (await agent.run(request(), body())).reward == 1
    assert seen[0]["tools"] == []
    task = body()
    task.responses_create_params.input = "no image"
    with pytest.raises(ValueError, match="at least one"):
        await agent.run(request(), task)


@pytest.mark.asyncio
async def test_retrieval_replay_and_visibility() -> None:
    agent, seen = make_agent(
        [response(output=[call("google_search", {"query": "red"})]), response("red")], retrieval={"mode": "replay"}
    )
    task = body()
    task.retrieval_replay = [
        {
            "tool_name": "google_search",
            "arguments": {"query": "red"},
            "output": {"ok": True, "context": "recorded red result"},
        }
    ]
    result = await agent.run(request(), task)
    assert result.reward == 1
    assert result.successful_tool_calls == 1
    assert len(seen[0]["tools"]) == 17
    assert "recorded red result" in seen[1]["input"][-1]["output"]


@pytest.mark.asyncio
async def test_retrieval_timeout_is_a_tool_error(monkeypatch) -> None:
    monkeypatch.setattr(Retrieval, "call", AsyncMock(side_effect=TimeoutError))
    agent, _ = make_agent(
        [response(output=[call("google_search", {"query": "red"})]), response("red")], retrieval={"mode": "replay"}
    )
    result = await agent.run(request(), body())
    assert result.tool_error_count == 1
    assert result.tool_trace[0]["output"]["error"] == "retrieval provider unavailable or timed out"


@pytest.mark.asyncio
async def test_incomplete_model_response_never_scores() -> None:
    incomplete = response("red")
    incomplete = type(incomplete).model_validate(
        incomplete.model_dump() | {"incomplete_details": {"reason": "max_output_tokens"}}
    )
    agent, seen = make_agent([incomplete])
    result = await agent.run(request(), body())
    assert result.reward == 0
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_responses_endpoint_propagates_cookies() -> None:
    agent, _ = make_agent([response("red")])
    outgoing = Response()
    result = await agent.responses(request(), outgoing, body().responses_create_params)
    assert result.output_text == "red"
    assert "model-session=test" in outgoing.headers["set-cookie"]


@pytest.mark.asyncio
async def test_multi_image_and_chain_indices() -> None:
    agent, _ = make_agent(
        [
            response(output=[call("flip", {"image_index": 1})]),
            response(output=[call("crop", {"image_index": 2, "bbox_2d": [0, 0, 500, 1000]})]),
            response("red"),
        ]
    )
    task = body()
    serialized = task.model_dump()
    serialized["responses_create_params"]["input"][0]["content"].append(
        {"type": "input_image", "image_url": image_data_url(Image.new("RGB", (20, 10), "blue")), "detail": "auto"}
    )
    task = AgenticMMERunRequest.model_validate(serialized)
    result = await agent.run(request(), task)
    assert result.tool_trace[0]["output"]["new_image_index"] == 2
    assert result.tool_trace[1]["output"]["new_image_index"] == 3
    assert result.tool_trace[1]["output"]["size"] == [10, 10]


@pytest.mark.asyncio
async def test_wire_input_items_keep_their_type() -> None:
    # ServerClient serializes with exclude_unset; the model server's converter needs every
    # input item's type (function_call_output items have no role to infer it from).
    agent, _ = make_agent([response(output=[call()]), response("<answer>red</answer>")])
    await agent.run(request(), body())
    model_calls = [
        c.kwargs for c in agent.server_client.post.call_args_list if c.kwargs["url_path"] == "/v1/responses"
    ]
    wire = model_calls[1]["json"].model_dump(exclude_unset=True)["input"]
    assert [item.get("type") for item in wire[-2:]] == ["function_call_output", "message"]
    assert all(item.get("type") or item.get("role") for item in wire)


@pytest.mark.asyncio
async def test_refused_model_call_ends_episode_unanswered() -> None:
    # e.g. an endpoint's request-size limit after several tool images: the rollout is scored 0
    # with the error recorded, instead of a 500 that aborts the whole collection.
    refused = FakeResponse(b"", {})
    refused.ok, refused.status = False, 400
    refused.read = AsyncMock(return_value=b'{"error": {"code": "request_too_large"}}')
    agent, seen = make_agent([response(output=[call()])])
    post = agent.server_client.post.side_effect

    async def post_or_refuse(**kwargs):
        if kwargs["url_path"] == "/v1/responses" and len(seen) == 1:
            return refused
        return await post(**kwargs)

    agent.server_client.post.side_effect = post_or_refuse
    result = await agent.run(request(), body())
    assert result.reward == 0 and result.failure_reason == "policy_error"
    assert "request_too_large" in result.policy_error and result.tool_call_count == 1


@pytest.mark.asyncio
async def test_no_round_or_tool_call_limit() -> None:
    outputs = [response(output=[call(call_id=f"c{i}")]) for i in range(20)] + [response("<answer>red</answer>")]
    agent, seen = make_agent(outputs, max_rounds=None, max_tool_calls=None)
    result = await agent.run(request(), body())
    assert result.reward == 1 and result.tool_call_count == 20 and result.tool_error_count == 0
    assert all(request_json["tools"] for request_json in seen)  # never forced into answer-only mode
