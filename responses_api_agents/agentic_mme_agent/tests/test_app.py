# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from time import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientConnectionError, ClientPayloadError
from fastapi import Request, Response
from omegaconf import OmegaConf
from PIL import Image

from nemo_gym.core import NeMoGymResponseUsage
from nemo_gym.models import CaptureStore
from nemo_gym.rollouts import RolloutCollectionHelper, run_health_checks
from nemo_gym.server_runtime.base_server import ServerClient
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ToolCallObservation
from resources_servers.agentic_mme.app import AgenticMMEConfig, AgenticMMEServer, AgenticMMEVerifyRequest
from resources_servers.agentic_mme.tests.test_app import response
from responses_api_agents.agentic_mme_agent.app import (
    AgenticMMEAgent,
    AgenticMMEAgentConfig,
    AgenticMMERunRequest,
    PolicyRequestError,
)
from responses_api_agents.agentic_mme_agent.retrieval import Retrieval
from responses_api_agents.agentic_mme_agent.tools import image_data_url


class FakeResponse:
    def __init__(self, payload, cookies=None):
        self.payload = payload
        self.cookies = cookies or {}
        self.ok = True
        self.status = 200
        self.headers = {}

    def release(self) -> None:
        pass

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


def make_agent(outputs, *, observability_enabled=False, **config):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": observability_enabled}
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
    # A path is required since url_path_for_request reads request.url.path (token-capture routes).
    return Request(
        {
            "type": "http",
            "headers": [],
            "method": "POST",
            "scheme": "http",
            "server": ("localhost", 80),
            "path": "/v1/responses",
            "query_string": b"",
        }
    )


@pytest.mark.asyncio
async def test_multiturn_images_tokens_and_grading() -> None:
    agent, seen = make_agent(
        [
            response(output=[call()]).model_copy(update={"id": "policy-crop"}),
            response("<answer>red</answer>").model_copy(update={"id": "policy-answer"}),
        ],
        observability_enabled=True,
    )
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
    observations = AgentObservationBundle.model_validate(result.model_dump()["ng_agent_observations"])
    [invocation] = [record for record in observations.records if isinstance(record, AgentInvocation)]
    [tool] = [record for record in observations.records if isinstance(record, ToolCallObservation)]
    assert invocation.status == "completed"
    assert [ref.response_id for ref in invocation.model_calls] == ["policy-crop", "policy-answer"]
    assert invocation.conversation[-1].model_dump() == result.response.output[-1].model_dump()
    assert len(invocation.conversation) == len(task.responses_create_params.input) + len(result.response.output)
    assert (tool.tool_call_id, tool.tool_name, tool.status) == ("c", "crop", "completed")
    assert tool.started_at <= tool.completed_at and tool.duration_ms >= 0


@pytest.mark.asyncio
async def test_collected_policy_turns_and_tool_attempts_are_observed(tmp_path: Path, monkeypatch) -> None:
    replies = [
        response(output=[call(call_id="crop-tool")]).model_copy(
            update={
                "id": "policy-crop",
                "status": "completed",
                "usage": NeMoGymResponseUsage(
                    input_tokens=11,
                    output_tokens=3,
                    total_tokens=14,
                    input_tokens_details={"cached_tokens": 0},
                    output_tokens_details={"reasoning_tokens": 0},
                ),
            }
        ),
        response("<answer>red</answer>").model_copy(
            update={
                "id": "policy-answer",
                "status": "completed",
                "usage": NeMoGymResponseUsage(
                    input_tokens=23,
                    output_tokens=7,
                    total_tokens=30,
                    input_tokens_details={"cached_tokens": 0},
                    output_tokens_details={"reasoning_tokens": 0},
                ),
            }
        ),
    ]
    agent, _ = make_agent(replies, observability_enabled=True)
    client = agent.server_client
    capture = CaptureStore(tmp_path / "model_calls")
    client.global_config_dict = OmegaConf.create(
        {
            "observability_enabled": True,
            "model_call_capture_dir": str(capture.root),
            "agent": {
                "responses_api_agents": {"agentic_mme_agent": {"model_server": agent.config.model_server.model_dump()}}
            },
            "agent_environment": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "agent"}}}},
        }
    )
    upstream = client.post
    capture_id = None

    async def post(*, server_name, url_path, json, **kwargs):
        nonlocal capture_id
        if server_name == "agent_environment":
            assert url_path == "/run"
            result = await agent.run(request(), AgenticMMERunRequest.model_validate(json))
            return FakeResponse(result.model_dump(mode="json"))
        started = time()
        reply = await upstream(server_name=server_name, url_path=url_path, json=json, **kwargs)
        if server_name == "model":
            assert url_path.startswith("/ng-rollout/")
            capture_id = url_path.split("/")[2]
            capture.record(
                capture_id,
                {
                    "model_call_id": "captured-" + reply.payload["id"],
                    "model_ref": {"type": "responses_api_models", "name": "model"},
                    "dialect": "responses",
                    "started_at": started,
                    "completed_at": time(),
                    "request": json.model_dump(mode="json"),
                    "response": reply.payload,
                    "status_code": 200,
                },
            )
        elif server_name == "grader":
            capture.record(
                capture_id,
                {
                    "model_call_id": "captured-judge",
                    "model_ref": {"type": "responses_api_models", "name": "judge"},
                    "dialect": "chat",
                    "request": {"messages": [{"role": "user", "content": "grade the answer"}]},
                    "response": {
                        "id": "judge-answer",
                        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "yes"}}],
                        "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
                    },
                    "status_code": 200,
                },
            )
            reply.payload["ng_verifier_invocations"] = [
                {
                    "invocation_id": "judge",
                    "status": "completed",
                    "model_calls": [{"model_call_id": "captured-judge"}],
                }
            ]
        return reply

    client.post = AsyncMock(side_effect=post)
    monkeypatch.setattr(
        "nemo_gym.rollouts.rollout_collection.setup_server_client_utils", lambda *args, **kwargs: client
    )
    row = body().model_dump(mode="json") | {
        "agent_ref": {"name": "agent"},
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "_ng_rollout_id": "health-case",
    }
    completions = RolloutCollectionHelper().run_examples([row], enrich_results=True)
    try:
        [(_, collected)] = [await future for future in completions]
    finally:
        await completions.aclose()
    trajectory = collected["ng_trajectory"]
    assert "agent_ref" not in collected
    assert collected["reward"] == 1
    assert len(trajectory["turns"]) == 2
    assert [turn["model_calls"][0]["response_id"] for turn in trajectory["turns"]] == [
        "policy-crop",
        "policy-answer",
    ]
    assert len(trajectory["model_calls"]) == 3
    assert [tool["tool_call_id"] for tool in trajectory["tool_calls"]] == ["crop-tool"]
    assert not {gap["code"] for gap in trajectory["gaps"]} & {
        "turns_unavailable",
        "conversation_unavailable",
        "tool_timing_unavailable",
    }
    assert collected["ng_perf"]["prompt_tokens"] == 11 + 23
    assert collected["ng_perf"]["completion_tokens"] == 3 + 7
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(collected) + "\n")
    health = run_health_checks(rollouts, workers=1)
    assert health.summary["run"]["verdicts"] == {"healthy": 1, "unhealthy": 0, "unobserved": 0}


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
    agent, seen = make_agent([response(output=[tool]), response("red")], max_tool_calls=1, observability_enabled=True)
    result = await agent.run(request(), body())
    assert result.reward == 1  # No tool-success bonus or arbitrary invalid-call penalty.
    assert result.tool_error_count == 1
    # No answer-only call: the next call keeps the tools, so its prompt extends the previous one.
    assert seen[1]["tools"] == seen[0]["tools"]
    assert seen[1]["tool_choice"] == "auto"
    observations = AgentObservationBundle.model_validate(result.ng_agent_observations)
    [attempt] = [record for record in observations.records if isinstance(record, ToolCallObservation)]
    assert attempt.tool_call_id == tool["call_id"] and attempt.status == "failed"
    assert attempt.duration_ms >= 0 and attempt.started_at <= attempt.completed_at


@pytest.mark.asyncio
async def test_parallel_calls_cannot_bypass_budget() -> None:
    agent, seen = make_agent(
        [response(output=[call(call_id="one"), call(call_id="two")]), response("red")], max_tool_calls=1
    )
    result = await agent.run(request(), body())
    assert result.successful_tool_calls == 1
    assert result.tool_error_count == 1
    assert result.tool_trace[1]["output"]["error"].startswith("tool budget")
    # A call beyond the budget ends the episode unanswered: no further model call.
    assert len(seen) == 1 and result.reward == 0 and result.response.status == "incomplete"
    # Running out of budget is not a length cut, so RL's overlong filtering keeps the zero.
    assert result.response.incomplete_details is None and result.failure_reason == "missing_answer"
    # All tool replies precede new user messages, including over-budget replies.
    assert [item.type for item in result.response.output] == [
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
        "message",
    ]


@pytest.mark.asyncio
async def test_using_the_whole_tool_budget_still_allows_an_answer() -> None:
    agent, seen = make_agent([response(output=[call()]), response("<answer>red</answer>")], max_tool_calls=1)
    result = await agent.run(request(), body())
    assert result.reward == 1 and result.successful_tool_calls == 1
    assert len(seen) == 2 and seen[1]["tools"] == seen[0]["tools"]


@pytest.mark.asyncio
async def test_every_model_call_extends_the_previous_one() -> None:
    # Token capture links a rollout's calls by prompt prefix; the budget must never change the
    # tools or tool_choice of a later call.
    agent, seen = make_agent([response(output=[call(call_id=f"c{i}")]) for i in range(3)], max_rounds=2)
    result = await agent.run(request(), body())
    assert len(seen) == 3 and result.reward == 0
    for earlier, later in zip(seen, seen[1:]):
        assert later["tools"] == earlier["tools"] and later["tool_choice"] == earlier["tool_choice"] == "auto"
        assert later["input"][: len(earlier["input"])] == earlier["input"]


@pytest.mark.asyncio
async def test_round_budget_and_refusing_to_answer() -> None:
    agent, seen = make_agent([response(output=[]), response(output=[call()])], max_rounds=1)
    result = await agent.run(request(), body())
    assert len(seen) == 2
    assert result.reward == 0
    assert result.response.status == "incomplete" and result.response.incomplete_details is None
    # The last allowed call is a normal call: its tool call runs, then the rounds are used up.
    assert result.successful_tool_calls == 1


@pytest.mark.asyncio
async def test_two_rollouts_are_isolated() -> None:
    agent, _ = make_agent(
        [
            response(output=[call(call_id="first-tool")]).model_copy(update={"id": "first-crop"}),
            response("red").model_copy(update={"id": "first-answer"}),
            response(output=[call(call_id="second-tool")]).model_copy(update={"id": "second-crop"}),
            response("red").model_copy(update={"id": "second-answer"}),
        ],
        observability_enabled=True,
    )
    first = await agent.run(request(), body())
    first_observations = first.model_dump()["ng_agent_observations"]
    second = await agent.run(request(), body())
    assert first.tool_trace[0]["output"]["new_image_index"] == 1
    assert second.tool_trace[0]["output"]["new_image_index"] == 1
    assert first.model_dump()["ng_agent_observations"] == first_observations
    for result, expected in ((first, ["first-crop", "first-answer"]), (second, ["second-crop", "second-answer"])):
        observations = AgentObservationBundle.model_validate(result.ng_agent_observations)
        [invocation] = [record for record in observations.records if isinstance(record, AgentInvocation)]
        assert [ref.response_id for ref in invocation.model_calls] == expected


@pytest.mark.asyncio
async def test_no_tool_budget_and_missing_images() -> None:
    agent, seen = make_agent([response("red")], max_tool_calls=0)
    result = await agent.run(request(), body())
    assert result.reward == 1
    observations = AgentObservationBundle.model_validate(result.ng_agent_observations)
    [invocation] = [record for record in observations.records if isinstance(record, AgentInvocation)]
    assert invocation.status == "completed"
    assert len(seen[0]["tools"]) == 14  # tools stay visible; any call would end the episode
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
            "output": {"ok": True, "context": "recorded red result", "raw": {"answerBox": {"answer": "crimson"}}},
        }
    ]
    result = await agent.run(request(), task)
    assert result.reward == 1
    assert result.successful_tool_calls == 1
    assert len(seen[0]["tools"]) == 17
    assert "recorded red result" in seen[1]["input"][-1]["output"]
    # The provider answer stays in the trace; the model sees only the compact context.
    assert "crimson" not in json.dumps(seen[1]["input"]) and result.tool_trace[0]["output"]["raw"]


@pytest.mark.asyncio
async def test_leak_guard_is_invisible_to_the_model() -> None:
    agent, seen = make_agent(
        [
            response(output=[call("google_search", {"query": "red"})]),
            response(output=[call("fetch_webpage", {"url": "https://blog.example/post"})]),
            response("red"),
        ],
        retrieval={"mode": "replay"},
    )
    task = body()
    filtered_page = {
        "ok": False,
        "tool": "fetch_webpage",
        "url": "https://blog.example/post",
        "error": "no page content available for this URL",
        "contaminated": 1,
    }
    task.retrieval_replay = [
        {
            "tool_name": "google_search",
            "arguments": {"query": "red"},
            "output": {
                "ok": True,
                "tool": "google_search",
                "context": "No results.",
                "raw": {},
                "contaminated": 3,
                "cache": "exact",
            },
        },
        {"tool_name": "fetch_webpage", "arguments": {"url": "https://blog.example/post"}, "output": filtered_page},
    ]
    result = await agent.run(request(), task)
    search = json.loads(seen[1]["input"][-1]["output"])
    page = json.loads(seen[2]["input"][-1]["output"])
    assert search == {"ok": True, "tool": "google_search", "context": "No results."}
    # Exactly what an ordinary failed fetch shows.
    assert page == {"ok": False, "error": "no page content available for this URL"}
    assert [event["output"].get("contaminated") for event in result.tool_trace] == [3, 1]
    assert result.tool_trace[0]["output"]["cache"] == "exact"


@pytest.mark.asyncio
async def test_trace_keeps_tool_results_without_image_data() -> None:
    agent, seen = make_agent([response(output=[call()]), response("<answer>red</answer>")])
    result = await agent.run(request(), body())
    output = result.tool_trace[0]["output"]
    assert output["ok"] and "image_url" not in output and output["new_image_index"] == 1
    assert "data:image" in json.dumps(seen[1]["input"])  # the model still gets the image


@pytest.mark.asyncio
async def test_retrieval_timeout_is_a_tool_error(monkeypatch) -> None:
    clock = {"wall": 100.0, "steady": 10.0}

    async def timeout(*args, **kwargs):
        clock["wall"] += 2.5
        clock["steady"] += 2.5
        raise TimeoutError

    monkeypatch.setattr("responses_api_agents.agentic_mme_agent.app.time", lambda: clock["wall"])
    monkeypatch.setattr("responses_api_agents.agentic_mme_agent.app.perf_counter", lambda: clock["steady"])
    monkeypatch.setattr(Retrieval, "call", AsyncMock(side_effect=timeout))
    agent, _ = make_agent(
        [response(output=[call("google_search", {"query": "red"})]), response("red")],
        retrieval={"mode": "replay"},
        observability_enabled=True,
    )
    result = await agent.run(request(), body())
    assert result.tool_error_count == 1
    assert result.tool_trace[0]["output"]["error"] == "retrieval provider unavailable or timed out"
    observations = AgentObservationBundle.model_validate(result.ng_agent_observations)
    [attempt] = [record for record in observations.records if isinstance(record, ToolCallObservation)]
    assert attempt.status == "timeout" and attempt.error_type == "TimeoutError"
    assert (attempt.started_at, attempt.completed_at, attempt.duration_ms) == (100.0, 102.5, 2500.0)


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


class UnreadableResponse(FakeResponse):
    """HTTP 200 whose body read fails or does not parse."""

    def __init__(self, body=None, error=None):
        super().__init__(None)
        self.body, self.error = body, error

    async def read(self) -> bytes:
        if self.error is not None:
            raise self.error
        return self.body


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_call", [0, 1], ids=["first_call", "mid_episode"])
@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError(),
        ClientConnectionError("connection reset"),
        UnreadableResponse(error=ClientPayloadError("response payload is not completed")),
        UnreadableResponse(body=b"<html>bad gateway</html>"),
        UnreadableResponse(body=b'{"not": "a response"}'),
    ],
    ids=["timeout", "connection", "payload", "not_json", "not_a_response"],
)
async def test_failed_model_call_without_status_goes_to_failures_sidecar(failing_call, failure) -> None:
    # A policy call that fails before giving an HTTP status is infrastructure: run() raises, so
    # rollout collection records the task in the failures sidecar and retries it.
    agent, seen = make_agent([response(output=[call()]), response("red")], observability_enabled=True)
    post = agent.server_client.post.side_effect
    model_calls = 0

    async def post_or_fail(**kwargs):
        nonlocal model_calls
        if kwargs["url_path"] == "/v1/responses":
            model_calls += 1
            if model_calls == failing_call + 1:
                if isinstance(failure, Exception):
                    raise failure
                return failure
        return await post(**kwargs)

    agent.server_client.post.side_effect = post_or_fail
    with pytest.raises(PolicyRequestError):
        await agent.run(request(), body())
    assert not any(c.kwargs["url_path"] == "/verify" for c in agent.server_client.post.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_body"),
    [
        (401, b'{"error": "invalid api key"}'),
        (429, b'{"error": "rate limit exceeded"}'),
        (500, b"internal server error"),
        (503, b"service unavailable"),
        (400, b'{"error": "unsupported parameter"}'),
        (400, b'{"error": "At most 1 image(s) may be provided in one prompt."}'),
    ],
    ids=["auth", "rate_limit", "server_error", "unavailable", "other_400", "serving_image_limit"],
)
async def test_refused_model_call_outside_the_model_goes_to_failures_sidecar(status, error_body) -> None:
    refused = FakeResponse(b"", {})
    refused.ok, refused.status = False, status
    refused.read = AsyncMock(return_value=error_body)
    agent, seen = make_agent([response(output=[call()])], observability_enabled=True)
    post = agent.server_client.post.side_effect

    async def post_or_refuse(**kwargs):
        if kwargs["url_path"] == "/v1/responses" and len(seen) == 1:
            return refused
        return await post(**kwargs)

    agent.server_client.post.side_effect = post_or_refuse
    with pytest.raises(PolicyRequestError, match=f"HTTP {status}"):
        await agent.run(request(), body())


@pytest.mark.asyncio
async def test_image_limit_is_a_tool_error_and_the_model_still_answers() -> None:
    agent, seen = make_agent(
        [response(output=[call(call_id="a")]), response(output=[call(call_id="b")]), response("<answer>red</answer>")],
        max_images_per_request=2,
    )
    result = await agent.run(request(), body())
    assert result.reward == 1 and result.successful_tool_calls == 1 and result.tool_error_count == 1
    assert result.tool_trace[1]["output"]["error"].startswith("image limit reached")
    images = [
        part
        for item in seen[-1]["input"]
        for part in (item.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "input_image"
    ]
    assert len(images) == 2


@pytest.mark.asyncio
async def test_no_round_or_tool_call_limit() -> None:
    outputs = [response(output=[call(call_id=f"c{i}")]) for i in range(20)] + [response("<answer>red</answer>")]
    agent, seen = make_agent(outputs, max_rounds=None, max_tool_calls=None)
    result = await agent.run(request(), body())
    assert result.reward == 1 and result.tool_call_count == 20 and result.tool_error_count == 0
    assert all(request_json["tools"] for request_json in seen)  # never forced into answer-only mode
