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
"""Replay and evidence contracts for the SPEED-Bench agent."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi import Request, Response
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord
from responses_api_agents.speed_bench_agent.app import (
    SpeedBenchAgent,
    SpeedBenchAgentConfig,
    SpeedBenchAgentRunRequest,
    _content_text,
)


def _http(
    payload: dict | bytes,
    *,
    cookies: dict | None = None,
    status: int = 200,
    headers: dict | None = None,
) -> MagicMock:
    content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    result = MagicMock(status=status, ok=status < 400, cookies=cookies or {}, headers=CIMultiDict(headers or {}))
    result.read = AsyncMock(return_value=content)
    result.content.read = AsyncMock(return_value=content)
    if status >= 400:
        url = URL("http://speed-agent/v1/responses")
        result.raise_for_status.side_effect = ClientResponseError(
            RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url),
            (),
            status=status,
            headers=result.headers,
        )
    return result


def _model_response(turn: int) -> dict:
    return {
        "id": f"response-{turn}",
        "created_at": 0.0,
        "model": "policy",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": f"message-{turn}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": f"answer-{turn}", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10 * turn,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": turn,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 11 * turn,
        },
        "metrics": {"spec_decode": {"accepted_tokens": turn}},
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _agent(*, capture: bool = True) -> tuple[SpeedBenchAgent, MagicMock]:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": capture}
    agent = SpeedBenchAgent(
        config=SpeedBenchAgentConfig(
            host="127.0.0.1",
            port=8080,
            name="speed_agent",
            entrypoint="app.py",
            resources_server=ResourcesServerRef(type="resources_servers", name="speed_resources"),
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
        ),
        server_client=client,
    )
    return agent, client


@pytest.mark.parametrize("capture", [False, True])
def test_multiturn_replay_preserves_metrics_and_owns_each_model_call(capture: bool) -> None:
    agent, transport = _agent(capture=capture)
    replies = [_model_response(1), _model_response(2)]
    sent = []

    async def post(**kwargs):
        sent.append(kwargs | {"json": kwargs["json"].model_dump(mode="json")})
        return _http(replies[len(sent) - 1])

    transport.post = AsyncMock(side_effect=post)
    with TestClient(agent.setup_webserver()) as client:
        reply = client.post(
            "/ng-rollout/task-0/v1/responses",
            json={"input": [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]},
        )
    assert reply.status_code == 200, reply.text
    result = reply.json()
    calls = sent
    assert [call["url_path"] for call in calls] == ["/ng-rollout/task-0/v1/responses"] * 2
    assert [[item["role"] for item in call["json"]["input"]] for call in calls] == [
        ["user"],
        ["user", "assistant", "user"],
    ]
    assert [item["content"] for item in calls[1]["json"]["input"]] == ["first", "answer-1", "second"]
    assert [item["content"][0]["text"] for item in result["output"]] == ["answer-1", "answer-2"]
    assert result["usage"]["total_tokens"] == 33
    assert result["metrics"]["requests"] == [reply["metrics"] for reply in replies]
    assert result["metrics"]["generation_duration"] >= 0
    if capture:
        trajectory = TrajectoryRecord.model_validate(result["_ng_trajectory"])
        invocation = trajectory.invocations[0]
        assert invocation.invocation_id == "root"
        assert invocation.status == "completed"
        assert invocation.in_session is False
        assert [ref.response_id for ref in invocation.model_calls] == ["response-1", "response-2"]
        assert all(ref.model_ref == agent.config.model_server for ref in invocation.model_calls)
        assert [turn.turn_no for turn in trajectory.turns] == [1, 2]
        assert [[ref.response_id for ref in turn.model_calls] for turn in trajectory.turns] == [
            ["response-1"],
            ["response-2"],
        ]
        assert len(trajectory.turns[0].question) == 1
        assert len(trajectory.turns[1].question) == 3
        assert [item.role for item in invocation.conversation] == ["user", "assistant", "user", "assistant"]
        assert all(call["headers"] == {"x-session-id": "root"} for call in calls)
        assert not trajectory.gaps
    else:
        assert "_ng_trajectory" not in result
        assert all("headers" not in call for call in calls)


@pytest.mark.parametrize("failure_stage", [None, "model", "verify"])
def test_run_preserves_evidence_through_self_dispatch(failure_stage: str | None) -> None:
    agent, transport = _agent()
    verify_payloads = []
    self_replies = []
    model_calls = []

    async def post(*, server_name, url_path, json, cookies, **kwargs):
        payload = json.model_dump(mode="json") if hasattr(json, "model_dump") else json
        if server_name == "policy":
            model_calls.append(payload)
            if failure_stage == "model" and len(model_calls) == 2:
                raise RuntimeError("model transport failed")
            return _http(_model_response(len(model_calls)))
        if server_name == "speed_agent":
            reply = await asyncio.to_thread(client.post, url_path, json=payload, cookies=cookies)
            self_replies.append(reply)
            return _http(
                reply.content, cookies=dict(reply.cookies), status=reply.status_code, headers=dict(reply.headers)
            )
        if url_path == "/verify":
            verify_payloads.append(payload)
            if failure_stage == "verify":
                return _http({"detail": "verification failed"}, status=503)
            return _http(payload | {"reward": 0.0})
        assert url_path == "/seed_session"
        return _http({})

    transport.post = AsyncMock(side_effect=post)
    app = agent.setup_webserver()
    agent.setup_exception_middleware(app)
    with TestClient(app, raise_server_exceptions=False) as client:
        reply = client.post(
            "/run",
            json={
                "task_id": "source-task",
                "_ng_rollout_id": "task-0",
                "responses_create_params": {
                    "input": [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]
                },
            },
        )
    assert reply.status_code == (500 if failure_stage else 200), reply.text
    assert self_replies[0].status_code == (500 if failure_stage == "model" else 200)
    result = reply.json()
    if failure_stage == "model":
        assert "ng_trajectory" not in result
        assert "ng_trajectory" not in self_replies[0].json()
        assert "_ng_trajectory" not in self_replies[0].json()
        observations = AgentObservationBundle.model_validate(result["ng_agent_observations"])
        assert observations.model_dump(mode="json") == self_replies[0].json()["ng_agent_observations"]
        assert observations.source == "speed_bench_agent"
        assert not observations.gaps
        invocation = observations.records[0]
        assert invocation.invocation_id == "root"
        assert not verify_payloads
        assert invocation.status == "failed"
        assert invocation.error_type == "RuntimeError"
        assert [ref.response_id for ref in invocation.model_calls] == ["response-1"]
        assert [item.role for item in invocation.conversation] == ["user", "assistant"]
    else:
        trajectory = TrajectoryRecord.model_validate(result["ng_trajectory"])
        assert trajectory.task_id == "source-task"
        assert trajectory.rollout_id == "task-0"
        assert [(turn.task_id, turn.rollout_id) for turn in trajectory.turns] == [("source-task", "task-0")] * 2
        invocation = trajectory.invocations[0]
        assert invocation.status == "completed"
        assert [ref.response_id for ref in invocation.model_calls] == ["response-1", "response-2"]
        assert "_ng_trajectory" in self_replies[0].json()
        assert "_ng_trajectory" not in verify_payloads[0]["response"]
        if failure_stage == "verify":
            assert invocation.verification_skipped is None
            assert "verification failed" in result["detail"]
            assert "reward" not in result
        else:
            assert invocation.verification_skipped is False
            assert result["reward"] == 0.0
            assert result["response"]["usage"]["total_tokens"] == 33
            assert "_ng_trajectory" not in result["response"]


def test_missing_response_id_keeps_the_unavailable_reference_explicit() -> None:
    agent, transport = _agent()
    transport.post = AsyncMock(side_effect=[_http(_model_response(1)), _http(_model_response(2) | {"id": ""})])
    with TestClient(agent.setup_webserver()) as client:
        reply = client.post(
            "/ng-rollout/task-0/v1/responses",
            json={"input": [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]},
        )
    assert reply.status_code == 200, reply.text
    result = reply.json()
    trajectory = TrajectoryRecord.model_validate(result["_ng_trajectory"])
    assert [ref.response_id for ref in trajectory.invocations[0].model_calls] == ["response-1"]
    assert [turn.turn_no for turn in trajectory.turns] == [1, 2]
    assert trajectory.turns[1].model_calls == []
    assert [(gap.code, gap.invocation_id, gap.detail) for gap in trajectory.gaps] == [
        ("model_call_reference_unavailable", "root", "turn:2")
    ]
    assert result["usage"]["total_tokens"] == 33
    assert [item["content"][0]["text"] for item in result["output"]] == ["answer-1", "answer-2"]


async def test_cancelled_model_call_keeps_partial_identity_free_observations() -> None:
    agent, transport = _agent()
    cancelled = asyncio.CancelledError("model request cancelled")
    transport.post = AsyncMock(side_effect=[_http(_model_response(1)), cancelled])
    request = Request(
        {
            "type": "http",
            "path": "/ng-rollout/task-0/v1/responses",
            "path_params": {"rollout_id": "task-0"},
            "headers": [],
        }
    )
    with pytest.raises(asyncio.CancelledError) as raised:
        await agent.responses(
            request,
            Response(),
            NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]
            ),
        )
    assert raised.value is cancelled
    assert not hasattr(request.state, "nemo_gym_agent_trajectory")
    observations = AgentObservationBundle.model_validate(request.state.nemo_gym_agent_observations)
    invocation = observations.records[0]
    assert invocation.status == "incomplete"
    assert invocation.error_type == "CancelledError"
    assert [ref.response_id for ref in invocation.model_calls] == ["response-1"]
    assert [item.role for item in invocation.conversation] == ["user", "assistant"]
    assert transport.post.await_count == 2


@pytest.mark.parametrize(
    "failure_content",
    [
        b"upstream proxy unavailable",
        b'{"ng_trajectory": {"invalid": true}}',
        json.dumps(
            {"ng_trajectory": TrajectoryRecord(task_id="other-task", rollout_id="other-rollout").model_dump()}
        ).encode(),
    ],
)
async def test_failed_self_dispatch_preserves_original_http_error(failure_content: bytes) -> None:
    agent, transport = _agent()
    failure = _http(failure_content, status=502, headers={"x-upstream-detail": "original"})
    error = failure.raise_for_status.side_effect
    cause = RuntimeError("original cause")
    error.__cause__ = cause
    transport.post = AsyncMock(side_effect=[_http({}), failure])
    request = Request({"type": "http", "headers": []})
    body = SpeedBenchAgentRunRequest.model_validate(
        {"task_id": "source-task", "_ng_rollout_id": "task-0", "responses_create_params": {"input": "first"}}
    )
    with pytest.raises(ClientResponseError) as raised:
        await agent.run(request, body)
    assert raised.value is error
    assert error.__cause__ is cause
    assert error.status == 502
    assert error.headers["x-upstream-detail"] == "original"
    assert error.response_content == failure_content
    failure.content.read.assert_awaited_once_with()
    failure.read.assert_not_awaited()
    assert not hasattr(request.state, "nemo_gym_agent_trajectory")
    assert transport.post.await_count == 2


async def test_successful_self_dispatch_rejects_another_rollouts_trajectory() -> None:
    agent, transport = _agent()
    response = _model_response(1) | {
        "_ng_trajectory": TrajectoryRecord(task_id="other-task", rollout_id="other-rollout").model_dump()
    }
    transport.post = AsyncMock(side_effect=[_http({}), _http(response)])
    request = Request({"type": "http", "headers": []})
    body = SpeedBenchAgentRunRequest.model_validate(
        {"task_id": "source-task", "_ng_rollout_id": "task-0", "responses_create_params": {"input": "first"}}
    )
    with pytest.raises(ValueError, match="trajectory does not match this run's rollout"):
        await agent.run(request, body)
    assert not hasattr(request.state, "nemo_gym_agent_trajectory")
    assert transport.post.await_count == 2


@pytest.mark.parametrize(
    ("first_status", "expected"),
    [
        ("completed", "completed"),
        ("failed", "failed"),
        ("incomplete", "incomplete"),
        ("cancelled", "incomplete"),
        ("queued", "unknown"),
        ("in_progress", "unknown"),
        (None, "unknown"),
    ],
)
async def test_later_completed_turn_preserves_earlier_noncompletion(first_status: str | None, expected: str) -> None:
    agent, transport = _agent()
    first = _model_response(1) | {"status": first_status}
    transport.post = AsyncMock(side_effect=[_http(first), _http(_model_response(2))])
    request = Request(
        {
            "type": "http",
            "path": "/ng-rollout/task-0/v1/responses",
            "path_params": {"rollout_id": "task-0"},
            "headers": [],
        }
    )
    response = await agent.responses(
        request,
        Response(),
        NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]
        ),
    )
    trajectory = TrajectoryRecord.model_validate(response.model_dump(mode="json")["_ng_trajectory"])
    assert trajectory.invocations[0].status == expected
    assert [ref.response_id for ref in trajectory.invocations[0].model_calls] == ["response-1", "response-2"]
    assert response.status == "completed"
    assert response.usage.total_tokens == 33


def test_content_text_string_passthrough():
    assert _content_text("hello") == "hello"


def test_content_text_list_with_text_attr():
    class _Item:
        text = "world"

    assert _content_text([_Item(), _Item()]) == "worldworld"


def test_content_text_list_with_dict_text_key():
    assert _content_text([{"text": "a"}, {"text": "b"}]) == "ab"


def test_content_text_list_falls_back_to_content_key():
    assert _content_text([{"content": "x"}]) == "x"


def test_content_text_list_skips_unknown_items():
    assert _content_text([{"foo": "bar"}, {"text": "ok"}]) == "ok"


def test_content_text_other_types_returns_empty():
    assert _content_text(None) == ""
    assert _content_text(42) == ""
