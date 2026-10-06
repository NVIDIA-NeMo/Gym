# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import orjson
import pytest
from pydantic import ValidationError

from environment_servers.single_agent_turn.tests.test_app import (
    _agent_response,
    _Client,
    _environment_server,
    _request,
    _Response,
)
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput


@pytest.mark.parametrize("partial", [False, True])
async def test_agent_deadline_stops_then_grades_preserved_state(monkeypatch, partial: bool) -> None:
    environment, client = _environment_server()
    request = _request()
    request.task.task_input.agent_timeout_seconds = 0.01
    saved = _agent_response().model_copy(update={"status": "incomplete"})
    receipt = orjson.loads(client.responses[3].body)
    if partial:
        receipt["partial_response"] = saved.model_dump(mode="json")
    receipt["agent_observations"] = {"source": "test", "records": [{"kind": "sandbox", "role": "agent"}]}
    responses = client.responses
    client.responses = [*responses[:2], _Response(receipt), responses[4], responses[5]]
    original_post = _Client.post

    async def post(self, server_name, url_path, **kwargs):
        if url_path.endswith("/v1/responses"):
            self.calls.append((server_name, url_path, kwargs))
            await asyncio.Event().wait()
        if url_path in {"/seed_session", "/verify"}:
            # Setup and verification have their own budgets, outside the agent timer.
            await asyncio.sleep(0.02)
        if url_path == "/verify":
            body = orjson.loads(self.responses[0].body)
            body["response"] = kwargs["json"]["response"]
            self.responses[0] = _Response(body)
        return await original_post(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    result = await asyncio.wait_for(environment.run_request(request), 2)

    assert result.failure is None
    assert result.result.agent_timed_out is True
    assert result.result.agent_timeout_seconds == 0.01
    assert result.result.reward == 1
    assert result.result.ng_agent_observations.source == "test"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    assert client.calls[4][2]["cookies"] == {"session": "updated-cookie"}
    if partial:
        assert result.result.response == saved
    else:
        response = result.result.response
        assert response.output == []
        assert response.status == "incomplete"
        assert response.metadata["agent_timed_out"] == "true"
        assert response.metadata["response_source"] == "environment_timeout_envelope"
        assert response.usage is None  # Missing captured usage must not be invented as zero.
    assert not client.responses


async def test_transport_timeout_before_agent_deadline_does_not_grade() -> None:
    environment, client = _environment_server()
    request = _request()
    request.task.task_input.agent_timeout_seconds = 10
    responses = client.responses
    client.responses = [*responses[:2], TimeoutError("upstream socket timed out"), responses[3], responses[5]]

    result = await environment.run_request(request)

    assert result.result is None
    assert result.failure.stage == "agent"
    assert result.failure.message == "upstream socket timed out"
    assert result.failure.terminal is False
    assert not any(path == "/verify" for _, path, _ in client.calls)
    assert not client.responses


async def test_agent_deadline_cleanup_failure_blocks_grading(monkeypatch) -> None:
    environment, client = _environment_server()
    request = _request()
    request.task.task_input.agent_timeout_seconds = 0.01
    responses = client.responses
    client.responses = [*responses[:2], RuntimeError("agent could not stop"), responses[3], responses[5]]
    original_post = _Client.post

    async def post(self, server_name, url_path, **kwargs):
        if url_path.endswith("/v1/responses"):
            self.calls.append((server_name, url_path, kwargs))
            await asyncio.Event().wait()
        return await original_post(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    result = await asyncio.wait_for(environment.run_request(request), 2)

    assert result.result is None
    assert result.failure.stage == "cleanup"
    assert result.failure.message == "agent could not stop"
    assert not any(path == "/verify" for _, path, _ in client.calls)


async def test_no_deadline_preserves_normal_result() -> None:
    environment, _ = _environment_server()
    request = _request()
    assert request.task.task_input.agent_timeout_seconds is None
    result = await environment.run_request(request)
    assert result.result.agent_timed_out is False
    assert result.result.agent_timeout_seconds is None
    assert result.result.response == _agent_response()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_agent_budget_must_be_positive_and_finite(timeout: float) -> None:
    with pytest.raises(ValidationError):
        SingleAgentTurnTaskInput(
            responses_create_params={"input": "task"}, task_data={}, agent_timeout_seconds=timeout
        )
