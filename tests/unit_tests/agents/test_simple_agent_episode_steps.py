# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""simple_agent in an episode whose steps a caller drives: one step per activation whatever its max_steps, the
steps counted by their activation's index, one invocation kept for the episode, and one model server session."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from nemo_gym.agents.agent_activation_types import AgentContinuationRequirements
from nemo_gym.agents.base_responses_api_agent import (
    AGENT_ACTIVATION_HEADER,
    AGENT_EPISODE_STEP_LIMIT_HEADER,
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
)
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.server_runtime.tool_access import DirectHTTPToolAccess
from nemo_gym.trajectory.trajectory_record import AgentInvocation, AgentObservationBundle, ModelCallRef
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    _end_episode_invocation,
    _episode_step,
    _EpisodeStep,
    _extend_episode_invocation,
)


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
ACTIVATION_PATH = "/ng-rollout/rollout/v1/responses"
TASK = [{"role": "user", "content": "weather?"}]


def _request(headers: dict[str, str]) -> Request:
    return Request({"type": "http", "headers": [(name.encode(), value.encode()) for name, value in headers.items()]})


def _invocation(invocation_id: str, *calls: str, status: str = "completed") -> AgentInvocation:
    return AgentInvocation(
        invocation_id=invocation_id, status=status, model_calls=[ModelCallRef(model_call_id=call) for call in calls]
    )


def _model_reply(step: int, cookies: dict[str, str] | None = None) -> MagicMock:
    """The model's reply in step ``step``: one tool call, so only a step limit stops the agent's own loop."""
    call = {
        "id": f"fc-{step}",
        "call_id": f"call-{step}",
        "name": "get_weather",
        "arguments": "{}",
        "type": "function_call",
        "status": "completed",
    }
    payload = {
        "id": f"resp-{step}",
        "created_at": 1.0,
        "model": "model",
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [call],
    }
    reply = MagicMock(status=200, ok=True, cookies=cookies or {})
    reply.read = AsyncMock(return_value=json.dumps(payload))
    return reply


def _agent(
    monkeypatch: pytest.MonkeyPatch, replies: list, *, max_steps: int | None = None
) -> tuple[SimpleAgent, AsyncMock, AsyncMock]:
    """A simple agent whose tool calls answer ``{}``, its model's replies in order, and the model and tool calls."""
    config = SimpleAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="simple",
        model_server=ModelServerRef(type="responses_api_models", name="model"),
        # Unset, the agent's own loop has no step limit.
        **({"max_steps": max_steps} if max_steps is not None else {}),
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"observability_enabled": True}
    server_client.post = AsyncMock(side_effect=replies)
    tool_reply = MagicMock(status=200, ok=True, cookies={})
    tool_reply.content.read = AsyncMock(return_value=b"{}")
    tool_calls = AsyncMock(return_value=tool_reply)
    monkeypatch.setattr("responses_api_agents.simple_agent.app.http_request", tool_calls)
    return SimpleAgent(config=config, server_client=server_client), server_client.post, tool_calls


def _seed_body(continuation: bool = False) -> dict:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session",
        episode_id=EPISODE,
        task_id=TaskId(taskset="example", task_id="0"),
        tool_accesses=[
            DirectHTTPToolAccess(name="weather.direct_http", required=True, base_url="http://resources:8000")
        ],
        continuation=AgentContinuationRequirements() if continuation else None,
    ).model_dump(mode="json")


def _session(
    monkeypatch: pytest.MonkeyPatch,
    replies: list,
    *,
    max_steps: int | None = None,
    raise_server_exceptions: bool = True,
    continuation: bool = False,
) -> tuple[TestClient, AsyncMock]:
    """A simple agent with an open session whose tool calls answer ``{}``, and its model's replies in order."""
    agent, model_calls, _ = _agent(monkeypatch, replies, max_steps=max_steps)
    client = TestClient(agent.setup_webserver(), raise_server_exceptions=raise_server_exceptions)
    seed = client.post("/v1/agent_sessions", json=_seed_body(continuation))
    assert seed.status_code == 200
    return client, model_calls


def _step(client: TestClient, conversation: list, step: int, limit: int) -> list:
    """Send step ``step`` of an episode limited to ``limit`` steps; returns the conversation that follows it."""
    reply = client.post(
        ACTIVATION_PATH,
        json={"input": conversation},
        headers={AGENT_ACTIVATION_HEADER: str(step), AGENT_EPISODE_STEP_LIMIT_HEADER: str(limit)},
    )
    assert reply.status_code == 200, reply.text
    return [*conversation, *reply.json()["output"]]


def _invocations(client: TestClient) -> list[dict]:
    close = client.post(
        "/v1/agent_sessions/close",
        json=AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=EPISODE).model_dump(mode="json"),
    )
    return [record for record in close.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"]


def test_the_step_and_its_limit_are_the_callers_or_none() -> None:
    assert _episode_step(_request({})) is None
    assert _episode_step(_request({AGENT_ACTIVATION_HEADER: "3"})) is None
    assert _episode_step(_request({AGENT_ACTIVATION_HEADER: "3", AGENT_EPISODE_STEP_LIMIT_HEADER: "50"})) == (
        _EpisodeStep(index=3, limit=50)
    )


@pytest.mark.parametrize("raw", ["0", "-3", "fifty", ""])
@pytest.mark.parametrize("header", [AGENT_EPISODE_STEP_LIMIT_HEADER, AGENT_ACTIVATION_HEADER])
def test_a_step_or_limit_that_is_not_a_positive_integer_is_refused(header: str, raw: str) -> None:
    headers = {AGENT_ACTIVATION_HEADER: "1", AGENT_EPISODE_STEP_LIMIT_HEADER: "50"} | {header: raw}

    with pytest.raises(HTTPException) as refused:
        _episode_step(_request(headers))

    assert refused.value.status_code == 422
    assert header in refused.value.detail


def test_a_step_limit_without_the_steps_index_is_refused() -> None:
    with pytest.raises(HTTPException) as refused:
        _episode_step(_request({AGENT_EPISODE_STEP_LIMIT_HEADER: "50"}))

    assert refused.value.status_code == 422
    assert AGENT_ACTIVATION_HEADER in refused.value.detail


def test_a_steps_invocation_extends_the_episodes_record_with_its_model_calls() -> None:
    other = _invocation("subagent", "s1")
    records = [_invocation("root", "m1"), other]

    left = _extend_episode_invocation(records, [_invocation("root", "m2"), _invocation("helper", "h1")])

    assert [record.invocation_id for record in records] == ["root", "subagent"]
    assert [call.model_call_id for call in records[0].model_calls] == ["m1", "m2"]
    assert records[1] is other
    assert [invocation.invocation_id for invocation in left] == ["helper"]


def test_the_first_steps_invocation_is_appended_as_it_is() -> None:
    first = [_invocation("root", "m1")]

    assert _extend_episode_invocation([], first) is first
    assert _extend_episode_invocation([_invocation("subagent", "s1")], first) is first
    assert _extend_episode_invocation([_invocation("root", "m1")], []) == []


def test_a_failed_step_ends_the_episodes_invocation_and_no_other_record() -> None:
    other = _invocation("subagent", "s1")
    observations = AgentObservationBundle(source="simple_agent", records=[_invocation("root", "m1"), other])

    ended = _end_episode_invocation(observations, "root", status="failed", error_type="RuntimeError")
    unmatched = _end_episode_invocation(observations, "helper", status="failed", error_type="RuntimeError")

    assert [(record.status, record.error_type) for record in ended.records] == [
        ("failed", "RuntimeError"),
        ("completed", None),
    ]
    assert [call.model_call_id for call in ended.records[0].model_calls] == ["m1"]
    assert observations.records[0].status == "completed"
    assert unmatched.records == observations.records


@pytest.mark.parametrize("max_steps", [None, 1, 50])
def test_each_step_is_one_model_call_whatever_the_agents_max_steps(
    monkeypatch: pytest.MonkeyPatch, max_steps: int | None
) -> None:
    client, model_calls = _session(monkeypatch, [_model_reply(1), _model_reply(2)], max_steps=max_steps)

    conversation = _step(client, TASK, 1, limit=5)
    conversation = _step(client, conversation, 2, limit=5)

    # Without the caller's limit, a model that keeps calling tools would keep the agent's own loop going.
    assert model_calls.await_count == 2
    assert [(item["type"], item["call_id"]) for item in conversation[1:]] == [
        ("function_call", "call-1"),
        ("function_call_output", "call-1"),
        ("function_call", "call-2"),
        ("function_call_output", "call-2"),
    ]


@pytest.mark.parametrize(("limit", "status"), [(3, "incomplete"), (5, "completed")])
def test_the_episodes_one_invocation_is_incomplete_only_when_its_last_step_reaches_the_limit(
    monkeypatch: pytest.MonkeyPatch, limit: int, status: str
) -> None:
    client, _ = _session(monkeypatch, [_model_reply(step) for step in (1, 2, 3)])

    conversation = TASK
    for step in (1, 2, 3):
        conversation = _step(client, conversation, step, limit)

    [invocation] = _invocations(client)
    assert (invocation["invocation_id"], invocation["status"]) == ("root", status)
    assert [call["response_id"] for call in invocation["model_calls"]] == ["resp-1", "resp-2", "resp-3"]


def test_a_step_sent_again_keeps_its_number(monkeypatch: pytest.MonkeyPatch) -> None:
    """A step the caller sends again, as after a dropped connection, is not counted as the next one, so the episode's
    second step still completes below its limit of three."""
    client, _ = _session(monkeypatch, [_model_reply(step) for step in (1, 2, 3)])

    conversation = _step(client, TASK, 1, limit=3)
    _step(client, TASK, 1, limit=3)
    _step(client, conversation, 2, limit=3)

    [invocation] = _invocations(client)
    assert invocation["status"] == "completed"


def test_a_step_sent_again_after_it_answered_gets_its_reply_without_running_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without checkpointing, a step sent again after its reply was lost, as after a dropped connection, does not call
    the model or run its tool calls a second time (a second email, say): it gets the reply the step gave."""
    agent, model_calls, tool_calls = _agent(monkeypatch, [_model_reply(1, {"model_session": "m1"}), _model_reply(2)])
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=_seed_body()).status_code == 200
    headers = {AGENT_ACTIVATION_HEADER: "1", AGENT_EPISODE_STEP_LIMIT_HEADER: "3"}

    first = client.post(ACTIVATION_PATH, json={"input": TASK}, headers=headers)
    again = client.post(ACTIVATION_PATH, json={"input": TASK}, headers=headers)

    assert first.status_code == again.status_code == 200
    assert again.json() == first.json()
    assert again.cookies.get("model_session") == first.cookies.get("model_session") == "m1"
    assert (model_calls.await_count, tool_calls.await_count) == (1, 1)


async def test_a_step_sent_again_while_it_runs_shares_its_run_after_the_first_caller_went_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step whose first request goes away while the step runs (a dropped connection cancels its handler) keeps
    running, and the step sent again gets its reply: one model call and one tool call."""
    model_called, answer = asyncio.Event(), asyncio.Event()

    async def model(*args: object, **kwargs: object) -> MagicMock:
        model_called.set()
        await answer.wait()
        return _model_reply(1)

    agent, model_calls, tool_calls = _agent(monkeypatch, [])
    model_calls.side_effect = model
    headers = {AGENT_ACTIVATION_HEADER: "1", AGENT_EPISODE_STEP_LIMIT_HEADER: "3"}
    transport = httpx.ASGITransport(app=agent.setup_webserver())
    async with httpx.AsyncClient(transport=transport, base_url="http://agent") as client:
        assert (await client.post("/v1/agent_sessions", json=_seed_body())).status_code == 200
        first = asyncio.create_task(client.post(ACTIVATION_PATH, json={"input": TASK}, headers=headers))
        await model_called.wait()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        again = asyncio.create_task(client.post(ACTIVATION_PATH, json={"input": TASK}, headers=headers))
        answer.set()
        reply = await again

    assert reply.status_code == 200, reply.text
    assert reply.json()["output"][0]["call_id"] == "call-1"
    assert (model_calls.await_count, tool_calls.await_count) == (1, 1)


async def test_closing_the_session_cancels_a_step_whose_caller_went_away(monkeypatch: pytest.MonkeyPatch) -> None:
    """A step that runs on after its caller went away ends when the session closes, as it ended with its request."""
    model_called, cancelled = asyncio.Event(), asyncio.Event()

    async def model(*args: object, **kwargs: object) -> MagicMock:
        model_called.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent, model_calls, tool_calls = _agent(monkeypatch, [])
    model_calls.side_effect = model
    headers = {AGENT_ACTIVATION_HEADER: "1", AGENT_EPISODE_STEP_LIMIT_HEADER: "3"}
    transport = httpx.ASGITransport(app=agent.setup_webserver())
    async with httpx.AsyncClient(transport=transport, base_url="http://agent") as client:
        assert (await client.post("/v1/agent_sessions", json=_seed_body())).status_code == 200
        first = asyncio.create_task(client.post(ACTIVATION_PATH, json={"input": TASK}, headers=headers))
        await model_called.wait()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        assert not cancelled.is_set()
        close = await client.post(
            "/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=EPISODE).model_dump(
                mode="json"
            ),
        )

    assert close.status_code == 200, close.text
    assert cancelled.is_set()
    assert tool_calls.await_count == 0


def test_a_step_older_than_the_sessions_latest_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    client, model_calls = _session(monkeypatch, [_model_reply(1), _model_reply(2)], raise_server_exceptions=False)
    conversation = _step(client, TASK, 1, limit=3)
    _step(client, conversation, 2, limit=3)

    late = client.post(
        ACTIVATION_PATH,
        json={"input": TASK},
        headers={AGENT_ACTIVATION_HEADER: "1", AGENT_EPISODE_STEP_LIMIT_HEADER: "3"},
    )

    assert late.status_code == 409
    assert model_calls.await_count == 2


@pytest.mark.parametrize(("driven", "sent"), [(True, [{}, {"model_session": "m1"}]), (False, [None, None])])
def test_the_steps_keep_the_episodes_model_server_session(
    monkeypatch: pytest.MonkeyPatch, driven: bool, sent: list
) -> None:
    replies = [_model_reply(1, {"model_session": "m1"}), _model_reply(2, {"model_session": "m1"})]
    client, model_calls = _session(monkeypatch, replies, max_steps=1)

    if driven:
        _step(client, _step(client, TASK, 1, limit=5), 2, limit=5)
    else:
        # Activations without the caller's limit start a model server session each, as before.
        for _ in range(2):
            assert client.post(ACTIVATION_PATH, json={"input": TASK}).status_code == 200

    assert [call.kwargs["cookies"] for call in model_calls.await_args_list] == sent


def test_a_continuation_session_is_not_driven_one_step_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each activation of a continuation session runs a whole turn on new input, so a caller that names a step limit
    is refused before the model is called."""
    client, model_calls = _session(monkeypatch, [_model_reply(1)], continuation=True)

    refused = client.post(
        ACTIVATION_PATH,
        json={"input": TASK},
        headers={AGENT_ACTIVATION_HEADER: "1", AGENT_EPISODE_STEP_LIMIT_HEADER: "5"},
    )

    assert refused.status_code == 422
    assert AGENT_EPISODE_STEP_LIMIT_HEADER in refused.json()["detail"]
    assert model_calls.await_count == 0


def test_a_step_that_raises_ends_the_episodes_invocation_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = [_model_reply(1), RuntimeError("model server unreachable")]
    client, _ = _session(monkeypatch, replies, raise_server_exceptions=False)

    conversation = _step(client, TASK, 1, limit=5)
    failed = client.post(
        ACTIVATION_PATH,
        json={"input": conversation},
        headers={AGENT_ACTIVATION_HEADER: "2", AGENT_EPISODE_STEP_LIMIT_HEADER: "5"},
    )

    assert failed.status_code == 500
    [invocation] = _invocations(client)
    assert (invocation["status"], invocation["error_type"]) == ("failed", "RuntimeError")
    assert [call["response_id"] for call in invocation["model_calls"]] == ["resp-1"]
