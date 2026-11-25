# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The step loop (max_episode_steps): the agent driven one step per activation, ending on the task state."""

import logging
import re
from types import SimpleNamespace
from typing import Any

import orjson
import pytest
from aiohttp import ClientConnectionError, ClientResponseError
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

import environment_servers.single_agent_turn.app as single_agent_turn_app
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.agents.base_responses_api_agent import AGENT_ACTIVATION_HEADER, AGENT_EPISODE_STEP_LIMIT_HEADER
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.core.failure_transport import RequestFailureLog
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.server_runtime.base_environment_server import dependency_failure
from nemo_gym.server_runtime.base_resources_server import RESOURCES_TASK_STATE_PATH
from tests.helpers.single_agent_turn import _request


ACTIVATION_PATH = "/ng-rollout/rollout-a2/v1/responses"
USAGE = {
    "input_tokens": 10,
    "output_tokens": 4,
    "total_tokens": 14,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 1},
}
SESSION_LOST_BODY = orjson.dumps({"detail": {"failure_kind": "session_lost", "message": "This session has ended."}})


class _Cookie:
    def __init__(self, value: str) -> None:
        self.value = value


class _Response:
    ok = True
    status = 200

    def __init__(self, body: dict, cookies: dict[str, str] | None = None) -> None:
        self.body = orjson.dumps(body)
        self.headers: dict[str, str] = {}
        self.cookies = {name: _Cookie(value) for name, value in (cookies or {}).items()}

    async def read(self) -> bytes:
        return self.body


class _ErrorResponse:
    """A reply with an HTTP error status, which raises as aiohttp's does."""

    ok = False

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body
        self.headers: dict[str, str] = {}
        self.cookies: dict[str, _Cookie] = {}

    async def read(self) -> bytes:
        return self.body

    def raise_for_status(self) -> None:
        url = f"http://resources:8000{RESOURCES_TASK_STATE_PATH}"
        request_info = SimpleNamespace(url=url, method="GET", headers={}, real_url=url)
        raise ClientResponseError(request_info, (), status=self.status, message="error", headers={})


def _tool_step(step: int, tool: str = "jira_add_comment") -> dict[str, Any]:
    """One step of the simple agent: a tool call and its output."""
    return {
        "output": [
            {
                "type": "function_call",
                "id": f"fc{step}",
                "call_id": f"call{step}",
                "name": tool,
                "arguments": "{}",
                "status": "completed",
            },
            {"type": "function_call_output", "call_id": f"call{step}", "output": f'{{"ok": {step}}}'},
        ]
    }


def _answer_step(step: int) -> dict[str, Any]:
    """A step whose model answers without calling a tool."""
    return {
        "output": [
            {
                "type": "message",
                "id": f"msg{step}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
            }
        ]
    }


def _reply(step: dict[str, Any], status: str | None = "completed", **fields: Any) -> dict[str, Any]:
    """The agent's reply to one activation; ``status=None`` leaves the status out, as some model servers do."""
    reply = {
        "id": "response",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "tools": [],
        "usage": USAGE,
        **step,
        **fields,
    }
    return reply if status is None else {**reply, "status": status}


class _Client(ServerClient):
    """The agent and the resources server: the agent replies with ``steps`` in order, and the task has ended from the
    step ``ended_after`` on. The activation numbered ``failing_activation`` loses its connection, and the task state
    read fails with ``task_state_error`` once that many steps ran, or answers ``task_state_reply``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, str, dict]]
    steps: list[dict]
    ended_after: int | None = None
    activations: int = 0
    failing_activation: int | None = None
    task_state_error: Exception | None = None
    task_state_error_from: int = 1
    task_state_reply: Any = None

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        body = kwargs.get("json")
        if isinstance(body, BaseModel):
            body = body.model_dump(mode="json")
        self.calls.append(("POST", server_name, url_path, {**kwargs, "json": body}))
        if url_path == "/seed_session":
            return _Response({"resources_session_id": body["resources_session_id"]}, {"resources": "seeded"})
        if url_path == "/v1/agent_sessions":
            return _Response({"agent_session_id": body["agent_session_id"]}, {"agent": "seeded"})
        if url_path.endswith("/v1/responses"):
            self.activations += 1
            if self.activations == self.failing_activation:
                raise ClientConnectionError("agent connection dropped")
            return _Response(self.steps[self.activations - 1])
        if url_path == "/v1/agent_sessions/close":
            return _Response({"agent_session_id": body["agent_session_id"]})
        if url_path == "/verify":
            return _Response({**body, "reward": 1.0})
        if url_path == "/close_session":
            return _Response({"resources_session_id": body["resources_session_id"]})
        raise AssertionError(f"unexpected POST {server_name}{url_path}")

    async def get(self, server_name: str, url_path: str, **kwargs) -> _Response | _ErrorResponse:
        self.calls.append(("GET", server_name, url_path, kwargs))
        assert (server_name, url_path) == ("resources", RESOURCES_TASK_STATE_PATH)
        if self.task_state_error is not None and self.activations >= self.task_state_error_from:
            raise self.task_state_error
        if self.task_state_reply is not None:
            return self.task_state_reply
        ended = self.ended_after is not None and self.activations >= self.ended_after
        return _Response({"task_ended": ended})

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"

    def activation_calls(self) -> list[dict]:
        return [kwargs for _, _, path, kwargs in self.calls if path.endswith("/v1/responses")]

    def paths(self) -> list[str]:
        return [f"{method} {path}" for method, _, path, _ in self.calls]


def _server(
    steps: list[dict],
    *,
    max_episode_steps: int | None = 5,
    end_on_task_state: bool = True,
    ended_after: int | None = None,
    agent: dict[str, Any] | None = None,
    agent_type: str = "simple_agent",
) -> tuple[SingleAgentTurnEnvironmentServer, _Client]:
    agent_config = {"host": "agent", "port": 8001, "entrypoint": "app.py", **(agent or {})}
    global_config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"officeenv": {"host": "resources", "port": 8000}}},
            "agent": {"responses_api_agents": {agent_type: agent_config}},
        }
    )
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        calls=[],
        steps=steps,
        ended_after=ended_after,
    )
    config = SingleAgentTurnEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "resources"},
        agent_server={"type": "responses_api_agents", "name": "agent"},
        cleanup_timeout_seconds=10,
        resources_tool_transports=["direct_http"],
        max_episode_steps=max_episode_steps,
        end_on_task_state=end_on_task_state and max_episode_steps is not None,
    )
    return SingleAgentTurnEnvironmentServer(config=config, server_client=client), client


def _verified_response(client: _Client) -> dict:
    return next(kwargs["json"]["response"] for _, _, path, kwargs in client.calls if path == "/verify")


async def test_the_episode_ends_at_the_step_after_which_the_task_has_ended() -> None:
    server, client = _server(
        [_reply(_tool_step(1)), _reply(_tool_step(2, "user_complete_task")), _reply(_tool_step(3))], ended_after=2
    )

    result = await server.run_request(_request())

    assert result.result.reward == 1.0
    assert client.activations == 2
    assert client.paths() == [
        "POST /seed_session",
        "POST /v1/agent_sessions",
        f"POST {ACTIVATION_PATH}",
        f"GET {RESOURCES_TASK_STATE_PATH}",
        f"POST {ACTIVATION_PATH}",
        f"GET {RESOURCES_TASK_STATE_PATH}",
        "POST /v1/agent_sessions/close",
        "POST /verify",
        "POST /close_session",
    ]
    response = _verified_response(client)
    assert [item["type"] for item in response["output"]] == ["function_call", "function_call_output"] * 2
    assert [item["call_id"] for item in response["output"]] == ["call1", "call1", "call2", "call2"]
    # Summed over the steps, as the agent's own loop sums its model calls.
    assert (response["usage"]["input_tokens"], response["usage"]["output_tokens"]) == (20, 8)
    assert response["usage"]["output_tokens_details"]["reasoning_tokens"] == 2
    # The task ended the episode, not the step limit.
    assert "ng_termination_reason" not in (response["metadata"] or {})


async def test_each_step_continues_the_conversation_and_names_the_episodes_step_limit() -> None:
    server, client = _server([_reply(_tool_step(1)), _reply(_tool_step(2)), _reply(_answer_step(3))])

    await server.run_request(_request())

    activations = client.activation_calls()
    assert [call["headers"] for call in activations] == [
        {AGENT_ACTIVATION_HEADER: str(step), AGENT_EPISODE_STEP_LIMIT_HEADER: "5"} for step in (1, 2, 3)
    ]
    inputs = [call["json"]["input"] for call in activations]
    assert [(item["role"], item["content"]) for item in inputs[0]] == [("user", "task")]
    assert [(item["type"], item["call_id"]) for item in inputs[2][1:]] == [
        ("function_call", "call1"),
        ("function_call_output", "call1"),
        ("function_call", "call2"),
        ("function_call_output", "call2"),
    ]
    assert inputs[1][1:] == inputs[2][1:3]
    # Every activation carries the agent session's cookies, as the seed set them.
    assert all(call["cookies"] == {"agent": "seeded"} for call in activations)


async def test_the_episode_ends_when_the_model_answers_without_a_tool() -> None:
    server, client = _server([_reply(_tool_step(1)), _reply(_answer_step(2)), _reply(_tool_step(3))])

    await server.run_request(_request())

    assert client.activations == 2
    # No tool ran in the last step, so the task state is not read after it.
    assert client.paths().count(f"GET {RESOURCES_TASK_STATE_PATH}") == 1
    response = _verified_response(client)
    assert [item["type"] for item in response["output"]] == ["function_call", "function_call_output", "message"]
    assert "ng_termination_reason" not in (response["metadata"] or {})


async def test_the_episode_ends_at_a_step_that_did_not_complete() -> None:
    server, client = _server([_reply(_tool_step(1), status="incomplete"), _reply(_tool_step(2))])

    await server.run_request(_request())

    assert client.activations == 1
    assert _verified_response(client)["status"] == "incomplete"


async def test_a_reply_without_a_status_goes_on_to_the_next_step() -> None:
    # A model server may answer without a status, and the simple agent passes the omission on.
    server, client = _server([_reply(_tool_step(1), status=None), _reply(_answer_step(2), status=None)])

    await server.run_request(_request())

    assert client.activations == 2
    assert client.paths().count(f"GET {RESOURCES_TASK_STATE_PATH}") == 1
    response = _verified_response(client)
    assert [item["type"] for item in response["output"]] == ["function_call", "function_call_output", "message"]
    assert (response["usage"]["input_tokens"], response["usage"]["output_tokens"]) == (20, 8)


@pytest.mark.parametrize("status", [None, "completed"])
async def test_a_truncated_reply_ends_the_episode_whatever_its_status(status: str | None) -> None:
    truncated = _reply(_tool_step(1), status=status, incomplete_details={"reason": "max_output_tokens"})
    server, client = _server([truncated, _reply(_tool_step(2))])

    await server.run_request(_request())

    assert client.activations == 1
    assert client.paths().count(f"GET {RESOURCES_TASK_STATE_PATH}") == 0


async def test_the_episode_ends_after_max_episode_steps_and_its_reply_says_so() -> None:
    server, client = _server([_reply(_tool_step(step)) for step in range(1, 5)], max_episode_steps=3)

    result = await server.run_request(_request())

    assert result.result.reward == 1.0
    assert client.activations == 3
    response = _verified_response(client)
    assert len(response["output"]) == 6
    # Only the metadata names the cutoff: the status stays the last step's, and incomplete_details stays unset.
    assert response["metadata"] == {"ng_termination_reason": "max_episode_steps"}
    assert (response["status"], response["incomplete_details"]) == ("completed", None)


async def test_the_task_state_is_not_read_after_the_last_step() -> None:
    # A read after the last step would fail; the episode ends there either way, so it is verified.
    server, client = _server([_reply(_tool_step(1)), _reply(_tool_step(2))], max_episode_steps=2)
    client.task_state_error = ConnectionError("resources server down")
    client.task_state_error_from = 2

    result = await server.run_request(_request())

    assert result.result.reward == 1.0
    assert client.activations == 2
    assert client.paths().count(f"GET {RESOURCES_TASK_STATE_PATH}") == 1
    assert "POST /verify" in client.paths()


async def test_without_end_on_task_state_only_the_steps_end_the_episode() -> None:
    server, client = _server(
        [_reply(_tool_step(1)), _reply(_tool_step(2)), _reply(_answer_step(3))], end_on_task_state=False, ended_after=1
    )

    await server.run_request(_request())

    assert client.activations == 3
    assert f"GET {RESOURCES_TASK_STATE_PATH}" not in client.paths()


async def test_the_task_state_is_read_with_the_resources_session_cookie() -> None:
    server, client = _server([_reply(_tool_step(1)), _reply(_answer_step(2))])

    await server.run_request(_request())

    reads = [kwargs for method, _, _, kwargs in client.calls if method == "GET"]
    assert reads == [{"cookies": {"resources": "seeded"}}]


async def test_a_failed_task_state_read_fails_the_episode_and_keeps_the_step_before_it() -> None:
    server, client = _server([_reply(_tool_step(1)), _reply(_tool_step(2))])
    client.task_state_error = ConnectionError("resources server down")

    result = await server.run_request(_request())

    assert result.result is None
    assert result.failure.stage == "agent"
    assert "resources server down" in result.failure.failure_reason
    assert client.activations == 1
    assert "POST /verify" not in client.paths()
    partial = result.failure.partial_response
    assert [(item.type, item.call_id) for item in partial.output] == [
        ("function_call", "call1"),
        ("function_call_output", "call1"),
    ]
    assert (partial.usage.input_tokens, partial.usage.output_tokens) == (10, 4)


async def test_a_step_that_fails_keeps_the_steps_before_it_with_their_usage() -> None:
    server, client = _server([_reply(_tool_step(1)), _reply(_tool_step(2)), _reply(_tool_step(3))])
    client.failing_activation = 3

    result = await server.run_request(_request())

    assert result.failure.stage == "agent"
    assert result.failure.terminal is False
    assert "POST /verify" not in client.paths()
    partial = result.failure.partial_response
    assert [(item.type, item.call_id) for item in partial.output] == [
        ("function_call", "call1"),
        ("function_call_output", "call1"),
        ("function_call", "call2"),
        ("function_call_output", "call2"),
    ]
    usage = partial.usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (20, 8, 28)
    assert usage.output_tokens_details.reasoning_tokens == 2


async def test_a_failure_before_the_first_reply_has_no_partial_response() -> None:
    server, client = _server([_reply(_tool_step(1))])
    client.failing_activation = 1

    result = await server.run_request(_request())

    assert result.failure.stage == "agent"
    assert result.failure.partial_response is None


async def test_a_partial_response_the_failure_carries_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    carried = NeMoGymResponse.model_validate(_reply(_answer_step(9)) | {"id": "carried"})

    async def task_ended(self: SingleAgentTurnEnvironmentServer, handles: Any) -> bool:
        raise dependency_failure(ConnectionError("lost"), stage="agent", partial_response=carried)

    monkeypatch.setattr(SingleAgentTurnEnvironmentServer, "_task_ended", task_ended)
    server, _ = _server([_reply(_tool_step(1)), _reply(_tool_step(2))])

    result = await server.run_request(_request())

    assert result.failure.partial_response.id == "carried"


async def test_a_lost_resources_session_ends_the_episode_and_goes_on_to_verification() -> None:
    # The same loss a tool call finds goes to /verify, which masks the episode.
    server, client = _server([_reply(_tool_step(1)), _reply(_tool_step(2))])
    client.task_state_reply = _ErrorResponse(404, SESSION_LOST_BODY)

    result = await server.run_request(_request())

    assert result.failure is None
    assert client.activations == 1
    assert client.paths()[-3:] == ["POST /v1/agent_sessions/close", "POST /verify", "POST /close_session"]


@pytest.mark.parametrize(
    ("status", "body", "terminal"),
    [
        (404, b'{"detail": "Not Found"}', True),
        (404, orjson.dumps({"detail": {"failure_kind": "verifier_error", "message": "no"}}), True),
        (404, b"[]", True),
        (404, b"not json", True),
        (500, SESSION_LOST_BODY, False),
    ],
)
async def test_any_other_failed_task_state_read_fails_the_episode(status: int, body: bytes, terminal: bool) -> None:
    server, client = _server([_reply(_tool_step(1)), _reply(_tool_step(2))])
    client.task_state_reply = _ErrorResponse(status, body)

    result = await server.run_request(_request())

    assert result.result is None
    assert (result.failure.stage, result.failure.terminal) == ("agent", terminal)
    assert "POST /verify" not in client.paths()


async def test_every_steps_failure_report_is_kept_not_only_the_last_steps() -> None:
    def report(event_id: str) -> dict[str, Any]:
        event = {"event_id": event_id, "boundary": "model", "failure_kind": "model_request_error"}
        return {"failure_id": None, "events": {event_id: {**event, "operation": "model_request"}}}

    steps = [
        _reply(_tool_step(1), failure_report=report("step-1")),
        _reply(_tool_step(2)),
        _reply(_answer_step(3), failure_report=report("step-3")),
    ]
    server, _ = _server(steps)

    with RequestFailureLog.capture() as log:
        await server.run_request(_request())

    assert {"step-1", "step-3"} <= log.report.events.keys()


async def test_without_max_episode_steps_the_agent_runs_its_own_loop_in_one_activation() -> None:
    server, client = _server([_reply(_tool_step(1))], max_episode_steps=None, agent={"max_steps": 50})

    result = await server.run_request(_request())

    assert result.result.reward == 1.0
    assert client.activations == 1
    assert f"GET {RESOURCES_TASK_STATE_PATH}" not in client.paths()
    assert client.activation_calls()[0]["headers"] == {AGENT_ACTIVATION_HEADER: "1"}


async def test_the_legacy_server_runs_a_flat_row_through_the_step_loop() -> None:
    server, client = _server(
        [_reply(_tool_step(1)), _reply(_tool_step(2, "user_complete_task")), _reply(_tool_step(3))], ended_after=2
    )
    legacy = SingleAgentTurnLegacyEnvironmentServer(config=server.config, server_client=client)

    result = await legacy.run_legacy(
        {"_ng_task_index": 0, "_ng_rollout_index": 0, "responses_create_params": {"input": "task"}}
    )

    assert (legacy.config.max_episode_steps, legacy.config.end_on_task_state) == (5, True)
    assert result["reward"] == 1.0
    assert [call["headers"] for call in client.activation_calls()] == [
        {AGENT_ACTIVATION_HEADER: str(step), AGENT_EPISODE_STEP_LIMIT_HEADER: "5"} for step in (1, 2)
    ]
    assert client.paths().count(f"GET {RESOURCES_TASK_STATE_PATH}") == 2


def test_driven_episodes_take_part_in_checkpoints_as_restarts() -> None:
    assert _server([])[0].checkpoint_mode_in_effect() == "restart_only"
    assert _server([], max_episode_steps=None, agent={"max_steps": 50})[0].checkpoint_mode_in_effect() == "exported"


@pytest.mark.parametrize(
    "agent",
    [
        {},
        {"max_steps": None},
        {"max_steps": 50},
        {"max_steps": "1"},
        {"execute_tools": True},
        {"execute_tools": "true"},
    ],
)
def test_a_simple_agent_that_runs_its_tools_is_driven_whatever_its_max_steps(agent: dict) -> None:
    # The step-limit header makes each activation one step, so the agent's own max_steps does not matter.
    server, _ = _server([], agent=agent)

    assert server.config.max_episode_steps == 5


@pytest.mark.parametrize(("agent", "warned"), [({}, False), ({"max_steps": None}, False), ({"max_steps": 150}, True)])
def test_a_max_steps_the_step_loop_does_not_use_is_named_at_startup(
    agent: dict, warned: bool, caplog: pytest.LogCaptureFixture
) -> None:
    # An override that raises the agent's own limit, as OfficeEnv's README once said to, loads but bounds nothing.
    with caplog.at_level(logging.WARNING, logger=single_agent_turn_app.__name__):
        _server([], agent=agent)

    warnings = [record.getMessage() for record in caplog.records if "max_steps" in record.getMessage()]
    assert warnings == (
        [
            "Environment Server 'environment' drives 'agent' one step per activation, so its max_steps (150) is not used: "
            "max_episode_steps (5) bounds the episode"
        ]
        if warned
        else []
    )


@pytest.mark.parametrize(
    ("agent", "agent_type", "problem"),
    [
        ({"execute_tools": False}, "simple_agent", "does not run the tools it calls (execute_tools: false)"),
        # As the agent's config reads the setting: from YAML or ${oc.env:...,false}, a string.
        ({"execute_tools": "false"}, "simple_agent", "does not run the tools it calls (execute_tools: false)"),
        ({"execute_tools": "False"}, "simple_agent", "does not run the tools it calls (execute_tools: false)"),
        ({}, "an_agent_without_sessions", "its type 'an_agent_without_sessions' keeps no agent session"),
        # Keeping a session is not taking one step per activation: these run their own loop in one.
        ({"max_steps": 1}, "hermes_agent", "its type 'hermes_agent' does not take one step per activation"),
        ({"max_steps": 1}, "browsecomp_agent", "its type 'browsecomp_agent' does not take one step per activation"),
    ],
)
def test_an_agent_that_cannot_take_one_step_per_activation_stops_the_server_at_startup(
    agent: dict, agent_type: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=f"drives 'agent' one step per activation .*{re.escape(problem)}"):
        _server([], agent=agent, agent_type=agent_type)


def test_an_agent_that_runs_its_own_loop_is_checked_only_when_driven_one_step_at_a_time() -> None:
    server, _ = _server([], max_episode_steps=None, agent_type="hermes_agent")

    assert server.config.max_episode_steps is None


def test_without_max_episode_steps_end_on_task_state_stops_the_agents_own_loop_instead() -> None:
    # tests/unit_tests/orchestration/test_single_agent_turn_task_end_stop.py covers that run.
    config = SingleAgentTurnEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "resources"},
        agent_server={"type": "responses_api_agents", "name": "agent"},
        cleanup_timeout_seconds=10,
        end_on_task_state=True,
    )
    assert (config.end_on_task_state, config.max_episode_steps) == (True, None)
