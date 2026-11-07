# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path
from typing import Any

import orjson
import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from environment_servers.officeenv.app import (
    TASK_STATE_PATH,
    OfficeEnvEnvironmentServer,
    OfficeEnvEnvironmentServerConfig,
)
from nemo_gym.agents import AGENT_ACTIVATION_HEADER
from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.server_runtime import SimpleServer


ROW = {"_ng_task_index": 3, "_ng_rollout_index": 0, "responses_create_params": {"input": "Close the ticket."}}
USAGE = {
    "input_tokens": 10,
    "output_tokens": 4,
    "total_tokens": 14,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 1},
}


def test_module_exports_app_for_multi_worker_import(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_app = object()
    monkeypatch.setattr("nemo_gym.utils.server_process.is_nemo_gym_fastapi_entrypoint", lambda _: True)
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda _: worker_app))

    namespace = runpy.run_path(str(Path(__file__).parents[1] / "app.py"), run_name="officeenv.worker_test")

    assert namespace["app"] is worker_app


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


def _tool_step(step: int, tool: str = "jira_add_comment") -> dict[str, Any]:
    """One step of the simple agent with max_steps 1: a tool call and its output."""
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


def _reply(step: dict[str, Any], status: str = "completed") -> dict[str, Any]:
    return {
        "id": "response",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "tools": [],
        "status": status,
        "usage": USAGE,
        **step,
    }


class _Client(ServerClient):
    """The agent and the OfficeEnv Resources Server: the agent replies with ``steps`` in order, and the task has
    ended from the step ``ended_after`` on."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, str, dict]]
    steps: list[dict]
    ended_after: int | None = None
    activations: int = 0
    task_state_error: Exception | None = None

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
            return _Response(self.steps[self.activations - 1])
        if url_path == "/v1/agent_sessions/close":
            return _Response({"agent_session_id": body["agent_session_id"]})
        if url_path == "/verify":
            return _Response({**body, "reward": 1.0})
        if url_path == "/close_session":
            return _Response({"resources_session_id": body["resources_session_id"]})
        raise AssertionError(f"unexpected POST {server_name}{url_path}")

    async def get(self, server_name: str, url_path: str, **kwargs) -> _Response:
        self.calls.append(("GET", server_name, url_path, kwargs))
        assert (server_name, url_path) == ("resources", TASK_STATE_PATH)
        if self.task_state_error is not None:
            raise self.task_state_error
        ended = self.ended_after is not None and self.activations >= self.ended_after
        return _Response({"task_ended": ended})

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"

    def activation_calls(self) -> list[dict]:
        return [kwargs for method, _, path, kwargs in self.calls if path.endswith("/v1/responses")]

    def paths(self) -> list[str]:
        return [f"{method} {path}" for method, _, path, _ in self.calls]


def _server(
    steps: list[dict], *, max_steps: int | None = 5, ended_after: int | None = None, agent_steps: Any = 1
) -> tuple[OfficeEnvEnvironmentServer, _Client]:
    agent = {"host": "agent", "port": 8001, "entrypoint": "app.py", "max_steps": agent_steps}
    global_config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"officeenv": {"host": "resources", "port": 8000}}},
            "agent": {"responses_api_agents": {"simple_agent": agent}},
        }
    )
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        calls=[],
        steps=steps,
        ended_after=ended_after,
    )
    config = OfficeEnvEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "resources"},
        agent_server={"type": "responses_api_agents", "name": "agent"},
        resources_tool_transports=["direct_http"],
        max_steps=max_steps,
    )
    return OfficeEnvEnvironmentServer(config=config, server_client=client), client


def _verified_response(client: _Client) -> dict:
    return next(kwargs["json"]["response"] for method, _, path, kwargs in client.calls if path == "/verify")


async def test_the_episode_ends_at_the_step_after_which_the_task_has_ended() -> None:
    server, client = _server(
        [_reply(_tool_step(1)), _reply(_tool_step(2, "user_complete_task")), _reply(_tool_step(3))]
    )
    client.ended_after = 2

    result = await server.run_legacy(ROW)

    assert result["reward"] == 1.0
    assert client.activations == 2
    assert client.paths() == [
        "POST /seed_session",
        "POST /v1/agent_sessions",
        "POST /ng-rollout/3-0/v1/responses",
        f"GET {TASK_STATE_PATH}",
        "POST /ng-rollout/3-0/v1/responses",
        f"GET {TASK_STATE_PATH}",
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


async def test_each_step_continues_the_conversation_with_the_steps_before_it() -> None:
    server, client = _server([_reply(_tool_step(1)), _reply(_tool_step(2)), _reply(_answer_step(3))])

    await server.run_legacy(ROW)

    activations = client.activation_calls()
    assert [call["headers"][AGENT_ACTIVATION_HEADER] for call in activations] == ["1", "2", "3"]
    inputs = [call["json"]["input"] for call in activations]
    assert [(item["role"], item["content"]) for item in inputs[0]] == [("user", "Close the ticket.")]
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

    await server.run_legacy(ROW)

    assert client.activations == 2
    # No tool ran in the last step, so the task state is not read after it.
    assert client.paths().count(f"GET {TASK_STATE_PATH}") == 1
    assert [item["type"] for item in _verified_response(client)["output"]] == [
        "function_call",
        "function_call_output",
        "message",
    ]


async def test_the_episode_ends_at_a_step_that_did_not_complete() -> None:
    server, client = _server([_reply(_tool_step(1), status="incomplete"), _reply(_tool_step(2))])

    await server.run_legacy(ROW)

    assert client.activations == 1
    assert _verified_response(client)["status"] == "incomplete"


async def test_the_episode_ends_after_max_steps() -> None:
    server, client = _server([_reply(_tool_step(step)) for step in range(1, 5)], max_steps=3)

    await server.run_legacy(ROW)

    assert client.activations == 3
    assert len(_verified_response(client)["output"]) == 6


async def test_the_task_state_is_read_with_the_resources_session_cookie() -> None:
    server, client = _server([_reply(_tool_step(1)), _reply(_answer_step(2))])

    await server.run_legacy(ROW)

    reads = [kwargs for method, _, _, kwargs in client.calls if method == "GET"]
    assert reads == [{"cookies": {"resources": "seeded"}}]


async def test_a_failed_task_state_read_fails_the_episode() -> None:
    server, client = _server([_reply(_tool_step(1)), _reply(_tool_step(2))])
    client.task_state_error = ConnectionError("resources server down")

    result = await server.run_legacy(ROW)

    assert result["_ng_failure_stage"] == "agent"
    assert "resources server down" in result["_ng_failure_message"]
    assert client.activations == 1
    assert "POST /verify" not in client.paths()


async def test_without_max_steps_the_agent_runs_its_own_loop_in_one_activation() -> None:
    server, client = _server([_reply(_tool_step(1))], max_steps=None, agent_steps=50)

    result = await server.run_legacy(ROW)

    assert result["reward"] == 1.0
    assert client.activations == 1
    assert f"GET {TASK_STATE_PATH}" not in client.paths()
    assert client.activation_calls()[0]["headers"] == {AGENT_ACTIVATION_HEADER: "1"}


@pytest.mark.parametrize("agent_steps", [None, 50, 2])
def test_the_server_refuses_an_agent_that_takes_several_steps_per_activation(agent_steps: Any) -> None:
    server, _ = _server([], agent_steps=agent_steps)

    with pytest.raises(ValueError, match="must take one step per activation"):
        server.setup_webserver()


def test_the_server_starts_with_a_one_step_agent_or_without_max_steps() -> None:
    _server([])[0].setup_webserver()
    _server([], max_steps=None, agent_steps=50)[0].setup_webserver()


def test_the_config_binds_the_simple_agent_one_step_per_activation() -> None:
    config = OmegaConf.load(Path(__file__).parents[3] / "resources_servers/officeenv/configs/officeenv.yaml")
    environment = config.officeenv_environment_server.environment_servers.officeenv
    agent = config.officeenv_simple_agent.responses_api_agents.simple_agent

    assert environment.agent_server.name == "officeenv_simple_agent"
    assert environment.resources_server.name == "officeenv_resources_server"
    assert list(environment.resources_tool_transports) == ["direct_http"]
    assert (environment.max_steps, agent.max_steps) == (50, 1)
    assert config.officeenv_resources_server.resources_servers.officeenv.close_tools_on_task_end is True
    OfficeEnvEnvironmentServerConfig.model_validate(
        {**OmegaConf.to_container(environment), "name": "officeenv", "host": "h", "port": 1}
    )
