# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.agent_utils.sandbox_session import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT
from nemo_gym.agent_utils.supervisor_client import parse_cleanup_receipt
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.tasks.harbor.materialize import materialize_task
from nemo_gym.tasks.harbor.task import load_task
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig, PiAgentRunRequest, _parse_recorded_events


def seed(*, session_id: str = "pi-session") -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id=session_id,
        episode_id=EpisodeId(rollout_id="pi-smoke", attempt=2),
        task_id=TaskId(taskset="swe-pro", task_id="task"),
        sandbox_access={
            "connection": {
                "kind": "direct",
                "provider_config_ref": "sandbox",
                "descriptor": {"sandbox_id": "resources-owned"},
            },
            "workdir": "/app",
        },
    )


def events(*, stop_reason="stop") -> str:
    return "\n".join(
        json.dumps([float(i), event])
        for i, event in enumerate(
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "responseId": "call-1",
                        "content": [
                            {"type": "thinking", "thinking": "Inspect the repository"},
                            {"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "pwd"}},
                        ],
                        "usage": {"input": 10, "output": 3, "cacheRead": 2},
                        "stopReason": "toolUse",
                    },
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": "tool-1",
                        "toolName": "bash",
                        "content": [{"type": "text", "text": "/app"}],
                    },
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "responseId": "call-2",
                        "content": [{"type": "text", "text": "Fixed"}],
                        "usage": {"input": 5, "output": 2, "cacheRead": 0},
                        "stopReason": stop_reason,
                        "errorMessage": "model error" if stop_reason == "error" else None,
                    },
                },
                {"type": "agent_end", "messages": [{"role": "assistant", "stopReason": stop_reason}]},
            ]
        )
    )


def staged(sandbox) -> dict:
    """The activation as the runner sees it: params.sh sourced by bash, plus the staged prompt."""
    script = 'source /dev/stdin; printf "%s\\0" "$cwd" "${#command[@]}" "${command[@]}" "${env[@]}"'
    params = sandbox.files[f"{sandbox.directory}/params.sh"]
    values = subprocess.run(
        ["bash", "-c", script], input=params, capture_output=True, text=True, check=True
    ).stdout.split("\0")[:-1]
    count = int(values[1])
    return {
        "cwd": values[0],
        "command": values[2 : 2 + count],
        "env": dict(item.split("=", 1) for item in values[2 + count :]),
        "prompt": sandbox.files[f"{sandbox.directory}/prompt.txt"],
    }


class Sandbox:
    def __init__(self):
        self.files = {}
        self.result = {
            "return_code": 0,
            "timed_out": False,
            "cleanup_confirmed": True,
            "error": None,
            "hostname": "task-container",
            "pid": 123,
        }
        self.events = events()
        self.blocked = False
        self.started = asyncio.Event()
        self.exited = asyncio.Event()
        self.exec = AsyncMock(side_effect=self.execute)
        self.stop = AsyncMock()
        self.disconnect = AsyncMock()
        self.launch = AsyncMock(side_effect=self.run)
        self.signal = AsyncMock(side_effect=self.stop_runner)
        self.cleanup_available = True

    async def execute(self, command, **kwargs):
        if "--receipt" in command and "process_supervisor.sh" in command:
            return await self.launch(command, **kwargs)
        if "ln -s stop " in command:
            await self.signal()
        return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")

    async def upload(self, source, destination):
        self.files[destination] = Path(source).read_text()

    async def download(self, source, destination):
        Path(destination).write_text(self.files[source])

    def save_result(self):
        if not self.cleanup_available:
            return
        self.files[f"{self.directory}/cleanup.json"] = json.dumps(
            {k: v for k, v in self.result.items() if k not in ("hostname", "pid")}
        )
        self.files[f"{self.directory}/runtime.json"] = json.dumps({k: self.result[k] for k in ("hostname", "pid")})
        self.files[f"{self.directory}/events.jsonl"] = self.events

    async def run(self, command, **kwargs):
        self.directory = next(path.rsplit("/", 1)[0] for path in self.files if path.endswith("/params.sh"))
        assert staged(self)["cwd"] == getattr(self, "expected_workdir", "/app")
        assert kwargs["cwd"] == getattr(self, "expected_workdir", "/app")
        assert f"bash {self.directory}/{HARNESS_RUNNER_FILE} {self.directory}/params.sh" in command
        assert "process_supervisor.sh" in command
        self.started.set()
        if not self.blocked:
            self.exited.set()
        await self.exited.wait()
        self.save_result()
        return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")

    async def stop_runner(self):
        self.directory = next(path.rsplit("/", 1)[0] for path in self.files if path.endswith("/params.sh"))
        self.result["timed_out"] = True
        self.save_result()
        self.exited.set()


@pytest.fixture
def setup():
    sandbox = Sandbox()
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {"policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}}
    )
    client._build_server_base_url.return_value = "http://model.example:9000"
    config = PiAgentConfig(
        name="pi",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        model_server={"type": "responses_api_models", "name": "policy"},
        model="test-model",
        pi_version="0.80.2",
        session_close_timeout_seconds=1,
    )
    module = "responses_api_agents.pi_agent.app"
    with (
        patch(f"{module}.ensure_pi", side_effect=AssertionError("sandbox sessions must not install host Pi")),
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider"),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
    ):
        agent = PiAgent(config=config, server_client=client)
        yield agent, sandbox


def close_body(session_id):
    return {"agent_session_id": session_id, "episode_id": seed().episode_id.model_dump()}


def mcp_grant(name, *, required=True):
    return {
        "kind": "mcp",
        "name": name,
        "required": required,
        "connection": {"url": "http://resources.example:7000/mcp", "headers": {"x-tool-session": f"token-{name}"}},
    }


def seed_with_tools(*accesses):
    return AgentSeedSessionRequest.model_validate({**seed().model_dump(mode="json"), "tool_accesses": list(accesses)})


def test_http_session_flow_runs_pi_in_borrowed_sandbox(setup):
    agent, sandbox = setup
    with patch.object(agent, "_run_pi", AsyncMock(side_effect=AssertionError("host Pi must not run"))):
        with TestClient(agent.setup_webserver()) as client:
            created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
            assert created.status_code == 200, created.text
            session_id = created.json()["agent_session_id"]
            directory = agent._session_records[session_id].state.session.session_dir
            assert f"{directory}/process_supervisor.sh" not in sandbox.files
            installer = f"{directory}/install_pi_runtime.sh"
            assert installer in sandbox.files
            assert agent.config.resources_server is None
            assert not sandbox.launch.called
            assert not any(path.startswith("/app/") for path in sandbox.files)
            install_call = sandbox.exec.await_args_list[1]
            assert install_call.args[0] == (f"bash {installer} /tmp/nemo-gym-pi-node-22.19.0-0.80.2 0.80.2")
            assert install_call.kwargs["timeout_s"] == agent.config.sandbox_install_timeout_seconds
            result = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code"})
            assert result.status_code == 200, result.text
            assert f"{directory}/process_supervisor.sh" in sandbox.files
            body = result.json()
            assert body["status"] == "completed"
            assert [item["type"] for item in body["output"]] == [
                "reasoning",
                "function_call",
                "function_call_output",
                "message",
            ]
            assert body["usage"]["total_tokens"] == 22
            assert body["usage"]["input_tokens_details"]["cached_tokens"] is None
            assert body["metadata"]["harness_execution"] == "sandbox"
            assert "_ng_agent_observations" not in body
            payload = staged(sandbox)
            assert payload["prompt"] == "Fix the code"
            assert payload["command"][0].endswith("/node/bin/node")
            assert "PATH" not in payload["env"]
            assert "HOME" not in payload["env"]
            assert not any(key.startswith("XDG_") for key in payload["env"])
            extension = f"{sandbox.directory}/output-limit.mjs"
            assert sandbox.files[extension] == Path(__file__).parents[1].joinpath("output-limit.mjs").read_text()
            assert payload["command"][payload["command"].index("--extension") + 1] == extension
            assert f"{sandbox.directory}/runtime-guards.mjs" in payload["command"]
            assert f"{sandbox.directory}/runtime-guards.mjs" in sandbox.files
            assert payload["env"]["NEMO_GYM_PI_BASH_TIMEOUT"] == "900"
            models = json.loads(sandbox.files[f"{sandbox.directory}/home/.pi/agent/models.json"])
            assert models["providers"]["nemo"]["models"][0]["maxTokens"] == agent.config.max_output_tokens
            assert models["providers"]["nemo"]["baseUrl"] == "http://model.example:9000/ng-rollout/pi-smoke-a2/v1"
            closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
            assert closed.status_code == 200, closed.text
            observations = closed.json()["agent_observations"]
            assert observations["source"] == "pi"
            assert "no_sandbox_runtime" not in [gap["code"] for gap in observations["gaps"]]
            assert len(observations["records"][0]["model_calls"]) == 2
    assert not any(record.state is not None for record in agent._session_records.values())
    assert agent._local_setup_task is None
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    agent.server_client.post.assert_not_called()


async def test_install_failure_preserves_stdout_and_stderr(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None, stdout="", stderr=""),
        SimpleNamespace(
            return_code=1, error_type=None, stdout="Node cannot load libstdc++.so.6", stderr="exit status 1"
        ),
        SimpleNamespace(return_code=0, error_type=None, stdout="", stderr=""),
    ]
    with pytest.raises(RuntimeError) as error:
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), seed())
    assert "exit status 1" in str(error.value)
    assert "Node cannot load libstdc++.so.6" in str(error.value)
    sandbox.disconnect.assert_awaited_once()
    sandbox.launch.assert_not_awaited()
    sandbox.stop.assert_not_awaited()
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize("stage", ["prepare", "install"])
@pytest.mark.parametrize("error_type", ["timeout", "sandbox"])
@pytest.mark.parametrize("owned", [False, True])
async def test_provider_error_type_blocks_setup_even_with_zero_exit(setup, stage, error_type, owned):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}})
    body = seed()
    if owned:
        agent.config.sandbox_provider = "sandbox"
        body = body.model_copy(update={"sandbox_access": None})
        sandbox.start = AsyncMock()
    ok = SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
    error = SimpleNamespace(return_code=0, error_type=error_type, stdout="bootstrap output", stderr="provider failed")
    sandbox.exec.side_effect = ([ok] if owned else []) + ([error, ok] if stage == "prepare" else [ok, error, ok])
    with patch("responses_api_agents.pi_agent.app.AsyncSandbox", return_value=sandbox) as sandbox_class:
        sandbox_class.connect = AsyncMock(return_value=sandbox)
        with pytest.raises(RuntimeError, match=f"error_type={error_type}") as failed:
            await agent.seed_agent_session(request, body)
    assert "bootstrap output" in str(failed.value)
    assert "provider failed" in str(failed.value)
    assert not request.session
    sandbox.launch.assert_not_awaited()
    if owned:
        sandbox.start.assert_awaited_once()
        sandbox.stop.assert_awaited_once()
    else:
        sandbox.disconnect.assert_awaited_once()
        sandbox.stop.assert_not_awaited()
    assert not any(record.state is not None for record in agent._session_records.values())


@pytest.mark.parametrize("limit", [0, -1, 128, 2**53])
def test_invalid_output_limit_does_not_consume_activation(setup, limit):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        seeded = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert seeded.status_code == 200
        response = client.post(
            "/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code", "max_output_tokens": limit}
        )
        assert response.status_code == 422
        assert next(iter(agent._session_records.values())).state.task is None
        sandbox.launch.assert_not_awaited()
        response = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code"})
        assert response.status_code == 200


def test_session_output_limit_uses_config_default(setup):
    agent, sandbox = setup
    agent.config.max_output_tokens = 4096
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        assert client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code"}).status_code == 200
        models = json.loads(sandbox.files[f"{sandbox.directory}/home/.pi/agent/models.json"])
        assert models["providers"]["nemo"]["models"][0]["maxTokens"] == 4096


def test_invalid_session_config_output_limit_does_not_consume_activation(setup):
    agent, sandbox = setup
    agent.config.max_output_tokens = 0
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        response = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code"})
        assert response.status_code == 422
        assert next(iter(agent._session_records.values())).state.task is None
        sandbox.launch.assert_not_awaited()


def test_direct_run_without_resources_rejected_before_execution(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/run", json={"responses_create_params": {"input": "task"}})
    assert response.status_code == 422
    assert "use EnvironmentServer /run" in response.json()["detail"]
    sandbox.exec.assert_not_awaited()
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize("option", ["no-sandbox", "required-tool", "no-model", "unpinned"])
def test_unsupported_seed_rejected_before_connection(setup, option):
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    if option == "no-sandbox":
        body["sandbox_access"] = None
    elif option == "no-model":
        agent.config.model_server = None
    elif option == "unpinned":
        agent.config.pi_version = "latest"
    else:
        body["tool_accesses"] = [
            {"kind": "direct_http", "name": "tools", "base_url": "http://resources", "required": True}
        ]
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=body).status_code == 422
    sandbox.exec.assert_not_awaited()


# gym_mcp.mjs builds tool names from the grant name and allows at most 64 characters.
_UNUSABLE_NAME = "its name is not 1 to 62 letters, digits, '-' and '_'"
_OVERLAP = "its tools could not be told apart from those of grant"


@pytest.mark.parametrize(
    "tool_accesses,detail",
    [
        ([mcp_grant("resources.mcp")], f"Pi cannot use these required MCP grants: resources.mcp ({_UNUSABLE_NAME})"),
        ([mcp_grant("r" * 63)], f"Pi cannot use these required MCP grants: {'r' * 63} ({_UNUSABLE_NAME})"),
        (
            # `resources_v2_finish` could be either grant's tool.
            [mcp_grant("resources"), mcp_grant("resources_v2")],
            "Pi cannot use these required MCP grants: "
            f"resources ({_OVERLAP} resources_v2); resources_v2 ({_OVERLAP} resources)",
        ),
    ],
)
def test_required_grants_pi_cannot_use_are_refused_at_seed(setup, tool_accesses, detail):
    agent, sandbox = setup
    body = {**seed().model_dump(mode="json"), "tool_accesses": tool_accesses}
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=body)
    assert response.status_code == 422
    assert response.json()["detail"] == detail
    sandbox.exec.assert_not_awaited()


def test_several_workers_seed_sessions(setup):
    """Session routing sends a session's later requests to the worker that seeded it."""
    agent, sandbox = setup
    agent.config.num_workers = 2
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        assert created.json()["agent_session_id"] in agent._session_records


def test_cookie_identity_and_single_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        assert client.post("/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/ng-rollout/wrong-a2/v1/responses", json={"input": "task"}).status_code == 409
        bad = close_body(session_id)
        bad["episode_id"]["attempt"] = 99
        assert client.post("/v1/agent_sessions/close", json=bad).status_code == 409
        first = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "task"})
        retry = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "task"})
        assert first.status_code == retry.status_code == 200
        assert first.json() == retry.json()
        assert client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "changed"}).status_code == 409
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.launch.assert_awaited_once()


@pytest.mark.parametrize("reason,expected", [("aborted", "incomplete"), ("length", "incomplete")])
def test_failed_or_partial_pi_output_is_preserved(setup, reason, expected):
    agent, sandbox = setup
    sandbox.events = events(stop_reason=reason)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200
        assert result.json()["status"] == expected
        assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200


@pytest.mark.parametrize(
    "override",
    [
        {"temperature": 0.2},
        {"top_p": 0.9},
        {"tools": [{"type": "function", "name": "foo"}]},
        {"input": [{"role": "assistant", "content": "old turn"}]},
    ],
)
def test_unsupported_request_is_not_silently_ignored(setup, override):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "task", **override})
        assert result.status_code == 422, result.text
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("return_code", [1, 137])
@pytest.mark.parametrize("timed_out", [False, True])
def test_aborted_preserves_worker_failure_unless_deadline_confirmed(setup, return_code, timed_out):
    agent, sandbox = setup
    sandbox.events = events(stop_reason="aborted")
    sandbox.result.update(return_code=return_code, timed_out=timed_out)
    app = agent.setup_webserver()
    agent.setup_exception_middleware(app)
    with TestClient(app) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "task"})
        if timed_out:
            assert result.status_code == 200
            assert result.json()["status"] == "incomplete"
            assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        else:
            assert result.status_code == 500
            assert f"Pi exited with code {return_code}" in result.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        assert closed.json()["agent_observations"]["records"]
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_rejected_request_does_not_consume_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/pi-smoke-a2/v1/responses"
        assert client.post(path, json={"input": "task", "temperature": 0.2}).status_code == 422
        sandbox.launch.assert_not_awaited()
        accepted = client.post(path, json={"input": "task"})
        assert accepted.status_code == 200, accepted.text
        assert client.post(path, json={"input": "changed"}).status_code == 409
    sandbox.launch.assert_awaited_once()


def test_no_session_keeps_existing_local_path(setup):
    agent, sandbox = setup
    with patch.object(agent, "_create_episode", AsyncMock(side_effect=RuntimeError("legacy path reached"))) as legacy:
        with TestClient(agent.setup_webserver()) as client:
            with pytest.raises(RuntimeError, match="legacy path reached"):
                client.post("/v1/responses", json={"input": "task"})
        legacy.assert_awaited_once()
    sandbox.launch.assert_not_awaited()


async def activate(agent, sandbox, body=None):
    request = Request({"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": "pi-smoke-a2"}})
    seeded = await agent.seed_agent_session(request, body or seed())
    task = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.wait_for(sandbox.started.wait(), 2)
    return request, seeded.agent_session_id, task


async def test_sandbox_runtime_guards_reach_the_pi_invocation(setup):
    agent, sandbox = setup
    agent.config.sandbox_bash_timeout_seconds = 123
    agent.config.timeout = 10800
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    assert response.status == "completed"
    payload = staged(sandbox)
    assert payload["env"]["NEMO_GYM_PI_BASH_TIMEOUT"] == "123"
    assert f"{sandbox.directory}/runtime-guards.mjs" in payload["command"]
    assert "tool_call" in sandbox.files[f"{sandbox.directory}/runtime-guards.mjs"]
    settings = json.loads(sandbox.files[f"{sandbox.directory}/home/.pi/agent/settings.json"])
    assert settings["httpIdleTimeoutMs"] == 10800000
    assert settings["retry"]["provider"]["timeoutMs"] == 10800000
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_mcp_grants_reach_pi_through_the_gym_extension(setup):
    agent, sandbox = setup
    agent.config.sandbox_mcp_timeout_seconds = 120
    agent.config.finish_tool_names = ["finish"]
    body = seed_with_tools(
        mcp_grant("resources"),
        mcp_grant("web-extra", required=False),
        # Pi has no direct HTTP client, and an optional grant may be left unused.
        {"kind": "direct_http", "name": "unused", "base_url": "http://resources", "required": False},
    )
    request, session_id, task = await activate(agent, sandbox, body)
    response = await task
    assert response.status == "completed"
    payload = staged(sandbox)
    extension = f"{sandbox.directory}/gym_mcp.mjs"
    assert sandbox.files[extension] == Path(__file__).parents[1].joinpath("gym_mcp.mjs").read_text()
    assert payload["command"][payload["command"].index(extension) - 1] == "--extension"
    assert payload["command"].index(extension) < payload["command"].index("--no-skills")
    assert payload["env"]["NEMO_GYM_PI_MCP_CONFIG"] == f"{sandbox.directory}/mcp-servers.json"
    servers = json.loads(sandbox.files[f"{sandbox.directory}/mcp-servers.json"])
    assert servers == {
        "resources": {
            "url": "http://resources.example:7000/mcp",
            "headers": {"x-tool-session": "token-resources"},
            "timeout": 120000,
            "required": True,
            "terminal_tools": ["finish"],
        },
        "web-extra": {
            "url": "http://resources.example:7000/mcp",
            "headers": {"x-tool-session": "token-web-extra"},
            "timeout": 120000,
            "required": False,
            "terminal_tools": ["finish"],
        },
    }
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


async def test_optional_mcp_grant_that_pi_cannot_use_is_left_out(setup):
    agent, sandbox = setup
    name, problem = "resources.mcp", _UNUSABLE_NAME
    body = seed_with_tools(mcp_grant("resources"), mcp_grant(name, required=False))
    request, session_id, task = await activate(agent, sandbox, body)
    assert (await task).status == "completed"
    assert list(json.loads(sandbox.files[f"{sandbox.directory}/mcp-servers.json"])) == ["resources"]
    close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    gaps = [(gap.code, gap.detail) for gap in close.agent_observations.gaps]
    assert ("mcp_server_skipped", f"{name}: {problem}") in gaps


async def test_no_mcp_grant_stages_no_tool_extension(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    payload = staged(sandbox)
    assert not any(path.endswith("gym_mcp.mjs") for path in [*payload["command"], *sandbox.files])
    assert "NEMO_GYM_PI_MCP_CONFIG" not in payload["env"]
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


@pytest.mark.parametrize("granted", [True, False])
async def test_mcp_setup_exit_fails_the_episode_with_its_cause(setup, granted):
    agent, sandbox = setup
    sandbox.result["return_code"] = 78
    cause = "resources: TypeError (ECONNREFUSED)"
    sandbox.events = json.dumps([0.0, {"type": "ng_gym_mcp_setup_failed", "error": cause}])
    request, session_id, task = await activate(
        agent, sandbox, seed_with_tools(mcp_grant("resources")) if granted else None
    )
    message = (
        f"Required Gym MCP tools could not be initialized: {re.escape(cause)}" if granted else "Pi exited with code 78"
    )
    with pytest.raises(RuntimeError, match=message):
        await task
    close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    if granted:
        # Episode failure rows carry the close observations, so the cause reaches them too.
        assert ("mcp_setup_failed", cause) in [(gap.code, gap.detail) for gap in close.agent_observations.gaps]


def finish_events(*, details, is_error=False, tool="resources_finish", with_bash=False):
    """A run whose last assistant turn calls a granted MCP tool, as when gym_mcp.mjs ends the run."""
    last_calls = [{"type": "toolCall", "id": "tool-2", "name": tool, "arguments": {"paths": ["/r.docx"]}}]
    last_results = [
        {
            "role": "toolResult",
            "toolCallId": "tool-2",
            "toolName": tool,
            "content": [{"type": "text", "text": "done"}],
            "details": details,
            "isError": is_error,
        }
    ]
    if with_bash:
        last_calls.append({"type": "toolCall", "id": "tool-3", "name": "bash", "arguments": {"command": "ls"}})
        last_results.append(
            {
                "role": "toolResult",
                "toolCallId": "tool-3",
                "toolName": "bash",
                "content": [{"type": "text", "text": "r.docx"}],
                "details": {},
                "isError": False,
            }
        )
    messages = [
        {
            "role": "assistant",
            "responseId": "call-1",
            "content": [{"type": "toolCall", "id": "tool-1", "name": "bash", "arguments": {"command": "ls"}}],
            "usage": {"input": 10, "output": 3, "cacheRead": 2},
            "stopReason": "toolUse",
        },
        {
            "role": "toolResult",
            "toolCallId": "tool-1",
            "toolName": "bash",
            "content": [{"type": "text", "text": "report.docx"}],
            "details": {},
            "isError": False,
        },
        {
            "role": "assistant",
            "responseId": "call-2",
            "content": last_calls,
            "usage": {"input": 5, "output": 2, "cacheRead": 0},
            "stopReason": "toolUse",
        },
        *last_results,
    ]
    recorded = [*({"type": "message_end", "message": message} for message in messages)]
    recorded.append({"type": "agent_end", "messages": messages})
    return "\n".join(json.dumps([float(i), event]) for i, event in enumerate(recorded))


async def test_finish_tool_ends_the_run_and_output_uses_gym_mcp_names(setup):
    agent, sandbox = setup
    sandbox.events = finish_events(details={"terminal": True})
    grants = seed_with_tools(mcp_grant("web"), mcp_grant("resources"))
    request, session_id, task = await activate(agent, sandbox, grants)
    response = await task
    assert response.status == "completed"
    assert response.error is None
    # The granted prefix becomes Gym's MCP naming, and Pi's own tools keep their names.
    assert [item.name for item in response.output if item.type == "function_call"] == [
        "bash",
        "mcp__resources__finish",
    ]
    close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    # Observations keep the names the model saw and record the run as completed.
    assert "resources_finish" in json.dumps(close.agent_observations.model_dump(mode="json"))
    assert [record.status for record in close.agent_observations.records if record.kind == "agent_invocation"] == [
        "completed"
    ]
    assert "invocation_outcome_unavailable" not in [gap.code for gap in close.agent_observations.gaps]


@pytest.mark.parametrize(
    "details,is_error,with_bash",
    [
        ({}, False, False),
        ({"terminal": True}, True, False),
        ({"terminal": True}, False, True),
    ],
)
async def test_run_ending_on_other_tool_calls_is_not_a_result(setup, details, is_error, with_bash):
    agent, sandbox = setup
    sandbox.events = finish_events(details=details, is_error=is_error, with_bash=with_bash)
    request, session_id, task = await activate(agent, sandbox, seed_with_tools(mcp_grant("resources")))
    with pytest.raises(RuntimeError, match="Pi ended without a terminal assistant result"):
        await task
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


async def test_skipped_optional_mcp_server_is_an_observation_gap(setup):
    agent, sandbox = setup
    skipped = {"type": "ng_gym_mcp_server_skipped", "server": "extra", "error": "TypeError (ECONNREFUSED)"}
    sandbox.events = json.dumps([0.0, skipped]) + "\n" + events()
    request, session_id, task = await activate(agent, sandbox, seed_with_tools(mcp_grant("extra", required=False)))
    assert (await task).status == "completed"
    close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    gaps = [(gap.code, gap.detail) for gap in close.agent_observations.gaps]
    assert ("mcp_server_skipped", "extra: TypeError (ECONNREFUSED)") in gaps


async def test_interrupted_run_keeps_the_skipped_mcp_server_gaps(setup):
    agent, sandbox = setup
    sandbox.blocked = True
    skipped = {"type": "ng_gym_mcp_server_skipped", "server": "extra", "error": "TypeError (ECONNREFUSED)"}
    sandbox.events = json.dumps([0.0, skipped])
    grants = seed_with_tools(mcp_grant("extra", required=False), mcp_grant("extra.mcp", required=False))
    request, session_id, task = await activate(agent, sandbox, grants)
    close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    gaps = [(gap.code, gap.detail) for gap in close.agent_observations.gaps]
    assert ("mcp_server_skipped", "extra: TypeError (ECONNREFUSED)") in gaps
    assert ("mcp_server_skipped", f"extra.mcp: {_UNUSABLE_NAME}") in gaps


@pytest.mark.parametrize("value", [0, 0.0004, 2_147_484])
def test_mcp_timeout_stays_within_node_timer_range(value):
    with pytest.raises(ValidationError):
        PiAgentConfig(name="pi", host="localhost", port=1, entrypoint="app.py", sandbox_mcp_timeout_seconds=value)


async def test_close_cancels_active_pi_before_detaching(setup):
    agent, sandbox = setup
    sandbox.blocked = True
    request, session_id, task = await activate(agent, sandbox)
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    sandbox.signal.assert_awaited_once()
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_terminal_termination_failure_releases_borrowed_session(setup):
    agent, sandbox = setup
    sandbox.result["cleanup_confirmed"] = False
    request, session_id, task = await activate(agent, sandbox)
    with pytest.raises(RuntimeError, match="harness termination was not confirmed"):
        await task
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert agent._session_records[session_id].state is None
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_disconnect_failure_retains_session_for_retry(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    sandbox.disconnect.side_effect = [RuntimeError("provider unavailable"), None]
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert agent._session_records[session_id].state is not None
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert agent._session_records[session_id].state is None


@pytest.mark.parametrize("log_available", [False, True])
async def test_missing_events_use_shared_session_log_without_blocking_close(setup, log_available):
    agent, sandbox = setup
    download = sandbox.download
    log_reads = []

    async def download_artifact(source, destination):
        if source.endswith("/events.jsonl"):
            raise FileNotFoundError("events unavailable")
        if source.endswith("/output.log"):
            log_reads.append(source)
            if not log_available:
                raise FileNotFoundError("log unavailable")
            Path(destination).write_text("diagnostic from shared supervisor")
            return
        await download(source, destination)

    sandbox.download = AsyncMock(side_effect=download_artifact)
    request, session_id, task = await activate(agent, sandbox)
    with pytest.raises(RuntimeError, match="Pi sandbox runner returned no valid result") as failure:
        await task
    assert ("diagnostic from shared supervisor" in str(failure.value)) is log_available
    assert isinstance(failure.value.__cause__, FileNotFoundError)
    state = agent._session_records[session_id].state
    assert log_reads == [f"{state.session.session_dir}/output.log"]
    assert state.session.cleanup["cleanup_confirmed"] is True
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_cleanup_receipt_is_required():
    with pytest.raises(ValueError):
        parse_cleanup_receipt({"return_code": 0, "error": None})


@pytest.mark.parametrize("artifact", ["missing_runtime", "invalid_runtime", "unknown_exit", "runtime_overrides_exit"])
async def test_worker_diagnostics_never_override_exit_or_cleanup(setup, artifact):
    agent, sandbox = setup
    download = sandbox.download
    if artifact == "unknown_exit":
        sandbox.result["return_code"] = None
    elif artifact == "runtime_overrides_exit":
        sandbox.result["return_code"] = 7

    async def download_artifact(source, destination):
        if source.endswith("/runtime.json"):
            if artifact == "missing_runtime":
                raise FileNotFoundError(source)
            if artifact == "invalid_runtime":
                Path(destination).write_text(json.dumps({"hostname": "sandbox", "pid": "invalid"}))
                return
            if artifact == "runtime_overrides_exit":
                Path(destination).write_text(json.dumps({"hostname": "sandbox", "pid": 123, "return_code": 0}))
                return
        await download(source, destination)

    sandbox.download = AsyncMock(side_effect=download_artifact)
    request, session_id, task = await activate(agent, sandbox)
    if artifact in ("missing_runtime", "invalid_runtime", "unknown_exit"):
        response = await task
        assert response.status == "completed"
        if artifact != "unknown_exit":
            assert "harness_hostname" not in response.metadata
    else:
        with pytest.raises(RuntimeError, match="exited with code 7"):
            await task
    state = agent._session_records[session_id].state
    assert state.session.cleanup["cleanup_confirmed"] is True
    if artifact != "unknown_exit":
        assert state.runtime_info is None
        assert any(gap.code == "runtime_info_unavailable" for gap in state.observations.gaps)
    if artifact == "unknown_exit":
        assert state.session.cleanup["return_code"] is None
        assert any(gap.code == "worker_exit_code_unavailable" for gap in state.observations.gaps)
    elif artifact == "runtime_overrides_exit":
        assert state.session.cleanup["return_code"] == 7
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert agent._session_records[session_id].state is None
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_instructions_and_text_parts_reach_pi_without_other_provider_credentials(setup):
    agent, sandbox = setup
    agent.config.system_prompt = "config instruction"
    agent.config.models_config = {"providers": {"unrelated": {"apiKey": "must-not-copy"}}}
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post(
            "/ng-rollout/pi-smoke-a2/v1/responses",
            json={
                "instructions": "request instruction",
                "input": [
                    {"role": "system", "content": "input instruction"},
                    {"role": "user", "content": [{"type": "input_text", "text": "task"}]},
                ],
            },
        )
        assert response.status_code == 200, response.text
        payload = staged(sandbox)
        assert payload["command"][-2:] == [
            "--append-system-prompt",
            "config instruction\n\nrequest instruction\n\ninput instruction",
        ]
        assert payload["prompt"] == "task"
        models = json.loads(sandbox.files[f"{sandbox.directory}/home/.pi/agent/models.json"])
        assert list(models["providers"]) == ["nemo"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        conversation = closed.json()["agent_observations"]["records"][0]["conversation"]
        assert conversation[0]["content"] == payload["command"][-1]


def test_observation_parse_failure_preserves_response(setup):
    agent, _ = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        with patch("responses_api_agents.pi_agent.app._build_pi_observations", side_effect=ValueError("bad event")):
            response = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        assert closed.json()["agent_observations"]["gaps"][0]["code"] == "observation_parse_failed"


def test_close_retry_and_stale_activation_do_not_run_host_pi(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "task"}).raise_for_status()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        repeated = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert repeated.status_code == 200
        assert repeated.json() == closed.json()
        bad = close_body(session_id)
        bad["episode_id"]["attempt"] = 99
        assert client.post("/v1/agent_sessions/close", json=bad).status_code == 409
        with patch.object(agent, "_run_pi", AsyncMock(side_effect=AssertionError("host Pi must not run"))):
            assert client.post("/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 409
        client.cookies.clear()
        replacement = seed().model_copy(update={"agent_session_id": "replacement-session"})
        assert client.post("/v1/agent_sessions", json=replacement.model_dump(mode="json")).status_code == 200
    sandbox.disconnect.assert_awaited_once()


async def test_concurrent_closes_share_receipt(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    entered, release = asyncio.Event(), asyncio.Event()

    async def disconnect():
        entered.set()
        await release.wait()

    sandbox.disconnect.side_effect = disconnect
    close = AgentCloseSessionRequest(**close_body(session_id))
    first = asyncio.create_task(agent.close_agent_session(request, close))
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(agent.close_agent_session(request, close))
    await asyncio.sleep(0)
    release.set()
    first_result, second_result = await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert first_result == second_result
    assert first_result is not second_result
    sandbox.disconnect.assert_awaited_once()


async def test_unknown_launch_outcome_fails_closed(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "pi-smoke-a2"}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    sandbox.launch.side_effect = TimeoutError("lost launch response")
    sandbox.cleanup_available = False
    with pytest.raises(TimeoutError):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert agent._session_records[session_id].state is not None
    sandbox.disconnect.assert_not_awaited()
    sandbox.stop.assert_not_awaited()


async def test_install_failure_disconnects_without_stopping_owner(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        SimpleNamespace(return_code=1, error_type=None, stderr="npm failed", stdout=""),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="npm failed"):
        await agent.seed_agent_session(request, seed())
    assert not any(record.state is not None for record in agent._session_records.values())
    assert not request.session
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_failed_connection_closes_provider_without_publishing_session(setup, error_type):
    agent, sandbox = setup
    provider = SimpleNamespace(aclose=AsyncMock())
    request = Request({"type": "http", "session": {}})
    module = "responses_api_agents.pi_agent.app"
    with (
        patch(f"{module}.create_provider", return_value=provider),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(side_effect=error_type("connection failed"))),
        pytest.raises(error_type),
    ):
        await agent.seed_agent_session(request, seed())
    provider.aclose.assert_awaited_once()
    assert not request.session
    assert not any(record.state is not None for record in agent._session_records.values())
    sandbox.exec.assert_not_awaited()
    sandbox.stop.assert_not_awaited()


async def test_invalid_runtime_metadata_keeps_valid_output_and_confirmed_cleanup(setup):
    agent, sandbox = setup
    sandbox.result["pid"] = "invalid-pid"
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    assert response.status == "completed"
    assert "harness_pid" not in response.metadata
    state = agent._session_records[session_id].state
    assert state.runtime_info is None and state.session.cleanup["cleanup_confirmed"] is True
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert agent._session_records[session_id].state is None
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_cancelled_install_never_publishes_session_or_launches_pi(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        asyncio.CancelledError(),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(asyncio.CancelledError):
        await agent.seed_agent_session(request, seed())
    assert not any(record.state is not None for record in agent._session_records.values())
    assert not request.session
    sandbox.launch.assert_not_awaited()
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize(
    "control",
    [
        {"model": "different-model"},
        {"include": ["reasoning.encrypted_content"]},
        {"store": True},
        {"service_tier": "priority"},
        {"prompt_cache_key": "cache"},
        {"prompt_cache_retention": "24h"},
        {"safety_identifier": "caller"},
        {"stream_options": {"include_obfuscation": True}},
        {"user": "caller"},
    ],
)
def test_unsupported_controls_do_not_consume_activation(setup, control):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200
        response = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code", **control})
        assert response.status_code == 422, response.text
        assert next(iter(agent._session_records.values())).state.task is None
        sandbox.launch.assert_not_awaited()
        response = client.post(
            "/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code", "model": "test-model"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "completed"


def sandbox_event_response(agent, sandbox, recorded_events):
    sandbox.events = "\n".join(json.dumps([i, event]) for i, event in enumerate(recorded_events))
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200
        response = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code"})
        assert response.status_code == 200, response.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    return response.json(), closed.json()["agent_observations"]


@pytest.mark.parametrize("final_stop, expected", [("stop", "completed"), ("length", "incomplete")])
def test_retry_uses_terminal_assistant_outcome(setup, final_stop, expected):
    agent, sandbox = setup
    initial = {
        "role": "assistant",
        "responseId": "retry-1",
        "content": [],
        "usage": {"input": 0, "output": 0, "cacheRead": 0},
        "stopReason": "error",
        "errorMessage": "503 Service unavailable",
    }
    final = {
        "role": "assistant",
        "responseId": "retry-2",
        "content": [{"type": "text", "text": "Result"}],
        "usage": {"input": 5, "output": 2, "cacheRead": 0},
        "stopReason": final_stop,
        "errorMessage": "retry exhausted" if final_stop == "error" else None,
    }
    response, observations = sandbox_event_response(
        agent,
        sandbox,
        [
            {"type": "message_end", "message": initial},
            {"type": "agent_end", "messages": [initial], "willRetry": True},
            {
                "type": "auto_retry_start",
                "attempt": 1,
                "maxAttempts": 3,
                "delayMs": 1,
                "errorMessage": initial["errorMessage"],
            },
            {"type": "message_end", "message": final},
            {"type": "auto_retry_end", "success": final_stop != "error", "attempt": 1},
            {"type": "agent_end", "messages": [final], "willRetry": False},
        ],
    )
    assert response["status"] == expected
    assert response["error"] == (
        {"code": "server_error", "message": "retry exhausted"} if final_stop == "error" else None
    )
    assert response["output"][-1]["content"][0]["text"] == "Result"
    assert response["usage"]["total_tokens"] == 7
    assert observations["records"][0]["status"] == expected
    assert len(observations["records"][0]["model_calls"]) == 2


def test_multiturn_message_ids_are_unique_and_tool_ids_preserved(setup):
    agent, sandbox = setup
    recorded = [json.loads(line)[1] for line in events().splitlines()]
    recorded[0]["message"]["content"].insert(1, {"type": "text", "text": "Inspecting"})
    response, _ = sandbox_event_response(agent, sandbox, recorded)
    output = response["output"]
    messages = [item for item in output if item["type"] == "message"]
    assert [item["content"][0]["text"] for item in messages] == ["Inspecting", "Fixed"]
    ids = [item["id"] for item in output if "id" in item]
    assert len(ids) == len(set(ids))
    assert [item["call_id"] for item in output if item["type"].startswith("function_call")] == ["tool-1", "tool-1"]


@pytest.mark.parametrize(
    "cache_read, expected", [(0, None), (3, 5), (None, None), (-1, None), ("3", None), ("invalid", None), (True, None)]
)
def test_optional_usage_details_preserve_unknown_contributors(setup, cache_read, expected):
    agent, sandbox = setup
    recorded = [json.loads(line)[1] for line in events().splitlines()]
    final_usage = recorded[2]["message"]["usage"]
    if cache_read is None:
        final_usage.pop("cacheRead")
    else:
        final_usage["cacheRead"] = cache_read
    response, observations = sandbox_event_response(agent, sandbox, recorded)
    assert response["usage"]["input_tokens_details"]["cached_tokens"] == expected
    gaps = {gap["code"] for gap in observations["gaps"]}
    assert ("cached_token_usage_unavailable" in gaps) is (expected is None)
    assert "reasoning_token_usage_unavailable" in gaps
    assert response["usage"]["input_tokens"] == (20 if cache_read == 3 and type(cache_read) is int else 17)
    assert response["usage"]["output_tokens"] == 5
    assert response["usage"]["output_tokens_details"]["reasoning_tokens"] is None
    assert response["output"][0]["type"] == "reasoning"


def test_defaulted_cache_zero_remains_unknown(setup):
    agent, sandbox = setup
    recorded = [json.loads(line)[1] for line in events().splitlines()]
    for event in recorded:
        message = event.get("message", {})
        if message.get("role") == "assistant":
            message["usage"]["cacheRead"] = 0
    response, _ = sandbox_event_response(agent, sandbox, recorded)
    assert response["usage"]["input_tokens_details"]["cached_tokens"] is None


@pytest.mark.parametrize("benchmark", ["swebench_pro", "independent"])
async def test_default_config_collects_through_environment_run(setup, monkeypatch, benchmark):
    """One Pi config works with SWE-bench and an unrelated Resources contract."""
    from environment_servers.single_agent_turn.app import (
        SingleAgentTurnEnvironmentServer,
        SingleAgentTurnEnvironmentServerConfig,
    )
    from nemo_gym.config import GlobalConfigDictParser
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
    from nemo_gym.server_utils import BaseServerConfig
    from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest

    agent, sandbox = setup
    root = Path(__file__).parents[3]
    parser = GlobalConfigDictParser()
    config_paths = [
        root / "responses_api_agents/pi_agent/configs/pi_agent.yaml",
        root / "environment_servers/single_agent_turn/configs/single_agent_turn.yaml",
    ]
    if benchmark == "swebench_pro":
        config_paths.append(root / "resources_servers/swebench_pro/configs/swebench_pro.yaml")
        resources_name = "swebench_pro_resources_server"
        taskset = "swebench_pro"
        task_data = {"instance_id": "instance"}
    else:
        resources_name = "workspace_resources"
        taskset = "workspace_fixture"
        task_data = {"expected_output": "done"}
    _, configs = parser.load_extra_config_paths([str(path) for path in config_paths])
    config = OmegaConf.merge(*configs)
    config.environment_routing_mode = "taskset"
    config.environment_server_routes = {taskset: "single_agent_turn"}
    environment_config = config.single_agent_turn.environment_servers.single_agent_turn
    environment_config.resources_server.name = resources_name
    environment_config.agent_server.name = "pi_agent"
    parser._recursively_swap_keys(config)
    assert config.environment_routing_mode == "taskset"
    environment_name = config.environment_server_routes[taskset]
    environment_config = config[environment_name].environment_servers.single_agent_turn
    agent_name = environment_config.agent_server.name
    resources_name = environment_config.resources_server.name
    assert config[agent_name].responses_api_agents.pi_agent.resources_server is None
    config[agent_name].responses_api_agents.pi_agent.model = "test-model"
    agent.config = PiAgentConfig(
        name=agent_name,
        host="localhost",
        port=8001,
        **OmegaConf.to_container(config[agent_name].responses_api_agents.pi_agent, resolve=True),
    )
    config.policy_model = {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}
    transport = ServerClient(head_server_config=BaseServerConfig(host="head", port=1), global_config_dict=config)
    agent.server_client = transport
    environment = SingleAgentTurnEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name=environment_name,
            host="localhost",
            port=8002,
            **OmegaConf.to_container(environment_config, resolve=True),
        ),
        server_client=transport,
    )
    calls = []
    cookies = {}

    class Response:
        ok = True

        def __init__(self, body, *, cookie=None):
            self.body = json.dumps(body).encode()
            self.cookies = {} if cookie is None else {"session": SimpleNamespace(value=cookie)}
            self.headers = {}

        async def read(self):
            return self.body

    async def post(self, server_name, url_path, **kwargs):
        body = kwargs["json"]
        calls.append((server_name, url_path))
        if server_name == environment_name:
            assert url_path == "/run"
            result = await environment.run_request(SingleAgentTurnRequest.model_validate(body))
            return Response(result.model_dump(mode="json"))
        if server_name == resources_name:
            if url_path == "/seed_session":
                assert body["task_data"] == task_data
                return Response(
                    {
                        "resources_session_id": body["resources_session_id"],
                        "sandbox_access": seed().sandbox_access.model_dump(),
                    },
                    cookie="resources-cookie",
                )
            assert kwargs["cookies"] == {"session": "resources-cookie"}
            if url_path == "/verify":
                assert not any(record.state is not None for record in agent._session_records.values())
                assert sandbox.disconnect.await_count == 1
                for key, value in task_data.items():
                    assert body[key] == value
                assert body["responses_create_params"]["input"] == "Fix it"
                assert body["response"]["usage"]["total_tokens"] == 22
                assert "verification_input" not in body
                return Response({**body, "reward": 1.0})
            assert url_path == "/close_session"
            return Response({"resources_session_id": body["resources_session_id"]})
        assert server_name == agent_name
        request = Request({"type": "http", "session": cookies})
        if url_path == "/v1/agent_sessions":
            result = await agent.seed_agent_session(request, AgentSeedSessionRequest.model_validate(body))
            assert result.agent_session_id == body["agent_session_id"]
        elif url_path == "/v1/agent_sessions/close":
            result = await agent.close_agent_session(request, AgentCloseSessionRequest.model_validate(body))
        else:
            assert url_path.endswith("/v1/responses")
            request.scope["path_params"] = {"rollout_id": url_path.split("/")[2]}
            result = await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming.model_validate(body))
        return Response(result.model_dump(mode="json"), cookie="agent-cookie")

    monkeypatch.setattr(ServerClient, "post", post)
    monkeypatch.setattr(ServerClient, "_resolve_base_url", lambda self, name: f"http://{name}:8000")
    monkeypatch.setattr(RolloutCollectionHelper, "setup_server_client", lambda self, head=None: transport)
    materialized = {
        "task_id": {"taskset": taskset, "task_id": "instance"},
        "task_input": {"responses_create_params": {"input": "Fix it"}, "task_data": task_data},
    }
    collection_config = RolloutCollectionConfig(
        input_jsonl_fpath="input.jsonl",
        output_jsonl_fpath="output.jsonl",
        environment_routing_mode=config.environment_routing_mode,
        environment_server_routes=OmegaConf.to_container(config.environment_server_routes),
        num_repeats=1,
    )
    rows = RolloutCollectionHelper._preprocess_raw_rows(
        [(0, json.dumps(materialized), materialized)], collection_config
    )
    _, result = await next(RolloutCollectionHelper().run_examples(rows))
    assert result["failure"] is None
    assert result["result"]["reward"] == 1.0
    assert result["result"]["response"]["usage"]["total_tokens"] == 22
    assert result["result"]["ng_agent_observations"]["source"] == "pi"
    assert calls == [
        (environment_name, "/run"),
        (resources_name, "/seed_session"),
        (agent_name, "/v1/agent_sessions"),
        (agent_name, "/ng-rollout/0-0/v1/responses"),
        (agent_name, "/v1/agent_sessions/close"),
        (resources_name, "/verify"),
        (resources_name, "/close_session"),
    ]
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failed_setup_preserves_handle_until_cleanup_confirmed(setup, cleanup_fails):
    agent, sandbox = setup
    ok = SimpleNamespace(error_type=None, return_code=0, stdout="", stderr="")
    failed = SimpleNamespace(error_type=None, return_code=1, stdout="", stderr="install failed")
    sandbox.exec.side_effect = [ok, failed, ok]
    if cleanup_fails:
        sandbox.disconnect.side_effect = RuntimeError("disconnect failed")
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="install failed"):
        await agent.seed_agent_session(request, seed())
    session_id = seed().agent_session_id
    if cleanup_fails:
        assert agent._session_records[session_id].state.session.closing
        with pytest.raises(HTTPException):
            await agent.seed_agent_session(Request({"type": "http", "session": {}}), seed())
    else:
        assert not any(record.state is not None for record in agent._session_records.values())
    sandbox.exec.side_effect = sandbox.execute
    sandbox.disconnect.side_effect = None
    result = await agent.close_agent_session(
        Request({"type": "http", "session": {}}), AgentCloseSessionRequest(**close_body(session_id))
    )
    assert result.agent_session_id == session_id
    assert not any(record.state is not None for record in agent._session_records.values())
    sandbox.stop.assert_not_awaited()


async def test_caller_id_never_controls_filesystem_path(setup):
    agent, _ = setup
    body = seed(session_id="../../task-repository\nunsafe")
    request = Request({"type": "http", "session": {}})
    response = await agent.seed_agent_session(request, body)
    assert response.agent_session_id == body.agent_session_id
    directory = agent._session_records[body.agent_session_id].state.session.session_dir
    assert Path(directory).parent == Path("/tmp/nemo-gym-pi-sessions")
    assert len(Path(directory).name) == 32
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


@pytest.mark.parametrize("marker", [None, "", [], {}, 0])
@pytest.mark.parametrize("endpoint", ["seed", "responses", "close", "run"])
async def test_malformed_session_marker_never_runs_host_pi(setup, marker, endpoint):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {"agent_session_id": marker}})
    with patch.object(agent, "_create_episode", AsyncMock()) as host:
        with pytest.raises(HTTPException, match="Invalid agent session marker"):
            if endpoint == "seed":
                await agent.seed_agent_session(request, seed())
            elif endpoint == "responses":
                await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
            elif endpoint == "close":
                await agent.close_agent_session(
                    request, AgentCloseSessionRequest(**close_body(seed().agent_session_id))
                )
            else:
                await agent.run(request, PiAgentRunRequest(responses_create_params={"input": "task"}))
        host.assert_not_awaited()
    sandbox.exec.assert_not_awaited()
    sandbox.launch.assert_not_awaited()


async def test_session_marker_blocks_legacy_run(setup):
    agent, _ = setup
    request = Request({"type": "http", "session": {"agent_session_id": "expired-session"}})
    with pytest.raises(HTTPException, match="EnvironmentServer /run"):
        await agent.run(request, PiAgentRunRequest(responses_create_params={"input": "task"}))


async def test_cancelled_http_waiter_does_not_cancel_shared_activation(setup):
    agent, sandbox = setup
    sandbox.blocked = True
    request, session_id, waiter = await activate(agent, sandbox)
    state = agent._session_records[session_id].state
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not state.task.done()
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    retry = asyncio.create_task(agent.responses(request, body))
    sandbox.exited.set()
    response = await asyncio.wait_for(retry, 2)
    response.output.clear()
    replay = await agent.responses(request, body)
    assert replay.output
    sandbox.launch.assert_awaited_once()
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


@pytest.mark.parametrize(
    "control",
    [
        {"max_output_tokens": 128},
        {"temperature": 0},
        {"top_p": 1},
        {"metadata": {"ignored": "value"}},
        {"store": True},
    ],
)
async def test_sandbox_rejects_unsupported_controls(setup, control):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "pi-smoke-a2"}})
    await agent.seed_agent_session(request, seed())
    with patch.object(agent, "_run_pi", AsyncMock()) as run:
        with pytest.raises(HTTPException) as error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task", **control))
        assert error.value.status_code == 422
        run.assert_not_awaited()
        sandbox.launch.assert_not_awaited()


async def test_local_instructions_match_sandbox_prompt(setup):
    agent, _ = setup
    agent.config.system_prompt = "config"
    body = NeMoGymResponseCreateParamsNonStreaming(
        instructions="request", input=[{"role": "system", "content": "input"}, {"role": "user", "content": "task"}]
    )
    with patch.object(agent, "_run_pi", AsyncMock(return_value=([], {}, "test-model", []))) as run:
        await agent.responses(Request({"type": "http", "session": {}}), body)
    assert run.call_args.args == ("task", "config\n\nrequest\n\ninput")


async def _run_locally(command, *, limit=8, **kwargs):
    """Run a sandbox command as a local process group, killed if it outlives `limit` seconds."""
    process = await asyncio.create_subprocess_exec(
        "sh",
        "-c",
        command,
        cwd=kwargs.get("cwd"),
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=limit)
        return SimpleNamespace(
            return_code=process.returncode,
            error_type=None,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )
    finally:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()


def _run_session_locally(sandbox, state, directory: Path, *, limit: float = 8) -> None:
    """Point a seeded session's files and commands at local paths and processes."""
    state.session.session_dir = str(directory)
    shutil.copyfile(HARNESS_RUNNER_SCRIPT, directory / HARNESS_RUNNER_FILE)
    sandbox.upload = AsyncMock(side_effect=shutil.copyfile)
    sandbox.download = AsyncMock(side_effect=shutil.copyfile)
    sandbox.exec.side_effect = functools.partial(_run_locally, limit=limit)


@pytest.fixture
async def local_session(setup, tmp_path):
    """Exercise the adapter's actual launch and stop commands against real Linux processes."""
    agent, sandbox = setup
    body = seed()
    body.sandbox_access.workdir = str(tmp_path)
    await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    state = agent._session_records[body.agent_session_id].state
    directory = tmp_path / "session"
    directory.mkdir()
    _run_session_locally(sandbox, state, directory)
    return state, _run_locally


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox contract")
@pytest.mark.parametrize("failure", ["cancel", "failed-before-spawn"])
async def test_close_fences_delayed_launch_before_and_after_removing_session(local_session, failure):
    state, execute = local_session
    waiting = asyncio.Event()
    commands = []

    async def queued_exec(command, **kwargs):
        if "--receipt" in command and "process_supervisor.sh" in command:
            commands.append(command)
            waiting.set()
            if failure == "cancel":
                await asyncio.Event().wait()
            raise OSError("lost launch response")
        return await execute(command, **kwargs)

    state.session.sandbox.exec.side_effect = queued_exec
    state.task = asyncio.create_task(state.execute(command=["true"], env={}, prompt="", timeout=5, close_timeout=2))
    await asyncio.wait_for(waiting.wait(), 2)
    if failure == "cancel":
        state.task.cancel()
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else OSError):
        await state.task
    directory = Path(state.session.session_dir)
    assert state.session.cleanup["cleanup_confirmed"] is True
    assert (directory / "launch.claim").readlink() == Path("stop")
    assert (await execute(commands[0])).return_code == 0
    assert not (directory / "supervisor.pid").exists()
    await state.close(2)
    assert not directory.exists()
    assert (await execute(commands[0])).return_code == 0
    assert not directory.exists()
    state.session.sandbox.disconnect.assert_awaited_once()
    state.session.sandbox.stop.assert_not_awaited()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox contract")
async def test_launch_claim_without_receipt_blocks_close(local_session):
    state, _ = local_session
    (Path(state.session.session_dir) / "launch.claim").symlink_to("launch")
    state.session.launch_started = True
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await state.close(1)
    state.session.sandbox.disconnect.assert_not_awaited()
    assert state.session.cleanup is None


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox contract")
async def test_adapter_uses_shared_supervisor_and_captures_real_events(local_session):
    state, _ = local_session
    # Diagnostics and malformed lines reach stdout too; only [seen_at, object] records reach the transcript.
    script = 'echo starting; echo "{\\"prompt\\": \\"$(cat)\\", \\"cwd\\": \\"$PWD\\"}"; echo "{broken"'
    raw = await state.execute(
        command=["bash", "-c", script], env={}, prompt="real invocation", timeout=3, close_timeout=2
    )
    diagnostics = {
        name: path.read_text(errors="replace")
        for name in ("cleanup.json", "runtime.json", "output.log", "stderr.log", "events.jsonl")
        if (path := Path(state.session.session_dir) / name).exists()
    }
    assert state.session.cleanup["cleanup_confirmed"] and state.session.cleanup["return_code"] == 0, diagnostics
    assert raw.strip(), diagnostics
    assert "{broken" in raw
    events = _parse_recorded_events(raw)
    assert [event for _, event in events] == [{"prompt": "real invocation", "cwd": state.session.workdir}]
    assert state.runtime_info.hostname
    await state.close(2)
    assert not Path(state.session.session_dir).exists()
    state.session.sandbox.stop.assert_not_awaited()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox contract")
async def test_close_confirms_cleanup_before_provider_cancellation(local_session):
    state, _ = local_session
    processes_path = Path(state.session.session_dir).parent / "children.json"
    code = (
        "import json,os,subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        f"Path({str(processes_path)!r}).write_text(json.dumps([os.getpid(),child.pid])); time.sleep(60)"
    )
    state.task = asyncio.create_task(
        state.execute(command=[sys.executable, "-c", code], env={}, prompt="", timeout=30, close_timeout=3)
    )
    processes = []
    try:
        async with asyncio.timeout(5):
            while not processes_path.exists():
                await asyncio.sleep(0.01)
        processes = json.loads(processes_path.read_text())
        processes.append(json.loads((Path(state.session.session_dir) / "runtime.json").read_text())["pid"])
        await state.close(3)
        assert state.session.closed and state.session.cleanup["cleanup_confirmed"] is True
        cli_pid, task_pid, wrapper_pid = processes
        assert not Path(f"/proc/{cli_pid}").exists()
        assert not Path(f"/proc/{wrapper_pid}").exists()
        # The in-flight detached tool is stopped before borrowed-session release; init may still hold it as a zombie.
        stat = Path(f"/proc/{task_pid}/stat")
        assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] in "ZX"
        state.session.sandbox.disconnect.assert_awaited_once()
        state.session.sandbox.stop.assert_not_awaited()
    finally:
        # This fixture uses host processes rather than a real container owner.
        for pid in processes:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not state.task.done():
            state.task.cancel()
        await asyncio.gather(state.task, return_exceptions=True)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox contract")
async def test_receipt_read_failure_does_not_signal_reused_pid(local_session):
    state, _ = local_session
    unrelated = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(60)")
    directory = Path(state.session.session_dir)
    (directory / "launch.claim").symlink_to("launch")
    (directory / "supervisor.pid").write_text(str(unrelated.pid))
    (directory / "cleanup.json").write_text(json.dumps({"cleanup_confirmed": True, "error": None}))
    state.session.launch_started = True
    downloads = 0

    async def download(source, destination):
        nonlocal downloads
        downloads += 1
        if downloads == 1:
            raise OSError("transient receipt download failure")
        shutil.copyfile(source, destination)

    state.session.sandbox.download.side_effect = download
    try:
        await state.session.stop_harness(timeout=2)
        assert state.session.cleanup["cleanup_confirmed"] is True
        assert unrelated.returncode is None
        os.kill(unrelated.pid, 0)
    finally:
        if unrelated.returncode is None:
            unrelated.terminate()
        await unrelated.wait()


def _pinned_pi_package() -> tuple[str, Path]:
    """Node and the installed Pi package, when it is the version the shipped config pins (as CI installs)."""
    config = OmegaConf.load(Path(__file__).parents[1] / "configs/pi_agent.yaml")
    pinned = config.pi_agent.responses_api_agents.pi_agent.pi_version
    node, pi = shutil.which("node"), shutil.which("pi")
    if not node or not pi:
        pytest.skip("Installed Pi and Node are required for the real Pi contract")
    package = Path(pi).resolve().parents[1]
    manifest = package / "package.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("version") != pinned:
        pytest.skip(f"This integration targets the pinned Pi {pinned}")
    return node, package


@contextmanager
def _http_server(handle):
    """Serve POST bodies on 127.0.0.1 with ``handle(body) -> (status, content_type, payload)``."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            status, content_type, payload = handle(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _finish_tool_server(calls):
    """A stateless JSON MCP endpoint, as Gym's Resources servers expose, that records `finish` calls."""
    finish = {
        "name": "finish",
        "description": "Submit the deliverables",
        "inputSchema": {"type": "object", "properties": {"paths": {"type": "array"}}, "required": ["paths"]},
    }

    def handle(body):
        if body["method"] == "notifications/initialized":
            return 202, "application/json", b""
        if body["method"] == "tools/call":
            calls.append(body["params"]["arguments"])
        result = {
            "initialize": {"protocolVersion": "2025-03-26", "capabilities": {}},
            "tools/list": {"tools": [finish]},
            "tools/call": {"content": [{"type": "text", "text": "Submitted"}]},
        }[body["method"]]
        return 200, "application/json", json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()

    return handle


def _finishing_model(requests, replies):
    """A streaming chat model that calls `resources_finish` with each reply in turn, then answers in text.

    A reply is (arguments, finish_reason, cut_off); a cut-off reply reports the request's whole output budget.
    """

    def handle(body):
        requests.append(body)

        def chunk(delta, finish=None, choices=None, usage=None):
            data = {"id": f"chat-{len(requests)}", "object": "chat.completion.chunk", "created": 0}
            data["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}] if choices is None else choices
            data |= {"model": body.get("model"), **({"usage": usage} if usage else {})}
            return f"data: {json.dumps(data)}\n\n"

        arguments, finish_reason, cut_off = (
            (*replies[len(requests) - 1],) if len(requests) <= len(replies) else 3 * (None,)
        )
        if arguments is None:
            stream = chunk({"role": "assistant", "content": "Done."}) + chunk({}, "stop")
        else:
            call = {"index": 0, "id": f"call-{len(requests)}", "type": "function"}
            call["function"] = {"name": "resources_finish", "arguments": arguments}
            stream = chunk({"role": "assistant", "tool_calls": [call]}) + chunk({}, finish_reason)
        output = body["max_tokens"] if cut_off else 5
        stream += chunk(
            None, choices=[], usage={"prompt_tokens": 10, "completion_tokens": output, "total_tokens": 10 + output}
        )
        return 200, "text/event-stream", f"{stream}data: [DONE]\n\n".encode()

    return handle


def _long_turn_then_overflow(requests, summary_status=200):
    """A streaming chat model whose first reply reasons for 100 KB and calls `bash`.

    Every later request fails with a context-length 400, except Pi's compaction requests, which carry no tools: they
    get a summary, or `summary_status` with no body.
    """

    def handle(body):
        requests.append(body)
        if "tools" not in body:
            if summary_status != 200:
                return summary_status, "application/json", b""
            delta, finish = {"role": "assistant", "content": "Summary."}, "stop"
        elif len(requests) > 1:
            error = {"error": {"message": "This model's maximum context length is 262144 tokens."}}
            return 400, "application/json", json.dumps(error).encode()
        else:
            call = {"index": 0, "id": "call-1", "type": "function"}
            call["function"] = {"name": "bash", "arguments": json.dumps({"command": "true"})}
            delta = {"role": "assistant", "reasoning_content": "r" * 100_000, "tool_calls": [call]}
            finish = "tool_calls"
        chunk = {"id": "chat-1", "object": "chat.completion.chunk", "created": 0, "model": body.get("model")}
        chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
        return 200, "text/event-stream", f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()

    return handle


@asynccontextmanager
async def _pinned_pi_activation(setup, tmp_path, monkeypatch, *, grants, replies=(), model=None):
    """One activation of the pinned Pi through the real runner and supervisor, with local MCP and model servers.

    `grants(up, down)` builds the episode's grants from a working MCP URL and one that refuses connections.
    The model server runs the handler `model(run.requests)` if given, or else `_finishing_model` with `replies`.
    Yields the activation to await; close runs on exit, and its response is left on the yielded namespace.
    """
    agent, sandbox = setup
    node, package = _pinned_pi_package()
    # Pi sends fetch, including gym_mcp.mjs's, through proxies set in the environment.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    runtime = tmp_path / "runtime"
    (runtime / "node/bin").mkdir(parents=True)
    (runtime / "node/bin/node").symlink_to(node)
    (runtime / "pi/node_modules/@earendil-works").mkdir(parents=True)
    (runtime / "pi/node_modules/@earendil-works/pi-coding-agent").symlink_to(package)
    agent.config.finish_tool_names = ["finish"]
    workdir = tmp_path / "work"
    workdir.mkdir()
    run = SimpleNamespace(requests=[], calls=[], close=None)
    with (
        socket.socket() as refused,
        _http_server(_finish_tool_server(run.calls)) as up,
        _http_server(model(run.requests) if model else _finishing_model(run.requests, replies)) as model_url,
    ):
        refused.bind(("127.0.0.1", 0))  # Bound but not listening, so connections to it are refused.
        run.down = f"http://127.0.0.1:{refused.getsockname()[1]}"
        agent.server_client._build_server_base_url.return_value = model_url
        body = seed_with_tools(*grants(f"{up}/mcp", f"{run.down}/mcp"))
        body.sandbox_access.workdir = str(workdir)
        request = Request({"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": "pi-smoke-a2"}})
        await agent.seed_agent_session(request, body)
        state = agent._session_records[body.agent_session_id].state
        state.runtime = str(runtime)
        directory = tmp_path / "session"
        (directory / "home/.pi/agent").mkdir(parents=True)
        # A hung run outlasts the activation wait before the local exec limit, so close can stop it through
        # its supervisor (whose stop handshake polls once a second) and leave no processes behind.
        agent.config.session_close_timeout_seconds = 10
        _run_session_locally(sandbox, state, directory, limit=60)
        run.activation = asyncio.wait_for(
            agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="Write the report.")), 30
        )
        try:
            yield run
        finally:
            close_request = AgentCloseSessionRequest(**close_body(body.agent_session_id))
            run.close = await agent.close_agent_session(request, close_request)


def _url_grant(name, url, *, required=True):
    return {**mcp_grant(name, required=required), "connection": {"url": url}}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox contract")
@pytest.mark.parametrize("case", ["finish", "finish-cut-at-output-limit", "finish-cut-as-vllm-reports-it"])
async def test_pinned_pi_ends_the_run_on_finish(setup, tmp_path, monkeypatch, case):
    paths = ["/app/report.docx", "/app/data.xlsx"]
    replies = [(json.dumps({"paths": paths}), "tool_calls", False)]
    if case.startswith("finish-cut"):
        # Pi parses the cut-off arguments as `{"paths": ["/app/report.docx"]}`; the call must not run. A server may
        # report `length` (here under a lower limit than the request's), while vLLM reports `tool_calls` for a reply
        # it parsed a call from, even when the reply used its whole output budget.
        vllm = case == "finish-cut-as-vllm-reports-it"
        replies.insert(0, ('{"paths": ["/app/report.docx", ', "tool_calls" if vllm else "length", vllm))

    def grants(up, down):
        return [_url_grant("resources", up)]

    async with _pinned_pi_activation(setup, tmp_path, monkeypatch, grants=grants, replies=replies) as run:
        response = await run.activation
    assert response.status == "completed", response.error
    assert "resources_finish" in [tool["function"]["name"] for tool in run.requests[0]["tools"]]
    # The server records only the complete finish call, and no model call follows it.
    assert run.calls == [{"paths": paths}]
    assert len(run.requests) == len(replies)
    assert [item.name for item in response.output if item.type == "function_call"] == ["mcp__resources__finish"] * len(
        replies
    )
    # Pi's agent_end carries the terminal tool result, so the invocation record agrees with the response.
    records = run.close.agent_observations.records
    assert [record.status for record in records if record.kind == "agent_invocation"] == ["completed"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox contract")
async def test_pinned_pi_stops_before_inference_when_a_required_grant_is_down(setup, tmp_path, monkeypatch):
    replies = [(json.dumps({"paths": ["/app/report.docx"]}), "tool_calls", False)]

    def grants(up, down):
        return [_url_grant("resources", down)]

    async with _pinned_pi_activation(setup, tmp_path, monkeypatch, grants=grants, replies=replies) as run:
        cause = f"resources: TypeError (ECONNREFUSED) at {run.down}"
        with pytest.raises(RuntimeError, match=re.escape(f"MCP tools could not be initialized: {cause}")):
            await run.activation
    assert run.requests == run.calls == []
    assert ("mcp_setup_failed", cause) in [(gap.code, gap.detail) for gap in run.close.agent_observations.gaps]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox contract")
@pytest.mark.parametrize(
    ("summary_status", "requests_with_tools"), [(200, [True, True, False, True]), (503, [True, True, False])]
)
async def test_pinned_pi_reports_a_context_overflow_after_a_long_turn(
    setup, tmp_path, monkeypatch, summary_status, requests_with_tools
):
    # Pi is still writing the long turn's lines when the 400 arrives, so outcome.mjs's event must not
    # split one of them or be lost. Gym grades the final 400 as a model limit only when that event reaches it.
    # With compaction on, Pi first compacts and retries; a failed compaction call is reported as the run's failure.
    model = functools.partial(_long_turn_then_overflow, summary_status=summary_status)
    async with _pinned_pi_activation(setup, tmp_path, monkeypatch, grants=lambda up, down: [], model=model) as run:
        response = await run.activation
    # Pi's compaction request is the one without tools; Pi retries the overflowed call only after it worked.
    assert ["tools" in request for request in run.requests] == requests_with_tools
    # The long turn ran and reached Gym whole; without it the pipe never fills.
    assert [item.type for item in response.output] == ["reasoning", "function_call", "function_call_output"]
    if summary_status == 200:
        assert (response.status, response.error) == ("incomplete", None)
    else:
        assert response.status == "failed"
        assert re.match(r"Context overflow recovery failed: .*\b503\b", response.error.message)
        # Not the overflow's 400, which the default policy would grade.
        assert json.loads(response.metadata["ng_agent_outcome"]) == {"failure_kind": "unknown"}


@pytest.mark.parametrize("field", ["timeout", "sandbox_install_timeout_seconds", "session_close_timeout_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_pi_rejects_invalid_execution_deadlines(field, value):
    with pytest.raises(ValidationError):
        PiAgentConfig(name="pi", host="localhost", port=1, entrypoint="app.py", **{field: value})


_COMPACTION_CALL_FAILED = "Context overflow recovery failed: Summarization failed: 503 status code (no body)"


@pytest.mark.parametrize(
    ("context_overflow", "compaction", "error"),
    [
        (False, [], "model error"),
        (True, [], None),
        # Pi 0.80.2's events after the overflow: it compacted, its retry overflowed again, and it gave up.
        (
            True,
            [
                {"type": "compaction_start", "reason": "overflow"},
                {"type": "compaction_end", "reason": "overflow", "result": {"summary": "Summary"}, "willRetry": True},
                {
                    "type": "compaction_end",
                    "reason": "overflow",
                    "errorMessage": "Context overflow recovery failed after one compact-and-retry attempt.",
                },
            ],
            None,
        ),
        # Its compaction call failed, so the overflowed call was never retried.
        (
            True,
            [
                {"type": "compaction_start", "reason": "overflow"},
                {"type": "compaction_end", "reason": "overflow", "errorMessage": _COMPACTION_CALL_FAILED},
            ],
            _COMPACTION_CALL_FAILED,
        ),
        # Pi also compacts after a no-body 400, which Gym does not count as an overflow.
        (
            False,
            [
                {"type": "compaction_start", "reason": "overflow"},
                {"type": "compaction_end", "reason": "overflow", "errorMessage": _COMPACTION_CALL_FAILED},
            ],
            _COMPACTION_CALL_FAILED,
        ),
        # A threshold compaction that fails after a non-overflow error must not replace that error.
        (
            False,
            [
                {"type": "compaction_start", "reason": "threshold"},
                {
                    "type": "compaction_end",
                    "reason": "threshold",
                    "errorMessage": "Auto-compaction failed: Summarization failed: 503 status code (no body)",
                },
            ],
            "model error",
        ),
    ],
)
async def test_provider_failure_is_reported_but_context_limit_preserves_patch(
    setup, context_overflow, compaction, error
):
    agent, sandbox = setup
    recorded = [json.loads(line)[1] for line in events(stop_reason="error").splitlines()]
    recorded.insert(-2, {"type": "ng_pi_outcome", "context_overflow": context_overflow})
    if context_overflow:
        # Pi's overflow error on the agent_end that Gym classifies. The default policy grades this 400.
        recorded[-1]["messages"][0]["errorMessage"] = "400 This model's maximum context length is 262144 tokens."
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate([*recorded, *compaction]))
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    if error is None:
        assert response.status == "incomplete"
        assert response.error is None
        assert response.output[-1].content[0].text == "Fixed"
    else:
        assert (response.status, response.error.message) == ("failed", error)
        assert json.loads(response.metadata["ng_agent_outcome"]) == {"failure_kind": "unknown"}
        # Replay the report without running the harness again; the environment owns policy.
        assert await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")) == response
    close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert len(close.agent_observations.records[0].model_calls) == 2
    assert close.agent_observations.records[0].status == ("failed" if error else "incomplete")
    sandbox.launch.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("interrupted", [False, True])
async def test_session_retry_ownership_uses_provider_correlation(setup, interrupted) -> None:
    agent, sandbox = setup
    failure = {
        "type": "message_end",
        "message": {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": "429 retry"},
    }
    recorded = [failure, {"type": "auto_retry_start", "attempt": 1}, failure]
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(recorded))
    if not interrupted:
        sandbox.events += "\n" + events()
    sandbox.blocked = interrupted
    request, session_id, task = await activate(agent, sandbox)
    if interrupted:
        close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert (await task).status == "completed"
        close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    provider = json.loads(sandbox.files[f"{sandbox.directory}/home/.pi/agent/models.json"])["providers"]["nemo"]
    assert provider["headers"] == {"x-session-id": "pi-smoke-a2"}
    root = close.agent_observations.records[0]
    assert root.invocation_id == provider["headers"]["x-session-id"]
    gaps = [gap.code for gap in close.agent_observations.gaps]
    assert "model_call_ownership_unavailable" not in gaps
    if interrupted:
        from nemo_gym.observability.model_calls import ModelCallRecord
        from nemo_gym.rollout_observability import join_agent_model_calls

        assert "agent_activation_interrupted" in gaps
        assert root.status == "incomplete"
        assert root.model_calls == []  # Rejected attempts have no provider response IDs.
        captured = [
            ModelCallRecord(model_call_id=f"attempt-{i}", client_session_id=root.invocation_id, call_index=i)
            for i in range(2)
        ]
        joined = join_agent_model_calls(close.agent_observations, captured)
        assert [ref.model_call_id for ref in joined.records[0].model_calls] == ["attempt-0", "attempt-1"]
        assert all(ref.response_id is None for ref in joined.records[0].model_calls)
        assert "model_call_ownership_unavailable" not in [gap.code for gap in joined.gaps]
    else:
        assert len(root.model_calls) == 2


@pytest.mark.parametrize(
    "error_message",
    ["400 rejected request", "401 invalid key", "429 exhausted retries", "503 unavailable", "Connection error."],
)
@pytest.mark.parametrize("prior_turn", [False, True])
@pytest.mark.parametrize("partial_output", [False, True])
async def test_terminal_model_error_reports_partial_output_and_usage(
    setup: tuple[PiAgent, Sandbox], prior_turn: bool, partial_output: bool, error_message: str
) -> None:
    agent, sandbox = setup
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "partial answer"}] if partial_output else [],
        "stopReason": "error",
        "errorMessage": error_message,
        "usage": {"input": 5 if partial_output else 0, "output": 2 if partial_output else 0, "cacheRead": 0},
    }
    recorded = []
    if prior_turn:
        recorded = [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "responseId": "earlier-response",
                    "content": [{"type": "text", "text": "earlier work"}],
                    "stopReason": "stop",
                    "usage": {"input": 7, "output": 3, "cacheRead": 2},
                },
            }
        ]
    recorded.extend(
        [
            {"type": "message_end", "message": message},
            {"type": "agent_end", "messages": [message]},
        ]
    )
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(recorded))
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    assert response.status == "failed"
    assert response.error.message == error_message
    expected_text = (["earlier work"] if prior_turn else []) + (["partial answer"] if partial_output else [])
    assert [item.content[0].text for item in response.output] == expected_text
    assert response.usage.input_tokens == (9 if prior_turn else 0) + (5 if partial_output else 0)
    assert response.usage.output_tokens == (3 if prior_turn else 0) + (2 if partial_output else 0)
    assert response.usage.total_tokens == (12 if prior_turn else 0) + (7 if partial_output else 0)
    close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    root = close.agent_observations.records[0]
    assert root.status == "failed"
    assert [call.response_id for call in root.model_calls] == (["earlier-response"] if prior_turn else [])
    assert len(root.conversation) == 1 + len(expected_text)  # No fabricated assistant turn for an empty rejection.
    sandbox.launch.assert_awaited_once()


@pytest.mark.parametrize(
    "error_message,return_code,ended,timed_out,expected_error",
    [
        ("400 rejected request", 1, True, False, "Pi exited with code 1"),
        ("400 rejected request", 137, True, False, "Pi exited with code 137"),
        ("400 rejected request", None, True, False, "400 rejected request"),
        ("400 rejected request", 0, False, False, "400 rejected request"),
        ("400 rejected request", 0, True, True, "400 rejected request"),
    ],
)
async def test_model_error_does_not_hide_incomplete_execution(
    setup: tuple[PiAgent, Sandbox],
    error_message: str,
    return_code: int | None,
    ended: bool,
    timed_out: bool,
    expected_error: str,
) -> None:
    agent, sandbox = setup
    message = {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": error_message}
    recorded = [{"type": "message_end", "message": message}]
    if ended:
        recorded.append({"type": "agent_end", "messages": [message]})
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(recorded))
    sandbox.result.update(return_code=return_code, timed_out=timed_out)
    request, session_id, task = await activate(agent, sandbox)
    with pytest.raises(RuntimeError, match=expected_error):
        await task
    close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert close.agent_observations.records
    sandbox.launch.assert_awaited_once()


async def test_model_rejection_does_not_hide_cleanup_failure(setup: tuple[PiAgent, Sandbox]) -> None:
    agent, sandbox = setup
    recorded = [json.loads(line)[1] for line in events(stop_reason="error").splitlines()]
    recorded[-2]["message"]["errorMessage"] = "400 rejected request"
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(recorded))
    sandbox.result["error"] = "cleanup failed"
    request, session_id, task = await activate(agent, sandbox)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await task
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


@pytest.mark.parametrize("owned", [False, True])
def test_sandbox_source_controls_ownership_and_session_routing(setup, owned):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/agent-workspace"}
    sandbox.expected_workdir = "/agent-workspace" if owned else "/app"
    body = seed()
    if owned:
        body.sandbox_access = None
    sandbox.start = AsyncMock()
    module = "responses_api_agents.pi_agent.app"
    with (
        patch(f"{module}.AsyncSandbox", return_value=sandbox) as factory,
        patch(f"{module}.resolve_provider_config") as resolve,
        TestClient(agent.setup_webserver()) as client,
    ):
        factory.connect = AsyncMock(return_value=sandbox)
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        state = agent._session_records[body.agent_session_id].state
        assert state.session.owns_sandbox is owned
        assert state.session.workdir == sandbox.expected_workdir
        resolve.assert_called_once_with("agent-provider" if owned else "sandbox", {})
        workspace_calls = [call for call in sandbox.exec.await_args_list if call.args[0].startswith("mkdir -p --")]
        assert len(workspace_calls) == int(owned)
        if owned:
            assert workspace_calls[0].args[0] == "mkdir -p -- /agent-workspace"
            assert workspace_calls[0].kwargs["cwd"] == "/"
        if owned:
            factory.connect.assert_not_awaited()
            spec = sandbox.start.await_args.args[0]
            assert spec.image == "test-image"
            assert spec.workdir == "/agent-workspace"
        else:
            factory.assert_not_called()
            factory.connect.assert_awaited_once()
        response = client.post(
            f"/ng-rollout/{body.episode_id.capture_key}/v1/responses", json={"input": "Fix the code"}
        )
        assert response.status_code == 200, response.text
        close_request = {"agent_session_id": body.agent_session_id, "episode_id": body.episode_id.model_dump()}
        closed = client.post("/v1/agent_sessions/close", json=close_request)
        assert closed.status_code == 200, closed.text
        assert client.post("/v1/agent_sessions/close", json=close_request).json() == closed.json()
        if owned:
            sandbox.stop.assert_awaited_once()
            sandbox.disconnect.assert_not_awaited()
        else:
            sandbox.stop.assert_not_awaited()
            sandbox.disconnect.assert_awaited_once()
        assert (
            client.post(f"/ng-rollout/{body.episode_id.capture_key}/v1/responses", json={"input": "task"}).status_code
            == 409
        )


def test_owned_stop_failure_blocks_close_until_retry(setup):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/workspace"}
    sandbox.start = AsyncMock()
    body = seed()
    body.sandbox_access = None
    with (
        patch("responses_api_agents.pi_agent.app.AsyncSandbox", return_value=sandbox),
        TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client,
    ):
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        state = agent._session_records[body.agent_session_id].state
        assert state.session.workdir == "/workspace"
        # An owned sandbox can be destroyed even when no runner receipt was returned.
        state.session.launch_started = True
        sandbox.stop.side_effect = [RuntimeError("provider stop failed"), None]
        close_request = {"agent_session_id": body.agent_session_id, "episode_id": body.episode_id.model_dump()}
        assert client.post("/v1/agent_sessions/close", json=close_request).status_code == 500
        assert not state.session.closed
        assert (
            client.post(f"/ng-rollout/{body.episode_id.capture_key}/v1/responses", json={"input": "task"}).status_code
            == 409
        )
        assert client.post("/v1/agent_sessions/close", json=close_request).status_code == 200
        assert sandbox.stop.await_count == 2
        sandbox.disconnect.assert_not_awaited()


@pytest.mark.parametrize("stage", ["start", "workdir", "install"])
def test_owned_setup_failure_preserves_error_and_retryable_cleanup(setup, stage):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/app"}
    sandbox.start = AsyncMock()
    body = seed()
    body.sandbox_access = None
    original = RuntimeError(f"{stage} failed")
    if stage == "start":
        sandbox.start.side_effect = original
    else:
        execute = sandbox.exec.side_effect

        async def fail_at_stage(command, **kwargs):
            if (stage == "workdir" and command.startswith("mkdir -p --")) or (
                stage == "install" and command.startswith("bash ") and "install_pi_runtime.sh" in command
            ):
                raise original
            return await execute(command, **kwargs)

        sandbox.exec.side_effect = fail_at_stage
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with (
        patch("responses_api_agents.pi_agent.app.AsyncSandbox", return_value=sandbox),
        TestClient(agent.setup_webserver()) as client,
    ):
        with pytest.raises(RuntimeError) as error:
            client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert error.value is original
        assert client.post("/v1/agent_sessions", json=body.model_dump(mode="json")).status_code == 409
        state = agent._session_records[body.agent_session_id].state
        assert state.session.closing and not state.session.closed
        response = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": body.agent_session_id, "episode_id": body.episode_id.model_dump()},
        )
        assert response.status_code == 200, response.text
        assert sandbox.stop.await_count == 2
        sandbox.disconnect.assert_not_awaited()


def test_borrow_connection_failure_never_creates_replacement(setup):
    agent, sandbox = setup
    agent.config.sandbox_provider = "fallback-must-not-be-used"
    module = "responses_api_agents.pi_agent.app"
    with (
        patch(f"{module}.AsyncSandbox") as factory,
        patch(f"{module}.create_provider", return_value=SimpleNamespace(aclose=AsyncMock())),
        TestClient(agent.setup_webserver()) as client,
    ):
        factory.connect = AsyncMock(side_effect=RuntimeError("borrow failed"))
        with pytest.raises(RuntimeError, match="borrow failed"):
            client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        factory.assert_not_called()
        sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("workdir", [None, "relative"])
def test_owned_workdir_is_validated_before_creation(setup, workdir):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"workdir": workdir}
    body = seed()
    body.sandbox_access = None
    with (
        patch("responses_api_agents.pi_agent.app.AsyncSandbox") as factory,
        TestClient(agent.setup_webserver()) as client,
    ):
        assert client.post("/v1/agent_sessions", json=body.model_dump(mode="json")).status_code == 422
        factory.assert_not_called()
        sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("owned", [False, True])
async def test_close_keeps_captured_events_after_sandbox_release(setup, owned):
    agent, sandbox = setup
    sandbox.events += "\n{"  # A terminated writer can leave a partial final record.
    request = Request(
        {"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": seed().episode_id.capture_key}}
    )
    seeded = await agent.seed_agent_session(request, seed())
    state = agent._session_records[seeded.agent_session_id].state
    state.session.owns_sandbox = owned
    captured = asyncio.Event()
    execute = state.execute

    async def pause_after_capture(*args, **kwargs):
        await execute(*args, **kwargs)
        captured.set()
        await asyncio.Future()

    state.execute = pause_after_capture
    sandbox.stop.side_effect = sandbox.files.clear
    sandbox.disconnect.side_effect = sandbox.files.clear
    task = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.wait_for(captured.wait(), 2)
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(seeded.agent_session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandbox.files == {}
    assert state.session.closed
    assert state.session.artifacts == sandbox.events
    assert closed.agent_observations is not None
    observations = closed.agent_observations.model_dump(mode="json")
    assert any(record.get("kind") == "tool_call" for record in observations["records"])
    assert "Fixed" in json.dumps(observations["records"])
    assert any(gap["code"] == "agent_activation_interrupted" for gap in observations["gaps"])
    assert sandbox.stop.await_count == int(owned)
    assert sandbox.disconnect.await_count == int(not owned)


@pytest.mark.parametrize("owned", [False, True])
async def test_malformed_exit_diagnostics_keep_valid_output_and_cleanup(setup, owned):
    agent, sandbox = setup
    sandbox.result["return_code"] = "invalid-diagnostic"
    request, session_id, task = await activate(agent, sandbox)
    state = agent._session_records[session_id].state
    state.session.owns_sandbox = owned
    response = await task
    assert response.status == "completed"
    assert response.output
    assert state.session.cleanup["cleanup_confirmed"] is True
    assert state.session.cleanup["return_code"] is None
    assert any(gap.code == "worker_exit_code_unavailable" for gap in state.observations.gaps)
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert closed.agent_observations.records
    assert sandbox.stop.await_count == int(owned)
    assert sandbox.disconnect.await_count == int(not owned)


async def test_missing_exit_and_empty_output_are_not_success(setup):
    agent, sandbox = setup
    sandbox.result["return_code"] = None
    sandbox.events = ""
    request, session_id, task = await activate(agent, sandbox)
    with pytest.raises(RuntimeError, match="no assistant result"):
        await task
    state = agent._session_records[session_id].state
    assert state.session.cleanup["cleanup_confirmed"] is True
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize(
    "budget,expected", [(None, 1800), ("900.0", 900), ("12.5", 12.5), ("3600", 1800), ("nan", 1800)]
)
def test_task_timeout_metadata_reaches_sandbox_supervisor(
    setup: tuple[PiAgent, Sandbox], tmp_path: Path, budget: str | None, expected: float
) -> None:
    agent, sandbox = setup
    agent.config.timeout = 1800
    (tmp_path / "task.toml").write_text("[agent]\ntimeout_sec = 900.0\n")
    (tmp_path / "instruction.md").write_text("Fix the code")
    (tmp_path / "tests").mkdir()
    body = materialize_task(load_task(tmp_path), "timeout-test").task_input.responses_create_params
    if budget != "900.0":
        body.metadata = None if budget is None else {"agent_timeout_sec": budget}
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        response = client.post(
            "/ng-rollout/pi-smoke-a2/v1/responses", json=body.model_dump(mode="json", exclude_unset=True)
        )
        assert response.status_code == 200, response.text
        command = shlex.split(sandbox.launch.await_args.args[0])
        assert float(command[command.index("--timeout") + 1]) == expected
        assert agent.config.timeout == 1800
        assert response.json()["metadata"]["agent_timeout_sec"] == str(float(expected))
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text


def test_task_timeout_does_not_allow_other_metadata_or_consume_activation(
    setup: tuple[PiAgent, Sandbox],
) -> None:
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        path = "/ng-rollout/pi-smoke-a2/v1/responses"
        rejected = client.post(
            path, json={"input": "Fix the code", "metadata": {"agent_timeout_sec": "900.0", "ignored": "value"}}
        )
        assert rejected.status_code == 422
        assert "metadata" in rejected.json()["detail"]
        sandbox.launch.assert_not_awaited()
        accepted = client.post(path, json={"input": "Fix the code", "metadata": {"agent_timeout_sec": "900.0"}})
        assert accepted.status_code == 200, accepted.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text


@pytest.mark.parametrize(
    "message,status",
    [
        ("400 bad request", 400),
        ("HTTP 503: unavailable", 503),
        ("500 status code (no body)", 500),
        ("401 unauthorized", 401),
    ],
)
def test_native_model_error_is_reported_with_classification(setup, message, status):
    agent, sandbox = setup
    native = {"role": "assistant", "stopReason": "error", "errorMessage": message, "content": []}
    response, observations = sandbox_event_response(
        agent,
        sandbox,
        [
            {"type": "message_end", "message": native},
            {"type": "agent_end", "messages": [native]},
        ],
    )
    assert response["status"] == "failed"
    assert response["error"]["message"] == message
    assert json.loads(response["metadata"]["ng_agent_outcome"]) == {
        "failure_kind": "model_request_error",
        "status_code": status,
    }
    assert observations["records"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "message,failure_kind",
    [("Request timed out.", "model_transport_error"), ("Unrecognized failure", "unknown")],
)
def test_terminal_provider_failure_uses_default_episode_policy(setup, message, failure_kind):
    from nemo_gym.agent_outcome import OutcomeAction, agent_outcome_policy
    from nemo_gym.openai_utils import NeMoGymResponse

    agent, sandbox = setup
    native = {"role": "assistant", "stopReason": "error", "errorMessage": message, "content": []}
    response, observations = sandbox_event_response(
        agent,
        sandbox,
        [
            {"type": "message_end", "message": native},
            {"type": "agent_end", "messages": [native]},
        ],
    )
    assert response["status"] == "failed"
    assert response["error"]["message"] == message
    assert json.loads(response["metadata"]["ng_agent_outcome"]) == {"failure_kind": failure_kind}
    assert observations["records"][0]["status"] == "failed"
    policy = agent_outcome_policy(NeMoGymResponse.model_validate(response))
    assert policy == OutcomeAction.RETRY


def test_process_exit_before_promised_retry_is_execution_failure(setup):
    agent, sandbox = setup
    message = {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": "503 unavailable"}
    sandbox.events = "\n".join(
        json.dumps([float(i), event])
        for i, event in enumerate(
            [
                {"type": "message_end", "message": message},
                {"type": "agent_end", "messages": [message], "willRetry": True},
            ]
        )
    )
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 500
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        assert closed.json()["agent_observations"]["records"][0]["status"] == "failed"
