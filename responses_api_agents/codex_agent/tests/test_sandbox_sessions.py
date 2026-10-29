# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.agent_utils.supervisor_client import parse_cleanup_receipt
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import SandboxStatus
from nemo_gym.tasks.harbor.materialize import materialize_task
from nemo_gym.tasks.harbor.task import load_task
from responses_api_agents.codex_agent.app import CodexAgent, CodexAgentConfig, _model_endpoint_error


def seed() -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="codex-test-session",
        episode_id=EpisodeId(rollout_id="codex-smoke", attempt=2),
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


@pytest.mark.parametrize("version,warn", [("0.144.4", False), ("0.145.0", True)])
def test_session_warns_about_untested_codex_version(setup, caplog, version, warn):
    agent, sandbox = setup
    agent.config.codex_version = version
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).raise_for_status()
    assert ("error classification and transcript parsing are validated on 0.144.4" in caplog.text) == warn


def events(*, stop_reason="stop") -> str:
    items = [
        {"type": "item.completed", "item": {"id": "think-1", "type": "reasoning", "text": "Inspect the repository"}},
        {"type": "item.started", "item": {"id": "tool-1", "type": "command_execution", "command": "pwd"}},
        {
            "type": "item.completed",
            "item": {
                "id": "tool-1",
                "type": "command_execution",
                "command": "pwd",
                "aggregated_output": "/app",
                "exit_code": 0,
                "status": "completed",
            },
        },
        {"type": "item.completed", "item": {"id": "msg-1", "type": "agent_message", "text": "Fixed"}},
    ]
    if stop_reason == "stop":
        items.append(
            {"type": "turn.completed", "usage": {"input_tokens": 17, "output_tokens": 5, "cached_input_tokens": 2}}
        )
    else:
        items.append({"type": "turn.failed", "error": {"message": "model error"}})
    return "\n".join(json.dumps([float(i), event]) for i, event in enumerate(items))


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
        self.exec = AsyncMock(
            side_effect=self.execute,
            return_value=SimpleNamespace(error_type=None, return_code=0, stdout="", stderr=""),
        )
        self.stop = AsyncMock()
        self.disconnect = AsyncMock()
        self.launch = AsyncMock(side_effect=self.create)
        self.signals = []

    async def upload(self, source, destination):
        self.files[destination] = Path(source).read_text()

    async def download(self, source, destination):
        Path(destination).write_text(self.files[source])

    def publish(self):
        self.files[f"{self.directory}/cleanup.json"] = json.dumps(
            {key: value for key, value in self.result.items() if key not in ("hostname", "pid")}
        )
        self.files[f"{self.directory}/runtime.json"] = json.dumps(
            {key: self.result[key] for key in ("hostname", "pid")}
        )
        self.files[f"{self.directory}/events.jsonl"] = self.events

    async def execute(self, command, **kwargs):
        if "exec bash " in command and "process_supervisor.sh" in command:
            return await self.launch(command=command, **kwargs)
        if "touch " in command and "stop.request" in command:
            self.signals.append("SIGTERM")
            if hasattr(self, "directory"):
                self.result["timed_out"] = True
                self.publish()
                self.exited.set()
        return self.exec.return_value

    async def create(self, **kwargs):
        payload_path = next(path for path in self.files if path.endswith("/input.json"))
        payload = json.loads(self.files[payload_path])
        assert payload["cwd"] == getattr(self, "expected_workdir", "/app")
        assert kwargs["cwd"] == getattr(self, "expected_workdir", "/app")
        assert "/sandbox_runner " in kwargs["command"] and "/bin/gym-runner" in kwargs["command"]
        self.directory = payload["directory"]
        self.started.set()
        if not self.blocked:
            self.exited.set()
        await self.exited.wait()
        self.publish()
        return self.exec.return_value


@pytest.fixture
def setup():
    sandbox = Sandbox()
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {"policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}}
    )
    client._build_server_base_url.return_value = "http://model.example:9000"
    config = CodexAgentConfig(
        name="codex",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        model_server={"type": "responses_api_models", "name": "policy"},
        model="test-model",
        codex_version="0.144.4",
        session_close_timeout_seconds=1,
    )
    module = "responses_api_agents.codex_agent.app"
    with (
        patch(f"{module}.ensure_codex", side_effect=AssertionError("agent sessions must not install host Codex")),
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider"),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
    ):
        agent = CodexAgent(config=config, server_client=client)
        yield agent, sandbox


def close_body(session_id):
    return {"agent_session_id": session_id, "episode_id": seed().episode_id.model_dump()}


def test_http_agent_session_flow_runs_codex_in_borrowed_sandbox(setup):
    agent, sandbox = setup
    with patch.object(agent, "_run_codex", AsyncMock(side_effect=AssertionError("host Codex must not run"))):
        with TestClient(agent.setup_webserver()) as client:
            created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
            assert created.status_code == 200, created.text
            session_id = created.json()["agent_session_id"]
            directory = agent._session_records[session_id].state.session.session_dir
            assert f"{directory}/process_supervisor.sh" not in sandbox.files
            installer = f"{directory}/install_codex_runtime.sh"
            assert installer in sandbox.files
            staged = {"sandbox_runner": "sandbox_runner.py", "sandbox_transport.py": "sandbox_transport.py"}
            for name, source in staged.items():
                assert sandbox.files[f"{directory}/{name}"] == (Path(__file__).parents[1] / source).read_text()
            assert agent.config.resources_server is None
            assert not sandbox.launch.called
            assert not any(path.startswith("/app/") for path in sandbox.files)
            install_call = sandbox.exec.await_args_list[1]
            assert install_call.args[0] == (f"bash {installer} /tmp/nemo-gym-codex-node-22.19.0-0.144.4 0.144.4")
            assert install_call.kwargs["timeout_s"] == agent.config.sandbox_install_timeout_seconds
            result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Fix the code"})
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
            assert body["usage"]["input_tokens_details"]["cached_tokens"] == 2
            assert body["metadata"]["harness_execution"] == "sandbox"
            assert "_ng_agent_observations" not in body
            payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
            assert payload["prompt"] == "Fix the code"
            assert payload["command"][0].endswith("/node/bin/node")
            assert "PATH" not in payload["env"]
            assert "HOME" not in payload["env"]
            assert not any(key.startswith("XDG_") for key in payload["env"])
            config = tomllib.loads(sandbox.files[f"{sandbox.directory}/home/.codex/config.toml"])
            assert (
                config["model_providers"]["gym"]["base_url"]
                == "http://model.example:9000/ng-rollout/codex-smoke-a2/v1"
            )
            assert payload["command"][-2:] == ["--", "-"]
            closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
            assert closed.status_code == 200, closed.text
            observations = closed.json()["agent_observations"]
            assert observations["source"] == "codex"
            assert "no_sandbox_runtime" not in [gap["code"] for gap in observations["gaps"]]
            assert "model_call_join_key_unavailable" in [gap["code"] for gap in observations["gaps"]]
    assert not any(record.state is not None for record in agent._session_records.values())
    assert agent._local_setup_task is None
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "3"])
def test_sandbox_usage_details_without_measured_positive_counts_are_unknown(setup, value) -> None:
    agent, sandbox = setup
    records = [json.loads(line) for line in sandbox.events.splitlines()]
    records[-1][1]["usage"].update(cached_input_tokens=value, reasoning_output_tokens=value)
    sandbox.events = "\n".join(json.dumps(record) for record in records)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        response = result.json()
        assert response["status"] == "completed"
        assert response["usage"]["total_tokens"] == 22
        assert response["usage"]["input_tokens_details"]["cached_tokens"] is None
        assert response["usage"]["output_tokens_details"]["reasoning_tokens"] is None
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id)).json()
        gaps = {gap["code"] for gap in closed["agent_observations"]["gaps"]}
        assert {"cached_token_usage_unavailable", "reasoning_token_usage_unavailable"} <= gaps


@pytest.mark.parametrize("unknown_turn", [None, 0, 1])
def test_sandbox_usage_details_require_known_counts_from_every_completed_turn(setup, unknown_turn) -> None:
    agent, sandbox = setup
    records = [json.loads(line) for line in sandbox.events.splitlines()]
    records[-1][1]["usage"]["reasoning_output_tokens"] = 1
    second_usage = {"input_tokens": 7, "output_tokens": 3, "cached_input_tokens": 4, "reasoning_output_tokens": 2}
    if unknown_turn is not None:
        # Missing details at either end must not turn a subtotal into a complete measurement.
        unknown = records[-1][1]["usage"] if unknown_turn == 0 else second_usage
        unknown.pop("cached_input_tokens")
        unknown.pop("reasoning_output_tokens")
    records.append([float(len(records)), {"type": "turn.completed", "usage": second_usage}])
    sandbox.events = "\n".join(json.dumps(record) for record in records)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        response = result.json()
        assert response["status"] == "completed"
        assert response["usage"]["total_tokens"] == 32
        assert response["usage"]["input_tokens_details"]["cached_tokens"] == (6 if unknown_turn is None else None)
        assert response["usage"]["output_tokens_details"]["reasoning_tokens"] == (3 if unknown_turn is None else None)
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id)).json()
        gaps = {gap["code"] for gap in closed["agent_observations"]["gaps"]}
        assert ("cached_token_usage_unavailable" in gaps) == (unknown_turn is not None)
        assert ("reasoning_token_usage_unavailable" in gaps) == (unknown_turn is not None)


def test_recovered_stream_error_keeps_success_and_available_usage_with_coverage_gap(setup) -> None:
    agent, sandbox = setup
    records = [json.loads(line) for line in sandbox.events.splitlines()]
    records.insert(
        0, [0.0, {"type": "error", "message": "Reconnecting... 1/5 (stream disconnected before completion)"}]
    )
    sandbox.events = "\n".join(json.dumps(record) for record in records)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        response = result.json()
        assert response["status"] == "completed"
        assert response["error"] is None
        assert response["usage"]["total_tokens"] == 22
        assert [item["type"] for item in response["output"]] == [
            "reasoning",
            "function_call",
            "function_call_output",
            "message",
        ]
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id)).json()
        assert closed["agent_observations"]["records"][0]["status"] == "completed"
        gaps = {gap["code"]: gap for gap in closed["agent_observations"]["gaps"]}
        assert "recovered retries" in gaps["partial_model_usage_unavailable"]["detail"]


def test_sandbox_custom_model_context_budget_reaches_sandbox_config(setup) -> None:
    agent, sandbox = setup
    agent.config = CodexAgentConfig(
        **(agent.config.model_dump() | {"model_context_window": 40960, "model_auto_compact_token_limit": 32768})
    )
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Fix multiply"})
        assert result.status_code == 200, result.text
        config = tomllib.loads(sandbox.files[f"{sandbox.directory}/home/.codex/config.toml"])
        assert config["model_context_window"] == 40960
        assert config["model_auto_compact_token_limit"] == 32768
        assert config["model"] == "test-model"
        assert config["model_providers"]["gym"]["base_url"] == "http://model.example:9000/ng-rollout/codex-smoke-a2/v1"
        assert config["features"] == {"multi_agent": False, "code_mode": False}
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    assert agent.config.extra_config == {}


def test_codex_provider_retries_and_idle_timeout_reach_sandbox_config(setup) -> None:
    # extra_config is refused in sandbox sessions, so these provider settings have fields of their own.
    agent, sandbox = setup
    agent.config = CodexAgentConfig(
        **(
            agent.config.model_dump()
            | {"request_max_retries": 8, "stream_max_retries": 10, "stream_idle_timeout_ms": 90_000_000}
        )
    )
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Fix multiply"})
        assert result.status_code == 200, result.text
        config = tomllib.loads(sandbox.files[f"{sandbox.directory}/home/.codex/config.toml"])
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    provider = config["model_providers"]["gym"]
    assert (provider["request_max_retries"], provider["stream_max_retries"]) == (8, 10)
    assert provider["stream_idle_timeout_ms"] == 90_000_000


def test_codex_cli_default_features_and_session_state_reach_the_sandbox_launch(setup) -> None:
    # A passthrough model server can relay every tool shape, so a sandbox launch can match a plain
    # `codex exec`: no [features] overrides and, as in every sandbox session, no --ephemeral.
    agent, sandbox = setup
    agent.config.features = {"multi_agent": None, "code_mode": None}
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Fix multiply"})
        assert result.status_code == 200, result.text
        config = tomllib.loads(sandbox.files[f"{sandbox.directory}/home/.codex/config.toml"])
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    assert "features" not in config
    assert config["model_providers"]["gym"]["base_url"] == "http://model.example:9000/ng-rollout/codex-smoke-a2/v1"
    assert "--ephemeral" not in payload["command"]
    assert payload["command"][-2:] == ["--", "-"]


def test_direct_run_without_resources_rejected_before_execution(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/run", json={"responses_create_params": {"input": "task"}})
    assert response.status_code == 422
    assert "requires resources_server" in response.json()["detail"]
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
        agent.config.codex_version = "latest"
    else:
        body["tool_accesses"] = [
            {"kind": "direct_http", "name": "tools", "base_url": "http://resources", "required": True}
        ]
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=body).status_code == 422
    sandbox.exec.assert_not_awaited()


def mcp_grant(name="gdpval_resources_server", *, required=True, url="http://resources.example:7000/mcp"):
    return {
        "kind": "mcp",
        "name": name,
        "required": required,
        "connection": {"url": url, "headers": {"X-NeMo-Gym-Session-Token": f"token-{name}"}},
    }


def test_mcp_grants_reach_the_sandbox_codex_config(setup):
    agent, sandbox = setup
    agent.config = CodexAgentConfig(
        **(
            agent.config.model_dump()
            | {
                "mcp_tool_timeout_seconds": 300,
                "tool_accesses": [
                    mcp_grant("search-tools", required=False, url="http://search.example:7100/mcp"),
                    # The episode's grant of the same name replaces this one.
                    mcp_grant(url="http://stale.example:1/mcp"),
                ],
            }
        )
    )
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [
        mcp_grant(),
        # An optional direct HTTP grant stays unused rather than failing the seed.
        {"kind": "direct_http", "name": "direct.tools", "base_url": "http://resources", "required": False},
    ]
    transcript = [json.loads(line) for line in sandbox.events.splitlines()]
    mcp_call = {"id": "mcp-1", "type": "mcp_tool_call", "server": "gdpval_resources_server", "tool": "finish"}
    transcript[-2:-2] = [
        [1.5, {"type": "item.started", "item": mcp_call}],
        [1.6, {"type": "item.completed", "item": mcp_call | {"status": "completed", "arguments": {}}}],
    ]
    sandbox.events = "\n".join(json.dumps(record) for record in transcript)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Fix the code"})
        assert result.status_code == 200, result.text
        assert [item["name"] for item in result.json()["output"] if item["type"] == "function_call"] == [
            "exec_command",
            "finish",
        ]
        config = tomllib.loads(sandbox.files[f"{sandbox.directory}/home/.codex/config.toml"])
        assert config["mcp_servers"] == {
            "search-tools": {
                "url": "http://search.example:7100/mcp",
                "http_headers": {"X-NeMo-Gym-Session-Token": "token-search-tools"},
                "tool_timeout_sec": 300.0,
            },
            "gdpval_resources_server": {
                "url": "http://resources.example:7000/mcp",
                "http_headers": {"X-NeMo-Gym-Session-Token": "token-gdpval_resources_server"},
                "required": True,
                "tool_timeout_sec": 300.0,
            },
        }
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
        # The token reaches Codex only through its config, never the command line.
        assert "token-gdpval_resources_server" not in json.dumps(payload)
        closed = client.post("/v1/agent_sessions/close", json=close_body(body["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    tools = {
        record["tool_call_id"]: record["tool_name"]
        for record in closed.json()["agent_observations"]["records"]
        if record["kind"] == "tool_call"
    }
    # Observations name an MCP call by its tool, as the response does.
    assert tools == {"tool-1": "exec_command", "mcp-1": "finish"}


@pytest.mark.parametrize(
    "grant, message",
    [
        (
            {"kind": "direct_http", "name": "resources.direct_http", "base_url": "http://r", "required": True},
            "supports only MCP tool grants; required grants it cannot use: resources.direct_http",
        ),
        (mcp_grant("resources.mcp"), "letters, digits, '-' and '_': resources.mcp"),
    ],
)
def test_unusable_required_tool_grants_are_rejected_before_connection(setup, grant, message):
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant]
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=body)
    assert response.status_code == 422
    assert message in response.json()["detail"]
    sandbox.exec.assert_not_awaited()


def test_optional_mcp_grant_with_an_unusable_name_is_left_out(setup):
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [mcp_grant(), mcp_grant("search.mcp", required=False)]
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=body).json()["agent_session_id"]
        assert client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"}).status_code == 200
        config = tomllib.loads(sandbox.files[f"{sandbox.directory}/home/.codex/config.toml"])
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id)).json()
    assert list(config["mcp_servers"]) == ["gdpval_resources_server"]
    gaps = [(gap["code"], gap["detail"]) for gap in closed["agent_observations"]["gaps"]]
    assert ("mcp_server_skipped", "search.mcp: its name is not letters, digits, '-' and '_'") in gaps


@pytest.mark.parametrize("value", [0, -1, float("nan"), 601])
def test_mcp_tool_timeout_must_be_positive_and_within_the_run_timeout(setup, value):
    agent, _ = setup
    with pytest.raises(ValidationError):
        CodexAgentConfig(**(agent.config.model_dump() | {"timeout": 600, "mcp_tool_timeout_seconds": value}))
    CodexAgentConfig(**(agent.config.model_dump() | {"timeout": 600, "mcp_tool_timeout_seconds": 600}))


@pytest.mark.parametrize("started", [False, True])
def test_startup_failure_reports_the_codex_stderr_tail(setup, started):
    agent, sandbox = setup
    # Codex repeats a failed server's error body, up to 8 KiB, after the line's start that names the server.
    stderr = (
        "rmcp worker log line\n" * 300
        + "Error: required MCP servers failed to initialize: gdpval_resources_server: "
        + "x" * 9000
        + " (-32603)"
    )
    sandbox.result["return_code"] = 1
    thread = [{"type": "thread.started", "thread_id": "t"}]
    if started:
        thread.append({"type": "turn.started"})
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(thread))
    publish = sandbox.publish

    def publish_with_stderr():
        publish()
        sandbox.files[f"{sandbox.directory}/stderr.log"] = stderr

    sandbox.publish = publish_with_stderr
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id)).json()
    assert result.status_code == 502
    detail = result.json()["detail"]
    gaps = {gap["code"]: gap["detail"] for gap in closed["agent_observations"]["gaps"]}
    if started:
        # A turn started, so the event stream explains the failure; stderr is not appended.
        assert "stderr:" not in detail
        assert "agent_startup_failed" not in gaps
    else:
        assert "required MCP servers failed to initialize: gdpval_resources_server" in detail
        assert detail.endswith(" (-32603)")
        assert len(detail) < 2100, "the stderr tail is bounded"
        assert "x" * 600 not in detail, "long stderr lines are shortened"
        # Environment Servers record the observations, not the 502 body.
        assert gaps["agent_startup_failed"] in detail


def test_startup_failure_stderr_hides_the_grant_tokens(setup):
    agent, sandbox = setup
    api_key = 'user="alice" api_key=secret-123 note=abcdef'
    sandbox.result["return_code"] = 1
    sandbox.events = json.dumps([0.0, {"type": "thread.started", "thread_id": "t"}])
    publish = sandbox.publish

    def publish_with_stderr():
        publish()
        sandbox.files[f"{sandbox.directory}/stderr.log"] = (
            "MCP error body: X-NeMo-Gym-Session-Token: token-gdpval_resources_server\n"
            # Codex cuts long error bodies, which can end a token early.
            "HTTP 401: {'x-nemo-gym-session-token': 'token-gd... (truncated 2048 bytes)\n"
            "authorization: Digest user=u nonce=s3cret\n"
            'HTTP 401: x-api-key: user="alice" api_key=secret-123 note=abc... (truncated 2048 bytes)\n'
            'retry: user="ali,user="alice" api_key=secret-123 note=abcdef\n'
            # An error body can echo the headers as JSON.
            f"HTTP 403: {json.dumps({'x-api-key': api_key})}\n"
            "Error: required MCP servers failed to initialize: gdpval_resources_server"
        )

    sandbox.publish = publish_with_stderr
    grant = mcp_grant()
    grant["connection"]["headers"]["Authorization"] = "Digest user=u nonce=s3cret"
    # Shares its start with Authorization, so redacting one must not hide the other's start.
    grant["connection"]["headers"]["Proxy-Authorization"] = "Digest user=bob nonce=beta-secret"
    grant["connection"]["headers"]["X-Api-Key"] = api_key
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant]
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=body).status_code == 200
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        closed = client.post("/v1/agent_sessions/close", json=close_body(body["agent_session_id"])).json()
    assert result.status_code == 502
    gap = {gap["code"]: gap["detail"] for gap in closed["agent_observations"]["gaps"]}["agent_startup_failed"]
    for text in (result.json()["detail"], gap):
        assert "token-gd" not in text
        assert "s3cret" not in text
        assert "secret-123" not in text
        assert "X-NeMo-Gym-Session-Token: [redacted]" in text
        assert "'x-nemo-gym-session-token': '[redacted] (truncated" in text
        assert "x-api-key: [redacted] (truncated" in text
        assert "retry: [redacted]\n" in text
        assert 'HTTP 403: {"x-api-key": "[redacted]"}' in text
        assert "required MCP servers failed to initialize: gdpval_resources_server" in text


def test_unreadable_startup_stderr_does_not_hold_the_activation(setup):
    agent, sandbox = setup
    sandbox.result["return_code"] = 1
    sandbox.events = json.dumps([0.0, {"type": "thread.started", "thread_id": "t"}])
    download = sandbox.download

    async def hanging_stderr_download(source, destination):
        if source.endswith("/stderr.log"):
            await asyncio.sleep(10)  # Longer than the 1 s bound, but finite so a regression fails instead of hanging.
        await download(source, destination)

    sandbox.download = hanging_stderr_download
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        started = time.monotonic()
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        elapsed = time.monotonic() - started
    assert result.status_code == 502
    assert result.json()["detail"] == "Codex execution failed: Codex exited with code 1"
    # The read is bounded by session_close_timeout_seconds (1 s in this fixture).
    assert elapsed < 5


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
        first = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert first.status_code == 200
        assert client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"}).json() == first.json()
        assert client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "changed"}).status_code == 409
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.launch.assert_awaited_once()


@pytest.mark.parametrize("reason,expected", [("error", "failed"), ("aborted", "failed")])
def test_failed_or_partial_codex_output_is_preserved(setup, reason, expected):
    agent, sandbox = setup
    sandbox.events = events(stop_reason=reason)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 502
        assert "model error" in result.json()["detail"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        invocation = closed.json()["agent_observations"]["records"][0]
        assert invocation["status"] == expected
        assert invocation["conversation"][-1]["content"][0]["text"] == "Fixed"


@pytest.mark.parametrize(
    "case",
    [
        # Graded, as Harbor grades after any agent exit:
        "codex-killed",  # e.g. the kernel's OOM kill picked Codex: the runner exits 128+9
        "runner-killed",  # e.g. the agent's `pkill python3` hit the runner: a negative status
        "failed-turn",
        "timed-out",  # an error event in a run that hit its time limit: graded as a timeout
        # Retried as before:
        "flag-off",
        "not-started",  # Codex failed before its first turn, as when a required MCP server does not connect
        "cleanup-error",
        "sandbox-stopped",
        "sandbox-unknown",
        "sandbox-silent",
        "probe-fails",
    ],
)
def test_failed_codex_run_is_graded_only_while_its_sandbox_still_runs(setup, case) -> None:
    agent, sandbox = setup
    agent.config.verify_after_agent_error = case != "flag-off"
    records = [json.loads(line) for line in events().splitlines()]
    if case == "failed-turn":
        records = [json.loads(line) for line in events(stop_reason="error").splitlines()]
    elif case == "timed-out":
        sandbox.result.update(timed_out=True, return_code=-15)
        records[-1] = [9.0, {"type": "item.completed", "item": {"type": "error", "message": "tool call failed"}}]
    elif case == "not-started":
        sandbox.result["return_code"] = 1
        records = []
    else:
        # Codex died mid-turn: no turn.completed.
        sandbox.result["return_code"] = -9 if case == "runner-killed" else 247
        records = records[:-1]
    if case != "not-started":
        records.insert(0, [-1.0, {"type": "turn.started"}])
    sandbox.events = "\n".join(json.dumps(record) for record in records)
    if case == "cleanup-error":
        sandbox.result["error"] = "cleanup: Sandbox tool processes remain alive"
    sandbox.status = AsyncMock(
        return_value={"sandbox-stopped": SandboxStatus.STOPPED, "sandbox-unknown": SandboxStatus.UNKNOWN}.get(
            case, SandboxStatus.RUNNING
        )
    )
    execute = sandbox.exec.side_effect

    async def probe_or_execute(command, **kwargs):
        if command == "true":
            if case == "sandbox-silent":
                raise ConnectionError("sandbox backend unreachable")
            return SimpleNamespace(
                error_type=None, return_code=1 if case == "probe-fails" else 0, stdout="", stderr=""
            )
        return await execute(command, **kwargs)

    sandbox.exec.side_effect = probe_or_execute
    graded = case in ("codex-killed", "runner-killed", "failed-turn", "timed-out")
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    probed = any(call.args[:1] == ("true",) for call in sandbox.exec.await_args_list)
    assert probed is (graded or case in ("sandbox-silent", "probe-fails"))
    expected_status = "incomplete" if case == "timed-out" else "failed"
    assert closed.json()["agent_observations"]["records"][0]["status"] == expected_status
    if not graded:
        assert result.status_code == 502, result.text
        return
    # A response, not an execution error: Resources grades it as a nonzero agent exit.
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["status"] == expected_status
    assert body["error"] is None
    assert "Fixed" in json.dumps(body["output"])
    assert body["metadata"]["graded_agent_error"].startswith(f"exit {sandbox.result['return_code']}: ")
    # No classified cause, so the Environment Server's outcome policy verifies it.
    assert "ng_agent_outcome" not in body["metadata"]


# Codex 0.154.0's final error once its retries against a failing endpoint are spent (a top-level error
# event, then turn.failed with the same message), captured from the CLI, with the retry notices it printed.
CODEX_0154_ENDPOINT_FAILURES = {
    "502": ("unexpected status 502 Bad Gateway: Unknown error, url: http://127.0.0.1:44879/v1/responses", 5),
    "503": ("unexpected status 503 Service Unavailable: Unknown error, url: http://127.0.0.1:36063/v1/responses", 5),
    "504": ("unexpected status 504 Gateway Timeout: Unknown error, url: http://127.0.0.1:47353/v1/responses", 5),
    "500": ("We're currently experiencing high demand, which may cause temporary errors.", 5),
    "transport": (
        "stream disconnected before completion: Transport error: network error: error decoding response body",
        5,
    ),
    "closed": ("stream disconnected before completion: stream closed before response.completed", 5),
    "send": (
        "stream disconnected before completion: error sending request for url (http://127.0.0.1:46443/v1/responses)",
        5,
    ),
    "idle": ("stream disconnected before completion: idle timeout waiting for SSE", 5),
    "429": ("exceeded retry limit, last status: 429 Too Many Requests", 0),
}
# Printed while the endpoint refuses connections; Codex keeps waiting rather than failing the turn.
CODEX_0154_WAITING_FOR_NETWORK = "Reconnecting... waiting for network (Connection failed: error sending request)"
CODEX_OUTPUT_CAP = "stream disconnected before completion: Incomplete response returned, reason: max_output_tokens"


def codex_error(message: str) -> list[dict]:
    """Codex 0.154.0's final error: a top-level error event, then turn.failed."""
    return [{"type": "error", "message": message}, {"type": "turn.failed", "error": {"message": message}}]


def retry_notice(message: str, attempt: int = 1) -> dict:
    return {"type": "error", "message": f"Reconnecting... {attempt}/5 ({message})"}


def run_codex_transcript(agent, sandbox, transcript: list[dict]):
    """Run one activation over ``transcript``; return the response, the closed session and whether the
    sandbox was probed for liveness (only a failed run that is not an endpoint failure is)."""
    agent.config.verify_after_agent_error = True
    sandbox.status = AsyncMock(return_value=SandboxStatus.RUNNING)
    sandbox.events = "\n".join(json.dumps([float(index), event]) for index, event in enumerate(transcript))
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    probed = any(call.args[:1] == ("true",) for call in sandbox.exec.await_args_list)
    return result, closed.json(), probed


@pytest.mark.parametrize(
    "message",
    [
        *(message for message, _ in CODEX_0154_ENDPOINT_FAILURES.values()),
        # The same format for a 500 (0.154.0 reported the captured 500 as "high demand").
        "unexpected status 500 Internal Server Error: Unknown error, url: http://127.0.0.1:1/v1/responses",
    ],
)
def test_codex_0154_endpoint_failures_are_model_endpoint_errors(message) -> None:
    final = [(9.0, event) for event in codex_error(message)]
    assert _model_endpoint_error(final) is not None
    assert all(_model_endpoint_error([event]) is not None for event in final)
    assert _model_endpoint_error([(1.0, retry_notice(message))]) is not None
    assert _model_endpoint_error([(1.0, {"type": "error", "message": CODEX_0154_WAITING_FOR_NETWORK})]) is not None


@pytest.mark.parametrize(
    "message",
    [
        CODEX_OUTPUT_CAP,  # the output cap is a gradable outcome, not an endpoint failure
        "",  # 0.154.0's message for a 400
        "unexpected status 400 Bad Request: Unknown error, url: http://127.0.0.1:1/v1/responses",
        "unexpected status 404 Not Found: Unknown error, url: http://127.0.0.1:1/v1/responses",
        "model error",
    ],
)
def test_other_codex_errors_are_not_model_endpoint_errors(message) -> None:
    assert _model_endpoint_error([(9.0, event) for event in codex_error(message)]) is None


MODEL_OUTPUT = {"type": "item.completed", "item": {"id": "msg-1", "type": "agent_message", "text": "Fixed"}}
WAITING = {"type": "error", "message": CODEX_0154_WAITING_FOR_NETWORK}
BAD_GATEWAY = CODEX_0154_ENDPOINT_FAILURES["502"][0]
IDLE = CODEX_0154_ENDPOINT_FAILURES["idle"][0]
BAD_REQUEST = "unexpected status 400 Bad Request: Unknown error, url: http://127.0.0.1:1/v1/responses"


@pytest.mark.parametrize(
    "transcript, ended_by_endpoint",
    [
        # Retries followed by model output were recovered; the run then ended for another reason.
        ([retry_notice(BAD_GATEWAY), MODEL_OUTPUT], False),  # e.g. then OOM-killed
        ([retry_notice(BAD_GATEWAY), MODEL_OUTPUT, *codex_error("model error")], False),
        ([WAITING, MODEL_OUTPUT, {"type": "turn.completed", "usage": {}}], False),
        # Only the terminal failure counts: the last call's retries timed out, but it failed on a 400.
        ([MODEL_OUTPUT, retry_notice(IDLE), retry_notice(IDLE, 2), *codex_error(BAD_REQUEST)], False),
        # The outage ended the run: retries spent, or Codex killed while it still retried or waited.
        ([retry_notice(BAD_GATEWAY), MODEL_OUTPUT, retry_notice(BAD_GATEWAY), *codex_error(BAD_GATEWAY)], True),
        ([MODEL_OUTPUT, retry_notice(BAD_GATEWAY)], True),
        ([MODEL_OUTPUT, WAITING, WAITING], True),
    ],
)
def test_only_the_failure_that_ended_the_run_is_checked(transcript, ended_by_endpoint) -> None:
    events = [(float(index), event) for index, event in enumerate(transcript)]
    assert (_model_endpoint_error(events) is not None) is ended_by_endpoint


@pytest.mark.parametrize("ending", ["retries-spent", "killed-while-retrying"])
@pytest.mark.parametrize("case", list(CODEX_0154_ENDPOINT_FAILURES))
def test_codex_0154_endpoint_outage_is_retried_not_graded(setup, case, ending) -> None:
    from environment_servers.single_agent_turn.app import OutcomeAction, agent_outcome_policy
    from nemo_gym.core import NeMoGymResponse

    agent, sandbox = setup
    message, retries = CODEX_0154_ENDPOINT_FAILURES[case]
    transcript = [
        {"type": "turn.started"},
        *(retry_notice(message, attempt) for attempt in range(1, retries + 1)),
        *codex_error(message),
    ]
    if ending == "retries-spent":
        sandbox.result["return_code"] = 1
    else:
        sandbox.result["return_code"] = -9
        transcript.pop()  # killed after the error event, before turn.failed
    result, closed, probed = run_codex_transcript(agent, sandbox, transcript)
    # The sandbox was never probed for grading, and the episode is rerun.
    assert not probed
    assert closed["agent_observations"]["records"][0]["status"] == "failed"
    if ending == "killed-while-retrying":
        # An execution error.
        assert result.status_code == 502, result.text
        return
    # Codex reported the failure: its classified cause reaches the outcome policy, which retries it.
    assert result.status_code == 200, result.text
    response = NeMoGymResponse.model_validate(result.json())
    assert (response.status, response.error.message) == ("failed", message)
    assert "graded_agent_error" not in response.metadata
    assert agent_outcome_policy(response) == OutcomeAction.RETRY


# Codex's own messages for an endpoint at capacity (HTTP 503) and a timed-out model request (see
# test_native_errors_use_captured_attempt_facts).
CODEX_NATIVE_ENDPOINT_FAILURES = {
    "capacity": "Selected model is at capacity. Please try a different model.",
    "request-timeout": "request timed out",
}


@pytest.mark.parametrize("ending", ["reported", "killed-after-reporting"])
@pytest.mark.parametrize("case", list(CODEX_NATIVE_ENDPOINT_FAILURES))
def test_native_endpoint_failures_are_retried_not_graded(setup, case, ending) -> None:
    from environment_servers.single_agent_turn.app import OutcomeAction, agent_outcome_policy
    from nemo_gym.core import NeMoGymResponse

    agent, sandbox = setup
    message = CODEX_NATIVE_ENDPOINT_FAILURES[case]
    assert _model_endpoint_error([(9.0, event) for event in codex_error(message)]) is not None
    # Codex exits 1 on the reported failure, or its runner is killed right after it reports it.
    sandbox.result["return_code"] = 1 if ending == "reported" else -9
    result, closed, probed = run_codex_transcript(agent, sandbox, [{"type": "turn.started"}, *codex_error(message)])
    assert not probed
    assert closed["agent_observations"]["records"][0]["status"] == "failed"
    if ending == "killed-after-reporting":
        assert result.status_code == 502, result.text
        return
    assert result.status_code == 200, result.text
    response = NeMoGymResponse.model_validate(result.json())
    assert (response.status, response.error.message) == ("failed", message)
    assert "graded_agent_error" not in response.metadata
    assert agent_outcome_policy(response) == OutcomeAction.RETRY


@pytest.mark.parametrize(
    "message, action",
    [
        ("model error", "retry"),
        ("stream error: Connection refused", "retry"),
        (BAD_REQUEST, "verify"),  # the outcome policy grades a request the model server rejected
    ],
)
def test_reported_provider_failure_keeps_its_classified_outcome(setup, message, action) -> None:
    """Codex exited 1 on turn.failed: the outcome policy decides it, not verify_after_agent_error."""
    from environment_servers.single_agent_turn.app import OutcomeAction, agent_outcome_policy
    from nemo_gym.core import NeMoGymResponse

    agent, sandbox = setup
    sandbox.result["return_code"] = 1
    transcript = [{"type": "turn.started"}, MODEL_OUTPUT, *codex_error(message)]
    result, closed, probed = run_codex_transcript(agent, sandbox, transcript)
    assert result.status_code == 200, result.text
    response = NeMoGymResponse.model_validate(result.json())
    assert (response.status, response.error.message) == ("failed", message)
    assert "graded_agent_error" not in response.metadata
    assert agent_outcome_policy(response) == OutcomeAction(action)
    assert not probed


@pytest.mark.parametrize("ending", ["timed-out", "completed"])
def test_recovered_codex_0154_retries_keep_the_run_gradable(setup, ending) -> None:
    # Every endpoint message as a recovered retry; the run then hits its time limit or completes.
    agent, sandbox = setup
    transcript = [
        {"type": "turn.started"},
        *(retry_notice(message) for message, _ in CODEX_0154_ENDPOINT_FAILURES.values()),
        {"type": "error", "message": CODEX_0154_WAITING_FOR_NETWORK},
        *(json.loads(line)[1] for line in events().splitlines()),
    ]
    if ending == "timed-out":
        sandbox.result.update(timed_out=True, return_code=-15)
        transcript.pop()  # stopped in its turn: no turn.completed
    result, closed, probed = run_codex_transcript(agent, sandbox, transcript)
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["error"] is None
    assert body["status"] == ("incomplete" if ending == "timed-out" else "completed")
    # Not an agent error at all: graded as the time limit (or the completion) it is.
    assert "graded_agent_error" not in body["metadata"]
    assert not probed
    assert "Fixed" in json.dumps(body["output"])


@pytest.mark.parametrize("ending", ["oom-kill", "runner-killed"])
def test_recovered_codex_0154_retries_then_an_agent_error_is_graded(setup, ending) -> None:
    # Endpoint retries early in the run, recovered; the run then ends on an unrelated agent error.
    agent, sandbox = setup
    transcript = [
        {"type": "turn.started"},
        *(retry_notice(message) for message, _ in CODEX_0154_ENDPOINT_FAILURES.values()),
        *(retry_notice(message) for message in CODEX_NATIVE_ENDPOINT_FAILURES.values()),
        WAITING,
        *[json.loads(line)[1] for line in events().splitlines()][:-1],  # model output, no turn.completed
    ]
    # The kernel's OOM kill picked Codex, or the agent's `pkill python3` hit the runner.
    sandbox.result["return_code"] = 247 if ending == "oom-kill" else -9
    result, closed, probed = run_codex_transcript(agent, sandbox, transcript)
    # Graded as a nonzero agent exit, as the reference runs grade it.
    assert result.status_code == 200, result.text
    body = result.json()
    assert (body["status"], body["error"]) == ("failed", None)
    assert body["metadata"]["graded_agent_error"].startswith(f"exit {sandbox.result['return_code']}: ")
    assert probed
    assert closed["agent_observations"]["records"][0]["status"] == "failed"


def test_codex_0154_killed_while_waiting_for_the_network_is_rerun(setup) -> None:
    agent, sandbox = setup
    sandbox.result["return_code"] = 247
    transcript = [
        {"type": "turn.started"},
        *[json.loads(line)[1] for line in events().splitlines()][:-1],
        WAITING,
        WAITING,
    ]
    result, _, probed = run_codex_transcript(agent, sandbox, transcript)
    assert result.status_code == 502, result.text
    assert not probed


def test_codex_0154_output_cap_after_a_recovered_retry_is_gradable(setup) -> None:
    agent, sandbox = setup
    sandbox.result["return_code"] = 1
    transcript = [
        {"type": "turn.started"},
        retry_notice(CODEX_0154_ENDPOINT_FAILURES["idle"][0]),
        {"type": "item.completed", "item": {"id": "partial", "type": "reasoning", "text": "Inspect first"}},
        *codex_error(CODEX_OUTPUT_CAP),
    ]
    result, _, probed = run_codex_transcript(agent, sandbox, transcript)
    assert result.status_code == 200, result.text
    body = result.json()
    assert (body["status"], body["incomplete_details"]["reason"], body["error"]) == (
        "incomplete",
        "max_output_tokens",
        None,
    )
    assert not probed


def test_incomplete_model_response_retains_reasoning_and_reports_usage_gap(setup) -> None:
    agent, sandbox = setup
    message = "stream disconnected before completion: Incomplete response returned, reason: max_output_tokens"
    sandbox.result["return_code"] = 1
    sandbox.events = "\n".join(
        json.dumps([float(index), event])
        for index, event in enumerate(
            [
                {"type": "turn.started"},
                {"type": "item.completed", "item": {"id": "partial", "type": "reasoning", "text": "Inspect first"}},
                {"type": "turn.failed", "error": {"message": message}},
            ]
        )
    )
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Fix multiply"})
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["status"] == "incomplete"
        assert body["incomplete_details"]["reason"] == "max_output_tokens"
        assert body["error"] is None
        assert body["output"][0]["summary"][0]["text"] == "Inspect first"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
        observations = closed.json()["agent_observations"]
        assert observations["records"][0]["status"] == "incomplete"
        assert observations["records"][0]["conversation"][-1]["summary"][0]["text"] == "Inspect first"
        assert "partial_model_usage_unavailable" in [gap["code"] for gap in observations["gaps"]]


@pytest.mark.parametrize(
    "condition", ["completed", "failed-turn", "failed-exit", "in-turn", "other-error", "later-error"]
)
def test_startup_model_metadata_advisory_requires_successful_turn(setup, condition) -> None:
    agent, sandbox = setup
    warning = (
        "Model metadata for `test-model` not found. Defaulting to fallback metadata; "
        "this can degrade performance and cause issues."
    )
    diagnostic = {
        "type": "item.completed",
        "item": {"id": "startup", "type": "error", "message": warning},
    }
    started = {"type": "turn.started"}
    if condition == "failed-turn":
        sandbox.events = events(stop_reason="error")
    elif condition == "failed-exit":
        sandbox.result["return_code"] = 1
    elif condition == "other-error":
        diagnostic["item"]["message"] = "Failed to initialize model client"
    prefix = [started, diagnostic] if condition == "in-turn" else [diagnostic, started]
    if condition == "later-error":
        prefix.append(
            {"type": "item.completed", "item": {"id": "failed", "type": "error", "message": "Model call failed"}}
        )
    sandbox.events = "\n".join(json.dumps([-2 + i, event]) for i, event in enumerate(prefix)) + "\n" + sandbox.events
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == (200 if condition in ("completed", "timed-out") else 502), response.text
        body = response.json()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
        observations = closed.json()["agent_observations"]
        warnings = [gap for gap in observations["gaps"] if gap["code"] == "model_metadata_fallback"]
        if condition == "completed":
            assert body["status"] == "completed"
            assert body["error"] is None
            assert body["usage"]["total_tokens"] == 22
            assert [gap["detail"] for gap in warnings] == [warning]
            assert observations["records"][0]["status"] == "completed"
        else:
            assert "detail" in body
            assert diagnostic["item"]["message"] in body["detail"]
            if condition == "failed-turn":
                assert "model error" in body["detail"]
            if condition == "later-error":
                assert "Model call failed" in body["detail"]
            assert not warnings
            assert observations["records"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "condition",
    ["completed", "failed-turn", "failed-exit", "timed-out", "no-terminal", "other-error", "near-match", "no-start"],
)
def test_advisories_remain_visible_on_completion_or_gradable_limits(setup, condition) -> None:
    agent, sandbox = setup
    warning = (
        "Heads up: Long threads and multiple compactions can cause the model to be less accurate. "
        "Start a new thread when possible to keep threads small and targeted."
    )
    startup = (
        "Model metadata for `test-model` not found. Defaulting to fallback metadata; "
        "this can degrade performance and cause issues."
    )
    transcript = [
        {"type": "item.completed", "item": {"id": "startup", "type": "error", "message": startup}},
        *([] if condition == "no-start" else [{"type": "turn.started"}]),
    ]
    transcript.extend(json.loads(line)[1] for line in events().splitlines())
    # The real long Ansible trace had six successful compactions followed by a
    # normal final message and turn.completed, with all usage still available.
    for index in range(6):
        transcript.insert(
            -2,
            {
                "type": "item.completed",
                "item": {
                    "id": f"compaction-warning-{index}",
                    "type": "error",
                    "message": warning + (" Extra error" if condition == "near-match" else ""),
                },
            },
        )
    if condition == "failed-turn":
        transcript.append({"type": "turn.failed", "error": {"message": "Final model call failed"}})
    elif condition == "failed-exit":
        sandbox.result["return_code"] = 1
    elif condition == "timed-out":
        sandbox.result["timed_out"] = True
    elif condition == "no-terminal":
        transcript.pop()
    elif condition == "other-error":
        transcript.insert(
            -1, {"type": "item.completed", "item": {"type": "error", "message": "Unknown model call error"}}
        )
    sandbox.events = "\n".join(json.dumps([float(index), event]) for index, event in enumerate(transcript))
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == (200 if condition in ("completed", "timed-out") else 502), result.text
        body = result.json()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    observations = closed.json()["agent_observations"]
    advisories = [gap for gap in observations["gaps"] if gap["code"] == "compaction_accuracy_advisory"]
    startup_gaps = [gap for gap in observations["gaps"] if gap["code"] == "model_metadata_fallback"]
    if condition in ("completed", "timed-out"):
        expected_status = "incomplete" if condition == "timed-out" else "completed"
        assert body["status"] == expected_status
        assert body["error"] is None
        assert body["usage"]["total_tokens"] == 22
        assert [gap["detail"] for gap in advisories] == [warning] * 6
        assert [gap["detail"] for gap in startup_gaps] == [startup]
        assert observations["records"][0]["status"] == expected_status
    else:
        assert "detail" in body
        assert warning in body["detail"]
        assert startup in body["detail"]
        if condition == "other-error":
            assert "Unknown model call error" in body["detail"]
        if condition == "failed-turn":
            assert "Final model call failed" in body["detail"]
        assert not advisories
        assert not startup_gaps
        assert observations["records"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "override",
    [
        {"max_output_tokens": 123},
        {"temperature": 0.2},
        {"seed": 42},
        {"top_p": 0.9},
        {"tools": [{"type": "function", "name": "foo"}]},
        {"input": [{"role": "assistant", "content": "old turn"}]},
    ],
)
def test_unsupported_request_is_not_silently_ignored(setup, override):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task", **override})
        assert result.status_code == 422, result.text
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.launch.assert_not_awaited()


def test_rejected_request_does_not_consume_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/codex-smoke-a2/v1/responses"
        assert client.post(path, json={"input": "task", "temperature": 0.2}).status_code == 422
        sandbox.launch.assert_not_awaited()
        accepted = client.post(path, json={"input": "task"})
        assert accepted.status_code == 200, accepted.text
        assert client.post(path, json={"input": "task"}).json() == accepted.json()
        assert client.post(path, json={"input": "changed task"}).status_code == 409
    sandbox.launch.assert_awaited_once()


@pytest.mark.parametrize(
    "metadata",
    [
        {"extra_body": '{"temperature": 0.0, "max_tokens": 1}'},
        {"extra_body": "{}"},
        {"extra_body": ""},
        {"extra_body": "not-json"},
        {"chat_template_kwargs": '{"enable_thinking": false}'},
    ],
)
def test_metadata_overrides_rejected_without_consuming_activation(setup, metadata):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/codex-smoke-a2/v1/responses"
        rejected = client.post(path, json={"input": "task", "metadata": metadata})
        assert rejected.status_code == 422, rejected.text
        assert next(iter(metadata)) in rejected.json()["detail"]
        assert "model server" in rejected.json()["detail"]
        sandbox.launch.assert_not_awaited()
        accepted = client.post(path, json={"input": "task"})
        assert accepted.status_code == 200, accepted.text
        assert client.post(path, json={"input": "task"}).json() == accepted.json()
        client.post("/v1/agent_sessions/close", json=close_body(seed().agent_session_id)).raise_for_status()
    sandbox.launch.assert_awaited_once()


def test_no_session_keeps_existing_local_path(setup):
    agent, sandbox = setup
    with patch.object(agent, "_create_response", AsyncMock(side_effect=RuntimeError("legacy path reached"))) as legacy:
        with TestClient(agent.setup_webserver()) as client:
            with pytest.raises(RuntimeError, match="legacy path reached"):
                client.post("/v1/responses", json={"input": "task"})
        legacy.assert_awaited_once()
    sandbox.launch.assert_not_awaited()


async def activate(agent, sandbox):
    request = Request({"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": "codex-smoke-a2"}})
    seeded = await agent.seed_agent_session(request, seed())
    task = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.wait_for(sandbox.started.wait(), 2)
    return request, seeded.agent_session_id, task


async def test_close_cancels_active_codex_before_detaching(setup):
    agent, sandbox = setup
    sandbox.blocked = True
    request, session_id, task = await activate(agent, sandbox)
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandbox.signals == ["SIGTERM"]
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_close_during_the_startup_stderr_read_keeps_the_observations(setup):
    agent, sandbox = setup
    sandbox.result["return_code"] = 1
    sandbox.events = json.dumps([0.0, {"type": "thread.started", "thread_id": "t"}])
    download = sandbox.download
    reading = asyncio.Event()

    async def hanging_stderr_download(source, destination):
        if source.endswith("/stderr.log"):
            reading.set()
            await asyncio.sleep(10)  # Bounded so a regression fails instead of hanging.
        await download(source, destination)

    sandbox.download = hanging_stderr_download
    request, session_id, task = await activate(agent, sandbox)
    await asyncio.wait_for(reading.wait(), 5)
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [record.status for record in closed.agent_observations.records] == ["failed"]


async def test_close_during_the_grading_probe_keeps_the_observations(setup):
    agent, sandbox = setup
    agent.config.verify_after_agent_error = True
    sandbox.result["return_code"] = -9  # killed mid-turn, after a tool call and a message
    records = [json.loads(line) for line in events().splitlines()][:-1]
    records.insert(0, [-1.0, {"type": "turn.started"}])
    sandbox.events = "\n".join(json.dumps(record) for record in records)
    probing = asyncio.Event()

    async def hanging_status():
        probing.set()
        await asyncio.sleep(10)  # Bounded so a regression fails instead of hanging.
        return SandboxStatus.RUNNING

    sandbox.status = hanging_status
    request, session_id, task = await activate(agent, sandbox)
    await asyncio.wait_for(probing.wait(), 5)
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    invocation, *tools = closed.agent_observations.records
    assert invocation.status == "failed"
    assert "Fixed" in json.dumps([item.model_dump(mode="json") for item in invocation.conversation])
    assert [tool.tool_call_id for tool in tools] == ["tool-1"]
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
    with pytest.raises(RuntimeError, match="Codex sandbox runner returned no valid result") as failure:
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


async def test_harness_agent_request_without_scope_keeps_local_path(setup):
    agent, sandbox = setup
    request = SimpleNamespace(path_params={})
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    with patch.object(agent, "_create_response", AsyncMock()) as create:
        result = await agent.responses(request, body)
        assert result is create.return_value
        create.assert_awaited_once_with(body)
    sandbox.exec.assert_not_awaited()


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
        with pytest.raises(HTTPException, match="exited with code 7"):
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


def test_instructions_and_text_parts_reach_codex_without_other_provider_credentials(setup):
    agent, sandbox = setup
    agent.config.system_prompt = "config instruction"
    agent.config.openai_api_key = "must-not-copy"
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post(
            "/ng-rollout/codex-smoke-a2/v1/responses",
            json={
                "instructions": "request instruction",
                "input": [
                    {"role": "system", "content": "input instruction"},
                    {"role": "user", "content": [{"type": "input_text", "text": "task"}]},
                ],
            },
        )
        assert response.status_code == 200, response.text
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
        config = tomllib.loads(sandbox.files[f"{sandbox.directory}/home/.codex/config.toml"])
        assert config["developer_instructions"] == "config instruction\n\nrequest instruction\n\ninput instruction"
        assert payload["prompt"] == "task"
        assert payload["env"]["OPENAI_API_KEY"] == "gym"
        assert "must-not-copy" not in json.dumps(sandbox.files)
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        conversation = closed.json()["agent_observations"]["records"][0]["conversation"]
        assert conversation[0]["content"] == config["developer_instructions"]


def test_close_retry_and_stale_activation_do_not_run_host_codex(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"}).raise_for_status()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        repeated = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert repeated.status_code == 200
        assert repeated.json() == closed.json()
        bad = close_body(session_id)
        bad["episode_id"]["attempt"] = 99
        assert client.post("/v1/agent_sessions/close", json=bad).status_code == 409
        with patch.object(agent, "_run_codex", AsyncMock(side_effect=AssertionError("host Codex must not run"))):
            assert client.post("/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 409
    sandbox.disconnect.assert_awaited_once()


def test_http_close_retry_survives_other_session_closes(setup, monkeypatch):
    agent, sandbox = setup
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: 100.0)
    with TestClient(agent.setup_webserver()) as client:

        def seed_and_close(index):
            client.cookies.clear()
            body = seed().model_dump(mode="json")
            body["agent_session_id"] = f"codex-test-session-{index}"
            body["episode_id"] = {"rollout_id": f"episode-{index}"}
            created = client.post("/v1/agent_sessions", json=body)
            assert created.status_code == 200
            cookies = dict(client.cookies)
            close = {"agent_session_id": created.json()["agent_session_id"], "episode_id": body["episode_id"]}
            result = client.post("/v1/agent_sessions/close", json=close)
            assert result.status_code == 200
            return cookies, close, result.json()

        cookies, close, first = seed_and_close(0)
        for index in range(1, 66):
            seed_and_close(index)
        client.cookies.clear()
        client.cookies.update(cookies)
        retry = client.post("/v1/agent_sessions/close", json=close)
        assert retry.status_code == 200
        assert retry.json() == first
    assert sandbox.disconnect.await_count == 66


async def test_close_receipt_expires_without_extending_on_retry(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    close = AgentCloseSessionRequest(**close_body(session_id))
    first = await agent.close_agent_session(request, close)
    clock[0] = 109.0
    assert await agent.close_agent_session(request, close) == first
    clock[0] = 110.0
    with pytest.raises(HTTPException) as error:
        await agent.close_agent_session(request, close)
    assert error.value.status_code == 409
    assert not agent._closed_session_records
    with pytest.raises(HTTPException) as stale_seed:
        await agent.seed_agent_session(request, seed())
    assert stale_seed.value.status_code == 409
    with patch.object(agent, "_create_response", AsyncMock(side_effect=AssertionError("host fallback"))):
        with pytest.raises(HTTPException) as error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert error.value.status_code == 409
    sandbox.disconnect.assert_awaited_once()


async def test_close_retry_window_starts_after_cleanup(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id

    async def disconnect():
        clock[0] = 200.0

    sandbox.disconnect.side_effect = disconnect
    close = AgentCloseSessionRequest(**close_body(session_id))
    first = await agent.close_agent_session(request, close)
    clock[0] = 209.0
    assert await agent.close_agent_session(request, close) == first
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
    sandbox.disconnect.assert_awaited_once()


async def test_seed_prunes_expired_close_receipts(setup, monkeypatch):
    agent, _ = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: clock[0])
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    clock[0] += agent.config.session_close_retry_window_seconds
    request.session.clear()
    await agent.seed_agent_session(request, seed().model_copy(update={"agent_session_id": "fresh-session"}))
    assert not agent._closed_session_records


@pytest.mark.parametrize("window", [0, -1, float("inf")])
def test_close_retry_window_must_be_positive_and_finite(setup, window):
    agent, _ = setup
    with pytest.raises(ValidationError):
        CodexAgentConfig(**(agent.config.model_dump() | {"session_close_retry_window_seconds": window}))


async def test_unknown_launch_outcome_fails_closed(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "codex-smoke-a2"}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    sandbox.launch.side_effect = TimeoutError("lost launch response")
    with pytest.raises(TimeoutError):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert agent._session_records[session_id].state is not None
    sandbox.disconnect.assert_not_awaited()
    sandbox.stop.assert_not_awaited()


async def test_lost_execution_is_never_graded(setup):
    """A provider error (a lost or unresponsive sandbox) stays an execution error that bypasses
    verification even with verify_after_agent_error, so the episode is retried."""
    agent, sandbox = setup
    agent.config.verify_after_agent_error = True
    sandbox.status = AsyncMock(return_value=SandboxStatus.RUNNING)
    sandbox.result["return_code"] = 247
    create = sandbox.launch.side_effect

    async def lost(**kwargs):
        await create(**kwargs)
        raise RuntimeError("OpenSandbox background command status unresponsive")

    sandbox.launch.side_effect = lost
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "codex-smoke-a2"}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    with pytest.raises(RuntimeError, match="status unresponsive"):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert closed.agent_observations.records[0].status == "incomplete"
    assert not any(call.args[:1] == ("true",) for call in sandbox.exec.await_args_list)
    sandbox.status.assert_not_awaited()


async def test_install_failure_disconnects_without_stopping_owner(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(error_type=None, return_code=0),
        SimpleNamespace(error_type=None, return_code=1, stdout="", stderr="npm failed"),
        SimpleNamespace(error_type=None, return_code=0),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="npm failed"):
        await agent.seed_agent_session(request, seed())
    assert not any(record.state is not None for record in agent._session_records.values())
    assert not request.session
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_cancelled_install_never_publishes_session_or_launches_codex(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(error_type=None, return_code=0),
        asyncio.CancelledError(),
        SimpleNamespace(error_type=None, return_code=0),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(asyncio.CancelledError):
        await agent.seed_agent_session(request, seed())
    assert not any(record.state is not None for record in agent._session_records.values())
    assert not request.session
    sandbox.launch.assert_not_awaited()
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("stage", ["prepare", "install", "close"])
async def test_provider_error_type_blocks_success_even_with_zero_exit(setup, stage):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}})
    error = SimpleNamespace(return_code=0, error_type="timeout", stdout="", stderr="provider timeout")
    ok = SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
    if stage == "prepare":
        sandbox.exec.side_effect = [error, ok]
    elif stage == "install":
        sandbox.exec.side_effect = [ok, error, ok]
    if stage != "close":
        with pytest.raises(RuntimeError, match="timeout"):
            await agent.seed_agent_session(request, seed())
        assert not request.session
        sandbox.disconnect.assert_awaited_once()
        if stage == "prepare":
            assert sandbox.exec.await_count == 2
            assert "mv /tmp/nemo-gym-codex-sessions/" in sandbox.exec.await_args.args[0]
        return
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    sandbox.exec.return_value = error
    with pytest.raises(RuntimeError, match="Could not remove Codex session files"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert agent._session_records[session_id].state is not None
    sandbox.disconnect.assert_not_awaited()
    sandbox.exec.return_value = ok
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    sandbox.disconnect.assert_awaited_once()


async def test_timeout_preserves_partial_tools_and_close_observations(setup):
    agent, sandbox = setup
    sandbox.result.update(timed_out=True, return_code=-9)
    sandbox.events = "\n".join(sandbox.events.splitlines()[:-1])
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    assert response.status == "incomplete"
    assert [item.type for item in response.output] == ["reasoning", "function_call", "function_call_output", "message"]
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert closed.agent_observations.records[0].status == "incomplete"
    assert "partial_model_usage_unavailable" in [gap.code for gap in closed.agent_observations.gaps]


async def test_independent_sessions_do_not_share_identity_or_configuration(setup):
    agent, sandbox = setup
    first = Request({"type": "http", "session": {}})
    second = Request({"type": "http", "session": {}})
    a = (await agent.seed_agent_session(first, seed())).agent_session_id
    other = seed().model_copy(update={"agent_session_id": "codex-other", "episode_id": EpisodeId(rollout_id="other")})
    b = (await agent.seed_agent_session(second, other)).agent_session_id
    assert a != b
    assert agent._session_records[a].state.session.session_dir != agent._session_records[b].state.session.session_dir
    with pytest.raises(HTTPException):
        await agent.close_agent_session(second, AgentCloseSessionRequest(**close_body(a)))
    assert agent._session_records[a].state is not None and agent._session_records[b].state is not None


async def test_local_runtime_is_lazy_once_and_retries_failed_setup(setup):
    agent, sandbox = setup
    with patch(
        "responses_api_agents.codex_agent.app.ensure_codex", side_effect=[RuntimeError("install failed"), None]
    ) as install:
        assert not install.called
        with pytest.raises(RuntimeError, match="install failed"):
            await agent._ensure_local_runtime()
        await asyncio.gather(agent._ensure_local_runtime(), agent._ensure_local_runtime())
        assert install.call_count == 2
    sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "wrong-model"),
        ("prompt_cache_key", "key"),
        ("max_tool_calls", 2),
        ("reasoning", {"effort": "high"}),
        ("top_p", 0.9),
    ],
)
def test_model_boundary_options_are_rejected_before_activation(setup, field, value):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        response = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task", field: value})
        assert response.status_code == 422
        sandbox.launch.assert_not_awaited()
        accepted = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert accepted.status_code == 200, accepted.text


@pytest.mark.parametrize("marker", [None, 0, [], {}, ""])
async def test_malformed_session_markers_cannot_fall_back_to_host(setup, marker):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {"agent_session_id": marker}})
    with patch.object(agent, "_create_response", AsyncMock(side_effect=AssertionError("host fallback"))) as legacy:
        with pytest.raises(HTTPException) as error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert error.value.status_code == 409
        from responses_api_agents.codex_agent.app import CodexAgentRunRequest

        with pytest.raises(HTTPException) as error:
            await agent.run(request, CodexAgentRunRequest(responses_create_params={"input": "task"}))
        assert error.value.status_code == 409
        legacy.assert_not_awaited()
    with pytest.raises(HTTPException) as error:
        await agent.seed_agent_session(request, seed())
    assert error.value.status_code == 409
    with pytest.raises(HTTPException) as error:
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(seed().agent_session_id)))
    assert error.value.status_code == 409
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("complete", [False, True])
async def test_latest_item_update_survives_cancel_without_duplicate_completion(setup, complete):
    agent, sandbox = setup
    started = {"id": "running-test", "type": "command_execution", "command": "pytest -x", "status": "in_progress"}
    updated = started | {"aggregated_output": "collected 42 tests\ntest_first PASSED\n"}
    items = [{"type": "item.started", "item": started}, {"type": "item.updated", "item": updated}]
    if complete:
        items += [
            {"type": "item.completed", "item": updated | {"status": "completed", "exit_code": 0}},
            {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}},
        ]
    sandbox.events = "\n".join(json.dumps([float(i), item]) for i, item in enumerate(items))
    sandbox.blocked = not complete
    request, session_id, task = await activate(agent, sandbox)
    if complete:
        response = await task
        assert len(response.output) == 2
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    if not complete:
        with pytest.raises(asyncio.CancelledError):
            await task
    conversation = closed.agent_observations.records[0].conversation
    calls = [item for item in conversation if item.type == "function_call"]
    outputs = [item for item in conversation if item.type == "function_call_output"]
    assert len(calls) == len(outputs) == 1
    assert json.loads(calls[0].arguments) == {"cmd": "pytest -x"}
    assert outputs[0].output == "collected 42 tests\ntest_first PASSED\n"
    assert calls[0].status == outputs[0].status == ("completed" if complete else "incomplete")


@pytest.mark.parametrize("alias", ["workdir", "sessions", "runtime"])
def test_sandbox_prepare_rejects_symlink_overlap_before_writing(tmp_path, alias):
    from responses_api_agents.codex_agent.app import _sandbox_prepare_command

    repository = tmp_path / "repository"
    repository.mkdir()
    sessions = tmp_path / "sessions"
    runtime = tmp_path / "runtime"
    workdir = repository
    if alias == "workdir":
        workdir = tmp_path / "alias"
        workdir.symlink_to(tmp_path, target_is_directory=True)
    elif alias == "sessions":
        sessions.symlink_to(repository, target_is_directory=True)
    else:
        runtime.symlink_to(repository, target_is_directory=True)
    directory = sessions / "session-1"
    result = subprocess.run(
        ["bash", "-c", _sandbox_prepare_command(str(workdir), str(directory), str(runtime))],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert result.returncode != 0
    assert "must be disjoint" in result.stderr
    assert not directory.exists()


def test_sandbox_prepare_accepts_disjoint_quoted_paths(tmp_path):
    from responses_api_agents.codex_agent.app import _sandbox_prepare_command

    repository = tmp_path / "task with 'quote"
    repository.mkdir()
    directory = tmp_path / "session with 'quote"
    result = subprocess.run(
        ["bash", "-c", _sandbox_prepare_command(str(repository), str(directory), str(tmp_path / "runtime"))],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (directory / "home/.codex").is_dir()
    assert not list(repository.iterdir())


async def _local_exec(command, **kwargs):
    process = await asyncio.create_subprocess_exec(
        "sh", "-c", command, cwd=kwargs.get("cwd"), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return SimpleNamespace(
        return_code=process.returncode, error_type=None, stdout=stdout.decode(), stderr=stderr.decode()
    )


async def test_session_rollouts_are_copied_to_the_logs_dir(tmp_path):
    """Codex's session rollouts (per-call token counts) survive the session directory, as Harbor keeps them."""
    from nemo_gym.agent_utils.sandbox_session import SandboxSession
    from responses_api_agents.codex_agent.sandbox import CodexSandboxSession

    def state(directory, logs_dir):
        session = SandboxSession(
            sandbox=SimpleNamespace(exec=_local_exec), session_dir=str(directory), workdir="/app", harness="Codex"
        )
        return CodexSandboxSession(request=seed(), session=session, runtime="/tmp/runtime", logs_dir=logs_dir)

    rollout = tmp_path / "session/home/.codex/sessions/2026/10/16/rollout-2026-10-16T22-15-00-abc.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text('{"type":"event_msg","payload":{"type":"token_count"}}\n')
    await state(tmp_path / "session", str(tmp_path / "logs/agent")).copy_session_rollouts()
    copied = tmp_path / "logs/agent/sessions/2026/10/16" / rollout.name
    assert copied.read_text() == rollout.read_text()

    # --ephemeral (no sessions directory) and no logs_dir are both no-ops.
    (tmp_path / "ephemeral").mkdir()
    await state(tmp_path / "ephemeral", str(tmp_path / "logs2")).copy_session_rollouts()
    assert not (tmp_path / "logs2").exists()
    await state(tmp_path / "session", None).copy_session_rollouts()


async def test_collect_copies_session_rollouts_after_reading_events(setup):
    agent, sandbox = setup
    agent.config.sandbox_logs_dir = "/logs/agent"
    body = seed()
    await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    state = agent._session_records[body.agent_session_id].state
    assert state.logs_dir == "/logs/agent"
    sandbox.files[f"{state.session.session_dir}/events.jsonl"] = events()
    sandbox.exec.reset_mock()
    assert await state.collect_artifacts() == events()
    copy = sandbox.exec.await_args_list[-1]
    assert f"{state.session.session_dir}/home/.codex/sessions" in copy.args[0]
    assert "/logs/agent/sessions" in copy.args[0]


def test_a_slow_session_rollout_copy_does_not_cost_the_run(setup):
    """The optional copy ends before the capture budget does, so the run is still returned and graded."""
    agent, sandbox = setup
    agent.config.sandbox_logs_dir = "/logs/agent"
    agent.config.session_close_timeout_seconds = 3
    execute = sandbox.exec.side_effect

    async def hanging_copy(command, **kwargs):
        if "cp -R" in command:
            await asyncio.sleep(10)  # Past the capture budget; bounded so a regression fails instead of hanging.
        return await execute(command, **kwargs)

    sandbox.exec.side_effect = hanging_copy
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["status"] == "completed"
    assert body["usage"]["total_tokens"] == 22
    assert closed.status_code == 200, closed.text
    invocation = closed.json()["agent_observations"]["records"][0]
    assert invocation["status"] == "completed"
    assert "Fixed" in json.dumps(invocation["conversation"])
    assert any("cp -R" in call.args[0] for call in sandbox.exec.await_args_list)


@pytest.mark.parametrize("value", ["/", "relative", "/logs/../tmp"])
def test_sandbox_logs_dir_must_be_an_absolute_directory_other_than_root(value):
    with pytest.raises(ValidationError):
        CodexAgentConfig(
            name="codex",
            host="localhost",
            port=8001,
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy"},
            codex_version="0.144.4",
            sandbox_logs_dir=value,
        )


def test_sandbox_prepare_accepts_root_workdir(tmp_path):
    """A task image without WORKDIR lends "/": the root holds the session and runtime paths, as under Harbor."""
    from responses_api_agents.codex_agent.app import _sandbox_prepare_command

    directory = tmp_path / "sessions" / "session-1"
    result = subprocess.run(
        ["bash", "-c", _sandbox_prepare_command("/", str(directory), str(tmp_path / "runtime"))],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (directory / "home/.codex").is_dir()


def test_sandbox_prepare_rejects_other_workdir_holding_session_files(tmp_path):
    from responses_api_agents.codex_agent.app import _sandbox_prepare_command

    directory = tmp_path / "sessions" / "session-1"
    result = subprocess.run(
        ["bash", "-c", _sandbox_prepare_command(str(tmp_path), str(directory), str(tmp_path / "runtime"))],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert result.returncode != 0
    assert "must be disjoint" in result.stderr
    assert not directory.exists()


async def test_root_workdir_seeds_with_runtime_and_session_under_tmp(setup):
    agent, sandbox = setup
    body = seed()
    body.sandbox_access.workdir = "/"
    await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    state = agent._session_records[body.agent_session_id].state
    assert state.session.workdir == "/"
    assert state.session.session_dir.startswith("/tmp/nemo-gym-codex-sessions/")
    assert state.runtime.startswith("/tmp/nemo-gym-codex-node-")
    assert any(
        call.kwargs.get("cwd") == "/" and "install_codex_runtime.sh" in call.args[0]
        for call in sandbox.exec.await_args_list
    )
    await agent.close_agent_session(
        Request({"type": "http", "session": {}}), AgentCloseSessionRequest(**close_body(body.agent_session_id))
    )


@pytest.mark.parametrize(
    "workdir", ["/tmp", "/tmp/nemo-gym-codex-sessions/other", "/tmp/nemo-gym-codex-node-22.19.0-0.144.4"]
)
def test_seed_rejects_workdirs_that_hold_runtime_or_session_files(setup, workdir):
    agent, sandbox = setup
    body = seed()
    body.sandbox_access.workdir = workdir
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
    assert created.status_code == 422
    assert "outside SandboxAccess.workdir" in created.text
    sandbox.exec.assert_not_awaited()


async def test_seed_is_idempotent_and_binds_complete_payload(setup):
    agent, sandbox = setup
    first = Request({"type": "http", "session": {}})
    second = Request({"type": "http", "session": {}})
    body = seed()
    with patch.object(agent, "_seed_agent_session_state", wraps=agent._seed_agent_session_state) as initialize:
        results = await asyncio.gather(agent.seed_agent_session(first, body), agent.seed_agent_session(second, body))
        assert [result.agent_session_id for result in results] == [body.agent_session_id] * 2
        assert initialize.await_count == 1
        changed = body.model_copy(deep=True)
        changed.sandbox_access.workdir = "/other"
        with pytest.raises(HTTPException) as error:
            await agent.seed_agent_session(second, changed)
        assert error.value.status_code == 409
    await agent.close_agent_session(second, AgentCloseSessionRequest(**close_body(body.agent_session_id)))
    sandbox.disconnect.assert_awaited_once()


async def test_cookie_less_close_cleans_lost_seed_response(setup):
    agent, sandbox = setup
    body = seed()
    await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    request = Request({"type": "http", "session": {}})
    close = AgentCloseSessionRequest(**close_body(body.agent_session_id))
    first = await agent.close_agent_session(request, close)
    assert await agent.close_agent_session(Request({"type": "http", "session": {}}), close) == first
    assert not any(record.state is not None for record in agent._session_records.values())
    sandbox.disconnect.assert_awaited_once()


async def test_close_before_seed_leaves_bounded_tombstone(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    body = seed()
    close = AgentCloseSessionRequest(**close_body(body.agent_session_id))
    stale = Request({"type": "http", "session": {}})
    await agent.close_agent_session(stale, close)
    with pytest.raises(HTTPException):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    sandbox.exec.assert_not_awaited()
    clock[0] = 110.0
    with pytest.raises(HTTPException, match="expired"):
        await agent.close_agent_session(stale, close)
    with pytest.raises(HTTPException, match="expired"):
        await agent.seed_agent_session(stale, body)
    assert not agent._closed_session_records


async def test_close_waits_for_racing_seed_and_removes_completed_setup(setup):
    agent, sandbox = setup
    entered = asyncio.Event()
    release = asyncio.Event()
    initialize = agent._seed_agent_session_state

    async def blocked_initialize(body):
        entered.set()
        await release.wait()
        return await initialize(body)

    with patch.object(agent, "_seed_agent_session_state", side_effect=blocked_initialize):
        creation = asyncio.create_task(agent.seed_agent_session(Request({"type": "http", "session": {}}), seed()))
        await entered.wait()
        closure = asyncio.create_task(
            agent.close_agent_session(
                Request({"type": "http", "session": {}}),
                AgentCloseSessionRequest(**close_body(seed().agent_session_id)),
            )
        )
        await asyncio.sleep(0)
        assert not closure.done()
        release.set()
        await asyncio.gather(creation, closure)
    assert not any(record.state is not None for record in agent._session_records.values())
    sandbox.disconnect.assert_awaited_once()


async def test_caller_identifier_is_not_a_filesystem_path(setup):
    agent, sandbox = setup
    body = seed().model_copy(update={"agent_session_id": "../caller supplied/id"})
    request = Request({"type": "http", "session": {}})
    created = await agent.seed_agent_session(request, body)
    assert created.agent_session_id == body.agent_session_id
    directory = Path(agent._session_records[created.agent_session_id].state.session.session_dir)
    assert directory.parent == Path("/tmp/nemo-gym-codex-sessions")
    assert len(directory.name) == 32
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(created.agent_session_id)))


async def test_session_lifecycle_uses_shared_bookkeeping(setup):
    from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

    agent, sandbox = setup
    assert type(agent).close_agent_session is SimpleResponsesAPIAgent.close_agent_session
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    assert not hasattr(agent, "_session_reapers")
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("outcome", ["completed", "provider-failure", "graded-failure"])
async def test_independent_configs_collect_flat_rows_through_environment_run(setup, monkeypatch, outcome):
    """Exercise the checked-in recipe through collector, real environment, and real Codex lifecycle.

    A failed Codex run bypasses verification unless verify_after_agent_error grades it."""
    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
    from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
    from nemo_gym.config import GlobalConfigDictParser
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
    from nemo_gym.server_utils import BaseServerConfig

    agent, sandbox = setup
    provider_failure = outcome == "provider-failure"
    if outcome != "completed":
        # Codex started its turn, which then failed.
        sandbox.events = "\n".join([json.dumps([-1.0, {"type": "turn.started"}]), events(stop_reason="error")])
    root = Path(__file__).parents[3]
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths(
        [
            str(root / "resources_servers/swebench_pro/configs/swebench_pro.yaml"),
            str(root / "responses_api_agents/codex_agent/configs/codex_agent.yaml"),
            str(root / "environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml"),
        ]
    )
    config = OmegaConf.merge(*configs)
    config.policy_model_name = "test-model"
    environment_name = "single_agent_turn_legacy"
    environment_config = config[environment_name].environment_servers.single_agent_turn_legacy
    environment_config.agent_server.name = "codex_agent"
    environment_config.resources_server.name = "swebench_pro_resources_server"
    agent_name = environment_config.agent_server.name
    resources_name = environment_config.resources_server.name
    assert config[agent_name].responses_api_agents.codex_agent.resources_server is None
    agent.config = CodexAgentConfig(
        name=agent_name,
        host="localhost",
        port=8001,
        **OmegaConf.to_container(config[agent_name].responses_api_agents.codex_agent, resolve=True),
    )
    if outcome == "graded-failure":
        agent.config.verify_after_agent_error = True
        sandbox.status = AsyncMock(return_value=SandboxStatus.RUNNING)
    config.policy_model = {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}
    transport = ServerClient(head_server_config=BaseServerConfig(host="head", port=1), global_config_dict=config)
    agent.server_client = transport
    environment = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name=environment_name,
            host="localhost",
            port=8002,
            **OmegaConf.to_container(environment_config, resolve=True),
        ),
        server_client=transport,
    )
    calls = []
    close_receipts = []
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
            return Response(await environment.run_legacy(body))
        if server_name == resources_name:
            if url_path == "/seed_session":
                assert body["task_data"]["instance_id"] == "instance"
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
                if outcome == "completed":
                    assert body["response"]["usage"]["total_tokens"] == 22
                else:
                    # The graded run did model work after the CLI's last total: its usage is unknown, not zero.
                    assert body["response"]["usage"] is None
                assert body["response"]["status"] == ("completed" if outcome == "completed" else "failed")
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
            close_receipts.append(result)
        else:
            assert url_path.endswith("/v1/responses")
            request.scope["path_params"] = {"rollout_id": url_path.split("/")[2]}
            result = await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming.model_validate(body))
        return Response(result.model_dump(mode="json"), cookie="agent-cookie")

    monkeypatch.setattr(ServerClient, "post", post)
    monkeypatch.setattr(ServerClient, "_resolve_base_url", lambda self, name: f"http://{name}:8000")
    monkeypatch.setattr(RolloutCollectionHelper, "setup_server_client", lambda self, head=None: transport)
    materialized = {"responses_create_params": {"input": "Fix it"}, "instance_id": "instance"}
    collection_config = RolloutCollectionConfig(
        input_jsonl_fpath="input.jsonl",
        output_jsonl_fpath="output.jsonl",
        agent_name=agent_name,
        num_repeats=1,
    )
    rows = RolloutCollectionHelper._preprocess_raw_rows(
        [(0, json.dumps(materialized), materialized)], collection_config
    )
    _, result = await next(RolloutCollectionHelper().run_examples(rows))
    if provider_failure:
        assert "reward" not in result
        assert result["_ng_failure_stage"] == "agent"
        assert "model error" in result["_ng_failure_message"]
        assert (resources_name, "/verify") not in calls
    else:
        assert result.get("reward") == 1.0, result
        if outcome == "completed":
            assert result["response"]["usage"]["total_tokens"] == 22
        else:
            assert result["response"]["usage"] is None
        assert result["ng_agent_observations"]["source"] == "codex"
    assert close_receipts[-1].agent_observations.source == "codex"
    assert close_receipts[-1].agent_observations.records[0].status == (
        "completed" if outcome == "completed" else "failed"
    )
    expected = [
        (environment_name, "/run"),
        (resources_name, "/seed_session"),
        (agent_name, "/v1/agent_sessions"),
        (agent_name, "/ng-rollout/0-0/v1/responses"),
        (agent_name, "/v1/agent_sessions/close"),
        (resources_name, "/verify"),
        (resources_name, "/close_session"),
    ]
    if provider_failure:
        expected.remove((resources_name, "/verify"))
    assert calls == expected
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
    with pytest.raises(RuntimeError, match="installer exited"):
        await agent.seed_agent_session(request, seed())
    session_id = seed().agent_session_id
    if cleanup_fails:
        assert agent._session_records[session_id].state.session.closing
        with pytest.raises(HTTPException):
            await agent.seed_agent_session(Request({"type": "http", "session": {}}), seed())
    else:
        assert not any(record.state is not None for record in agent._session_records.values())
    sandbox.exec.side_effect = None
    sandbox.disconnect.side_effect = None
    result = await agent.close_agent_session(
        Request({"type": "http", "session": {}}), AgentCloseSessionRequest(**close_body(session_id))
    )
    assert result.agent_session_id == session_id
    assert not any(record.state is not None for record in agent._session_records.values())
    sandbox.stop.assert_not_awaited()


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
    state.session.session_dir = str(directory)
    from responses_api_agents.codex_agent import sandbox_runner

    shutil.copyfile(sandbox_runner.__file__, directory / "sandbox_runner")
    sandbox.upload = AsyncMock(side_effect=shutil.copyfile)
    sandbox.download = AsyncMock(side_effect=shutil.copyfile)

    async def execute(command, **kwargs):
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
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=8)
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

    sandbox.exec.side_effect = execute
    return state, execute


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
    state.task = asyncio.create_task(state.execute({}, timeout=5, close_timeout=2))
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
    payload = {
        "directory": state.session.session_dir,
        "cwd": state.request.sandbox_access.workdir,
        "command": [sys.executable, "-c", "import json,sys; print(json.dumps({'prompt':sys.stdin.read()}))"],
        "env": {},
        "prompt": "real invocation",
    }
    raw = await state.execute(payload, timeout=3, close_timeout=2)
    _, event = json.loads(raw)
    assert event == {"prompt": "real invocation"}
    assert state.session.cleanup["cleanup_confirmed"] and state.session.cleanup["return_code"] == 0
    assert state.runtime_info.hostname
    await state.close(2)
    assert not Path(state.session.session_dir).exists()
    state.session.sandbox.stop.assert_not_awaited()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox contract")
async def test_close_confirms_cleanup_before_provider_cancellation(local_session):
    state, _ = local_session
    processes_path = Path(state.session.session_dir).parent / "children.json"
    # A completed tool leaves a service adopted before close. A direct child of
    # the still-running CLI would instead be an in-flight tool and must be stopped.
    service_tool = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
        "start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print(child.pid)"
    )
    code = (
        "import json,os,subprocess,sys,time; from pathlib import Path; "
        f"service_pid=int(subprocess.check_output([sys.executable,'-c',{service_tool!r}],text=True)); "
        f"Path({str(processes_path)!r}).write_text(json.dumps([os.getpid(),service_pid])); time.sleep(60)"
    )
    payload = {
        "directory": state.session.session_dir,
        "cwd": state.request.sandbox_access.workdir,
        "command": [sys.executable, "-c", code],
        "env": {},
        "prompt": "",
    }
    state.task = asyncio.create_task(state.execute(payload, timeout=30, close_timeout=3))
    processes = []
    try:
        async with asyncio.timeout(5):
            while not processes_path.exists():
                await asyncio.sleep(0.01)
        processes = json.loads(processes_path.read_text())
        processes.append(json.loads((Path(state.session.session_dir) / "runtime.json").read_text())["pid"])
        await state.close(3)
        assert state.session.closed and state.session.cleanup["cleanup_confirmed"] is True
        cli_pid, service_pid, wrapper_pid = processes
        assert all(not Path(f"/proc/{pid}").exists() for pid in (cli_pid, wrapper_pid))
        assert Path(f"/proc/{service_pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        state.session.sandbox.disconnect.assert_awaited_once()
        state.session.sandbox.stop.assert_not_awaited()
    finally:
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


@pytest.mark.parametrize("outcome", ["completed", "provider", "runtime", "output-limit", "wall-time"])
async def test_execution_outcomes_preserve_evidence_and_only_limits_are_gradable(setup, outcome):
    agent, sandbox = setup
    if outcome == "provider":
        sandbox.events = events(stop_reason="error")
    elif outcome == "runtime":
        sandbox.result["return_code"] = 7
    elif outcome == "output-limit":
        sandbox.result["return_code"] = 1
        rows = [json.loads(line) for line in events().splitlines()[:-1]]
        rows.append(
            [
                5.0,
                {
                    "type": "turn.failed",
                    "error": {
                        "message": "stream disconnected before completion: Incomplete response returned, reason: max_output_tokens"
                    },
                },
            ]
        )
        sandbox.events = "\n".join(json.dumps(row) for row in rows)
    elif outcome == "wall-time":
        sandbox.result.update(timed_out=True, return_code=-9)
        sandbox.events = "\n".join(events().splitlines()[:-1])
    request, session_id, task = await activate(agent, sandbox)
    if outcome in ("provider", "runtime"):
        with pytest.raises(HTTPException) as failed:
            await task
        assert failed.value.status_code == 502
        with pytest.raises(HTTPException) as replay:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert replay.value.detail == failed.value.detail
    else:
        result = await task
        assert result.status == ("completed" if outcome == "completed" else "incomplete")
        assert result.error is None
        assert result.output[-1].content[0].text == "Fixed"
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    invocation = closed.agent_observations.records[0]
    assert invocation.status == (
        "failed" if outcome in ("provider", "runtime") else "completed" if outcome == "completed" else "incomplete"
    )
    assert invocation.conversation[-1].content[0].text == "Fixed"
    sandbox.launch.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_setup_is_cleanup_only_and_preserves_original_exception(setup, cancelled):
    agent, sandbox = setup
    original = asyncio.CancelledError() if cancelled else RuntimeError("original install failure")
    ok = SimpleNamespace(error_type=None, return_code=0, stdout="", stderr="")
    sandbox.exec.side_effect = [ok, original, ok]
    sandbox.disconnect.side_effect = RuntimeError("cleanup unavailable")
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": seed().episode_id.capture_key}})
    with pytest.raises(type(original)) as raised:
        await agent.seed_agent_session(request, seed())
    assert raised.value is original
    assert request.session == {}
    request.session["agent_session_id"] = seed().agent_session_id
    with pytest.raises(HTTPException) as activation:
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert activation.value.status_code == 409
    sandbox.launch.assert_not_awaited()
    sandbox.exec.side_effect = None
    sandbox.disconnect.side_effect = None
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(seed().agent_session_id)))
    sandbox.stop.assert_not_awaited()


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
    module = "responses_api_agents.codex_agent.app"
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
        patch("responses_api_agents.codex_agent.app.AsyncSandbox", return_value=sandbox),
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
                stage == "install" and command.startswith("bash ") and "install_codex_runtime.sh" in command
            ):
                raise original
            return await execute(command, **kwargs)

        sandbox.exec.side_effect = fail_at_stage
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with (
        patch("responses_api_agents.codex_agent.app.AsyncSandbox", return_value=sandbox),
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
    module = "responses_api_agents.codex_agent.app"
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
        patch("responses_api_agents.codex_agent.app.AsyncSandbox") as factory,
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
    with pytest.raises(HTTPException, match="without turn.completed"):
        await task
    state = agent._session_records[session_id].state
    assert state.session.cleanup["cleanup_confirmed"] is True
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize(
    "budget,expected", [(None, 1800), ("900.0", 900), ("12.5", 12.5), ("3600", 1800), ("nan", 1800)]
)
def test_task_timeout_metadata_reaches_sandbox_supervisor(
    setup: tuple[CodexAgent, Sandbox], tmp_path: Path, budget: str | None, expected: float
) -> None:
    agent, sandbox = setup
    agent.config.timeout = 1800
    (tmp_path / "task.toml").write_text("[agent]\ntimeout_sec = 900.0\n")
    (tmp_path / "instruction.md").write_text("Fix the code")
    (tmp_path / "tests").mkdir()
    body = materialize_task(load_task(tmp_path), "timeout-test").task_input.responses_create_params
    if budget != "900.0":
        body.metadata = None if budget is None else {"agent_timeout_sec": budget}
    if body.metadata is not None:
        body.metadata["instance_id"] = "task-1"
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        response = client.post(
            "/ng-rollout/codex-smoke-a2/v1/responses", json=body.model_dump(mode="json", exclude_unset=True)
        )
        assert response.status_code == 200, response.text
        command = shlex.split(sandbox.launch.await_args.kwargs["command"])
        assert float(command[command.index("--timeout") + 1]) == expected
        assert agent.config.timeout == 1800
        assert response.json()["metadata"]["agent_timeout_sec"] == str(float(expected))
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text


def test_reasoning_effort_requires_an_explicit_opt_in(setup) -> None:
    agent, sandbox = setup
    agent.config.reasoning_effort = "max"
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert response.status_code == 422
    assert "model_server_forwards_reasoning_effort" in response.json()["detail"]
    sandbox.exec.assert_not_awaited()


def test_opted_in_reasoning_effort_reaches_the_sandbox_config(setup) -> None:
    agent, sandbox = setup
    agent.config.reasoning_effort = "max"
    agent.config.model_server_forwards_reasoning_effort = True
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Fix multiply"})
        assert result.status_code == 200, result.text
        config = tomllib.loads(sandbox.files[f"{sandbox.directory}/home/.codex/config.toml"])
        assert config["model_reasoning_effort"] == "max"
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200


@pytest.mark.parametrize("inherit", [False, True])
def test_sandbox_environment_inheritance_is_opt_in(setup, inherit) -> None:
    agent, sandbox = setup
    agent.config.inherit_sandbox_environment = inherit
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Fix multiply"})
        assert result.status_code == 200, result.text
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    assert payload["inherit_env"] is inherit
    # HOME and the XDG directories come from the sandbox either way; only CODEX_HOME is private.
    assert payload["env"] == {
        "CODEX_HOME": f"{sandbox.directory}/home/.codex",
        "OPENAI_API_KEY": "gym",  # pragma: allowlist secret
    }


@pytest.mark.parametrize("status", [400, 401, 429, 503])
async def test_native_model_error_is_reported_with_classification(setup, status):
    agent, sandbox = setup
    message = f"unexpected status {status}: provider error"
    sandbox.events = "\n".join(
        json.dumps([float(i), event])
        for i, event in enumerate(
            [
                {
                    "type": "item.completed",
                    "item": {
                        "type": "error",
                        "message": "Model metadata for `test-model` not found. Defaulting to fallback metadata; "
                        "this can degrade performance and cause issues.",
                    },
                },
                {"type": "turn.started"},
                {"type": "error", "message": f"Reconnecting... 1/5 ({message})"},
                {"type": "turn.failed", "error": {"message": message}},
            ]
        )
    )
    sandbox.result["return_code"] = 1
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    assert response.status == "failed"
    assert response.error.message.endswith(message)
    assert json.loads(response.metadata["ng_agent_outcome"]) == {
        "failure_kind": "model_request_error",
        "status_code": status,
    }
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert closed.agent_observations.records[0].status == "failed"


async def test_native_failure_without_http_status_reports_unknown(setup):
    agent, sandbox = setup
    message = json.dumps({"error": {"message": "rejected", "code": "400"}})
    sandbox.events = json.dumps([1.0, {"type": "turn.failed", "error": {"message": message}}])
    sandbox.result["return_code"] = 1
    request, session_id, task = await activate(agent, sandbox)
    response = await task
    assert response.status == "failed"
    assert response.error.message == message
    assert json.loads(response.metadata["ng_agent_outcome"]) == {"failure_kind": "unknown"}
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


def native_transcript(items):
    return "\n".join(
        json.dumps({"type": "response_item", "timestamp": f"2026-10-19T12:00:0{index}Z", "payload": item})
        for index, item in enumerate(items)
    )


@pytest.mark.parametrize("custom", [False, True])
def test_session_uses_native_call_identity_arguments_and_results(setup, custom):
    agent, sandbox = setup
    native_output = "Process exited with code 7\nOutput:\noriginal result"
    transcript = native_transcript(
        [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
            {
                "type": "custom_tool_call" if custom else "function_call",
                "call_id": "provider-call",
                "name": "apply_patch" if custom else "exec_command",
                ("input" if custom else "arguments"): "*** Begin Patch\n*** End Patch"
                if custom
                else '{"cmd":"exit 7"}',
            },
            {
                "type": "custom_tool_call_output" if custom else "function_call_output",
                "call_id": "provider-call",
                "output": native_output,
            },
        ]
    )
    if custom:
        transcript += "\n" + json.dumps(
            {"type": "event_msg", "payload": {"type": "patch_apply_end", "call_id": "provider-call", "success": False}}
        )
    original_publish = sandbox.publish

    def publish():
        original_publish()
        sandbox.files[f"{sandbox.directory}/native_session.jsonl"] = transcript
        sandbox.files[f"{sandbox.directory}/model_attempts.jsonl"] = '{"status":200}\n'

    sandbox.publish = publish
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200, response.text
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
        assert "--ephemeral" not in payload["command"]
        assert payload["invocation_id"] == "codex-smoke-a2"
        assert payload["model_base_url"] == "http://model.example:9000/ng-rollout/codex-smoke-a2/v1"
        config = tomllib.loads(sandbox.files[f"{sandbox.directory}/home/.codex/config.toml"])
        assert config["model_providers"]["gym"]["http_headers"] == {"x-session-id": "codex-smoke-a2"}
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id)).json()
    bundle = closed["agent_observations"]
    invocation, tool = bundle["records"]
    assert invocation["conversation"][1]["call_id"] == tool["tool_call_id"] == "provider-call"
    assert invocation["conversation"][1]["input" if custom else "arguments"] == (
        "*** Begin Patch\n*** End Patch" if custom else '{"cmd":"exit 7"}'
    )
    assert invocation["conversation"][2]["output"] == native_output
    assert tool["status"] == "failed" and tool["error_type"] == ("tool_error" if custom else "nonzero_exit")
    assert not any(gap["code"] == "model_call_join_key_unavailable" for gap in bundle["gaps"])


@pytest.mark.parametrize(
    "statuses,timed_out,returncode,worker_error,extra_error,accepted",
    [
        ([400], False, 1, None, None, True),
        ([200, 400], False, 1, None, None, True),
        ([429, 400], False, 1, None, None, True),
        ([400, 200], False, 1, None, None, True),
        ([401], False, 1, None, None, True),
        ([403], False, 1, None, None, True),
        ([429, 429, 429], False, 1, None, None, True),
        ([500], False, 1, None, None, True),
        ([], False, 1, None, None, True),
        ([429, None], False, 1, None, None, True),
        ([400], True, 1, None, None, False),
        ([400], False, None, None, None, False),
        ([400], False, 0, None, None, False),
        ([400], False, 7, None, None, False),
        ([400], False, 1, "worker failed", None, False),
        ([400], False, 1, None, "sandbox state lost", False),
    ],
)
def test_terminal_model_error_preserves_provider_and_worker_error_boundary(
    setup, statuses, timed_out, returncode, worker_error, extra_error, accepted
):
    agent, sandbox = setup
    sandbox.result.update(return_code=returncode, timed_out=timed_out, error=worker_error)
    records = [
        {"type": "turn.started"},
        {"type": "turn.failed", "error": {"message": "model rejected request"}},
    ]
    if extra_error:
        records.insert(1, {"type": "item.completed", "item": {"type": "error", "message": extra_error}})
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(records))
    original_publish = sandbox.publish

    def publish():
        original_publish()
        sandbox.files[f"{sandbox.directory}/native_session.jsonl"] = native_transcript(
            [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]}]
        )
        sandbox.files[f"{sandbox.directory}/model_attempts.jsonl"] = "\n".join(
            json.dumps({"status": status}) for status in statuses
        )

    sandbox.publish = publish
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == (200 if accepted else 502), response.text
        if accepted:
            body = response.json()
            assert body["status"] == "failed"
            assert body["error"] == {
                "code": "rate_limit_exceeded" if statuses and statuses[-1] == 429 else "server_error",
                "message": "model rejected request",
            }
            assert body["output"] == []
            outcome = json.loads(body["metadata"]["ng_agent_outcome"])
            if statuses and statuses[-1] is None:
                assert outcome == {"failure_kind": "model_transport_error"}
            elif statuses and statuses[-1] >= 400:
                assert outcome == {"failure_kind": "model_request_error", "status_code": statuses[-1]}
            else:
                assert outcome == {"failure_kind": "unknown"}
            if 200 in statuses or not statuses:
                assert body["usage"] is None
            else:
                assert body["usage"]["total_tokens"] == 0
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id)).json()
        invocation = closed["agent_observations"]["records"][0]
        assert invocation["status"] == "failed"
        if accepted:
            assert invocation["error_type"] == "provider_error"
        assert all(item.get("role") == "user" for item in invocation["conversation"])


@pytest.mark.parametrize("usage_source", ["native", "cli", "missing"])
def test_later_rejection_preserves_real_tool_output_and_available_usage(setup, usage_source):
    agent, sandbox = setup
    sandbox.result["return_code"] = 1
    # Pinned 0.144.4 emits this shape after a successful tool call then HTTP 400:
    # no turn.completed, but the native token_count retains the completed call's usage.
    rejection = '{"error":{"message":"controlled model failure","type":"conformance_error","code":"400"}}'
    tool = {"id": "item_1", "type": "command_execution", "command": "printf retained", "status": "in_progress"}
    records = [
        {"type": "turn.started"},
        {"type": "item.started", "item": tool},
        {
            "type": "item.completed",
            "item": tool | {"status": "completed", "exit_code": 0, "aggregated_output": "retained"},
        },
        {"type": "error", "message": rejection},
        {"type": "turn.failed", "error": {"message": rejection}},
    ]
    if usage_source == "cli":
        records.insert(-2, {"type": "turn.completed", "usage": {"input_tokens": 17, "output_tokens": 5}})
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(records))
    native_output = "Process exited with code 0\nOutput:\nretained"
    transcript = native_transcript(
        [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
            {
                "type": "function_call",
                "call_id": "provider-call",
                "name": "exec_command",
                "arguments": '{"cmd":"printf retained"}',
            },
            {"type": "function_call_output", "call_id": "provider-call", "output": native_output},
        ]
    )
    if usage_source != "missing":
        transcript += "\n" + json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 20,
                            "output_tokens": 8,
                            "total_tokens": 28,
                            "cached_input_tokens": 3,
                            "reasoning_output_tokens": 2,
                        }
                    },
                },
            }
        )
    original_publish = sandbox.publish

    def publish():
        original_publish()
        sandbox.files[f"{sandbox.directory}/native_session.jsonl"] = transcript
        sandbox.files[f"{sandbox.directory}/model_attempts.jsonl"] = '{"status":200}\n{"status":400}\n'

    sandbox.publish = publish
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "failed"
        assert body["error"] == {"code": "server_error", "message": rejection}
        assert [item["type"] for item in body["output"]] == ["function_call", "function_call_output"]
        assert body["output"][1]["output"] == "retained"
        if usage_source == "native":
            assert body["usage"] == {
                "input_tokens": 20,
                "output_tokens": 8,
                "total_tokens": 28,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 2},
            }
        elif usage_source == "cli":
            assert body["usage"]["input_tokens"] == 17
            assert body["usage"]["output_tokens"] == 5
            assert body["usage"]["total_tokens"] == 22
        else:
            assert body["usage"] is None
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id)).json()
        invocation, observed_tool = closed["agent_observations"]["records"]
        assert invocation["status"] == "failed"
        assert observed_tool["status"] == "completed"
        assert invocation["conversation"][1]["call_id"] == observed_tool["tool_call_id"] == "provider-call"
        assert invocation["conversation"][2]["output"] == native_output


@pytest.mark.parametrize("usage_source", ["native", "missing"])
def test_graded_failure_reports_native_usage_or_none(setup, usage_source):
    """A killed run that is graded reports the native cumulative usage, or none, never the CLI's zero."""
    agent, sandbox = setup
    agent.config.verify_after_agent_error = True
    sandbox.status = AsyncMock(return_value=SandboxStatus.RUNNING)
    sandbox.result["return_code"] = -9  # killed after a completed tool call, before turn.completed
    tool = {"id": "item_1", "type": "command_execution", "command": "printf retained", "status": "in_progress"}
    records = [
        {"type": "turn.started"},
        {"type": "item.started", "item": tool},
        {"type": "item.completed", "item": tool | {"status": "completed", "exit_code": 0, "aggregated_output": "ok"}},
    ]
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(records))
    transcript = native_transcript(
        [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]}]
    )
    if usage_source == "native":
        transcript += "\n" + json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 20,
                            "output_tokens": 8,
                            "total_tokens": 28,
                            "cached_input_tokens": 3,
                            "reasoning_output_tokens": 2,
                        }
                    },
                },
            }
        )
    original_publish = sandbox.publish

    def publish():
        original_publish()
        sandbox.files[f"{sandbox.directory}/native_session.jsonl"] = transcript
        sandbox.files[f"{sandbox.directory}/model_attempts.jsonl"] = '{"status":200}\n'

    sandbox.publish = publish
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        client.post("/v1/agent_sessions/close", json=close_body(session_id))
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["status"], body["error"]) == ("failed", None)
    assert body["metadata"]["graded_agent_error"].startswith("exit -9: ")
    assert "ng_agent_outcome" not in body["metadata"]
    if usage_source == "native":
        assert body["usage"] == {
            "input_tokens": 20,
            "output_tokens": 8,
            "total_tokens": 28,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        }
    else:
        assert body["usage"] is None


async def test_activation_failure_takes_precedence_over_recorded_rejection(setup):
    agent, sandbox = setup
    state = await agent._seed_agent_session_state(seed())
    state.session.cleanup = sandbox.result | {"return_code": 1}
    state.session.artifacts = json.dumps(
        [1.0, {"type": "turn.failed", "error": {"message": "model rejected request"}}]
    )
    state.model_attempts = [{"status": 200}, {"status": 400}]
    with patch.object(state, "execute", AsyncMock(side_effect=RuntimeError("sandbox connection lost"))):
        with pytest.raises(RuntimeError, match="sandbox connection lost"):
            await agent._sandbox_response(
                state, NeMoGymResponseCreateParamsNonStreaming(input="task"), prompt="task", system=""
            )
    assert state.observations.records[0].status == "incomplete"


@pytest.mark.asyncio
async def test_failed_native_snapshot_preserves_collected_cli_events(setup):
    agent, sandbox = setup
    state = await agent._seed_agent_session_state(seed())
    original_execute = sandbox.execute

    async def execute(command, **kwargs):
        if "--snapshot" in command:
            raise OSError("snapshot transport failed")
        return await original_execute(command, **kwargs)

    sandbox.exec.side_effect = execute
    try:
        response = await agent._sandbox_response(
            state, NeMoGymResponseCreateParamsNonStreaming(input="task"), prompt="task", system=""
        )
        assert response.output[-1].content[0].text == "Fixed"
        assert state.observations.records[0].conversation[-1].content[0].text == "Fixed"
        assert any(gap.code == "model_call_join_key_unavailable" for gap in state.observations.gaps)
    finally:
        await state.close(1)


@pytest.mark.parametrize("transport_error", ["timed out", "IncompleteRead(0 bytes read, 20 more expected)", None])
def test_post_header_failure_reaches_outcome_policy(setup, transport_error):
    from nemo_gym.agent_outcome import OutcomeAction, agent_outcome_policy
    from nemo_gym.core import NeMoGymResponse

    agent, sandbox = setup
    sandbox.result["return_code"] = 1
    records = [
        {"type": "turn.started"},
        {"type": "turn.failed", "error": {"message": "stream disconnected before completion: server error"}},
    ]
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(records))
    original_publish = sandbox.publish

    def publish():
        original_publish()
        attempt = {"status": 200}
        if transport_error is not None:
            attempt["transport_error"] = transport_error
        sandbox.files[f"{sandbox.directory}/model_attempts.jsonl"] = json.dumps(attempt) + "\n"

    sandbox.publish = publish
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "failed"
        assert body["usage"] is None
        outcome = json.loads(body["metadata"]["ng_agent_outcome"])
        assert outcome == {"failure_kind": "model_transport_error" if transport_error is not None else "unknown"}
        policy = agent_outcome_policy(NeMoGymResponse.model_validate(body))
        assert policy == OutcomeAction.RETRY
        client.post("/v1/agent_sessions/close", json=close_body(session_id)).raise_for_status()


# Native Display strings from codex-rs/protocol/src/error.rs at rust-v0.144.4.
@pytest.mark.parametrize(
    "attempt,message,expected",
    [
        (
            {"status": 400},
            '{"error":{"message":"invalid input","code":"bad_request"}}',
            {"failure_kind": "model_request_error", "status_code": 400},
        ),
        (
            {"status": 429},
            "exceeded retry limit, last status: 429 Too Many Requests",
            {"failure_kind": "model_request_error", "status_code": 429},
        ),
        (
            {"status": 500},
            "We're currently experiencing high demand, which may cause temporary errors.",
            {"failure_kind": "model_request_error", "status_code": 500},
        ),
        (
            {"status": 503},
            "Selected model is at capacity. Please try a different model.",
            {"failure_kind": "model_request_error", "status_code": 503},
        ),
        ({"status": None, "error": "timed out"}, "request timed out", {"failure_kind": "model_transport_error"}),
        (
            {"status": 429, "transport_error": "timed out"},
            "request timed out",
            {"failure_kind": "model_transport_error"},
        ),
        (
            {"status": 200, "transport_error": "timed out"},
            "request timed out",
            {"failure_kind": "model_transport_error"},
        ),
    ],
)
def test_native_errors_use_captured_attempt_facts(setup, attempt, message, expected):
    from nemo_gym.agent_outcome import OutcomeAction, agent_outcome_policy
    from nemo_gym.core import NeMoGymResponse

    agent, sandbox = setup
    sandbox.result["return_code"] = 1
    sandbox.events = json.dumps([1.0, {"type": "turn.failed", "error": {"message": message}}])
    original_publish = sandbox.publish

    def publish():
        original_publish()
        sandbox.files[f"{sandbox.directory}/model_attempts.jsonl"] = json.dumps(attempt) + "\n"

    sandbox.publish = publish
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200
        result = NeMoGymResponse.model_validate(response.json())
        assert result.status == "failed"
        assert result.error.message == message
        assert result.error.code == ("rate_limit_exceeded" if expected.get("status_code") == 429 else "server_error")
        assert json.loads(result.metadata["ng_agent_outcome"]) == expected
        assert agent_outcome_policy(result) == (
            OutcomeAction.VERIFY if attempt["status"] == 400 else OutcomeAction.RETRY
        )
        client.post("/v1/agent_sessions/close", json=close_body(session_id)).raise_for_status()
