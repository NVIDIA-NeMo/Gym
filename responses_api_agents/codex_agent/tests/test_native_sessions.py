# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
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
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.codex_agent.app import CodexAgent, CodexAgentConfig


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
        if "exec python3 -I " in command and "process_supervisor.py" in command:
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
        assert "sandbox_runner.py" in kwargs["command"]
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
        patch(f"{module}.ensure_codex", side_effect=AssertionError("native sessions must not install host Codex")),
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider"),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
    ):
        agent = CodexAgent(config=config, server_client=client)
        yield agent, sandbox


def close_body(session_id):
    return {"agent_session_id": session_id, "episode_id": seed().episode_id.model_dump()}


def test_http_native_flow_runs_codex_in_borrowed_sandbox(setup):
    agent, sandbox = setup
    with patch.object(agent, "_run_codex", AsyncMock(side_effect=AssertionError("host Codex must not run"))):
        with TestClient(agent.setup_webserver()) as client:
            created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
            assert created.status_code == 200, created.text
            session_id = created.json()["agent_session_id"]
            directory = agent._session_records[session_id].state.session.session_dir
            assert f"{directory}/process_supervisor.py" not in sandbox.files
            installer = f"{directory}/install_codex_runtime.sh"
            assert installer in sandbox.files
            assert agent.config.resources_server is None
            assert not sandbox.launch.called
            assert not any(path.startswith("/app/") for path in sandbox.files)
            install_call = sandbox.exec.await_args_list[1]
            assert install_call.args[0] == (f"bash {installer} /tmp/nemo-gym-codex-node-22.19.0-0.144.4 0.144.4")
            assert install_call.kwargs["timeout_s"] == agent.config.sandbox_install_timeout_seconds
            result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Fix the code"})
            assert result.status_code == 200, result.text
            assert f"{directory}/process_supervisor.py" in sandbox.files
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
def test_native_usage_details_without_measured_positive_counts_are_unknown(setup, value) -> None:
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
def test_native_usage_details_require_known_counts_from_every_completed_turn(setup, unknown_turn) -> None:
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


def test_native_custom_model_context_budget_reaches_sandbox_config(setup) -> None:
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


def test_shared_seed_rejects_multiple_workers_before_initialization(setup):
    agent, sandbox = setup
    agent.config.num_workers = 2
    with TestClient(agent.setup_webserver()) as client:
        # Worker count is server configuration, checked by the shared route before seed.
        with pytest.raises(ValueError, match="Agent sessions require num_workers=1"):
            client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert not agent._session_records
    sandbox.exec.assert_not_awaited()


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


async def test_failed_cleanup_keeps_handles_and_prevents_close(setup):
    agent, sandbox = setup
    sandbox.result["cleanup_confirmed"] = False
    request, session_id, task = await activate(agent, sandbox)
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await task
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert agent._session_records[session_id].state is not None
    sandbox.disconnect.assert_not_awaited()


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


@pytest.mark.parametrize("provider_failure", [False, True])
async def test_independent_configs_collect_flat_rows_through_environment_run(setup, monkeypatch, provider_failure):
    """Exercise the checked-in recipe through collector, real environment, and real Codex lifecycle."""
    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
    from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
    from nemo_gym.global_config import GlobalConfigDictParser
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
    from nemo_gym.server_utils import BaseServerConfig

    agent, sandbox = setup
    if provider_failure:
        sandbox.events = events(stop_reason="error")
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
                assert body["response"]["usage"]["total_tokens"] == 22
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
        assert result["response"]["usage"]["total_tokens"] == 22
        assert result["ng_agent_observations"]["source"] == "codex"
    assert close_receipts[-1].agent_observations.source == "codex"
    assert close_receipts[-1].agent_observations.records[0].status == ("failed" if provider_failure else "completed")
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

    shutil.copyfile(sandbox_runner.__file__, directory / "sandbox_runner.py")
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
        if "--receipt" in command and "process_supervisor.py" in command:
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
    code = (
        "import json,os,subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        f"Path({str(processes_path)!r}).write_text(json.dumps([os.getpid(),child.pid])); time.sleep(60)"
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
        assert all(not Path(f"/proc/{pid}").exists() for pid in processes)
        state.session.sandbox.disconnect.assert_awaited_once()
        state.session.sandbox.stop.assert_not_awaited()
    finally:
        if not state.session.closed:
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
def test_sandbox_source_controls_ownership_and_native_routing(setup, owned):
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
