# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.codex_agent.app import CodexAgent, CodexAgentConfig
from responses_api_agents.codex_agent.sandbox import CodexSandboxResult


def seed() -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
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
        self.exec = AsyncMock(return_value=SimpleNamespace(error_type=None, return_code=0, stdout="", stderr=""))
        self.stop = AsyncMock()
        self.disconnect = AsyncMock()
        self.runner = SimpleNamespace(
            wait_exit=AsyncMock(side_effect=self.wait_exit),
            send_signal=AsyncMock(side_effect=self.signal),
            close=AsyncMock(),
        )
        self.pty = SimpleNamespace(create=AsyncMock(side_effect=self.create))

    async def upload(self, source, destination):
        self.files[destination] = Path(source).read_text()

    async def download(self, source, destination):
        Path(destination).write_text(self.files[source])

    async def create(self, **kwargs):
        payload_path = next(path for path in self.files if path.endswith("/input.json"))
        payload = json.loads(self.files[payload_path])
        assert payload["cwd"] == "/app"
        assert kwargs["cwd"] == "/app"
        assert "sandbox_runner.py" in kwargs["command"]
        self.directory = payload["directory"]
        self.started.set()
        if not self.blocked:
            self.exited.set()
        return self.runner

    async def signal(self, name):
        assert name == "SIGTERM"
        self.result["timed_out"] = True
        self.exited.set()

    async def wait_exit(self):
        await self.exited.wait()
        self.files[f"{self.directory}/result.json"] = json.dumps(self.result)
        self.files[f"{self.directory}/events.jsonl"] = self.events
        return 0


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
            directory = f"/tmp/nemo-gym-codex-sessions/{session_id}"
            installer = f"{directory}/install_codex_runtime.sh"
            assert installer in sandbox.files
            assert agent.config.resources_server is None
            assert not sandbox.pty.create.called
            assert not any(path.startswith("/app/") for path in sandbox.files)
            install_call = sandbox.exec.await_args_list[1]
            assert install_call.args[0] == (f"bash {installer} /tmp/nemo-gym-codex-node-22.19.0-0.144.4 0.144.4")
            assert install_call.kwargs["timeout_s"] == agent.config.sandbox_install_timeout_seconds
            result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Fix the code"})
            assert result.status_code == 200, result.text
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
    assert not agent._sandbox_sessions
    assert agent._local_setup_task is None
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    agent.server_client.post.assert_not_called()


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


@pytest.mark.parametrize("option", ["no-sandbox", "worker", "required-tool", "no-model", "unpinned"])
def test_unsupported_seed_rejected_before_connection(setup, option):
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    if option == "no-sandbox":
        body["sandbox_access"] = None
    elif option == "worker":
        agent.config.num_workers = 2
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


def test_cookie_identity_and_single_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 409
        assert client.post("/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/ng-rollout/wrong-a2/v1/responses", json={"input": "task"}).status_code == 409
        bad = close_body(session_id)
        bad["episode_id"]["attempt"] = 99
        assert client.post("/v1/agent_sessions/close", json=bad).status_code == 409
        assert client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"}).status_code == 200
        assert client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.pty.create.assert_awaited_once()


@pytest.mark.parametrize("reason,expected", [("error", "failed"), ("aborted", "failed")])
def test_failed_or_partial_codex_output_is_preserved(setup, reason, expected):
    agent, sandbox = setup
    sandbox.events = events(stop_reason=reason)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200
        assert result.json()["status"] == expected
        assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200


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
        assert body["status"] == "failed"
        assert body["error"]["message"] == message
        assert body["output"][0]["summary"][0]["text"] == "Inspect first"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
        observations = closed.json()["agent_observations"]
        assert observations["records"][0]["status"] == "failed"
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
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["output"][-1]["content"][0]["text"] == "Fixed"
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
            assert body["status"] == "failed"
            assert diagnostic["item"]["message"] in body["error"]["message"]
            if condition == "failed-turn":
                assert "model error" in body["error"]["message"]
            if condition == "later-error":
                assert "Model call failed" in body["error"]["message"]
            assert not warnings
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
    sandbox.pty.create.assert_not_awaited()


def test_rejected_request_does_not_consume_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/codex-smoke-a2/v1/responses"
        assert client.post(path, json={"input": "task", "temperature": 0.2}).status_code == 422
        sandbox.pty.create.assert_not_awaited()
        accepted = client.post(path, json={"input": "task"})
        assert accepted.status_code == 200, accepted.text
        assert client.post(path, json={"input": "task"}).status_code == 409
    sandbox.pty.create.assert_awaited_once()


def test_no_session_keeps_existing_local_path(setup):
    agent, sandbox = setup
    with patch.object(agent, "_create_response", AsyncMock(side_effect=RuntimeError("legacy path reached"))) as legacy:
        with TestClient(agent.setup_webserver()) as client:
            with pytest.raises(RuntimeError, match="legacy path reached"):
                client.post("/v1/responses", json={"input": "task"})
        legacy.assert_awaited_once()
    sandbox.pty.create.assert_not_awaited()


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
    sandbox.runner.send_signal.assert_awaited_once_with("SIGTERM")
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
    assert session_id in agent._sandbox_sessions
    sandbox.disconnect.assert_not_awaited()
    sandbox.runner.close.assert_not_awaited()


async def test_disconnect_failure_retains_session_for_retry(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    sandbox.disconnect.side_effect = [RuntimeError("provider unavailable"), None]
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in agent._sandbox_sessions
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id not in agent._sandbox_sessions
    sandbox.runner.close.assert_awaited_once()


def test_cleanup_receipt_is_required():
    with pytest.raises(ValueError):
        CodexSandboxResult.model_validate({"return_code": 0, "error": None})


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
    monkeypatch.setattr("responses_api_agents.codex_agent.app.monotonic", lambda: 100.0)
    with TestClient(agent.setup_webserver()) as client:

        def seed_and_close(index):
            client.cookies.clear()
            body = seed().model_dump(mode="json")
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
    monkeypatch.setattr("responses_api_agents.codex_agent.app.monotonic", lambda: clock[0])
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
    assert not agent._closed_sandbox_sessions
    with patch.object(agent, "_create_response", AsyncMock(side_effect=AssertionError("host fallback"))):
        with pytest.raises(HTTPException) as error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert error.value.status_code == 409
    sandbox.disconnect.assert_awaited_once()


async def test_close_retry_window_starts_after_cleanup(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.codex_agent.app.monotonic", lambda: clock[0])
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
    assert first_result is second_result
    sandbox.disconnect.assert_awaited_once()


async def test_seed_prunes_expired_close_receipts(setup, monkeypatch):
    agent, _ = setup
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.codex_agent.app.monotonic", lambda: clock[0])
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    clock[0] += agent.config.session_close_retry_window_seconds
    request.session.clear()
    await agent.seed_agent_session(request, seed())
    assert not agent._closed_sandbox_sessions


@pytest.mark.parametrize("window", [0, -1, float("inf")])
def test_close_retry_window_must_be_positive_and_finite(setup, window):
    agent, _ = setup
    with pytest.raises(ValidationError):
        CodexAgentConfig(**(agent.config.model_dump() | {"session_close_retry_window_seconds": window}))


async def test_unknown_launch_outcome_fails_closed(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "codex-smoke-a2"}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    sandbox.pty.create.side_effect = TimeoutError("lost launch response")
    with pytest.raises(TimeoutError):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in agent._sandbox_sessions
    sandbox.disconnect.assert_not_awaited()
    sandbox.stop.assert_not_awaited()


async def test_install_failure_disconnects_without_stopping_owner(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(error_type=None, return_code=0),
        SimpleNamespace(error_type=None, return_code=1, stderr="npm failed"),
        SimpleNamespace(error_type=None, return_code=0),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="npm failed"):
        await agent.seed_agent_session(request, seed())
    assert not agent._sandbox_sessions
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
    assert not agent._sandbox_sessions
    assert not request.session
    sandbox.pty.create.assert_not_awaited()
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
            sandbox.exec.assert_awaited_once()
        return
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    sandbox.exec.return_value = error
    with pytest.raises(RuntimeError, match="Could not remove Codex session files"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in agent._sandbox_sessions
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
    other = seed().model_copy(update={"episode_id": EpisodeId(rollout_id="other")})
    b = (await agent.seed_agent_session(second, other)).agent_session_id
    assert a != b
    assert agent._sandbox_sessions[a].directory != agent._sandbox_sessions[b].directory
    with pytest.raises(HTTPException):
        await agent.close_agent_session(second, AgentCloseSessionRequest(**close_body(a)))
    assert a in agent._sandbox_sessions and b in agent._sandbox_sessions


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
        sandbox.pty.create.assert_not_awaited()
        accepted = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "task"})
        assert accepted.status_code == 200, accepted.text


@pytest.mark.parametrize("marker", [None, 0, [], {}, ""])
async def test_malformed_session_markers_cannot_fall_back_to_host(setup, marker):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {"nemo_gym_codex_sandbox_session": marker}})
    with patch.object(agent, "_create_response", AsyncMock(side_effect=AssertionError("host fallback"))) as legacy:
        with pytest.raises(HTTPException) as error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert error.value.status_code == 409
        from responses_api_agents.codex_agent.app import CodexAgentRunRequest

        with pytest.raises(HTTPException) as error:
            await agent.run(request, CodexAgentRunRequest(responses_create_params={"input": "task"}))
        assert error.value.status_code == 409
        legacy.assert_not_awaited()
    sandbox.pty.create.assert_not_awaited()


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
