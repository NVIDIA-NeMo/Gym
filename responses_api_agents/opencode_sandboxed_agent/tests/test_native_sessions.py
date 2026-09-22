# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
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
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent, OpenCodeSandboxedAgentConfig
from responses_api_agents.opencode_sandboxed_agent.sandbox import OpenCodeSandboxResult


def seed() -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        episode_id=EpisodeId(rollout_id="opencode-smoke", attempt=2),
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


def events() -> str:
    return json.dumps(
        {
            "messages": [
                {"info": {"role": "user"}, "parts": [{"type": "text", "text": "task"}]},
                {
                    "info": {
                        "role": "assistant",
                        "time": {"completed": 100},
                        "finish": "stop",
                        "tokens": {"input": 10, "output": 2, "reasoning": 1, "cache": {"read": 3}, "total": 12},
                    },
                    "parts": [
                        {"type": "reasoning", "text": "Inspect"},
                        {
                            "type": "tool",
                            "callID": "tool-1",
                            "tool": "bash",
                            "state": {"input": {"command": "pwd"}, "output": "/app"},
                        },
                        {"type": "text", "text": "Fixed"},
                    ],
                },
            ]
        }
    )


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
        self.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None))
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
        self.files[f"{self.directory}/export.json"] = self.events
        return 0


@pytest.fixture
def setup():
    sandbox = Sandbox()
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {"policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}}
    )
    client._build_server_base_url.return_value = "http://model.example:9000"
    config = OpenCodeSandboxedAgentConfig(
        name="pi",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        model_server={"type": "responses_api_models", "name": "policy"},
        opencode_version="1.17.11",
        resources_server={"type": "resources_servers", "name": "resources"},
        sandbox_provider="unused",
        sandbox_config={},
        sandbox_timeout=30,
        opencode_max_context_window=32000,
        session_close_timeout_seconds=1,
    )
    module = "responses_api_agents.opencode_sandboxed_agent.app"
    with (
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider"),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
    ):
        agent = OpenCodeSandboxedAgent(config=config, server_client=client)
        yield agent, sandbox


def close_body(session_id):
    return {"agent_session_id": session_id, "episode_id": seed().episode_id.model_dump()}


def test_http_native_flow_runs_opencode_in_borrowed_sandbox(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        assert not sandbox.pty.create.called
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "Fix the code"})
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["status"] == "completed"
        assert [item["type"] for item in body["output"]] == [
            "reasoning",
            "function_call",
            "function_call_output",
            "message",
        ]
        assert body["usage"]["total_tokens"] == 16
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
        assert payload["prompt"] == "Fix the code"
        assert payload["command"][0].endswith("nemo-gym-opencode-runtime-1.17.11/opencode")
        assert payload["env"]["HOME"].startswith("/tmp/")
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        assert (
            config["provider"]["nemo_gym"]["options"]["baseURL"]
            == "http://model.example:9000/ng-rollout/opencode-smoke-a2/v1"
        )
        assert client.post("/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"}).status_code == 409
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    assert not agent._native_sessions
    assert not any(path.startswith("/app/") for path in sandbox.files)
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize(
    "override",
    [
        {"max_output_tokens": 123},
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
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task", **override})
        assert result.status_code == 422, result.text
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.pty.create.assert_not_awaited()


def test_rejected_request_does_not_consume_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/opencode-smoke-a2/v1/responses"
        assert client.post(path, json={"input": "task", "temperature": 0.2}).status_code == 422
        sandbox.pty.create.assert_not_awaited()
        accepted = client.post(path, json={"input": "task"})
        assert accepted.status_code == 200, accepted.text
        assert client.post(path, json={"input": "task"}).status_code == 409
    sandbox.pty.create.assert_awaited_once()


async def activate(agent, sandbox):
    request = Request(
        {"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": "opencode-smoke-a2"}}
    )
    seeded = await agent.seed_agent_session(request, seed())
    task = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.wait_for(sandbox.started.wait(), 2)
    return request, seeded.agent_session_id, task


async def test_close_cancels_active_opencode_before_detaching(setup):
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
    assert (await task).status == "failed"
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in agent._native_sessions
    sandbox.disconnect.assert_not_awaited()
    sandbox.runner.close.assert_not_awaited()


async def test_disconnect_failure_retains_session_for_retry(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    sandbox.disconnect.side_effect = [RuntimeError("provider unavailable"), None]
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in agent._native_sessions
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id not in agent._native_sessions
    sandbox.runner.close.assert_awaited_once()


def test_cleanup_receipt_is_required():
    with pytest.raises(ValueError):
        OpenCodeSandboxResult.model_validate({"return_code": 0, "error": None})


def test_instructions_and_text_parts_reach_opencode(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        result = client.post(
            "/ng-rollout/opencode-smoke-a2/v1/responses",
            json={
                "instructions": "outer",
                "input": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": [{"type": "input_text", "text": "task"}]},
                ],
            },
        )
        assert result.status_code == 200, result.text
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        assert sandbox.files[config["instructions"][0]] == "outer\n\nsystem"
        assert payload["prompt"] == "task"
        assert config["enabled_providers"] == ["nemo_gym"]


def test_http_close_retry_survives_other_session_closes(setup, monkeypatch):
    agent, sandbox = setup
    monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.monotonic", lambda: 100.0)
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
    monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.monotonic", lambda: clock[0])
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
    assert not agent._closed_native_sessions
    with pytest.raises(HTTPException) as error:
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert error.value.status_code == 409
    sandbox.disconnect.assert_awaited_once()


async def test_close_retry_window_starts_after_cleanup(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.monotonic", lambda: clock[0])
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
    monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.monotonic", lambda: clock[0])
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    clock[0] += agent.config.session_close_retry_window_seconds
    await agent.seed_agent_session(request, seed())
    assert not agent._closed_native_sessions


@pytest.mark.parametrize("window", [0, -1, float("inf")])
def test_close_retry_window_must_be_positive_and_finite(setup, window):
    agent, _ = setup
    with pytest.raises(ValidationError):
        OpenCodeSandboxedAgentConfig(**(agent.config.model_dump() | {"session_close_retry_window_seconds": window}))


async def test_unknown_launch_outcome_fails_closed(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "opencode-smoke-a2"}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    sandbox.pty.create.side_effect = TimeoutError("lost launch response")
    result = await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert result.status == "failed"
    assert "lost launch response" in result.error.message
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in agent._native_sessions
    sandbox.disconnect.assert_not_awaited()
    sandbox.stop.assert_not_awaited()


async def test_install_failure_disconnects_without_stopping_owner(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        SimpleNamespace(return_code=1, stderr="curl failed", stdout="", error_type=None),
        SimpleNamespace(return_code=0),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="curl failed"):
        await agent.seed_agent_session(request, seed())
    assert not agent._native_sessions
    assert not request.session
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_cancelled_install_never_publishes_session_or_launches_opencode(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        asyncio.CancelledError(),
        SimpleNamespace(return_code=0),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(asyncio.CancelledError):
        await agent.seed_agent_session(request, seed())
    assert not agent._native_sessions
    assert not request.session
    sandbox.pty.create.assert_not_awaited()
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_native_usage_restores_cached_and_reasoning_tokens_across_subagents():
    export = {
        "usage_messages": [
            {
                "role": "assistant",
                "tokens": {"input": 10, "output": 3, "reasoning": 2, "cache": {"read": 4, "write": 1}},
            },
            {"role": "assistant", "tokens": {"input": 7, "output": 5, "reasoning": 1, "cache": {"read": 2}}},
            {"role": "user"},
        ]
    }
    usage = OpenCodeSandboxedAgent._native_usage(export)
    assert usage.input_tokens == 24
    assert usage.output_tokens == 11
    assert usage.total_tokens == 35
    assert usage.input_tokens_details.cached_tokens == 6
    assert usage.output_tokens_details.reasoning_tokens == 3
    assert OpenCodeSandboxedAgent._native_usage({"messages": []}) is None


@pytest.mark.parametrize("option", ["missing-sandbox", "worker", "required-tool", "workdir", "unpinned", "provider"])
def test_invalid_seed_never_connects(setup, option):
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    if option == "missing-sandbox":
        body["sandbox_access"] = None
    elif option == "worker":
        agent.config.num_workers = 2
    elif option == "required-tool":
        body["tool_accesses"] = [
            {"kind": "direct_http", "name": "tools", "base_url": "http://resources", "required": True}
        ]
    elif option == "workdir":
        body["sandbox_access"]["workdir"] = "/"
    elif option == "unpinned":
        agent.config.opencode_version = "latest"
    else:
        agent.config.opencode_config = {"provider": {"other": {"apiKey": "must-not-copy"}}}
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=body).status_code == 422
    sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("kind", ["error", "timeout", "length"])
def test_partial_output_survives_model_failure_and_timeout(setup, kind):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    if kind == "error":
        export["messages"][-1]["info"]["error"] = {"message": "model rejected request"}
    elif kind == "timeout":
        sandbox.result["timed_out"] = True
        sandbox.result["return_code"] = -9
    else:
        export["messages"][-1]["info"]["finish"] = "length"
    sandbox.events = json.dumps(export)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        session_id = created.json()["agent_session_id"]
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.json()["status"] == ("failed" if kind == "error" else "incomplete")
        assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        retry = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert retry.json() == closed.json()
        assert client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"}).status_code == 409
    sandbox.disconnect.assert_awaited_once()


async def test_file_cleanup_failure_keeps_connection_and_can_retry(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=1, error_type=None),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    close = AgentCloseSessionRequest(**close_body(session_id))
    with pytest.raises(RuntimeError, match="session files"):
        await agent.close_agent_session(request, close)
    sandbox.disconnect.assert_not_awaited()
    assert agent._native_sessions[session_id].closing
    await agent.close_agent_session(request, close)
    sandbox.disconnect.assert_awaited_once()


def test_malformed_later_artifact_keeps_partial_output(setup):
    from nemo_gym.rollout_observability import AgentObservationBundle

    agent, _ = setup
    export = json.loads(events())
    export["messages"][1]["parts"].extend(
        [
            {"type": "future-event"},
            {
                "type": "tool",
                "callID": "failed-tool",
                "tool": "bash",
                "state": {"input": {}, "status": "error", "error": "failed tool"},
            },
        ]
    )
    observations = AgentObservationBundle(source="opencode")
    output = agent._native_output(export, observations)
    assert output[3].content[0].text == "Fixed"
    assert output[-1].output == "failed tool"
    assert output[-1].status == "incomplete"
    assert observations.gaps[0].code == "agent_artifact_record_unparseable"
