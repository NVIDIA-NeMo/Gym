# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig
from responses_api_agents.opencode_agent.sandbox import OpenCodeSandboxResult


def seed() -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id=f"opencode-test-{uuid4().hex}",
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
    config = OpenCodeAgentConfig(
        name="opencode",
        execution_mode="sandbox",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        model_server={"type": "responses_api_models", "name": "policy"},
        opencode_version="1.17.11",
        resources_server={"type": "resources_servers", "name": "resources"},
        sandbox_provider="unused",
        sandbox_config={},
        timeout=30,
        context_window=32000,
        session_close_timeout_seconds=1,
    )
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(f"{module}.ensure_opencode", side_effect=AssertionError("Native sessions must not install on the host")),
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider"),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
    ):
        agent = OpenCodeAgent(config=config, server_client=client)
        yield agent, sandbox


def close_body(session_id):
    return {"agent_session_id": session_id, "episode_id": seed().episode_id.model_dump()}


def capture_observations(sandbox, tmp_path):
    """Persist the same assistant turns used by the fake runner as real SQLite artifacts."""
    database = tmp_path / "observations.db"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            create table session(id text, parent_id text, time_created integer);
            create table message(id text, session_id text, data text, time_created integer);
            create table part(id text, message_id text, session_id text, data text, time_created integer);
            insert into session values('root', null, 0);
        """)
        for index, message in enumerate(json.loads(sandbox.events)["messages"]):
            message_id = f"message-{index}"
            connection.execute(
                "insert into message values (?, ?, ?, ?)",
                (message_id, "root", json.dumps(message["info"]), index),
            )
            for part_index, part in enumerate(message["parts"]):
                connection.execute(
                    "insert into part values (?, ?, ?, ?, ?)",
                    (f"part-{index}-{part_index}", message_id, "root", json.dumps(part), part_index),
                )
    original = sandbox.download

    async def download(source, destination):
        if source.endswith("/observations.db"):
            Path(destination).write_bytes(database.read_bytes())
        else:
            await original(source, destination)

    sandbox.download = download
    return database


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


@pytest.mark.parametrize("part_type", ["text", "reasoning"])
@pytest.mark.parametrize("text", ["literal <think> opening", "literal </think> closing", " <think>x</think> \n"])
def test_http_preserves_typed_literal_think_tags_and_unique_ids(setup, tmp_path, part_type, text):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    assistant = export["messages"][1]
    tool = assistant["parts"][1]
    assistant["parts"] = [{"type": part_type, "text": text}, tool, {"type": part_type, "text": text}]
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["status"] == "completed"
        output = body["output"]
        expected_type = "message" if part_type == "text" else "reasoning"
        text_key = "content" if part_type == "text" else "summary"
        assert [item["type"] for item in output] == [
            expected_type,
            "function_call",
            "function_call_output",
            expected_type,
        ]
        assert output[0][text_key][0]["text"] == output[3][text_key][0]["text"] == text
        assert output[0]["id"] != output[3]["id"]
        assert output[1]["call_id"] == output[2]["call_id"] == "tool-1"
        assert output[1]["name"] == "bash"
        assert json.loads(output[1]["arguments"]) == {"command": "pwd"}
        assert output[2]["output"] == "/app"
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    invocation = next(
        record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
    )
    persisted_parts = [item for item in invocation["conversation"] if item.get("role") != "user"]
    assert [item["type"] for item in persisted_parts] == [item["type"] for item in output]
    assert persisted_parts[0][text_key][0]["text"] == persisted_parts[3][text_key][0]["text"] == text
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


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


async def test_close_cancels_active_opencode_before_detaching(setup, tmp_path):
    agent, sandbox = setup
    sandbox.blocked = True
    capture_observations(sandbox, tmp_path)
    request, session_id, task = await activate(agent, sandbox)
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    sandbox.runner.send_signal.assert_awaited_once_with("SIGTERM")
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    invocation = next(record for record in closed.agent_observations.records if record.kind == "agent_invocation")
    assert invocation.status == "incomplete"
    assert invocation.error_type == "cancelled"
    assert invocation.conversation[-1].content[0].text == "Fixed"


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
    monkeypatch.setattr("responses_api_agents.opencode_agent.app.monotonic", lambda: 100.0)
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
    monkeypatch.setattr("responses_api_agents.opencode_agent.app.monotonic", lambda: clock[0])
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
    monkeypatch.setattr("responses_api_agents.opencode_agent.app.monotonic", lambda: clock[0])
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
    monkeypatch.setattr("responses_api_agents.opencode_agent.app.monotonic", lambda: clock[0])
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
        OpenCodeAgentConfig(**(agent.config.model_dump() | {"session_close_retry_window_seconds": window}))


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
    usage = OpenCodeAgent._native_usage(export)
    assert usage.input_tokens == 24
    assert usage.output_tokens == 11
    assert usage.total_tokens == 35
    assert usage.input_tokens_details.cached_tokens == 6
    assert usage.output_tokens_details.reasoning_tokens == 3
    assert OpenCodeAgent._native_usage({"messages": []}) is None


@pytest.mark.parametrize("field", ["cache", "reasoning"])
@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "3", "bad"])
def test_native_usage_does_not_treat_defaulted_or_invalid_details_as_measurements(field, value):
    tokens = {"input": 10, "output": 3, "reasoning": 2, "cache": {"read": 4, "write": 1}}
    if field == "cache":
        tokens["cache"]["read"] = value
    else:
        tokens["reasoning"] = value
    usage = OpenCodeAgent._native_usage({"usage_messages": [{"role": "assistant", "tokens": tokens}]})
    # Invalid optional fields neither erase measured base counts nor get coerced into totals.
    assert usage.input_tokens == (11 if field == "cache" else 15)
    assert usage.output_tokens == (3 if field == "reasoning" else 5)
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert usage.input_tokens_details.cached_tokens == (None if field == "cache" else 4)
    assert usage.output_tokens_details.reasoning_tokens == (None if field == "reasoning" else 2)


@pytest.mark.parametrize("unknown_turn", [0, 1])
@pytest.mark.parametrize("unknown_kind", ["zero", "absent", "missing_usage"])
def test_native_optional_usage_stays_unknown_across_root_and_subagent_calls(unknown_turn, unknown_kind):
    infos = [
        {"role": "assistant", "tokens": {"input": 10, "output": 3, "reasoning": 2, "cache": {"read": 4, "write": 1}}},
        {"role": "assistant", "tokens": {"input": 7, "output": 5, "reasoning": 1, "cache": {"read": 2}}},
    ]
    if unknown_kind == "missing_usage":
        infos[unknown_turn].pop("tokens")
    elif unknown_kind == "absent":
        infos[unknown_turn]["tokens"].pop("reasoning")
        infos[unknown_turn]["tokens"]["cache"].pop("read")
    else:
        infos[unknown_turn]["tokens"]["reasoning"] = 0
        infos[unknown_turn]["tokens"]["cache"]["read"] = 0
    usage = OpenCodeAgent._native_usage({"usage_messages": infos})
    assert usage.input_tokens_details.cached_tokens is None
    assert usage.output_tokens_details.reasoning_tokens is None
    if unknown_kind == "missing_usage":
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
            (9, 6, 15) if unknown_turn == 0 else (15, 5, 20)
        )
    else:
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
            (20, 9, 29) if unknown_turn == 0 else (22, 10, 32)
        )


@pytest.mark.parametrize("cache,reasoning", [(0, 0), (4, 0), (0, 2), (4, 2)])
def test_native_http_usage_gaps_follow_persisted_optional_counter_availability(setup, tmp_path, cache, reasoning):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    tokens = export["messages"][1]["info"]["tokens"]
    tokens["cache"] = {"read": cache, "write": 1}
    tokens["reasoning"] = reasoning
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        activated = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        activated.raise_for_status()
        response = activated.json()
        assert response["status"] == "completed"
        assert response["usage"] == {
            "input_tokens": 11 + cache,
            "output_tokens": 2 + reasoning,
            "total_tokens": 13 + cache + reasoning,
            "input_tokens_details": {"cached_tokens": cache or None},
            "output_tokens_details": {"reasoning_tokens": reasoning or None},
        }
        assert [item["type"] for item in response["output"]] == [
            "reasoning",
            "function_call",
            "function_call_output",
            "message",
        ]
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        closed.raise_for_status()
    gaps = {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
    assert ("token_usage_detail_unavailable" in gaps) == (cache == 0 or reasoning == 0)


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
def test_partial_output_survives_model_failure_and_timeout(setup, kind, tmp_path):
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
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        session_id = created.json()["agent_session_id"]
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.json()["status"] == ("failed" if kind == "error" else "incomplete")
        assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        invocation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
        )
        assert invocation["status"] == result.json()["status"]
        assert invocation["conversation"][-1]["content"][0]["text"] == "Fixed"
        retry = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert retry.json() == closed.json()
        assert client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"}).status_code == 409
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize(
    ("finish", "expected"),
    [
        ("stop", "completed"),
        ("length", "incomplete"),
        ("content-filter", "incomplete"),
        ("tool-calls", "incomplete"),
        ("error", "failed"),
        ("unknown", "failed"),
        (None, "failed"),
    ],
)
def test_completed_model_turn_is_not_always_terminal(setup, tmp_path, finish, expected):
    agent, sandbox = setup
    export = json.loads(sandbox.events)
    export["messages"][-1]["info"]["finish"] = finish
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200
        assert result.json()["status"] == expected
        assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        if expected == "failed":
            assert "terminal assistant result" in result.json()["error"]["message"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        invocation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
        )
        assert invocation["status"] == expected


@pytest.mark.parametrize(("finish", "expected"), [("stop", "completed"), ("tool-calls", "incomplete")])
def test_timeout_preserves_completed_children_and_marks_unfinished_children(setup, tmp_path, finish, expected):
    agent, sandbox = setup
    sandbox.result["timed_out"] = True
    sandbox.result["return_code"] = -9
    database = capture_observations(sandbox, tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("insert into session values ('child', 'root', 1)")
        for index, reason in enumerate(("tool-calls", finish)):
            connection.execute(
                "insert into message values (?, ?, ?, ?)",
                (
                    f"child-message-{index}",
                    "child",
                    json.dumps({"role": "assistant", "finish": reason, "time": {"completed": 100 + index}}),
                    index,
                ),
            )
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.json()["status"] == "incomplete"
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        invocations = {
            record["invocation_id"]: record["status"]
            for record in closed.json()["agent_observations"]["records"]
            if record["kind"] == "agent_invocation"
        }
        assert invocations == {"root": "incomplete", "child": expected}


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


@pytest.mark.parametrize(
    "workdir",
    [
        "/",
        "/tmp",
        "/tmp/",
        "/tmp/../tmp/task",
        "/tmp/nemo-gym-opencode-sessions",
        "/tmp/nemo-gym-opencode-runtime-1.17.11/repo",
    ],
)
async def test_adapter_workdir_overlap_is_rejected_before_connection(setup, workdir):
    agent, sandbox = setup
    body = seed()
    body.sandbox_access.workdir = workdir
    with pytest.raises(HTTPException) as error:
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    assert error.value.status_code == 422
    sandbox.exec.assert_not_awaited()


async def test_resolved_workdir_check_runs_before_session_files_are_created(setup, tmp_path):
    import shlex
    import subprocess
    import sys

    agent, sandbox = setup
    await agent.seed_agent_session(Request({"type": "http", "session": {}}), seed())
    command = shlex.split(sandbox.exec.await_args_list[0].args[0])
    script = command[3]
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    workdir = tmp_path / "task"
    workdir.symlink_to(sessions, target_is_directory=True)
    rejected = subprocess.run(
        [sys.executable, "-I", "-c", script, str(workdir), str(sessions), str(tmp_path / "runtime")],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "overlaps the task workdir" in rejected.stderr
    ordinary = tmp_path / "repo"
    ordinary.mkdir()
    accepted = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ordinary), str(sessions), str(tmp_path / "runtime")],
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert list(sessions.iterdir()) == []


@pytest.mark.parametrize("marker", [None, "", [], {}, 0, "closed-session"])
async def test_native_markers_block_legacy_run_and_responses(setup, marker):
    from responses_api_agents.opencode_agent.app import OpenCodeAgentRunRequest

    agent, sandbox = setup
    request = Request({"type": "http", "session": {"nemo_gym_opencode_native_session": marker}})
    with pytest.raises(HTTPException) as error:
        await agent.run(request, OpenCodeAgentRunRequest(responses_create_params={"input": "task"}))
    assert error.value.status_code == 409
    with pytest.raises(HTTPException) as error:
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert error.value.status_code == 409
    agent.server_client.post.assert_not_called()
    sandbox.pty.create.assert_not_awaited()


def test_close_response_cookie_blocks_legacy_run(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).raise_for_status()
        response = client.post("/run", json={"responses_create_params": {"input": "task"}})
        assert response.status_code == 409
    agent.server_client.post.assert_not_called()


async def test_caller_assigned_seed_is_serialized_and_binds_all_inputs(setup):
    agent, sandbox = setup
    body = seed()
    first_request = Request({"type": "http", "session": {}})
    retry_request = Request({"type": "http", "session": {}})
    first, retry = await asyncio.gather(
        agent.seed_agent_session(first_request, body), agent.seed_agent_session(retry_request, body)
    )
    assert first.agent_session_id == retry.agent_session_id == body.agent_session_id
    assert sandbox.exec.await_count == 2
    changed = body.model_copy(deep=True)
    changed.sandbox_access.workdir = "/different"
    with pytest.raises(HTTPException, match="different seed inputs"):
        await agent.seed_agent_session(retry_request, changed)
    await agent.close_agent_session(
        Request({"type": "http", "session": {}}), AgentCloseSessionRequest(**close_body(body.agent_session_id))
    )
    sandbox.disconnect.assert_awaited_once()


async def test_close_without_seed_cookie_blocks_delayed_seed(setup):
    agent, sandbox = setup
    body = seed()
    request = Request({"type": "http", "session": {}})
    close = AgentCloseSessionRequest(**close_body(body.agent_session_id))
    assert (await agent.close_agent_session(request, close)).agent_session_id == body.agent_session_id
    with pytest.raises(HTTPException, match="already closed"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    sandbox.exec.assert_not_awaited()
    sandbox.disconnect.assert_not_awaited()


async def test_close_waits_for_inflight_seed_even_without_cookie(setup):
    agent, sandbox = setup
    body = seed()
    entered, release = asyncio.Event(), asyncio.Event()
    initialize = agent._initialize_agent_session_state

    async def delayed_initialize(*args):
        entered.set()
        await release.wait()
        return await initialize(*args)

    with patch.object(agent, "_initialize_agent_session_state", delayed_initialize):
        seeded = asyncio.create_task(agent.seed_agent_session(Request({"type": "http", "session": {}}), body))
        await entered.wait()
        closed = asyncio.create_task(
            agent.close_agent_session(
                Request({"type": "http", "session": {}}), AgentCloseSessionRequest(**close_body(body.agent_session_id))
            )
        )
        await asyncio.sleep(0)
        assert not closed.done()
        release.set()
        await seeded
        await closed
    assert not agent._native_sessions
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("fail_cleanup", [False, True])
async def test_abandoned_session_expires_and_failed_cleanup_stays_closed_to_activation(setup, fail_cleanup):
    agent, sandbox = setup
    agent.config.session_lifetime_seconds = 0.01
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "opencode-smoke-a2"}})
    body = seed()
    await agent.seed_agent_session(request, body)
    if fail_cleanup:
        sandbox.disconnect.side_effect = RuntimeError("provider unavailable")
    expiry = agent._native_session_expiry_tasks[body.agent_session_id]
    await asyncio.wait_for(asyncio.shield(expiry), 2)
    sandbox.disconnect.assert_awaited_once()
    assert bool(agent._native_sessions) is fail_cleanup
    with pytest.raises(HTTPException):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    if fail_cleanup:
        sandbox.disconnect.side_effect = None
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


async def test_caller_session_id_is_never_used_as_a_filesystem_path(setup):
    agent, sandbox = setup
    body = seed().model_copy(update={"agent_session_id": "../../outside; echo untrusted"})
    request = Request({"type": "http", "session": {}})
    await agent.seed_agent_session(request, body)
    state = agent._native_sessions[body.agent_session_id]
    assert Path(state.directory).parent == Path("/tmp/nemo-gym-opencode-sessions")
    assert body.agent_session_id not in state.directory
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))
