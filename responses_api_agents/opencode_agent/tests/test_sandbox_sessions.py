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

from nemo_gym.agent_utils.supervisor_client import parse_cleanup_receipt
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig


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
        }
        self.runtime_info = {"hostname": "task-container", "pid": 123}
        self.events = events()
        self.blocked = False
        self.started = asyncio.Event()
        self.exited = asyncio.Event()
        self.exec = AsyncMock(
            side_effect=self.run_exec,
            return_value=SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None),
        )
        self.stop = AsyncMock()
        self.disconnect = AsyncMock()
        self.launch = AsyncMock(side_effect=self.create)
        self.request_stop = AsyncMock(side_effect=self.signal)

    @property
    def pty(self):
        raise AssertionError("This provider supports exec only; no PTY API")

    async def run_exec(self, command, **kwargs):
        if "--receipt" in command and "process_supervisor.py" in command:
            return await self.launch(command=command, **kwargs)
        if "stop.request" in command:
            await self.request_stop()
        return self.exec.return_value

    async def upload(self, source, destination):
        self.files[destination] = Path(source).read_text()

    async def download(self, source, destination):
        Path(destination).write_text(self.files[source])

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
        await self.wait_exit()
        return SimpleNamespace(error_type=None, return_code=0, stdout="", stderr="")

    async def signal(self):
        if hasattr(self, "directory"):
            self.result["timed_out"] = True
            self.exited.set()
            await asyncio.sleep(0)

    async def wait_exit(self):
        await self.exited.wait()
        self.files[f"{self.directory}/cleanup.json"] = json.dumps(self.result)
        self.files[f"{self.directory}/runtime.json"] = json.dumps(self.runtime_info)
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
        max_output_tokens=4096,
        session_close_timeout_seconds=1,
    )
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(
            f"{module}.ensure_opencode", side_effect=AssertionError("Sandbox sessions must not install on the host")
        ),
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider"),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
    ):
        agent = OpenCodeAgent(config=config, server_client=client)
        yield agent, sandbox


def active_sessions(agent):
    return {key: record.state for key, record in agent._session_records.items() if record.state is not None}


@pytest.mark.parametrize("stage", ["prepare", "install"])
@pytest.mark.parametrize("error_type", ["timeout", "sandbox"])
@pytest.mark.parametrize("owned", [False, True])
async def test_provider_error_type_blocks_setup_even_with_zero_exit(setup, stage, error_type, owned):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}})
    body = seed()
    if owned:
        agent.config.sandbox_provider = "sandbox"
        agent.config.sandbox_config = {"image": "test-image", "workdir": "/app"}
        body = body.model_copy(update={"sandbox_access": None})
        sandbox.start = AsyncMock()
    ok = SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
    error = SimpleNamespace(return_code=0, error_type=error_type, stdout="bootstrap output", stderr="provider failed")
    sandbox.exec.side_effect = ([ok] if owned else []) + ([error, ok] if stage == "prepare" else [ok, error, ok])
    with patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox) as sandbox_class:
        sandbox_class.connect = AsyncMock(return_value=sandbox)
        with pytest.raises(RuntimeError, match=f"error={error_type}") as failed:
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


@pytest.mark.parametrize("output_budget", [4096, 32000])
def test_http_session_flow_runs_opencode_in_borrowed_sandbox(setup, output_budget):
    agent, sandbox = setup
    agent.config.max_output_tokens = output_budget
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        assert not sandbox.launch.called
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
        assert config["provider"]["nemo_gym"]["models"]["dummy_model"]["limit"] == {
            "context": 32000,
            "input": 32000,
            "output": output_budget,
        }
        assert payload["env"]["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == str(output_budget)
        assert (
            config["provider"]["nemo_gym"]["options"]["baseURL"]
            == "http://model.example:9000/ng-rollout/opencode-smoke-a2/v1"
        )
        assert client.post("/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"}).status_code == 409
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    assert not active_sessions(agent)
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
    sandbox.launch.assert_not_awaited()


def test_rejected_request_does_not_consume_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/opencode-smoke-a2/v1/responses"
        assert client.post(path, json={"input": "task", "temperature": 0.2}).status_code == 422
        sandbox.launch.assert_not_awaited()
        accepted = client.post(path, json={"input": "task"})
        assert accepted.status_code == 200, accepted.text
        assert client.post(path, json={"input": "task"}).json() == accepted.json()
        assert client.post(path, json={"input": "changed"}).status_code == 409
    sandbox.launch.assert_awaited_once()


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
    sandbox.request_stop.assert_awaited_once()
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
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await task
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in active_sessions(agent)
    sandbox.disconnect.assert_not_awaited()
    assert not any("rm -rf" in call.args[0] for call in sandbox.exec.await_args_list)


async def test_disconnect_failure_retains_session_for_retry(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    sandbox.disconnect.side_effect = [RuntimeError("provider unavailable"), None]
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in active_sessions(agent)
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id not in active_sessions(agent)
    sandbox.launch.assert_awaited_once()


def test_cleanup_receipt_is_required():
    with pytest.raises(ValueError):
        parse_cleanup_receipt({"return_code": 0, "error": None})


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
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: 100.0)
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
    assert first_result is not second_result
    sandbox.disconnect.assert_awaited_once()


async def test_seed_prunes_expired_close_receipts(setup, monkeypatch):
    agent, _ = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: clock[0])
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    clock[0] += agent.config.session_close_retry_window_seconds
    await agent.seed_agent_session(Request({"type": "http", "session": {}}), seed())
    assert not agent._closed_session_records


@pytest.mark.parametrize("window", [0, -1, float("inf")])
def test_close_retry_window_must_be_positive_and_finite(setup, window):
    agent, _ = setup
    with pytest.raises(ValidationError):
        OpenCodeAgentConfig(**(agent.config.model_dump() | {"session_close_retry_window_seconds": window}))


async def test_unknown_launch_outcome_fails_closed(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "opencode-smoke-a2"}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    sandbox.launch.side_effect = TimeoutError("lost launch response")
    with pytest.raises(TimeoutError, match="lost launch response"):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert session_id in active_sessions(agent)
    sandbox.disconnect.assert_not_awaited()
    sandbox.stop.assert_not_awaited()


async def test_install_failure_disconnects_without_stopping_owner(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        SimpleNamespace(return_code=1, stderr="curl failed", stdout="", error_type=None),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="curl failed"):
        await agent.seed_agent_session(request, seed())
    assert not active_sessions(agent)
    assert not request.session
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_cancelled_install_never_publishes_session_or_launches_opencode(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        asyncio.CancelledError(),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(asyncio.CancelledError):
        await agent.seed_agent_session(request, seed())
    assert not active_sessions(agent)
    assert not request.session
    sandbox.launch.assert_not_awaited()
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_reported_usage_restores_cached_and_reasoning_tokens_across_subagents():
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
    usage = OpenCodeAgent._session_usage(export)
    assert usage.input_tokens == 24
    assert usage.output_tokens == 11
    assert usage.total_tokens == 35
    assert usage.input_tokens_details.cached_tokens == 6
    assert usage.output_tokens_details.reasoning_tokens == 3
    assert OpenCodeAgent._session_usage({"messages": []}) is None


@pytest.mark.parametrize("field", ["cache", "reasoning"])
@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "3", "bad"])
def test_reported_usage_does_not_treat_defaulted_or_invalid_details_as_measurements(field, value):
    tokens = {"input": 10, "output": 3, "reasoning": 2, "cache": {"read": 4, "write": 1}}
    if field == "cache":
        tokens["cache"]["read"] = value
    else:
        tokens["reasoning"] = value
    usage = OpenCodeAgent._session_usage({"usage_messages": [{"role": "assistant", "tokens": tokens}]})
    # Invalid optional fields neither erase measured base counts nor get coerced into totals.
    assert usage.input_tokens == (11 if field == "cache" else 15)
    assert usage.output_tokens == (3 if field == "reasoning" else 5)
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert usage.input_tokens_details.cached_tokens == (None if field == "cache" else 4)
    assert usage.output_tokens_details.reasoning_tokens == (None if field == "reasoning" else 2)


@pytest.mark.parametrize("unknown_turn", [0, 1])
@pytest.mark.parametrize("unknown_kind", ["zero", "absent", "missing_usage"])
def test_optional_reported_usage_stays_unknown_across_root_and_subagent_calls(unknown_turn, unknown_kind):
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
    usage = OpenCodeAgent._session_usage({"usage_messages": infos})
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
def test_session_http_usage_gaps_follow_persisted_optional_counter_availability(setup, tmp_path, cache, reasoning):
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
        agent.config.sandbox_provider = ""
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
        if option == "worker":
            with pytest.raises(ValueError, match="num_workers=1"):
                client.post("/v1/agent_sessions", json=body)
        else:
            assert client.post("/v1/agent_sessions", json=body).status_code == 422
    sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("output_budget", [0, -1, 32001])
def test_invalid_output_budget_rejected_before_connecting(setup, output_budget):
    agent, sandbox = setup
    agent.config.max_output_tokens = output_budget
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert response.status_code == 422
    assert "max_output_tokens" in response.json()["detail"]
    sandbox.exec.assert_not_awaited()
    sandbox.launch.assert_not_awaited()


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
        if kind == "error":
            assert result.status_code == 502
            assert "model rejected request" in result.json()["detail"]
        else:
            assert result.json()["status"] == "incomplete"
            assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        invocation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
        )
        assert invocation["status"] == ("failed" if kind == "error" else "incomplete")
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
        if expected == "failed":
            assert result.status_code == 502
            assert "terminal assistant result" in result.json()["detail"]
        else:
            assert result.status_code == 200
            assert result.json()["status"] == expected
            assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200
        invocation = next(
            record for record in closed.json()["agent_observations"]["records"] if record["kind"] == "agent_invocation"
        )
        assert invocation["status"] == expected
        assert invocation["conversation"][-1]["content"][0]["text"] == "Fixed"


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
        SimpleNamespace(return_code=1, error_type=None, stderr="file cleanup failed"),
        SimpleNamespace(return_code=0, error_type=None, stderr=""),
    ]
    close = AgentCloseSessionRequest(**close_body(session_id))
    with pytest.raises(RuntimeError, match="session files"):
        await agent.close_agent_session(request, close)
    sandbox.disconnect.assert_not_awaited()
    assert agent._session_records[session_id].closing
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
    output = agent._session_output(export, observations)
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
    script = command[command.index("-I") + 2]
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
    with pytest.raises(HTTPException, match="another seed request"):
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
    initialize = agent._seed_agent_session_state

    async def delayed_initialize(*args):
        entered.set()
        await release.wait()
        return await initialize(*args)

    with patch.object(agent, "_seed_agent_session_state", delayed_initialize):
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
    assert not active_sessions(agent)
    sandbox.disconnect.assert_awaited_once()


async def test_active_session_has_no_adapter_expiry_timer(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}})
    body = seed()
    await agent.seed_agent_session(request, body)
    await asyncio.sleep(0)
    assert "session_lifetime_seconds" not in type(agent.config).model_fields
    assert body.agent_session_id in active_sessions(agent)
    sandbox.disconnect.assert_not_awaited()
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


async def test_caller_session_id_is_never_used_as_a_filesystem_path(setup):
    agent, sandbox = setup
    body = seed().model_copy(update={"agent_session_id": "../../outside; echo untrusted"})
    request = Request({"type": "http", "session": {}})
    await agent.seed_agent_session(request, body)
    state = active_sessions(agent)[body.agent_session_id]
    assert Path(state.session.session_dir).parent == Path("/tmp/nemo-gym-opencode-sessions")
    assert body.agent_session_id not in state.session.session_dir
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


async def test_expired_receipt_cookie_cannot_seed_or_activate(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    body = seed()
    request = Request({"type": "http", "session": {}})
    await agent.seed_agent_session(request, body)
    close = AgentCloseSessionRequest(**close_body(body.agent_session_id))
    await agent.close_agent_session(request, close)
    clock[0] = 111.0
    with pytest.raises(HTTPException, match="expired"):
        await agent.close_agent_session(request, close)
    with pytest.raises(HTTPException, match="expired"):
        await agent.seed_agent_session(request, body)
    with pytest.raises(HTTPException):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("failure", ["remove", "disconnect"])
async def test_failed_setup_cleanup_retains_state_for_cookieless_close(setup, failure):
    agent, sandbox = setup
    body = seed()
    success = SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)
    failed = SimpleNamespace(return_code=1, stdout="", stderr="installer failed", error_type=None)
    sandbox.exec.side_effect = [success, failed, failed if failure == "remove" else success]
    if failure == "disconnect":
        sandbox.disconnect.side_effect = RuntimeError("disconnect failed")
    with pytest.raises(RuntimeError, match="installer failed"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    assert agent._session_records[body.agent_session_id].closing
    with pytest.raises(HTTPException, match="closing"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    sandbox.exec.side_effect = None
    sandbox.exec.return_value = success
    sandbox.disconnect.side_effect = None
    closed = await agent.close_agent_session(
        Request({"type": "http", "session": {}}), AgentCloseSessionRequest(**close_body(body.agent_session_id))
    )
    assert closed.agent_observations is not None
    assert body.agent_session_id not in active_sessions(agent)
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
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(f"{module}.AsyncSandbox", return_value=sandbox) as factory,
        patch(f"{module}.resolve_provider_config") as resolve,
        TestClient(agent.setup_webserver()) as client,
    ):
        factory.connect = AsyncMock(return_value=sandbox)
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        state = active_sessions(agent)[body.agent_session_id]
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
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox),
        TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client,
    ):
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        state = active_sessions(agent)[body.agent_session_id]
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
                stage == "install" and command.startswith("bash ") and "install_opencode_runtime.sh" in command
            ):
                raise original
            return await execute(command, **kwargs)

        sandbox.exec.side_effect = fail_at_stage
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with (
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox),
        TestClient(agent.setup_webserver()) as client,
    ):
        with pytest.raises(RuntimeError) as error:
            client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert error.value is original
        assert client.post("/v1/agent_sessions", json=body.model_dump(mode="json")).status_code == 409
        state = active_sessions(agent)[body.agent_session_id]
        assert state.session.closing and not state.session.closed
        response = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": body.agent_session_id, "episode_id": body.episode_id.model_dump()},
        )
        assert response.status_code == 200, response.text
        assert sandbox.stop.await_count == 2
        sandbox.disconnect.assert_not_awaited()


def test_failed_connection_closes_provider_without_publishing_session(setup):
    agent, sandbox = setup
    agent.config.sandbox_provider = "fallback-must-not-be-used"
    module = "responses_api_agents.opencode_agent.app"
    provider = SimpleNamespace(aclose=AsyncMock())
    with (
        patch(f"{module}.AsyncSandbox") as factory,
        patch(f"{module}.create_provider", return_value=provider),
        TestClient(agent.setup_webserver()) as client,
    ):
        factory.connect = AsyncMock(side_effect=RuntimeError("borrow failed"))
        with pytest.raises(RuntimeError, match="borrow failed"):
            client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        factory.assert_not_called()
        sandbox.stop.assert_not_awaited()
        provider.aclose.assert_awaited_once()
        assert not active_sessions(agent)


@pytest.mark.parametrize("workdir", [None, "relative"])
def test_owned_workdir_is_validated_before_creation(setup, workdir):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"workdir": workdir}
    body = seed()
    body.sandbox_access = None
    with (
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox") as factory,
        TestClient(agent.setup_webserver()) as client,
    ):
        assert client.post("/v1/agent_sessions", json=body.model_dump(mode="json")).status_code == 422
        factory.assert_not_called()
        sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("invalid_runtime", [False, True])
async def test_close_keeps_captured_events_after_sandbox_release(setup, tmp_path, owned, invalid_runtime):
    agent, sandbox = setup
    body = seed()
    if owned:
        body.sandbox_access = None
        agent.config.sandbox_provider = "agent-provider"
        agent.config.sandbox_config = {"image": "test-image"}
    if invalid_runtime:
        sandbox.runtime_info = {"hostname": "worker", "pid": "not-an-integer"}
        sandbox.result["return_code"] = "not-an-integer"
    sandbox.start = AsyncMock()
    capture_observations(sandbox, tmp_path)
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "opencode-smoke-a2"}})
    with patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox) as factory:
        factory.connect = AsyncMock(return_value=sandbox)
        await agent.seed_agent_session(request, body)
        state = agent._session_records[body.agent_session_id].state
        directory = state.session.session_dir
        assert f"{directory}/process_supervisor.py" not in sandbox.files
        sandbox.blocked = True
        task = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
        await asyncio.wait_for(sandbox.started.wait(), 2)
        assert f"{directory}/process_supervisor.py" in sandbox.files

        async def release():
            assert state.session.cleanup["cleanup_confirmed"] is True
            assert state.session.artifacts is not None
            sandbox.files.clear()
            sandbox.download = AsyncMock(side_effect=AssertionError("output read after release"))

        sandbox.stop.side_effect = release
        sandbox.disconnect.side_effect = release
        close = AgentCloseSessionRequest(**close_body(body.agent_session_id))
        response = await agent.close_agent_session(request, close)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert response.agent_observations is not None
        invocations = [record for record in response.agent_observations.records if record.kind == "agent_invocation"]
        assert any(record.conversation for record in invocations)
        assert state.session.closed
        assert await agent.close_agent_session(request, close) == response
        (sandbox.stop if owned else sandbox.disconnect).assert_awaited_once()
        (sandbox.disconnect if owned else sandbox.stop).assert_not_awaited()


@pytest.mark.parametrize("diagnostics", [{"return_code": "0"}, {"extra": 1}])
def test_optional_diagnostics_preserve_valid_terminal_result(setup, diagnostics):
    agent, sandbox = setup
    sandbox.result.update(diagnostics)
    sandbox.runtime_info = {"pid": "invalid"}
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        response.raise_for_status()
        assert response.json()["status"] == "completed"
        assert response.json()["output"][-1]["content"][0]["text"] == "Fixed"
        assert "harness_hostname" not in response.json()["metadata"]
        assert "harness_pid" not in response.json()["metadata"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        closed.raise_for_status()
        gaps = {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
        assert "runtime_info_unavailable" in gaps
        assert ("worker_exit_code_unavailable" in gaps) == ("return_code" in diagnostics)


async def test_disconnected_waiter_and_identical_retry_share_one_activation(setup):
    agent, sandbox = setup
    sandbox.blocked = True
    request, session_id, waiter = await activate(agent, sandbox)
    retry = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    state = agent._session_records[session_id].state
    assert not state.task.done()
    sandbox.request_stop.assert_not_awaited()
    with pytest.raises(HTTPException, match="retry the same request"):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="different task"))
    sandbox.exited.set()
    response = await asyncio.wait_for(retry, 2)
    original = response.model_copy(deep=True)
    response.output.clear()
    assert await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")) == original
    sandbox.launch.assert_awaited_once()
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
