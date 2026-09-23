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
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig, PiAgentRunRequest
from responses_api_agents.pi_agent.sandbox import PiSandboxResult


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
        self.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr=""))
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
        patch(f"{module}.ensure_pi", side_effect=AssertionError("native sessions must not install host Pi")),
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider"),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
    ):
        agent = PiAgent(config=config, server_client=client)
        yield agent, sandbox


def close_body(session_id):
    return {"agent_session_id": session_id, "episode_id": seed().episode_id.model_dump()}


def test_http_native_flow_runs_pi_in_borrowed_sandbox(setup):
    agent, sandbox = setup
    with patch.object(agent, "_run_pi", AsyncMock(side_effect=AssertionError("host Pi must not run"))):
        with TestClient(agent.setup_webserver()) as client:
            created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
            assert created.status_code == 200, created.text
            session_id = created.json()["agent_session_id"]
            directory = agent._sandbox_sessions[session_id].directory
            installer = f"{directory}/install_pi_runtime.sh"
            assert installer in sandbox.files
            assert agent.config.resources_server is None
            assert not sandbox.pty.create.called
            assert not any(path.startswith("/app/") for path in sandbox.files)
            install_call = sandbox.exec.await_args_list[1]
            assert install_call.args[0] == (f"bash {installer} /tmp/nemo-gym-pi-node-22.19.0-0.80.2 0.80.2")
            assert install_call.kwargs["timeout_s"] == agent.config.sandbox_install_timeout_seconds
            result = client.post(
                "/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code", "max_output_tokens": 123}
            )
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
            assert body["usage"]["input_tokens_details"]["cached_tokens"] is None
            assert body["metadata"]["harness_execution"] == "sandbox"
            assert "_ng_agent_observations" not in body
            payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
            assert payload["prompt"] == "Fix the code"
            assert payload["command"][0].endswith("/node/bin/node")
            assert "PATH" not in payload["env"]
            extension = f"{sandbox.directory}/output-limit.mjs"
            assert sandbox.files[extension] == Path(__file__).parents[1].joinpath("output-limit.mjs").read_text()
            assert payload["command"][payload["command"].index("--extension") + 1] == extension
            models = json.loads(sandbox.files[f"{sandbox.directory}/home/.pi/agent/models.json"])
            assert models["providers"]["nemo"]["models"][0]["maxTokens"] == 123
            assert models["providers"]["nemo"]["baseUrl"] == "http://model.example:9000/ng-rollout/pi-smoke-a2/v1"
            closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
            assert closed.status_code == 200, closed.text
            observations = closed.json()["agent_observations"]
            assert observations["source"] == "pi"
            assert "no_sandbox_runtime" not in [gap["code"] for gap in observations["gaps"]]
            assert len(observations["records"][0]["model_calls"]) == 2
    assert not agent._sandbox_sessions
    assert agent._local_setup_task is None
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, 2**53])
def test_invalid_output_limit_does_not_consume_activation(setup, limit):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        seeded = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert seeded.status_code == 200
        response = client.post(
            "/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code", "max_output_tokens": limit}
        )
        assert response.status_code == 422
        assert not next(iter(agent._sandbox_sessions.values())).activated
        sandbox.pty.create.assert_not_awaited()
        response = client.post(
            "/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code", "max_output_tokens": 128}
        )
        assert response.status_code == 200


def test_native_output_limit_uses_config_default(setup):
    agent, sandbox = setup
    agent.config.max_output_tokens = 4096
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        assert client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code"}).status_code == 200
        models = json.loads(sandbox.files[f"{sandbox.directory}/home/.pi/agent/models.json"])
        assert models["providers"]["nemo"]["models"][0]["maxTokens"] == 4096


def test_invalid_native_config_output_limit_does_not_consume_activation(setup):
    agent, sandbox = setup
    agent.config.max_output_tokens = 0
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        response = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code"})
        assert response.status_code == 422
        assert not next(iter(agent._sandbox_sessions.values())).activated
        sandbox.pty.create.assert_not_awaited()


def test_direct_run_without_resources_rejected_before_execution(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/run", json={"responses_create_params": {"input": "task"}})
    assert response.status_code == 422
    assert "use EnvironmentServer /run" in response.json()["detail"]
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
        agent.config.pi_version = "latest"
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
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        assert client.post("/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/ng-rollout/wrong-a2/v1/responses", json={"input": "task"}).status_code == 409
        bad = close_body(session_id)
        bad["episode_id"]["attempt"] = 99
        assert client.post("/v1/agent_sessions/close", json=bad).status_code == 409
        assert client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "task"}).status_code == 200
        assert client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.pty.create.assert_awaited_once()


@pytest.mark.parametrize("reason,expected", [("error", "failed"), ("aborted", "failed"), ("length", "incomplete")])
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
    sandbox.pty.create.assert_not_awaited()


def test_rejected_request_does_not_consume_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/pi-smoke-a2/v1/responses"
        assert client.post(path, json={"input": "task", "temperature": 0.2}).status_code == 422
        sandbox.pty.create.assert_not_awaited()
        accepted = client.post(path, json={"input": "task"})
        assert accepted.status_code == 200, accepted.text
        assert client.post(path, json={"input": "task"}).status_code == 409
    sandbox.pty.create.assert_awaited_once()


def test_no_session_keeps_existing_local_path(setup):
    agent, sandbox = setup
    with patch.object(agent, "_create_episode", AsyncMock(side_effect=RuntimeError("legacy path reached"))) as legacy:
        with TestClient(agent.setup_webserver()) as client:
            with pytest.raises(RuntimeError, match="legacy path reached"):
                client.post("/v1/responses", json={"input": "task"})
        legacy.assert_awaited_once()
    sandbox.pty.create.assert_not_awaited()


async def activate(agent, sandbox):
    request = Request({"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": "pi-smoke-a2"}})
    seeded = await agent.seed_agent_session(request, seed())
    task = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.wait_for(sandbox.started.wait(), 2)
    return request, seeded.agent_session_id, task


async def test_close_cancels_active_pi_before_detaching(setup):
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
        PiSandboxResult.model_validate({"return_code": 0, "error": None})


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
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
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
        replacement = seed().model_copy(update={"agent_session_id": "replacement-session"})
        assert client.post("/v1/agent_sessions", json=replacement.model_dump(mode="json")).status_code == 200
    sandbox.disconnect.assert_awaited_once()


def test_http_close_retry_survives_other_session_closes(setup, monkeypatch):
    agent, sandbox = setup
    monkeypatch.setattr("responses_api_agents.pi_agent.app.monotonic", lambda: 100.0)
    with TestClient(agent.setup_webserver()) as client:

        def seed_and_close(index):
            client.cookies.clear()
            body = seed().model_dump(mode="json")
            body["agent_session_id"] = f"session-{index}"
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
    monkeypatch.setattr("responses_api_agents.pi_agent.app.monotonic", lambda: clock[0])
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
    with patch.object(agent, "_create_episode", AsyncMock(side_effect=AssertionError("host fallback"))):
        with pytest.raises(HTTPException) as error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert error.value.status_code == 409
    sandbox.disconnect.assert_awaited_once()


async def test_close_retry_window_starts_after_cleanup(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.pi_agent.app.monotonic", lambda: clock[0])
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
    monkeypatch.setattr("responses_api_agents.pi_agent.app.monotonic", lambda: clock[0])
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    clock[0] += agent.config.session_close_retry_window_seconds
    await agent.seed_agent_session(request, seed().model_copy(update={"agent_session_id": "replacement-session"}))
    assert not agent._closed_sandbox_sessions


@pytest.mark.parametrize("window", [0, -1, float("inf")])
def test_close_retry_window_must_be_positive_and_finite(setup, window):
    agent, _ = setup
    with pytest.raises(ValidationError):
        PiAgentConfig(**(agent.config.model_dump() | {"session_close_retry_window_seconds": window}))


async def test_unknown_launch_outcome_fails_closed(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "pi-smoke-a2"}})
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
        SimpleNamespace(return_code=0),
        SimpleNamespace(return_code=1, stderr="npm failed"),
        SimpleNamespace(return_code=0),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="npm failed"):
        await agent.seed_agent_session(request, seed())
    assert not agent._sandbox_sessions
    assert not request.session
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_cancelled_install_never_publishes_session_or_launches_pi(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0),
        asyncio.CancelledError(),
        SimpleNamespace(return_code=0),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(asyncio.CancelledError):
        await agent.seed_agent_session(request, seed())
    assert not agent._sandbox_sessions
    assert not request.session
    sandbox.pty.create.assert_not_awaited()
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
        assert not next(iter(agent._sandbox_sessions.values())).activated
        sandbox.pty.create.assert_not_awaited()
        response = client.post(
            "/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code", "model": "test-model"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "completed"


def native_event_response(agent, sandbox, recorded_events):
    sandbox.events = "\n".join(json.dumps([i, event]) for i, event in enumerate(recorded_events))
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200
        response = client.post("/ng-rollout/pi-smoke-a2/v1/responses", json={"input": "Fix the code"})
        assert response.status_code == 200, response.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    return response.json(), closed.json()["agent_observations"]


@pytest.mark.parametrize(
    "final_stop, expected", [("stop", "completed"), ("length", "incomplete"), ("error", "failed")]
)
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
    response, observations = native_event_response(
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
    response, _ = native_event_response(agent, sandbox, recorded)
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
    response, observations = native_event_response(agent, sandbox, recorded)
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
    response, _ = native_event_response(agent, sandbox, recorded)
    assert response["usage"]["input_tokens_details"]["cached_tokens"] is None


async def test_native_recipe_collects_through_environment_run(setup, monkeypatch):
    """Exercise the checked-in recipe through collector, real environment, and real Pi lifecycle."""
    from environment_servers.single_agent_turn.app import (
        SingleAgentTurnEnvironmentServer,
        SingleAgentTurnEnvironmentServerConfig,
    )
    from nemo_gym.global_config import GlobalConfigDictParser
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
    from nemo_gym.server_utils import BaseServerConfig
    from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest

    agent, sandbox = setup
    recipe_path = Path(__file__).parents[3] / "benchmarks/swebench/pro/pi_native.yaml"
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(recipe_path)])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    assert config.environment_routing_mode == "taskset"
    environment_name = config.environment_server_routes["swebench_pro:smoke"]
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
                assert body.task_data == {"instance_id": "instance"}
                return Response(
                    {
                        "resources_session_id": body.resources_session_id,
                        "sandbox_access": seed().sandbox_access.model_dump(),
                    },
                    cookie="resources-cookie",
                )
            assert kwargs["cookies"] == {"session": "resources-cookie"}
            if url_path == "/verify":
                assert not agent._sandbox_sessions
                assert sandbox.disconnect.await_count == 1
                return Response({**body.verification_input.model_dump(mode="json"), "reward": 1.0})
            assert url_path == "/close_session"
            return Response({"resources_session_id": body.resources_session_id})
        assert server_name == agent_name
        request = Request({"type": "http", "session": cookies})
        if url_path == "/v1/agent_sessions":
            result = await agent.seed_agent_session(request, body)
            assert result.agent_session_id == body.agent_session_id
        elif url_path == "/v1/agent_sessions/close":
            result = await agent.close_agent_session(request, body)
        else:
            assert url_path.endswith("/v1/responses")
            request.scope["path_params"] = {"rollout_id": url_path.split("/")[2]}
            result = await agent.responses(request, body)
        return Response(result.model_dump(mode="json"), cookie="agent-cookie")

    monkeypatch.setattr(ServerClient, "post", post)
    monkeypatch.setattr(ServerClient, "_resolve_base_url", lambda self, name: f"http://{name}:8000")
    monkeypatch.setattr(RolloutCollectionHelper, "setup_server_client", lambda self, head=None: transport)
    materialized = {
        "task_id": {"taskset": "swebench_pro:smoke", "task_id": "instance"},
        "task_input": {"responses_create_params": {"input": "Fix it"}, "task_data": {"instance_id": "instance"}},
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
    assert result["result"]["verification"]["reward"] == 1.0
    assert result["result"]["verification"]["response"]["usage"]["total_tokens"] == 22
    assert result["result"]["agent_observations"]["source"] == "pi"
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
        assert agent._sandbox_sessions[session_id].closing
        with pytest.raises(HTTPException):
            await agent.seed_agent_session(Request({"type": "http", "session": {}}), seed())
    else:
        assert not agent._sandbox_sessions
    sandbox.exec.side_effect = None
    sandbox.disconnect.side_effect = None
    result = await agent.close_agent_session(
        Request({"type": "http", "session": {}}), AgentCloseSessionRequest(**close_body(session_id))
    )
    assert result.agent_session_id == session_id
    assert not agent._sandbox_sessions
    assert not agent._session_expiry_tasks
    sandbox.stop.assert_not_awaited()


async def test_caller_id_seed_retries_share_one_session_without_cookies(setup):
    agent, sandbox = setup
    body = seed(session_id="caller-assigned-id")
    requests = [Request({"type": "http", "session": {}}) for _ in range(2)]
    first, second = await asyncio.gather(*(agent.seed_agent_session(request, body) for request in requests))
    assert first.agent_session_id == second.agent_session_id == body.agent_session_id
    assert len(agent._sandbox_sessions) == 1
    assert sandbox.exec.await_count == 2  # One path check and one installation.
    assert requests[0].session == requests[1].session
    await agent.close_agent_session(
        Request({"type": "http", "session": {}}),
        AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id),
    )
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("field", ["episode", "task", "workdir"])
async def test_caller_id_rejects_changed_seed_binding(setup, field):
    agent, _ = setup
    body = seed()
    request = Request({"type": "http", "session": {}})
    await agent.seed_agent_session(request, body)
    changed = body.model_copy(deep=True)
    if field == "episode":
        changed.episode_id = changed.episode_id.model_copy(update={"attempt": changed.episode_id.attempt + 1})
    elif field == "task":
        changed.task_id = changed.task_id.model_copy(update={"task_id": "another-task"})
    else:
        changed.sandbox_access.workdir = "/another-repository"
    with pytest.raises(HTTPException, match="different seed"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), changed)
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


async def test_close_before_seed_prevents_late_creation(setup):
    agent, sandbox = setup
    body = seed()
    request = Request({"type": "http", "session": {}})
    close = AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)
    first = await agent.close_agent_session(request, close)
    assert await agent.close_agent_session(Request({"type": "http", "session": {}}), close) is first
    with pytest.raises(HTTPException, match="already closed"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    sandbox.exec.assert_not_awaited()


async def test_close_serializes_with_inflight_seed(setup):
    agent, sandbox = setup
    body = seed()
    entered, release = asyncio.Event(), asyncio.Event()
    original = agent._initialize_agent_session_state

    async def blocked_initialize(*args):
        entered.set()
        await release.wait()
        return await original(*args)

    with patch.object(agent, "_initialize_agent_session_state", side_effect=blocked_initialize):
        pending = asyncio.create_task(agent.seed_agent_session(Request({"type": "http", "session": {}}), body))
        await entered.wait()
        close = asyncio.create_task(
            agent.close_agent_session(
                Request({"type": "http", "session": {}}),
                AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id),
            )
        )
        await asyncio.sleep(0)
        assert not close.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(pending, close), 2)
    assert not agent._sandbox_sessions
    sandbox.disconnect.assert_awaited_once()


async def test_caller_id_never_controls_filesystem_path(setup):
    agent, _ = setup
    body = seed(session_id="../../task-repository\nunsafe")
    request = Request({"type": "http", "session": {}})
    response = await agent.seed_agent_session(request, body)
    assert response.agent_session_id == body.agent_session_id
    directory = agent._sandbox_sessions[body.agent_session_id].directory
    assert Path(directory).parent == Path("/tmp/nemo-gym-pi-sessions")
    assert len(Path(directory).name) == 32
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


async def test_abandoned_session_expires_through_normal_cleanup(setup):
    agent, sandbox = setup
    agent.config.session_lifetime_seconds = 0.01
    body = seed()
    await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    expiry = agent._session_expiry_tasks[body.agent_session_id]
    await asyncio.wait_for(asyncio.shield(expiry), 2)
    assert not agent._sandbox_sessions
    assert body.agent_session_id in agent._closed_sandbox_sessions
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_failed_expiry_retains_state_and_rejects_activation(setup, caplog):
    agent, sandbox = setup
    agent.config.session_lifetime_seconds = 0.01
    body = seed()
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": body.episode_id.capture_key}})
    await agent.seed_agent_session(request, body)
    sandbox.disconnect.side_effect = RuntimeError("disconnect unavailable")
    await asyncio.wait_for(asyncio.shield(agent._session_expiry_tasks[body.agent_session_id]), 2)
    assert agent._sandbox_sessions[body.agent_session_id].closing
    assert "retaining failed state" in caplog.text
    with pytest.raises(HTTPException):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    sandbox.pty.create.assert_not_awaited()
    sandbox.disconnect.side_effect = None
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


@pytest.mark.parametrize("lifetime", [0, -1, float("inf"), float("nan")])
def test_session_lifetime_is_positive_and_finite(setup, lifetime):
    agent, _ = setup
    with pytest.raises(ValidationError):
        PiAgentConfig(**(agent.config.model_dump() | {"session_lifetime_seconds": lifetime}))


@pytest.mark.parametrize("marker", [None, "", [], {}, 0])
@pytest.mark.parametrize("endpoint", ["seed", "responses", "close", "run"])
async def test_malformed_native_marker_never_runs_host_pi(setup, marker, endpoint):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {"nemo_gym_pi_sandbox_session": marker}})
    with patch.object(agent, "_create_episode", AsyncMock()) as host:
        with pytest.raises(HTTPException, match="Invalid native Pi session marker"):
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
    sandbox.pty.create.assert_not_awaited()


async def test_native_marker_blocks_legacy_run(setup):
    agent, _ = setup
    request = Request({"type": "http", "session": {"nemo_gym_pi_sandbox_session": "expired-session"}})
    with pytest.raises(HTTPException, match="EnvironmentServer /run"):
        await agent.run(request, PiAgentRunRequest(responses_create_params={"input": "task"}))


async def test_receipt_expiry_does_not_create_empty_cookieless_success(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.pi_agent.app.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    body = seed()
    request = Request({"type": "http", "session": {}})
    await agent.seed_agent_session(request, body)
    close = AgentCloseSessionRequest(**close_body(body.agent_session_id))
    await agent.close_agent_session(request, close)
    clock[0] = 111.0
    with pytest.raises(HTTPException, match="receipt has expired"):
        await agent.close_agent_session(Request({"type": "http", "session": {}}), close)
    with pytest.raises(HTTPException, match="already closed"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    # Even after tombstone pruning, a stale cookie cannot create a new session or receipt.
    clock[0] += agent.config.session_lifetime_seconds
    with pytest.raises(HTTPException, match="expired"):
        await agent.close_agent_session(request, close)
    with pytest.raises(HTTPException, match="expired"):
        await agent.seed_agent_session(request, body)
    sandbox.disconnect.assert_awaited_once()


async def test_expiring_tombstone_preserves_lock_with_a_queued_waiter(setup, monkeypatch):
    agent, _ = setup
    body = seed()
    session_id = body.agent_session_id
    monkeypatch.setattr("responses_api_agents.pi_agent.app.monotonic", lambda: 10.0)
    agent._closed_session_ids[session_id] = (body.episode_id, 1.0)
    lock = agent._session_locks.setdefault(session_id, asyncio.Lock())
    await lock.acquire()

    async def wait_for_lock():
        async with agent._session_lock(session_id):
            assert agent._session_locks[session_id] is lock

    waiter = asyncio.create_task(wait_for_lock())
    await asyncio.sleep(0)
    # release wakes the queued waiter without letting it reacquire until this coroutine yields.
    lock.release()
    assert not lock.locked()
    agent._expire_closed_agent_sessions()
    assert agent._session_locks[session_id] is lock
    await waiter
    assert session_id not in agent._session_locks
