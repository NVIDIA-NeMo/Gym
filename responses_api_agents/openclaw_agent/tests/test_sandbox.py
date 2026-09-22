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
from responses_api_agents.openclaw_agent.app import OpenClawAgent, OpenClawAgentConfig
from responses_api_agents.openclaw_agent.sandbox import OpenClawSandboxResult


def seed() -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        episode_id=EpisodeId(rollout_id="openclaw-smoke", attempt=2),
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
        json.dumps(event)
        for i, event in enumerate(
            [
                {
                    "type": "message",
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
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": "tool-1",
                        "toolName": "bash",
                        "content": [{"type": "text", "text": "/app"}],
                    },
                },
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "responseId": "call-2",
                        "content": [{"type": "text", "text": "Fixed"}],
                        "usage": {"input": 5, "output": 2},
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
        self.files[f"{self.directory}/stdout.log"] = ""
        self.files[f"{self.directory}/stderr.log"] = "model error"
        session_id = self.directory.rsplit("/", 1)[-1]
        self.files[f"{self.directory}/home/.openclaw/agents/main/sessions/{session_id}.jsonl"] = self.events
        return 0


@pytest.fixture
def setup():
    sandbox = Sandbox()
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {"policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}}
    )
    client._build_server_base_url.return_value = "http://model.example:9000"
    config = OpenClawAgentConfig(
        name="openclaw",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        model_server={"type": "responses_api_models", "name": "policy"},
        model="test-model",
        openclaw_version="2026.6.11",
        session_close_timeout_seconds=1,
    )
    module = "responses_api_agents.openclaw_agent.app"
    with (
        patch(
            f"{module}.ensure_openclaw", side_effect=AssertionError("native sessions must not install host OpenClaw")
        ),
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider"),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
    ):
        agent = OpenClawAgent(config=config, server_client=client)
        yield agent, sandbox


def close_body(session_id):
    return {"agent_session_id": session_id, "episode_id": seed().episode_id.model_dump()}


def test_http_native_flow_runs_openclaw_in_borrowed_sandbox(setup):
    agent, sandbox = setup
    with patch.object(agent, "_run_openclaw", AsyncMock(side_effect=AssertionError("host OpenClaw must not run"))):
        with TestClient(agent.setup_webserver()) as client:
            created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
            assert created.status_code == 200, created.text
            session_id = created.json()["agent_session_id"]
            directory = f"/tmp/nemo-gym-openclaw-sessions/{session_id}"
            installer = f"{directory}/install_openclaw_runtime.sh"
            assert installer in sandbox.files
            assert agent.config.resources_server is None
            assert not sandbox.pty.create.called
            assert not any(path.startswith("/app/") for path in sandbox.files)
            install_call = sandbox.exec.await_args_list[1]
            assert install_call.args[0] == (
                f"bash {installer} /tmp/nemo-gym-openclaw-node-22.19.0-2026.6.11 2026.6.11"
            )
            assert install_call.kwargs["timeout_s"] == agent.config.sandbox_install_timeout_seconds
            result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "Fix the code"})
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
            models = json.loads(sandbox.files[f"{sandbox.directory}/home/.openclaw/openclaw.json"])
            assert models["agents"]["defaults"]["workspace"] == "/app"
            assert models["models"]["providers"]["nemo"]["models"][0]["compat"]["supportsUsageInStreaming"] is True
            assert (
                models["models"]["providers"]["nemo"]["baseUrl"]
                == "http://model.example:9000/ng-rollout/openclaw-smoke-a2/v1"
            )
            closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
            assert closed.status_code == 200, closed.text
            observations = closed.json()["agent_observations"]
            assert observations["source"] == "openclaw"
            assert "no_sandbox_runtime" not in [gap["code"] for gap in observations["gaps"]]
            assert len(observations["records"][0]["model_calls"]) == 2
    assert not agent._sandbox_sessions
    assert agent._local_setup_task is None
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    agent.server_client.post.assert_not_called()


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
        agent.config.openclaw_version = "latest"
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
        assert client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"}).status_code == 200
        assert client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.pty.create.assert_awaited_once()


@pytest.mark.parametrize("reason,expected", [("error", "failed"), ("aborted", "failed"), ("length", "incomplete")])
def test_failed_or_partial_openclaw_output_is_preserved(setup, reason, expected):
    agent, sandbox = setup
    sandbox.events = events(stop_reason=reason)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200
        assert result.json()["status"] == expected
        assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200


@pytest.mark.parametrize(
    "override",
    [
        {"temperature": 0.2},
        {"max_output_tokens": 123},
        {"top_k": 3},
        {"input": ""},
        {"top_p": 0.9},
        {"tools": [{"type": "function", "name": "foo"}]},
        {"input": [{"role": "assistant", "content": "old turn"}]},
    ],
)
def test_unsupported_request_is_not_silently_ignored(setup, override):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task", **override})
        assert result.status_code == 422, result.text
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.pty.create.assert_not_awaited()


def test_rejected_request_does_not_consume_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/openclaw-smoke-a2/v1/responses"
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
    request = Request(
        {"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": "openclaw-smoke-a2"}}
    )
    seeded = await agent.seed_agent_session(request, seed())
    task = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await asyncio.wait_for(sandbox.started.wait(), 2)
    return request, seeded.agent_session_id, task


async def test_close_cancels_active_openclaw_before_detaching(setup):
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
        OpenClawSandboxResult.model_validate({"return_code": 0, "error": None})


def test_instructions_and_text_parts_reach_openclaw_without_other_provider_credentials(setup):
    agent, sandbox = setup
    agent.config.system_prompt = "config instruction"
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post(
            "/ng-rollout/openclaw-smoke-a2/v1/responses",
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
        assert payload["prompt"] == "config instruction\n\nrequest instruction\n\ninput instruction\n\ntask"
        models = json.loads(sandbox.files[f"{sandbox.directory}/home/.openclaw/openclaw.json"])
        assert list(models["models"]["providers"]) == ["nemo"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        conversation = closed.json()["agent_observations"]["records"][0]["conversation"]
        assert conversation[0]["content"] == "config instruction\n\nrequest instruction\n\ninput instruction"


def test_observation_parse_failure_preserves_response(setup):
    agent, _ = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        with patch(
            "responses_api_agents.openclaw_agent.app.build_openclaw_observations",
            side_effect=ValueError("bad event"),
        ):
            response = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        assert closed.json()["agent_observations"]["gaps"][0]["code"] == "observation_capture_failed"


def test_close_retry_and_stale_activation_do_not_run_host_openclaw(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"}).raise_for_status()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        repeated = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert repeated.status_code == 200
        assert repeated.json() == closed.json()
        bad = close_body(session_id)
        bad["episode_id"]["attempt"] = 99
        assert client.post("/v1/agent_sessions/close", json=bad).status_code == 409
        with patch.object(agent, "_run_openclaw", AsyncMock(side_effect=AssertionError("host OpenClaw must not run"))):
            assert client.post("/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
    sandbox.disconnect.assert_awaited_once()


def test_http_close_retry_survives_other_session_closes(setup, monkeypatch):
    agent, sandbox = setup
    monkeypatch.setattr("responses_api_agents.openclaw_agent.app.monotonic", lambda: 100.0)
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
    monkeypatch.setattr("responses_api_agents.openclaw_agent.app.monotonic", lambda: clock[0])
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
    monkeypatch.setattr("responses_api_agents.openclaw_agent.app.monotonic", lambda: clock[0])
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
    monkeypatch.setattr("responses_api_agents.openclaw_agent.app.monotonic", lambda: clock[0])
    request = Request({"type": "http", "session": {}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    clock[0] += agent.config.session_close_retry_window_seconds
    await agent.seed_agent_session(request, seed())
    assert not agent._closed_sandbox_sessions


@pytest.mark.parametrize("window", [0, -1, float("inf")])
def test_close_retry_window_must_be_positive_and_finite(setup, window):
    agent, _ = setup
    with pytest.raises(ValidationError):
        OpenClawAgentConfig(**(agent.config.model_dump() | {"session_close_retry_window_seconds": window}))


async def test_unknown_launch_outcome_fails_closed(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "openclaw-smoke-a2"}})
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
    sandbox.exec.side_effect = [SimpleNamespace(return_code=0), SimpleNamespace(return_code=1, stderr="npm failed")]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="npm failed"):
        await agent.seed_agent_session(request, seed())
    assert not agent._sandbox_sessions
    assert not request.session
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_cancelled_install_never_publishes_session_or_launches_openclaw(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [SimpleNamespace(return_code=0), asyncio.CancelledError()]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(asyncio.CancelledError):
        await agent.seed_agent_session(request, seed())
    assert not agent._sandbox_sessions
    assert not request.session
    sandbox.pty.create.assert_not_awaited()
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize(
    "override",
    [{"model": "different-model"}, {"store": False}, {"include": []}, {"service_tier": "priority"}],
)
def test_model_and_other_request_controls_are_explicitly_rejected(setup, override):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task", **override})
        assert result.status_code == 422, result.text
    sandbox.pty.create.assert_not_awaited()


async def test_cleanup_command_provider_error_blocks_close_and_can_retry(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type="timeout", stderr="cleanup timed out"),
        SimpleNamespace(return_code=0, error_type=None, stderr=""),
    ]
    with pytest.raises(RuntimeError, match="cleanup timed out"):
        await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    sandbox.disconnect.assert_not_awaited()
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    sandbox.disconnect.assert_awaited_once()


async def test_simultaneous_activation_is_rejected_before_launch(setup):
    agent, sandbox = setup
    sandbox.blocked = True
    request, session_id, task = await activate(agent, sandbox)
    with pytest.raises(HTTPException) as error:
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="second task"))
    assert error.value.status_code == 409
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    sandbox.pty.create.assert_awaited_once()


async def test_close_start_prevents_activation(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "openclaw-smoke-a2"}})
    session_id = (await agent.seed_agent_session(request, seed())).agent_session_id
    entered, release = asyncio.Event(), asyncio.Event()

    async def disconnect():
        entered.set()
        await release.wait()

    sandbox.disconnect.side_effect = disconnect
    close = asyncio.create_task(agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id))))
    await entered.wait()
    with pytest.raises(HTTPException) as error:
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert error.value.status_code == 409
    release.set()
    await close
    sandbox.pty.create.assert_not_awaited()


async def test_cancelled_activation_keeps_partial_observations_for_close(setup):
    agent, sandbox = setup
    sandbox.blocked = True
    request, session_id, task = await activate(agent, sandbox)
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    with pytest.raises(asyncio.CancelledError):
        await task
    invocation = closed.agent_observations.records[0]
    assert invocation.status == "incomplete"
    assert invocation.conversation[-1].content[0].text == "Fixed"
    assert len(invocation.model_calls) == 2


def test_usage_sums_cache_writes_and_failed_calls(setup):
    agent, sandbox = setup
    transcript = [json.loads(line) for line in events(stop_reason="error").splitlines()]
    transcript[0]["message"]["usage"]["cacheWrite"] = 4
    sandbox.events = "\n".join(json.dumps(event) for event in transcript)
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        response = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
    assert response.json()["status"] == "failed"
    assert response.json()["usage"]["input_tokens"] == 21
    assert response.json()["usage"]["output_tokens"] == 5
    assert response.json()["usage"]["total_tokens"] == 26


async def test_two_sessions_keep_workspaces_model_routes_and_observations_separate(setup):
    agent, first = setup
    second = Sandbox()
    second.events = second.events.replace("Fixed", "Second result")
    request1 = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "openclaw-smoke-a2"}})
    request2 = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "second-a2"}})
    seed2 = seed()
    seed2.episode_id = EpisodeId(rollout_id="second", attempt=2)
    with patch("responses_api_agents.openclaw_agent.app.AsyncSandbox.connect", AsyncMock(side_effect=[first, second])):
        created1 = await agent.seed_agent_session(request1, seed())
        created2 = await agent.seed_agent_session(request2, seed2)
    result1, result2 = await asyncio.gather(
        agent.responses(request1, NeMoGymResponseCreateParamsNonStreaming(input="first task")),
        agent.responses(request2, NeMoGymResponseCreateParamsNonStreaming(input="second task")),
    )
    assert result1.output[-1].content[0].text == "Fixed"
    assert result2.output[-1].content[0].text == "Second result"
    assert first.directory != second.directory
    assert created1.agent_session_id != created2.agent_session_id
    for sandbox, route in [(first, "openclaw-smoke-a2"), (second, "second-a2")]:
        config = json.loads(sandbox.files[f"{sandbox.directory}/home/.openclaw/openclaw.json"])
        assert config["models"]["providers"]["nemo"]["baseUrl"].endswith(f"/ng-rollout/{route}/v1")
    await agent.close_agent_session(request1, AgentCloseSessionRequest(**close_body(created1.agent_session_id)))
    await agent.close_agent_session(
        request2, AgentCloseSessionRequest(agent_session_id=created2.agent_session_id, episode_id=seed2.episode_id)
    )
    first.disconnect.assert_awaited_once()
    second.disconnect.assert_awaited_once()


@pytest.mark.parametrize("marker", [None, 7, "expired-session"])
async def test_invalid_or_expired_native_cookie_never_uses_local_execution(setup, marker):
    agent, sandbox = setup
    request = Request(
        {
            "type": "http",
            "session": {"nemo_gym_openclaw_sandbox_session": marker},
            "path_params": {"rollout_id": "openclaw-smoke-a2"},
        }
    )
    with patch.object(agent, "_create_response", AsyncMock(side_effect=AssertionError("host fallback"))):
        with pytest.raises(HTTPException) as error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert error.value.status_code == 409
    sandbox.pty.create.assert_not_awaited()


def test_native_close_cookie_blocks_legacy_run_with_configured_resources(setup):
    from nemo_gym.config_types import ResourcesServerRef

    agent, sandbox = setup
    agent.config.resources_server = ResourcesServerRef(type="resources_servers", name="legacy-resources")
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        assert client.cookies  # Exercise the actual signed cookie returned by close.
        response = client.post("/run", json={"responses_create_params": {"input": "task"}})
        assert response.status_code == 409
    agent.server_client.post.assert_not_called()
    sandbox.pty.create.assert_not_awaited()


def test_zero_harness_counters_are_reported_as_missing_usage(setup):
    agent, sandbox = setup
    transcript = [json.loads(line) for line in events().splitlines()]
    for event in transcript:
        if event.get("message", {}).get("role") == "assistant":
            event["message"]["usage"] = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    sandbox.events = "\n".join(json.dumps(event) for event in transcript)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200
        assert response.json()["usage"]["total_tokens"] == 0
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
    gaps = closed.json()["agent_observations"]["gaps"]
    assert any(gap["code"] == "model_call_usage_unavailable" and "zero" in gap["detail"] for gap in gaps)


async def test_invalid_envelope_usage_does_not_discard_valid_transcript(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    state = agent._sandbox_sessions[session_id]
    response = await agent._collect_sandbox_response(
        state,
        NeMoGymResponseCreateParamsNonStreaming(input="task"),
        prompt="task",
        system="",
        stdout=json.dumps({"meta": {"agentMeta": {"usage": {"input": "not-a-number"}}}}),
    )
    assert response.status == "completed"
    assert response.output[-1].content[0].text == "Fixed"
    assert response.usage.total_tokens == 22
    assert "agent_stdout_unparseable" in {gap.code for gap in state.observations.gaps}


@pytest.mark.parametrize("owned_root", ["sessions", "runtime"])
@pytest.mark.parametrize("overlap", ["same", "ancestor", "descendant", "symlink", "dotdot"])
def test_workdir_cannot_overlap_adapter_storage(tmp_path, owned_root, overlap):
    import subprocess
    import sys

    from responses_api_agents.openclaw_agent.sandbox import _SANDBOX_PATH_CHECK

    sessions, runtime = tmp_path / "sessions", tmp_path / "runtime"
    for directory in (sessions, runtime):
        directory.mkdir()
        (directory / "repo").mkdir()
    owned = tmp_path / owned_root
    if overlap == "same":
        workdir = owned
    elif overlap == "ancestor":
        workdir = tmp_path
    elif overlap == "descendant":
        workdir = owned / "repo"
    elif overlap == "symlink":
        workdir = tmp_path / "task-alias"
        workdir.symlink_to(owned)
    else:
        workdir = owned / "repo" / ".."
    result = subprocess.run(
        [sys.executable, "-c", _SANDBOX_PATH_CHECK, str(workdir), str(sessions), str(runtime)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=5,
    )
    assert result.returncode != 0
    assert "overlaps adapter-owned storage" in result.stderr


def test_workdir_validation_accepts_separate_repository_and_rejects_missing(tmp_path):
    import subprocess
    import sys

    from responses_api_agents.openclaw_agent.sandbox import _SANDBOX_PATH_CHECK

    workdir = tmp_path / "task"
    command = [
        sys.executable,
        "-c",
        _SANDBOX_PATH_CHECK,
        str(workdir),
        str(tmp_path / "sessions"),
        str(tmp_path / "runtime"),
    ]
    missing = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=5)
    assert missing.returncode != 0
    assert "task workdir is missing" in missing.stderr
    workdir.mkdir()
    valid = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=5)
    assert valid.returncode == 0, valid.stderr
    assert not (tmp_path / "sessions").exists()
    assert not (tmp_path / "runtime").exists()


async def test_path_check_failure_never_uploads_or_installs(setup):
    agent, sandbox = setup
    sandbox.exec.return_value = SimpleNamespace(return_code=1, stderr="task workdir overlaps adapter-owned storage")
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="overlaps adapter-owned storage"):
        await agent.seed_agent_session(request, seed())
    assert not sandbox.files
    assert not request.session
    sandbox.exec.assert_awaited_once()
    sandbox.disconnect.assert_awaited_once()
