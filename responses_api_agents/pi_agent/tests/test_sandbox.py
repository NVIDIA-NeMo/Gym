# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import signal
import sys
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
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig, PiAgentRunRequest


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
        self.exec = AsyncMock(side_effect=self.execute)
        self.stop = AsyncMock()
        self.disconnect = AsyncMock()
        self.launch = AsyncMock(side_effect=self.run)
        self.signal = AsyncMock(side_effect=self.stop_runner)
        self.cleanup_available = True

    async def execute(self, command, **kwargs):
        if "--receipt" in command and "process_supervisor.py" in command:
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
        payload_path = next(path for path in self.files if path.endswith("/input.json"))
        payload = json.loads(self.files[payload_path])
        assert payload["cwd"] == getattr(self, "expected_workdir", "/app")
        assert kwargs["cwd"] == getattr(self, "expected_workdir", "/app")
        assert "sandbox_runner.py" in command
        assert "process_supervisor.py" in command
        self.directory = payload["directory"]
        self.started.set()
        if not self.blocked:
            self.exited.set()
        await self.exited.wait()
        self.save_result()
        return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")

    async def stop_runner(self):
        self.directory = next(path.rsplit("/", 1)[0] for path in self.files if path.endswith("/input.json"))
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


def test_http_session_flow_runs_pi_in_borrowed_sandbox(setup):
    agent, sandbox = setup
    with patch.object(agent, "_run_pi", AsyncMock(side_effect=AssertionError("host Pi must not run"))):
        with TestClient(agent.setup_webserver()) as client:
            created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
            assert created.status_code == 200, created.text
            session_id = created.json()["agent_session_id"]
            directory = agent._session_records[session_id].state.session.session_dir
            assert f"{directory}/process_supervisor.py" not in sandbox.files
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


async def activate(agent, sandbox):
    request = Request({"type": "http", "headers": [], "session": {}, "path_params": {"rollout_id": "pi-smoke-a2"}})
    seeded = await agent.seed_agent_session(request, seed())
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
    payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
    assert payload["env"]["NEMO_GYM_PI_BASH_TIMEOUT"] == "123"
    assert f"{sandbox.directory}/runtime-guards.mjs" in payload["command"]
    assert "tool_call" in sandbox.files[f"{sandbox.directory}/runtime-guards.mjs"]
    settings = json.loads(sandbox.files[f"{sandbox.directory}/home/.pi/agent/settings.json"])
    assert settings["httpIdleTimeoutMs"] == 10800000
    assert settings["retry"]["provider"]["timeoutMs"] == 10800000
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


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
    from nemo_gym.global_config import GlobalConfigDictParser
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
    from responses_api_agents.pi_agent import sandbox_runner

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
        # This stdlib-only event producer must not load the host's editable installs or site hooks.
        "command": [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import json,sys; print(json.dumps({'prompt':sys.stdin.read()}))",
        ],
        "env": {},
        "prompt": "real invocation",
    }
    raw = await state.execute(payload, timeout=3, close_timeout=2)
    diagnostics = {
        name: path.read_text(errors="replace")
        for name in ("cleanup.json", "runtime.json", "output.log", "stderr.log", "events.jsonl")
        if (path := Path(state.session.session_dir) / name).exists()
    }
    assert state.session.cleanup["cleanup_confirmed"] and state.session.cleanup["return_code"] == 0, diagnostics
    assert raw.strip(), diagnostics
    _, event = json.loads(raw)
    assert event == {"prompt": "real invocation"}
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


@pytest.mark.parametrize("field", ["timeout", "sandbox_install_timeout_seconds", "session_close_timeout_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_pi_rejects_invalid_execution_deadlines(field, value):
    with pytest.raises(ValidationError):
        PiAgentConfig(name="pi", host="localhost", port=1, entrypoint="app.py", **{field: value})


@pytest.mark.parametrize("context_overflow", [False, True])
async def test_provider_failure_is_not_gradable_but_context_limit_preserves_patch(setup, context_overflow):
    agent, sandbox = setup
    recorded = [json.loads(line)[1] for line in events(stop_reason="error").splitlines()]
    recorded.insert(-2, {"type": "ng_pi_outcome", "context_overflow": context_overflow})
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(recorded))
    request, session_id, task = await activate(agent, sandbox)
    if context_overflow:
        response = await task
        assert response.status == "incomplete"
        assert response.error is None
        assert response.output[-1].content[0].text == "Fixed"
    else:
        with pytest.raises(RuntimeError, match="Pi agent failed: model error"):
            await task
        # The shared activation replays the failure instead of running the harness again.
        with pytest.raises(RuntimeError, match="Pi agent failed: model error"):
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    close = await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))
    assert len(close.agent_observations.records[0].model_calls) == 2
    assert close.agent_observations.records[0].status == ("incomplete" if context_overflow else "failed")
    sandbox.launch.assert_awaited_once()
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
