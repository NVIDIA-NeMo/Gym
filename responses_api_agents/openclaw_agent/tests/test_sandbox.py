# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from copy import deepcopy
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
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseOutputMessage
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.openclaw_agent.app import OpenClawAgent, OpenClawAgentConfig, _unique_usage_messages


def seed(*, session_id: str | None = None) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id=session_id or f"openclaw-{uuid4().hex}",
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
        }
        self.runtime_info = {"hostname": "task-container", "pid": 123}
        self.events = events()
        self.blocked = False
        self.started = asyncio.Event()
        self.exited = asyncio.Event()
        self.exec = AsyncMock(
            side_effect=self.run_exec,
            return_value=SimpleNamespace(error_type=None, return_code=0, stdout="", stderr=""),
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
            f"{module}.ensure_openclaw", side_effect=AssertionError("sandbox sessions must not install host OpenClaw")
        ),
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider"),
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
    ):
        agent = OpenClawAgent(config=config, server_client=client)
        yield agent, sandbox


@pytest.mark.parametrize("stage", ["prepare", "install"])
@pytest.mark.parametrize("error_type", ["timeout", "sandbox"])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [("bootstrap output", "provider failed"), (None, "provider failed"), ("bootstrap output", None), (None, None)],
)
async def test_provider_error_type_blocks_setup_even_with_zero_exit(setup, stage, error_type, owned, stdout, stderr):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}})
    body = seed()
    if owned:
        agent.config.sandbox_provider = "sandbox"
        agent.config.sandbox_config = {"image": "test-image", "workdir": "/app"}
        body = body.model_copy(update={"sandbox_access": None})
        sandbox.start = AsyncMock()
    ok = SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
    error = SandboxExecResult(return_code=0, error_type=error_type, stdout=stdout, stderr=stderr)
    sandbox.exec.side_effect = ([ok] if owned else []) + ([error, ok] if stage == "prepare" else [ok, error, ok])
    with patch("responses_api_agents.openclaw_agent.app.AsyncSandbox", return_value=sandbox) as sandbox_class:
        sandbox_class.connect = AsyncMock(return_value=sandbox)
        with pytest.raises(RuntimeError, match=f"error={error_type}") as failed:
            await agent.seed_agent_session(request, body)
    assert "exit 0" in str(failed.value)
    if stdout is not None:
        assert stdout in str(failed.value)
    if stderr is not None:
        assert stderr in str(failed.value)
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


@pytest.mark.parametrize("model_timeout_seconds", [600, 1200])
def test_http_sandbox_flow_runs_openclaw_in_borrowed_sandbox(setup, model_timeout_seconds):
    agent, sandbox = setup
    agent.config.model_timeout_seconds = model_timeout_seconds
    with patch.object(agent, "_run_openclaw", AsyncMock(side_effect=AssertionError("host OpenClaw must not run"))):
        with TestClient(agent.setup_webserver()) as client:
            created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
            assert created.status_code == 200, created.text
            session_id = created.json()["agent_session_id"]
            directory = agent._session_records[session_id].state.session.session_dir
            installer = f"{directory}/install_openclaw_runtime.sh"
            assert installer in sandbox.files
            assert agent.config.resources_server is None
            assert not sandbox.launch.called
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
            assert body["usage"]["input_tokens_details"]["cached_tokens"] is None
            assert body["usage"]["output_tokens_details"]["reasoning_tokens"] is None
            assert body["metadata"]["harness_execution"] == "sandbox"
            assert "_ng_agent_observations" not in body
            payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
            assert payload["prompt"] == "Fix the code"
            assert payload["command"][0].endswith("/node/bin/node")
            assert "--timeout 900" in sandbox.launch.await_args.kwargs["command"]
            assert "timeout" not in payload
            assert "PATH" not in payload["env"]
            models = json.loads(sandbox.files[f"{sandbox.directory}/home/.openclaw/openclaw.json"])
            assert models["agents"]["defaults"]["workspace"] == "/app"
            assert models["models"]["providers"]["nemo"]["timeoutSeconds"] == model_timeout_seconds
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
            # Interrupted auxiliary calls need not persist a compaction event,
            # so a plain transcript cannot establish full usage coverage either.
            assert "auxiliary_model_usage_unavailable" in [gap["code"] for gap in observations["gaps"]]
            assert len(observations["records"][0]["model_calls"]) == 2
    assert all(record.state is None for record in agent._session_records.values())
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
        if option == "worker":
            with pytest.raises(ValueError, match="num_workers=1"):
                client.post("/v1/agent_sessions", json=body)
        else:
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
        first = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        retry = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert first.status_code == retry.status_code == 200
        assert first.json() == retry.json()
        assert (
            client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "different"}).status_code == 409
        )
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200
    sandbox.launch.assert_awaited_once()


@pytest.mark.parametrize("reason", ["error", "aborted", "length"])
def test_failed_or_partial_openclaw_output_is_preserved(setup, reason):
    agent, sandbox = setup
    sandbox.events = events(stop_reason=reason)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        path = "/ng-rollout/openclaw-smoke-a2/v1/responses"
        result = client.post(path, json={"input": "task"})
        if reason == "length":
            assert result.status_code == 200
            assert result.json()["status"] == "incomplete"
            assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        else:
            assert result.status_code == 502
            assert "model" in result.json()["detail"]
            retry = client.post(path, json={"input": "task"})
            assert retry.status_code == 502
            assert retry.json() == result.json()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        invocation = closed.json()["agent_observations"]["records"][0]
        assert invocation["status"] == ("incomplete" if reason == "length" else "failed")
        assert invocation["conversation"][-1]["content"][0]["text"] == "Fixed"
    sandbox.launch.assert_awaited_once()


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
    sandbox.launch.assert_not_awaited()


def test_rejected_request_does_not_consume_activation(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
        path = "/ng-rollout/openclaw-smoke-a2/v1/responses"
        assert client.post(path, json={"input": "task", "temperature": 0.2}).status_code == 422
        sandbox.launch.assert_not_awaited()
        accepted = client.post(path, json={"input": "task"})
        assert accepted.status_code == 200, accepted.text
        assert client.post(path, json={"input": "task"}).json() == accepted.json()
    sandbox.launch.assert_awaited_once()


def test_request_allowlist_rejects_future_fields_but_accepts_defaults(setup):
    class ExtendedRequest(NeMoGymResponseCreateParamsNonStreaming):
        future_control: str | None = None

    agent, sandbox = setup
    assert agent._sandbox_input(ExtendedRequest(input="task", stream=False, background=False)) == ("task", "")
    with pytest.raises(HTTPException, match="future_control") as error:
        agent._sandbox_input(ExtendedRequest(input="task", future_control="unsupported"))
    assert error.value.status_code == 422
    with pytest.raises(HTTPException, match="metadata"):
        agent._sandbox_input(ExtendedRequest(input="task", metadata={"ignored": "value"}))
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize(
    "metadata,hint",
    [
        ({"extra_body": json.dumps({"seed": 1})}, "request field metadata"),
        ({"chat_template_kwargs": json.dumps({"enable_thinking": True})}, "Gym model server"),
    ],
)
def test_metadata_rejection_preserves_activation_and_configuration_hint(setup, metadata, hint):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        path = "/ng-rollout/openclaw-smoke-a2/v1/responses"
        rejected = client.post(path, json={"input": "task", "metadata": metadata})
        assert rejected.status_code == 422
        assert hint in rejected.json()["detail"]
        sandbox.launch.assert_not_awaited()
        client.post(path, json={"input": "task"}).raise_for_status()
        sandbox.launch.assert_awaited_once()
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).raise_for_status()


def test_no_session_keeps_existing_local_path(setup):
    agent, sandbox = setup
    with patch.object(agent, "_create_response", AsyncMock(side_effect=RuntimeError("legacy path reached"))) as legacy:
        with TestClient(agent.setup_webserver()) as client:
            with pytest.raises(RuntimeError, match="legacy path reached"):
                client.post("/v1/responses", json={"input": "task"})
        legacy.assert_awaited_once()
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("path", ["/v1/responses", "/ng-rollout/local-smoke/v1/responses"])
def test_default_config_unseeded_responses_run_local_cli(setup, path: str) -> None:
    agent, sandbox = setup
    config = OmegaConf.load(Path(__file__).parents[1] / "configs/openclaw_agent.yaml")
    config.policy_model_name = "local-model"
    agent.config = OpenClawAgentConfig.model_validate(
        OmegaConf.to_container(config.openclaw_agent.responses_api_agents.openclaw_agent, resolve=True)
        | {"host": "localhost", "port": 8001, "name": "openclaw"}
    )
    assert agent.config.resources_server is None
    assert agent.config.sandbox_provider is None
    message = NeMoGymResponseOutputMessage(
        id="msg-local",
        role="assistant",
        type="message",
        status="completed",
        content=[{"type": "output_text", "text": "local result", "annotations": []}],
    )
    with (
        patch.object(
            agent,
            "_run_openclaw",
            AsyncMock(
                return_value=(
                    [message],
                    {"input_tokens": 3, "output_tokens": 2},
                    "local-model",
                )
            ),
        ) as local,
        patch("responses_api_agents.openclaw_agent.app.create_provider") as provider,
        TestClient(agent.setup_webserver()) as client,
    ):
        response = client.post(path, json={"input": "task", "temperature": 0.7})
    assert response.status_code == 200, response.text
    assert response.json()["output"][0]["content"][0]["text"] == "local result"
    assert response.json()["usage"]["total_tokens"] == 5
    local.assert_awaited_once()
    assert local.call_args.args == ("task", "")
    assert local.call_args.kwargs["rollout_id"] == ("local-smoke" if "ng-rollout" in path else None)
    provider.assert_not_called()
    sandbox.launch.assert_not_awaited()


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
    sandbox.request_stop.assert_awaited_once()
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


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
    replay = await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert replay == original
    sandbox.launch.assert_awaited_once()
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


@pytest.mark.parametrize("stop_reason", ["stop", "aborted"])
def test_wall_limit_returns_gradable_partial_output(setup, stop_reason):
    agent, sandbox = setup
    sandbox.result.update(timed_out=True, return_code=-15)
    sandbox.events = events(stop_reason=stop_reason)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200
        assert response.json()["status"] == "incomplete"
        assert response.json()["output"][-1]["content"][0]["text"] == "Fixed"
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200


@pytest.mark.parametrize("stop_reason", ["stop", "length", "aborted"])
@pytest.mark.parametrize("failure", ["runner", "exit"])
def test_runtime_failures_cannot_become_successful_responses(setup, failure, stop_reason):
    agent, sandbox = setup
    sandbox.result.update(error="runner failed" if failure == "runner" else None, return_code=1)
    sandbox.events = events(stop_reason=stop_reason)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 502
        expected = "runner failed" if failure == "runner" else "aborted" if stop_reason == "aborted" else "exited 1"
        assert expected in response.json()["detail"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        invocation = closed.json()["agent_observations"]["records"][0]
        assert invocation["status"] == "failed"
        assert invocation["conversation"][-1]["content"][0]["text"] == "Fixed"


async def test_local_and_sandbox_prompts_include_the_same_instructions(setup):
    agent, _ = setup
    agent.config.system_prompt = "configured system"
    body = NeMoGymResponseCreateParamsNonStreaming(
        instructions="request instructions",
        input=[{"role": "system", "content": "input system"}, {"role": "user", "content": "task"}],
    )
    expected = ("task", "configured system\n\nrequest instructions\n\ninput system")
    assert agent._sandbox_input(body) == expected
    with patch.object(agent, "_run_openclaw", AsyncMock(return_value=([], {}, "test-model"))) as run:
        await agent._create_response(body)
    assert run.await_args.args == expected


@pytest.mark.parametrize("marker", [None, "expired"])
async def test_obsolete_sandbox_cookie_never_falls_back_to_host(setup, marker):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {"nemo_gym_openclaw_sandbox_session": marker}})
    with patch.object(agent, "_create_response", AsyncMock(side_effect=AssertionError("host fallback"))):
        with pytest.raises(HTTPException, match="Obsolete"):
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    sandbox.exec.assert_not_awaited()


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
    assert not any("rm -rf" in call.args[0] for call in sandbox.exec.await_args_list)


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
    sandbox.launch.assert_awaited_once()


def test_cleanup_receipt_is_required():
    with pytest.raises(ValueError):
        parse_cleanup_receipt({"return_code": 0, "error": None})


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
        client.cookies.clear()
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
    sandbox.disconnect.assert_awaited_once()


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
    with patch.object(agent, "_create_episode", AsyncMock(side_effect=AssertionError("host fallback"))):
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
        OpenClawAgentConfig(**(agent.config.model_dump() | {"session_close_retry_window_seconds": window}))


async def test_unknown_launch_outcome_fails_closed(setup):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "openclaw-smoke-a2"}})
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
        SimpleNamespace(return_code=0, error_type=None),
        SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="npm failed"),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="npm failed"):
        await agent.seed_agent_session(request, seed())
    assert all(record.state is None for record in agent._session_records.values())
    assert not request.session
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_cancelled_install_never_publishes_session_or_launches_openclaw(setup):
    agent, sandbox = setup
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        asyncio.CancelledError(),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    request = Request({"type": "http", "session": {}})
    with pytest.raises(asyncio.CancelledError):
        await agent.seed_agent_session(request, seed())
    assert all(record.state is None for record in agent._session_records.values())
    assert not request.session
    sandbox.launch.assert_not_awaited()
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
    sandbox.launch.assert_not_awaited()


async def test_cleanup_command_provider_error_blocks_close_and_can_retry(setup):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type="timeout", stderr="cleanup timed out"),
        SimpleNamespace(return_code=0, error_type=None, stderr=""),
    ]
    with pytest.raises(RuntimeError, match="Could not remove OpenClaw session files"):
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
    sandbox.launch.assert_awaited_once()


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
    sandbox.launch.assert_not_awaited()


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


async def test_usage_sums_cache_writes_and_failed_calls(setup):
    agent, sandbox = setup
    transcript = [json.loads(line) for line in events(stop_reason="error").splitlines()]
    transcript[0]["message"]["usage"]["cacheWrite"] = 4
    sandbox.events = "\n".join(json.dumps(event) for event in transcript)
    request, session_id, task = await activate(agent, sandbox)
    with pytest.raises(HTTPException, match="model error"):
        await task
    state = agent._session_records[session_id].state
    response = await agent._collect_sandbox_response(
        state, NeMoGymResponseCreateParamsNonStreaming(input="task"), prompt="task", system=""
    )
    assert response.status == "failed"
    assert response.usage.input_tokens == 21
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 26
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


@pytest.mark.parametrize(
    "first_cache,second_cache,expected_cache,expected_input",
    [
        (2, {}, None, 17),
        (2, {"cacheRead": None}, None, 17),
        (2, {"cacheRead": -1}, None, 17),
        (2, {"cacheRead": True}, None, 17),
        (2, {"cacheRead": "3"}, None, 17),
        (2, {"cacheRead": 3.0}, None, 17),
        (None, {"cacheRead": 2}, None, 17),
        (0, {"cacheRead": 0}, None, 15),
        (2, {"cacheRead": 0}, None, 17),
        (0, {"cacheRead": 2}, None, 17),
        (2, {"cacheRead": 3}, 5, 20),
    ],
)
def test_sandbox_usage_details_preserve_positive_counts_and_treat_defaulted_zero_as_unknown(
    setup, first_cache, second_cache, expected_cache, expected_input
):
    agent, sandbox = setup
    transcript = [json.loads(line) for line in events().splitlines()]
    transcript[0]["message"]["usage"]["cacheRead"] = first_cache
    transcript[2]["message"]["usage"].update(second_cache)
    sandbox.events = "\n".join(json.dumps(event) for event in transcript)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        response = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        response.raise_for_status()
        body = response.json()
        assert body["status"] == "completed"
        assert body["output"][0]["type"] == "reasoning"
        assert body["usage"] == {
            "input_tokens": expected_input,
            "output_tokens": 5,
            "total_tokens": expected_input + 5,
            "input_tokens_details": {"cached_tokens": expected_cache},
            "output_tokens_details": {"reasoning_tokens": None},
        }
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        closed.raise_for_status()
    gaps = {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
    assert ("cached_token_usage_unavailable" in gaps) == (expected_cache is None)
    assert "reasoning_token_usage_unavailable" in gaps


@pytest.mark.parametrize("second_usage", [None, "unavailable"])
def test_missing_call_usage_keeps_sandbox_cache_aggregate_unknown(setup, second_usage):
    agent, sandbox = setup
    transcript = [json.loads(line) for line in events().splitlines()]
    transcript[2]["message"]["usage"] = second_usage
    sandbox.events = "\n".join(json.dumps(event) for event in transcript)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        response = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        response.raise_for_status()
        assert response.json()["usage"] == {
            "input_tokens": 12,
            "output_tokens": 3,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": None},
            "output_tokens_details": {"reasoning_tokens": None},
        }
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        closed.raise_for_status()
    gaps = {gap["code"] for gap in closed.json()["agent_observations"]["gaps"]}
    assert {"model_call_usage_unavailable", "cached_token_usage_unavailable"} <= gaps


def test_rewritten_transcript_and_cli_mirror_count_each_model_call_once(setup):
    agent, sandbox = setup
    # Scalar usage from the real Ansible run: eight main calls, three rewritten
    # copies, then a CLI mirror of all main-call usage. Auxiliary compaction
    # calls (2505 input / 839 output) are not represented in this transcript.
    counters = [(4817, 27), (8072, 26), (10442, 27), (13697, 27), (9838, 26), (12208, 27), (15463, 27), (18718, 27)]
    transcript = [
        {
            "type": "message",
            "id": f"entry-{index}",
            "parentId": f"entry-{index - 1}" if index else None,
            "message": {
                "role": "assistant",
                "api": "openai-completions",
                "responseId": f"call-{index}",
                "content": [
                    {"type": "toolCall", "id": f"tool-{index}", "name": "read", "arguments": {"path": "/app/a"}}
                ],
                "usage": {"input": input_tokens, "output": output_tokens, "cacheRead": 0, "cacheWrite": 0},
                "stopReason": "toolUse",
            },
        }
        for index, (input_tokens, output_tokens) in enumerate(counters)
    ]
    for index in (1, 2, 3):
        rewritten = deepcopy(transcript[index])
        rewritten["id"] = f"rewritten-{index}"
        transcript.append(rewritten)
    transcript.extend(
        [
            {
                "type": "compaction",
                "tokensBefore": 22737,
                "summary": "Partial task history",
                "firstKeptEntryId": "entry-0",
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "api": "openai-completions",
                    "content": [],
                    "stopReason": "error",
                    "errorMessage": "Context overflow recovery exhausted",
                    "usage": {"input": 0, "output": 0},
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "api": "cli",
                    "content": [{"type": "text", "text": "Context overflow recovery exhausted"}],
                    "stopReason": "stop",
                    "usage": {"input": 93255, "output": 214, "cacheRead": 0, "cacheWrite": 0},
                },
            },
        ]
    )
    sandbox.events = "\n".join(json.dumps(event) for event in transcript)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 502, response.text
        assert response.json()["detail"] == "Context overflow recovery exhausted"
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    observations = closed.json()["agent_observations"]
    invocation = observations["records"][0]
    assert invocation["status"] == "failed"
    assert len(invocation["model_calls"]) == 8
    assert sum(item["type"] == "function_call" for item in invocation["conversation"]) == 11
    assert any(record["kind"] == "context_compaction" for record in observations["records"])
    gaps = {gap["code"] for gap in observations["gaps"]}
    assert "auxiliary_model_usage_unavailable" in gaps
    assert "agent_conversation_branching_unavailable" in gaps
    assert "model_call_usage_identity_unavailable" in gaps


@pytest.mark.parametrize("conflict", ["content", "usage"])
def test_conflicting_response_id_usage_is_excluded_and_reported(setup, conflict):
    agent, sandbox = setup
    transcript = [json.loads(line) for line in events().splitlines()]
    collision = deepcopy(transcript[0])
    if conflict == "content":
        collision["message"]["content"][0]["thinking"] = "Different call, same ID and usage"
    else:
        collision["message"]["usage"]["input"] += 1
    transcript.insert(1, collision)
    sandbox.events = "\n".join(json.dumps(event) for event in transcript)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["usage"]["input_tokens"] == 5
        assert body["usage"]["output_tokens"] == 2
        assert sum(item["type"] == "function_call" for item in body["output"]) == 2
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    gaps = closed.json()["agent_observations"]["gaps"]
    assert any(gap["code"] == "model_call_usage_identity_ambiguous" and "call-1" in gap["detail"] for gap in gaps)


@pytest.mark.parametrize("missing_id", [None, "", " ", 42])
def test_missing_response_ids_do_not_deduplicate_distinct_calls(missing_id):
    message = {"role": "assistant", "content": "Repeated output", "usage": {"input": 7, "output": 3}}
    if missing_id is not None:
        message["responseId"] = missing_id
    messages = [message, deepcopy(message)]
    before = deepcopy(messages)
    selected, gaps = _unique_usage_messages(messages)
    assert selected == messages
    assert len(selected) == 2
    assert sum(message["usage"]["input"] for message in selected) == 14
    assert len(gaps) == 2
    assert {gap.code for gap in gaps} == {"model_call_usage_identity_unavailable"}
    assert messages == before


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
async def test_invalid_or_expired_sandbox_cookie_never_uses_local_execution(setup, marker):
    agent, sandbox = setup
    request = Request(
        {
            "type": "http",
            "session": {"agent_session_id": marker},
            "path_params": {"rollout_id": "openclaw-smoke-a2"},
        }
    )
    with patch.object(agent, "_create_response", AsyncMock(side_effect=AssertionError("host fallback"))):
        with pytest.raises(HTTPException) as error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        assert error.value.status_code == 409
    sandbox.launch.assert_not_awaited()


def test_sandbox_close_cookie_blocks_legacy_run_with_configured_resources(setup):
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
    sandbox.launch.assert_not_awaited()


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
    state = agent._session_records[session_id].state
    state.session.artifacts["stdout.log"] = json.dumps({"meta": {"agentMeta": {"usage": {"input": "not-a-number"}}}})
    response = await agent._collect_sandbox_response(
        state,
        NeMoGymResponseCreateParamsNonStreaming(input="task"),
        prompt="task",
        system="",
    )
    assert response.status == "completed"
    assert response.output[-1].content[0].text == "Fixed"
    assert response.usage.total_tokens == 22
    assert "agent_stdout_unparseable" in {gap.code for gap in state.observations.gaps}


@pytest.mark.parametrize(
    "cache,expected_cache",
    [
        ({}, None),
        ({"cacheRead": None}, None),
        ({"cacheRead": -1}, None),
        ({"cacheRead": True}, None),
        ({"cacheRead": "5"}, None),
        ({"cacheRead": "bad"}, None),
        ({"cacheRead": 3.0}, None),
        ({"cacheRead": 0}, None),
        ({"cacheRead": 5}, 5),
    ],
)
async def test_sandbox_envelope_fallback_preserves_known_and_unknown_cache_details(setup, cache, expected_cache):
    agent, sandbox = setup
    request, session_id, task = await activate(agent, sandbox)
    await task
    state = agent._session_records[session_id].state
    state.session.artifacts = {
        "stdout.log": json.dumps(
            {
                "payloads": [{"text": "Retained final output."}],
                "meta": {"agentMeta": {"usage": {"input": 10, "output": 3, **cache}}},
            }
        )
    }
    response = await agent._collect_sandbox_response(
        state,
        NeMoGymResponseCreateParamsNonStreaming(input="task"),
        prompt="task",
        system="",
    )
    assert response.output[0].content[0].text == "Retained final output."
    assert response.usage.input_tokens == 10 + (expected_cache or 0)
    assert response.usage.output_tokens == 3
    assert response.usage.input_tokens_details.cached_tokens == expected_cache
    assert response.usage.output_tokens_details.reasoning_tokens is None
    gaps = {gap.code for gap in state.observations.gaps}
    assert ("cached_token_usage_unavailable" in gaps) == (expected_cache is None)
    assert "model_call_usage_unavailable" in gaps
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(session_id)))


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
    sandbox.exec.return_value = SimpleNamespace(
        return_code=1, error_type=None, stdout="", stderr="task workdir overlaps adapter-owned storage"
    )
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError, match="overlaps adapter-owned storage"):
        await agent.seed_agent_session(request, seed())
    assert not sandbox.files
    assert not request.session
    sandbox.exec.assert_awaited_once()
    sandbox.disconnect.assert_awaited_once()


async def test_caller_id_seed_retries_share_one_session_without_cookies(setup):
    agent, sandbox = setup
    body = seed(session_id="caller-assigned-id")
    requests = [Request({"type": "http", "session": {}}) for _ in range(2)]
    first, second = await asyncio.gather(*(agent.seed_agent_session(request, body) for request in requests))
    assert first.agent_session_id == second.agent_session_id == body.agent_session_id
    assert len(agent._session_records) == 1
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
    with pytest.raises(HTTPException, match="another seed"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), changed)
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


async def test_close_before_seed_prevents_late_creation(setup):
    agent, sandbox = setup
    body = seed()
    request = Request({"type": "http", "session": {}})
    close = AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)
    first = await agent.close_agent_session(request, close)
    assert await agent.close_agent_session(Request({"type": "http", "session": {}}), close) == first
    with pytest.raises(HTTPException, match="already closed"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    sandbox.exec.assert_not_awaited()


async def test_close_serializes_with_inflight_seed(setup):
    agent, sandbox = setup
    body = seed()
    entered, release = asyncio.Event(), asyncio.Event()
    original = agent._seed_agent_session_state

    async def blocked_initialize(*args):
        entered.set()
        await release.wait()
        return await original(*args)

    with patch.object(agent, "_seed_agent_session_state", side_effect=blocked_initialize):
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
    assert all(record.state is None for record in agent._session_records.values())
    sandbox.disconnect.assert_awaited_once()


async def test_caller_id_never_controls_filesystem_path(setup):
    agent, _ = setup
    body = seed(session_id="../../task-repository\nunsafe")
    request = Request({"type": "http", "session": {}})
    response = await agent.seed_agent_session(request, body)
    assert response.agent_session_id == body.agent_session_id
    directory = agent._session_records[body.agent_session_id].state.session.session_dir
    assert Path(directory).parent == Path("/tmp/nemo-gym-openclaw-sessions")
    assert len(Path(directory).name) == 32
    await agent.close_agent_session(request, AgentCloseSessionRequest(**close_body(body.agent_session_id)))


@pytest.mark.parametrize(
    "field", ["timeout", "model_timeout_seconds", "sandbox_install_timeout_seconds", "session_close_timeout_seconds"]
)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_deadlines_are_positive_and_finite(setup, field, value):
    agent, _ = setup
    with pytest.raises(ValidationError):
        OpenClawAgentConfig(**(agent.config.model_dump() | {field: value}))


@pytest.mark.parametrize("failed", [False, True])
async def test_independent_configs_route_prepared_rows_through_environment_run(setup, monkeypatch, failed):
    from http.cookies import SimpleCookie

    import orjson

    import nemo_gym.rollout_collection as collection
    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
    from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    agent, sandbox = setup
    global_config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "resources_servers/swebench_pro/configs/swebench_pro.yaml",
                        "responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml",
                        "environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml",
                    ],
                    "single_agent_turn_legacy": {
                        "environment_servers": {
                            "single_agent_turn_legacy": {
                                "resources_server": {
                                    "type": "resources_servers",
                                    "name": "swebench_pro_resources_server",
                                },
                                "agent_server": {"type": "responses_api_agents", "name": "openclaw_agent"},
                            }
                        }
                    },
                    "policy_model_name": "test-model",
                    "policy_model": {"responses_api_models": {"openai_model": {"entrypoint": "app.py"}}},
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    # The benchmark's legacy allowlist predates this sandbox harness. Keep its
    # definition untouched and explicitly opt in, as the CLI smoke does.
    monkeypatch.setenv("NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING", "1")
    if failed:
        sandbox.events = events(stop_reason="error")
    env_name = "single_agent_turn_legacy"
    env_config = global_config[env_name].environment_servers.single_agent_turn_legacy
    agent_name = env_config.agent_server.name
    resources_name = env_config.resources_server.name
    agent.config = OpenClawAgentConfig(
        name=agent_name,
        **OmegaConf.to_container(global_config[agent_name].responses_api_agents.openclaw_agent, resolve=True),
    )
    assert agent.config.resources_server is None
    assert not agent.config.openclaw_config
    agent.server_client.global_config_dict = global_config
    agent.server_client._resolve_base_url.return_value = "http://resources.example:8000"
    environment = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(name=env_name, **OmegaConf.to_container(env_config)),
        server_client=agent.server_client,
    )
    calls = []
    resources_id = None

    class Reply:
        def __init__(self, payload, *, status=200, cookies=None):
            self.status = status
            self.ok = status < 400
            self.cookies = SimpleCookie(cookies or {})
            self.payload = payload

        async def read(self):
            return orjson.dumps(self.payload)

        def raise_for_status(self):
            assert self.ok, self.payload

    async def post(*, server_name, url_path, json, cookies=None, **kwargs):
        nonlocal resources_id
        payload = json.model_dump(mode="json") if hasattr(json, "model_dump") else json
        calls.append((server_name, url_path, payload))
        if server_name == resources_name:
            if url_path == "/seed_session":
                resources_id = payload["resources_session_id"]
                assert payload["task_data"]["instance_id"] == "smoke-instance"
                return Reply(
                    {"resources_session_id": resources_id, "sandbox_access": seed().sandbox_access.model_dump()},
                    cookies={"resources": resources_id},
                )
            assert cookies == {"resources": resources_id}
            if url_path == "/verify":
                sandbox.disconnect.assert_awaited_once()
                assert payload["response"]["status"] == "completed"
                return Reply({"reward": 1.0, **payload})
            assert url_path == "/close_session"
            assert payload["resources_session_id"] == resources_id
            return Reply({"resources_session_id": resources_id})
        assert server_name in {env_name, agent_name}
        assert not (server_name == agent_name and url_path == "/run")
        client = environment_http if server_name == env_name else agent_http
        client.cookies.clear()
        client.cookies.update(cookies or {})
        response = await asyncio.to_thread(client.post, url_path, json=payload)
        return Reply(response.json(), status=response.status_code, cookies=dict(response.cookies))

    agent.server_client.post = AsyncMock(side_effect=post)
    monkeypatch.setattr(collection, "setup_server_client_utils", lambda *args, **kwargs: agent.server_client)
    prepared = {
        "responses_create_params": {"input": "Fix the code"},
        "instance_id": "smoke-instance",
        "task_source": resources_name,
    }
    config = collection.RolloutCollectionConfig(
        input_jsonl_fpath="unused.jsonl",
        output_jsonl_fpath="unused-rollouts.jsonl",
        environment_routing_mode="legacy",
        environment_server_name=env_name,
        num_repeats=1,
    )
    rows = collection.RolloutCollectionHelper._preprocess_raw_rows(
        [(0, orjson.dumps(prepared).decode(), prepared)], config
    )
    with (
        TestClient(agent.setup_webserver()) as agent_http,
        TestClient(environment.setup_webserver()) as environment_http,
    ):
        _, result = await next(collection.RolloutCollectionHelper().run_examples(rows))
    if failed:
        assert result["_ng_failure_class"] == "environment_server_failed", result
        assert result["_ng_failure_stage"] == "agent"
        assert "reward" not in result
    else:
        assert "verification" not in result
        assert result.get("reward") == 1.0, result
        assert result["ng_agent_observations"]["source"] == "openclaw"
    expected = [
        (env_name, "/run"),
        (resources_name, "/seed_session"),
        (agent_name, "/v1/agent_sessions"),
        (agent_name, "/ng-rollout/0-0/v1/responses"),
        (agent_name, "/v1/agent_sessions/close"),
    ]
    if not failed:
        expected.append((resources_name, "/verify"))
    expected.append((resources_name, "/close_session"))
    assert [(server_name, path) for server_name, path, _ in calls] == expected
    assert calls[2][2]["agent_session_id"] == calls[4][2]["agent_session_id"]
    assert calls[0][2]["instance_id"] == prepared["instance_id"]
    assert calls[0][2]["responses_create_params"]["input"] == prepared["responses_create_params"]["input"]
    assert all(record.state is None for record in agent._session_records.values())
    sandbox.stop.assert_not_awaited()


async def test_failed_seed_with_confirmed_cleanup_accepts_cookie_less_close(setup):
    agent, sandbox = setup
    body = seed()
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="npm unavailable"),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    with pytest.raises(RuntimeError, match="npm unavailable"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    close = AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)
    result = await agent.close_agent_session(Request({"type": "http", "session": {}}), close)
    assert result.agent_session_id == body.agent_session_id
    sandbox.disconnect.assert_awaited_once()
    assert all(record.state is None for record in agent._session_records.values())


async def test_failed_seed_cleanup_retains_state_until_explicit_retry(setup):
    agent, sandbox = setup
    body = seed()
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="npm unavailable"),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    sandbox.disconnect.side_effect = RuntimeError("disconnect unavailable")
    with pytest.raises(RuntimeError, match="npm unavailable"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    assert agent._session_records[body.agent_session_id].state.session.closing
    assert body.agent_session_id not in agent._closed_session_records
    with pytest.raises(HTTPException, match="closing"):
        await agent.seed_agent_session(Request({"type": "http", "session": {}}), body)
    sandbox.exec.side_effect = None
    sandbox.disconnect.side_effect = None
    close = AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)
    await agent.close_agent_session(Request({"type": "http", "session": {}}), close)
    assert all(record.state is None for record in agent._session_records.values())
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("marker", [None, [], {}, 7])
async def test_seed_rejects_malformed_sandbox_cookie_before_connecting(setup, marker):
    agent, sandbox = setup
    request = Request({"type": "http", "session": {"agent_session_id": marker}})
    with pytest.raises(HTTPException, match="Invalid agent session marker"):
        await agent.seed_agent_session(request, seed())
    sandbox.exec.assert_not_awaited()


async def test_stale_cookie_cannot_recreate_session_or_close_receipt_after_full_retention(setup, monkeypatch):
    agent, sandbox = setup
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 5
    request = Request({"type": "http", "session": {}})
    body = seed()
    await agent.seed_agent_session(request, body)
    close = AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)
    await agent.close_agent_session(request, close)
    clock[0] = 111.0
    with pytest.raises(HTTPException, match="expired"):
        await agent.close_agent_session(request, close)
    with pytest.raises(HTTPException, match="expired"):
        await agent.seed_agent_session(request, body)
    assert not agent._closed_session_records
    assert all(record.state is None for record in agent._session_records.values())
    assert not agent._session_records
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("owned", [False, True])
def test_sandbox_source_controls_ownership_and_sandbox_routing(setup, owned):
    agent, sandbox = setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/agent-workspace"}
    sandbox.expected_workdir = "/agent-workspace" if owned else "/app"
    body = seed()
    if owned:
        body.sandbox_access = None
    sandbox.start = AsyncMock()
    module = "responses_api_agents.openclaw_agent.app"
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
        patch("responses_api_agents.openclaw_agent.app.AsyncSandbox", return_value=sandbox),
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
                stage == "install" and command.startswith("bash ") and "install_openclaw_runtime.sh" in command
            ):
                raise original
            return await execute(command, **kwargs)

        sandbox.exec.side_effect = fail_at_stage
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    with (
        patch("responses_api_agents.openclaw_agent.app.AsyncSandbox", return_value=sandbox),
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
    module = "responses_api_agents.openclaw_agent.app"
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
        patch("responses_api_agents.openclaw_agent.app.AsyncSandbox") as factory,
        TestClient(agent.setup_webserver()) as client,
    ):
        assert client.post("/v1/agent_sessions", json=body.model_dump(mode="json")).status_code == 422
        factory.assert_not_called()
        sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("invalid_runtime", [False, True])
async def test_close_retains_capture_after_provider_release(setup, tmp_path, owned, invalid_runtime):
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

    request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "openclaw-smoke-a2"}})
    with patch("responses_api_agents.openclaw_agent.app.AsyncSandbox", return_value=sandbox) as factory:
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
        response = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
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
