# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from nemo_gym.agents import AGENT_SESSION_COOKIE_KEY, AgentSeedRejectedError, AgentSeedSessionRequest, SandboxSession
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.trajectory import AgentObservationBundle
from responses_api_agents.nemo_fabric_agent import app
from responses_api_agents.nemo_fabric_agent.sandbox import FabricSandboxSession
from responses_api_agents.nemo_fabric_agent.sandbox_runtime import runtime_archive, runtime_requirements


def seed(*, borrowed=True, **updates):
    return AgentSeedSessionRequest.model_validate(
        {
            "agent_session_id": "session",
            "episode_id": {"rollout_id": "rollout"},
            "task_id": {"taskset": "fixture", "task_id": "task"},
            "sandbox_access": {
                "connection": {"provider_config_ref": "sandbox", "descriptor": {"id": "borrowed"}},
                "workdir": "/app",
            }
            if borrowed
            else None,
            **updates,
        }
    )


def agent(**updates):
    return app.NeMoFabricAgent(
        config=app.NeMoFabricAgentConfig(
            name="fabric",
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            model="gym-policy-model",
            **{"adapter_id": "nvidia.fabric.mini-swe-agent", **updates},
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={"sandbox": {"docker": {}}}),
    )


def request(*, rollout="rollout"):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "session": {AGENT_SESSION_COOKIE_KEY: "session"},
            "path_params": {"rollout_id": rollout},
        }
    )


def response(instance, body):
    return instance._response_from_result(body, {"status": "succeeded", "output": {"response": "answer"}}, [])[0]


@pytest.mark.parametrize("borrowed", [True, False])
async def test_seed_keeps_sandbox_ownership_and_does_not_run_harness(monkeypatch, model_url, borrowed):
    remote = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None)),
        upload=AsyncMock(),
        start=AsyncMock(),
    )
    constructor = MagicMock(return_value=remote)
    constructor.connect = AsyncMock(return_value=remote)
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    monkeypatch.setattr(app, "create_provider", MagicMock())
    install = AsyncMock(return_value="/opt/private/python")
    monkeypatch.setattr(app, "prepare_fabric_runtime", install)
    state = await agent(sandbox_provider="sandbox")._seed_agent_session_state(seed(borrowed=borrowed))
    assert state.session.owns_sandbox is not borrowed
    assert state.session.workdir == "/app"
    assert state.python == "/opt/private/python"
    assert state.task is None
    assert remote.start.await_count == (0 if borrowed else 1)
    if borrowed:
        assert constructor.connect.await_args.args == ({"id": "borrowed"},)
        assert "mkdir -p -- /app" not in remote.exec.await_args.args[0]


async def test_required_tools_are_refused(monkeypatch):
    instance = agent()
    monkeypatch.setattr(
        type(instance), "effective_tool_accesses", lambda *_: [SimpleNamespace(required=True, name="required")]
    )
    with pytest.raises(AgentSeedRejectedError, match="required remote"):
        await instance._seed_agent_session_state(seed())


async def test_retries_join_one_activation_and_http_cancellation_does_not_kill_it(monkeypatch):
    instance = agent()
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    started, finish = asyncio.Event(), asyncio.Event()
    state = FabricSandboxSession(request=seed(), session=SimpleNamespace(closing=False))
    monkeypatch.setattr(type(instance), "_require_agent_session", lambda *_: state)
    calls = []

    async def run(*_):
        calls.append(True)
        started.set()
        await finish.wait()
        return response(instance, body)

    monkeypatch.setattr(type(instance), "_sandbox_response", run)
    first = asyncio.create_task(instance.responses(request(), body))
    await asyncio.wait_for(started.wait(), timeout=2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not state.task.cancelled()
    with pytest.raises(HTTPException) as conflict:
        await instance.responses(request(), NeMoGymResponseCreateParamsNonStreaming(input="different"))
    assert conflict.value.status_code == 409
    with pytest.raises(HTTPException) as mismatch:
        await instance.responses(request(rollout="other"), body)
    assert mismatch.value.status_code == 409
    finish.set()
    replay = await instance.responses(request(), body)
    assert len(calls) == 1
    assert replay.output[0].content[0].text == "answer"


async def test_close_joins_parsing_and_returns_cleanup_evidence():
    instance = agent()
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    closed = asyncio.Event()

    async def close(*, timeout):
        closed.set()

    bundle = AgentObservationBundle(source="nemo_fabric")

    async def parse():
        await closed.wait()
        state.observations = bundle
        return response(instance, body)

    state = FabricSandboxSession(
        request=seed(), session=SimpleNamespace(close=close, artifacts={}, cleanup_confirmed=True)
    )
    state.task = asyncio.create_task(parse())
    result = await instance._close_agent_session_state(state)
    assert result.cleanup_confirmed is True
    assert result.agent_observations == bundle
    assert state.task.done()


async def test_stage_uses_private_python_and_preserves_task_environment(monkeypatch):
    sandbox = SimpleNamespace(upload=AsyncMock())
    binding = seed()
    binding.sandbox_access.env = {"TASK_FLAG": "hello world"}
    state = FabricSandboxSession(
        request=binding,
        python="/opt/private/python",
        session=SandboxSession(sandbox=sandbox, workdir="/app", session_dir="/tmp/session", harness="Fabric"),
    )
    uploads = {}

    async def upload(_sandbox, *, path, text):
        uploads[path] = text

    monkeypatch.setattr("responses_api_agents.nemo_fabric_agent.sandbox.upload_text", upload)
    payload = {"input": "hello", "workdir": "/app"}
    command = await state.stage_activation(payload)
    assert command.stdlib_runner is False
    assert command.argv == ["bash", "/tmp/session/harness_runner.sh", "/tmp/session/params.sh"]
    assert json.loads(uploads["/tmp/session/input.json"]) == payload
    params = uploads["/tmp/session/params.sh"]
    assert "command=(/opt/private/python -I -B /tmp/session/fabric_runner /tmp/session)\n" in params
    assert "stdin=/dev/null" in params and "stdout=/tmp/session/stdout.log" in params
    assert "TASK_FLAG=hello world" in params
    assert "HOME=" not in params and "PATH=" not in params and "VIRTUAL_ENV=" not in params


def test_runtime_fingerprint_varies_with_adapter_and_relay():
    profiles = [
        runtime_requirements(name, relay=relay)
        for name, relay in [
            ("nvidia.fabric.mini-swe-agent", False),
            ("nvidia.fabric.claude", False),
            ("nvidia.fabric.claude", True),
        ]
    ]
    assert len({runtime_archive(profile)[1] for profile in profiles}) == 3
    assert all("==0.5.0" in profile[0] for profile in profiles)
    assert (
        runtime_requirements("nvidia.fabric.langchain.deepagents", relay=False)[0] == "nemo-fabric[deepagents]==0.5.0"
    )
    with pytest.raises(ValueError, match="No sandbox runtime profile"):
        runtime_requirements("nvidia.fabric.unknown", relay=False)


async def test_activation_preserves_model_capture_route_and_reports_unknown_usage(monkeypatch):
    instance = agent()
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    captured = {}
    native = {"status": "succeeded", "output": {"response": "answer", "usage": {"api_calls": 5}}}

    async def execute(*, stage_activation, collect, timeout, close_timeout):
        await stage_activation()
        assert timeout == 75
        return native

    binding = seed(agent_timeout_sec=75)
    state = FabricSandboxSession(
        request=binding,
        session=SimpleNamespace(
            workdir="/app",
            session_dir="/tmp/fabric",
            execute=execute,
            cleanup={"timed_out": False, "return_code": 0},
            closing=False,
        ),
        observations=AgentObservationBundle(source="nemo_fabric"),
    )

    async def stage(_self, payload):
        captured.update(payload)

    monkeypatch.setattr(FabricSandboxSession, "stage_activation", stage)
    monkeypatch.setattr(
        type(instance), "resolve_model_base_url", lambda _self, name, rollout: f"http://model/ng-rollout/{rollout}/v1"
    )
    result = await instance._sandbox_response(state, body)
    assert captured["config"]["models"]["default"]["base_url"] == "http://model/ng-rollout/rollout/v1"
    assert captured["config"]["environment"]["workspace"] == "/app"
    assert result.status == "completed"
    assert result.usage is None
    assert any(gap.code == "usage_unavailable" for gap in state.observations.gaps)


@pytest.mark.parametrize("confirmed", [False, True])
async def test_only_confirmed_timeout_can_return_an_incomplete_response(monkeypatch, confirmed):
    instance = agent()
    state = FabricSandboxSession(
        request=seed(),
        session=SimpleNamespace(
            workdir="/app",
            session_dir="/tmp/fabric",
            execute=AsyncMock(side_effect=RuntimeError("no result")),
            cleanup={"timed_out": True, "return_code": 143},
            cleanup_confirmed=confirmed,
            artifacts=None,
            closing=False,
            expired=False,
        ),
        observations=AgentObservationBundle(source="nemo_fabric"),
    )
    monkeypatch.setattr(type(instance), "resolve_model_base_url", lambda *_: "http://model/v1")
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    if confirmed:
        result = await instance._sandbox_response(state, body)
        assert result.status == "incomplete" and result.usage is None
        assert state.observations.records[0].outcome == "timeout"
    else:
        with pytest.raises(RuntimeError, match="no result"):
            await instance._sandbox_response(state, body)


async def test_unbound_compatibility_run_refuses_before_seeding_resources():
    instance = agent()
    with pytest.raises(HTTPException) as error:
        await instance.run(request(), app.NeMoFabricAgentRunRequest(responses_create_params={"input": "hello"}))
    assert error.value.status_code == 422
    instance.server_client.post.assert_not_called()
