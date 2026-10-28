# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox mode contract tests at the HTTP and lifecycle boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest, AgentSessionSetupError
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from responses_api_agents.langchain_deepagents_agent import sandbox_agent
from responses_api_agents.langchain_deepagents_agent.sandbox import DeepAgentsSandboxSession
from responses_api_agents.langchain_deepagents_agent.sandbox_agent import SandboxedDeepAgent, SandboxedDeepAgentConfig


def seed(**kwargs):
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="probe", task_id="task"),
        **kwargs,
    )


def access(**kwargs):
    return SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"id": "borrowed"}),
        workdir=kwargs.pop("workdir", "/task"),
        **kwargs,
    )


def agent(**kwargs):
    config = SandboxedDeepAgentConfig(
        host="127.0.0.1",
        port=12345,
        name="agent",
        entrypoint="sandbox_agent.py",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        **kwargs,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return SandboxedDeepAgent(config=config, server_client=client)


def state(body=None):
    session = SimpleNamespace(
        workdir="/task",
        stop_request_path="/private/stop",
        artifacts=None,
        close=AsyncMock(),
        owns_sandbox=False,
        cleanup={"cleanup_confirmed": True, "return_code": 0, "timed_out": False, "error": None},
    )
    return DeepAgentsSandboxSession(
        request=body or seed(sandbox_access=access()), session=session, sandbox_id="borrowed"
    )


def output(usage=None):
    return {
        "status": "completed",
        "error": None,
        "error_status": None,
        "stop_reason": None,
        "usages": [usage],
        "gaps": [],
        "tools": [],
        "turns": [],
        "invocations": [
            {
                "kind": "agent_invocation",
                "invocation_id": "root",
                "status": "completed",
                "model_calls": [],
                "conversation": [],
            }
        ],
        "output_items": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }
        ],
    }


@pytest.mark.parametrize(
    "body,reason",
    [
        (seed(), "requires sandbox_access"),
        (seed(sandbox_access=access(workdir="relative")), "absolute workdir"),
        (seed(sandbox_access=access(user="user")), "user override"),
        (
            seed(
                sandbox_access=access(),
                tool_accesses=[DirectHTTPToolAccess(name="resource", required=True, base_url="http://resource")],
            ),
            "required tool grants",
        ),
        (
            seed(
                sandbox_access=access(),
                tool_accesses=[
                    MCPToolAccess(
                        name="resource",
                        required=True,
                        connection=MCPStreamableHTTPConnection(url="http://resource/mcp"),
                    )
                ],
            ),
            "required tool grants",
        ),
    ],
)
def test_unsupported_seed_is_final_422_without_allocating_sandbox(body, reason, monkeypatch):
    create = MagicMock()
    monkeypatch.setattr(sandbox_agent, "create_provider", create)
    assert reason in check_agent_seed_rejected(agent().setup_webserver(), body)
    create.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_activation_joins_and_disconnect_does_not_cancel(tmp_path, monkeypatch):
    instance, current = agent(), state()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def activate(self, body, retained):
        calls.append(body)
        entered.set()
        await release.wait()
        from nemo_gym.openai_utils import NeMoGymResponse

        return NeMoGymResponse(
            id="response",
            created_at=0,
            model="model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tools=[],
            tool_choice="auto",
        )

    monkeypatch.setattr(SandboxedDeepAgent, "_require_agent_session", lambda self, key: current)
    monkeypatch.setattr(SandboxedDeepAgent, "_agent_session_id_from_request", lambda self, request: "session")
    monkeypatch.setattr(SandboxedDeepAgent, "_activate", activate)
    request = SimpleNamespace(path_params={"rollout_id": "rollout"})
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    first = asyncio.create_task(instance.responses(request, body))
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(instance.responses(request, body))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not current.task.done()
    with pytest.raises(HTTPException) as changed:
        await instance.responses(request, body.model_copy(update={"input": "changed"}))
    assert changed.value.status_code == 409
    release.set()
    assert (await asyncio.wait_for(second, 2)).id == "response"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_activation_forwards_controls_and_unknown_usage(tmp_path, monkeypatch):
    instance, current = (
        agent(),
        state(seed(sandbox_access=access(env={"TASK_VALUE": "preserved"}), agent_timeout_sec=60)),
    )
    captured = []

    async def execute(self, payload, **kwargs):
        captured.append((payload, kwargs))
        self.session.artifacts = output()
        return self.session.artifacts

    monkeypatch.setattr(DeepAgentsSandboxSession, "execute", execute)
    monkeypatch.setattr(
        SandboxedDeepAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://model/ng-rollout/rollout/v1"
    )
    response = await instance._activate(
        NeMoGymResponseCreateParamsNonStreaming(
            input="task",
            temperature=0.3,
            top_p=0.8,
            max_output_tokens=123,
            instructions="system",
            reasoning={"summary": "auto"},
        ),
        current,
    )
    payload, kwargs = captured[0]
    assert payload["model_defaults"] == {
        "model": "default",
        "temperature": 0.3,
        "top_p": 0.8,
        "max_output_tokens": 123,
        "parallel_tool_calls": True,
        "reasoning": {"summary": "auto"},
    }
    assert payload["model_url"] == "http://model/ng-rollout/rollout/v1/responses"
    assert payload["task_env"] == {"TASK_VALUE": "preserved"}
    assert kwargs["timeout"] == 60
    assert response.usage is None and response.output_text == "done"
    closed = await instance._close_agent_session_state(current)
    current.session.close.assert_awaited_once()
    assert closed.resources_cookies is None
    assert any(g.code == "model_usage_unavailable" for g in closed.agent_observations.gaps)


@pytest.mark.asyncio
async def test_setup_failure_disconnects_borrowed_sandbox_and_retains_failed_cleanup(monkeypatch):
    instance = agent()
    provider = SimpleNamespace(name="local", aclose=AsyncMock())
    sandbox = SimpleNamespace(sandbox_id="borrowed")
    monkeypatch.setattr(sandbox_agent, "resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(sandbox_agent, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(sandbox_agent, "create_provider", lambda config: provider)
    monkeypatch.setattr(sandbox_agent.AsyncSandbox, "connect", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(
        DeepAgentsSandboxSession, "install_runtime", AsyncMock(side_effect=RuntimeError("install failed"))
    )
    cleanup = AsyncMock()
    monkeypatch.setattr(DeepAgentsSandboxSession, "close", cleanup)
    with pytest.raises(RuntimeError, match="install failed"):
        await instance._seed_agent_session_state(seed(sandbox_access=access()))
    cleanup.assert_awaited_once()
    cleanup.side_effect = RuntimeError("disconnect failed")
    with pytest.raises(AgentSessionSetupError) as error:
        await instance._seed_agent_session_state(seed(sandbox_access=access()))
    assert isinstance(error.value.state, DeepAgentsSandboxSession)
    assert str(error.value.error) == "install failed"


def test_sessionless_activation_is_rejected():
    with TestClient(agent().setup_webserver()) as client:
        response = client.post("/v1/responses", json={"input": "task"})
    assert response.status_code == 409


@pytest.mark.parametrize(
    "native_status,stop_reason,error_status,timed_out,expected,error_type",
    [
        ("completed", None, None, False, "completed", None),
        ("failed", None, 400, False, "failed", "ModelRequestError"),
        ("incomplete", "wall_time", None, True, "timeout", None),
        ("incomplete", "cancelled", None, False, "cancelled", None),
        ("incomplete", "max_iterations", None, False, "failed", "GraphRecursionError"),
    ],
)
def test_close_bundle_retains_actual_supervisor_outcome(
    native_status, stop_reason, error_status, timed_out, expected, error_type
):
    instance, current = agent(), state()
    current.session.artifacts = output() | {
        "status": native_status,
        "stop_reason": stop_reason,
        "error_status": error_status,
    }
    current.session.cleanup["timed_out"] = timed_out
    current.session.cleanup["return_code"] = 143 if timed_out else 0
    bundle, _ = instance._artifacts(current)
    (sandbox,) = [r for r in bundle.records if r.kind == "sandbox"]
    assert sandbox.sandbox_id == "borrowed" and sandbox.provider == "sandbox"
    assert sandbox.outcome == expected and sandbox.error_type == error_type
    assert sandbox.exit_code == (143 if timed_out else 0)
    assert sandbox.cpu_time_s is None and sandbox.peak_memory_mib is None
    current.session.cleanup = None
    bundle, _ = instance._artifacts(current)
    assert not any(r.kind == "sandbox" for r in bundle.records)
    assert any(g.code == "sandbox_outcome_unavailable" for g in bundle.gaps)
