# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentSeedSessionRequest, agent_outcome_policy
from nemo_gym.agents.testing.agent_session_conformance import check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, TaskId
from responses_api_agents.spatialclaw_agent import app, sessions
from responses_api_agents.spatialclaw_agent.tests.test_observability import agent_config


def seed(**updates):
    body = AgentSeedSessionRequest(
        agent_session_id="native-session",
        episode_id=EpisodeId(rollout_id="native-1"),
        task_id=TaskId(taskset="spatialclaw:test", task_id="one"),
        tool_accesses=[
            {
                "kind": "direct_http",
                "name": "spatialclaw",
                "required": True,
                "base_url": "http://resources",
                "cookies": {"resources": "scope"},
            }
        ],
    )
    return AgentSeedSessionRequest.model_validate(body.model_dump() | updates)


def agent():
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return app.SpatialClawAgent(config=agent_config(num_workers=1), server_client=client)


def reply(failure=None):
    result = NeMoGymResponse(
        id="r",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    extras = {"_ng_agent_observations": {"source": "spatialclaw", "records": [], "gaps": []}}
    if failure:
        extras["_ng_agent_failure"] = failure
    return result.model_copy(update=extras)


def test_sessions_run_once_replay_close_and_reject_stale_marker(monkeypatch):
    native = agent()
    monkeypatch.setattr(app, "resource_call", AsyncMock(return_value={}))
    execute = AsyncMock(return_value=reply())
    monkeypatch.setattr(app.SpatialClawAgent, "_responses", execute)
    with TestClient(native.setup_webserver()) as client:
        body = seed().model_dump(mode="json")
        assert client.post("/v1/agent_sessions", json=body).status_code == 200
        assert client.post("/v1/agent_sessions", json=body).status_code == 200
        assert client.post("/ng-rollout/wrong/v1/responses", json={"input": "q"}).status_code == 409
        path = "/ng-rollout/native-1/v1/responses"
        first = client.post(path, json={"input": "q"})
        assert first.status_code == 200, first.text
        assert client.post(path, json={"input": "q"}).json() == first.json()
        assert client.post(path, json={"input": "changed"}).status_code == 409
        execute.assert_awaited_once()
        close = {"agent_session_id": "native-session", "episode_id": seed().episode_id.model_dump()}
        closed = client.post("/v1/agent_sessions/close", json=close)
        assert closed.status_code == 200, closed.text
        assert closed.json()["agent_observations"]["source"] == "spatialclaw"
        assert closed.json()["resources_cookies"] == {"resources": "scope"}
        assert client.post("/v1/agent_sessions/close", json=close).json() == closed.json()
        assert client.post(path, json={"input": "q"}).status_code == 409
        assert client.post("/v1/agent_sessions", json=body).status_code == 409


def test_seed_rejects_unsupported_grants_terminally():
    native = agent()
    unsupported = seed(
        tool_accesses=[
            {
                "kind": "mcp",
                "name": "tools",
                "required": True,
                "connection": {"transport": "streamable_http", "url": "http://tools/mcp"},
            }
        ]
    )
    assert "direct_http" in check_agent_seed_rejected(native.setup_webserver(), unsupported)


@pytest.mark.parametrize(
    "failure,action",
    [
        ({"http_status": 400, "mask_sample": False}, "verify"),
        ({"http_status": 503, "mask_sample": True}, "retry"),
        ({"failure_kind": "agent_timeout", "mask_sample": False}, "verify"),
        ({"failure_kind": "spatialclaw:media_error", "mask_sample": True}, "retry"),
    ],
)
async def test_session_failures_use_environment_outcome_policy(monkeypatch, failure, action):
    native = agent()
    state = await native._seed_agent_session_state(seed())
    monkeypatch.setattr(app, "resource_call", AsyncMock(return_value={}))
    monkeypatch.setattr(app.SpatialClawAgent, "_responses", AsyncMock(return_value=reply(failure)))
    result = await native._session_response(state, MagicMock(), SimpleNamespace())
    assert result.status == "failed"
    assert AGENT_OUTCOME_METADATA_KEY in result.metadata
    assert agent_outcome_policy(result).value == action
    assert "_ng_agent_failure" not in (result.model_extra or {})
    assert state.observations.source == "spatialclaw"


async def test_close_cancels_activation_and_reports_missing_observations():
    native = agent()
    state = await native._seed_agent_session_state(seed())
    state.task = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    closed = await native._close_agent_session_state(state)
    assert state.task.cancelled()
    assert closed.agent_observations.gaps[0].code == "observation_capture_failed"


async def test_remote_kernel_uses_granted_episode_credentials(monkeypatch):
    access = seed().tool_accesses[0]
    response = SimpleNamespace()
    call = AsyncMock(return_value=response)
    monkeypatch.setattr(sessions, "request", call)
    monkeypatch.setattr(sessions, "raise_for_status", AsyncMock())
    monkeypatch.setattr(
        sessions,
        "get_response_json",
        AsyncMock(
            return_value={
                "result": {
                    "stdout": "ok",
                    "stderr": "",
                    "error": None,
                    "display_data": [],
                    "execution_time_sec": 0.1,
                }
            }
        ),
    )
    kernel = sessions.RemoteKernel(access)
    result = await kernel.execute("print('ok')", timeout=30)
    assert result.stdout == "ok" and result.error is None
    kwargs = call.call_args.kwargs
    assert kwargs["url"] == "http://resources/kernel"
    assert kwargs["cookies"] == {"resources": "scope"}
    assert kwargs["json"] == {"operation": "execute", "code": "print('ok')", "timeout": 30}


async def test_remote_kernel_restarts_and_keeps_resource_ownership(monkeypatch):
    calls = []
    values = {
        "start": "new-kernel", "shutdown": None, "reset_namespace": None,
        "get_variables": {"x": {"type": "int", "value": "2"}},
        "check_sentinel": {"text": "A"}, "clear_sentinel": None,
    }

    async def resource(access, path, payload):
        calls.append(payload)
        return {"result": values[payload["operation"]]}

    monkeypatch.setattr(sessions, "resource_call", resource)
    pool = sessions.RemoteKernelPool(seed().tool_accesses[0])
    kernel = await pool.acquire()
    kernel.set_init_code("import math")
    kernel.set_injection_code("x=2")
    await kernel.shutdown()
    assert not kernel.is_running
    assert await kernel.start() == "new-kernel" and kernel.is_running
    await kernel.reset_namespace()
    assert await kernel.get_variables() == {"x": {"type": "int", "value": "2"}}
    assert await kernel.check_sentinel() == {"text": "A"}
    await kernel.clear_sentinel()
    assert calls[-1] == {"operation": "clear_sentinel", "name": "_return_answer_result"}
    before = len(calls)
    await pool.release(kernel)
    await pool.shutdown_all()
    assert len(calls) == before  # Agent close never destroys Resources' kernel.


@pytest.mark.parametrize("accesses", [[], [seed().tool_accesses[0], {
    "kind": "direct_http", "name": "other", "required": True, "base_url": "http://other",
}]])
async def test_native_seed_requires_one_resource_kernel(accesses):
    from nemo_gym.agents import AgentSeedRejectedError
    with pytest.raises(AgentSeedRejectedError, match="exactly one"):
        await agent()._seed_agent_session_state(seed(tool_accesses=accesses))


async def test_swallowed_resource_error_cannot_become_a_native_correct_fallback(monkeypatch, tmp_path):
    import sys
    from types import ModuleType

    native = app.SpatialClawAgent(
        config=agent_config(workspace_root=str(tmp_path / "workspaces")), server_client=MagicMock(spec=ServerClient)
    )
    state = await native._seed_agent_session_state(seed())
    monkeypatch.setattr(app.SpatialClawAgent, "_resolve_spatialclaw_root", AsyncMock(return_value=tmp_path))
    monkeypatch.setattr(app.SpatialClawAgent, "_build_spatialclaw_config", lambda *args: SimpleNamespace())

    class Workflow:
        def __init__(self, config):
            self.llm_client = SimpleNamespace(_client_pool={}, close=AsyncMock())

        async def arun(self, **kwargs):
            try:
                await self._kernel_pool.kernel.execute("print(1)")
            except RuntimeError:
                pass  # The native execute node catches errors and may produce a fallback.
            return {"final_answer": {"text": "A"}, "termination_reason": "fallback"}

    config = ModuleType("spatial_agent.config")
    config.set_config = lambda value: None
    workflow = ModuleType("spatial_agent.workflow")
    workflow.SpatialAgentWorkflow = Workflow
    monkeypatch.setitem(sys.modules, "spatial_agent.config", config)
    monkeypatch.setitem(sys.modules, "spatial_agent.workflow", workflow)

    async def resource(access, path, payload):
        if path == "/get_context":
            return {"instruction": "q", "images": ["/frame.png"], "metadata": {}}
        raise RuntimeError("resource service unavailable")

    monkeypatch.setattr(app, "resource_call", resource)
    monkeypatch.setattr(sessions, "resource_call", resource)
    response = await native._session_response(
        state, SimpleNamespace(cookies={}), NeMoGymResponseCreateParamsNonStreaming(input="q")
    )
    assert response.status == "failed"
    assert response.output_text == ""
    assert agent_outcome_policy(response).value == "retry"
