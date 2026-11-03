# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from nemo_gym.agents import ROLLOUT_STOP_METADATA_KEY, AgentCloseSessionRequest
from nemo_gym.config import ResourcesServerRef
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.langgraph_agent.tests.test_app import (
    FLAVORS,
    _agent,
    _mock_model_response,
    _seed,
)
from responses_api_agents.langgraph_agent.tests.test_app import (
    sandbox_boundary as sandbox_boundary,
)


@pytest.mark.parametrize("agent_class,config_class,workflow,options", FLAVORS)
@pytest.mark.parametrize("training", [False, True])
def test_graph_session_replays_and_preserves_capture(
    sandbox_boundary, agent_class, config_class, workflow, options, training
):
    sandbox, factory = sandbox_boundary
    agent = _agent(agent_class, config_class, options)
    seed = _seed()
    path = (
        f"/ng-rollout/{seed.episode_id.capture_key}"
        + ("/training-token-capture" if training else "")
        + "/v1/responses"
    )
    body = {"input": [{"role": "user", "content": "What is 6 * 7?"}]}
    with TestClient(agent.setup_webserver()) as client:
        seeded = client.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).json() == seeded.json()
        assert client.post("/ng-rollout/wrong/v1/responses", json=body).status_code == 409
        assert client.post("/v1/responses", json=body).status_code == 409
        result = client.post(path, json=body)
        assert result.status_code == 200, result.text
        assert result.json()["output"] == sandbox.response["output"]
        assert result.json()["usage"] == sandbox.response["usage"]
        assert client.post(path, json=body).json() == result.json()
        assert client.post(path, json={"input": "different"}).status_code == 409
        other_path = (
            path.replace("/training-token-capture", "")
            if training
            else path.replace("/v1/responses", "/training-token-capture/v1/responses")
        )
        assert client.post(other_path, json=body).status_code == 409
        assert sandbox.disconnects == sandbox.stops == 0
        payloads = [json.loads(value) for key, value in sandbox.uploads.items() if key.endswith("/input.json")]
        assert len(payloads) == 1
        assert payloads[0]["model_url"] == "http://model-server:8080" + path
        assert payloads[0]["workflow"] == workflow
        assert "cookies" not in payloads[0]
        close_body = {"agent_session_id": seed.agent_session_id, "episode_id": seed.episode_id.model_dump()}
        closed = client.post("/v1/agent_sessions/close", json=close_body)
        assert closed.status_code == 200, closed.text
        assert closed.json()["resources_cookies"] is None
        assert closed.json()["agent_observations"]["gaps"][0]["code"] == "observation_capture_failed"
        assert client.post("/v1/agent_sessions/close", json=close_body).json() == closed.json()
        assert client.post(path, json=body).status_code == 409
        assert agent._session_records[seed.agent_session_id].state is None
    factory.connect.assert_awaited_once()
    assert sandbox.disconnects == 1
    assert sandbox.stops == 0
    assert sandbox.events.index("cleanup-receipt") < sandbox.events.index("disconnect")
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize("kind", ["direct_http", "mcp"])
@pytest.mark.parametrize("configured", [False, True])
def test_graph_rejects_required_tools_before_connecting(sandbox_boundary, kind, configured):
    _, factory = sandbox_boundary
    agent = _agent()
    seed = _seed().model_dump(mode="json")
    grant = {"kind": kind, "name": "required-tools", "required": True}
    grant.update(
        {"base_url": "http://resources:8000"}
        if kind == "direct_http"
        else {"connection": {"transport": "streamable_http", "url": "http://resources:8000/mcp"}}
    )
    if configured:
        agent.config = type(agent.config).model_validate(agent.config.model_dump() | {"tool_accesses": [grant]})
    else:
        seed["tool_accesses"] = [grant]
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed)
    assert response.status_code == 422, response.text
    assert "required-tools" in response.json()["detail"]
    factory.connect.assert_not_awaited()
    assert not agent._session_records


def test_model_url_carries_trace_context_after_the_capture_prefix(sandbox_boundary, monkeypatch):
    from responses_api_agents.langgraph_agent import app as module

    sandbox, _ = sandbox_boundary
    traceparent = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
    monkeypatch.setattr(module, "trace_context_path_segment", lambda: f"/ng-trace/{traceparent}")
    agent = _agent()
    seed = _seed()
    prefix = f"/ng-rollout/{seed.episode_id.capture_key}/training-token-capture"
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).status_code == 200
        assert client.post(f"{prefix}/v1/responses", json={"input": "question"}).status_code == 200
    payload = json.loads(next(value for path, value in sandbox.uploads.items() if path.endswith("/input.json")))
    assert payload["model_url"] == f"http://model-server:8080{prefix}/ng-trace/{traceparent}/v1/responses"


@pytest.mark.parametrize("stopped", [False, True])
def test_graph_stopped_at_its_deadline_returns_an_incomplete_response(sandbox_boundary, stopped):
    # A graph the supervisor ended at its deadline, or through a rollout stop, is verified as incomplete
    # instead of failing with a 500 the Environment Server would retry.
    sandbox, _ = sandbox_boundary
    sandbox.output = {"error": "LangGraph workflow cancelled.", "error_type": "CancelledError", "runtime": {}}
    sandbox.timed_out = True
    agent = _agent()
    seed = _seed()
    rollout_id = seed.episode_id.capture_key
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).status_code == 200
        if stopped:
            stop = client.post(f"/rollouts/{rollout_id}/stop", json={"reason": "budget"})
            assert stop.status_code == 200, stop.text
            assert stop.json()["stopped"] is True
        result = client.post(f"/ng-rollout/{rollout_id}/v1/responses", json={"input": "question"})
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "incomplete"
    assert result.json()["output"] == []
    assert result.json()["usage"] is None
    assert (ROLLOUT_STOP_METADATA_KEY in result.json()["metadata"]) is stopped


def test_optional_tools_and_failed_activation_can_close(sandbox_boundary):
    sandbox, _ = sandbox_boundary
    sandbox.output = {"error": "model unavailable", "traceback": "runner failure"}
    agent = _agent()
    seed = _seed().model_dump(mode="json")
    seed["tool_accesses"] = [
        {"kind": "direct_http", "name": "optional", "required": False, "base_url": "http://resources:8000"}
    ]
    path = f"/ng-rollout/{_seed().episode_id.capture_key}/v1/responses"
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions", json=seed).status_code == 200
        assert client.post(path, json={"input": "question"}).status_code == 500
        assert client.post(path, json={"input": "question"}).status_code == 500
        assert sum("exec bash" in command for command, _, _ in sandbox.commands) == 1
        close = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": seed["agent_session_id"], "episode_id": seed["episode_id"]},
        )
        assert close.status_code == 200, close.text
        assert close.json()["agent_observations"]["gaps"]
    assert sandbox.disconnects == 1
    assert sandbox.stops == 0


async def test_close_during_execution_confirms_cleanup_and_drains_callers(sandbox_boundary, monkeypatch):
    sandbox, _ = sandbox_boundary
    agent = _agent()
    seed = _seed()
    request = SimpleNamespace(
        session={},
        cookies={},
        path_params={"rollout_id": seed.episode_id.capture_key},
        url=SimpleNamespace(path=f"/ng-rollout/{seed.episode_id.capture_key}/v1/responses"),
    )
    started = asyncio.Event()
    stopped = asyncio.Event()
    original_exec = sandbox.exec

    async def execute(command, **kwargs):
        if "exec bash" in command:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sandbox.events.append("provider-exec-stopped")
                stopped.set()
        return await original_exec(command, **kwargs)

    monkeypatch.setattr(sandbox, "exec", execute)
    await agent.seed_agent_session(request, seed)
    body = NeMoGymResponseCreateParamsNonStreaming(input="question")
    running = asyncio.create_task(agent.responses(request, body))
    await asyncio.wait_for(started.wait(), timeout=5)
    replay = asyncio.create_task(agent.responses(request, body))
    await asyncio.sleep(0)
    close = AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
    receipts = await asyncio.gather(*(agent.close_agent_session(request, close) for _ in range(2)))
    assert receipts[0] == receipts[1]
    assert stopped.is_set()
    for task in (running, replay):
        with pytest.raises(asyncio.CancelledError):
            await task
    assert sandbox.disconnects == 1
    assert sandbox.stops == 0
    assert sandbox.events.index("cleanup-receipt") < sandbox.events.index("provider-exec-stopped")
    assert sandbox.events.index("provider-exec-stopped") < sandbox.events.index("disconnect")
    assert agent._session_records[seed.agent_session_id].state is None


async def test_disconnected_waiter_does_not_restart_graph(sandbox_boundary, monkeypatch):
    sandbox, _ = sandbox_boundary
    agent = _agent()
    seed = _seed()
    request = SimpleNamespace(
        session={},
        cookies={},
        path_params={"rollout_id": seed.episode_id.capture_key},
        url=SimpleNamespace(path=f"/ng-rollout/{seed.episode_id.capture_key}/v1/responses"),
    )
    started = asyncio.Event()
    finish = asyncio.Event()
    original_exec = sandbox.exec

    async def execute(command, **kwargs):
        if "exec bash" in command:
            started.set()
            await finish.wait()
        return await original_exec(command, **kwargs)

    monkeypatch.setattr(sandbox, "exec", execute)
    await agent.seed_agent_session(request, seed)
    body = NeMoGymResponseCreateParamsNonStreaming(input="question")
    waiter = asyncio.create_task(agent.responses(request, body))
    await asyncio.wait_for(started.wait(), timeout=5)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    finish.set()
    replay = await asyncio.wait_for(agent.responses(request, body), timeout=5)
    assert replay.output[0].content[0].text == "<answer>42</answer>"
    assert sum("exec bash" in command for command, _, _ in sandbox.commands) == 1
    await agent.close_agent_session(
        request, AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
    )
    assert sandbox.disconnects == 1


@pytest.mark.parametrize("owned", [False, True])
async def test_release_failure_retains_handle_until_close_retry(sandbox_boundary, owned):
    sandbox, _ = sandbox_boundary
    agent = _agent(sandbox_provider="runtime")
    seed = _seed(borrowed=not owned)
    request = SimpleNamespace(session={})
    await agent.seed_agent_session(request, seed)
    changed = seed.model_copy(update={"task_id": {"taskset": "test", "task_id": "different"}})
    with pytest.raises(HTTPException) as error:
        await agent.seed_agent_session(request, changed)
    assert error.value.status_code == 409
    sandbox.release_error = RuntimeError("release temporarily failed")
    close = AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
    with pytest.raises(RuntimeError, match="release temporarily failed"):
        await agent.close_agent_session(request, close)
    assert agent._session_records[seed.agent_session_id].state is not None
    assert agent._session_records[seed.agent_session_id].closing
    sandbox.release_error = None
    await agent.close_agent_session(request, close)
    assert agent._session_records[seed.agent_session_id].state is None
    assert (sandbox.stops, sandbox.disconnects) == ((2, 0) if owned else (0, 2))


@pytest.mark.parametrize("agent_class,config_class,workflow,options", FLAVORS)
def test_legacy_run_preserves_verifier_fields_and_resource_cookies(agent_class, config_class, workflow, options):
    agent = _agent(agent_class, config_class, options)
    agent.config.resources_server = ResourcesServerRef(type="resources_servers", name="resources")
    agent.server_client.global_config_dict = {"observability_enabled": True}
    seeded = _mock_model_response()
    seeded.cookies = {"resources": "seed-cookie"}
    generated = _mock_model_response()
    generated.cookies = {"model": "model-cookie"}
    verification = {
        "responses_create_params": {"input": "question"},
        "response": generated.json.return_value,
        "reward": 0.0,
        "question": "question",
        "expected_answer": "42",
        "_ng_failure_class": "judge_failure",
    }
    agent.server_client.post.side_effect = [seeded, generated, _mock_model_response(verification)]
    row = {
        "responses_create_params": {"input": "question"},
        "question": "question",
        "expected_answer": "42",
        "metadata": {"source_dataset": "test"},
        "_ng_rollout_id": "legacy-rollout",
    }
    with TestClient(agent.setup_webserver()) as client:
        result = client.post("/run", json=row)
    assert result.status_code == 200, result.text
    assert result.json()["_ng_failure_class"] == "judge_failure"
    seed_call, response_call, verify_call = [call.kwargs for call in agent.server_client.post.call_args_list]
    assert seed_call["json"]["expected_answer"] == "42"
    assert response_call["url_path"] == "/ng-rollout/legacy-rollout/v1/responses"
    assert not response_call.get("cookies")
    assert verify_call["cookies"] == {"resources": "seed-cookie"}
    assert verify_call["json"]["metadata"] == row["metadata"]
    assert verify_call["json"]["question"] == "question"


def _direct_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": "/v1/responses",
            "path_params": {},
            "headers": [],
            "session": {"caller": "preserved"},
        }
    )


async def test_transient_sessions_do_not_bind_the_callers_cookie(sandbox_boundary):
    sandbox, _ = sandbox_boundary
    agent = _agent(sandbox_provider="runtime")
    request = _direct_request()
    body = NeMoGymResponseCreateParamsNonStreaming(input="question")

    await agent.responses(request, body)
    await agent.responses(request, body)

    assert request.session == {"caller": "preserved"}
    assert len(sandbox.started) == sandbox.stops == 2
    assert sandbox.disconnects == 0
    assert sum("exec bash" in command for command, _, _ in sandbox.commands) == 2
    assert all(record.state is None for record in agent._session_records.values())
    assert not agent._failed_seed_records


@pytest.mark.parametrize("failure", ["setup", "activation", "release"])
@pytest.mark.parametrize("recovery", ["manual", "expiry"])
async def test_transient_failure_keeps_handle_for_cleanup_retry(sandbox_boundary, monkeypatch, failure, recovery):
    from unittest.mock import AsyncMock

    from responses_api_agents.langgraph_agent.sandbox import LangGraphSandboxSession

    sandbox, _ = sandbox_boundary
    sandbox.release_error = RuntimeError("release unavailable")
    if failure == "setup":
        monkeypatch.setattr(
            LangGraphSandboxSession, "install_runtime", AsyncMock(side_effect=RuntimeError("install failed"))
        )
    agent = _agent(sandbox_provider="runtime")
    request = _direct_request()

    if failure == "activation":
        sandbox.output = {"error": "model failed", "error_type": "upstream"}
    expected_error = {"setup": "install failed", "activation": "model failed", "release": "release unavailable"}[
        failure
    ]
    with pytest.raises(RuntimeError, match=expected_error):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="question"))

    assert request.session == {"caller": "preserved"}
    assert len(agent._failed_seed_records) == 1
    identifier, record = next(iter(agent._failed_seed_records.items()))
    assert record.state is not None and record.closing
    assert record.state.session.sandbox is sandbox
    assert not record.state.session.closed
    releases_before_retry = sandbox.stops
    assert releases_before_retry >= 1
    assert sandbox.disconnects == 0
    if failure == "setup":
        assert not any("exec bash" in command for command, _, _ in sandbox.commands)

    sandbox.release_error = None
    if recovery == "manual":
        await agent.close_agent_session(
            Request({"type": "http", "session": {}}),
            AgentCloseSessionRequest(agent_session_id=identifier, episode_id=record.episode_id),
        )
        assert agent._session_records[identifier].state is None
    else:
        record.expires_at = 0
        agent.expire_agent_session_records()
        pending = tuple(agent._failed_seed_releases)
        assert len(pending) == 1
        await asyncio.wait_for(asyncio.gather(*pending), timeout=5)
        assert identifier not in agent._session_records
    assert not agent._failed_seed_records
    assert sandbox.stops == releases_before_retry + 1
    assert sandbox.disconnects == 0


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("activated", [False, True])
def test_shutdown_releases_unclosed_sessions(sandbox_boundary, owned, activated):
    sandbox, _ = sandbox_boundary
    agent = _agent(sandbox_provider="runtime")
    seed = _seed(borrowed=not owned)
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))
        assert response.status_code == 200, response.text
        if activated:
            response = client.post(
                f"/ng-rollout/{seed.episode_id.capture_key}/v1/responses", json={"input": "question"}
            )
            assert response.status_code == 200, response.text
        assert sandbox.stops == sandbox.disconnects == 0
        assert agent._session_records[seed.agent_session_id].state is not None
    assert agent._session_records[seed.agent_session_id].state is None
    assert (sandbox.stops, sandbox.disconnects) == ((1, 0) if owned else (0, 1))
    if activated:
        release = "stop" if owned else "disconnect"
        assert sandbox.events.index("collect-output") < sandbox.events.index(release)


def test_shutdown_retries_retained_transient_cleanup(sandbox_boundary):
    sandbox, _ = sandbox_boundary
    sandbox.release_error = RuntimeError("release unavailable")
    agent = _agent(sandbox_provider="runtime")
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        response = client.post("/v1/responses", json={"input": "question"})
        assert response.status_code == 500
        assert len(agent._failed_seed_records) == 1
        identifier = next(iter(agent._failed_seed_records))
        assert agent._session_records[identifier].state is not None
        attempts_before_shutdown = sandbox.stops
        sandbox.release_error = None
    assert agent._session_records[identifier].state is None
    assert not agent._failed_seed_records
    assert sandbox.stops == attempts_before_shutdown + 1
    assert sandbox.disconnects == 0
