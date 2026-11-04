# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed, activation, timeout and close behavior of the Fabric agent session, through its own routes."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from nemo_gym.agents import (
    AGENT_SESSION_COOKIE_KEY,
    AGENT_TIMEOUT_METADATA_KEY,
    ROLLOUT_STOP_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentSeedRejectedError,
    AgentSessionSetupError,
    check_agent_seed_rejected,
)
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from nemo_gym.trajectory import AgentObservationBundle
from responses_api_agents.nemo_fabric_agent import app
from responses_api_agents.nemo_fabric_agent.sandbox import FabricSandboxSession
from responses_api_agents.nemo_fabric_agent.sandbox_runner import Invocation

from .test_sessions import agent, request, response, seed


def _mcp(name, *, required):
    return MCPToolAccess(
        name=name, required=required, connection=MCPStreamableHTTPConnection(url="http://resources:8000/mcp")
    )


def _remote(*, disconnect_error=None):
    """A sandbox connection whose release is observable."""
    return SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")),
        upload=AsyncMock(),
        start=AsyncMock(),
        stop=AsyncMock(),
        disconnect=AsyncMock(side_effect=disconnect_error),
        uploads_independent_of_execs=True,
    )


def _patch_sandbox(monkeypatch, remote, *, install="/opt/private/python"):
    constructor = MagicMock(return_value=remote)
    constructor.connect = AsyncMock(return_value=remote)
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    monkeypatch.setattr(app, "create_provider", MagicMock(return_value=SimpleNamespace(aclose=AsyncMock())))
    monkeypatch.setattr(app.NeMoFabricAgent, "resolve_model_base_url", lambda *_: "http://model.example/v1")
    installer = AsyncMock(return_value=install) if isinstance(install, str) else install
    monkeypatch.setattr(app, "prepare_fabric_runtime", installer)
    return installer


def _sandbox_access(**updates):
    return {
        "connection": {"provider_config_ref": "sandbox", "descriptor": {"id": "borrowed"}},
        "workdir": "/app",
        **updates,
    }


# --- seed refusals, through the real /v1/agent_sessions route -------------------------------------------------


@pytest.mark.parametrize(
    ("config", "updates", "reason"),
    [
        ({}, {"tool_accesses": [_mcp("tools", required=True)]}, "required remote grants"),
        (
            {},
            {"tool_accesses": [DirectHTTPToolAccess(name="memory", required=True, base_url="http://resources:8000")]},
            "required remote grants",
        ),
        ({}, {"sandbox_access": _sandbox_access(user="root")}, "task-specific agent user"),
        ({}, {"continuation": {}}, "multi-turn continuation"),
        ({}, {"borrowed": False}, "sandbox_access or a configured sandbox_provider"),
        ({"adapter_id": "nvidia.fabric.unknown"}, {}, "No sandbox runtime profile"),
        ({"fabric_config": {"mcp": {"servers": {"x": {}}}}}, {}, "MCP configuration or skills"),
        ({"fabric_config": {"relay": {"project": "x"}}}, {}, "artifact and Relay paths"),
        ({"fabric_config": {"runtime": {"artifacts": "/app/out"}}}, {}, "artifact and Relay paths"),
        ({"fabric_config": {"environment": {"workspace": "/other"}}}, {}, "environment is provided by Resources"),
        ({"fabric_config": {"models": {"judge": {"model": "other"}}}}, {}, "default model role"),
        ({"num_workers": 2}, {}, "num_workers=1"),
        ({"send_model_base_url": False}, {}, "Gym model-server URL"),
        ({"cwd": "/other"}, {}, "cwd conflicts"),
        *[
            ({}, {"sandbox_access": _sandbox_access(workdir=workdir)}, "absolute task directory")
            for workdir in ["app", "/", "/tmp", "/opt", "/app/../etc", "/app\x00x"]
        ],
        *[
            ({}, {"sandbox_access": _sandbox_access(workdir=workdir)}, "separate from runtime/session storage")
            for workdir in ["/tmp/nemo-gym-fabric-sessions/x", "/opt/nemo-gym-fabric/python"]
        ],
        # Deterministic configuration errors are final: a plain ValueError or TypeError would be a retried 500.
        (
            {"sandbox_provider": "sandbox", "sandbox_config": {"bogus": 1}},
            {"borrowed": False},
            "Invalid Fabric sandbox configuration",
        ),
        ({"sandbox_provider": "missing"}, {"borrowed": False}, "Invalid Fabric sandbox configuration"),
        (
            {},
            {
                "sandbox_access": _sandbox_access(
                    connection={"provider_config_ref": "missing", "descriptor": {"id": "borrowed"}}
                )
            },
            "Invalid Fabric sandbox provider",
        ),
    ],
)
def test_unsupported_seed_is_refused_with_a_final_422_before_any_sandbox_is_touched(
    monkeypatch, config, updates, reason
):
    provider = MagicMock()
    monkeypatch.setattr(app, "create_provider", provider)
    instance = agent(**config)

    detail = check_agent_seed_rejected(instance.setup_webserver(), seed(**updates))

    assert reason in detail
    provider.assert_not_called()
    assert not instance._session_records


def test_training_token_capture_is_refused_because_sessions_are_evaluation_only():
    instance = agent()
    instance.server_client.global_config_dict = {"token_id_capture": {"enabled": True, "all_agents": True}}

    assert "evaluation-only" in check_agent_seed_rejected(instance.setup_webserver(), seed())


async def test_loopback_model_url_is_refused_for_a_remote_sandbox_provider_before_it_is_created(monkeypatch):
    provider = MagicMock()
    monkeypatch.setattr(app, "create_provider", provider)
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr(app.NeMoFabricAgent, "resolve_model_base_url", lambda *_: "http://127.0.0.1:8000/v1")

    with pytest.raises(AgentSeedRejectedError, match="loopback"):
        await agent()._seed_agent_session_state(seed())

    provider.assert_not_called()


async def test_optional_grants_are_recorded_as_gaps_and_do_not_fail_the_seed(monkeypatch):
    _patch_sandbox(monkeypatch, _remote())

    state = await agent()._seed_agent_session_state(seed(tool_accesses=[_mcp("weather", required=False)]))

    gaps = {gap.code: gap.detail for gap in state.observations.gaps}
    assert gaps["optional_tool_access_ignored"] == "weather"
    assert "agent_invocation_unobserved" in gaps
    assert state.observations.source == "nemo_fabric"


@pytest.mark.parametrize(("sandbox_config", "ttl"), [({}, 120 + 300 + 600), ({"ttl_s": 5}, 5)])
async def test_an_owned_sandbox_gets_a_ttl_so_a_server_crash_cannot_leak_it(monkeypatch, sandbox_config, ttl):
    remote = _remote()
    _patch_sandbox(monkeypatch, remote)
    instance = agent(
        sandbox_provider="sandbox", sandbox_config=sandbox_config, sandbox_install_timeout_seconds=120, timeout=300
    )

    await instance._seed_agent_session_state(seed(borrowed=False))

    assert remote.start.await_args.args[0].ttl_s == ttl


# --- seed failure: release and the original error ------------------------------------------------------------


@pytest.mark.parametrize("borrowed", [True, False])
async def test_failed_runtime_install_releases_what_the_seed_acquired(monkeypatch, borrowed):
    remote = _remote()
    _patch_sandbox(monkeypatch, remote, install=AsyncMock(side_effect=RuntimeError("pip failed")))

    with pytest.raises(RuntimeError, match="pip failed"):
        await agent(sandbox_provider="sandbox")._seed_agent_session_state(seed(borrowed=borrowed))

    if borrowed:  # Resources keeps the task sandbox; the borrowed connection is only released.
        remote.disconnect.assert_awaited_once()
        remote.stop.assert_not_awaited()
    else:  # A sandbox the agent started is stopped.
        remote.stop.assert_awaited_once()
        remote.disconnect.assert_not_awaited()


async def test_failed_sandbox_connect_closes_the_provider_the_seed_created(monkeypatch):
    _patch_sandbox(monkeypatch, _remote())
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", MagicMock(return_value=provider))
    app.AsyncSandbox.connect.side_effect = ConnectionError("sandbox is gone")

    with pytest.raises(ConnectionError, match="sandbox is gone"):
        await agent()._seed_agent_session_state(seed())

    provider.aclose.assert_awaited_once()


async def test_failed_release_after_a_failed_install_hands_the_state_back_for_close(monkeypatch):
    remote = _remote(disconnect_error=RuntimeError("release failed"))
    _patch_sandbox(monkeypatch, remote, install=AsyncMock(side_effect=RuntimeError("pip failed")))

    with pytest.raises(AgentSessionSetupError) as error:
        await agent()._seed_agent_session_state(seed())

    assert isinstance(error.value.state, FabricSandboxSession)
    assert str(error.value.error) == "pip failed"


async def test_install_that_exceeds_its_timeout_fails_the_seed_and_releases_the_sandbox(monkeypatch):
    remote = _remote()

    async def hang(*_, **__):
        await asyncio.sleep(30)

    _patch_sandbox(monkeypatch, remote, install=AsyncMock(side_effect=hang))

    with pytest.raises(TimeoutError):
        await agent(sandbox_install_timeout_seconds=1)._seed_agent_session_state(seed())

    remote.disconnect.assert_awaited_once()


# --- the activation routes -------------------------------------------------------------------------------------


def _request(session, *, rollout="rollout"):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "session": session,
            "path_params": {"rollout_id": rollout},
        }
    )


async def test_activation_without_a_session_and_after_close_is_a_409_not_a_fallback_to_the_host_path(monkeypatch):
    _patch_sandbox(monkeypatch, _remote())
    instance = agent()
    instance._create_response = AsyncMock(side_effect=AssertionError("host path must not run"))
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")

    # A session marker with no seeded session behind it.
    with pytest.raises(HTTPException) as unknown:
        await instance.responses(_request({AGENT_SESSION_COOKIE_KEY: "session"}), body)
    assert unknown.value.status_code == 409

    cookies = {}
    binding = seed()
    await instance.seed_agent_session(_request(cookies), binding)
    monkeypatch.setattr(
        type(instance), "_sandbox_response", lambda self, state, body: asyncio.sleep(0, response(self, body))
    )
    assert (await instance.responses(_request(cookies), body)).output[0].content[0].text == "answer"

    closed = await instance.close_agent_session(
        _request(cookies), AgentCloseSessionRequest(agent_session_id="session", episode_id=binding.episode_id)
    )
    assert closed.cleanup_confirmed is True
    with pytest.raises(HTTPException) as stale:
        await instance.responses(_request(cookies), body)
    assert stale.value.status_code == 409
    instance._create_response.assert_not_called()


async def test_activation_after_close_started_is_a_409(monkeypatch):
    instance = agent()
    state = FabricSandboxSession(request=seed(), session=SimpleNamespace(closing=True))
    monkeypatch.setattr(type(instance), "_require_agent_session", lambda *_: state)

    with pytest.raises(HTTPException) as error:
        await instance.responses(request(), NeMoGymResponseCreateParamsNonStreaming(input="hello"))

    assert error.value.status_code == 409 and "closing" in error.value.detail


async def test_a_failed_activation_is_not_rerun_by_a_retry(monkeypatch):
    instance = agent()
    state = FabricSandboxSession(request=seed(), session=SimpleNamespace(closing=False))
    monkeypatch.setattr(type(instance), "_require_agent_session", lambda *_: state)
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    calls = []

    async def fail(*_):
        calls.append(True)
        raise RuntimeError("worker died")

    monkeypatch.setattr(type(instance), "_sandbox_response", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="worker died"):
            await instance.responses(request(), body)
    assert len(calls) == 1


async def test_activation_route_refuses_unsupported_fields_before_starting_the_harness(monkeypatch):
    instance = agent()
    state = FabricSandboxSession(request=seed(), session=SimpleNamespace(closing=False))
    monkeypatch.setattr(type(instance), "_require_agent_session", lambda *_: state)
    started = []

    async def run(*_):
        started.append(True)

    monkeypatch.setattr(type(instance), "_sandbox_response", run)

    with pytest.raises(HTTPException) as error:
        await instance.responses(request(), NeMoGymResponseCreateParamsNonStreaming(input="hello", temperature=0))

    assert error.value.status_code == 422
    assert state.task is None and not started


@pytest.mark.parametrize("attempt", [0, 2])
async def test_retried_attempt_uses_its_capture_key_for_the_route_and_the_model_url(monkeypatch, attempt):
    instance = agent()
    binding = seed(episode_id={"rollout_id": "rollout", "attempt": attempt})
    key = "rollout" if attempt == 0 else "rollout-a2"
    assert binding.episode_id.capture_key == key
    captured = {}

    async def execute(*, stage_activation, collect, timeout, close_timeout):
        await stage_activation()
        return {"status": "succeeded", "output": {"response": "answer"}}

    state = _activation_state(execute, binding=binding)

    async def stage(_self, payload):
        captured.update(payload)

    monkeypatch.setattr(FabricSandboxSession, "stage_activation", stage)
    monkeypatch.setattr(
        type(instance), "resolve_model_base_url", lambda _self, name, rollout: f"http://model/ng-rollout/{rollout}/v1"
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    await instance._sandbox_response(state, body)
    assert captured["request_id"] == key
    assert captured["config"]["models"]["default"]["base_url"] == f"http://model/ng-rollout/{key}/v1"

    # The route that a retried attempt's model calls use is also the only one its activation is accepted on.
    monkeypatch.setattr(type(instance), "_require_agent_session", lambda *_: state)
    state.session.closing = False
    state.activation_request = body
    state.task = asyncio.create_task(asyncio.sleep(0, response(instance, body)))
    wrong = "rollout-a2" if attempt == 0 else "rollout"
    with pytest.raises(HTTPException) as mismatch:
        await instance.responses(request(rollout=wrong), body)
    assert mismatch.value.status_code == 409
    assert (await instance.responses(request(rollout=key), body)).status == "completed"


# --- request validation ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"temperature": 0},
        {"max_output_tokens": 5},
        {"previous_response_id": "previous"},
        {"tools": [{"type": "function", "name": "f", "parameters": {}, "strict": False}]},
        {"tool_choice": "required"},
        {"parallel_tool_calls": False},
        {"top_p": 0.5},
        {"reasoning": {"effort": "high"}},
        {"store": True},
        {"metadata": {"other": "x"}},
        {"model": "some-other-model"},
        {
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "x"},
            ]
        },
        {
            "input": [
                {"role": "user", "content": [{"type": "input_image", "image_url": "http://x/y.png", "detail": "auto"}]}
            ]
        },
        {"input": [{"role": "tool", "content": "x"}]},
        {"input": []},
        {"input": [{"role": "system", "content": "only a system prompt"}]},
    ],
)
def test_unsupported_or_empty_requests_are_refused(fields):
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "hello", **fields})
    with pytest.raises(HTTPException) as error:
        agent()._validate_session_request(body)
    assert error.value.status_code == 422


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"model": "gym-policy-model"},
        {"model": "policy_model"},
        {"metadata": {AGENT_TIMEOUT_METADATA_KEY: "30"}},
        {"instructions": "be brief"},
        {
            "input": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": [{"type": "input_text", "text": "q"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "a", "annotations": []}]},
                {"role": "developer", "content": "d"},
                {"role": "user", "content": "q2"},
            ]
        },
    ],
)
def test_text_only_requests_the_environment_server_sends_are_accepted(fields):
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "hello", **fields})
    agent()._validate_session_request(body)


# --- _sandbox_response outcomes --------------------------------------------------------------------------------


def _activation_state(execute, *, binding=None, cleanup=None, **session):
    return FabricSandboxSession(
        request=binding or seed(),
        session=SimpleNamespace(
            workdir="/app",
            session_dir="/tmp/fabric",
            execute=execute,
            cleanup=cleanup if cleanup is not None else {"timed_out": False, "return_code": 0},
            cleanup_confirmed=True,
            artifacts=None,
            closing=False,
            expired=False,
            **session,
        ),
        observations=AgentObservationBundle(source="nemo_fabric"),
    )


def _returning(result):
    async def execute(*, stage_activation, collect, timeout, close_timeout):
        return result

    return execute


async def test_known_token_usage_is_reported_and_unknown_usage_is_not_invented(model_url):
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    known = _activation_state(
        _returning(
            {"status": "succeeded", "output": {"response": "a", "usage": {"input_tokens": 3, "output_tokens": 4}}}
        )
    )

    result = await agent()._sandbox_response(known, body)

    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens) == (3, 4, 7)
    assert not [gap for gap in known.observations.gaps if gap.code == "usage_unavailable"]


async def test_fabric_receives_the_same_time_budget_gym_enforces_and_a_payload_the_worker_accepts(
    model_url, monkeypatch
):
    captured, payload = {}, {}

    async def execute(*, stage_activation, collect, timeout, close_timeout):
        await stage_activation()
        captured["timeout"] = timeout
        return {"status": "succeeded", "output": {"response": "a"}}

    async def stage(_self, activation):
        payload.update(activation)

    state = _activation_state(execute, binding=seed(agent_timeout_sec=90))
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello", metadata={AGENT_TIMEOUT_METADATA_KEY: "45"})
    monkeypatch.setattr(FabricSandboxSession, "stage_activation", stage)
    await agent()._sandbox_response(state, body)

    assert captured["timeout"] == 45
    assert payload["config"]["runtime"]["timeout_seconds"] == 45
    assert payload["config"]["runtime"]["artifacts"] == "/tmp/fabric/artifacts"
    # The worker has no Gym dependency, so nothing but this check ties its input model to the payload built here.
    Invocation.model_validate(payload)


async def test_worker_nonzero_exit_without_a_timeout_fails_the_activation_and_is_recorded(model_url):
    state = _activation_state(
        _returning({"status": "succeeded", "output": {"response": "a"}}),
        cleanup={"timed_out": False, "return_code": 3},
    )

    with pytest.raises(RuntimeError, match="exited unsuccessfully: 3"):
        await agent()._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="hello"))

    assert state.observations.records[0].outcome == "failed" and state.observations.records[0].exit_code == 3


async def test_activation_that_loses_the_race_with_close_is_cancelled_not_returned(model_url):
    async def execute(*, stage_activation, collect, timeout, close_timeout):
        state.session.closing = True
        return {"status": "succeeded", "output": {"response": "a"}}

    state = _activation_state(execute)

    with pytest.raises(asyncio.CancelledError):
        await agent()._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="hello"))


async def test_an_adapter_timeout_is_an_incomplete_rollout_to_verify_not_a_server_error(model_url):
    # Fabric gives its adapters a deadline ten seconds before the one Gym enforces, so Claude and Codex report their own.
    timeout = {"status": "failed", "error": {"code": "timeout", "message": "Claude invocation timed out"}}
    state = _activation_state(_returning(timeout))

    result = await agent(adapter_id="nvidia.fabric.claude")._sandbox_response(
        state, NeMoGymResponseCreateParamsNonStreaming(input="hello")
    )

    assert result.status == "incomplete" and result.usage is None
    assert state.observations.records[0].outcome == "timeout"


async def test_any_other_failed_result_still_fails_the_activation(model_url):
    failed = {"status": "failed", "error": {"code": "claude_missing_result", "message": "no result"}}
    state = _activation_state(_returning(failed))

    with pytest.raises(RuntimeError, match="NeMo Fabric invocation failed: no result"):
        await agent(adapter_id="nvidia.fabric.claude")._sandbox_response(
            state, NeMoGymResponseCreateParamsNonStreaming(input="hello")
        )


@pytest.mark.parametrize("usage", [{"api_calls": 2}, {"input_tokens": 5, "output_tokens": 6}])
async def test_confirmed_timeout_with_a_partial_result_is_incomplete_and_keeps_only_known_usage(model_url, usage):
    partial = {"status": "failed", "error": {"code": "x"}, "output": {"response": "partial", "usage": usage}}
    state = _activation_state(
        AsyncMock(side_effect=TimeoutError("deadline")), cleanup={"timed_out": True, "return_code": 143}
    )
    state.session.artifacts = partial

    result = await agent()._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="hello"))

    assert result.status == "incomplete"
    assert result.output[-1].content[0].text == "partial"
    assert (result.usage is None) is ("input_tokens" not in usage)
    assert state.observations.records[0].outcome == "timeout"


async def test_timeout_without_any_result_declares_the_gap_and_a_stop_is_marked(model_url):
    state = _activation_state(
        AsyncMock(side_effect=TimeoutError("deadline")),
        cleanup={"timed_out": True, "return_code": 143},
        expire_reason="user stop",
    )
    state.session.expired = True

    result = await agent()._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="hello"))

    assert result.status == "incomplete" and result.output == []
    assert "partial_fabric_result_unavailable" in {gap.code for gap in state.observations.gaps}
    assert json.loads(result.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": "user stop"}


async def test_a_timeout_that_races_with_close_is_not_returned_as_an_incomplete_response(model_url):
    state = _activation_state(
        AsyncMock(side_effect=TimeoutError("deadline")), cleanup={"timed_out": True, "return_code": 143}
    )
    state.session.closing = True

    with pytest.raises(TimeoutError, match="deadline"):
        await agent()._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="hello"))


async def test_concurrency_limit_also_bounds_session_activations(model_url):
    instance = agent(concurrency=1)
    running, peak, first_running, release = 0, 0, asyncio.Event(), asyncio.Event()

    async def execute(*, stage_activation, collect, timeout, close_timeout):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        first_running.set()
        await release.wait()
        running -= 1
        return {"status": "succeeded", "output": {"response": "a"}}

    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    tasks = [asyncio.create_task(instance._sandbox_response(_activation_state(execute), body)) for _ in range(2)]
    await asyncio.wait_for(first_running.wait(), timeout=2)
    await asyncio.sleep(0.05)  # the second activation had every chance to start
    assert running == 1
    release.set()
    await asyncio.gather(*tasks)
    assert peak == 1


async def test_session_relay_records_under_the_session_directory_and_the_response_comes_from_atif(
    model_url, monkeypatch
):
    captured = {}
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "message": "Let me look.",
                "tool_calls": [{"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": "ls"}}],
                "observation": {"results": [{"source_call_id": "c1", "content": "README.txt"}]},
            },
            {"source": "agent", "message": "answer"},
        ]
    }

    async def execute(*, stage_activation, collect, timeout, close_timeout):
        await stage_activation()
        return {"status": "succeeded", "output": {"response": "answer"}, "relay_atif": [trajectory]}

    async def stage(_self, payload):
        captured.update(payload)

    monkeypatch.setattr(FabricSandboxSession, "stage_activation", stage)
    state = _activation_state(execute)

    result = await agent(relay=True)._sandbox_response(state, NeMoGymResponseCreateParamsNonStreaming(input="hello"))

    config = captured["config"]
    assert config["relay"]["output_dir"] == "/tmp/fabric/relay/atif"
    assert config["relay"]["observability"]["atif"]["enabled"] is True
    assert config["telemetry"]["providers"] == {"relay": {}}
    assert config["runtime"]["artifacts"] == "/tmp/fabric/artifacts"  # the session's, not a host path
    assert [item.type for item in result.output] == ["message", "function_call", "function_call_output", "message"]
    assert result.output[2].output == "README.txt"


def test_host_relay_loader_drops_a_file_that_is_not_a_trajectory_and_orders_the_rest(tmp_path):
    contents = {
        "a_list": [],
        "null_step": {"steps": [None]},
        "valid": {"steps": [{"source": "agent", "timestamp": "1"}]},
    }
    artifacts = []
    for name, content in contents.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content))
        artifacts.append({"kind": "atif", "path": str(path)})

    loaded = app._atif_trajectories({"relay_artifacts": artifacts})

    # A step-less trajectory sorts by "None", after a timestamped one.
    assert loaded == [contents["valid"], contents["null_step"]]


# --- close ---------------------------------------------------------------------------------------------------


async def test_close_cancels_an_unfinished_activation_that_produced_no_artifacts():
    started = asyncio.Event()

    async def hang():
        started.set()
        await asyncio.Event().wait()

    state = FabricSandboxSession(
        request=seed(), session=SimpleNamespace(close=AsyncMock(), artifacts=None, cleanup_confirmed=True)
    )
    state.task = asyncio.create_task(hang())
    await started.wait()

    await state.close(1)

    assert state.task.cancelled()
    state.session.close.assert_awaited_once_with(timeout=1)


async def test_close_after_a_failed_activation_still_returns_cleanup_evidence():
    async def fail():
        raise RuntimeError("worker died")

    bundle = AgentObservationBundle(source="nemo_fabric")
    state = FabricSandboxSession(
        request=seed(),
        session=SimpleNamespace(close=AsyncMock(), artifacts=None, cleanup_confirmed=False),
        observations=bundle,
    )
    state.task = asyncio.create_task(fail())
    await asyncio.wait({state.task})  # the activation has already failed when close arrives

    result = await agent()._close_agent_session_state(state)

    assert result.cleanup_confirmed is False
    assert result.agent_observations is bundle


async def test_close_before_any_activation_reports_the_unobserved_activation():
    state = FabricSandboxSession(
        request=seed(), session=SimpleNamespace(close=AsyncMock(), artifacts=None, cleanup_confirmed=True)
    )

    result = await agent()._close_agent_session_state(state)

    assert result.cleanup_confirmed is True
    assert [gap.code for gap in result.agent_observations.gaps] == ["agent_activation_unobserved"]


async def test_close_that_cannot_release_raises_so_the_base_keeps_the_state_for_retry():
    state = FabricSandboxSession(
        request=seed(),
        session=SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("release failed")), artifacts=None),
    )

    with pytest.raises(RuntimeError, match="release failed"):
        await agent()._close_agent_session_state(state)


async def test_collect_artifacts_reads_the_result_and_refuses_a_missing_or_malformed_one(monkeypatch):
    state = FabricSandboxSession(
        request=seed(), session=SimpleNamespace(sandbox=SimpleNamespace(), session_dir="/tmp/session")
    )
    files = {}

    async def read_text(_sandbox, *, path):
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]

    monkeypatch.setattr("responses_api_agents.nemo_fabric_agent.sandbox.read_text", read_text)

    files["/tmp/session/result.json"] = json.dumps({"status": "succeeded"})
    assert await state.collect_artifacts() == {"status": "succeeded"}

    files["/tmp/session/result.json"] = "[1]"
    with pytest.raises(ValueError, match="JSON object"):
        await state.collect_artifacts()

    del files["/tmp/session/result.json"]
    files["/tmp/session/stderr.log"] = "x" * 9000 + "Traceback: boom"
    with pytest.raises(RuntimeError, match="Traceback: boom") as error:
        await state.collect_artifacts()
    assert "/tmp/session" in str(error.value)
