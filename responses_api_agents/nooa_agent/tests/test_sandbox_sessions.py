# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.agent_utils.supervisor_client import SUPERVISOR_FILE
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent import app as app_module
from responses_api_agents.nooa_agent.app import NOOAAgent
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_app import request, seed
from responses_api_agents.nooa_agent.tests.test_config import agent_config, invocation_config
from responses_api_agents.nooa_agent.tests.test_sandbox_runner import MemorySandbox, complete_files


def access() -> dict:
    return {
        "connection": {"kind": "direct", "provider_config_ref": "provider", "descriptor": {"id": "borrowed"}},
        "workdir": "/app",
    }


@pytest.fixture
def setup(monkeypatch):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy": {"responses_api_models": {"model": {"host": "model", "port": 8000}}}}
    client._build_server_base_url.return_value = "http://model:8000"
    instance = NOOAAgent(config=agent_config(nooa=invocation_config(execution_mode="sandboxed")), server_client=client)
    sandbox = MemorySandbox()
    connect = AsyncMock(return_value=sandbox)
    staging = AsyncMock(return_value="/opt/nooa/bin/python")
    monkeypatch.setattr(app_module.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(app_module, "resolve_provider_config", MagicMock())
    monkeypatch.setattr(app_module, "create_provider", MagicMock())
    monkeypatch.setattr(app_module, "prepare_nooa_runtime", staging)
    return instance, sandbox, staging, connect


async def close(instance: NOOAAgent, session: str = "session"):
    return await instance.close_agent_session(
        request(session_id=session), AgentCloseSessionRequest(agent_session_id=session, episode_id=seed().episode_id)
    )


async def test_missing_resources_sandbox_is_rejected(setup) -> None:
    instance, sandbox, staging, connect = setup
    with pytest.raises(HTTPException, match="requires Resources-provided"):
        await instance.seed_agent_session(request(), seed())
    connect.assert_not_awaited()
    staging.assert_not_awaited()
    sandbox.stop.assert_not_awaited()


async def test_runtime_staging_failure_keeps_borrowed_connection_for_close(setup) -> None:
    instance, sandbox, staging, _ = setup
    staging.side_effect = RuntimeError("unsupported runtime")
    with pytest.raises(RuntimeError, match="unsupported runtime"):
        await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    state = instance._session_records["session"].state
    assert state.runner.sandbox is sandbox
    await close(instance)
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    instance.server_client.post.assert_not_called()


@pytest.mark.parametrize("stage", ["runtime", "directory"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_setup_cannot_be_retried_as_a_successful_seed(setup, stage: str, cancelled: bool) -> None:
    instance, sandbox, staging, connect = setup
    error = asyncio.CancelledError() if cancelled else RuntimeError("setup failed")
    if stage == "runtime":
        staging.side_effect = error
    else:
        sandbox.exec.side_effect = error
    body = seed(sandbox_access=access())
    with pytest.raises(type(error)) as caught:
        await instance.seed_agent_session(request(), body)
    assert caught.value is error
    record = instance._session_records["session"]
    assert record.state.runner.sandbox is sandbox
    assert record.closing is True
    with pytest.raises(HTTPException, match="closing"):
        await instance.seed_agent_session(request(), body)
    with pytest.raises(HTTPException, match="closing"):
        instance._require_agent_session("session")
    connect.assert_awaited_once()
    staging.assert_awaited_once()
    sandbox.disconnect.assert_not_awaited()
    sandbox.exec.side_effect = None
    receipt = await close(instance)
    assert await close(instance) == receipt
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("grant", [False, True])
async def test_close_returns_sandbox_evidence_cookies_and_replays_receipt(setup, grant: bool) -> None:
    instance, sandbox, _, _ = setup
    await instance.seed_agent_session(
        request(),
        seed(
            sandbox_access=access(),
            tool_accesses=[
                {"kind": "direct_http", "name": "tools", "required": True, "base_url": "http://resources:8000"}
            ]
            if grant
            else [],
        ),
    )
    state = instance._require_agent_session("session")
    complete_files(state.runner, sandbox)
    await instance.responses(request(session_id="session"), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    receipt = await close(instance)
    assert receipt.agent_observations.gaps[0].code == "test"
    assert receipt.resources_cookies == ({"resource": "new"} if grant else None)
    assert set(receipt.model_dump()) == {"agent_session_id", "agent_observations", "resources_cookies"}
    assert await close(instance) == receipt
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    instance.server_client.post.assert_not_called()


async def test_missing_cleanup_receipt_requires_retry_before_disconnect(setup) -> None:
    instance, sandbox, _, _ = setup
    await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    state = instance._require_agent_session("session")
    state.runner.session.launch_started = True
    with pytest.raises(RuntimeError, match="cannot confirm termination"):
        await close(instance)
    sandbox.disconnect.assert_not_awaited()
    instance.server_client.post.assert_not_called()
    complete_files(state.runner, sandbox)
    receipt = await close(instance)
    assert receipt.resources_cookies is None
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_distinct_sessions_keep_connections_cookies_and_state_separate(setup) -> None:
    instance, first_sandbox, _, connect = setup
    second_sandbox = MemorySandbox()
    connect.side_effect = [first_sandbox, second_sandbox]
    for name in ["session", "second"]:
        await instance.seed_agent_session(
            request(),
            seed(
                agent_session_id=name,
                sandbox_access=access(),
                tool_accesses=[
                    {
                        "kind": "direct_http",
                        "name": "resources",
                        "required": True,
                        "base_url": "http://resources:8000",
                        "cookies": {"session": name},
                    }
                ],
            ),
        )
    first = instance._require_agent_session("session")
    second = instance._require_agent_session("second")
    assert first.runner.directory != second.runner.directory
    assert first.runner.sandbox is first_sandbox
    assert second.runner.sandbox is second_sandbox
    first.resources_cookies["extra"] = "private"
    assert second.resources_cookies == {"session": "second"}
    await close(instance)
    assert not second.closing
    second_sandbox.disconnect.assert_not_awaited()
    await close(instance, "second")
    second_sandbox.disconnect.assert_awaited_once()


async def test_negative_cleanup_receipt_releases_borrowed_connection_with_capture_gap(setup) -> None:
    instance, sandbox, _, _ = setup
    await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    state = instance._require_agent_session("session")
    state.runner.session.launch_started = True
    sandbox.files[state.runner.directory + "/cleanup.json"] = (
        '{"cleanup_confirmed":false,"return_code":null,"timed_out":false,"error":"children still running"}'
    )
    receipt = await close(instance)
    assert "sandbox_capture_failed" in {gap.code for gap in receipt.agent_observations.gaps}
    assert await close(instance) == receipt
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_active_shared_session_retries_unknown_cleanup_before_cancelling_provider_exec(setup) -> None:
    instance, sandbox, _, _ = setup
    await instance.seed_agent_session(
        request(),
        seed(
            sandbox_access=access(),
            tool_accesses=[
                {"kind": "direct_http", "name": "resources", "required": True, "base_url": "http://resources"}
            ],
        ),
    )
    state = instance._require_agent_session("session")
    launched = asyncio.Event()
    confirmed = False
    events = []
    original_download = sandbox.download

    async def execute(command: str, **kwargs) -> SandboxExecResult:
        if f"/{SUPERVISOR_FILE}" in command:
            launched.set()
            try:
                await asyncio.Event().wait()
            finally:
                events.append("provider_cancel")
        if "touch " in command:
            events.append("stop")
            if confirmed:
                complete_files(state.runner, sandbox)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def download(source: str, destination: Path) -> None:
        if source.endswith("/result.json"):
            assert state.runner.session.cleanup["cleanup_confirmed"] is True
            events.append("capture")
        await original_download(source, destination)

    async def disconnect() -> None:
        events.append("disconnect")

    sandbox.exec.side_effect = execute
    sandbox.download = download
    sandbox.disconnect.side_effect = disconnect
    activation = asyncio.create_task(
        instance.responses(request(session_id="session"), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    )
    await asyncio.wait_for(launched.wait(), timeout=1)
    with pytest.raises(RuntimeError, match="cannot confirm termination"):
        await close(instance)
    assert events == ["stop"]
    assert not state.execution.done()
    assert instance._session_records["session"].state is state
    sandbox.disconnect.assert_not_awaited()
    confirmed = True
    receipt = await asyncio.wait_for(close(instance), timeout=1)
    with pytest.raises(HTTPException) as interrupted:
        await activation
    assert interrupted.value.status_code == 409
    assert events == ["stop", "stop", "capture", "provider_cancel", "disconnect"]
    assert NOOATerminationReason.CANCELLED in {gap.code for gap in receipt.agent_observations.gaps}
    assert receipt.resources_cookies == {"resource": "new"}
    assert state.execution.done()
    assert await close(instance) == receipt
    sandbox.stop.assert_not_awaited()


async def test_seed_wires_resolved_endpoint_runtime_and_limits_into_one_runner(setup) -> None:
    instance, sandbox, staging, connect = setup
    instance.config.sandbox_runner_timeout_seconds = 1234
    instance.config.session_close_timeout_seconds = 19
    instance.config.max_policy_calls = 7
    body = seed(sandbox_access={**access(), "workdir": "/task/repo"})
    await instance.seed_agent_session(request(), body)
    runner = instance._require_agent_session("session").runner
    assert runner.sandbox is sandbox
    assert runner.session.workdir == "/task/repo"
    assert runner.python == "/opt/nooa/bin/python"
    assert runner.model_base_url == "http://model:8000"
    assert runner.model_server_name == "policy"
    assert runner.timeout == 1234
    assert runner.close_timeout == 19
    assert runner.max_policy_calls == 7
    assert runner.invocation is instance.config.nooa
    instance.server_client._resolve_base_url.assert_not_called()
    instance.server_client._build_server_base_url.assert_called_once_with({"host": "model", "port": 8000})
    staging.assert_awaited_once_with(sandbox)
    connect.assert_awaited_once()
    await instance.seed_agent_session(request(), body)
    assert instance._require_agent_session("session").runner is runner
    connect.assert_awaited_once()
    await close(instance)


async def test_failed_connect_closes_provider_without_retaining_seeded_state(setup) -> None:
    instance, sandbox, staging, connect = setup
    provider = app_module.create_provider.return_value
    provider.aclose = AsyncMock()
    connect.side_effect = ConnectionError("cannot attach sandbox")
    with pytest.raises(ConnectionError, match="cannot attach"):
        await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    provider.aclose.assert_awaited_once()
    assert "session" not in instance._session_records
    staging.assert_not_awaited()
    sandbox.disconnect.assert_not_awaited()


async def test_worker_budget_failure_is_not_reported_as_retryable_dependency_outage(setup) -> None:
    import json

    from responses_api_agents.nooa_agent.runner import NOOARunFailure
    from responses_api_agents.nooa_agent.tests.test_sandbox_runner import checkpoint_artifact, cleanup_receipt

    instance, sandbox, _, _ = setup
    await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    runner = instance._require_agent_session("session").runner
    sandbox.files[runner.directory + "/cleanup.json"] = json.dumps(cleanup_receipt(timed_out=True))
    sandbox.files[runner.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    with pytest.raises(NOOARunFailure, match="worker exceeded its execution deadline"):
        await instance.responses(request(session_id="session"), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    receipt = await close(instance)
    assert NOOATerminationReason.INFRASTRUCTURE_ERROR in {gap.code for gap in receipt.agent_observations.gaps}


async def test_cancelled_request_stays_cancelled_even_when_session_is_closing(setup) -> None:
    instance, sandbox, _, _ = setup
    await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    state = instance._require_agent_session("session")
    active = asyncio.Event()

    async def run(body):
        active.set()
        await asyncio.Event().wait()

    state.runner.run = run
    activation = asyncio.create_task(
        instance.responses(request(session_id="session"), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    )
    await asyncio.wait_for(active.wait(), timeout=1)
    state.closing = True
    activation.cancel("caller cancelled")
    with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
        await activation
    assert not state.execution.done()
    await close(instance)
