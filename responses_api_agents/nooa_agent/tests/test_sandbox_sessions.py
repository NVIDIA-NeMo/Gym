# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent import app as app_module
from responses_api_agents.nooa_agent.app import NOOAAgent
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
    client.global_config_dict = {}
    client._resolve_base_url.return_value = "http://model:8000"
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
    state.runner.launched = True
    complete_files(state.runner, sandbox)
    receipt = await close(instance)
    assert receipt.agent_observations.gaps[0].code == "test"
    assert receipt.resources_cookies == ({"resource": "new"} if grant else None)
    assert set(receipt.model_dump()) == {"agent_session_id", "agent_observations", "resources_cookies"}
    assert await close(instance) == receipt
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    instance.server_client.post.assert_not_called()


async def test_cleanup_failure_requires_retry_before_disconnect(setup) -> None:
    instance, sandbox, _, _ = setup
    await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    state = instance._require_agent_session("session")
    state.runner.launched = True
    sandbox.files[state.runner.directory + "/cleanup.json"] = '{"cleanup_confirmed":false}'
    with pytest.raises(RuntimeError, match="unconfirmed"):
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
