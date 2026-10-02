# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import ServerClient


class _Agent(SimpleResponsesAPIAgent):
    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body):
        raise NotImplementedError

    async def _seed_agent_session_state(self, body):
        return AgentSessionState(request=body)

    async def _close_agent_session_state(self, state):
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(source="test"),
            resources_cookies={"session": "updated"},
        )


@pytest.fixture
def agent():
    result = _Agent(
        config=BaseResponsesAPIAgentConfig(host="localhost", port=1, entrypoint="app.py", name="test"),
        server_client=MagicMock(spec=ServerClient),
    )
    result._seed_agent_session_state = AsyncMock(wraps=result._seed_agent_session_state)
    result._close_agent_session_state = AsyncMock(wraps=result._close_agent_session_state)
    return result


def _seed(index=0):
    return AgentSeedSessionRequest(
        agent_session_id=f"session-{index}",
        episode_id={"rollout_id": f"episode-{index}"},
        task_id={"taskset": "test", "task_id": str(index)},
    )


def _close(seed):
    return AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)


async def test_simultaneous_seed_retries_initialize_once_and_bind_all_fields(agent):
    seed = _seed()
    first, second = await asyncio.gather(
        *(agent.seed_agent_session(SimpleNamespace(session={}), seed) for _ in range(2))
    )
    assert first == second
    agent._seed_agent_session_state.assert_awaited_once()
    changed = AgentSeedSessionRequest.model_validate(
        seed.model_dump()
        | {
            "tool_accesses": [
                {"kind": "direct_http", "name": "different", "required": True, "base_url": "http://tools"}
            ]
        }
    )
    with pytest.raises(HTTPException, match="another seed request"):
        await agent.seed_agent_session(SimpleNamespace(session={}), changed)


async def test_close_retries_preserve_observations_after_many_other_closes(agent):
    seed = _seed()
    await agent.seed_agent_session(SimpleNamespace(session={}), seed)
    first = await agent.close_agent_session(SimpleNamespace(session={}), _close(seed))
    for index in range(1, 67):
        await agent.close_agent_session(SimpleNamespace(session={}), _close(_seed(index)))
    retry = await agent.close_agent_session(SimpleNamespace(session={}), _close(seed))
    assert retry == first
    assert retry.agent_observations.source == "test"
    assert retry.resources_cookies == {"session": "updated"}
    agent._close_agent_session_state.assert_awaited_once()
    wrong = _close(seed).model_copy(update={"episode_id": _seed(1).episode_id})
    with pytest.raises(HTTPException, match="episode_id"):
        await agent.close_agent_session(SimpleNamespace(session={}), wrong)


async def test_seed_state_is_not_visible_before_setup_finishes(agent):
    seed = _seed()
    entered, release = asyncio.Event(), asyncio.Event()
    state = AgentSessionState(request=seed)

    async def initialize(body):
        entered.set()
        await release.wait()
        return state

    agent._seed_agent_session_state.side_effect = initialize
    task = asyncio.create_task(agent.seed_agent_session(SimpleNamespace(session={}), seed))
    await entered.wait()
    try:
        with pytest.raises(HTTPException) as error:
            agent._require_agent_session(seed.agent_session_id)
        assert error.value.status_code == 409
    finally:
        release.set()
        await task
    assert agent._require_agent_session(seed.agent_session_id) is state


async def test_ready_session_lookup_does_not_depend_on_bookkeeping_lock(agent):
    seed = _seed()
    await agent.seed_agent_session(SimpleNamespace(session={}), seed)
    state = agent._require_agent_session(seed.agent_session_id)
    async with agent._locked_agent_session(seed.agent_session_id):
        assert agent._require_agent_session(seed.agent_session_id) is state


async def test_failed_close_retains_state_and_can_retry(agent):
    seed = _seed()
    await agent.seed_agent_session(SimpleNamespace(session={}), seed)
    state = agent._require_agent_session(seed.agent_session_id)
    agent._close_agent_session_state.side_effect = RuntimeError("cleanup unavailable")
    with pytest.raises(RuntimeError, match="cleanup unavailable"):
        await agent.close_agent_session(SimpleNamespace(session={}), _close(seed))
    assert agent._session_records[seed.agent_session_id].state is state
    with pytest.raises(HTTPException, match="closing"):
        agent._require_agent_session(seed.agent_session_id)
    with pytest.raises(HTTPException, match="closing"):
        await agent.seed_agent_session(SimpleNamespace(session={}), seed)
    agent._close_agent_session_state.side_effect = None
    result = await agent.close_agent_session(SimpleNamespace(session={}), _close(seed))
    assert result.agent_observations.source == "test"
    assert agent._session_records[seed.agent_session_id].state is None


async def test_retry_window_begins_after_cleanup_and_is_not_extended(agent, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    seed = _seed()
    request = SimpleNamespace(session={})
    await agent.seed_agent_session(request, seed)

    async def cleanup(state):
        clock[0] = 200.0
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id)

    agent._close_agent_session_state.side_effect = cleanup
    first = await agent.close_agent_session(request, _close(seed))
    clock[0] = 209.0
    assert await agent.close_agent_session(request, _close(seed)) == first
    clock[0] = 210.0
    with pytest.raises(HTTPException, match="expired"):
        await agent.close_agent_session(request, _close(seed))
    assert not agent._closed_session_records
    assert not agent._session_records
    assert agent._agent_session_id_from_request(request) == seed.agent_session_id
    with pytest.raises(HTTPException):
        agent._require_agent_session(seed.agent_session_id)
    with pytest.raises(HTTPException, match="expired"):
        await agent.seed_agent_session(request, seed)


async def test_unknown_close_prevents_delayed_seed_during_retry_window(agent):
    seed = _seed()
    await agent.close_agent_session(SimpleNamespace(session={}), _close(seed))
    with pytest.raises(HTTPException, match="already closed"):
        await agent.seed_agent_session(SimpleNamespace(session={}), seed)
    agent._seed_agent_session_state.assert_not_awaited()
    agent._close_agent_session_state.assert_not_awaited()


async def test_failed_seed_releases_record_and_waiter_uses_current_lock(agent):
    seed = _seed()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def initialize(body):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
            raise RuntimeError("setup failed")
        await asyncio.sleep(0)
        return AgentSessionState(request=body)

    agent._seed_agent_session_state.side_effect = initialize
    first = asyncio.create_task(agent.seed_agent_session(SimpleNamespace(session={}), seed))
    await entered.wait()
    second = asyncio.create_task(agent.seed_agent_session(SimpleNamespace(session={}), seed))
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(RuntimeError, match="setup failed"):
        await first
    third = asyncio.create_task(agent.seed_agent_session(SimpleNamespace(session={}), seed))
    assert await second == await third
    assert calls == 2
    assert len(agent._session_records) == 1


@pytest.mark.parametrize("marker", [None, "", 0, [], {}])
async def test_malformed_cookie_never_selects_host_path(agent, marker):
    with pytest.raises(HTTPException, match="Invalid agent session marker"):
        await agent.seed_agent_session(SimpleNamespace(session={"agent_session_id": marker}), _seed())
    agent._seed_agent_session_state.assert_not_awaited()


@pytest.mark.parametrize("error", [RuntimeError("partial setup"), asyncio.CancelledError("partial setup")])
async def test_failed_seed_retains_partial_state_only_for_cleanup(agent, error):
    seed = _seed()

    async def initialize(body):
        raise AgentSessionSetupError(AgentSessionState(request=body), error=error)

    agent._seed_agent_session_state.side_effect = initialize
    with pytest.raises(type(error), match="partial setup") as raised:
        await agent.seed_agent_session(SimpleNamespace(session={}), seed)
    assert raised.value is error
    with pytest.raises(HTTPException, match="closing"):
        agent._require_agent_session(seed.agent_session_id)
    with pytest.raises(HTTPException, match="closing"):
        await agent.seed_agent_session(SimpleNamespace(session={}), seed)
    await agent.close_agent_session(SimpleNamespace(session={}), _close(seed))
    agent._close_agent_session_state.assert_awaited_once()
    assert agent._session_records[seed.agent_session_id].state is None


async def test_request_and_close_response_cannot_mutate_stored_binding(agent):
    seed = _seed()
    await agent.seed_agent_session(SimpleNamespace(session={}), seed)
    seed.task_id = seed.task_id.model_copy(update={"task_id": "changed"})
    assert agent._require_agent_session(seed.agent_session_id).request.task_id.task_id == "0"
    first = await agent.close_agent_session(SimpleNamespace(session={}), _close(seed))
    first.resources_cookies["session"] = "changed"
    retry = await agent.close_agent_session(SimpleNamespace(session={}), _close(seed))
    assert retry.resources_cookies == {"session": "updated"}


async def test_close_receipt_expiry_does_not_expire_active_sessions(agent, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.base_responses_api_agent.monotonic", lambda: clock[0])
    seed = _seed()
    await agent.seed_agent_session(SimpleNamespace(session={}), seed)
    state = agent._require_agent_session(seed.agent_session_id)
    await agent.close_agent_session(SimpleNamespace(session={}), _close(_seed(1)))
    clock[0] += 86400
    await agent.close_agent_session(SimpleNamespace(session={}), _close(_seed(2)))
    assert agent._require_agent_session(seed.agent_session_id) is state
    assert _seed(1).agent_session_id not in agent._session_records


@pytest.mark.parametrize("window", [0, -1, float("inf")])
def test_retry_window_must_be_positive_and_finite(window):
    with pytest.raises(ValidationError, match="session_close_retry_window_seconds"):
        BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="", session_close_retry_window_seconds=window)
