# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, PrivateAttr

from nemo_gym.agent_activation_types import AgentContinuationCapabilities, AgentContinuationRequirements
from nemo_gym.agent_utils.ordered_operations import ActivationConflictError, OrderedOperationLedger
from nemo_gym.base_responses_api_agent import (
    AGENT_ACTIVATION_HEADER,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


def response(text="done"):
    return NeMoGymResponse(
        id="reply",
        created_at=0,
        model="test",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
        metadata={"text": text},
    )


class Agent(SimpleResponsesAPIAgent):
    _inputs: list[str] = PrivateAttr(default_factory=list)
    _entered: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _release: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _cancelled: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _close_count: int = PrivateAttr(default=0)
    _cleanup_ok: bool = PrivateAttr(default=True)

    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body):
        raise NotImplementedError

    def _agent_continuation_capabilities(self):
        return AgentContinuationCapabilities()

    def _continuation_close_timeout(self):
        return 1.0

    def _validate_agent_activation(self, state, body):
        if not body.input:
            raise HTTPException(422, "Empty delta")

    async def _seed_agent_session_state(self, body):
        return AgentSessionState(request=body)

    async def _activate_agent_session_state(self, state, body, *, activation_id):
        text = body.input
        self._inputs.append(text)
        self._entered.set()
        try:
            await self._release.wait()
        except asyncio.CancelledError:
            self._cancelled.set()
            raise
        return response(text)

    async def _close_agent_session_state(self, state):
        self._close_count += 1
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, cleanup_confirmed=self._cleanup_ok
        )


@pytest.fixture
def agent():
    return Agent(
        config=BaseResponsesAPIAgentConfig(host="localhost", port=1, entrypoint="app.py", name="test"),
        server_client=MagicMock(spec=ServerClient),
    )


async def seed(agent, *, session_id="session", continuation=True):
    body = AgentSeedSessionRequest(
        agent_session_id=session_id,
        episode_id={"rollout_id": session_id},
        task_id={"taskset": "test", "task_id": "task"},
        continuation=AgentContinuationRequirements() if continuation else None,
    )
    request = SimpleNamespace(session={}, headers={}, path_params={})
    await agent.seed_agent_session(request, body)
    return request, body


async def activate(agent, request, index=1, text="initial"):
    request = SimpleNamespace(
        session=request.session,
        headers={AGENT_ACTIVATION_HEADER: str(index)},
        path_params=request.path_params,
    )
    return await agent.activate_agent_session(request, NeMoGymResponseCreateParamsNonStreaming(input=text))


def close(body):
    return AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)


async def test_two_turns_replay_disconnect_and_immutable_results(agent):
    request, body = await seed(agent)
    first = asyncio.create_task(activate(agent, request))
    await agent._entered.wait()
    joined = asyncio.create_task(activate(agent, request))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not agent._cancelled.is_set()
    with pytest.raises(HTTPException, match="different input"):
        await activate(agent, request, text="conflict")
    with pytest.raises(HTTPException, match="in flight"):
        await activate(agent, request, 2, "later")
    agent._release.set()
    result = await joined
    result.metadata["text"] = "mutated result"
    assert (await activate(agent, request)).metadata["text"] == "initial"
    await activate(agent, request, 2, "correction")
    assert agent._inputs == ["initial", "correction"]
    receipt = await agent.close_agent_session(request, close(body))
    assert receipt.cleanup_confirmed
    assert await agent.close_agent_session(request, close(body)) == receipt
    assert agent._close_count == 1
    with pytest.raises(HTTPException, match="closing"):
        await activate(agent, request, 3)


async def test_validation_does_not_consume_index(agent):
    request, body = await seed(agent)
    with pytest.raises(HTTPException) as failure:
        await activate(agent, request, text="")
    assert failure.value.status_code == 422
    agent._release.set()
    await activate(agent, request, text="valid")
    assert agent._inputs == ["valid"]


async def test_close_cancels_active_turn_and_fences(agent):
    request, body = await seed(agent)
    active = asyncio.create_task(activate(agent, request))
    await agent._entered.wait()
    receipt = await agent.close_agent_session(request, close(body))
    assert agent._cancelled.is_set() and receipt.cleanup_confirmed
    with pytest.raises(asyncio.CancelledError):
        await active
    with pytest.raises(HTTPException):
        await activate(agent, request)


async def test_failed_cleanup_blocks_receipt_and_can_retry(agent):
    request, body = await seed(agent)
    agent._cleanup_ok = False
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent.close_agent_session(request, close(body))
    agent._cleanup_ok = True
    assert (await agent.close_agent_session(request, close(body))).cleanup_confirmed
    assert agent._close_count == 2


@pytest.mark.parametrize("continuation", [False, True])
async def test_only_native_continuation_shields_close_waiter(agent, monkeypatch, continuation):
    request, body = await seed(agent, continuation=continuation)
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def cleanup(state):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return AgentCloseSessionResponse(agent_session_id=body.agent_session_id, cleanup_confirmed=True)

    monkeypatch.setattr(agent, "_close_agent_session_state", cleanup)
    waiter = asyncio.create_task(agent.close_agent_session(request, close(body)))
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert cancelled.is_set() is (not continuation)
    release.set()
    assert (await agent.close_agent_session(request, close(body))).cleanup_confirmed


async def test_requires_cookie_continuation_and_contiguous_header(agent):
    request, _ = await seed(agent, continuation=False)
    with pytest.raises(HTTPException, match="not seeded"):
        await activate(agent, request)
    request, _ = await seed(agent, session_id="native")
    with pytest.raises(HTTPException, match="out of order"):
        await activate(agent, request, 2)
    with pytest.raises(HTTPException, match="requires x-ng-agent-activation"):
        await agent.activate_agent_session(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    request.path_params = {"rollout_id": "wrong"}
    with pytest.raises(HTTPException, match="rollout"):
        await activate(agent, request)
    assert agent._inputs == []


async def test_unsupported_capabilities_fail_before_setup(agent, monkeypatch):
    monkeypatch.setattr(agent, "_agent_continuation_capabilities", lambda: None)
    with pytest.raises(HTTPException, match="does not support"):
        await seed(agent)
    assert agent._session_records == {}
    monkeypatch.setattr(
        agent, "_agent_continuation_capabilities", lambda: AgentContinuationCapabilities(observations=[])
    )
    with pytest.raises(HTTPException, match="lacks required"):
        await seed(agent)
    assert agent._session_records == {}


async def test_failed_operation_replays_and_blocks_cursor():
    ledger = OrderedOperationLedger[NeMoGymResponse]()
    calls = 0
    params = NeMoGymResponseCreateParamsNonStreaming(input="task")

    async def fail():
        nonlocal calls
        calls += 1
        raise RuntimeError("harness unavailable")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="harness unavailable"):
            await ledger.execute(index=1, request=params, operation=fail)
    with pytest.raises(ActivationConflictError, match="Previous operation failed"):
        await ledger.execute(index=2, request=params, operation=fail)
    assert calls == 1
    await ledger.close(timeout=1)


async def test_close_timeout_keeps_fence_until_remote_stop_finishes():
    ledger = OrderedOperationLedger[NeMoGymResponse]()
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_stop():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            raise

    params = NeMoGymResponseCreateParamsNonStreaming(input="task")
    waiter = asyncio.create_task(ledger.execute(index=1, request=params, operation=delayed_stop))
    await entered.wait()
    with pytest.raises(TimeoutError):
        await ledger.close(timeout=0.01)
    with pytest.raises(ActivationConflictError, match="closing"):
        await ledger.execute(index=2, request=params, operation=delayed_stop)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await ledger.close(timeout=1)


class CustomRequest(BaseModel):
    task_index: int


class CustomSchemaAgent(Agent):
    def _agent_continuation_capabilities(self):
        return None

    async def responses(self, req: CustomRequest):
        return {"task_index": req.task_index}


def test_existing_schema_route_and_close_payload_are_unchanged(agent):
    custom = CustomSchemaAgent(config=agent.config, server_client=agent.server_client)
    with TestClient(custom.setup_webserver()) as client:
        assert client.post("/v1/responses", json={"task_index": 7}).json() == {"task_index": 7}
    receipt = AgentCloseSessionResponse(agent_session_id="legacy")
    assert receipt.model_dump(mode="json") == {
        "agent_session_id": "legacy",
        "agent_observations": None,
        "resources_cookies": None,
    }
    assert not any(path.endswith("/activate") for path in agent.setup_webserver().openapi()["paths"])
