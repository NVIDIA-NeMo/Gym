# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import PrivateAttr

from nemo_gym.agent_runtime_policy import AgentRuntimePolicy
from nemo_gym.agent_utils.ordered_operations import OrderedOperationLedger
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.interactive_agent_types import (
    AgentActivationObservation,
    AgentActivationRequest,
    AgentActivationResponse,
    AgentContinuationCapabilities,
    AgentContinuationRequirements,
    InteractionBudget,
    ResourcesStepResponse,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def response(text: str = "done") -> NeMoGymResponse:
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

    async def _seed_agent_session_state(self, body):
        return AgentSessionState(request=body)

    async def _activate_agent_session_state(self, state, body, request):
        self._inputs.append(body.responses_create_params.input)
        self._entered.set()
        try:
            await self._release.wait()
        except asyncio.CancelledError:
            self._cancelled.set()
            raise
        return AgentActivationResponse(activation_id=body.activation_id, response=response(self._inputs[-1]))

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


async def seed(agent, *, session_id="session", continuation=True, runtime_policy=None, requires_budget=False):
    body = AgentSeedSessionRequest(
        agent_session_id=session_id,
        episode_id={"rollout_id": session_id},
        task_id={"taskset": "test", "task_id": "task"},
        continuation=AgentContinuationRequirements(requires_interaction_budget=requires_budget)
        if continuation
        else None,
        runtime_policy=runtime_policy,
    )
    request = SimpleNamespace(session={})
    await agent.seed_agent_session(request, body)
    return request, body


async def test_unsupported_runtime_policy_rejected_before_setup(agent, monkeypatch):
    async def setup(_):
        raise AssertionError("Unsupported policy must not create runtime state")

    monkeypatch.setattr(agent, "_seed_agent_session_state", setup)
    with pytest.raises(HTTPException, match="required runtime policy format") as failure:
        await seed(agent, runtime_policy={"format": "unknown.v1", "settings": {}})
    assert failure.value.status_code == 422


async def test_required_interaction_budget_capability_is_checked_before_setup(agent):
    with pytest.raises(HTTPException, match="required interaction budget") as failure:
        await seed(agent, requires_budget=True)
    assert failure.value.status_code == 422 and not agent._session_records


async def test_interaction_budget_is_immutable_across_activations_without_poisoning_rejected_ids(agent, monkeypatch):
    monkeypatch.setattr(
        agent,
        "_agent_continuation_capabilities",
        lambda: AgentContinuationCapabilities(supports_interaction_budget=True),
    )
    request, body = await seed(agent, requires_budget=True)
    budget = InteractionBudget(started_at_unix_seconds=100, deadline_unix_seconds=160)
    params = activation(body)
    with pytest.raises(HTTPException, match="requires the seeded interaction budget"):
        await agent.activate_agent_session(request, params)
    params.interaction_budget = budget
    invalid = params.model_copy(update={"activation_id": 3})
    with pytest.raises(HTTPException, match="out of order"):
        await agent.activate_agent_session(request, invalid)
    assert not agent._session_records[body.agent_session_id].interaction_budget_bound
    agent._release.set()
    original = await agent.activate_agent_session(request, params)
    assert await agent.activate_agent_session(request, params) == original
    next_activation = activation(body, index=1, text="next")
    next_activation.interaction_budget = InteractionBudget(started_at_unix_seconds=100, deadline_unix_seconds=170)
    with pytest.raises(HTTPException, match="already bound") as failure:
        await agent.activate_agent_session(request, next_activation)
    assert failure.value.status_code == 409
    next_activation.interaction_budget = budget
    await agent.activate_agent_session(request, next_activation)
    assert agent._inputs == ["initial", "next"]
    assert (await agent.close_agent_session(request, close(body))).cleanup_confirmed


async def test_optional_interaction_budget_requires_adapter_support_when_supplied(agent):
    request, body = await seed(agent)
    params = activation(body)
    params.interaction_budget = InteractionBudget(started_at_unix_seconds=100, deadline_unix_seconds=160)
    with pytest.raises(HTTPException, match="does not support interaction budgets"):
        await agent.activate_agent_session(request, params)
    assert not agent._inputs


async def test_adapter_validates_policy_and_seed_retry_cannot_change_it(agent, monkeypatch):
    def validate(policy: AgentRuntimePolicy) -> None:
        if policy.format != "fixture.v1" or policy.settings.keys() != {"mode"}:
            raise HTTPException(422, "Unsupported fixture policy")

    monkeypatch.setattr(agent, "_validate_agent_runtime_policy", validate)
    request, body = await seed(agent, runtime_policy={"format": "fixture.v1", "settings": {"mode": "native"}})
    assert (await agent.seed_agent_session(request, body)).agent_session_id == body.agent_session_id
    conflicting = body.model_copy(deep=True)
    conflicting.runtime_policy.settings["mode"] = "changed"
    with pytest.raises(HTTPException, match="another seed request"):
        await agent.seed_agent_session(request, conflicting)
    assert agent._require_agent_session(body.agent_session_id).request.runtime_policy.settings == {"mode": "native"}
    with pytest.raises(HTTPException, match="Unsupported fixture policy"):
        await seed(agent, session_id="bad", runtime_policy={"format": "fixture.v1", "settings": {"unknown": True}})


def activation(body, index=0, text="initial"):
    return AgentActivationRequest(
        agent_session_id=body.agent_session_id,
        episode_id=body.episode_id,
        activation_id=index,
        responses_create_params={"input": text},
    )


def close(body):
    return AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)


async def test_two_turns_join_replay_disconnect_and_immutable_inputs(agent):
    request, body = await seed(agent)
    params = activation(body)
    first = asyncio.create_task(agent.activate_agent_session(request, params))
    await agent._entered.wait()
    joined = asyncio.create_task(agent.activate_agent_session(request, params))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not agent._cancelled.is_set()
    with pytest.raises(HTTPException, match="different input"):
        await agent.activate_agent_session(request, activation(body, text="conflict"))
    with pytest.raises(HTTPException, match="in flight"):
        await agent.activate_agent_session(request, activation(body, 1, "later"))
    params.responses_create_params.input = "mutated by caller"
    agent._release.set()
    result = await joined
    result.response.metadata["text"] = "mutated result"
    replay = await agent.activate_agent_session(request, activation(body))
    assert replay.response.metadata["text"] == "initial"
    second = await agent.activate_agent_session(request, activation(body, 1, "correction"))
    assert second.activation_id == 1
    assert agent._inputs == ["initial", "correction"]
    receipt = await agent.close_agent_session(request, close(body))
    assert receipt.cleanup_confirmed
    assert [item.activation_id for item in receipt.activations] == [0, 1]
    assert await agent.close_agent_session(request, close(body)) == receipt
    assert agent._close_count == 1
    with pytest.raises(HTTPException, match="closing"):
        await agent.activate_agent_session(request, activation(body, 2, "stale"))


async def test_close_racing_with_active_turn_cancels_awaits_and_fences(agent):
    request, body = await seed(agent)
    active = asyncio.create_task(agent.activate_agent_session(request, activation(body)))
    await agent._entered.wait()
    receipt = await agent.close_agent_session(request, close(body))
    assert agent._cancelled.is_set()
    assert receipt.cleanup_confirmed and receipt.activations == []
    with pytest.raises(asyncio.CancelledError):
        await active
    with pytest.raises(HTTPException):
        await agent.activate_agent_session(request, activation(body))


async def test_failed_cleanup_blocks_receipt_and_can_retry(agent):
    request, body = await seed(agent)
    agent._cleanup_ok = False
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent.close_agent_session(request, close(body))
    with pytest.raises(HTTPException, match="closing"):
        await agent.activate_agent_session(request, activation(body))
    agent._cleanup_ok = True
    receipt = await agent.close_agent_session(request, close(body))
    assert receipt.cleanup_confirmed and agent._close_count == 2


async def test_close_waiter_disconnect_does_not_cancel_cleanup(agent, monkeypatch):
    request, body = await seed(agent)
    entered, release = asyncio.Event(), asyncio.Event()

    async def cleanup(state):
        entered.set()
        await release.wait()
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id, cleanup_confirmed=True)

    monkeypatch.setattr(agent, "_close_agent_session_state", cleanup)
    waiter = asyncio.create_task(agent.close_agent_session(request, close(body)))
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    receipt = await agent.close_agent_session(request, close(body))
    assert receipt.cleanup_confirmed


async def test_requires_continuation_cookie_identity_and_contiguous_ids(agent):
    request, body = await seed(agent, continuation=False)
    with pytest.raises(HTTPException, match="not seeded"):
        await agent.activate_agent_session(request, activation(body))
    request, body = await seed(agent, session_id="interactive")
    with pytest.raises(HTTPException, match="cookie"):
        await agent.activate_agent_session(SimpleNamespace(session={}), activation(body))
    with pytest.raises(HTTPException, match="out of order"):
        await agent.activate_agent_session(request, activation(body, 1))
    wrong = activation(body).model_copy(
        update={"episode_id": body.episode_id.model_copy(update={"rollout_id": "wrong"})}
    )
    with pytest.raises(HTTPException, match="episode_id"):
        await agent.activate_agent_session(request, wrong)
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


async def test_failed_step_replays_failure_and_blocks_cursor_advancement():
    ledger = OrderedOperationLedger[ResourcesStepResponse]()
    calls = 0
    request = AgentContinuationRequirements()

    async def fail():
        nonlocal calls
        calls += 1
        raise RuntimeError("simulator unavailable")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="simulator unavailable"):
            await ledger.execute(index=0, request=request, operation=fail)
    with pytest.raises(HTTPException, match="Previous operation failed"):
        await ledger.execute(index=1, request=request, operation=fail)
    assert calls == 1
    await ledger.close(timeout=1)


async def test_session_isolation_while_first_session_in_flight(agent):
    one, seed_one = await seed(agent, session_id="one")
    two, seed_two = await seed(agent, session_id="two")
    first = asyncio.create_task(agent.activate_agent_session(one, activation(seed_one)))
    await agent._entered.wait()
    second = asyncio.create_task(agent.activate_agent_session(two, activation(seed_two, text="independent")))
    await asyncio.sleep(0)
    agent._release.set()
    await asyncio.gather(first, second)
    assert agent._inputs == ["initial", "independent"]


def test_observations_preserve_order_and_text_distinction():
    from pydantic import ValidationError

    observation = AgentActivationObservation(
        events=[
            {"sequence": 1, "kind": "text", "text": "visible"},
            {"sequence": 2, "kind": "reasoning", "text": "private"},
            {"sequence": 3, "kind": "tool_use", "name": "read", "result": "contents"},
        ]
    )
    assert observation.events[1].kind == "reasoning"
    with pytest.raises(ValidationError, match="strictly increasing"):
        AgentActivationObservation(events=list(reversed(observation.events)))


async def test_close_timeout_keeps_fence_until_cancelled_hook_finishes():
    ledger = OrderedOperationLedger[ResourcesStepResponse]()
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_remote_stop():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            raise

    waiter = asyncio.create_task(
        ledger.execute(index=0, request=AgentContinuationRequirements(), operation=delayed_remote_stop)
    )
    await entered.wait()
    with pytest.raises(TimeoutError):
        await ledger.close(timeout=0.01)
    with pytest.raises(HTTPException, match="closing"):
        await ledger.execute(index=1, request=AgentContinuationRequirements(), operation=delayed_remote_stop)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await ledger.close(timeout=1)
