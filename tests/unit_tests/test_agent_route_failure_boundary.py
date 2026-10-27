# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import MagicMock

from aiohttp import ClientResponseError

from nemo_gym.agent_utils.supervisor_client import HarnessTerminationError
from nemo_gym.typed_exceptions import (
    FAILURE_EVENT_HEADER,
    AgentFailure,
    ModelFailure,
    boundary_failure,
    decode_header,
    encode_header,
    remote_failure,
)
from tests.unit_tests import test_session_routing
from tests.unit_tests.test_agent_seed_failure_boundary import _served


def _inner_reply_error(failure: BaseException, *, server: str) -> ClientResponseError:
    """The error ``raise_for_status`` raises for a failed reply from another Gym server, with its rebuilt cause."""
    headers = {FAILURE_EVENT_HEADER: encode_header(boundary_failure(failure, server=server))}
    error = ClientResponseError(MagicMock(real_url="http://inner/v1/responses"), (), status=500, headers=headers)
    error.response_content = b'{"detail":"inner failed"}'
    error.__cause__ = remote_failure(headers)
    return error


def _post_responses(monkeypatch, raised: BaseException):
    async def responses(self, body=None):
        raise raised

    monkeypatch.setattr(test_session_routing._SessionAgent, "responses", responses)
    worker = _served()
    with worker.client:
        return worker, worker.client.post("/v1/responses", json={"input": "hi"})


def _agent_events(reply):
    failure = decode_header(reply.headers)
    return failure, [e for e in failure.failure_events.values() if e.type == "AgentFailure"]


def test_exhausted_model_calls_keep_the_inner_reply_body_and_add_one_agent_failure(monkeypatch) -> None:
    model = ModelFailure(message="model unavailable", operation="model.responses", http_status=503)
    worker, reply = _post_responses(monkeypatch, _inner_reply_error(model, server="model"))

    assert reply.status_code == 500
    key = worker.agent.get_session_middleware_key()
    assert reply.json() == f"Hit an exception in {key} calling an inner server: " + 'b\'{"detail":"inner failed"}\''
    failure, agents = _agent_events(reply)
    assert len(agents) == 1 and agents[0].event_id == failure.failure_id
    assert (agents[0].operation, agents[0].failure_kind) == ("agent.responses", None)
    assert model.event_id in failure.failure_events  # The model's own event, with its id, stays in the chain.


def test_a_run_that_called_this_agents_responses_is_not_wrapped_twice(monkeypatch) -> None:
    from nemo_gym.typed_exceptions import agent_boundary_failure  # Only this test needs the new helper.

    inner_agent = agent_boundary_failure(RuntimeError("x"), operation="agent.responses")
    try:
        raise inner_agent from RuntimeError("x")
    except AgentFailure as error:
        reply_error = _inner_reply_error(error, server="agent")
    worker, reply = _post_responses(monkeypatch, reply_error)

    _, agents = _agent_events(reply)
    assert [a.event_id for a in agents] == [inner_agent.event_id]


def test_a_generic_error_keeps_its_repr_body_and_a_harness_cause(monkeypatch) -> None:
    termination = HarnessTerminationError(harness="codex", receipt={"cleanup_confirmed": False, "error": "stuck"})
    _, reply = _post_responses(monkeypatch, termination)

    assert reply.status_code == 500 and reply.json() == repr(termination)
    failure, agents = _agent_events(reply)
    assert failure.failure_events[agents[0].cause_id].type == "HarnessFailure"


def test_cancellation_is_not_wrapped(monkeypatch) -> None:
    _, reply = _post_responses(monkeypatch, asyncio.CancelledError())

    # Starlette answers an endpoint cancellation itself (its body is not ours); the boundary adds nothing.
    assert reply.status_code == 500
    failure = decode_header(reply.headers)
    assert failure is None or not any(e.type == "AgentFailure" for e in (failure.failure_events or {}).values())
