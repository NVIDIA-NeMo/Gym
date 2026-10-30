# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import MagicMock

from aiohttp import ClientResponseError

from nemo_gym.agent_utils.supervisor_client import harness_termination_failure
from nemo_gym.failure_transport import FailureHeader
from nemo_gym.server_runtime.episode_types import FailureBoundary
from nemo_gym.typed_exceptions import NgFailureEvent, agent_route_failure, failure_report_for, rebuild_failure
from tests.unit_tests.server_runtime import test_session_routing
from tests.unit_tests.test_agent_seed_failure_boundary import _served


def _inner_reply_error(failure: BaseException) -> ClientResponseError:
    """The error raise_for_status raises for another Gym server, with its rebuilt cause."""
    headers = FailureHeader.encode(failure_report_for(failure))
    error = ClientResponseError(MagicMock(real_url="http://inner/v1/responses"), (), status=500, headers=headers)
    error.response_content = b'{"detail":"inner failed"}'
    error.__cause__ = rebuild_failure(FailureHeader.decode(headers))
    return error


def _post_responses(monkeypatch, raised: BaseException):
    async def responses(self, body=None):
        raise raised

    monkeypatch.setattr(test_session_routing._SessionAgent, "responses", responses)
    worker = _served()
    with worker.client:
        return worker, worker.client.post("/v1/responses", json={"input": "hi"})


def _agent_events(reply):
    report = FailureHeader.decode(reply.headers)
    return report, [event for event in report.events.values() if event.boundary is FailureBoundary.AGENT]


def test_exhausted_model_calls_keep_the_inner_reply_body_and_add_one_agent_failure(monkeypatch) -> None:
    model = NgFailureEvent(
        failure_kind="test:failure",
        boundary=FailureBoundary.MODEL,
        failure_reason="model unavailable",
        operation="model.responses",
        http_status=503,
    )
    worker, reply = _post_responses(monkeypatch, _inner_reply_error(model))

    assert reply.status_code == 500
    key = worker.agent.get_session_middleware_key()
    assert reply.json() == f"Hit an exception in {key} calling an inner server: " + 'b\'{"detail":"inner failed"}\''
    report, agents = _agent_events(reply)
    assert len(agents) == 1 and agents[0].event_id == report.failure_id
    assert (agents[0].operation, agents[0].failure_kind) == ("agent.responses", "agent_route_failed")
    assert model.event.event_id in report.events


def test_a_run_that_called_this_agents_responses_is_not_wrapped_twice(monkeypatch) -> None:
    cause = RuntimeError("x")
    inner_agent = agent_route_failure(cause, operation="agent.responses")
    inner_agent.__cause__ = cause
    _, reply = _post_responses(monkeypatch, _inner_reply_error(inner_agent))

    _, agents = _agent_events(reply)
    assert [event.event_id for event in agents] == [inner_agent.event.event_id]


def test_a_generic_error_keeps_its_repr_body_and_a_harness_cause(monkeypatch) -> None:
    termination = harness_termination_failure(harness="codex", receipt={"cleanup_confirmed": False, "error": "stuck"})
    _, reply = _post_responses(monkeypatch, termination)

    assert reply.status_code == 500 and reply.json() == repr(termination)
    report, agents = _agent_events(reply)
    assert report.events[agents[0].cause_id].boundary is FailureBoundary.HARNESS


def test_cancellation_is_not_wrapped(monkeypatch) -> None:
    _, reply = _post_responses(monkeypatch, asyncio.CancelledError())

    assert reply.status_code == 500
    report = FailureHeader.decode(reply.headers)
    assert report is None or not any(event.boundary is FailureBoundary.AGENT for event in report.events.values())
