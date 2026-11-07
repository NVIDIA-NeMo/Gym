# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failing agent route keeps the kind of its cause from the policy call to the row Collection saves."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiohttp import ClientResponseError

from nemo_gym.client.openai_client import MODEL_RESPONSE_TIMEOUT_REASON
from nemo_gym.core.failure_kinds import MODEL_REQUEST_ERROR
from nemo_gym.core.failure_records import (
    FailureBoundary,
    FailureEvent,
    FailureReport,
    ModelFailureDetails,
    SdkErrorDetails,
)
from nemo_gym.core.failure_transport import FailureHeader
from nemo_gym.core.typed_exceptions import NgFailureEvent, failure_report_for, rebuild_failure
from nemo_gym.rollouts.rollout_collection import _agent_request_failure_row, _failure_outcome
from tests.unit_tests.agents.test_agent_route_failure_boundary import _agent_events, _post_responses


def _model_attempts(*, timed_out: bool) -> FailureReport:
    """The policy model server's reply report: three failed attempts of one call, ending on the last."""
    operation_id = uuid4().hex
    attempts = [
        FailureEvent(
            boundary=FailureBoundary.MODEL,
            failure_kind=MODEL_REQUEST_ERROR,
            failure_reason=MODEL_RESPONSE_TIMEOUT_REASON if timed_out else "HTTP 500 from the model server",
            operation="model_request",
            operation_id=operation_id,
            attempt_id=uuid4().hex,
            http_status=None if timed_out else 500,
            details=ModelFailureDetails(request_upload_complete=True if timed_out else None),
            foreign_error=SdkErrorDetails(exception_type="TimeoutError", message="") if timed_out else None,
        )
        for _ in range(3)
    ]
    return FailureReport(failure_id=attempts[-1].event_id, events={event.event_id: event for event in attempts})


def _run_error(headers) -> ClientResponseError:
    return ClientResponseError(
        SimpleNamespace(url="http://agent/run", method="POST", headers={}, real_url="http://agent/run"),
        (),
        status=500,
        message="Internal Server Error",
        headers=headers,
    )


def _model_reply_error(report: FailureReport) -> ClientResponseError:
    """The error raise_for_status raises for the policy model server's failed reply, with its rebuilt cause."""
    headers = FailureHeader.encode(report)
    error = _run_error(headers)
    error.response_content = b'{"detail":"inner failed"}'
    error.__cause__ = rebuild_failure(FailureHeader.decode(headers))
    return error


def _route_failure_headers(cause: BaseException | None = None):
    """The header of a failed /run reply: the agent route failure, caused by ``cause`` when given."""
    route = NgFailureEvent(
        boundary=FailureBoundary.AGENT,
        failure_kind="agent_route_failed",
        failure_reason="Agent operation agent.run failed: RuntimeError",
        operation="agent.run",
    )
    route.__cause__ = cause
    return FailureHeader.encode(failure_report_for(route))


# Hop 1: the agent's /v1/responses, which a custom /run calls for the policy.


def test_a_confirmed_policy_timeout_leaves_the_responses_route_typed(monkeypatch) -> None:
    _, reply = _post_responses(monkeypatch, _model_reply_error(_model_attempts(timed_out=True)))

    assert reply.status_code == 500
    report, agents = _agent_events(reply)
    assert [(e.operation, e.failure_kind) for e in agents] == [
        ("agent.policy_request", "model_policy_response_timeout")
    ]
    assert agents[0].event_id == report.failure_id
    assert report.events[agents[0].cause_id].boundary is FailureBoundary.MODEL


def test_an_unconfirmed_model_failure_stays_a_route_failure(monkeypatch) -> None:
    _, reply = _post_responses(monkeypatch, _model_reply_error(_model_attempts(timed_out=False)))

    _, agents = _agent_events(reply)
    assert [(e.operation, e.failure_kind) for e in agents] == [("agent.responses", "agent_route_failed")]


# Hop 2: Collection turns the agent's failed /run reply into the row it saves.


def test_a_policy_timeout_in_a_custom_run_gives_a_typed_row(monkeypatch) -> None:
    _, reply = _post_responses(monkeypatch, _model_reply_error(_model_attempts(timed_out=True)))

    row = _agent_request_failure_row(_run_error(reply.headers), 500, reply.headers)

    assert row["_ng_failure_kind"] == "model_policy_response_timeout"
    saved = _failure_outcome({"_ng_run_id": "r", "_ng_task_index": 0, "_ng_rollout_index": 0}, row, "agent")
    assert saved.failure.failure_kind == saved.sidecar_failure_class == "model_policy_response_timeout"


def test_a_verifier_error_in_a_custom_run_gives_a_typed_row() -> None:
    verifier = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind="verifier_error",
        failure_reason="Verification failed: CritPtRateLimitExceeded",
        operation="verify",
        stage="verification",
        http_status=500,
    )
    headers = _route_failure_headers(verifier)

    row = _agent_request_failure_row(_run_error(headers), 500, headers)

    assert row["_ng_failure_kind"] == "verifier_error"


@pytest.mark.parametrize(
    ("status", "headers"),
    [
        pytest.param(500, None, id="no-report"),
        pytest.param(502, _route_failure_headers(RuntimeError("x")), id="the-server-did-not-run"),
        pytest.param(500, _route_failure_headers(RuntimeError("x")), id="routing-only-chain"),
    ],
)
def test_a_reply_that_names_no_cause_keeps_the_untyped_class(status, headers) -> None:
    row = _agent_request_failure_row(_run_error(headers), status, headers)

    assert "_ng_failure_kind" not in row
