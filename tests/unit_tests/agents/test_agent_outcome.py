# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponseError
from pydantic import ValidationError

from nemo_gym.agents.agent_outcome import (
    AGENT_OUTCOME_METADATA_KEY,
    POLICY_TIMEOUT_OPERATION,
    AgentOutcome,
    policy_call_failure,
)
from nemo_gym.agents.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client.openai_client import MODEL_RESPONSE_TIMEOUT_REASON
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.failure_kinds import FAILURE_KINDS, MODEL_POLICY_RESPONSE_TIMEOUT, MODEL_REQUEST_ERROR
from nemo_gym.core.failure_records import FailureBoundary, FailureEvent, FailureReport, ModelFailureDetails
from nemo_gym.core.failure_transport import FailureHeader, RequestFailureLog
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.core.typed_exceptions import rebuild_failure


def test_inherited_outcome_reporting_preserves_response_and_marks_unknown():
    class Agent(SimpleResponsesAPIAgent):
        async def responses(self, body):
            pass

        async def run(self, body):
            pass

    agent = Agent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )
    response = NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
        status="failed",
        error={"code": "server_error", "message": "Unclassified native failure"},
        metadata={"existing": "retained"},
    )
    reported = agent.report_outcome(response, {"error": "unrecognized"})
    assert reported.metadata == {"existing": "retained", AGENT_OUTCOME_METADATA_KEY: '{"failure_kind":"unknown"}'}
    assert reported.model_dump(exclude={"metadata"}) == response.model_dump(exclude={"metadata"})
    for status in ("completed", "incomplete"):
        response = response.model_copy(update={"status": status})
        assert agent.report_outcome(response, {}) is response


@pytest.mark.parametrize(
    "fields",
    [
        {"failure_kind": "model_request_error"},
        {"failure_kind": "unknown", "status_code": 503},
        {"failure_kind": "model_request_error", "status_code": True},
        {"failure_kind": "model_request_error", "status_code": 200},
    ],
)
def test_rejects_inconsistent_model_request_evidence(fields):
    with pytest.raises(ValidationError):
        AgentOutcome.model_validate(fields)


@pytest.mark.parametrize("kind", sorted(FAILURE_KINDS | {"unknown"}))
def test_accepts_shared_failure_vocabulary(kind):
    fields = {"failure_kind": kind}
    if kind == MODEL_REQUEST_ERROR:
        fields["status_code"] = 503
    outcome = AgentOutcome(**fields)
    assert AgentOutcome.model_validate_json(outcome.model_dump_json()) == outcome


@pytest.mark.parametrize(
    "kind", ["", "typo", "model_timeout_typo", "agent:custom", "unknown ", "limit", "invalid_output"]
)
def test_rejects_unregistered_failure_kinds(kind):
    import json

    fields = {"failure_kind": kind}
    with pytest.raises(ValidationError, match="Unregistered failure kind"):
        AgentOutcome(**fields)
    with pytest.raises(ValidationError, match="Unregistered failure kind"):
        AgentOutcome.model_validate_json(json.dumps(fields))


def _model_attempt(*, http_status: int | None = None, timed_out: bool = False, attempt: str = "1") -> FailureEvent:
    """One failed attempt of the model call ``call`` as a model server's client reports it."""
    return FailureEvent(
        boundary=FailureBoundary.MODEL,
        failure_kind=MODEL_REQUEST_ERROR,
        failure_reason=MODEL_RESPONSE_TIMEOUT_REASON if timed_out else "Provider answered an error",
        operation="model_request",
        operation_id="call",
        attempt_id=f"call-{attempt}",
        http_status=http_status,
        details=ModelFailureDetails(request_upload_complete=True) if timed_out else None,
    )


def _selecting(*events: FailureEvent) -> FailureReport:
    """A report of ``events`` whose last one is the selected failure."""
    return FailureReport(failure_id=events[-1].event_id, events={event.event_id: event for event in events})


def _failed_reply(status: int, report: FailureReport | None = None) -> ClientResponseError:
    """The HTTP error ``raise_for_status`` raises for a model server's failed reply that forwards ``report``."""
    request_info = SimpleNamespace(url="http://model", method="POST", headers={}, real_url="http://model")
    headers = FailureHeader.encode(report) if report is not None else None
    error = ClientResponseError(request_info, (), status=status, message="failed", headers=headers)
    # raise_for_status links the forwarded failure as the HTTP error's cause.
    error.__cause__ = rebuild_failure(FailureHeader.decode(headers))
    return error


def test_a_policy_call_failure_selects_the_forwarded_cause_with_the_reply_status():
    forwarded = _selecting(_model_attempt(http_status=503))
    classified = policy_call_failure(_failed_reply(503, forwarded))
    assert classified.outcome == AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=503)
    assert classified.timeout is None
    assert classified.report.failure_id == forwarded.failure_id
    assert classified.report.events.keys() == forwarded.events.keys()


def test_a_reply_without_a_report_is_a_request_error_that_selects_nothing():
    classified = policy_call_failure(_failed_reply(500))
    assert classified.outcome == AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=500)
    assert classified.report == FailureReport()


def test_a_confirmed_response_timeout_is_the_agents_own_failure_caused_by_the_reply():
    attempts = [_model_attempt(timed_out=True, attempt=str(n)) for n in range(2)]
    error = _failed_reply(504, _selecting(*attempts))
    classified = policy_call_failure(error)
    assert classified.outcome == AgentOutcome(failure_kind=MODEL_POLICY_RESPONSE_TIMEOUT)
    timeout = classified.timeout
    assert (timeout.boundary, timeout.failure_kind) == (FailureBoundary.AGENT, MODEL_POLICY_RESPONSE_TIMEOUT)
    assert (timeout.operation, timeout.operation_id) == (POLICY_TIMEOUT_OPERATION, "call")
    assert timeout.__cause__ is error
    assert classified.report.failure_id == timeout.event_id
    assert {attempt.event_id for attempt in attempts} <= classified.report.events.keys()


def test_the_request_logs_recovered_observations_join_the_report():
    recovered = _model_attempt(http_status=502, attempt="0")
    with RequestFailureLog.capture() as log:
        log.add(FailureReport(events={recovered.event_id: recovered}))
        forwarded = _selecting(_model_attempt(http_status=503))
        classified = policy_call_failure(_failed_reply(503, forwarded))
    assert recovered.event_id in classified.report.events
    assert classified.report.failure_id == forwarded.failure_id


# A forwarded report with a missing or repeated cause is invalid, and its header decodes as omitted.
@pytest.mark.parametrize("case", ["body-failure", "body-failure-as-cause", "omitted", "missing-cause", "cause-cycle"])
def test_a_reply_whose_selected_failure_may_hide_an_answer_is_withheld(case: str):
    body = _model_attempt(http_status=200)
    wrapper = FailureEvent(
        boundary=FailureBoundary.MODEL, failure_kind=MODEL_REQUEST_ERROR, failure_reason="wrapped", cause_id="x"
    )
    report = {
        "body-failure": _selecting(body),
        "body-failure-as-cause": _selecting(body, wrapper.model_copy(update={"cause_id": body.event_id})),
        "omitted": _selecting(_model_attempt(http_status=503)).model_copy(update={"omitted": True}),
        "missing-cause": FailureReport.model_construct(
            failure_id=wrapper.event_id, events={wrapper.event_id: wrapper}, omitted=False
        ),
        "cause-cycle": FailureReport.model_construct(
            failure_id=wrapper.event_id,
            events={wrapper.event_id: wrapper.model_copy(update={"cause_id": wrapper.event_id})},
            omitted=False,
        ),
    }[case]
    assert policy_call_failure(_failed_reply(500, report)) is None


def test_a_report_of_recovered_observations_only_is_not_withheld():
    observed = _model_attempt(http_status=200)
    classified = policy_call_failure(_failed_reply(503, FailureReport(events={observed.event_id: observed})))
    assert classified.outcome.status_code == 503
    assert classified.report.failure_id is None and observed.event_id in classified.report.events
