# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response fields and failure routing for the shared outcome ladder."""

import json
from types import SimpleNamespace

import pytest

from nemo_gym.agents.agent_outcome import (
    AGENT_OUTCOME_METADATA_KEY,
    AGENT_STOP_REASON_METADATA_KEY,
    AgentOutcome,
    AgentOutcomePolicyConfig,
    OutcomeAction,
    agent_response_action,
    model_http_error_outcome,
    reported_agent_failure,
)
from nemo_gym.agents.base_responses_api_agent import ROLLOUT_STOP_METADATA_KEY
from nemo_gym.client.openai_client import MODEL_RESPONSE_TIMEOUT_REASON
from nemo_gym.core.failure_kinds import (
    AGENT_RUN_ERROR,
    MODEL_POLICY_RESPONSE_TIMEOUT,
    MODEL_REQUEST_ERROR,
    MODEL_REQUEST_REJECTED,
    MODEL_TRANSPORT_ERROR,
    PROVIDER_UNAVAILABLE,
)
from nemo_gym.core.failure_records import FailureReport
from tests.unit_tests.agents.test_outcome_ladder import _agent, _evidence, _fault, _model_error, _response


# --- finish_response ------------------------------------------------------------------------------------------- #


def test_finish_response_drops_a_rollout_stop_marker_the_ladder_did_not_decide():
    # A marker left by an earlier report_rollout_stop must not survive a run the ladder decided otherwise.
    response = _response(status="completed", metadata={ROLLOUT_STOP_METADATA_KEY: '{"reason":"budget"}', "kept": "x"})
    finished = _agent().finish_response(response, _evidence())
    assert finished.status == "completed"
    assert finished.metadata == {"kept": "x", AGENT_STOP_REASON_METADATA_KEY: "completed"}


def test_finish_response_marks_a_stopped_run_whose_session_expired():
    # evidence.stopped from a rollout stop (no deadline involved) still gains the marker with the session's reason.
    session = SimpleNamespace(expired=True, expire_reason="budget")
    finished = _agent().finish_response(
        _response(status="completed"), _evidence(stopped=True, has_result=False), session=session
    )
    assert finished.status == "incomplete"
    assert json.loads(finished.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": "budget"}
    assert AGENT_OUTCOME_METADATA_KEY not in finished.metadata


def test_finish_response_keeps_the_adapters_incomplete_details():
    response = _response(status="incomplete", incomplete_details={"reason": "content_filter"})
    finished = _agent().finish_response(response, _evidence(model_limit="output"))
    assert finished.incomplete_details.reason == "content_filter"


@pytest.mark.parametrize(
    ("evidence", "stop_reason"),
    [
        (_evidence(timed_out=True, infra_fault=_fault(), has_result=False), "infra"),
        (_evidence(timed_out=True, model_replied=False, retry_deadline_before_reply=True), "timeout"),
        (_evidence(error=_model_error(503), model_replied=False), "error"),
        (_evidence(has_result=False), "no_result"),
    ],
)
def test_finish_response_gives_every_failed_row_an_error(evidence, stop_reason):
    finished = _agent().finish_response(_response(status="completed"), evidence)
    assert finished.status == "failed"
    assert finished.metadata[AGENT_STOP_REASON_METADATA_KEY] == stop_reason
    assert finished.error.code == "server_error" and finished.error.message


@pytest.mark.parametrize("kind", [PROVIDER_UNAVAILABLE, MODEL_TRANSPORT_ERROR])
def test_an_infra_fault_with_a_status_keeps_its_kind(kind):
    # Only model_request_error carries the fault's HTTP status; any other kind drops it rather than becoming unknown.
    evidence = _evidence(timed_out=True, infra_fault=_fault(kind, status=503), has_result=False)
    from nemo_gym.agents.agent_outcome import RunDecision, decide_outcome

    assert decide_outcome(evidence) == RunDecision(
        status="failed", stop_reason="infra", outcome=AgentOutcome(failure_kind=kind)
    )


# --- model_http_error_outcome ---------------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "body",
    [
        '{"code": "context_length_exceeded"}',  # a flat object without the envelope
        "null",
        "",
        b"\x80not utf-8",
        '{"error": null}',
    ],
)
def test_model_http_error_outcome_tolerates_other_bodies(body):
    assert model_http_error_outcome(400, error_body=body) == _model_error(400)


# --- agent_response_action: the shared Environment Server rule (single_agent_turn and Harbor) ------------------ #


def _failed(outcome: AgentOutcome, report: dict | None = None, *, status: str = "failed"):
    fields = {"metadata": {AGENT_OUTCOME_METADATA_KEY: outcome.model_dump_json(exclude_none=True)}}
    if report is not None:
        fields["failure_report"] = report
    return _response(status=status, **fields)


def _selected(kind: str, **event) -> dict:
    return {"failure_id": "selected", "events": {"selected": {"event_id": "selected", "failure_kind": kind, **event}}}


def _confirmed_policy_timeout() -> dict:
    model = {
        "event_id": "model",
        "boundary": "model",
        "failure_kind": MODEL_REQUEST_ERROR,
        "failure_reason": MODEL_RESPONSE_TIMEOUT_REASON,
        "operation": "model_request",
        "operation_id": "operation",
        "attempt_id": "attempt",
        "details": {"request_upload_complete": True},
    }
    agent = {
        "event_id": "agent",
        "boundary": "agent",
        "failure_kind": MODEL_POLICY_RESPONSE_TIMEOUT,
        "operation": "agent.policy_request",
        "operation_id": "operation",
        "cause_id": "model",
    }
    return {"failure_id": "agent", "events": {"agent": agent, "model": model}}


def _enclosing_500(*, connected: bool) -> dict:
    event = {
        "event_id": "selected",
        "boundary": "model",
        "failure_kind": MODEL_REQUEST_ERROR,
        "operation": "model_request",
        "operation_id": "operation",
        "attempt_id": "attempt-1",
        "details": {"request_upload_complete": connected},
        "foreign_error": {"exception_type": "aiohttp.client_exceptions.ClientConnectorError", "message": ""},
    }
    return {"failure_id": "selected", "events": {"selected": event}}


TRANSPORT = AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)
POLICY_TIMEOUT = AgentOutcome(failure_kind=MODEL_POLICY_RESPONSE_TIMEOUT)


@pytest.mark.parametrize(
    ("response", "action", "cause_kind"),
    [
        # No report: the outcome policy alone decides.
        (_failed(TRANSPORT), OutcomeAction.RETRY, None),
        # A report that selects nothing keeps the policy action (a plain provider 400 is graded).
        (_failed(_model_error(400), {"failure_id": None, "events": {}}), OutcomeAction.VERIFY, None),
        # An omitted report is not "selects nothing": the evidence cannot be checked, so the episode fails.
        (_failed(TRANSPORT, {"failure_id": None, "events": {}, "omitted": True}), OutcomeAction.FAIL, None),
        # A selected failure matching the outcome keeps the policy action and becomes the cause.
        (
            _failed(TRANSPORT, _selected(MODEL_TRANSPORT_ERROR, boundary="model")),
            OutcomeAction.RETRY,
            MODEL_TRANSPORT_ERROR,
        ),
        # A selected failure of another kind contradicts the outcome.
        (_failed(TRANSPORT, _selected(PROVIDER_UNAVAILABLE, boundary="agent")), OutcomeAction.FAIL, None),
        # A selected status other than the outcome's contradicts it.
        (
            _failed(_model_error(503), _selected(MODEL_REQUEST_ERROR, boundary="model", http_status=500)),
            OutcomeAction.FAIL,
            None,
        ),
        # An enclosing 500 (no selected HTTP status) is retried only when no attempt ever connected.
        (_failed(_model_error(500), _enclosing_500(connected=False)), OutcomeAction.RETRY, MODEL_REQUEST_ERROR),
        (_failed(_model_error(500), _enclosing_500(connected=True)), OutcomeAction.FAIL, MODEL_REQUEST_ERROR),
        # A confirmed policy-response timeout without a timeout rule keeps the policy's action (Harbor's path).
        (_failed(POLICY_TIMEOUT, _confirmed_policy_timeout()), OutcomeAction.FAIL, MODEL_POLICY_RESPONSE_TIMEOUT),
        # A response that did not fail is graded whatever report it carries.
        (
            _failed(TRANSPORT, _selected(PROVIDER_UNAVAILABLE, boundary="agent"), status="incomplete"),
            OutcomeAction.VERIFY,
            None,
        ),
    ],
    ids=[
        "no-report",
        "unselected",
        "omitted",
        "matching",
        "contradicting-kind",
        "contradicting-status",
        "enclosing-500-never-connected",
        "enclosing-500-connected",
        "policy-timeout-without-rule",
        "not-failed",
    ],
)
def test_agent_response_action(response, action, cause_kind):
    decided, cause = agent_response_action(response, config=AgentOutcomePolicyConfig())
    assert decided == action
    assert (cause.failure_kind if cause else None) == cause_kind


def test_agent_response_action_applies_the_timeout_rule_only_to_a_confirmed_policy_timeout():
    seen = []

    def timeout_action(fallback: OutcomeAction) -> OutcomeAction:
        seen.append(fallback)
        return OutcomeAction.COUNT_AS_ZERO

    config = AgentOutcomePolicyConfig()
    action, cause = agent_response_action(
        _failed(POLICY_TIMEOUT, _confirmed_policy_timeout()), config=config, timeout_action=timeout_action
    )
    assert (action, cause.failure_kind, seen) == (
        OutcomeAction.COUNT_AS_ZERO,
        MODEL_POLICY_RESPONSE_TIMEOUT,
        [OutcomeAction.FAIL],
    )
    # Any other selected failure never consults the timeout rule.
    action, _ = agent_response_action(
        _failed(TRANSPORT, _selected(MODEL_TRANSPORT_ERROR, boundary="model")),
        config=config,
        timeout_action=timeout_action,
    )
    assert action == OutcomeAction.RETRY and seen == [OutcomeAction.FAIL]


@pytest.mark.parametrize(
    "error_body",
    [
        '{"detail":{"error":{"code":"context_length_exceeded"}}}',
        '{"error":{"type":"context_length_exceeded"}}',
        '{"error":{"code":"CONTEXT_LENGTH_EXCEEDED"}}',
        '{"error":"context_length_exceeded"}',
        '\ufeff{"error":{"code":"context_length_exceeded"}}',
    ],
)
def test_http_classification_agrees_with_the_model_client(error_body):
    assert model_http_error_outcome(400, error_body=error_body).failure_kind == "model_context_limit"


@pytest.mark.parametrize("status", [0, 302, 600, "400", None])
def test_unexpected_http_status_is_unknown(status):
    assert model_http_error_outcome(status).failure_kind == "unknown"


def test_deeply_nested_error_body_is_not_a_context_overflow():
    depth = 100_000
    assert model_http_error_outcome(400, error_body="[" * depth + "]" * depth) == _model_error(400)


@pytest.mark.parametrize(
    "evidence,status",
    [
        (_evidence(), "completed"),
        (_evidence(context_limit=True), "incomplete"),
        (_evidence(has_result=False), "failed"),
    ],
)
def test_finished_fields_agree_with_status(evidence, status):
    response = _response(
        status="failed",
        error={"code": "server_error", "message": "native error"},
        incomplete_details={"reason": "max_output_tokens"},
    )
    finished = _agent().finish_response(response, evidence)
    assert finished.status == status
    assert (finished.error is not None) is (status == "failed")
    assert (finished.incomplete_details is not None) is (status == "incomplete")
    assert finished.output == response.output and finished.usage == response.usage
    assert response.error.message == "native error"


@pytest.mark.parametrize(
    "evidence,report,kind,action",
    [
        (
            _evidence(error=_model_error(400), model_replied=False),
            _selected(MODEL_REQUEST_ERROR, boundary="model", http_status=400),
            MODEL_REQUEST_REJECTED,
            OutcomeAction.VERIFY,
        ),
        (
            _evidence(has_result=False),
            _selected(MODEL_TRANSPORT_ERROR, boundary="model"),
            AGENT_RUN_ERROR,
            OutcomeAction.RETRY,
        ),
        (
            _evidence(timed_out=True, model_replied=False, retry_deadline_before_reply=True),
            _selected(MODEL_TRANSPORT_ERROR, boundary="model"),
            "unknown",
            OutcomeAction.RETRY,
        ),
        (
            _evidence(timed_out=True, infra_fault=_fault(PROVIDER_UNAVAILABLE, status=503)),
            _selected(PROVIDER_UNAVAILABLE, boundary="sandbox", http_status=503),
            PROVIDER_UNAVAILABLE,
            OutcomeAction.RETRY,
        ),
        (
            _evidence(timed_out=True, infra_fault=_fault(MODEL_REQUEST_ERROR)),
            _selected(MODEL_REQUEST_ERROR, boundary="model"),
            "unknown",
            OutcomeAction.RETRY,
        ),
    ],
)
def test_reclassified_failure_retains_its_cause_and_follows_the_decided_policy(evidence, report, kind, action):
    response = _response(status="failed", failure_report=report)
    finished = _agent().finish_response(response, evidence)
    config = AgentOutcomePolicyConfig(by_failure_kind={MODEL_REQUEST_REJECTED: "verify"})
    decided, cause = agent_response_action(finished, config=config)
    assert decided == action
    kept = FailureReport.model_validate(finished.model_extra["failure_report"])
    if kind == "unknown":
        assert cause is None and kept.failure_id is None
    else:
        assert cause.failure_kind == kind
        selected = kept.events[kept.failure_id]
        assert selected.boundary == "agent" and selected.operation == "agent.run"
        assert selected.cause_id == report["failure_id"]
        assert selected.http_status is None
    assert kept.events[report["failure_id"]] == FailureReport.model_validate(report).events[report["failure_id"]]
    assert response.model_extra["failure_report"] == report
    assert _agent().finish_response(finished, evidence).model_extra == finished.model_extra


def test_matching_failure_is_not_replaced():
    report = _selected(MODEL_REQUEST_ERROR, boundary="model", http_status=503)
    finished = _agent().finish_response(
        _response(status="failed", failure_report=report), _evidence(error=_model_error(503))
    )
    assert finished.model_extra["failure_report"] == report
    assert reported_agent_failure(finished).http_status == 503


def test_explicit_null_report_fails_consistency_check():
    response = _failed(TRANSPORT).model_copy(update={"failure_report": None})
    assert agent_response_action(response, config=AgentOutcomePolicyConfig()) == (OutcomeAction.FAIL, None)
