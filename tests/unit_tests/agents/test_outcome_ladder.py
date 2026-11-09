# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.agents.agent_outcome import (
    AGENT_OUTCOME_METADATA_KEY,
    AGENT_STOP_REASON_METADATA_KEY,
    AgentOutcome,
    AgentOutcomePolicyConfig,
    OutcomeAction,
    RunDecision,
    RunEvidence,
    agent_outcome_policy,
    decide_outcome,
)
from nemo_gym.agents.base_responses_api_agent import (
    ROLLOUT_STOP_METADATA_KEY,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.failure_kinds import (
    AGENT_INVALID_OUTPUT,
    AGENT_RUN_ERROR,
    MODEL_CONTEXT_LIMIT,
    MODEL_REQUEST_ERROR,
    MODEL_REQUEST_REJECTED,
    MODEL_TRANSPORT_ERROR,
    PROVIDER_UNAVAILABLE,
)
from nemo_gym.core.failure_records import FailureBoundary
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.core.typed_exceptions import NgFailureEvent


def _evidence(**fields) -> RunEvidence:
    return RunEvidence(**{"has_result": True, "timed_out": False, "model_replied": True, **fields})


def _model_error(status: int) -> AgentOutcome:
    return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status)


def _fault(kind: str = MODEL_TRANSPORT_ERROR, *, status: int | None = None) -> NgFailureEvent:
    return NgFailureEvent(
        boundary=FailureBoundary.MODEL, failure_kind=kind, failure_reason="recorded fault", http_status=status
    )


REJECTED = AgentOutcome(failure_kind=MODEL_REQUEST_REJECTED)
RUN_ERROR = AgentOutcome(failure_kind=AGENT_RUN_ERROR)
UNKNOWN = AgentOutcome(failure_kind="unknown")


# One case per ladder row, then the cases that show which row wins when several match.
# (evidence, decision, the default policy's action on the decided response)
LADDER = {
    # 1. A rollout stop is graded like a timeout.
    "stopped": (_evidence(stopped=True, timed_out=True), RunDecision("incomplete", "stopped"), OutcomeAction.VERIFY),
    # 2. A context limit is graded with the partial work (decision 1).
    "context_limit": (
        _evidence(context_limit=True, has_result=False),
        RunDecision("incomplete", "context_limit"),
        OutcomeAction.VERIFY,
    ),
    "context_limit_classified": (
        _evidence(error=AgentOutcome(failure_kind=MODEL_CONTEXT_LIMIT), has_result=False),
        RunDecision("incomplete", "context_limit"),
        OutcomeAction.VERIFY,
    ),
    # 3. A deadline spent on our servers is retried.
    "infra": (
        _evidence(timed_out=True, infra_fault=_fault(), has_result=False),
        RunDecision("failed", "infra", AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)),
        OutcomeAction.RETRY,
    ),
    "infra_status": (
        _evidence(timed_out=True, infra_fault=_fault(MODEL_REQUEST_ERROR, status=503), model_replied=False),
        RunDecision("failed", "infra", _model_error(503)),
        OutcomeAction.RETRY,
    ),
    "infra_unnameable": (
        _evidence(timed_out=True, infra_fault=_fault(MODEL_REQUEST_ERROR), model_replied=False),
        RunDecision("failed", "infra", UNKNOWN),
        OutcomeAction.RETRY,
    ),
    # 4. A deadline is graded with what the run kept.
    "timeout_after_reply": (_evidence(timed_out=True), RunDecision("incomplete", "timeout"), OutcomeAction.VERIFY),
    "timeout_before_reply": (
        _evidence(timed_out=True, model_replied=False, has_result=False),
        RunDecision("incomplete", "timeout"),
        OutcomeAction.VERIFY,
    ),
    # Until the deadline flip, an adapter that retried a deadline before any reply keeps doing so.
    "timeout_before_reply_kept_retried": (
        _evidence(timed_out=True, model_replied=False, retry_deadline_before_reply=True),
        RunDecision("failed", "timeout", UNKNOWN),
        OutcomeAction.RETRY,
    ),
    "timeout_unknown_reply_kept_retried": (
        _evidence(timed_out=True, model_replied=None, retry_deadline_before_reply=True),
        RunDecision("failed", "timeout", UNKNOWN),
        OutcomeAction.RETRY,
    ),
    "timeout_after_reply_not_retried": (
        _evidence(timed_out=True, model_replied=True, retry_deadline_before_reply=True),
        RunDecision("incomplete", "timeout"),
        OutcomeAction.VERIFY,
    ),
    # 5. An error the run ended on: the classified outcome, by the status table.
    "error_400_after_reply": (
        _evidence(error=_model_error(400)),
        RunDecision("failed", "error", _model_error(400)),
        OutcomeAction.VERIFY,
    ),
    "error_401_after_reply": (
        _evidence(error=_model_error(401)),
        RunDecision("failed", "error", _model_error(401)),
        OutcomeAction.FAIL,
    ),
    "error_429_before_reply": (
        _evidence(error=_model_error(429), model_replied=False),
        RunDecision("failed", "error", _model_error(429)),
        OutcomeAction.RETRY,
    ),
    "error_503_before_reply": (
        _evidence(error=_model_error(503), model_replied=False),
        RunDecision("failed", "error", _model_error(503)),
        OutcomeAction.RETRY,
    ),
    "error_transport": (
        _evidence(error=AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR), model_replied=False),
        RunDecision("failed", "error", AgentOutcome(failure_kind=MODEL_TRANSPORT_ERROR)),
        OutcomeAction.RETRY,
    ),
    "error_harness": (
        _evidence(error=AgentOutcome(failure_kind=AGENT_INVALID_OUTPUT)),
        RunDecision("failed", "error", AgentOutcome(failure_kind=AGENT_INVALID_OUTPUT)),
        OutcomeAction.VERIFY,
    ),
    # A 4xx before any reply is the run's setup, for every harness (decision 4).
    **{
        f"error_{status}_before_reply": (
            _evidence(error=_model_error(status), model_replied=False, has_result=False),
            RunDecision("failed", "error", REJECTED),
            OutcomeAction.FAIL,
        )
        for status in (400, 401, 403, 404, 413, 422)
    },
    # Without evidence that no reply arrived, the classified status decides.
    "error_400_reply_unknown": (
        _evidence(error=_model_error(400), model_replied=None),
        RunDecision("failed", "error", _model_error(400)),
        OutcomeAction.VERIFY,
    ),
    # 6. A model or harness limit is graded with the partial work.
    "output_limit": (_evidence(model_limit="output"), RunDecision("incomplete", "output_limit"), OutcomeAction.VERIFY),
    "turn_limit": (_evidence(model_limit="turns"), RunDecision("incomplete", "agent_limit"), OutcomeAction.VERIFY),
    # 7. A refusal is the model's answer (decision 5).
    "refusal": (_evidence(refusal=True, has_result=False), RunDecision("completed", "refusal"), OutcomeAction.VERIFY),
    # 8. A usable result.
    "completed": (_evidence(), RunDecision("completed", "completed"), OutcomeAction.VERIFY),
    # 9. No usable result: retried by default (decision 3).
    "no_result": (_evidence(has_result=False), RunDecision("failed", "no_result", RUN_ERROR), OutcomeAction.RETRY),
    "no_result_before_reply": (
        _evidence(has_result=False, model_replied=False),
        RunDecision("failed", "no_result", RUN_ERROR),
        OutcomeAction.RETRY,
    ),
    # Precedence: the first matching row wins.
    "stopped_over_context_limit": (
        _evidence(stopped=True, context_limit=True, timed_out=True, infra_fault=_fault()),
        RunDecision("incomplete", "stopped"),
        OutcomeAction.VERIFY,
    ),
    "context_limit_over_infra_timeout": (
        _evidence(context_limit=True, timed_out=True, infra_fault=_fault()),
        RunDecision("incomplete", "context_limit"),
        OutcomeAction.VERIFY,
    ),
    "infra_over_error": (
        _evidence(timed_out=True, infra_fault=_fault(PROVIDER_UNAVAILABLE), error=_model_error(400)),
        RunDecision("failed", "infra", AgentOutcome(failure_kind=PROVIDER_UNAVAILABLE)),
        OutcomeAction.RETRY,
    ),
    "timeout_over_error": (
        _evidence(timed_out=True, error=_model_error(401), model_replied=False),
        RunDecision("incomplete", "timeout"),
        OutcomeAction.VERIFY,
    ),
    "infra_ignored_without_deadline": (
        _evidence(infra_fault=_fault()),
        RunDecision("completed", "completed"),
        OutcomeAction.VERIFY,
    ),
    "error_over_limit": (
        _evidence(error=_model_error(500), model_limit="output"),
        RunDecision("failed", "error", _model_error(500)),
        OutcomeAction.RETRY,
    ),
    "limit_over_refusal": (
        _evidence(model_limit="turns", refusal=True),
        RunDecision("incomplete", "agent_limit"),
        OutcomeAction.VERIFY,
    ),
}


def _response(**fields) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
        **fields,
    )


def _agent() -> SimpleResponsesAPIAgent:
    class Agent(SimpleResponsesAPIAgent):
        async def responses(self, body):
            pass

    return Agent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.parametrize("case", list(LADDER))
def test_ladder_row_and_default_action(case):
    evidence, expected, action = LADDER[case]
    decision = decide_outcome(evidence)
    assert decision == expected
    # An outcome is written exactly on a failed response.
    assert (decision.outcome is not None) is (decision.status == "failed")
    # The Environment Server's default policy acts on the response the ladder decided.
    response = _agent().finish_response(_response(status="completed"), evidence)
    assert agent_outcome_policy(response, config=AgentOutcomePolicyConfig()) == action


def test_finish_response_sets_status_and_outcome_and_keeps_the_adapters_fields():
    response = _response(
        status="completed",
        error={"code": "server_error", "message": "native 503 body"},
        metadata={"existing": "kept", AGENT_OUTCOME_METADATA_KEY: '{"failure_kind":"model_context_limit"}'},
    )
    finished = _agent().finish_response(response, _evidence(error=_model_error(503), model_replied=False))
    assert finished.status == "failed"
    assert finished.error.message == "native 503 body"
    assert finished.metadata == {
        "existing": "kept",
        AGENT_STOP_REASON_METADATA_KEY: "error",
        AGENT_OUTCOME_METADATA_KEY: '{"failure_kind":"model_request_error","status_code":503}',
    }
    assert finished.model_dump(exclude={"status", "metadata"}) == response.model_dump(exclude={"status", "metadata"})


def test_finish_response_drops_a_stale_outcome_from_a_graded_response():
    # Hermes and Droid used to leave model_context_limit on their incomplete responses.
    response = _response(
        status="failed", metadata={AGENT_OUTCOME_METADATA_KEY: '{"failure_kind":"model_context_limit"}'}
    )
    finished = _agent().finish_response(response, _evidence(context_limit=True))
    assert finished.status == "incomplete"
    assert finished.metadata == {AGENT_STOP_REASON_METADATA_KEY: "context_limit"}
    assert agent_outcome_policy(finished) == OutcomeAction.VERIFY


def test_finish_response_names_a_failed_run_without_an_error():
    finished = _agent().finish_response(_response(status="completed"), _evidence(has_result=False))
    assert finished.status == "failed"
    assert finished.error.message == "The harness produced no usable result"
    assert json.loads(finished.metadata[AGENT_OUTCOME_METADATA_KEY]) == {"failure_kind": AGENT_RUN_ERROR}


def test_finish_response_marks_an_output_limit():
    finished = _agent().finish_response(_response(status="completed"), _evidence(model_limit="output"))
    assert finished.status == "incomplete"
    assert finished.incomplete_details.reason == "max_output_tokens"


@pytest.mark.parametrize(
    "expired, timed_out, status, marked",
    [
        # A deadline that a rollout stop caused is a stop, as report_rollout_stop marks it.
        (True, True, "incomplete", True),
        # A run that finished on its own before the stop took effect keeps its result.
        (True, False, "completed", False),
        # Without a stop, a deadline is a timeout.
        (False, True, "incomplete", False),
    ],
)
def test_finish_response_marks_a_rollout_stop(expired, timed_out, status, marked):
    session = SimpleNamespace(expired=expired, expire_reason="budget" if expired else None)
    finished = _agent().finish_response(_response(status="completed"), _evidence(timed_out=timed_out), session=session)
    assert finished.status == status
    assert (ROLLOUT_STOP_METADATA_KEY in finished.metadata) is marked
    if marked:
        assert json.loads(finished.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": "budget"}
        assert finished.metadata[AGENT_STOP_REASON_METADATA_KEY] == "stopped"
        assert AGENT_OUTCOME_METADATA_KEY not in finished.metadata


def test_rejected_request_is_never_graded_even_with_a_400_override():
    # The rejection carries no HTTP status, so a status rule cannot grade it as the model's answer.
    config = AgentOutcomePolicyConfig(by_status_code={400: "verify"}, default_action="retry")
    response = _agent().finish_response(
        _response(status="completed"), _evidence(error=_model_error(400), model_replied=False)
    )
    assert agent_outcome_policy(response, config=config) == OutcomeAction.FAIL


def test_harbor_style_override_grades_no_result():
    # A deployment that wants Harbor's behavior grades a run without a usable result.
    config = AgentOutcomePolicyConfig(by_failure_kind={AGENT_RUN_ERROR: "verify"})
    response = _agent().finish_response(_response(status="completed"), _evidence(has_result=False))
    assert agent_outcome_policy(response, config=config) == OutcomeAction.VERIFY
    # The other defaults survive the partial override.
    assert config.by_failure_kind[MODEL_REQUEST_REJECTED] == OutcomeAction.FAIL
    assert config.by_failure_kind[MODEL_CONTEXT_LIMIT] == OutcomeAction.VERIFY
