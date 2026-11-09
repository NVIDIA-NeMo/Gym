# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode's run evidence and the outcome ladder's decision for each run event of a sandbox session."""

import json

import pytest
from fastapi.testclient import TestClient

from nemo_gym.agents import OutcomeAction, RunEvidence, agent_outcome_policy
from nemo_gym.core import NeMoGymResponse
from responses_api_agents.opencode_agent.continuation import classify_opencode_run
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import (  # noqa: F401
    capture_observations,
    close_body,
    events,
    rejected_events,
    seed,
    setup,
)


CLEAN = {"return_code": 0, "error": None, "timed_out": False, "cleanup_confirmed": True}
TIMED_OUT = {"return_code": -9, "error": None, "timed_out": True, "cleanup_confirmed": True}


def classify(export: dict, cleanup: dict, *, continuation: bool = False):
    return classify_opencode_run(
        export, cleanup=cleanup, previous_session_id=None, session_id="s", stdout="", continuation=continuation
    )


def test_a_completed_run_is_a_usable_result_after_a_model_reply():
    assert classify(json.loads(events()), CLEAN).evidence == RunEvidence(
        has_result=True, timed_out=False, model_replied=True
    )


def test_a_deadline_before_any_assistant_message_keeps_its_retry():
    export = {"messages": json.loads(events())["messages"][:1]}
    outcome = classify(export, TIMED_OUT)
    assert outcome.evidence == RunEvidence(
        has_result=False, timed_out=True, model_replied=False, retry_deadline_before_reply=True
    )
    # A continuation's interrupted checkpoint can resume, so its deadline is graded as before.
    assert classify(export, TIMED_OUT, continuation=True).evidence.retry_deadline_before_reply is False


def test_a_model_error_on_the_first_request_is_no_reply():
    """An assistant turn that ended on an error is not a model reply (the idea PR #5798 proposed for OpenCode)."""
    outcome = classify(json.loads(rejected_events()), CLEAN | {"return_code": 1})
    assert outcome.evidence.model_replied is False
    assert outcome.recorded_error["error"]["data"]["statusCode"] == 400
    assert classify(json.loads(rejected_events(after_work=True)), CLEAN | {"return_code": 1}).evidence.model_replied


@pytest.mark.parametrize(
    "finish,limit,reason",
    [
        ("length", "output", "max_output_tokens"),
        ("content-filter", "output", "content_filter"),
        ("tool-calls", "turns", None),
    ],
)
def test_limits_are_incomplete_with_their_reason(finish, limit, reason):
    export = json.loads(events())
    export["messages"][-1]["info"]["finish"] = finish
    outcome = classify(export, CLEAN)
    assert (outcome.evidence.model_limit, outcome.evidence.has_result, outcome.incomplete_reason) == (
        limit,
        False,
        reason,
    )


def with_error(export: str, status: int) -> str:
    rejected = json.loads(export)
    rejected["messages"][-1]["info"]["error"]["data"]["statusCode"] = status
    return json.dumps(rejected)


def placeholder() -> str:
    """OpenCode saved the assistant message of a model call that had not answered yet."""
    export = json.loads(events())
    export["messages"][-1] = {"info": {"role": "assistant", "time": {"created": 100}}, "parts": []}
    return json.dumps(export)


# event -> (persisted export, receipt changes, status, ng_agent_outcome, Environment Server action). Before the
# outcome ladder the deadline-before-reply row and every agent_run_error row answered 502 (retried), as did a deadline
# with any recorded error; the first-request 400 and 401 were model_request_error (graded and failed).
EVENTS = {
    "context_overflow": (
        "overflow",
        {"return_code": 1},
        "incomplete",
        None,
        OutcomeAction.VERIFY,
    ),
    "deadline_before_reply": (
        json.dumps({"messages": json.loads(events())["messages"][:1]}),
        TIMED_OUT,
        "failed",
        {"failure_kind": "unknown"},
        OutcomeAction.RETRY,
    ),
    "deadline_during_first_model_call": (placeholder(), TIMED_OUT, "incomplete", None, OutcomeAction.VERIFY),
    "deadline_after_reply": (events(), TIMED_OUT, "incomplete", None, OutcomeAction.VERIFY),
    "deadline_after_reply_endpoint_503": (
        with_error(rejected_events(after_work=True), 503),
        TIMED_OUT,
        "failed",
        {"failure_kind": "model_request_error", "status_code": 503},
        OutcomeAction.RETRY,
    ),
    "deadline_after_reply_rejected_400": (
        rejected_events(after_work=True),
        TIMED_OUT,
        "incomplete",
        None,
        OutcomeAction.VERIFY,
    ),
    "no_result": (events(), {"return_code": 2}, "failed", {"failure_kind": "agent_run_error"}, OutcomeAction.RETRY),
    "persistent_503_after_reply": (
        with_error(rejected_events(after_work=True), 503),
        {"return_code": 1},
        "failed",
        {"failure_kind": "model_request_error", "status_code": 503},
        OutcomeAction.RETRY,
    ),
    "401_before_reply": (
        with_error(rejected_events(), 401),
        {"return_code": 1},
        "failed",
        {"failure_kind": "model_request_rejected"},
        OutcomeAction.FAIL,
    ),
    "400_after_reply": (
        rejected_events(after_work=True),
        {"return_code": 1},
        "failed",
        {"failure_kind": "model_request_error", "status_code": 400},
        OutcomeAction.VERIFY,
    ),
}


@pytest.mark.parametrize("event", EVENTS)
def test_each_run_event_gets_the_ladder_outcome(setup, tmp_path, event):  # noqa: F811
    agent, sandbox = setup
    export, receipt, status, outcome, action = EVENTS[event]
    if export == "overflow":
        overflow = json.loads(events())
        overflow["messages"][-1]["info"].update(
            finish="error", error={"name": "ContextOverflowError", "data": {"message": "context is too large"}}
        )
        export = json.dumps(overflow)
    sandbox.events = export
    sandbox.result.update(receipt)
    capture_observations(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
    assert result.status_code == 200, result.text
    response = NeMoGymResponse.model_validate(result.json())
    assert response.status == status
    assert json.loads(response.metadata.get("ng_agent_outcome", "null")) == outcome
    assert agent_outcome_policy(response) == action
    assert (response.error is not None) is (status == "failed")
    invocation = next(r for r in closed.json()["agent_observations"]["records"] if r["kind"] == "agent_invocation")
    assert invocation["status"] == status
