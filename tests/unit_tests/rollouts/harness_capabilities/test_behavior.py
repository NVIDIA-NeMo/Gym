# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check core behavioral comparisons without launching or importing harnesses."""

import copy
import json

import pytest

from nemo_gym.rollouts.harness_capabilities.behavior import (
    ExpectedOutcome,
    environment_action,
    inspect_behavior,
    model_checks,
    outcome_checks,
    tool_checks,
)
from tests.unit_tests.rollouts.harness_capabilities.synthetic import evidence_and_witness


def fingerprint(request: dict, status: int, response: dict) -> str:
    # The core accepts an injected comparison key; protocol normalization belongs to the runner.
    return json.dumps([request, status, response], sort_keys=True)


@pytest.fixture
def behavior_episode() -> tuple[dict, dict]:
    return evidence_and_witness()


def failed_model_checks(record: dict, attempts: list[dict]) -> set[str]:
    return {c["id"] for c in model_checks(record, attempts, fingerprint=fingerprint) if c["status"] == "fail"}


@pytest.mark.parametrize(
    "mutation", ["missing_calls", "response_id", "finish_reason", "prompt_tokens", "missing_tokens", "request_order"]
)
def test_canonical_behavioral_checks_use_the_independent_witness(behavior_episode, mutation):
    raw, witness = behavior_episode
    assert not failed_model_checks(raw, witness["attempts"])
    call = raw["ng_trajectory"]["model_calls"][1]
    if mutation == "missing_calls":
        del raw["ng_trajectory"]["model_calls"]
    elif mutation in ("response_id", "finish_reason"):
        call["response_metadata"][mutation] = "fabricated"
    elif mutation == "prompt_tokens":
        call["token_stats"]["prompt_tokens"] = 999
    elif mutation == "missing_tokens":
        del call["token_stats"]
    else:
        call["request"]["input"].reverse()
    issues = failed_model_checks(raw, witness["attempts"])
    assert issues
    assert issues <= {"model.saved_exchanges", "model.response_metadata", "model.token_counts"}


def test_canonical_error_response_cannot_invent_an_id():
    request = {"input": []}
    response = {"error": {"message": "prescribed HTTP 400"}}
    attempts = [{"request": request, "response": response, "status_code": 400}]
    metadata = {"status_code": 400, "response_id": None, "response_status": None, "finish_reason": None}
    call = {"request": request, "response": response, "response_metadata": metadata}
    record = {"ng_trajectory": {"model_calls": [call]}}
    assert not failed_model_checks(record, attempts)
    metadata["response_id"] = "invented"
    assert failed_model_checks(record, attempts) == {"model.response_metadata"}


@pytest.mark.parametrize(
    "field", ["prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cached_tokens"]
)
def test_canonical_omitted_usage_must_stay_unavailable(field):
    request = {"messages": [{"role": "user", "content": "hello"}]}
    response = {"id": "r", "object": "chat.completion", "created": 123, "choices": [{"finish_reason": "stop"}]}
    call = {
        "request": request,
        "response": copy.deepcopy(response),
        "response_metadata": {"status_code": 200, "response_id": "r", "finish_reason": "stop"},
    }
    record = {"ng_trajectory": {"model_calls": [call]}}
    attempts = [{"request": request, "response": response, "status_code": 200}]
    assert not failed_model_checks(record, attempts)
    call["token_stats"] = {field: 0}
    assert failed_model_checks(record, attempts) == {"model.token_counts"}


def test_canonical_counts_cannot_be_swapped_between_calls(behavior_episode):
    record, witness = behavior_episode
    calls = record["ng_trajectory"]["model_calls"]
    # Give the calls distinct, independently known counts.
    for i, call in enumerate(calls):
        call["token_stats"]["prompt_tokens"] = 10 + i
        witness["attempts"][i]["response"]["usage"]["input_tokens"] = 10 + i
        call["response"]["usage"]["input_tokens"] = 10 + i
    assert not failed_model_checks(record, witness["attempts"])
    calls[0]["token_stats"], calls[1]["token_stats"] = calls[1]["token_stats"], calls[0]["token_stats"]
    assert failed_model_checks(record, witness["attempts"]) == {"model.token_counts"}


def test_missing_tool_request_blocks_only_request_comparison(behavior_episode):
    record, witness = behavior_episode
    inv = record["ng_trajectory"]["invocations"][0]
    inv["conversation"] = [i for i in inv["conversation"] if i.get("call_id") != "tool-1"]
    checks = {c["id"]: c for c in tool_checks(record, witness["tool_calls"])}
    assert checks["tools.witness_join"]["status"] == "fail"
    assert checks["tools.witness_request"]["status"] == "fail"
    assert checks["tools.witness_status"]["status"] == "pass"
    assert checks["tools.witness_output"]["status"] == "pass"


def test_the_scripted_finish_call_is_not_compared_with_the_shell_witness(behavior_episode):
    record, witness = behavior_episode
    tools = record["ng_trajectory"]["tool_calls"]
    tools.append({**tools[0], "tool_call_id": "finish-1"})
    assert {c["id"]: c for c in tool_checks(record, witness["tool_calls"])}["tools.witness_ids"]["status"] == "fail"
    checks = {c["id"]: c for c in tool_checks(record, witness["tool_calls"], finish_id="finish-1")}
    assert checks["tools.witness_ids"]["status"] == "pass"


@pytest.mark.parametrize("finish_call,status", [({"id": "finish-1"}, "pass"), (None, "fail")])
def test_inspect_behavior_takes_the_finish_call_from_the_witness(behavior_episode, finish_call, status):
    record, witness = behavior_episode
    tools = record["ng_trajectory"]["tool_calls"]
    tools.append({**tools[0], "tool_call_id": "finish-1"})
    witness["finish_call"] = finish_call
    options = dict(http_errors=(), terminal_error=False, tool_steps=2, expected_reward=0.0, fingerprint=fingerprint)
    checks = {c["id"]: c for c in inspect_behavior(witness, record, **options)}
    assert checks["tools.witness_ids"]["status"] == status


@pytest.mark.parametrize(
    "mutation,failed",
    [
        (None, None),
        ("missing", "tools.witness_ids"),
        ("name", "tools.witness_request"),
        ("arguments", "tools.witness_request"),
        ("exit_code", "tools.witness_status"),
        ("outputs", "tools.witness_output"),
    ],
)
def test_additional_native_executions_require_complete_independent_witnesses(behavior_episode, mutation, failed):
    record, witness = behavior_episode
    trajectory = record["ng_trajectory"]
    original = trajectory["tool_calls"][0]
    trajectory["tool_calls"].append({**original, "tool_call_id": "native-prefill"})
    conversation = trajectory["invocations"][0]["conversation"]
    conversation.extend(
        [
            {**item, "call_id": "native-prefill"}
            for item in list(conversation)
            if item.get("call_id") == original["tool_call_id"]
        ]
    )
    native = {**witness["tool_calls"][0], "id": "native-prefill"}
    witness["native_tool_calls"] = [native]
    if mutation == "missing":
        witness["native_tool_calls"] = []
    elif mutation == "name":
        native["name"] = "wrong-tool"
    elif mutation == "arguments":
        native["arguments"] = {"code": "not executed"}
    elif mutation == "exit_code":
        native["exit_code"] = 7
    elif mutation == "outputs":
        native["outputs"] = ["not observed"]
    options = dict(http_errors=(), terminal_error=False, tool_steps=2, expected_reward=0.0, fingerprint=fingerprint)
    checks = {check["id"]: check for check in inspect_behavior(witness, record, **options)}
    assert checks["tools.executed"]["status"] == "pass"
    if failed:
        assert checks[failed]["status"] == "fail"
    else:
        assert all(check["status"] != "fail" for check in checks.values())


def _as_text_parts(outputs: list[str]) -> list[dict]:
    return [{"type": "text", "text": output} for output in outputs]


def test_tool_output_given_as_text_parts_matches_the_same_text(behavior_episode):
    record, witness = behavior_episode
    for call in witness["tool_calls"]:
        call["outputs"] = [_as_text_parts(call["outputs"])]
    assert {c["id"]: c for c in tool_checks(record, witness["tool_calls"])}["tools.witness_output"]["status"] == "pass"
    for call in record["ng_trajectory"]["tool_calls"]:
        call["output"] = _as_text_parts(["val", "ue"])
    assert {c["id"]: c for c in tool_checks(record, witness["tool_calls"])}["tools.witness_output"]["status"] == "pass"


def test_different_text_fails_the_output_check_in_any_form(behavior_episode):
    record, witness = behavior_episode
    witness["tool_calls"][0]["outputs"] = [_as_text_parts(["different"])]
    checks = {c["id"]: c for c in tool_checks(record, witness["tool_calls"])}
    assert checks["tools.witness_output"]["status"] == "fail"


def test_missing_output_witness_cannot_qualify_artifacts(behavior_episode):
    record, witness = behavior_episode
    options = dict(http_errors=(), terminal_error=False, tool_steps=2, expected_reward=0.0, fingerprint=fingerprint)
    assert all(c["status"] != "fail" for c in inspect_behavior(witness, record, **options))
    del witness["tool_calls"][0]["outputs"]
    checks = {c["id"]: c for c in inspect_behavior(witness, record, **options)}
    assert checks["tools.witness_output"]["status"] == "fail"
    assert checks["tools.witness_output"]["reasons"] == [
        "retained tool output differs from the independent tool witness"
    ]


@pytest.mark.parametrize("missing", [None, "judge_call", "grade_reference", "verifier_owner"])
def test_judge_scenario_requires_witnessed_grade_and_verifier_ownership(missing):
    from copy import deepcopy

    from scripts.harness_conformance.runner import _fingerprint

    from nemo_gym.rollouts.harness_capabilities.behavior import inspect_behavior
    from tests.unit_tests.rollouts.harness_capabilities.synthetic import evidence_and_witness

    record, witness = evidence_and_witness()
    call = deepcopy(record["ng_trajectory"]["model_calls"][0])
    call["model_call_id"] = "judge-call"
    call["response_metadata"]["response_id"] = "judge-response"
    record["ng_trajectory"]["model_calls"].append(call)
    owner = {"invocation_id": "judge", "model_calls": [{"model_call_id": "judge-call"}]}
    record["ng_trajectory"]["verifier_invocations"] = [owner]
    record["judgement_id"] = "judge-response"
    witness["judge_calls"] = [
        {"request": {"input": "CONFORMANCE_DONE"}, "response": {"id": "judge-response"}, "status_code": 200}
    ]
    if missing == "judge_call":
        witness["judge_calls"] = []
    elif missing == "grade_reference":
        record.pop("judgement_id")
    elif missing == "verifier_owner":
        record["ng_trajectory"]["verifier_invocations"] = []
        record["ng_trajectory"]["invocations"].append(owner)
    result = inspect_behavior(
        witness,
        record,
        http_errors=(),
        terminal_error=False,
        tool_steps=2,
        expected_reward=1.0,
        fingerprint=_fingerprint,
        judge_verifier=True,
    )
    check = next(c for c in result if c["id"] == "verifier.judge_reference")
    assert check["tier"] == "P1"
    assert check["status"] == ("pass" if missing is None else "fail")


@pytest.mark.parametrize(
    "status, outcome",
    [
        # Every request failed, so the model never replied: a non-retryable 4xx is the run's rejected request.
        *((status, {"failure_kind": "model_request_rejected"}) for status in (400, 401, 403, 404, 413, 422)),
        *(
            (status, {"failure_kind": "model_request_error", "status_code": status})
            for status in (408, 425, 429, 500, 503)
        ),
    ],
)
def test_terminal_error_expects_the_outcome_ladders_report(status, outcome):
    from nemo_gym.rollouts.harness_capabilities.behavior import _terminal_error_reported

    attempts = [{"status_code": status, "request": {}, "response": {"error": {"message": "controlled failure"}}}]

    def response(reported: dict) -> list[dict]:
        return [
            {
                "status": "failed",
                "error": {"message": f"HTTP {status}: controlled failure"},
                "metadata": {"ng_agent_outcome": json.dumps(reported)},
            }
        ]

    assert _terminal_error_reported(attempts, response(outcome), status)
    other = (
        {"failure_kind": "model_request_error", "status_code": status}
        if outcome["failure_kind"] == "model_request_rejected"
        else {"failure_kind": "model_request_rejected"}
    )
    assert not _terminal_error_reported(attempts, response(other), status)


def _response(status: str, outcome: dict | None = None) -> dict:
    metadata = {} if outcome is None else {"ng_agent_outcome": json.dumps(outcome)}
    return {"status": status, "error": None, "metadata": metadata}


@pytest.mark.parametrize(
    "expected, response, matches",
    [
        (ExpectedOutcome("incomplete", "verify"), _response("incomplete"), True),
        # The ladder removes ng_agent_outcome from a response it does not fail.
        (
            ExpectedOutcome("incomplete", "verify"),
            _response("incomplete", {"failure_kind": "model_context_limit"}),
            False,
        ),
        (ExpectedOutcome("incomplete", "verify"), _response("completed"), False),
        (
            ExpectedOutcome("failed", "verify", "model_request_error", 400),
            _response("failed", {"failure_kind": "model_request_error", "status_code": 400}),
            True,
        ),
        (
            ExpectedOutcome("failed", "verify", "model_request_error", 400),
            _response("failed", {"failure_kind": "model_request_error", "status_code": 502}),
            False,
        ),
        (
            ExpectedOutcome("failed", "retry", "agent_run_error"),
            _response("failed", {"failure_kind": "unknown"}),
            False,
        ),
        (ExpectedOutcome("failed", "retry", "agent_run_error"), _response("failed"), False),
        (
            ExpectedOutcome("failed", "retry", "agent_run_error"),
            {"status": "failed", "metadata": {"ng_agent_outcome": "{"}},
            False,
        ),
    ],
)
def test_expected_outcome_matches_status_and_exact_agent_outcome(expected, response, matches):
    assert expected.matches(response) is matches


@pytest.mark.parametrize(
    "arguments",
    [
        ("failed", "retry"),  # A failed outcome names its kind.
        ("incomplete", "verify", "unknown"),  # Only a failed response carries an outcome.
        ("failed", "verify", "model_request_error"),  # model_request_error needs the upstream status.
        ("failed", "fail", "invented_kind"),
    ],
)
def test_expected_outcome_rejects_an_outcome_no_agent_could_report(arguments):
    with pytest.raises(ValueError):
        ExpectedOutcome(*arguments)


@pytest.mark.parametrize(
    "record, action",
    [
        (None, None),
        ({"reward": 0.0, "evaluation_completed": True}, "verify"),
        ({"_ng_failure_class": "environment_server_failed", "_ng_failure_terminal": False}, "retry"),
        ({"_ng_failure_class": "environment_server_failed"}, "retry"),
        ({"_ng_failure_class": "environment_server_failed", "_ng_failure_terminal": True}, "fail"),
        (
            {
                "_ng_failure_class": "model_policy_response_timeout",
                "_ng_failure_terminal": True,
                "_ng_failure_counts_as_zero": True,
            },
            "count_as_zero",
        ),
    ],
)
def test_environment_action_reads_the_record_collection_kept(record, action):
    assert environment_action(record) == action


RETRIED = ExpectedOutcome("failed", "retry", "agent_run_error")
RETRY_ROW = {"_ng_failure_class": "environment_server_failed", "_ng_failure_terminal": False}


def _outcome_witness(*, http_status: int = 200, response: dict | None = None) -> dict:
    return {
        "attempts": [{"received_at": 100.0}],
        "agent_replies": [{"http_status": http_status, "replied_at": 101.0}],
        "agent_responses": [] if response is None else [response],
    }


def _statuses(checks: list[dict]) -> dict[str, str]:
    return {c["id"]: c["status"] for c in checks}


def test_outcome_checks_pass_for_the_ladders_response_and_action():
    witness = _outcome_witness(response=_response("failed", {"failure_kind": "agent_run_error"}))
    assert _statuses(outcome_checks(witness, RETRY_ROW, outcomes=(RETRIED,))) == {
        "outcome.http_ok": "pass",
        "outcome.ladder": "pass",
        "outcome.disposition": "pass",
        "outcome.target": "not_applicable",
        "model.retries_bounded": "not_applicable",
    }


def test_a_5xx_reply_fails_the_http_and_ladder_checks():
    # A bare 502 reaches the Environment Server without a response, so there is no outcome to match.
    statuses = _statuses(outcome_checks(_outcome_witness(http_status=502), RETRY_ROW, outcomes=(RETRIED,)))
    assert statuses["outcome.http_ok"] == "fail"
    assert statuses["outcome.ladder"] == "fail"
    assert statuses["outcome.disposition"] == "fail"


def test_a_wrong_failure_kind_fails_only_the_ladder_check():
    witness = _outcome_witness(response=_response("failed", {"failure_kind": "unknown"}))
    statuses = _statuses(outcome_checks(witness, RETRY_ROW, outcomes=(RETRIED,)))
    assert (statuses["outcome.http_ok"], statuses["outcome.ladder"]) == ("pass", "fail")


@pytest.mark.parametrize("terminal, status", [(False, "pass"), (True, "fail")])
def test_disposition_requires_the_matched_outcomes_terminal_flag(terminal, status):
    witness = _outcome_witness(response=_response("failed", {"failure_kind": "agent_run_error"}))
    record = {**RETRY_ROW, "_ng_failure_terminal": terminal}
    assert _statuses(outcome_checks(witness, record, outcomes=(RETRIED,)))["outcome.disposition"] == status


def test_disposition_follows_the_outcome_the_response_matched():
    outcomes = (ExpectedOutcome("incomplete", "verify"), ExpectedOutcome("failed", "retry", "unknown"))
    graded = {"reward": 0.0, "evaluation_completed": True}
    incomplete = _outcome_witness(response=_response("incomplete"))
    unknown = _outcome_witness(response=_response("failed", {"failure_kind": "unknown"}))
    assert _statuses(outcome_checks(incomplete, graded, outcomes=outcomes))["outcome.disposition"] == "pass"
    assert _statuses(outcome_checks(incomplete, RETRY_ROW, outcomes=outcomes))["outcome.disposition"] == "fail"
    assert _statuses(outcome_checks(unknown, RETRY_ROW, outcomes=outcomes))["outcome.disposition"] == "pass"
    assert _statuses(outcome_checks(unknown, graded, outcomes=outcomes))["outcome.disposition"] == "fail"


def test_target_outcome_is_a_separate_p1_check():
    outcomes = (ExpectedOutcome("incomplete", "verify"), ExpectedOutcome("failed", "retry", "unknown"))
    witness = _outcome_witness(response=_response("failed", {"failure_kind": "unknown"}))
    checks = {c["id"]: c for c in outcome_checks(witness, RETRY_ROW, outcomes=outcomes, target_outcome=outcomes[0])}
    assert checks["outcome.ladder"]["status"] == "pass"
    assert (checks["outcome.target"]["tier"], checks["outcome.target"]["status"]) == ("P1", "fail")


@pytest.mark.parametrize("replied_at, status", [(129.0, "pass"), (131.0, "fail")])
def test_retries_must_end_within_the_budget_from_the_first_request(replied_at, status):
    response = _response("failed", {"failure_kind": "model_request_error", "status_code": 503})
    witness = _outcome_witness(response=response)
    witness["attempts"] = [{"received_at": 100.0}, {"received_at": 110.0}]
    witness["agent_replies"] = [{"http_status": 200, "replied_at": replied_at}]
    outcomes = (ExpectedOutcome("failed", "retry", "model_request_error", 503),)
    checks = _statuses(outcome_checks(witness, RETRY_ROW, outcomes=outcomes, retry_budget_seconds=30))
    assert checks["model.retries_bounded"] == status


def test_scenarios_without_outcomes_skip_the_outcome_checks(behavior_episode):
    record, witness = behavior_episode
    options = dict(http_errors=(), terminal_error=False, tool_steps=2, expected_reward=0.0, fingerprint=fingerprint)
    checks = _statuses(inspect_behavior(witness, record, **options))
    assert {checks[key] for key in ("outcome.http_ok", "outcome.ladder", "outcome.disposition")} == {"not_applicable"}


KILLED_CHECKS = ("model.finished", "tools.executed", "tools.witness_ids", "tools.witness_output", "verifier.answer")
VERIFIER_CHECKS = ("verifier.outcome", "verifier.saved_reward", "verifier.saved_outcome")


def test_an_ungraded_outcome_drops_verifier_and_killed_tool_checks(behavior_episode):
    graded, witness = behavior_episode
    observed = _outcome_witness(response=_response("failed", {"failure_kind": "agent_run_error"}))
    witness.update(agent_replies=observed["agent_replies"], agent_responses=observed["agent_responses"])
    witness.update(finished=False, verifications=[], tool_calls=witness["tool_calls"][:1])
    witness["tool_calls"][0].update(result_seen=False, outputs=[])
    options = dict(
        http_errors=(),
        terminal_error=False,
        tool_steps=1,
        expected_reward=0.0,
        fingerprint=fingerprint,
        final_answer=False,
        tool_results=False,
        outcomes=(RETRIED,),
    )
    checks = _statuses(inspect_behavior(witness, RETRY_ROW, **options))
    assert {key: checks[key] for key in KILLED_CHECKS + VERIFIER_CHECKS} == dict.fromkeys(
        KILLED_CHECKS + VERIFIER_CHECKS, "not_applicable"
    )
    assert checks["outcome.disposition"] == "pass"
    # Graded instead of retried: verification evidence applies again, and the action check fails.
    checks = _statuses(inspect_behavior(witness, graded, **options))
    assert checks["verifier.outcome"] == "fail"  # The witness saw no verification.
    assert {checks[key] for key in KILLED_CHECKS} == {"not_applicable"}
    assert checks["outcome.disposition"] == "fail"


def test_an_incomplete_responses_reply_is_compared_by_its_incomplete_reason():
    request = {"input": [{"role": "user", "content": "hello"}]}
    response = {"id": "r", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    attempts = [{"request": request, "response": response, "status_code": 200}]
    metadata = {"status_code": 200, "response_id": "r", "response_status": "incomplete", "finish_reason": None}
    call = {"request": request, "response": response, "response_metadata": metadata}
    record = {"ng_trajectory": {"model_calls": [call]}}
    assert failed_model_checks(record, attempts) == {"model.response_metadata"}
    # Capture records the Responses stop reason as the call's finish reason.
    metadata["finish_reason"] = "max_output_tokens"
    assert not failed_model_checks(record, attempts)
