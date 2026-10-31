# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from nemo_gym.orchestration.recovery import _observed_trial_operations, decide_from_collected
from tests.unit_tests.test_tool_trial_coverage import fixture


def _judged(mutate):
    """The actual healthy CPU trial fixture relabelled as judge-graded; used for explicit consumer-boundary controls."""
    collected = copy.deepcopy(fixture("healthy"))
    observed = collected.results[0]["ng_trial_observations"]
    grade = observed["verifier_invocations"][0]["grading_operations"][0]
    grade["kind"] = "llm_judge"
    mutate(observed, grade)
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    return _observed_trial_operations(decision, collected)


def _judge_call(observed, *, http_attempts=1):
    call = copy.deepcopy(observed["model_calls"][0])
    meta = call["response_metadata"]
    call["model_call_id"], meta["response_id"] = "judge-call", "judge-response"
    meta["model_operations"][0].update(
        operation_id="judge-op", final_attempt_id="judge-attempt", http_attempts=http_attempts
    )
    observed["model_calls"].append(call)
    return call


def _dep(call):
    meta = call["response_metadata"]
    return {"model_call_id": None, "model_ref": dict(meta["model_ref"]), "response_id": meta["response_id"]}


def _own(observed, call):
    observed["verifier_invocations"][0]["model_calls"].append(
        {"model_call_id": call["model_call_id"], "model_ref": None, "response_id": None}
    )


def _missing_dependency(observed, grade):
    grade["dependencies"] = []


def _foreign_dependency(observed, grade):
    grade["dependencies"] = [
        {"model_call_id": None, "model_ref": {"type": "responses_api_models", "name": "x"}, "response_id": "other"}
    ]


def _duplicate_dependency(observed, grade):
    call = _judge_call(observed)
    _own(observed, call)
    grade["dependencies"] = [_dep(call), _dep(call)]


def _unowned_dependency(observed, grade):
    grade["dependencies"] = [_dep(_judge_call(observed))]


def _owned_call_not_a_dependency(observed, grade):
    call = _judge_call(observed)
    _own(observed, call)
    extra = _judge_call(observed)
    extra["model_call_id"], extra["response_metadata"]["response_id"] = "judge-call-2", "judge-response-2"
    _own(observed, extra)
    grade["dependencies"] = [_dep(call)]


def _policy_call_as_dependency(observed, grade):
    policy = observed["model_calls"][0]
    _own(observed, policy)
    grade["dependencies"] = [_dep(policy)]


def _recovered_judge_attempt(observed, grade):
    call = _judge_call(observed, http_attempts=2)
    _own(observed, call)
    grade["dependencies"] = [_dep(call)]


def _unknown_operation(observed, grade):
    call = _judge_call(observed)
    _own(observed, call)
    grade["dependencies"] = [_dep(call)]
    grade["outcome"] = "unknown"


@pytest.mark.parametrize(
    "mutate",
    [
        _missing_dependency,
        _foreign_dependency,
        _duplicate_dependency,
        _unowned_dependency,
        _owned_call_not_a_dependency,
        _policy_call_as_dependency,
        _recovered_judge_attempt,
        _unknown_operation,
    ],
)
def test_judge_graded_trials_without_exact_owned_single_attempt_dependencies_stay_unknown(mutate):
    assert _judged(mutate) is False


def test_the_programmatic_trial_fixture_still_passes():
    collected = fixture("healthy")
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    assert _observed_trial_operations(decision, collected) is True


@pytest.mark.parametrize("count", [1, 2])
def test_chat_judge_completion_uses_the_observed_chat_finish_reason(count):
    def mutate(observed, grade):
        grade["dependencies"] = []
        for number in range(count):
            call = _judge_call(observed)
            meta = call["response_metadata"]
            call["model_call_id"] = f"chat-judge-{number}"
            meta.update(
                response_id=f"chat-response-{number}", dialect="chat", response_status=None, finish_reason="stop"
            )
            meta["model_operations"][0].update(
                operation_id=f"chat-op-{number}", final_attempt_id=f"chat-attempt-{number}"
            )
            _own(observed, call)
            grade["dependencies"].append(_dep(call))

    assert _judged(mutate) is True


@pytest.mark.parametrize(
    "change",
    [
        "dialect",
        "finish_absent",
        "finish_unknown",
        "finish_length",
        "contradictory_status",
        "missing_id",
        "duplicate_attempt",
        "recovered",
    ],
)
def test_chat_judge_incomplete_or_unjoined_completion_stays_unknown(change):
    def mutate(observed, grade):
        calls = []
        for number in range(2):
            call = _judge_call(observed)
            meta = call["response_metadata"]
            call["model_call_id"] = f"chat-judge-{number}"
            meta.update(
                response_id=f"chat-response-{number}", dialect="chat", response_status=None, finish_reason="stop"
            )
            meta["model_operations"][0].update(
                operation_id=f"chat-op-{number}", final_attempt_id=f"chat-attempt-{number}"
            )
            _own(observed, call)
            calls.append(call)
        grade["dependencies"] = [_dep(call) for call in calls]
        meta = calls[1]["response_metadata"]
        if change == "dialect":
            meta["dialect"] = "responses"
        elif change == "finish_absent":
            meta["finish_reason"] = None
        elif change == "finish_unknown":
            meta["finish_reason"] = "unrecognized"
        elif change == "finish_length":
            meta["finish_reason"] = "length"
        elif change == "contradictory_status":
            meta["response_status"] = "failed"
        elif change == "missing_id":
            meta["model_operations"][0]["operation_id"] = None
        elif change == "duplicate_attempt":
            meta["model_operations"][0]["final_attempt_id"] = "chat-attempt-0"
        elif change == "recovered":
            meta["model_operations"][0]["http_attempts"] = 2

    assert _judged(mutate) is False


@pytest.mark.parametrize(
    "dialect,finish,expected",
    [("chat", "stop", ("chat", "stop")), ("private-unrecognized", "private-unrecognized", (None, None))],
)
def test_stdlib_collector_preserves_only_bounded_chat_completion_metadata(tmp_path, dialect, finish, expected):
    from tests.unit_tests.test_recovery_health_binding import collect, save_rows, setup

    artifacts, rows = setup(tmp_path, n=1)
    metadata = rows[0]["ng_trajectory"]["model_calls"][0]["response_metadata"]
    metadata.update(dialect=dialect, finish_reason=finish, response_status=None)
    save_rows(artifacts, rows)
    collected = collect(artifacts)
    projected = collected.results[0]["ng_trial_observations"]["model_calls"][0]["response_metadata"]
    assert (projected["dialect"], projected["finish_reason"]) == expected


def _recovered_fixture():
    collected = copy.deepcopy(fixture("healthy"))
    observed = collected.results[0]["ng_trial_observations"]
    grade = observed["verifier_invocations"][0]["grading_operations"][0]
    grade["kind"] = "llm_judge"
    call = _judge_call(observed)
    _own(observed, call)
    grade["dependencies"] = [_dep(call)]
    meta = call["response_metadata"]
    final = meta["model_operations"][0]
    final.update(retry_wait_s=0, request_elapsed_ms=2)
    failed = dict(
        final,
        operation_id="earlier-judge-op",
        final_attempt_id="earlier-judge-attempt",
        outcome="failed",
        failure_id="judge-error",
    )
    meta["model_operations"].insert(0, failed)
    event = dict(
        event_id="judge-error",
        boundary="model",
        operation="model_request",
        operation_id="earlier-judge-op",
        attempt_id="earlier-judge-attempt",
        cause_id=None,
        failure_kind="model_request_error",
        http_status=503,
    )
    meta["failure_report"]["events"] = {"judge-error": event}
    collected.results[0]["_ng_failure_events"] = {"judge-error": dict(event)}
    return collected, meta, grade


def test_recovered_judge_counts_are_separate_and_do_not_change_score_or_denominator():
    from nemo_gym.orchestration.recovery import trial_health

    c, _, _ = _recovered_fixture()
    before = copy.deepcopy(c.model_dump())
    d = decide_from_collected(c, max_infra_retries=1, max_walltime_s=None, live_job=False)
    counters = {}
    assert _observed_trial_operations(d, c, judge_recovery=counters) is True
    assert counters == {
        "judge_recovered_attempts": 1,
        "judge_recovered_operations": 1,
        "observed_grading_attempt_errors": 1,
        "observed_grading_failed_operations": 1,
    }
    assert trial_health(d, c)[0] is True
    assert d.expected == 1 and d.action.value == "noop"
    assert c.model_dump() == before


@pytest.mark.parametrize(
    "change",
    [
        "omitted",
        "selected",
        "missing_event",
        "orphan",
        "cycle",
        "wrong_operation",
        "wrong_attempt",
        "wrong_terminal",
        "wrong_count",
        "missing_elapsed",
        "infinite_elapsed",
        "negative_wait",
        "missing_status",
        "duplicate_operation",
        "duplicate_attempt",
        "final_failed",
        "wrong_grade",
        "missing_dependency",
        "unrelated_row_event",
    ],
)
def test_recovered_judge_missing_or_foreign_facts_cannot_accept(change):
    from nemo_gym.orchestration.recovery import trial_health

    c, meta, grade = _recovered_fixture()
    report = meta["failure_report"]
    event = report["events"]["judge-error"]
    failed, final = meta["model_operations"]
    if change == "omitted":
        report["omitted"] = True
    elif change == "selected":
        report["failure_id"] = "judge-error"
    elif change == "missing_event":
        report["events"] = {}
    elif change == "orphan":
        report["events"]["orphan"] = dict(event, event_id="orphan", operation_id="foreign")
    elif change == "cycle":
        event["cause_id"] = "judge-error"
    elif change == "wrong_operation":
        event["operation_id"] = "foreign"
    elif change == "wrong_attempt":
        event["attempt_id"] = "foreign"
    elif change == "wrong_terminal":
        failed["failure_id"] = "foreign"
    elif change == "wrong_count":
        failed["http_attempts"] = 2
    elif change == "missing_elapsed":
        failed.pop("request_elapsed_ms")
    elif change == "infinite_elapsed":
        failed["request_elapsed_ms"] = float("inf")
    elif change == "negative_wait":
        failed["retry_wait_s"] = -1
    elif change == "missing_status":
        event["http_status"] = None
    elif change == "duplicate_operation":
        final["operation_id"] = failed["operation_id"]
    elif change == "duplicate_attempt":
        final["final_attempt_id"] = failed["final_attempt_id"]
    elif change == "final_failed":
        final["outcome"] = "failed"
    elif change == "wrong_grade":
        grade["result_binding_sha256"] = "0" * 64
    elif change == "missing_dependency":
        grade["dependencies"] = []
    elif change == "unrelated_row_event":
        c.results[0]["_ng_failure_events"]["unrelated"] = dict(event, event_id="unrelated")
    d = decide_from_collected(c, max_infra_retries=1, max_walltime_s=None, live_job=False)
    counters = {}
    assert _observed_trial_operations(d, c, judge_recovery=counters) is False
    assert counters == {}
    assert trial_health(d, c)[0] is None


def test_recovered_event_id_cannot_be_reused_by_another_judge_dependency():
    c, meta, grade = _recovered_fixture()
    observed = c.results[0]["ng_trial_observations"]
    original = observed["model_calls"][-1]
    extra = copy.deepcopy(original)
    extra["model_call_id"] = "other-judge-call"
    extra_meta = extra["response_metadata"]
    extra_meta["response_id"] = "other-judge-response"
    for operation in extra_meta["model_operations"]:
        operation["operation_id"] += "-other"
        operation["final_attempt_id"] += "-other"
    event = extra_meta["failure_report"]["events"]["judge-error"]
    event["operation_id"] += "-other"
    event["attempt_id"] += "-other"
    observed["model_calls"].append(extra)
    _own(observed, extra)
    grade["dependencies"].append(_dep(extra))
    d = decide_from_collected(c, max_infra_retries=1, max_walltime_s=None, live_job=False)
    assert _observed_trial_operations(d, c) is False


@pytest.mark.parametrize(
    "dialect,status,finish",
    [
        ("responses", "completed", None),
        ("responses", "incomplete", None),
        ("chat", None, "stop"),
        ("chat", None, "length"),
    ],
)
def test_bound_returned_invalid_grading_is_a_trial_fault(dialect, status, finish):
    from nemo_gym.orchestration.recovery import trial_health

    c, meta, grade = _recovered_fixture()
    grade["outcome"] = "unknown"
    meta.update(dialect=dialect, response_status=status, finish_reason=finish)
    before = copy.deepcopy(c.model_dump())
    d = decide_from_collected(c, max_infra_retries=1, max_walltime_s=None, live_job=False)
    counters, faults = {}, []
    assert _observed_trial_operations(d, c, judge_recovery=counters, grading_failures=faults) is True
    assert len(faults) == 1
    assert counters == {
        "judge_recovered_attempts": 0,
        "judge_recovered_operations": 0,
        "observed_grading_attempt_errors": 1,
        "observed_grading_failed_operations": 1,
    }
    c.quality["observed_upstream_recovery"] = {"recovered_attempt_errors": 1, "recovered_operations": 1}
    before = copy.deepcopy(c.model_dump())
    assert trial_health(d, c)[0] is False
    assert d.expected == 1 and d.action.value == "noop"
    assert c.model_dump() == before
    grade["dependencies"] = []
    assert trial_health(d, c)[0] is None
