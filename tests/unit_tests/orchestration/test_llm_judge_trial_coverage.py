# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from nemo_gym.orchestration.recovery import _observed_trial_operations, decide_from_collected
from tests.unit_tests.orchestration.test_tool_trial_coverage import fixture


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
    from tests.unit_tests.orchestration.test_recovery_health_binding import collect, save_rows, setup

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
    from nemo_gym.orchestration.recovery import gate_summary_from_collected, trial_counts, trial_health

    c, meta, grade = _recovered_fixture()
    grade["outcome"] = "unknown"
    meta.update(dialect=dialect, response_status=status, finish_reason=finish)
    before = copy.deepcopy(c.model_dump())
    d = decide_from_collected(c, max_infra_retries=1, max_walltime_s=None, live_job=False)
    counters, faults = {}, []
    assert _observed_trial_operations(d, c, judge_recovery=counters, grading_failures=faults) is True
    assert faults == []  # the row is counted once, as its own judge_unparseable fault
    assert counters == {
        "judge_recovered_attempts": 0,
        "judge_recovered_operations": 0,
        "observed_grading_attempt_errors": 1,
        "observed_grading_failed_operations": 1,
    }
    c.quality["observed_upstream_recovery"] = {"recovered_attempt_errors": 1, "recovered_operations": 1}
    before = copy.deepcopy(c.model_dump())
    assert trial_health(d, c)[0] is False
    # The detailed counts follow the gate: the grading without a valid verdict is unhealthy, never healthy.
    expected_counts = {"observed_healthy": 0, "unhealthy": 1, "unobserved": 0, "verifier_recorded": 0}
    assert trial_counts(d, c) == expected_counts
    task = gate_summary_from_collected(d, c, max_unhealthy_fraction=None)
    assert task["trial_counts"] == expected_counts
    assert task["evaluation_health"]["evaluation"]["kinds"] == {"judge_unparseable": 1}
    assert task["failures"]["trial_failures"]["by_cause"]["judge"] == {"count": 1, "rate": 1.0}
    assert d.expected == 1 and d.action.value == "reverify"  # the unparsed grading is planned for a re-judge
    assert c.model_dump() == before
    # The unknown outcome alone counts the trial (a known lower bound over the limit), whatever joins its calls.
    grade["dependencies"] = []
    assert trial_health(d, c)[0] is False


def _transport_retried_judge(*, transport=1, event=True):
    """A judge request whose first connection failed without an HTTP status and was retried inside the same
    operation, which then completed: one HTTP attempt, one transport failure, one recorded event."""
    collected = copy.deepcopy(fixture("healthy"))
    observed = collected.results[0]["ng_trial_observations"]
    grade = observed["verifier_invocations"][0]["grading_operations"][0]
    grade["kind"] = "llm_judge"
    call = _judge_call(observed)
    _own(observed, call)
    grade["dependencies"] = [_dep(call)]
    meta = call["response_metadata"]
    meta["model_operations"][0].update(transport_failures=transport, retry_wait_s=0, request_elapsed_ms=2)
    if event:
        failure = dict(
            event_id="judge-transport",
            boundary="model",
            operation="model_request",
            operation_id="judge-op",
            attempt_id="judge-op-transport-1",
            cause_id=None,
            failure_kind="model_request_error",
            http_status=None,
        )
        meta["failure_report"]["events"] = {"judge-transport": failure}
        collected.results[0]["_ng_failure_events"] = {"judge-transport": dict(failure)}
    return collected


def test_a_judge_request_retried_after_a_transport_failure_is_recovered():
    from nemo_gym.orchestration.recovery import trial_health

    c = _transport_retried_judge()
    d = decide_from_collected(c, max_infra_retries=1, max_walltime_s=None, live_job=False)
    counters = {}
    assert _observed_trial_operations(d, c, judge_recovery=counters) is True
    assert counters["observed_grading_attempt_errors"] == 1 and counters["judge_recovered_attempts"] == 1
    assert trial_health(d, c)[0] is True


@pytest.mark.parametrize(("transport", "event"), [(0, True), (1, False)], ids=["uncounted-event", "missing-event"])
def test_a_transport_failure_must_be_counted_and_recorded(transport, event):
    c = _transport_retried_judge(transport=transport, event=event)
    d = decide_from_collected(c, max_infra_retries=1, max_walltime_s=None, live_job=False)
    assert _observed_trial_operations(d, c) is False


@pytest.mark.parametrize(
    "dialect,status,finish",
    [("responses", "incomplete", None), ("chat", None, "length"), ("chat", None, "content_filter")],
)
def test_a_completed_grade_on_a_cut_judge_reply_is_a_grading_without_a_valid_verdict(dialect, status, finish):
    collected = copy.deepcopy(fixture("healthy"))
    observed = collected.results[0]["ng_trial_observations"]
    grade = observed["verifier_invocations"][0]["grading_operations"][0]
    grade["kind"] = "llm_judge"
    call = _judge_call(observed)
    _own(observed, call)
    grade["dependencies"] = [_dep(call)]
    call["response_metadata"].update(dialect=dialect, response_status=status, finish_reason=finish)
    d = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    faults = []
    assert _observed_trial_operations(d, collected, grading_failures=faults) is True
    assert len(faults) == 1
    # A caller that does not count gradings without a valid verdict still cannot accept the trial.
    assert _observed_trial_operations(d, collected) is False


def _verifier_retried_judge(*, event=True, earlier=True, same_model=True):
    """A judge request the verifier itself retried: a failed call it owns, then the completed reply the grade rests on."""
    collected = copy.deepcopy(fixture("healthy"))
    observed = collected.results[0]["ng_trial_observations"]
    grade = observed["verifier_invocations"][0]["grading_operations"][0]
    grade["kind"] = "llm_judge"
    final = _judge_call(observed)
    final["started_at"] = 20.0
    failed = copy.deepcopy(final)
    failed_meta = failed["response_metadata"]
    failed["model_call_id"], failed["started_at"] = "failed-judge-call", 10.0 if earlier else 30.0
    failed_meta.update(response_id=None, status_code=500, error_category="upstream_error")
    if not same_model:
        failed_meta["model_ref"] = {"type": "responses_api_models", "name": "another_model"}
    failed_meta["model_operations"] = [
        dict(
            failed_meta["model_operations"][0],
            operation_id="failed-judge-op",
            final_attempt_id="failed-judge-attempt",
            outcome="failed",
            failure_id="failed-judge-event",
        )
    ]
    failed_meta["failure_report"] = {
        "failure_id": "failed-judge-event",
        "omitted": False,
        "events": {
            "failed-judge-event": dict(
                event_id="failed-judge-event",
                boundary="model",
                operation="model_request",
                operation_id="failed-judge-op",
                attempt_id="failed-judge-attempt",
                cause_id=None,
                failure_kind="model_request_error",
                http_status=500,
            )
        }
        if event
        else {},
    }
    observed["model_calls"].append(failed)
    _own(observed, failed)
    _own(observed, final)
    grade["dependencies"] = [_dep(final)]
    return collected


def _covered(collected):
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    counters = {}
    return _observed_trial_operations(decision, collected, judge_recovery=counters), counters


def test_a_judge_request_the_verifier_retried_is_a_recovered_grading_attempt():
    covered, counters = _covered(_verifier_retried_judge())

    assert covered is True
    assert counters == {
        "judge_recovered_attempts": 1,
        "judge_recovered_operations": 1,
        "observed_grading_attempt_errors": 1,
        "observed_grading_failed_operations": 1,
    }


@pytest.mark.parametrize(
    "change", [{"event": False}, {"earlier": False}, {"same_model": False}], ids=["unrecorded", "later", "other_model"]
)
def test_a_retried_judge_request_that_is_not_wholly_recorded_stays_unknown(change):
    assert _covered(_verifier_retried_judge(**change))[0] is False


def _retried_parts(collected):
    observed = collected.results[0]["ng_trial_observations"]
    grade = observed["verifier_invocations"][0]["grading_operations"][0]
    failed = next(call for call in observed["model_calls"] if call["model_call_id"] == "failed-judge-call")
    final = next(call for call in observed["model_calls"] if call["model_call_id"] == "judge-call")
    return observed, grade, failed, final


def _second_failed(collected, suffix="2", started=11.0):
    observed, _, failed, _ = _retried_parts(collected)
    other = copy.deepcopy(failed)
    other["model_call_id"], other["started_at"] = f"failed-judge-call-{suffix}", started
    meta = other["response_metadata"]
    event = copy.deepcopy(next(iter(meta["failure_report"]["events"].values())))
    event.update(
        event_id=f"failed-judge-event-{suffix}",
        operation_id=f"failed-judge-op-{suffix}",
        attempt_id=f"failed-judge-attempt-{suffix}",
    )
    meta["model_operations"][0].update(
        operation_id=f"failed-judge-op-{suffix}",
        final_attempt_id=f"failed-judge-attempt-{suffix}",
        failure_id=f"failed-judge-event-{suffix}",
    )
    meta["failure_report"] = {"failure_id": event["event_id"], "omitted": False, "events": {event["event_id"]: event}}
    observed["model_calls"].append(other)
    _own(observed, other)
    return other


def _all_counters(count):
    return {
        "judge_recovered_attempts": count,
        "judge_recovered_operations": count,
        "observed_grading_attempt_errors": count,
        "observed_grading_failed_operations": count,
    }


def _set_failed(collected, *, operation=None, event=None):
    _, _, failed, _ = _retried_parts(collected)
    meta = failed["response_metadata"]
    meta["model_operations"][0].update(operation or {})
    if event:
        ((_, recorded),) = meta["failure_report"]["events"].items()
        recorded.update(event)
    return collected


@pytest.mark.parametrize(
    "operation",
    [
        {"http_attempts": 2},
        {"http_attempts": 0},
        {"http_attempts": True},
        {"http_attempts": None},
        {"transport_failures": 1},
        {"transport_failures": None},
        {"transport_failures": "0"},
    ],
)
def test_a_retried_judge_request_with_unrecorded_attempts_stays_unknown(operation):
    covered, counters = _covered(_set_failed(_verifier_retried_judge(), operation=operation))

    assert covered is False
    assert counters == {}


def test_a_retried_judge_request_missing_its_counters_stays_unknown():
    collected = _verifier_retried_judge()
    _, _, failed, _ = _retried_parts(collected)
    del failed["response_metadata"]["model_operations"][0]["transport_failures"]

    assert _covered(collected) == (False, {})


def test_a_terminal_transport_failure_of_a_retried_judge_request_is_recovered():
    collected = _set_failed(
        _verifier_retried_judge(),
        operation={"http_attempts": 1, "transport_failures": 1},
        event={"http_status": None},
    )

    assert _covered(collected) == (True, _all_counters(1))


@pytest.mark.parametrize(
    "operation, event",
    [({"transport_failures": 1}, {}), ({}, {"http_status": None})],
    ids=["transport_count_without_transport_event", "transport_event_without_transport_count"],
)
def test_inconsistent_transport_counts_of_a_retried_judge_request_stay_unknown(operation, event):
    assert _covered(_set_failed(_verifier_retried_judge(), operation=operation, event=event)) == (False, {})


@pytest.mark.parametrize("identity", ["operation", "attempt"])
def test_a_retried_judge_request_cannot_share_an_id_with_the_completed_dependency(identity):
    if identity == "operation":
        mutation = dict(operation={"operation_id": "judge-op"}, event={"operation_id": "judge-op"})
    else:
        mutation = dict(operation={"final_attempt_id": "judge-attempt"}, event={"attempt_id": "judge-attempt"})

    assert _covered(_set_failed(_verifier_retried_judge(), **mutation)) == (False, {})


@pytest.mark.parametrize("shared", ["operation", "attempt", "event"])
def test_two_retried_judge_requests_cannot_share_an_id(shared):
    collected = _verifier_retried_judge()
    other = _second_failed(collected)
    meta = other["response_metadata"]
    operation = meta["model_operations"][0]
    ((key, event),) = meta["failure_report"]["events"].items()
    if shared == "operation":
        operation["operation_id"] = event["operation_id"] = "failed-judge-op"
    elif shared == "attempt":
        operation["final_attempt_id"] = event["attempt_id"] = "failed-judge-attempt"
    else:
        meta["failure_report"] = {
            "failure_id": "failed-judge-event",
            "omitted": False,
            "events": {"failed-judge-event": dict(event, event_id="failed-judge-event")},
        }
        operation["failure_id"] = "failed-judge-event"

    assert _covered(collected) == (False, {})


def test_two_retried_judge_requests_with_distinct_ids_are_both_recovered():
    collected = _verifier_retried_judge()
    _second_failed(collected)

    assert _covered(collected) == (True, _all_counters(2))


@pytest.mark.parametrize("starts", [(None, None), (None, 20.0), (20.0, None)])
def test_completed_judge_dependencies_need_no_timestamps_when_nothing_was_retried(starts):
    def mutate(observed, grade):
        grade["dependencies"] = []
        for number, started in enumerate(starts):
            call = _judge_call(observed)
            meta = call["response_metadata"]
            call["model_call_id"], call["started_at"] = f"two-judge-{number}", started
            meta["response_id"] = f"two-response-{number}"
            meta["model_operations"][0].update(
                operation_id=f"two-op-{number}", final_attempt_id=f"two-attempt-{number}"
            )
            _own(observed, call)
            grade["dependencies"].append(_dep(call))

    assert _judged(mutate) is True


def _earlier_dependency(collected, started=5.0):
    observed, grade, _, _ = _retried_parts(collected)
    call = _judge_call(observed)
    meta = call["response_metadata"]
    call["model_call_id"], call["started_at"] = "first-pass-call", started
    meta["response_id"] = "first-pass-response"
    meta["model_operations"][0].update(operation_id="first-pass-op", final_attempt_id="first-pass-attempt")
    _own(observed, call)
    grade["dependencies"].insert(0, _dep(call))
    return call


def test_a_retry_of_a_later_pass_is_recovered_beside_an_earlier_completed_pass():
    collected = _verifier_retried_judge()
    _earlier_dependency(collected)

    assert _covered(collected) == (True, _all_counters(1))


def test_a_retry_after_every_same_model_dependency_stays_unknown_even_if_another_model_is_later():
    collected = _verifier_retried_judge(earlier=False)
    _earlier_dependency(collected)
    later = _earlier_dependency(collected, started=40.0)
    later["response_metadata"]["model_ref"] = {"type": "responses_api_models", "name": "another_model"}
    grade = _retried_parts(collected)[1]
    grade["dependencies"][0] = _dep(later)

    assert _covered(collected)[0] is False
