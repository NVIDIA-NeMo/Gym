# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from nemo_gym.orchestration.recovery import _observed_trial_operations, decide_from_collected
from tests.unit_tests.test_tool_trial_coverage import fixture


def _judged(mutate):
    """The actual healthy CPU trial fixture relabelled as judge-graded; used only for negative controls."""
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
