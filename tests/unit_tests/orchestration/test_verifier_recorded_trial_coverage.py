# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A verifier that grades in process is observed through the grading record it saved with its verify result."""

import pytest

from nemo_gym.core.episode_context import rollout_context
from nemo_gym.orchestration.recovery import (
    _observed_trial_operations,
    decide_from_collected,
    trial_counts,
    trial_health,
)
from nemo_gym.server_runtime.verifier_observability import observe_verifier
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir
from tests.unit_tests.orchestration.test_llm_judge_trial_coverage import _dep, _judge_call, _own
from tests.unit_tests.orchestration.test_tool_trial_coverage import fixture


def _record(row, **change):
    return {
        "verifier": "mcqa",
        "score": row["reward"],
        "verdict": "scored",
        "result_binding_sha256": row["_ng_collected_grading_binding"],
        **change,
    }


def _in_process(case, **change):
    """The fixture's rows as an in-process verifier saves them: no grading operation, only its grading record."""
    collected = fixture(case)
    for row in collected.results:
        verifier = row["ng_trial_observations"]["verifier_invocations"][0]
        verifier["grading_operations"] = None
        verifier["grading_record"] = _record(row, **change)
    return collected


def _gate(collected):
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, reason = trial_health(decision, collected)
    return passed, reason, trial_counts(decision, collected), _observed_trial_operations(decision, collected)


def test_a_metric_only_run_is_fully_observed_as_verifier_recorded():
    passed, reason, counts, covered = _gate(_in_process("repeats"))
    assert covered is True and passed is True
    assert counts == {"observed_healthy": 2, "unhealthy": 0, "unobserved": 0, "verifier_recorded": 2}
    assert "grading (2 verifier-recorded, 0 observed through a recorded grading operation)" in reason
    assert "UNOBSERVED" not in reason and "lower bound" not in reason


def test_an_early_return_zero_score_with_its_verifier_record_is_observed():
    collected = _in_process("wrong")
    assert collected.results[0]["reward"] == 0
    passed, reason, counts, covered = _gate(collected)
    assert covered is True and passed is True
    assert counts["unobserved"] == 0 and counts["verifier_recorded"] == 1
    assert "1 verifier-recorded" in reason


@pytest.mark.parametrize("change", ["absent", "verifier", "score", "binding", "masked", "owned_call", "operation"])
def test_a_reward_without_the_verifiers_own_record_stays_unobserved(change):
    collected = _in_process("healthy")
    row = collected.results[0]
    verifier = row["ng_trial_observations"]["verifier_invocations"][0]
    if change == "absent":
        verifier.pop("grading_record")
    elif change == "verifier":
        verifier["grading_record"]["verifier"] = "another_verifier"
    elif change == "score":
        verifier["grading_record"]["score"] = 0.0
    elif change == "binding":
        verifier["grading_record"]["result_binding_sha256"] = "0" * 64
    elif change == "masked":
        verifier["grading_record"]["verdict"] = "masked"
    elif change == "owned_call":
        verifier["model_calls"] = [{"model_call_id": "grading-call", "model_ref": None, "response_id": None}]
    else:
        # A recorded grading operation decides the row; the record cannot stand in for an unbound one.
        verifier["grading_operations"] = [
            {
                "operation_id": "grade-1",
                "kind": "programmatic_calculation",
                "outcome": "unknown",
                "result_binding_sha256": None,
                "dependencies": [],
            }
        ]
    assert row["reward"] == 1
    passed, reason, counts, covered = _gate(collected)
    assert covered is False and passed is None
    assert "verifier-recorded" not in reason
    assert counts["verifier_recorded"] == 0
    if change != "operation":
        assert counts["unobserved"] == 1
        assert "UNOBSERVED 1 of 1 expected trials" in reason


def _judge_row(*, captured, recorded):
    """The healthy trial graded by one captured judge call, optionally with that capture lost."""
    collected = fixture("healthy")
    row = collected.results[0]
    observed = row["ng_trial_observations"]
    verifier = observed["verifier_invocations"][0]
    grade = verifier["grading_operations"][0]
    grade["kind"] = "llm_judge"
    call = _judge_call(observed)
    _own(observed, call)
    grade["dependencies"] = [_dep(call)]
    if not captured:
        observed["model_calls"].remove(call)
        verifier["model_calls"] = []
    if recorded:
        verifier["grading_record"] = _record(row)
    return collected


def test_a_judge_row_with_its_capture_is_observed_through_its_grading_operation():
    passed, reason, counts, covered = _gate(_judge_row(captured=True, recorded=False))
    assert covered is True and passed is True
    assert counts["verifier_recorded"] == 0 and counts["unobserved"] == 0
    assert "actual model/tool and bound grading operations cover every identity and repeat" in reason


@pytest.mark.parametrize("recorded", [False, True])
def test_a_judge_row_without_its_capture_stays_unobserved(recorded):
    passed, reason, counts, covered = _gate(_judge_row(captured=False, recorded=recorded))
    assert covered is False and passed is None
    assert "verifier-recorded" not in reason and counts["verifier_recorded"] == 0


def test_an_uninstrumented_judge_without_captures_or_a_record_stays_unobserved():
    passed, reason, counts, covered = _gate(fixture("unobserved-judge"))
    assert covered is False and passed is None
    assert counts["unobserved"] == 1 and counts["verifier_recorded"] == 0


@pytest.mark.asyncio
async def test_the_collector_keeps_the_saved_record_bound_to_the_saved_row(tmp_path):
    graded = {"response": {"id": "graded-response", "status": "completed", "output": []}, "reward": 1.0}

    async def verify():
        return graded

    with rollout_context("0-0"):
        returned = await observe_verifier(verify, enabled=True, verifier_name="metric_verifier")()
    (invocation,) = returned["ng_verifier_invocations"]
    row = ident(0, **graded, ng_trajectory={"verifier_invocations": [invocation]})
    job = write_job_dir(tmp_path / "job", n=1, results=[row], failures=[])

    [collected] = collect_with_score(job, "[]").results
    [verifier] = collected["ng_trial_observations"]["verifier_invocations"]
    assert verifier["grading_record"] == {
        "verifier": "metric_verifier",
        "score": 1.0,
        "verdict": "scored",
        "result_binding_sha256": collected["_ng_collected_grading_binding"],
    }
