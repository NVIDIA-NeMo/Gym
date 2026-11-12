# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A trial is observed by the grading bound to its saved answer; trajectory evidence is reported, never a condition.

The trajectory an agent records (its invocations, turns, call ownership, session flag, conversation and sandbox
timing) is observability evidence. A run whose every answer is bound to its grading gets an exact verdict, with the
trajectory evidence it lacks reported by kind; a trial whose answer is not bound to any grading stays unverified.
"""

from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run


SUBSTEPS = {
    "steps": 2,
    "passed": 2,
    "num_steps_total": 2,
    "num_steps_passed": 2,
    "problem_accuracy": True,
    "environment_results": 2,
}
TEST_RUN = {
    "resolved": True,
    "patch_applied": True,
    "timed_out": False,
    "error": False,
    "test_results": True,
    "model_patch": True,
}


def _graded_by(row: dict, **facts) -> None:
    """The verifier recorded no grading operation; a saved grading artifact records the grading instead."""
    row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    row["_ng_collected_grading_facts"] = facts


def _repository_patch_shape(row: dict) -> None:
    """A sandboxed coding agent: sandbox timing unrecorded, no session flag; a repository test run grades it."""
    _graded_by(row, test_run=TEST_RUN)
    row["evaluation_completed"] = True
    observed = row["ng_trial_observations"]
    observed["gaps"] = [{"code": "sandbox_lifecycle_timing_unavailable"}]
    observed["invocations"][0]["in_session"] = None


def _multi_step_code_shape(row: dict) -> None:
    """A multi-step code agent that records no conversation; per-sub-step tests grade it."""
    _graded_by(row, substep_tests=SUBSTEPS)
    row["ng_trial_observations"]["gaps"] = [{"code": "conversation_unavailable"}]


def _simulated_user_shape(row: dict, task: int) -> None:
    """A tool-transcript agent: the agent and a simulated user, neither owning its calls; the environment rewards."""
    _graded_by(row, environment_reward=row["reward"])
    observed = row["ng_trial_observations"]
    observed["verifier_invocations"] = []
    observed["invocations"] = [
        {"invocation_id": f"run-{task}:agent", "status": "unknown", "in_session": None, "model_calls": []},
        {"invocation_id": f"run-{task}:user_simulator", "status": "unknown", "in_session": None, "model_calls": []},
    ]
    observed["turns"] = []
    observed["gaps"] = [
        {"code": "tau_tool_transcript_only"},
        {"code": "tau_tool_transcript_only"},
        {"code": "model_call_ownership_unavailable"},
        {"code": "turns_unavailable"},
    ]


def _assert_exact_with_trajectory_reported(collected, *, kinds: dict, trials: int):
    _, (passed, reason, report), task = _gate(collected)
    assert passed is True, reason
    assert report["status"] == "PASS" and report["exact"] is True
    assert report["evaluation"]["count"] == 0
    assert report["trajectory_evidence"] == {"incomplete": trials, "trials": trials, "kinds": kinds}
    assert f"trajectory evidence incomplete: {trials} of {trials}" in reason
    assert task["trial_counts"]["unobserved"] == 0
    return reason


def test_a_repository_patch_run_with_its_trajectory_gaps_is_exact_and_reports_them():
    collected = _run(20)
    for row in collected.results:
        _repository_patch_shape(row)
    _assert_exact_with_trajectory_reported(
        collected, kinds={"in_session_unrecorded": 20, "sandbox_lifecycle_timing_unavailable": 20}, trials=20
    )


def test_a_multi_step_code_run_without_its_conversation_is_exact_and_reports_it():
    collected = _run(20)
    for row in collected.results:
        _multi_step_code_shape(row)
    _assert_exact_with_trajectory_reported(collected, kinds={"conversation_unavailable": 20}, trials=20)


def test_a_simulated_user_run_without_call_ownership_is_exact_and_reports_it():
    collected = _run(20)
    for task, row in enumerate(collected.results):
        _simulated_user_shape(row, task)
    _assert_exact_with_trajectory_reported(
        collected,
        kinds={
            "in_session_unrecorded": 20,
            "invocation_unknown": 20,
            "model_call_ownership_unavailable": 20,
            "multiple_invocations": 20,
            "tau_tool_transcript_only": 20,
            "turns_unavailable": 20,
        },
        trials=20,
    )


def test_a_simulated_user_run_that_ended_on_a_rejected_request_counts_it_as_a_model_fault():
    collected = _run(20)
    for task, row in enumerate(collected.results):
        _simulated_user_shape(row, task)
    # The policy's last request was rejected (HTTP 400) and the environment scored the rollout zero.
    row = collected.results[0]
    row["reward"] = 0.0
    row["_ng_collected_grading_facts"]["environment_reward"] = 0.0
    [call] = row["ng_trial_observations"]["model_calls"]
    [operation] = call["response_metadata"]["model_operations"]
    operation["outcome"], operation["failure_id"] = "failed", "rejected-1"
    call["response_metadata"]["failure_report"]["events"] = {
        "rejected-1": {
            "event_id": "rejected-1",
            "boundary": "model",
            "operation": "model_request",
            "operation_id": operation["operation_id"],
            "attempt_id": operation["final_attempt_id"],
            "cause_id": None,
            "failure_kind": "model_request_error",
            "http_status": 400,
        }
    }
    _, (passed, reason, report), _ = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["model"]["kinds"] == {"model_request_error": 1}
    assert report["evaluation"]["count"] == 0
    assert report["trajectory_evidence"]["incomplete"] == 19


def test_a_trial_with_a_trajectory_gap_and_no_grading_artifact_stays_unverified():
    collected = _run(20)
    for row in collected.results:
        _repository_patch_shape(row)
    # The last trial saved only its reward: nothing records how it was graded.
    collected.results[-1]["_ng_collected_grading_facts"] = {}
    _, (passed, reason, report), task = _gate(collected)
    assert passed is None and report["status"] == "UNVERIFIED" and report["exact"] is False
    assert reason.startswith("trial: UNOBSERVED 1 of 20 expected trials")
    assert task["trial_counts"]["unobserved"] == 1


def test_a_failed_request_on_a_graded_trial_that_did_not_end_there_is_never_hidden():
    collected = _run(20)
    for row in collected.results:
        _repository_patch_shape(row)
    # A rejected request the trial recorded but no recovery accounts for, on a trial scored as passed.
    row = collected.results[0]
    [call] = row["ng_trial_observations"]["model_calls"]
    [operation] = call["response_metadata"]["model_operations"]
    operation["outcome"], operation["failure_id"] = "failed", "rejected-1"
    _, (passed, _, report), _ = _gate(collected)
    assert passed is None and report["exact"] is False
