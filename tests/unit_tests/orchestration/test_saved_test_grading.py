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
"""Saved test results record how a trial was graded: a repository test run of the answer's patch, and per-sub-step
test results. Each binds the saved answer and the score; a bare reward never does."""

import pytest

from nemo_gym.orchestration.recovery import grading_artifact
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir
from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run


RESPONSE = {"id": "resp-answer", "status": "completed", "output": [{"type": "message", "role": "assistant"}]}


def _repository_test_run(task: int, *, resolved: bool = True, **changes) -> dict:
    """A row as a repository-patch verifier saves it: the patch, its test run's verdict and per-test results."""
    return (
        ident(
            task,
            reward=float(resolved is True),
            response=RESPONSE,
            evaluation_completed=True,
            eval_timed_out=False,
            resolved=resolved,
            patch_applied=True,
            model_patch="diff --git a/module.py b/module.py\n",
            test_results={
                "tests": [{"name": "tests/test_module.py::test_case", "status": "PASSED" if resolved else "FAILED"}]
            },
            test_output="STDOUT:\nRunning selected tests\n",
            error=None,
            verification_attempts=1,
        )
        | changes
    )


def _substep_tests(task: int, *, steps: list[bool] = (True, False, True), **changes) -> dict:
    """A row as a multi-step code verifier saves it: one test result per scored sub-step, and the problem verdict."""
    steps = list(steps)
    passed = sum(steps)
    return (
        ident(
            task,
            reward=float(all(steps)),
            response=RESPONSE,
            problem_id="7",
            step_results=steps,
            num_steps_passed=passed,
            num_steps_total=len(steps),
            problem_accuracy=all(steps),
            subtask_accuracy=passed / len(steps),
            scored_step_ids=[f"7.{number}" for number in range(1, len(steps) + 1)],
            step_environment_results=[{"current": step} for step in steps],
            step_environment_errors=[{"current": None if step else "AssertionError"} for step in steps],
            step_outputs=None,
        )
        | changes
    )


def _collected_rows(tmp_path, rows: list[dict]) -> list[dict]:
    job = write_job_dir(tmp_path / "job", n=len(rows), results=rows, failures=[])
    return collect_with_score(job, "[]").results


def test_the_collector_projects_only_the_test_runs_flags(tmp_path):
    [row] = _collected_rows(tmp_path, [_repository_test_run(0)])
    assert row["_ng_collected_grading_facts"]["test_run"] == {
        "resolved": True,
        "patch_applied": True,
        "timed_out": False,
        "error": False,
        "test_results": True,
        "model_patch": True,
    }
    assert "test_results" not in row and "model_patch" not in row and "test_output" not in row


def test_the_collector_projects_only_the_substep_counts(tmp_path):
    [row] = _collected_rows(tmp_path, [_substep_tests(0)])
    assert row["_ng_collected_grading_facts"]["substep_tests"] == {
        "steps": 3,
        "passed": 2,
        "num_steps_total": 3,
        "num_steps_passed": 2,
        "problem_accuracy": False,
        "environment_results": 3,
    }
    assert "step_results" not in row and "step_environment_errors" not in row


@pytest.mark.parametrize("resolved", [True, False])
def test_a_repository_test_run_records_the_grading(tmp_path, resolved):
    [row] = _collected_rows(tmp_path, [_repository_test_run(0, resolved=resolved)])
    assert grading_artifact(row) == "repository_test_run"


@pytest.mark.parametrize(
    "changes",
    [
        {"evaluation_completed": False},
        {"reward": 0.0},
        {"error": "patch extraction failed"},
        {"eval_timed_out": True},
        {"test_results": None},
        {"patch_applied": False},
        {"resolved": None},
    ],
    ids=["not-completed", "reward-disagrees", "error", "timed-out", "no-test-results", "not-applied", "no-verdict"],
)
def test_a_repository_test_run_that_does_not_bind_its_verdict_is_no_grading_record(tmp_path, changes):
    [row] = _collected_rows(tmp_path, [_repository_test_run(0, **changes)])
    assert grading_artifact(row) is None


@pytest.mark.parametrize("steps", [[True, True], [True, False, False]])
def test_substep_test_results_record_the_grading(tmp_path, steps):
    [row] = _collected_rows(tmp_path, [_substep_tests(0, steps=steps)])
    assert grading_artifact(row) == "substep_tests"


@pytest.mark.parametrize(
    "changes",
    [
        {"reward": 1.0},
        {"num_steps_passed": 3},
        {"num_steps_total": 4},
        {"problem_accuracy": True},
        {"step_environment_results": [{"current": True}]},
        {"step_results": [], "num_steps_passed": 0, "num_steps_total": 0, "step_environment_results": []},
    ],
    ids=["reward-disagrees", "passed-disagrees", "total-disagrees", "verdict-disagrees", "unaligned", "no-steps"],
)
def test_substep_results_that_do_not_bind_the_verdict_are_no_grading_record(tmp_path, changes):
    [row] = _collected_rows(tmp_path, [_substep_tests(0, **changes)])
    assert grading_artifact(row) is None


def _gate_with_facts(facts: dict, *, reward: float):
    collected = _run(1)
    row = collected.results[0]
    row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    row["reward"] = reward
    row["_ng_collected_grading_facts"] = facts
    return collected


@pytest.mark.parametrize(
    "facts,kept",
    [
        (
            {
                "test_run": {
                    "resolved": False,
                    "patch_applied": True,
                    "timed_out": False,
                    "error": False,
                    "test_results": True,
                    "model_patch": True,
                }
            },
            {"evaluation_completed": True},
        ),
        (
            {
                "substep_tests": {
                    "steps": 2,
                    "passed": 1,
                    "num_steps_total": 2,
                    "num_steps_passed": 1,
                    "problem_accuracy": False,
                    "environment_results": 2,
                }
            },
            {},
        ),
    ],
    ids=["repository-test-run", "substep-tests"],
)
def test_a_trial_graded_by_saved_test_results_is_observed_and_the_run_passes(facts, kept):
    collected = _gate_with_facts(facts, reward=0.0)
    collected.results[0].update(kept)
    _, (passed, reason, report), task = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert task["trial_counts"]["unobserved"] == 0
