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
"""Saved unit test results record the grading of a code answer when the reward follows from them.

A code generation verifier runs the answer's extracted code against the task's unit tests and saves one result per
test (``result``: ``True`` for a pass, ``False`` or a negative error code otherwise). Older runs recorded no grading
operation for it. The collector projects the counts only, and the row is observed when the reward is 1.0 exactly when
every test passed (0.0 otherwise), or the default penalty 0.0 for an answer whose reasoning format violated the rule.
"""

import json
from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import (
    CollectedEvidence,
    decide_from_collected,
    grading_artifact,
    trial_verdict,
)
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir


FIXTURE = Path(__file__).parent / "fixtures/cpu-grading-mcqa-healthy.json"
ANSWER = {
    "object": "response",
    "id": "resp-answer",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "```python\nprint(1)\n```"}],
        }
    ],
}


def _graded(result, reward: float, *, violation: float = 0.0) -> dict:
    """A row as the code generation verifier saves it after running the unit tests."""
    return ident(
        0,
        reward=reward,
        response=ANSWER,
        extracted_model_output="kept on the run host",
        extracted_model_code="print(1)",
        result=result,
        metadata={"error_code": -2, "error_message": "Wrong Answer", "traceback": "kept on the run host"},
        unit_tests_time_taken=0.5,
        reasoning_format_violation_rate=violation,
    )


def _collected(tmp_path, row: dict) -> dict:
    job = write_job_dir(tmp_path / "job", n=1, results=[row], failures=[])
    [collected] = collect_with_score(job, "[]").results
    return collected


def test_the_collector_projects_unit_test_counts_only(tmp_path):
    row = _collected(tmp_path, _graded([True, False, -2], 0.0))
    assert row["_ng_collected_grading_facts"]["unit_tests"] == {
        "tests": 3,
        "passed": 1,
        "all_passed": False,
        "code": True,
        "reasoning_format_violation_rate": 0.0,
    }
    assert "kept on the run host" not in json.dumps(row) and "Wrong Answer" not in json.dumps(row)


@pytest.mark.parametrize(
    "result,reward,violation",
    [
        ([True, True], 1.0, 0.0),
        ([True, False], 0.0, 0.0),
        ([True, -2, -3, -4], 0.0, 0.0),
        ([True, True], 0.0, 1.0),
        ([False], 0.0, 1.0),
    ],
    ids=["all-passed", "one-failed", "error-codes", "violation-penalty", "violation-and-failed"],
)
def test_unit_test_results_the_reward_follows_from_record_the_grading(tmp_path, result, reward, violation):
    row = _collected(tmp_path, _graded(result, reward, violation=violation))
    assert grading_artifact(row) == "unit_test_results"


@pytest.mark.parametrize(
    "result,reward,violation",
    [
        ([True, True], 0.0, 0.0),
        ([True, False], 1.0, 0.0),
        ([1, 1], 1.0, 0.0),
        ([True], 1.0, 1.0),
        ([True], 0.5, 0.0),
        ([True], 1.0, 0.5),
        ([], 0.0, 0.0),
    ],
    ids=[
        "passed-scored-zero",
        "failed-scored-one",
        "int-codes",
        "violation-scored-one",
        "partial",
        "rate",
        "no-tests",
    ],
)
def test_unit_test_results_the_reward_does_not_follow_from_stay_unobserved(tmp_path, result, reward, violation):
    row = _collected(tmp_path, _graded(result, reward, violation=violation))
    assert grading_artifact(row) is None


def test_an_unbound_answer_is_not_recorded_by_its_unit_test_results(tmp_path):
    row = _collected(tmp_path, _graded([True], 1.0) | {"response": ANSWER | {"id": None}})
    assert grading_artifact(row) is None


def test_a_row_without_unit_test_results_is_unchanged(tmp_path):
    # No code was extracted: nothing ran, and nothing records the grading.
    row = _collected(
        tmp_path,
        ident(
            0, reward=0.0, response=ANSWER, extracted_model_output="no code here", reasoning_format_violation_rate=0.0
        ),
    )
    assert "unit_tests" not in row.get("_ng_collected_grading_facts", {})
    assert grading_artifact(row) is None


@pytest.mark.parametrize("unit_tests,passed", [(True, True), (False, None)], ids=["recorded", "unrecorded"])
def test_the_trial_gate_observes_a_trial_through_its_unit_test_results(unit_tests, passed):
    collected = CollectedEvidence.model_validate_json(FIXTURE.read_text())
    row = collected.results[0]
    row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    if unit_tests:
        row["_ng_collected_grading_facts"] = {
            "unit_tests": {
                "tests": 2,
                "passed": 2,
                "all_passed": True,
                "code": True,
                "reasoning_format_violation_rate": 0.0,
            }
        }
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    result, reason, evaluation = trial_verdict(decision, collected)
    assert result is passed, reason
    assert evaluation["exact"] is unit_tests
