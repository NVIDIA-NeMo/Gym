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
"""A saved judge record counts only when a judge actually ran and its verdict agrees with the reward.

An empty ``judge_evaluations`` list records no judge; a judge evaluation's parsed verdict label must agree with the
reward. A rubric judge's saved verdict and score record the grading when the reward follows from them.
"""

import pytest

from nemo_gym.orchestration.recovery import grading_artifact
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir


COMPLETED = {"id": "resp-answer", "status": "completed", "output": [{"type": "message", "role": "assistant"}]}
INCOMPLETE = {"id": "resp-answer", "status": "incomplete", "output": [{"type": "reasoning"}]}


def _collected(tmp_path, row: dict) -> dict:
    job = write_job_dir(tmp_path / "job", n=1, results=[row], failures=[])
    [collected] = collect_with_score(job, "[]").results
    return collected


def _judge_evaluation(label: str) -> dict:
    return {"responses_create_params": {"input": "kept on the run host"}, "verdict_label": label}


def test_an_empty_judge_evaluation_list_is_no_judge_record_and_the_incomplete_answer_is_a_model_failure(tmp_path):
    row = _collected(tmp_path, ident(0, reward=0.0, response=INCOMPLETE, judge_evaluations=[]))
    # Nothing a judge saved: no grading facts are projected at all.
    assert not row.get("_ng_collected_grading_facts", {}).get("judge_evaluations")
    assert grading_artifact(row) is None
    from nemo_gym.orchestration.recovery import _model_failure

    assert _model_failure(row) is True


@pytest.mark.parametrize(
    "labels,reward,recorded",
    [
        (["Judgement: yes"], 1.0, True),
        (["Judgement: no"], 0.0, True),
        (["[[A=B]]", "[[A=B]]"], 1.0, True),
        (["[[A=B]]", "[[A!=B]]"], 0.0, True),
        (["Judgement: yes"], 0.0, False),
        (["Judgement: no"], 1.0, False),
        (["JUDGE_ERROR"], 0.0, False),
        (["an unparsed label"], 1.0, True),
    ],
    ids=["yes", "no", "equal-twice", "swap-disagrees", "yes-scored-zero", "no-scored-one", "judge-error", "unparsed"],
)
def test_a_judge_evaluation_records_the_grading_only_when_its_verdict_agrees_with_the_reward(
    tmp_path, labels, reward, recorded
):
    row = _collected(
        tmp_path,
        ident(0, reward=reward, response=COMPLETED, judge_evaluations=[_judge_evaluation(label) for label in labels]),
    )
    assert (grading_artifact(row) == "judge_record") is recorded
    facts = row["_ng_collected_grading_facts"]
    assert "Judgement" not in str(facts) and "A=B" not in str(facts)


def _rubric(**changes) -> dict:
    """A row as a rubric judge saves it: its verdict, the rubric score and the pass threshold."""
    return {
        "verdict": "YES",
        "judge_output": "kept on the run host",
        "rubric_score": 8.0,
        "rubric_score_normalized": 0.8,
        "scoring_threshold": 7.0,
        "scoring_max_points": 10.0,
        "invalid_judge_response": False,
    } | changes


def test_a_rubric_verdict_that_passes_the_threshold_records_the_grading(tmp_path):
    row = _collected(tmp_path, ident(0, reward=1.0, response=COMPLETED, **_rubric()))
    assert row["_ng_collected_grading_facts"]["rubric"] == {
        "verdict": True,
        "rubric_score": 8.0,
        "scoring_threshold": 7.0,
        "scoring_max_points": 10.0,
        "invalid_judge_response": False,
    }
    assert "judge_output" not in row
    assert grading_artifact(row) == "rubric_verdict"


@pytest.mark.parametrize(
    "reward,changes",
    [
        (0.0, {}),
        (1.0, {"verdict": "NO", "rubric_score": 3.0}),
        (1.0, {"rubric_score": 6.0}),
        (1.0, {"invalid_judge_response": True}),
        (1.0, {"rubric_score": None}),
        (1.0, {"verdict": "maybe"}),
    ],
    ids=["reward-disagrees", "verdict-no", "below-threshold", "invalid-judge", "no-score", "unknown-verdict"],
)
def test_a_rubric_verdict_that_does_not_bind_the_reward_stays_unverified(tmp_path, reward, changes):
    row = _collected(tmp_path, ident(0, reward=reward, response=COMPLETED, **_rubric(**changes)))
    assert grading_artifact(row) is None


def test_a_rubric_verdict_that_fails_the_threshold_records_a_zero(tmp_path):
    row = _collected(tmp_path, ident(0, reward=0.0, response=COMPLETED, **_rubric(verdict="NO", rubric_score=5.0)))
    assert grading_artifact(row) == "rubric_verdict"
