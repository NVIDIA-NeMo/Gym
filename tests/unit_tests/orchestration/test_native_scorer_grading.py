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
"""A verifier that runs a native scorer saves the scorer's metrics beside the answer.

The scorer runs outside the verifier and returns its metrics (``native_metrics``: a ``weighted_score`` in [0, 1] and a
binary ``execution`` flag) and its diagnostics. Older runs recorded no grading operation for it. The row is observed
when its reward equals the saved weighted score, the execution flag is binary and the diagnostics were saved; the
score is the scorer's own, not a re-run of it.
"""

import json

import pytest

from nemo_gym.orchestration.recovery import grading_artifact
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir


ANSWER = {
    "object": "response",
    "id": "resp-answer",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "kept on the run host"}],
        }
    ],
}


def _collected(tmp_path, reward=0.42, *, weighted=0.42, execution=1.0, diagnostics=None) -> dict:
    row = ident(
        0,
        reward=reward,
        response=ANSWER,
        native_metrics={"weighted_score": weighted, "execution": execution},
        native_diagnostics={"format_audit": {"valid": True, "issues": []}} if diagnostics is None else diagnostics,
    )
    job = write_job_dir(tmp_path / "job", n=1, results=[row], failures=[])
    [collected] = collect_with_score(job, "[]").results
    return collected


@pytest.mark.parametrize(("reward", "execution"), [(0.42, 1.0), (0.0, 0.0), (1.0, 1.0)])
def test_a_reward_equal_to_the_scorers_saved_score_records_the_grading(tmp_path, reward, execution):
    row = _collected(tmp_path, reward, weighted=reward, execution=execution)
    assert row["_ng_collected_grading_facts"]["native_score"] == {
        "weighted_score": reward,
        "execution": execution,
        "diagnostics": True,
    }
    assert "kept on the run host" not in json.dumps(row) and "format_audit" not in json.dumps(row)
    assert grading_artifact(row) == "native_scorer_score"


@pytest.mark.parametrize(
    "changes",
    [
        {"reward": 0.5},
        {"weighted": 1.5, "reward": 1.5},
        {"execution": 0.5},
        {"execution": None},
        {"diagnostics": {}},
    ],
    ids=["reward-not-score", "score-out-of-range", "execution-not-binary", "no-execution", "no-diagnostics"],
)
def test_a_reward_the_scorers_saved_metrics_do_not_record_stays_unobserved(tmp_path, changes):
    assert grading_artifact(_collected(tmp_path, **changes)) is None


def test_an_unbound_answer_is_not_recorded_by_its_scorer_metrics(tmp_path):
    row = ident(
        0,
        reward=0.42,
        response=ANSWER | {"id": None},
        native_metrics={"weighted_score": 0.42, "execution": 1.0},
        native_diagnostics={"format_audit": {}},
    )
    job = write_job_dir(tmp_path / "job", n=1, results=[row], failures=[])
    [collected] = collect_with_score(job, "[]").results
    assert grading_artifact(collected) is None
