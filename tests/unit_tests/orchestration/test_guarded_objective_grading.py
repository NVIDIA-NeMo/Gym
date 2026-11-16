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
"""An agent that runs its own guardrail-gated rubric saves the headline score and the counts it follows from.

The rubric scores a task 0.0 when a guardrail was violated, else the objectives passed over the objectives. Older runs
recorded no grading operation for it. The row is observed when its reward equals the saved headline and the headline
follows from the saved counts; the counts are the rubric's own, not a re-run of its assertions.

The fixtures are graded rows of a saved run, cut to the saved metrics, with ids and answer text replaced.
"""

import json
from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import grading_artifact
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir


ROWS = json.loads((Path(__file__).parent / "fixtures/saved-verifier-output-guarded-objectives.json").read_text())


def _collected(tmp_path, saved: dict, reward: float | None = None, **metrics) -> dict:
    response = saved["response"] | {"metrics": saved["response"]["metrics"] | metrics}
    row = ident(0, reward=saved["reward"] if reward is None else reward, response=response)
    job = write_job_dir(tmp_path / "job", n=1, results=[row], failures=[])
    [collected] = collect_with_score(job, "[]").results
    return collected


@pytest.mark.parametrize("saved", ROWS, ids=[row["_class"] for row in ROWS])
def test_a_headline_that_follows_from_its_counts_records_the_grading(tmp_path, saved):
    assert grading_artifact(_collected(tmp_path, saved)) == "guarded_objective_score"


@pytest.mark.parametrize(
    "cls,reward,metrics",
    [
        ("partial", 1.0, {}),
        ("partial", 0.75, {"objectives_passed": 2.0}),
        ("partial", 0.75, {"guardrails_violated": 1.0}),
        ("full", 1.0, {"objectives_passed": 5.0, "objectives_total": 4.0}),
        ("partial", 0.75, {"objectives_total": None}),
        ("zero", 0.0, {"objectives_total": 0.0, "guardrails_total": 0.0}),
        ("guardrail", 0.0, {"guardrails_total": 0.0}),
        ("partial", 0.75, {"aa_headline": None}),
    ],
    ids=[
        "reward-not-headline",
        "headline-not-counts",
        "guardrail-violated",
        "more-passed-than-total",
        "no-count",
        "no-counted-assertion",
        "more-violated-than-guardrails",
        "no-headline",
    ],
)
def test_a_headline_that_does_not_follow_from_its_counts_stays_unobserved(tmp_path, cls, reward, metrics):
    saved = next(row for row in ROWS if row["_class"] == cls)
    assert grading_artifact(_collected(tmp_path, saved, reward, **metrics)) is None
