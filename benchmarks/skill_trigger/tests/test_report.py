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
"""Check denominators and isolation of experimental results."""

import pytest

from benchmarks.skill_trigger.report import Attempt, summarize


def attempt(identifier, **kwargs):
    return Attempt(
        attempt_id=identifier,
        task_id="task",
        model="fixture",
        runtime="pinned",
        profile="pi",
        collection="example",
        release_sha256="a" * 64,
        case_kind="positive",
        **kwargs,
    )


def test_report_preserves_unscored_pending_and_duplicates():
    rows = [
        attempt("a", status="scored", reward=1),
        attempt("b", status="scored", reward=0),
        attempt("c", status="unscored", reason="Missing evidence"),
        attempt("d", status="pending"),
    ]
    summary = summarize(rows)[0]
    assert (summary["planned"], summary["scored"], summary["unscored"], summary["pending"]) == (4, 2, 1, 1)
    assert summary["rate"] == 0.5
    assert summary["numerator"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rows + rows[:1])
    assert summarize(rows[2:])[0]["rate"] is None


def test_negative_rate_is_opening_not_avoidance():
    value = attempt("a", status="scored", reward=1).model_copy(
        update={
            "case_kind": "negative",
            "all_skill_openings": 1,
            "session_start_openings": 1,
            "unnecessary_skill_openings": 0,
        }
    )
    summary = summarize([value])[0]
    assert summary["metric"] == "unnecessary_opening_rate"
    assert summary["rate"] == 0
    other = value.model_copy(update={"attempt_id": "b", "release_sha256": "b" * 64})
    assert len(summarize([value, other])) == 2


def test_invalid_outcome_is_rejected():
    with pytest.raises(ValueError):
        attempt("a", status="unscored", reward=0)
    with pytest.raises(ValueError):
        attempt("a", status="scored")


def test_negative_counts_are_required_and_consistent():
    row = attempt("a", status="scored", reward=1).model_dump()
    row["case_kind"] = "negative"
    with pytest.raises(ValueError, match="require all"):
        Attempt.model_validate(row)
    row.update(all_skill_openings=1, session_start_openings=1, unnecessary_skill_openings=0)
    summary = summarize([Attempt.model_validate(row)])[0]
    assert summary["opening_metrics"]["session_start_openings"]["rate"] == 1
    assert summary["opening_metrics"]["all_skill_openings"]["numerator"] == 1
    assert summary["rate"] == 0
    row["unnecessary_skill_openings"] = 1
    with pytest.raises(ValueError, match="contradict"):
        Attempt.model_validate(row)
