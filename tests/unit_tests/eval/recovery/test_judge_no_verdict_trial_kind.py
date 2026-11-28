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
"""A row whose judge gave no verdict when asked again is a counted, final ``judge_no_verdict`` failure.

It stays in the main output with reward 0 and is not masked. The trial gate counts it as an evaluation-side fault
with an exact count, so a run below the failure limit passes, and recovery never judges its answer again.
"""

import pytest

from nemo_gym.core.failure_kinds import JUDGE_NO_VERDICT, FailureClass, classify_failure
from nemo_gym.eval.recovery.grading import reverification_cause, saved_row_evidence
from tests.helpers.recovery_health import _gate, _run


def _run_with_one_judge_no_verdict(kind: str):
    """101 graded trials; the first one's judge gave no verdict, so its row scored zero with ``kind``."""
    collected = _run(101)
    row = collected.results[0]
    row["reward"] = 0.0
    row["mask_sample"] = False
    row["failure_kind"] = kind
    row["failure_reason"] = "The judge gave no verdict for game 2 (judge output incomplete: content_filter)"
    return collected, row


def test_judge_no_verdict_is_evaluation_side():
    assert classify_failure(failure_kind=JUDGE_NO_VERDICT) == FailureClass.EVALUATION


def test_one_judge_no_verdict_in_101_trials_passes_with_an_exact_count_and_is_not_judged_again():
    collected, row = _run_with_one_judge_no_verdict(JUDGE_NO_VERDICT)

    decision, (passed, reason, evaluation), _ = _gate(collected)

    assert decision.complete
    assert passed is True, reason
    assert evaluation["exact"] is True
    assert evaluation["evaluation"]["kinds"] == {JUDGE_NO_VERDICT: 1}
    assert reverification_cause(saved_row_evidence(row)) is None


@pytest.mark.parametrize("kind", ["arena:judge_no_verdict"])
def test_an_unclassified_server_kind_for_the_same_row_withholds_the_pass(kind):
    collected, _ = _run_with_one_judge_no_verdict(kind)

    _, (passed, reason, _), _ = _gate(collected)

    assert passed is None
    assert "unknown classification" in reason
