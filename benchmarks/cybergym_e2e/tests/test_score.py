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
"""score.classify: honest outcomes are scored; malformed/missing validator output is masked."""

from score import classify


EXPECTED = {"stage1": "passed", "stage2": "passed", "stage3": "passed"}


def test_full_pass_is_solved():
    got = {"stage1": "passed", "stage2": "passed", "stage3": "passed", "stage4": "passed"}
    assert classify(got, EXPECTED, True, True) == (True, False)


def test_honest_unsolved_is_scored_not_masked():
    # stage1 passed, stage2 failed -> stage3 legitimately skipped: a real model failure, never masked.
    got = {"stage1": "passed", "stage2": "failed", "stage3": "skipped"}
    assert classify(got, EXPECTED, True, True) == (False, False)


def test_missing_stage_key_is_masked():
    # The validator JSON omitted stage2/stage3 (-> None): inconclusive, must NOT score as a failure.
    solved, ve = classify({"stage1": "passed"}, EXPECTED, True, True)
    assert solved is False and ve is True


def test_error_status_is_masked():
    solved, ve = classify({"stage1": "error"}, {"stage1": "passed"}, True, True)
    assert solved is False and ve is True


def test_unknown_status_is_masked():
    _, ve = classify({"stage1": "weird"}, {"stage1": "passed"}, True, True)
    assert ve is True


def test_firewall_breach_blocks_solve():
    got = {k: "passed" for k in ("stage1", "stage2", "stage3", "stage4")}
    solved, ve = classify(got, EXPECTED, False, True)
    assert solved is False and ve is False  # firewall, not a validator failure


def test_poc_only_cannot_solve_end_to_end():
    assert classify({"stage1": "passed"}, {"stage1": "passed"}, True, False) == (False, False)
