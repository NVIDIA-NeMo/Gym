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

"""Tests for rollout identity: minting, validation, and the carriers."""

from __future__ import annotations

import pytest

from nemo_gym.rollout_id import (
    NG_ROLLOUT_PREFIX_RE,
    ROLLOUT_ID_HEADER,
    is_valid_rollout_id,
    make_rollout_id,
    slugify,
)


def test_make_rollout_id_is_deterministic_and_formatted():
    rid = make_rollout_id("Blackbox Toy", 3, 1, 0)
    assert rid == "blackbox-toy.00003.r01.a0"
    # Same coordinates -> same id.
    assert rid == make_rollout_id("Blackbox Toy", 3, 1, 0)
    # The attempt number distinguishes re-runs of the same task.
    assert make_rollout_id("Blackbox Toy", 3, 1, 1) == "blackbox-toy.00003.r01.a1"


def test_indices_are_zero_padded_for_sortability():
    a = make_rollout_id("d", 2, 0, 0)
    b = make_rollout_id("d", 10, 0, 0)
    assert a == "d.00002.r00.a0" and b == "d.00010.r00.a0"
    # Zero-padding makes the string order match the numeric order.
    assert a < b


def test_slugify_normalizes_and_never_empty():
    assert slugify("SWE Bench / Verified!") == "swe-bench-verified"
    assert slugify("") == "task"
    assert slugify("---") == "task"


def test_is_valid_rollout_id():
    assert is_valid_rollout_id("blackbox-toy.00003.r01.a0")
    assert not is_valid_rollout_id("has spaces")
    assert not is_valid_rollout_id("")
    assert not is_valid_rollout_id("a" * 200)


def test_make_rollout_id_rejects_out_of_charset_dataset():
    # A dataset that slugifies to something valid is fine; slugify guarantees
    # the charset, so a well-formed id always results.
    assert is_valid_rollout_id(make_rollout_id("weird/name!!", 0, 0, 0))


def test_prefix_regex_splits_id_and_remainder():
    rid = "blackbox-toy.00003.r01.a0"
    m = NG_ROLLOUT_PREFIX_RE.match(f"/ng-rollout/{rid}/v1/messages")
    assert m is not None
    assert m.group("rid") == rid
    assert m.group("rest") == "/v1/messages"


def test_prefix_regex_requires_a_remainder_path():
    rid = "blackbox-toy.00003.r01.a0"
    # No trailing path after the id -> no match (nothing to forward).
    assert NG_ROLLOUT_PREFIX_RE.match(f"/ng-rollout/{rid}") is None
    # A non-prefixed path is left alone.
    assert NG_ROLLOUT_PREFIX_RE.match("/v1/messages") is None


def test_header_name_is_stable():
    assert ROLLOUT_ID_HEADER == "x-nemo-gym-rollout-id"


@pytest.mark.parametrize("task,rollout,attempt", [(0, 0, 0), (99999, 99, 9)])
def test_generated_ids_are_always_valid(task, rollout, attempt):
    assert is_valid_rollout_id(make_rollout_id("ds", task, rollout, attempt))
