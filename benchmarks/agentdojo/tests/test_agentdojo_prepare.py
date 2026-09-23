# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.agentdojo.prepare import SUITE_COUNTS, _row


def test_materialized_matrix_count_is_derived_from_upstream() -> None:
    expected = sum(user_count * (1 + injection_count) for user_count, injection_count in SUITE_COUNTS.values())

    assert expected == 1046


def test_clean_and_attacked_rows_are_unambiguous() -> None:
    clean = _row(suite="banking", user_task_id="user_task_0")
    attacked = _row(
        suite="banking",
        user_task_id="user_task_0",
        injection_task_id="injection_task_0",
    )

    assert clean["condition"] == "clean"
    assert clean["attack"] is None
    assert clean["injection_task_id"] is None
    assert attacked["condition"] == "attacked"
    assert attacked["attack"] == "important_instructions"
    assert attacked["injection_task_id"] == "injection_task_0"
