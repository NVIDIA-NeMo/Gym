# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from benchmarks.agentdyn.prepare import (
    EXPECTED_ATTACKED_CASES,
    EXPECTED_USER_TASKS,
    SUITE_COUNTS,
    _row,
)


def test_agentdyn_matrix_matches_paper_scope() -> None:
    clean = sum(user_count for user_count, _ in SUITE_COUNTS.values())
    attacked = sum(user_count * injection_count for user_count, injection_count in SUITE_COUNTS.values())
    assert clean == EXPECTED_USER_TASKS == 60
    assert attacked == EXPECTED_ATTACKED_CASES == 560


def test_rows_keep_defense_orthogonal_to_task_matrix() -> None:
    row = _row(
        suite="dailylife",
        user_task_id="user_task_0",
        injection_task_id="injection_task_0",
    )
    assert row["defense"] is None
    assert row["attack"] == "important_instructions"
    assert row["condition"] == "attacked"
