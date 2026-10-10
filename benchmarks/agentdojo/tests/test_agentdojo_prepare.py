# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from benchmarks.agentdojo import prepare as prepare_module
from benchmarks.agentdojo.prepare import SUITE_TASKS, _row


def test_materialized_matrix_count_is_derived_from_upstream() -> None:
    expected = sum(user_count * (1 + len(injections)) for user_count, injections in SUITE_TASKS.values())

    assert expected == 1046


def test_prepared_rows_are_unique_and_use_slack_injection_numbering(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(prepare_module, "OUTPUT_PATH", tmp_path / "agentdojo_benchmark.jsonl")
    rows = [json.loads(line) for line in prepare_module.prepare().read_text().splitlines()]
    selectors = {(row["suite"], row["user_task_id"], row["injection_task_id"]) for row in rows}

    assert len(rows) == len(selectors) == 1046
    slack_injections = {
        row["injection_task_id"] for row in rows if row["suite"] == "slack" and row["injection_task_id"]
    }
    assert slack_injections == {f"injection_task_{number}" for number in range(1, 6)}


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
