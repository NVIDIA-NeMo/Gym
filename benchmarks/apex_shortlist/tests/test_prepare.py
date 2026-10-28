# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pydantic import TypeAdapter

from benchmarks.apex_shortlist import prepare
from nemo_gym.data.task_data import validate_jsonl_rows
from resources_servers.math_with_judge.task_data import TaskData


SOURCE_ROWS = [
    {"problem_idx": 7, "problem": "Compute 1+1.", "answer": 2},
    {"problem_idx": 3, "problem": "Compute 2+2.", "answer": 4},
]


def _prepared_rows(monkeypatch, tmp_path) -> list[dict]:
    monkeypatch.setattr(prepare, "load_dataset", lambda *args, **kwargs: SOURCE_ROWS)
    monkeypatch.setattr(prepare, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "benchmark.jsonl")

    output_fpath = prepare.prepare()
    return [json.loads(line) for line in output_fpath.read_text().splitlines()]


def test_prepared_rows_carry_the_source_problem_idx_as_task_id(monkeypatch, tmp_path) -> None:
    rows = _prepared_rows(monkeypatch, tmp_path)

    assert [row["task_id"] for row in rows] == ["7", "3"]
    assert [row["expected_answer"] for row in rows] == ["2", "4"]


def test_prepared_rows_declare_task_id_in_the_math_with_judge_schema(monkeypatch, tmp_path) -> None:
    lines = [json.dumps(row) for row in _prepared_rows(monkeypatch, tmp_path)]

    report = validate_jsonl_rows("math_with_judge", TypeAdapter(TaskData), "benchmark.jsonl", lines)

    assert report.clean
    assert "task_id" not in dict(report.unknown_keys)
