# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pydantic import TypeAdapter

from benchmarks.gpqa import prepare
from nemo_gym.data.task_data import validate_jsonl_rows
from resources_servers.mcqa.task_data import TaskData


def _source_row(record_id: str, question: str) -> dict:
    return {
        "Record ID": record_id,
        "Question": question,
        "Correct Answer": "right",
        "Incorrect Answer 1": "wrong 1",
        "Incorrect Answer 2": "wrong 2",
        "Incorrect Answer 3": "wrong 3",
    }


def _prepared_rows(monkeypatch, tmp_path) -> list[dict]:
    source_rows = [_source_row("recB", "Question B?"), _source_row("recA", "Question A?")]
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: source_rows)
    monkeypatch.setattr(prepare, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(prepare, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "gpqa_diamond_benchmark.jsonl")

    output_fpath = prepare.prepare()
    return [json.loads(line) for line in output_fpath.read_text().splitlines()]


def test_prepared_rows_carry_the_source_record_id_as_task_id(monkeypatch, tmp_path) -> None:
    rows = _prepared_rows(monkeypatch, tmp_path)

    assert [row["task_id"] for row in rows] == ["recB", "recA"]


def test_prepared_rows_declare_task_id_in_the_mcqa_schema(monkeypatch, tmp_path) -> None:
    lines = [json.dumps(row) for row in _prepared_rows(monkeypatch, tmp_path)]

    report = validate_jsonl_rows("mcqa", TypeAdapter(TaskData), "gpqa_diamond_benchmark.jsonl", lines)

    assert "task_id" not in dict(report.unknown_keys)
