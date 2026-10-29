# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pydantic import TypeAdapter

from benchmarks.hle_verified import prepare
from nemo_gym.data.task_data import validate_jsonl_rows
from resources_servers.equivalence_llm_judge.task_data import TaskData


def _source_row(row_id: str, verified_class: str, image: str | None = None) -> dict:
    return {
        "id": row_id,
        "question": f"Question {row_id}?",
        "answer": row_id,
        "answer_type": "exactMatch",
        "category": "Math",
        "raw_subject": "Algebra",
        "Verified_Classes": verified_class,
        "image": image,
    }


def _prepared_rows(monkeypatch, tmp_path) -> list[dict]:
    source_rows = [
        _source_row("id-b", "Gold subset"),
        _source_row("id-image", "Gold subset", image="data:image/png;base64,AAAA"),
        _source_row("id-uncertain", "Uncertain subset"),
        _source_row("id-a", "Revision subset"),
    ]
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: source_rows)
    monkeypatch.setattr("nemo_gym.config.global_config.get_global_config_dict", lambda: {})

    output_fpath = prepare.prepare(output_fpath=tmp_path / "hle_verified_benchmark.jsonl")
    return [json.loads(line) for line in output_fpath.read_text().splitlines()]


def test_prepared_rows_carry_the_source_id_as_task_id(monkeypatch, tmp_path) -> None:
    rows = _prepared_rows(monkeypatch, tmp_path)

    assert [row["task_id"] for row in rows] == ["id-b", "id-a"]


def test_prepared_rows_declare_task_id_in_the_equivalence_llm_judge_schema(monkeypatch, tmp_path) -> None:
    lines = [json.dumps(row) for row in _prepared_rows(monkeypatch, tmp_path)]

    report = validate_jsonl_rows("equivalence_llm_judge", TypeAdapter(TaskData), "hle_verified_benchmark.jsonl", lines)

    assert "task_id" not in dict(report.unknown_keys)
