# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from datasets import Dataset
from pydantic import TypeAdapter

from benchmarks.hle import prepare
from nemo_gym.data.task_data import validate_jsonl_rows
from resources_servers.equivalence_llm_judge.task_data import TaskData


SOURCE = Dataset.from_dict(
    {
        "id": ["id-b", "id-image", "id-a"],
        "question": ["Question B?", "Question with picture?", "Question A?"],
        "image": [None, "data:image/png;base64,AAAA", None],
        "answer": ["b", "i", "a"],
        "answer_type": ["exactMatch", "exactMatch", "exactMatch"],
    }
)


def _prepared_rows(monkeypatch, tmp_path) -> list[dict]:
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: SOURCE)
    monkeypatch.setattr("nemo_gym.config.global_config.get_global_config_dict", lambda: {})
    monkeypatch.setattr(prepare, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "hle_benchmark.jsonl")

    output_fpath = prepare.prepare()
    return [json.loads(line) for line in output_fpath.read_text().splitlines()]


def test_prepared_rows_carry_the_source_id_as_task_id(monkeypatch, tmp_path) -> None:
    rows = _prepared_rows(monkeypatch, tmp_path)

    assert [row["task_id"] for row in rows] == ["id-b", "id-a"]


def test_prepared_rows_declare_task_id_in_the_equivalence_llm_judge_schema(monkeypatch, tmp_path) -> None:
    lines = [json.dumps(row) for row in _prepared_rows(monkeypatch, tmp_path)]

    report = validate_jsonl_rows("equivalence_llm_judge", TypeAdapter(TaskData), "hle_benchmark.jsonl", lines)

    assert "task_id" not in dict(report.unknown_keys)
