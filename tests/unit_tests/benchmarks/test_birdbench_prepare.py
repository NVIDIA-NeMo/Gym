# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from pathlib import Path

from benchmarks.birdbench import prepare


class _Handle:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def sql_context_for_question(self, question: str) -> str:
        return ""


def test_prepared_rows_carry_the_source_question_id_as_task_id(tmp_path: Path, monkeypatch) -> None:
    databases = tmp_path / "dev_20240627" / "dev_databases"
    (databases / "db1").mkdir(parents=True)
    sqlite3.connect(str(databases / "db1" / "db1.sqlite")).close()
    entries = [
        {"question_id": 7, "db_id": "db1", "question": "b", "SQL": "SELECT 1", "difficulty": "simple"},
        {"question_id": 3, "db_id": "db1", "question": "a", "SQL": "SELECT 2", "difficulty": "simple"},
    ]
    (databases.parent / "dev.json").write_text(json.dumps(entries))
    monkeypatch.setattr(prepare, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(prepare, "ensure_bird_sql", lambda: databases)
    monkeypatch.setattr(prepare, "DbHandle", _Handle)

    rows = [json.loads(line) for line in prepare.prepare().read_text().splitlines()]

    assert [(row["task_id"], row["id"]) for row in rows] == [("7", 0), ("3", 1)]


def test_the_train_uberset_gives_every_row_its_train_json_position() -> None:
    train = [{"db_id": "db1", "question": f"q{i}", "SQL": f"s{i}"} for i in range(3)]
    verified = [{"question_id": 1, "db_id": "db1", "question": "q1", "original_question": None, "SQL": "fixed"}]

    assert [entry["question_id"] for entry in prepare.build_uberset(train, verified)] == [0, 2, 1]
