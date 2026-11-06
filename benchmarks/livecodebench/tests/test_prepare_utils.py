# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from benchmarks.livecodebench import prepare_utils


def _source_row(question_id: str, contest_date: str) -> dict:
    return {
        "question_id": question_id,
        "question_content": f"Solve {question_id}.",
        "contest_date": contest_date,
        "difficulty": "easy",
        "starter_code": "",
        "metadata": "{}",
        "public_test_cases": json.dumps([{"input": "1\n", "output": "1\n"}]),
        "private_test_cases": "",
    }


def _prepared_rows(monkeypatch, tmp_path) -> list[dict]:
    source_rows = [
        _source_row("abc365_b", "2024-08-10"),
        _source_row("abc300_z", "2023-01-01"),
        _source_row("abc365_a", "2024-08-10"),
    ]
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: source_rows)

    output_fpath = prepare_utils.prepare_from_hf_raw(
        tmp_path / "livecodebench_validation.jsonl", release_version="release_v6", date_from="2024-08-01"
    )
    return [json.loads(line) for line in output_fpath.read_text().splitlines()]


def test_prepared_rows_carry_the_source_question_id_as_task_id(monkeypatch, tmp_path) -> None:
    rows = _prepared_rows(monkeypatch, tmp_path)

    assert [row["task_id"] for row in rows] == ["abc365_b", "abc365_a"]
    assert [row["verifier_metadata"]["problem_id"] for row in rows] == ["abc365_b", "abc365_a"]
