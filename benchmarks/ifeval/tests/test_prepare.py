# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

from pydantic import TypeAdapter

from benchmarks.ifeval import prepare
from nemo_gym.task_data import validate_jsonl_rows
from resources_servers.instruction_following.task_data import TaskData


SOURCE_ROWS = [
    {
        "key": 1001,
        "prompt": "Write in lowercase.",
        "instruction_id_list": ["change_case:english_lowercase"],
        "kwargs": [{}],
    },
    {
        "key": 1005,
        "prompt": "Use 3 bullets.",
        "instruction_id_list": ["detectable_format:number_bullet_lists"],
        "kwargs": [{"num_bullets": 3}],
    },
]


def _prepared_rows(tmp_path: Path) -> list[dict]:
    def download(url: str, fpath: Path) -> None:
        Path(fpath).write_text("".join(json.dumps(row) + "\n" for row in SOURCE_ROWS))

    with (
        patch.object(prepare, "DATA_DIR", tmp_path),
        patch.object(prepare, "RAW_FPATH", tmp_path / "input_data.jsonl"),
        patch.object(prepare, "OUTPUT_FPATH", tmp_path / "ifeval_benchmark.jsonl"),
        patch.object(prepare.urllib.request, "urlretrieve", download),
    ):
        output_fpath = prepare.prepare()
    return [json.loads(line) for line in output_fpath.read_text().splitlines()]


def _validate(rows: list[dict]):
    lines = [json.dumps(row) for row in rows]
    return validate_jsonl_rows("instruction_following", TypeAdapter(TaskData), "ifeval_benchmark.jsonl", lines)


def test_prepared_rows_carry_the_source_key_as_task_id(tmp_path: Path) -> None:
    rows = _prepared_rows(tmp_path)

    assert [row["task_id"] for row in rows] == ["1001", "1005"]
    assert [row["id"] for row in rows] == [0, 1]
    assert rows[1]["verifier_metadata"]["kwargs"] == [{"num_bullets": 3}]


def test_prepared_rows_are_clean_under_the_instruction_following_schema(tmp_path: Path) -> None:
    report = _validate(_prepared_rows(tmp_path))

    assert report.clean
    assert dict(report.unknown_keys) == {}


def test_rows_without_task_id_stay_clean_and_misspelled_keys_are_still_reported(tmp_path: Path) -> None:
    rows = _prepared_rows(tmp_path)
    without_task_id = [{key: value for key, value in row.items() if key != "task_id"} for row in rows]
    misspelled = [{**row, "task_di": "1001"} for row in without_task_id]

    assert _validate(without_task_id).clean
    assert dict(_validate(misspelled).unknown_keys) == {"task_di": 2}
