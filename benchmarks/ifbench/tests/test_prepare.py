# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

from pydantic import TypeAdapter

from benchmarks.ifbench import prepare
from nemo_gym.data import validate_jsonl_rows
from resources_servers.ifbench.task_data import TaskData


SOURCE_ROWS = [
    {
        "key": "7",
        "prompt": "Include keyword nebula twice.",
        "instruction_id_list": ["count:keywords_multiple"],
        "kwargs": [{"keyword1": "nebula"}],
    },
    {
        "key": "3",
        "prompt": "Answer in 20 words.",
        "instruction_id_list": ["count:word_count_range"],
        "kwargs": [{"N": 20.0}],
    },
]


def _prepared_rows(tmp_path: Path) -> list[dict]:
    def download(url: str, fpath: Path) -> None:
        Path(fpath).write_text("".join(json.dumps(row) + "\n" for row in SOURCE_ROWS))

    with (
        patch.object(prepare, "DATA_DIR", tmp_path),
        patch.object(prepare, "RAW_FPATH", tmp_path / "IFBench_test.jsonl"),
        patch.object(prepare, "OUTPUT_FPATH", tmp_path / "ifbench_benchmark.jsonl"),
        patch.object(prepare.urllib.request, "urlretrieve", download),
    ):
        output_fpath = prepare.prepare()
    return [json.loads(line) for line in output_fpath.read_text().splitlines()]


def _validate(rows: list[dict]):
    lines = [json.dumps(row) for row in rows]
    return validate_jsonl_rows("ifbench", TypeAdapter(TaskData), "ifbench_benchmark.jsonl", lines)


def test_prepared_rows_carry_the_source_key_as_task_id(tmp_path: Path) -> None:
    rows = _prepared_rows(tmp_path)

    assert [row["task_id"] for row in rows] == ["7", "3"]
    assert [row["id"] for row in rows] == [0, 1]
    assert rows[1]["kwargs"] == [{"N": 20.0}]


def test_prepared_rows_are_clean_under_the_ifbench_schema(tmp_path: Path) -> None:
    report = _validate(_prepared_rows(tmp_path))

    assert report.clean
    assert dict(report.unknown_keys) == {}


def test_rows_without_task_id_stay_clean_and_misspelled_keys_are_still_reported(tmp_path: Path) -> None:
    rows = _prepared_rows(tmp_path)
    without_task_id = [{key: value for key, value in row.items() if key != "task_id"} for row in rows]
    misspelled = [{**row, "task_di": "7"} for row in without_task_id]

    assert _validate(without_task_id).clean
    assert dict(_validate(misspelled).unknown_keys) == {"task_di": 2}
