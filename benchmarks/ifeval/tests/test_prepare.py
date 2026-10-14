# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.ifeval import prepare


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


def test_prepared_rows_carry_the_source_key_as_task_id(tmp_path: Path) -> None:
    def download(url: str, fpath: Path) -> None:
        Path(fpath).write_text("".join(json.dumps(row) + "\n" for row in SOURCE_ROWS))

    with (
        patch.object(prepare, "DATA_DIR", tmp_path),
        patch.object(prepare, "RAW_FPATH", tmp_path / "input_data.jsonl"),
        patch.object(prepare, "OUTPUT_FPATH", tmp_path / "ifeval_benchmark.jsonl"),
        patch.object(prepare.urllib.request, "urlretrieve", download),
    ):
        output_fpath = prepare.prepare()

    rows = [json.loads(line) for line in output_fpath.read_text().splitlines()]
    assert [row["task_id"] for row in rows] == ["1001", "1005"]
    assert [row["id"] for row in rows] == [0, 1]
    assert rows[1]["verifier_metadata"]["kwargs"] == [{"num_bullets": 3}]
