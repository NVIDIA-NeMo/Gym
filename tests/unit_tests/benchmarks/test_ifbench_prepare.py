# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path
from unittest.mock import patch

from benchmarks.ifbench import prepare


METRICS_FPATH = Path(prepare.__file__).parent / "data" / "ifbench_benchmark_metrics.json"
PROMPTS = 300  # rows of IFBench_test.jsonl at the pinned commit


def test_the_download_is_pinned_to_a_commit() -> None:
    assert re.search(r"/allenai/IFBench/[0-9a-f]{40}/data/IFBench_test\.jsonl$", prepare.URL)


def test_the_committed_metrics_describe_the_pinned_prompts() -> None:
    metrics = json.loads(METRICS_FPATH.read_text())
    assert metrics["Number of examples"] == PROMPTS * metrics["num_repeats"]
    assert metrics["prompt"]["unique_count"] == PROMPTS
    assert metrics["id"]["Max"] == PROMPTS - 1


def test_prepared_rows_carry_the_source_key_as_task_id(tmp_path: Path) -> None:
    source = [
        {"key": "7", "prompt": "a", "instruction_id_list": ["count:keywords_multiple"], "kwargs": [{}]},
        {"key": "3", "prompt": "b", "instruction_id_list": ["count:word_count_range"], "kwargs": [{}]},
    ]

    def download(url: str, fpath: Path) -> None:
        Path(fpath).write_text("".join(json.dumps(row) + "\n" for row in source))

    with (
        patch.object(prepare, "DATA_DIR", tmp_path),
        patch.object(prepare, "RAW_FPATH", tmp_path / "IFBench_test.jsonl"),
        patch.object(prepare, "OUTPUT_FPATH", tmp_path / "ifbench_benchmark.jsonl"),
        patch.object(prepare.urllib.request, "urlretrieve", download),
    ):
        rows = [json.loads(line) for line in prepare.prepare().read_text().splitlines()]

    assert [(row["task_id"], row["id"]) for row in rows] == [("7", 0), ("3", 1)]
