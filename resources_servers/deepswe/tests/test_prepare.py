# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from resources_servers.deepswe.prepare import _write_jsonl
from resources_servers.deepswe.task_store import DeepSWETaskStore


def test_write_jsonl_uses_pinned_task_image(task_assets: Path, tmp_path: Path) -> None:
    store = DeepSWETaskStore(task_assets, expected_task_count=1)
    output_path = tmp_path / "benchmark.jsonl"

    _write_jsonl(store, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["task_id"] == "example-task"
    assert row["verifier_metadata"] == {"task_id": "example-task"}
    assert row["image"] == "public.example/project/example-task:v1.1"
