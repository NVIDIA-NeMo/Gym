# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from resources_servers.nl2repobench.prepare import _copy_task_assets, _write_jsonl
from resources_servers.nl2repobench.task_store import REQUIRED_TASK_FILES, NL2RepoBenchTaskStore


def test_copy_task_assets_copies_only_required_files(task_assets: Path, tmp_path: Path) -> None:
    source_task_dir = task_assets / "example-task"
    # Upstream source directories may carry extra files that must not be copied.
    (source_task_dir / "extra_notes.txt").write_text("do not copy me\n", encoding="utf-8")

    destination = tmp_path / "prepared-tasks"
    _copy_task_assets(task_assets, destination)

    copied_task_dir = destination / "example-task"
    assert copied_task_dir.is_dir()
    for relative_path in REQUIRED_TASK_FILES:
        assert (copied_task_dir / relative_path).is_file()
    assert not (copied_task_dir / "extra_notes.txt").exists()


def test_write_jsonl_produces_expected_row_shape(task_assets: Path, tmp_path: Path) -> None:
    store = NL2RepoBenchTaskStore(task_assets, expected_task_count=1)
    output_path = tmp_path / "benchmark.jsonl"

    _write_jsonl(store, output_path)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])

    assert row["task_id"] == "example-task"
    assert row["image"] == "ghcr.io/multimodal-art-projection/nl2repobench/example-task:1.0"
    assert row["subset"] == "nl2repobench-v1"
    assert row["split"] == "test"

    content = row["responses_create_params"]["input"][0]["content"]
    assert "implement the entire project" in content.lower()
    assert "example-task" in content

    verifier_metadata = row["verifier_metadata"]
    assert verifier_metadata["task_id"] == "example-task"
    assert verifier_metadata["test_case_count"] == 5
    assert verifier_metadata["test_commands"] == [
        "pip install -e .",
        "pytest --continue-on-collection-errors -n 0 tests -v -s",
    ]
    assert verifier_metadata["test_files"] == ["tests"]
