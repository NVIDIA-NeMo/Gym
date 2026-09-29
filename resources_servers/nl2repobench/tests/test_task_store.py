# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from resources_servers.nl2repobench.task_schema import Task
from resources_servers.nl2repobench.task_store import NL2RepoBenchTaskStore, task_id, task_image


def test_load_task_store(task_assets: Path) -> None:
    store = NL2RepoBenchTaskStore(task_assets, expected_task_count=1)
    task = store.get("example-task")

    assert len(store) == 1
    assert isinstance(task, Task)
    assert task_id(task) == "example-task"
    assert task_image(task) == "ghcr.io/multimodal-art-projection/nl2repobench/example-task:1.0"
    assert task.test_case_count == 5
    assert task.test_commands.commands[0] == "pip install -e ."
    assert task.test_files.files == ["tests"]
    assert list(store) == [task]


def test_missing_required_file_raises(task_assets: Path) -> None:
    (task_assets / "example-task" / "test_files.json").unlink()

    with pytest.raises(FileNotFoundError, match="missing required files"):
        NL2RepoBenchTaskStore(task_assets, expected_task_count=1)


def test_malformed_json_raises(task_assets: Path) -> None:
    (task_assets / "example-task" / "test_commands.json").write_text("not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        NL2RepoBenchTaskStore(task_assets, expected_task_count=1)


def test_non_integer_test_case_count_raises(task_assets: Path) -> None:
    (task_assets / "example-task" / "test_case_count.txt").write_text("not-a-number", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid test_case_count.txt"):
        NL2RepoBenchTaskStore(task_assets, expected_task_count=1)


def test_task_count_mismatch_raises(task_assets: Path) -> None:
    with pytest.raises(ValueError, match="Expected 2 NL2RepoBench tasks"):
        NL2RepoBenchTaskStore(task_assets, expected_task_count=2)


def test_two_task_store(two_task_assets: Path) -> None:
    store = NL2RepoBenchTaskStore(two_task_assets, expected_task_count=2)

    assert len(store) == 2
    assert set(store.task_ids) == {"example-task-one", "example-task-two"}
