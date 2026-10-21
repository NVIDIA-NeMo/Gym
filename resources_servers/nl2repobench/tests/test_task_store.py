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


def test_task_image_registry_template_overrides_default(task_assets: Path) -> None:
    store = NL2RepoBenchTaskStore(task_assets, expected_task_count=1)
    task = store.get("example-task")

    image = task_image(task, registry_template="my-registry.example.com/org/{name}:1.0")

    assert image == "my-registry.example.com/org/example-task:1.0"


def test_task_image_registry_template_lowercases_name(tmp_path: Path) -> None:
    # Not using the shared _write_task helper from conftest.py here: importing it via
    # `from tests.conftest import ...` collides with Gym's own root-level tests/conftest.py (both
    # packages are named "tests") when the full suite is run from the Gym repo root rather than
    # scoped to this directory - confirmed the hard way (ImportError under `pytest tests/`, passed
    # under `pytest resources_servers/nl2repobench/tests/test_task_store.py`).
    task_dir = tmp_path / "tasks" / "MixedCase_Task-1"
    task_dir.mkdir(parents=True)
    (task_dir / "start.md").write_text("# MixedCase_Task-1\n", encoding="utf-8")
    (task_dir / "test_case_count.txt").write_text("5", encoding="utf-8")
    (task_dir / "test_commands.json").write_text(
        json.dumps(["pip install -e .", "pytest --continue-on-collection-errors -n 0 tests -v -s"]),
        encoding="utf-8",
    )
    (task_dir / "test_files.json").write_text(json.dumps(["tests"]), encoding="utf-8")

    tasks_dir = tmp_path / "tasks"
    store = NL2RepoBenchTaskStore(tasks_dir, expected_task_count=1)
    task = store.get("MixedCase_Task-1")

    # Docker/OCI image references must be lowercase - a custom registry_template must not emit
    # an uppercase reference, which would fail to pull.
    image = task_image(task, registry_template="my-registry.example.com/org/{name}:1.0")

    assert image == "my-registry.example.com/org/mixedcase_task-1:1.0"
    # The default (upstream) registry is left untouched by this - task.name itself is unaffected.
    assert task.name == "MixedCase_Task-1"


def test_two_task_store(two_task_assets: Path) -> None:
    store = NL2RepoBenchTaskStore(two_task_assets, expected_task_count=2)

    assert len(store) == 2
    assert set(store.task_ids) == {"example-task-one", "example-task-two"}
