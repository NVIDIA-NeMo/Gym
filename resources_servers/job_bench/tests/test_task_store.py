# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from resources_servers.job_bench.task_store import (
    JobBenchTaskError,
    JobBenchTaskStore,
    build_task_id,
    discover_tasks,
    load_rubrics,
)


def test_discover_indexes_task_by_split_profession_and_task(cache_dir: Path, task_id: str) -> None:
    tasks = discover_tasks(cache_dir, "main")

    assert list(tasks) == [task_id]
    task = tasks[task_id]
    assert task.split == "main"
    assert task.profession == "biostatisticians"
    assert task.task_name == "task1"
    assert task.instructions == "Analyze the cohort.\n"


def test_task_exposes_rubric_weights_and_a_stable_digest(cache_dir: Path, task_id: str) -> None:
    task = discover_tasks(cache_dir, "main")[task_id]

    assert len(task.rubrics) == 2
    assert task.max_score == 15
    assert task.rubrics_sha256 == task.rubrics_sha256
    assert len(task.rubrics_sha256) == 64


def test_label_is_sandbox_metadata_safe(cache_dir: Path, task_id: str) -> None:
    task = discover_tasks(cache_dir, "main")[task_id]

    # Slashes are not valid in a provider label; they must be replaced, not dropped.
    assert task.label == "main-biostatisticians-task1"
    assert len(task.label) <= 63


def test_store_rejects_a_short_cache(cache_dir: Path) -> None:
    with pytest.raises(JobBenchTaskError, match="expected 2"):
        JobBenchTaskStore(cache_dir, "main", expected_task_count=2)


def test_store_rejects_an_unknown_split(cache_dir: Path) -> None:
    with pytest.raises(JobBenchTaskError, match="Unknown JobBench split"):
        JobBenchTaskStore(cache_dir, "hard", expected_task_count=1)


def test_store_reports_a_missing_split_directory(tmp_path: Path) -> None:
    with pytest.raises(JobBenchTaskError, match="not found"):
        JobBenchTaskStore(tmp_path, "main", expected_task_count=1)


def test_store_get_reports_unknown_task(cache_dir: Path, task_id: str) -> None:
    store = JobBenchTaskStore(cache_dir, "main", expected_task_count=1)

    assert task_id in store
    assert len(store) == 1
    assert [task.task_id for task in store.values()] == [task_id]
    with pytest.raises(JobBenchTaskError, match="Unknown JobBench task"):
        store.get("main/lawyers/task9")


def test_discover_rejects_a_task_missing_rubrics(cache_dir: Path, task_id: str) -> None:
    (cache_dir / "main" / "biostatisticians" / "task1" / "RUBRICS.json").unlink()

    with pytest.raises(JobBenchTaskError, match="missing RUBRICS.json"):
        discover_tasks(cache_dir, "main")


def test_discover_rejects_a_task_missing_instructions(cache_dir: Path) -> None:
    (cache_dir / "main" / "biostatisticians" / "task1" / "task_folder" / "TASK_INSTRUCTIONS.txt").unlink()

    with pytest.raises(JobBenchTaskError, match="TASK_INSTRUCTIONS.txt"):
        discover_tasks(cache_dir, "main")


def test_discover_ignores_directories_that_are_not_tasks(cache_dir: Path, task_id: str) -> None:
    (cache_dir / "main" / "biostatisticians" / "notes").mkdir()

    assert list(discover_tasks(cache_dir, "main")) == [task_id]


def test_load_rubrics_accepts_the_alternate_upstream_key(tmp_path: Path) -> None:
    path = tmp_path / "RUBRICS.json"
    path.write_text(json.dumps({"evaluation_rubrics": [{"rubric": "r", "weight": 3, "criterion": ["c"]}]}))

    assert load_rubrics(path)[0]["weight"] == 3


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"rubrics": []}, "non-empty"),
        ({"other": [1]}, "non-empty"),
        ({"rubrics": ["not-an-object"]}, "not an object"),
        ({"rubrics": [{"rubric": "r", "weight": "ten"}]}, "non-numeric weight"),
    ],
)
def test_load_rubrics_rejects_malformed_payloads(tmp_path: Path, payload: dict, message: str) -> None:
    path = tmp_path / "RUBRICS.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(JobBenchTaskError, match=message):
        load_rubrics(path)


def test_load_rubrics_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "RUBRICS.json"
    path.write_text("{not json")

    with pytest.raises(JobBenchTaskError, match="Malformed rubrics JSON"):
        load_rubrics(path)


def test_load_rubrics_reports_an_unreadable_file(tmp_path: Path) -> None:
    with pytest.raises(JobBenchTaskError, match="Cannot read rubrics"):
        load_rubrics(tmp_path / "absent.json")


def test_build_task_id_joins_the_three_components() -> None:
    assert build_task_id("easy", "lawyers", "task2") == "easy/lawyers/task2"
