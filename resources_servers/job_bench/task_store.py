# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local task store for the prepared JobBench control-plane cache.

``prepare.py`` materializes the upstream Hugging Face dataset into

    <cache_dir>/<split>/<profession>/<taskN>/
        task_folder/                 # the agent's working materials
        files_required_to_search/    # main split only; withheld from the agent by default
        RUBRICS.json                 # grading rubrics (never given to the agent)
        task_card.md                 # human-readable brief (never given to the agent)

Rubrics and search files are the answer key, so they stay in the gitignored cache
on the resources server and never travel in the model-visible JSONL.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


SPLITS = ("main", "easy")

# Task counts published by JobBench; a short count means a partial cache.
EXPECTED_TASK_COUNTS = {"main": 65, "easy": 63}

# Sandbox layout. These are baked into the agent prompt at prepare time and
# recreated by the resources server at seed time, so both must agree.
WORKSPACE_DIR = "/workspace"
TASK_FOLDER_DIR = f"{WORKSPACE_DIR}/task_folder"
OUTPUT_DIR = f"{WORKSPACE_DIR}/output"
SEARCH_FILES_DIR = f"{WORKSPACE_DIR}/files_required_to_search"

INSTRUCTIONS_FILENAME = "TASK_INSTRUCTIONS.txt"
RUBRICS_FILENAME = "RUBRICS.json"
TASK_CARD_FILENAME = "task_card.md"
TASK_FOLDER_NAME = "task_folder"
SEARCH_FILES_NAME = "files_required_to_search"

_TASK_DIR_PATTERN = re.compile(r"^task\d+$")
_LABEL_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class JobBenchTaskError(RuntimeError):
    """The prepared cache is missing or malformed for a requested task."""


@dataclass(frozen=True)
class JobBenchTask:
    """One prepared JobBench task rooted at ``task_dir``."""

    task_id: str
    split: str
    profession: str
    task_name: str
    task_dir: Path

    @property
    def task_folder(self) -> Path:
        return self.task_dir / TASK_FOLDER_NAME

    @property
    def search_files_dir(self) -> Path:
        return self.task_dir / SEARCH_FILES_NAME

    @property
    def rubrics_path(self) -> Path:
        return self.task_dir / RUBRICS_FILENAME

    @property
    def instructions_path(self) -> Path:
        return self.task_folder / INSTRUCTIONS_FILENAME

    @cached_property
    def rubrics(self) -> list[dict]:
        return load_rubrics(self.rubrics_path)

    @cached_property
    def rubrics_sha256(self) -> str:
        return hashlib.sha256(self.rubrics_path.read_bytes()).hexdigest()

    @property
    def max_score(self) -> int:
        return sum(int(rubric.get("weight", 0)) for rubric in self.rubrics)

    @property
    def instructions(self) -> str:
        return self.instructions_path.read_text(encoding="utf-8", errors="replace")

    @property
    def label(self) -> str:
        """A value safe to use as a sandbox metadata label."""
        return _LABEL_SAFE_PATTERN.sub("-", self.task_id)[:63]


def load_rubrics(rubrics_path: Path) -> list[dict]:
    """Read a ``RUBRICS.json`` file, accepting both upstream key spellings."""
    try:
        payload = json.loads(rubrics_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise JobBenchTaskError(f"Cannot read rubrics at {rubrics_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise JobBenchTaskError(f"Malformed rubrics JSON at {rubrics_path}: {error}") from error

    # run_judge.sh accepts either `.rubrics` or `.evaluation_rubrics`.
    rubrics = payload.get("rubrics")
    if rubrics is None:
        rubrics = payload.get("evaluation_rubrics")
    if not isinstance(rubrics, list) or not rubrics:
        raise JobBenchTaskError(f"No non-empty 'rubrics' array in {rubrics_path}")

    for index, rubric in enumerate(rubrics):
        if not isinstance(rubric, dict):
            raise JobBenchTaskError(f"Rubric {index} in {rubrics_path} is not an object")
        if not isinstance(rubric.get("weight", 0), (int, float)):
            raise JobBenchTaskError(f"Rubric {index} in {rubrics_path} has a non-numeric weight")
    return rubrics


def build_task_id(split: str, profession: str, task_name: str) -> str:
    return f"{split}/{profession}/{task_name}"


def discover_tasks(cache_dir: Path, split: str) -> dict[str, JobBenchTask]:
    """Index every ``<profession>/<taskN>`` directory under ``<cache_dir>/<split>``."""
    split_root = cache_dir / split
    if not split_root.is_dir():
        raise JobBenchTaskError(
            f"JobBench cache for split {split!r} not found at {split_root}. "
            "Run `python -m resources_servers.job_bench.prepare` first."
        )

    tasks: dict[str, JobBenchTask] = {}
    for profession_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        for task_dir in sorted(t for t in profession_dir.iterdir() if t.is_dir()):
            if not _TASK_DIR_PATTERN.match(task_dir.name):
                continue
            task = JobBenchTask(
                task_id=build_task_id(split, profession_dir.name, task_dir.name),
                split=split,
                profession=profession_dir.name,
                task_name=task_dir.name,
                task_dir=task_dir,
            )
            if not task.rubrics_path.is_file():
                raise JobBenchTaskError(f"Task {task.task_id} is missing {RUBRICS_FILENAME}")
            if not task.instructions_path.is_file():
                raise JobBenchTaskError(f"Task {task.task_id} is missing {TASK_FOLDER_NAME}/{INSTRUCTIONS_FILENAME}")
            tasks[task.task_id] = task
    return tasks


class JobBenchTaskStore:
    """Read-only index over a prepared JobBench cache for a single split."""

    def __init__(self, cache_dir: Path, split: str, *, expected_task_count: int | None = None) -> None:
        if split not in SPLITS:
            raise JobBenchTaskError(f"Unknown JobBench split {split!r}; expected one of {SPLITS}")
        self.cache_dir = cache_dir
        self.split = split
        self._tasks = discover_tasks(cache_dir, split)

        expected = expected_task_count if expected_task_count is not None else EXPECTED_TASK_COUNTS[split]
        if len(self._tasks) != expected:
            raise JobBenchTaskError(
                f"JobBench split {split!r} has {len(self._tasks)} tasks in {cache_dir}, expected {expected}. "
                "Re-run prepare.py (use FORCE to re-download a partial cache)."
            )

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: object) -> bool:
        return task_id in self._tasks

    def get(self, task_id: str) -> JobBenchTask:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise JobBenchTaskError(f"Unknown JobBench task {task_id!r} in split {self.split!r}") from None

    def values(self) -> list[JobBenchTask]:
        return list(self._tasks.values())
