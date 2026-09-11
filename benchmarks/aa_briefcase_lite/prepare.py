# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare AA-Briefcase-Lite rows from a pinned local public checkout."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "aa_briefcase_lite.jsonl"
DATASET_ENV = "AA_BRIEFCASE_LITE_DATASET_DIR"
REVISION_ENV = "AA_BRIEFCASE_LITE_REVISION"
REVISION_MARKER = ".aa-briefcase-lite-revision"
PINNED_REVISION = "4dec557b47d43867a1648c0974db1d8208c8b677"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dataset_revision(dataset_dir: Path) -> str:
    marker = dataset_dir / REVISION_MARKER
    if marker.is_file():
        revision = marker.read_text(encoding="utf-8").strip()
    else:
        try:
            revision = subprocess.run(
                ["git", "-C", str(dataset_dir), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"AA-Briefcase-Lite dataset must be a Git checkout or carry {REVISION_MARKER}: {dataset_dir}"
            ) from exc
    expected = os.environ.get(REVISION_ENV, PINNED_REVISION)
    if revision != expected:
        raise RuntimeError(f"AA-Briefcase-Lite revision mismatch: expected {expected}, found {revision}")
    return revision


def _validate_task_files(dataset_dir: Path, tasks: list[dict]) -> None:
    source_lists = {(tuple(task["shared_files"]), tuple(task["week_files"])) for task in tasks}
    if len(source_lists) != 1:
        raise ValueError("Lite tasks are expected to share one identical shared/week source pool")

    for task in tasks:
        paths = [
            task["task_md_path"],
            task["scenario_overview_path"],
            task["week_overview_path"],
            *task["shared_files"],
            *task["week_files"],
        ]
        for relative in paths:
            candidate = (dataset_dir / relative.rstrip("/")).resolve()
            if not candidate.is_relative_to(dataset_dir) or not candidate.exists():
                raise FileNotFoundError(f"Invalid or missing dataset path for {task['task_id']}: {relative}")


def prepare() -> Path:
    dataset_value = os.environ.get(DATASET_ENV)
    if not dataset_value:
        raise RuntimeError(f"Set {DATASET_ENV} to the pinned ArtificialAnalysis/AA-Briefcase-Lite checkout")
    dataset_dir = Path(dataset_value).resolve()
    tasks_path = dataset_dir / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"AA-Briefcase-Lite tasks file not found: {tasks_path}")

    revision = _dataset_revision(dataset_dir)
    tasks = _read_jsonl(tasks_path)
    if {task["task_id"] for task in tasks} != {"w1_t1", "w1_t2", "w1_t3", "w1_t4"}:
        raise ValueError("Expected the four public AA-Briefcase-Lite task IDs")
    _validate_task_files(dataset_dir, tasks)

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w", encoding="utf-8") as output:
        for task in tasks:
            record = {
                "responses_create_params": {"input": []},
                "task_id": task["task_id"],
                "week": task["week"],
                "dataset_dir": str(dataset_dir),
                "dataset_revision": revision,
                "task_md_path": task["task_md_path"],
                "deliverable_filenames": task["deliverable_filenames"],
                "shared_files": task["shared_files"],
                "week_files": task["week_files"],
                "scenario_overview_path": task["scenario_overview_path"],
                "week_overview_path": task["week_overview_path"],
            }
            output.write(json.dumps(record) + "\n")

    print(f"Wrote {len(tasks)} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
