# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare evaluation-only SWE-Atlas QA rows from the pinned upstream checkout."""

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
TASKS_DIR = PACKAGE_DIR / "data/source/data/qa"
DATASET_PATH = PACKAGE_DIR / "data/benchmark.jsonl"
SOURCE_REVISION = "c6673fe259c3493a2f834266abf63247a5d1507c"  # pragma: allowlist secret
SOURCE_REPOSITORY = "https://github.com/scaleapi/SWE-Atlas.git"
EXPECTED_TASK_COUNT = 124


def load_task(tasks_dir: Path, task_id: str) -> dict:
    """Read a task's TOML config, rejecting invalid IDs and paths outside tasks_dir."""
    task_dir = (tasks_dir / task_id).resolve()
    if not re.fullmatch(r"task-[0-9a-f]{24}", task_id) or task_dir.parent != tasks_dir.resolve():
        raise ValueError(f"Invalid SWE-Atlas task ID: {task_id!r}")
    with (task_dir / "task.toml").open("rb") as stream:
        return tomllib.load(stream)


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, errors="replace"
    ).stdout.strip()


def validate_source(source_dir: Path) -> None:
    """Reject a different revision or modified private evaluator assets."""
    revision = _git("rev-parse", "HEAD", cwd=source_dir)
    if revision != SOURCE_REVISION:
        raise ValueError(f"SWE-Atlas source revision is {revision}, expected {SOURCE_REVISION}")
    if _git("status", "--porcelain=v1", "--untracked-files=all", "--", "data/qa", cwd=source_dir):
        raise ValueError("SWE-Atlas source has uncommitted or untracked changes under data/qa")


def prepare(*, source_dir: Path = PACKAGE_DIR / "data/source", output_path: Path = DATASET_PATH) -> Path:
    """Clone if absent, validate the pinned clean QA source, and write evaluation rows."""
    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", SOURCE_REPOSITORY, str(source_dir))
        _git("checkout", "--detach", SOURCE_REVISION, cwd=source_dir)
    validate_source(source_dir)
    tasks_dir = source_dir / "data/qa"
    task_dirs = sorted(path for path in tasks_dir.iterdir() if path.is_dir())
    if len(task_dirs) != EXPECTED_TASK_COUNT:
        raise ValueError(f"Expected {EXPECTED_TASK_COUNT} SWE-Atlas QA tasks, found {len(task_dirs)}")
    rows = []
    for task_dir in task_dirs:
        load_task(tasks_dir, task_dir.name)
        rows.append(
            {
                "task_id": task_dir.name,
                "responses_create_params": {
                    "input": [{"role": "user", "content": (task_dir / "instruction.md").read_text(encoding="utf-8")}],
                },
                "verifier_metadata": {"task_id": task_dir.name},
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    """Prepare the benchmark using optional source and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PACKAGE_DIR / "data/source")
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    args = parser.parse_args()
    print(prepare(source_dir=args.source_dir, output_path=args.output))


if __name__ == "__main__":
    main()
