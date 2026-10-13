# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned DeepSWE v1.1 task assets and Gym JSONL data."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from resources_servers.deepswe.task_store import (
    DEEPSWE_SOURCE_REVISION,
    REQUIRED_TASK_FILES,
    DeepSWETaskStore,
    task_id,
    task_image,
)


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]
DEEPSWE_REPOSITORY = "https://github.com/datacurve-ai/deep-swe"
DEFAULT_SOURCE_DIR = PACKAGE_DIR / "data" / "cache" / "source"
DEFAULT_TASKS_DIR = PACKAGE_DIR / "data" / "cache" / "tasks"
DEFAULT_JSONL = NEMO_GYM_ROOT / "benchmarks" / "deepswe" / "data" / "deepswe_benchmark.jsonl"
DEFAULT_EXAMPLE_JSONL = PACKAGE_DIR / "data" / "example.jsonl"
CACHE_MARKER = ".nemo_gym_deepswe.json"


def _git_output(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def ensure_source(source_dir: str | Path, *, allow_download: bool = True) -> Path:
    """Return an exact checkout of the pinned DeepSWE source revision."""

    path = Path(source_dir).expanduser().resolve()
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"DeepSWE source checkout does not exist: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _git_output("clone", DEEPSWE_REPOSITORY, str(path))
        _git_output("checkout", "--detach", DEEPSWE_SOURCE_REVISION, cwd=path)
    if not (path / ".git").is_dir():
        raise ValueError(f"DeepSWE source is not a Git checkout: {path}")
    revision = _git_output("rev-parse", "HEAD", cwd=path)
    if revision != DEEPSWE_SOURCE_REVISION:
        raise ValueError(f"DeepSWE source {path} is at {revision}; expected pinned revision {DEEPSWE_SOURCE_REVISION}")
    return path


def _copy_task_assets(source_tasks_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=".deepswe-tasks-", dir=destination.parent))
    temporary_tasks = temporary_parent / "tasks"
    temporary_tasks.mkdir()
    try:
        for source_task_dir in sorted(path.parent for path in source_tasks_dir.glob("*/task.toml")):
            target_task_dir = temporary_tasks / source_task_dir.name
            for relative_path in REQUIRED_TASK_FILES:
                source_path = source_task_dir / relative_path
                target_path = target_task_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
        if destination.exists():
            shutil.rmtree(destination)
        temporary_tasks.rename(destination)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def _write_jsonl(
    store: DeepSWETaskStore,
    output_path: Path,
    *,
    limit: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = list(store)
    if limit is not None:
        tasks = tasks[:limit]
    with output_path.open("w", encoding="utf-8") as stream:
        for task in tasks:
            current_task_id = task_id(task)
            row = {
                "task_id": current_task_id,
                "image": task_image(task),
                "responses_create_params": {
                    "input": [{"role": "user", "content": task.instruction}],
                },
                "verifier_metadata": {"task_id": current_task_id},
                "subset": "deepswe-v1.1",
                "split": "test",
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    tasks_dir: str | Path = DEFAULT_TASKS_DIR,
    jsonl_path: str | Path = DEFAULT_JSONL,
    example_jsonl_path: str | Path = DEFAULT_EXAMPLE_JSONL,
    allow_download: bool = True,
) -> tuple[Path, Path]:
    """Materialize verifier-private assets and model-visible benchmark rows."""

    source = ensure_source(source_dir, allow_download=allow_download)
    source_tasks_dir = source / "tasks"
    DeepSWETaskStore(source_tasks_dir)

    prepared_tasks_dir = Path(tasks_dir).expanduser().resolve()
    _copy_task_assets(source_tasks_dir, prepared_tasks_dir)
    marker = {
        "source_repository": DEEPSWE_REPOSITORY,
        "source_revision": DEEPSWE_SOURCE_REVISION,
        "image_source": "task.toml",
    }
    (prepared_tasks_dir / CACHE_MARKER).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    store = DeepSWETaskStore(prepared_tasks_dir)
    output_path = Path(jsonl_path).expanduser().resolve()
    example_output_path = Path(example_jsonl_path).expanduser().resolve()
    _write_jsonl(store, output_path)
    _write_jsonl(store, example_output_path, limit=5)
    return prepared_tasks_dir, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--example-jsonl", type=Path, default=DEFAULT_EXAMPLE_JSONL)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    tasks_dir, jsonl_path = prepare(
        source_dir=args.source_dir,
        tasks_dir=args.tasks_dir,
        jsonl_path=args.jsonl,
        example_jsonl_path=args.example_jsonl,
        allow_download=not args.no_download,
    )
    print(f"Prepared DeepSWE verifier assets in {tasks_dir}")
    print(f"Wrote DeepSWE benchmark data to {jsonl_path}")


if __name__ == "__main__":
    main()
