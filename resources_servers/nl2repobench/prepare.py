# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned NL2RepoBench task assets and Gym JSONL data."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from resources_servers.nl2repobench.task_store import (
    NL2REPOBENCH_SOURCE_REVISION,
    REQUIRED_TASK_FILES,
    NL2RepoBenchTaskStore,
    task_id,
    task_image,
)


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]
NL2REPOBENCH_REPOSITORY_URL = "https://github.com/multimodal-art-projection/NL2RepoBench"
DEFAULT_SOURCE_DIR = PACKAGE_DIR / "data" / "cache" / "source"
DEFAULT_TASKS_DIR = PACKAGE_DIR / "data" / "cache" / "tasks"
DEFAULT_JSONL = NEMO_GYM_ROOT / "benchmarks" / "nl2repobench" / "data" / "nl2repobench_benchmark.jsonl"
CACHE_MARKER = ".nemo_gym_nl2repobench.json"

FIXED_INSTRUCTION = (
    "According to the start.md in the workspace, implement the entire project as per the "
    "requirements described in start.md."
)


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


def ensure_source(
    source_dir: str | Path,
    revision: str = NL2REPOBENCH_SOURCE_REVISION,
    *,
    allow_download: bool = True,
) -> Path:
    """Return an exact checkout of the pinned NL2RepoBench source revision."""

    path = Path(source_dir).expanduser().resolve()
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"NL2RepoBench source checkout does not exist: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _git_output("clone", NL2REPOBENCH_REPOSITORY_URL, str(path))
        _git_output("checkout", "--detach", revision, cwd=path)
    if not (path / ".git").is_dir():
        raise ValueError(f"NL2RepoBench source is not a Git checkout: {path}")
    current_revision = _git_output("rev-parse", "HEAD", cwd=path)
    if current_revision != revision:
        raise ValueError(f"NL2RepoBench source {path} is at {current_revision}; expected pinned revision {revision}")
    return path


def _copy_task_assets(source_test_files_dir: Path, destination_tasks_dir: Path) -> None:
    destination_tasks_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=".nl2repobench-tasks-", dir=destination_tasks_dir.parent))
    temporary_tasks = temporary_parent / "tasks"
    temporary_tasks.mkdir()
    try:
        for source_task_dir in sorted(path for path in source_test_files_dir.iterdir() if path.is_dir()):
            target_task_dir = temporary_tasks / source_task_dir.name
            target_task_dir.mkdir(parents=True, exist_ok=True)
            for relative_path in REQUIRED_TASK_FILES:
                source_path = source_task_dir / relative_path
                target_path = target_task_dir / relative_path
                shutil.copy2(source_path, target_path)
        if destination_tasks_dir.exists():
            shutil.rmtree(destination_tasks_dir)
        temporary_tasks.rename(destination_tasks_dir)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def _write_jsonl(store: NL2RepoBenchTaskStore, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for task in store:
            current_task_id = task_id(task)
            row = {
                "task_id": current_task_id,
                "image": task_image(task),
                "responses_create_params": {
                    "input": [{"role": "user", "content": f"{FIXED_INSTRUCTION}\n\n{task.start_md}"}],
                },
                "verifier_metadata": {
                    "task_id": current_task_id,
                    "test_commands": task.test_commands.commands,
                    "test_case_count": task.test_case_count,
                    "test_files": task.test_files.files,
                },
                "subset": "nl2repobench-v1",
                "split": "test",
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    tasks_dir: str | Path = DEFAULT_TASKS_DIR,
    jsonl_path: str | Path = DEFAULT_JSONL,
    source_revision: str = NL2REPOBENCH_SOURCE_REVISION,
    allow_download: bool = True,
) -> tuple[Path, Path]:
    """Materialize verifier-private assets and model-visible benchmark rows."""

    source = ensure_source(source_dir, source_revision, allow_download=allow_download)
    source_test_files_dir = source / "test_files"

    prepared_tasks_dir = Path(tasks_dir).expanduser().resolve()
    _copy_task_assets(source_test_files_dir, prepared_tasks_dir)
    marker = {
        "source_repository": NL2REPOBENCH_REPOSITORY_URL,
        "source_revision": source_revision,
    }
    (prepared_tasks_dir / CACHE_MARKER).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    store = NL2RepoBenchTaskStore(prepared_tasks_dir)
    output_path = Path(jsonl_path).expanduser().resolve()
    _write_jsonl(store, output_path)
    return prepared_tasks_dir, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--source-revision", type=str, default=NL2REPOBENCH_SOURCE_REVISION)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    tasks_dir, jsonl_path = prepare(
        source_dir=args.source_dir,
        tasks_dir=args.tasks_dir,
        jsonl_path=args.jsonl,
        source_revision=args.source_revision,
        allow_download=not args.no_download,
    )
    print(f"Prepared NL2RepoBench verifier assets in {tasks_dir}")
    print(f"Wrote NL2RepoBench benchmark data to {jsonl_path}")


if __name__ == "__main__":
    main()
