# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare source rows for the terminal_bench_2_1 benchmark."""

import json
import re
import tomllib
from glob import glob
from pathlib import Path
from subprocess import run


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark.jsonl"
NUM_TASKS = 89
TASK_REPOSITORY = "https://github.com/harbor-framework/terminal-bench-2-1"
_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def _git(repo_path: Path, *args: str) -> str:
    return run(["git", "-C", str(repo_path), *args], check=True, capture_output=True, text=True).stdout.strip()


def _check_out(repo_path: Path, task_revision: str) -> None:
    """Leave the task repository checked out at exactly ``task_revision``."""
    if _git(repo_path, "status", "--porcelain"):
        # The verifier uploads files from this checkout, so it is never reset under a run.
        raise ValueError(f"Task checkout {repo_path} is modified. Move it aside and retry; it will not be reset.")
    if _git(repo_path, "rev-parse", "HEAD") == task_revision:
        return
    if run(["git", "-C", str(repo_path), "cat-file", "-e", f"{task_revision}^{{commit}}"]).returncode != 0:
        _git(repo_path, "fetch", "--quiet", "origin", task_revision)
    _git(repo_path, "checkout", "--quiet", "--detach", task_revision)


def prepare(*, task_revision: str | None = None) -> Path:
    """Write the benchmark rows from the task repository.

    ``task_revision`` is the full commit of the task repository to check out. Without it the checkout is whatever
    ``git clone`` or an existing ``terminal-bench-2-1`` directory provides, so two runs can read different tasks.
    """
    if task_revision is not None and not _FULL_COMMIT.fullmatch(task_revision):
        raise ValueError(f"task_revision must be a full 40-character commit hash, got {task_revision!r}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    repo_path = BENCHMARK_DIR / "terminal-bench-2-1"
    if not repo_path.exists():
        run(["git", "clone", TASK_REPOSITORY], cwd=str(BENCHMARK_DIR), check=True)
    if task_revision is not None:
        _check_out(repo_path, task_revision)

    num_samples = 0
    task_ids = set()
    f_out = open(OUTPUT_PATH, "w")
    for task_dir in sorted(glob(f"{repo_path}/tasks/*")):
        task_dir = repo_path / "tasks" / task_dir
        if not task_dir.is_dir():
            continue

        with open(task_dir / "task.toml", "rb") as file:
            task_toml = tomllib.load(file)

        sample = {
            "responses_create_params": {
                "input": [{"role": "user", "content": (task_dir / "instruction.md").read_text()}]
            },
            "task_id": task_dir.name,
            "task_name": task_toml["task"]["name"],
            "docker_image": task_toml["environment"]["docker_image"],
            "task_folder": str(task_dir.relative_to(BENCHMARK_DIR.parent.parent)),
        }

        f_out.write(json.dumps(sample) + "\n")
        num_samples += 1
        task_ids.add(sample["task_id"])
    f_out.close()

    assert num_samples == NUM_TASKS, num_samples
    assert len(task_ids) == NUM_TASKS, f"task ids are not unique: {len(task_ids)} unique of {num_samples}"

    return OUTPUT_PATH


if __name__ == "__main__":
    prepare()
