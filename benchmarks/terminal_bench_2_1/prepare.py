# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare source rows for the terminal_bench_2_1 benchmark."""

import argparse
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


def _git(repository: Path, *arguments: str) -> str:
    return run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout.strip()


def prepare(*, task_revision: str | None = None, task_repository: str = TASK_REPOSITORY) -> Path:
    """Prepare task rows, optionally checking out a full task commit before reading them.

    Without a revision, preserve the existing checkout or clone the repository's default branch.
    Pinned preparation refuses local changes, including ignored files that could alter verifier assets.
    """
    if task_revision is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", task_revision):
            raise ValueError("task_revision must be a full 40-character commit SHA")
        task_revision = task_revision.lower()

    repo_path = BENCHMARK_DIR / "terminal-bench-2-1"
    if not repo_path.exists():
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--", task_repository, str(repo_path)], check=True)
    if task_revision is not None:
        # The stored URL, not `remote get-url`: that one applies url.*.insteadOf rewrites (a mirror, a token).
        if _git(repo_path, "config", "--get", "remote.origin.url") != task_repository:
            raise ValueError(
                f"Task checkout {repo_path} has a different origin; move it aside or use its repository URL"
            )
        # The verifier uploads every non-hidden file under a task's tests directory, so a change anywhere, an untracked
        # file, or an ignored file under tasks/*/tests/ counts. Ignored files elsewhere (bytecode caches) do not.
        if _git(repo_path, "status", "--porcelain", "--untracked-files=all") or _git(
            repo_path, "status", "--porcelain", "--ignored", "--untracked-files=all", "--", ":(glob)tasks/*/tests/**"
        ):
            raise ValueError(f"Task checkout {repo_path} has local changes; move them aside before pinning a revision")
        available = run(["git", "-C", str(repo_path), "cat-file", "-e", task_revision], capture_output=True)
        if available.returncode:
            _git(repo_path, "fetch", "origin", task_revision)
        if _git(repo_path, "cat-file", "-t", task_revision) != "commit":
            raise ValueError("task_revision must identify a commit, not another Git object")
        _git(repo_path, "checkout", "--detach", task_revision)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-revision", help="Full task commit SHA; defaults to the existing checkout or repository HEAD"
    )
    parser.add_argument("--task-repository", default=TASK_REPOSITORY, help="Task repository URL to clone")
    args = parser.parse_args()
    prepare(task_revision=args.task_revision, task_repository=args.task_repository)
