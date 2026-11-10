# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the task folders and source rows for the openthoughts_tblite benchmark.

The tasks are fetched at a pinned commit and copied to data/tasks/. Each task.toml gains the prebuilt image
of its task as [environment] docker_image; nothing else in the tasks is changed, so every task keeps its own
agent and verifier time limits.
"""

import fcntl
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory


BENCHMARK_DIR = Path(__file__).resolve().parent
GYM_ROOT = BENCHMARK_DIR.parent.parent
TASKS_DIR = BENCHMARK_DIR / "data" / "tasks"
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark.jsonl"

TASK_REPOSITORY = "https://github.com/open-thoughts/OpenThoughts-TBLite.git"
# Harbor registry openthoughts-tblite@2.0
TASK_REVISION = "f075e463472c7790b85793b392dff1fff20cc0e3"  # pragma: allowlist secret
# The tasks ship only Dockerfiles. These are copies of NousResearch's builds (nousresearch/tblite-<task>:latest)
# under fixed tags, since a `latest` tag can be repointed.
DOCKER_IMAGE = "docker.io/kbuchanan31150/tblite:{task}"
NUM_TASKS = 100


def _source() -> str:
    return f"{TASK_REPOSITORY}@{TASK_REVISION} image={DOCKER_IMAGE}"


def _with_docker_image(text: str, image: str) -> str:
    """task.toml text with [environment] docker_image set; fails on any other change."""
    config = tomllib.loads(text)
    if "docker_image" in config.get("environment", {}):
        raise ValueError(f"task.toml already names an image: {config['environment']['docker_image']}")
    line = f"docker_image = {json.dumps(image)}"
    header = re.search(r"^\[environment\][ \t]*(#.*)?$", text, flags=re.M)
    if header is None:
        updated = text.rstrip("\n") + f"\n\n[environment]\n{line}\n"
    else:
        updated = text[: header.end()] + "\n" + line + text[header.end() :]
    config.setdefault("environment", {})["docker_image"] = image
    if tomllib.loads(updated) != config:
        raise ValueError("task.toml changed beyond [environment] docker_image")
    return updated


def _fetch_tasks() -> None:
    """Copy the pinned task folders, with their images, to TASKS_DIR."""
    TASKS_DIR.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="tblite-", dir=TASKS_DIR.parent) as directory:
        checkout = Path(directory) / "checkout"
        staged = Path(directory) / "tasks"
        for args in (
            ["init", "--quiet", str(checkout)],
            ["-C", str(checkout), "fetch", "--quiet", "--depth", "1", TASK_REPOSITORY, TASK_REVISION],
            ["-C", str(checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        ):
            subprocess.run(["git", *args], check=True)
        task_dirs = sorted(path.parent for path in checkout.glob("*/task.toml"))
        if len(task_dirs) != NUM_TASKS:
            raise ValueError(f"Expected {NUM_TASKS} tasks at {TASK_REVISION}, found {len(task_dirs)}")
        for task_dir in task_dirs:
            shutil.copytree(task_dir, staged / task_dir.name)
            task_toml = staged / task_dir.name / "task.toml"
            image = DOCKER_IMAGE.format(task=task_dir.name)
            task_toml.write_text(_with_docker_image(task_toml.read_text(encoding="utf-8"), image), encoding="utf-8")
        (staged / ".source").write_text(_source() + "\n")
        # Publish only a complete copy; a failed fetch leaves the previous one in place.
        if TASKS_DIR.exists():
            shutil.rmtree(TASKS_DIR)
        os.replace(staged, TASKS_DIR)


def _tasks_are_current() -> bool:
    marker = TASKS_DIR / ".source"
    return marker.is_file() and marker.read_text().strip() == _source()


def is_prepared_data_current(jsonl_fpath: Path) -> bool:
    return Path(jsonl_fpath).is_file() and _tasks_are_current()


def prepare() -> Path:
    TASKS_DIR.parent.mkdir(parents=True, exist_ok=True)
    with (TASKS_DIR.parent / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not _tasks_are_current():
            _fetch_tasks()

        rows = []
        for task_dir in sorted(path.parent for path in TASKS_DIR.glob("*/task.toml")):
            task = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
            if not (task_dir / "tests" / "test.sh").is_file():
                raise ValueError(f"{task_dir}: no tests/test.sh")
            rows.append(
                {
                    "responses_create_params": {
                        "input": [
                            {"role": "user", "content": (task_dir / "instruction.md").read_text(encoding="utf-8")}
                        ]
                    },
                    "task_id": task_dir.name,
                    # The tasks have no [task] name; Harbor names each by its folder.
                    "task_name": task_dir.name,
                    "docker_image": task["environment"]["docker_image"],
                    "task_folder": str(task_dir.relative_to(GYM_ROOT)),
                }
            )
        if len(rows) != NUM_TASKS:
            raise ValueError(f"Expected {NUM_TASKS} tasks in {TASKS_DIR}, found {len(rows)}")

        temporary_path = OUTPUT_PATH.with_suffix(".jsonl.tmp")
        temporary_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        os.replace(temporary_path, OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(prepare())
