# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned, ordered Terminal-Bench inputs used by the Inkling recipe."""

import fcntl
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from benchmarks.terminal_bench_2_1.prepare_terminal_guidance import TERMINAL_INTERACTION_GUIDANCE


BENCHMARK_DIR = Path(__file__).resolve().parent
GYM_ROOT = BENCHMARK_DIR.parent.parent
REFERENCE_PATH = BENCHMARK_DIR / "inkling_small_reference.json"
SOURCE_PATH = BENCHMARK_DIR / "data" / "inkling-small-tasks"
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark_inkling_small.jsonl"


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.strip()


def _ensure_source(*, url: str, revision: str) -> None:
    if not SOURCE_PATH.exists():
        # Publish only a completed checkout; a failed clone cannot poison later preparation.
        with TemporaryDirectory(prefix="inkling-clone-", dir=SOURCE_PATH.parent) as directory:
            checkout = Path(directory) / "tasks"
            subprocess.run(["git", "clone", "--no-checkout", url, str(checkout)], check=True)
            _git(checkout, "checkout", "--detach", revision)
            os.replace(checkout, SOURCE_PATH)
    actual = _git(SOURCE_PATH, "rev-parse", "HEAD")
    if actual != revision:
        raise ValueError(
            f"Expected task revision {revision}, found {actual} in {SOURCE_PATH}. Move it aside and retry."
        )
    # The verifier uploads files from this checkout, including untracked/ignored files.
    if _git(SOURCE_PATH, "status", "--porcelain", "--untracked-files=all", "--ignored"):
        raise ValueError(f"Task checkout {SOURCE_PATH} is modified. Move it aside and retry; it will not be reset.")


def prepare() -> Path:
    """Validate the pinned sources and atomically regenerate all 89 guided task rows."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_PATH.parent / ".inkling-small-prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        _ensure_source(url=reference["task_repository"], revision=reference["task_revision"])
        order = reference["task_order"]
        actual_tasks = {path.name for path in (SOURCE_PATH / "tasks").iterdir() if path.is_dir()}
        if len(order) != 89 or len(set(order)) != 89 or actual_tasks != set(order):
            raise ValueError("Task list does not match the reference's 89 unique tasks.")

        lines = []
        digest = hashlib.sha256()
        for name in order:
            task_dir = SOURCE_PATH / "tasks" / name
            task = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
            row = {
                "responses_create_params": {
                    "input": [
                        {"role": "user", "content": (task_dir / "instruction.md").read_text(encoding="utf-8")},
                        {"role": "user", "content": TERMINAL_INTERACTION_GUIDANCE},
                    ]
                },
                "task_name": task["task"]["name"],
                "docker_image": task["environment"]["docker_image"],
                "task_folder": name,
            }
            # Normalize only the checkout location, retaining the task identity and order.
            digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
            row["task_folder"] = str(task_dir.relative_to(GYM_ROOT))
            lines.append(json.dumps(row) + "\n")
        if digest.hexdigest() != reference["normalized_rows_sha256"]:
            raise ValueError("Task instructions, guidance, image tags, or order differ from the reference inputs.")

        temporary_path = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=OUTPUT_PATH.parent, delete=False) as output:
                temporary_path = Path(output.name)
                output.writelines(lines)
            os.replace(temporary_path, OUTPUT_PATH)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(prepare())
