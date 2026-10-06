# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare all 89 pinned Terminal-Bench 2.1 tasks for native NOOA evaluation."""

import json
import math
import subprocess
import tomllib
from pathlib import Path
from typing import TypedDict

from resources_servers.terminal_bench_2_1.task_metadata import read_image_startup


BENCHMARK_DIR = Path(__file__).resolve().parent
TASK_REPOSITORY = "https://github.com/harbor-framework/terminal-bench-2-1"
TASK_REVISION = "7131e4375048a0e408a8fb404b5f499d726b695b"
TASK_COUNT = 89


class NativeTaskRow(TypedDict):
    """Native task identity and the single-agent input prepared from task files."""

    task_id: dict[str, str]
    task_input: dict[str, object]


def task_row(task_dir: Path, *, deployed_task_dir: Path | None = None) -> NativeTaskRow:
    """Keep the canonical instruction, image and separate agent/verifier budgets."""
    with (task_dir / "task.toml").open("rb") as reader:
        task = tomllib.load(reader)
    name = task["task"]["name"]
    instruction = (task_dir / "instruction.md").read_text()
    image = task["environment"]["docker_image"]
    if not all(isinstance(value, str) and value.strip() for value in (name, instruction, image)):
        raise ValueError(f"Missing task identity, instruction or image: {task_dir}")
    if not (task_dir / "tests/test.sh").is_file():
        raise ValueError(f"Missing task verifier: {task_dir}")
    budgets = {}
    for section in ("agent", "verifier"):
        value = task[section]["timeout_sec"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {section} timeout: {task_dir}")
        budgets[f"{section}_timeout_seconds"] = float(value)
    # Relative paths remain portable when the prepared task checkout ships with Gym.
    try:
        task_folder = str(task_dir.relative_to(BENCHMARK_DIR.parents[1]))
    except ValueError:
        task_folder = str(task_dir.resolve())
    if deployed_task_dir is not None:
        if not deployed_task_dir.is_absolute():
            raise ValueError("Deployed task directory must be absolute")
        task_folder = str(deployed_task_dir)
    startup = read_image_startup(task_dir)
    return {
        "task_id": {"taskset": "terminal-bench-2.1-nooa", "task_id": name},
        "task_input": {
            "responses_create_params": {
                "input": [{"role": "user", "content": instruction}],
                "max_output_tokens": 32768,
                "temperature": 1.0,
                "top_p": 1.0,
            },
            "agent_timeout_seconds": budgets.pop("agent_timeout_seconds"),
            "task_data": {
                "task_name": name,
                "docker_image": image,
                "task_folder": task_folder,
                "task_revision": TASK_REVISION,
                **({"image_startup": startup.model_dump()} if startup is not None else {}),
                **budgets,
            },
        },
    }


def prepare_native(
    *,
    repository_path: Path | None = None,
    output: Path | None = None,
    deployed_repository_path: Path | None = None,
) -> Path:
    """Validate pinned tasks and optionally map their relative paths to a deployed checkout.

    Task identities are namespaced labels, not repository-relative filesystem paths.
    The deployment root must contain the same pinned task checkout.
    """
    if deployed_repository_path is not None and not deployed_repository_path.is_absolute():
        raise ValueError("Deployed repository path must be absolute")
    repository_path = repository_path or BENCHMARK_DIR / "terminal-bench-2-1"
    output = output or BENCHMARK_DIR / "data/nooa.jsonl"
    if not repository_path.exists():
        subprocess.run(["git", "clone", "--no-checkout", TASK_REPOSITORY, str(repository_path)], check=True)
        subprocess.run(["git", "-C", str(repository_path), "checkout", "--detach", TASK_REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(repository_path), "rev-parse", "HEAD"], text=True).strip()
    if revision != TASK_REVISION:
        raise ValueError(
            f"Expected Terminal-Bench task revision {TASK_REVISION}, found {revision}; use a separate checkout"
        )
    changes = subprocess.check_output(
        ["git", "-C", str(repository_path), "status", "--porcelain", "--untracked-files=all", "--", "tasks"],
        text=True,
    )
    if changes:
        raise ValueError("Terminal-Bench task checkout contains modified or untracked task files")
    rows = [
        task_row(
            path.parent,
            deployed_task_dir=(
                deployed_repository_path / path.parent.relative_to(repository_path)
                if deployed_repository_path is not None
                else None
            ),
        )
        for path in sorted((repository_path / "tasks").glob("*/task.toml"))
    ]
    identities = {row["task_id"]["task_id"] for row in rows}
    if len(rows) != TASK_COUNT or len(identities) != TASK_COUNT:
        raise ValueError(
            f"Expected {TASK_COUNT} unique Terminal-Bench tasks, found {len(rows)} rows/{len(identities)} IDs"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return output


if __name__ == "__main__":
    prepare_native()
