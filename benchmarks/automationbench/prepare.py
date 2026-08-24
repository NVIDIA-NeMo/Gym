# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export public or path-provided AutomationBench V1 tasks to Gym JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
REPO_ROOT = BENCHMARK_DIR.parents[1]
VERIFIERS_AGENT_DIR = REPO_ROOT / "responses_api_agents" / "verifiers_agent"
CREATE_DATASET = VERIFIERS_AGENT_DIR / "scripts" / "create_dataset.py"
TASKSET_ID = "automationbench_v1"
AGENT_NAME = "automationbench_benchmark_agent"
PUBLIC_OUTPUT = DATA_DIR / "automationbench_benchmark.jsonl"
V1_TASKS_OUTPUT = DATA_DIR / "v1_tasks.jsonl"
STAGED_TASKS_DIR = DATA_DIR / "tasks"


def _interpreter() -> str:
    component_python = VERIFIERS_AGENT_DIR / ".venv" / "bin" / "python"
    return str(component_python) if component_python.exists() else sys.executable


def _export(
    output: Path,
    *,
    taskset_config: dict,
    size: int = -1,
    agent_name: str = AGENT_NAME,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _interpreter(),
        str(CREATE_DATASET),
        "--taskset",
        TASKSET_ID,
        "--taskset-config",
        json.dumps(taskset_config, separators=(",", ":")),
        "--size",
        str(size),
        "--agent-name",
        agent_name,
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, cwd=VERIFIERS_AGENT_DIR)
    return output


def prepare(
    force: bool = False,
    domains: str = "all",
    num_examples: int = -1,
) -> Path:
    """Prepare the public six-domain benchmark for ``gym eval prepare``."""
    if PUBLIC_OUTPUT.exists() and not force:
        return PUBLIC_OUTPUT
    return _export(
        PUBLIC_OUTPUT,
        taskset_config={"domains": domains, "num_examples": num_examples},
    )


def stage_tasks(tasks_dir: Path, *, force: bool = False) -> Path:
    """Point the ignored stable runtime path at an external task directory."""
    source = tasks_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"AutomationBench tasks directory does not exist: {source}")
    if not any(source.glob("task_*.json")):
        raise ValueError(f"AutomationBench tasks directory has no task_*.json files: {source}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if STAGED_TASKS_DIR.is_symlink():
        if STAGED_TASKS_DIR.resolve() == source:
            return STAGED_TASKS_DIR
        if not force:
            raise FileExistsError(
                f"{STAGED_TASKS_DIR} points to {STAGED_TASKS_DIR.resolve()}; pass --force to replace it"
            )
        STAGED_TASKS_DIR.unlink()
    elif STAGED_TASKS_DIR.exists():
        if STAGED_TASKS_DIR.resolve() == source:
            return STAGED_TASKS_DIR
        raise FileExistsError(f"{STAGED_TASKS_DIR} already exists and is not a replaceable symlink")
    STAGED_TASKS_DIR.symlink_to(source, target_is_directory=True)
    return STAGED_TASKS_DIR


def prepare_tasks(
    tasks_dir: Path,
    *,
    output: Path = V1_TASKS_OUTPUT,
    size: int = -1,
    force: bool = False,
) -> Path:
    staged = stage_tasks(tasks_dir, force=force)
    if output.exists() and not force:
        return output
    return _export(
        output,
        taskset_config={"tasks_dir": str(staged)},
        size=size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, help="Directory containing V1 task_*.json files")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--domains", default="all")
    parser.add_argument("--size", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.tasks_dir:
        output = args.output or V1_TASKS_OUTPUT
        prepared = prepare_tasks(args.tasks_dir, output=output, size=args.size, force=args.force)
    else:
        if args.output and args.output != PUBLIC_OUTPUT:
            raise ValueError(f"public benchmark output is fixed at {PUBLIC_OUTPUT}")
        prepared = prepare(force=args.force, domains=args.domains, num_examples=args.size)
    print(prepared)


if __name__ == "__main__":
    main()
