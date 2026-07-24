#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the separately distributed Prime browser tasks."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


ENV_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ENV_ROOT / "data" / "tasks"
MANIFEST = ENV_ROOT / "data" / "validation.jsonl"
DATASET_NAME = "browser_all_opensandbox"
EXPECTED_TASKS = 30
SETUP_SCRIPT = ENV_ROOT / "sandbox_setup" / "setup_opensandbox.sh"
BROWSER_OPEN = ENV_ROOT / "browser_tools" / "browser_open"


def task_names(tasks_dir: Path, expected_tasks: int) -> list[str]:
    names = sorted(path.name for path in tasks_dir.iterdir() if path.is_dir() and (path / "task.toml").is_file())
    if len(names) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} tasks in {tasks_dir}, found {len(names)}")
    return names


def write_manifest(path: Path, names: list[str]) -> None:
    rows = (
        {
            "instance_id": f"{DATASET_NAME}::{name}",
            "responses_create_params": {
                "input": [],
                "temperature": 0.6,
                "top_p": 0.95,
                "max_output_tokens": 32768,
            },
            "agent_ref": {
                "type": "responses_api_agents",
                "name": "harbor_agent",
            },
        }
        for name in names
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def set_toml_string(path: Path, table: str, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = f"[{table}]"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise ValueError(f"{path} has no {header} table") from exc

    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("[")),
        len(lines),
    )
    assignment = f'{key} = "{value}"'
    for index in range(start + 1, end):
        if re.match(rf"^\s*{re.escape(key)}\s*=", lines[index]):
            lines[index] = assignment
            break
    else:
        lines.insert(start + 1, assignment)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(
    source_dir: Path,
    output_dir: Path,
    manifest_path: Path = MANIFEST,
    *,
    overwrite: bool = False,
    expected_tasks: int = EXPECTED_TASKS,
) -> int:
    tasks_dir = source_dir / "tasks"
    start_sims = source_dir / "scripts" / "start_local_sims.py"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"task directory not found: {tasks_dir}")
    if not start_sims.is_file():
        raise FileNotFoundError(f"browser launcher not found: {start_sims}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    names = task_names(tasks_dir, expected_tasks)

    for name in names:
        target = output_dir / name
        shutil.copytree(
            tasks_dir / name,
            target,
            ignore=shutil.ignore_patterns("._*"),
            symlinks=False,
        )
        set_toml_string(
            target / "task.toml",
            "environment",
            "docker_image",
            "python:3.12-slim-bookworm",
        )
        environment = target / "environment"
        shutil.copy2(SETUP_SCRIPT, environment / "setup_opensandbox.sh")
        shutil.copy2(BROWSER_OPEN, environment / "browser_open")
        shutil.copy2(start_sims, environment / "start_local_sims.py")

    write_manifest(manifest_path, names)
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Extracted browser bundle containing tasks/ and scripts/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Prepared task directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=MANIFEST,
        help=f"Generated Gym input JSONL. Default: {MANIFEST}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    args = parser.parse_args()
    count = prepare(
        args.source_dir.resolve(),
        args.output_dir.resolve(),
        args.manifest_path.resolve(),
        overwrite=args.overwrite,
    )
    print(f"Prepared {count} browser tasks in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
