# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Legal Agent Bench's benchmark index from its shared task cache."""

from __future__ import annotations

import json
import os
from pathlib import Path

from nemo_gym.config import get_global_config_dict
from resources_servers.legal_agent_bench.environment_prepare import prepare_environment
from resources_servers.legal_agent_bench.prepare import (
    EXPECTED_TASK_COUNT,
    INDEX_FILENAME,
    prepare_assets,
    read_task_selection,
)


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "legal_agent_bench_benchmark.jsonl"


def prepare(
    *, force: bool = False, input: str | Path | None = None, output: str | Path | None = None, image: str | None = None
) -> Path:
    """Prepare all LAB tasks, or select them with +prepare_script_args.input=/path/to/tasks.jsonl."""
    options = {}
    for key, variable in (
        ("tasks_dir", "LEGAL_AGENT_BENCH_TASK_CACHE_DIR"),
        ("skills_dir", "LEGAL_AGENT_BENCH_SKILLS_DIR"),
    ):
        if value := os.environ.get(variable):
            options[key] = value
    if input is not None:
        task_ids = read_task_selection(input)
        options["task_ids"] = task_ids
    prepared = prepare_assets("all", force=force, **options)
    tasks_dir = prepared.get("tasks")
    if tasks_dir is None:
        raise RuntimeError("LAB asset preparation did not return a task cache")

    source_index = Path(input).expanduser() if input is not None else Path(tasks_dir) / INDEX_FILENAME
    rows = []
    for number, line in enumerate(source_index.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected an object")
        except ValueError as error:
            raise ValueError(f"Invalid LAB task index JSON on line {number}: {error}") from error
        rows.append(row)
    if input is None and len(rows) != EXPECTED_TASK_COUNT:
        raise ValueError(f"Expected {EXPECTED_TASK_COUNT} LAB benchmark rows, found {len(rows)}")
    lab = get_global_config_dict().get("lab", {})
    return prepare_environment(
        tasks_dir=Path(tasks_dir),
        skills_dir=Path(prepared["skills"]),
        rows=rows,
        output=Path(output) if output is not None else OUTPUT_FPATH,
        image=image or os.environ.get("LEGAL_AGENT_BENCH_IMAGE"),
        agent_timeout_seconds=float(lab.get("agent_timeout_seconds", 10800)),
        verifier_timeout_seconds=float(lab.get("verifier_timeout_seconds", 3600)),
    )


if __name__ == "__main__":
    prepare()
