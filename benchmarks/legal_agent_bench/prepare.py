# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Legal Agent Bench's benchmark index from its shared task cache."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from resources_servers.legal_agent_bench.prepare import EXPECTED_TASK_COUNT, INDEX_FILENAME, prepare_assets


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "legal_agent_bench_benchmark.jsonl"
BENCHMARK_AGENT_NAME = "legal_agent_bench_benchmark_harbor_agent"


def _render_benchmark_index(source_index: Path) -> str:
    rendered_rows: list[str] = []

    with source_index.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid LAB task index JSON on line {line_number}: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(f"LAB task index line {line_number} must contain a JSON object")

            row["agent_ref"] = {
                "name": BENCHMARK_AGENT_NAME,
                "type": "responses_api_agents",
            }
            rendered_rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    if len(rendered_rows) != EXPECTED_TASK_COUNT:
        raise ValueError(f"Expected {EXPECTED_TASK_COUNT} LAB benchmark rows, found {len(rendered_rows)}")

    return "".join(rendered_rows)


def _atomic_write(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(dir=output_path.parent, prefix=f".{output_path.name}.")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare(*, force: bool = False) -> Path:
    """Prepare shared LAB assets and write the benchmark-specific task index."""
    prepared = prepare_assets("all", force=force)
    tasks_dir = prepared.get("tasks")
    if tasks_dir is None:
        raise RuntimeError("LAB asset preparation did not return a task cache")

    content = _render_benchmark_index(Path(tasks_dir) / INDEX_FILENAME)
    _atomic_write(OUTPUT_FPATH, content)
    print(f"Wrote {EXPECTED_TASK_COUNT} Legal Agent Bench tasks to {OUTPUT_FPATH}", flush=True)
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
