# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize BenchCAD using the pinned upstream downloaders and prompt builders."""

import argparse
import asyncio
import json
from pathlib import Path

from responses_api_agents.benchcad_agent.runtime import TASKS, UPSTREAM_REVISION, ensure_runtime, run_worker
from responses_api_agents.benchcad_agent.worker import DATASET_REVISION


ROOT = Path(__file__).resolve().parent


async def prepare_async(*, tasks: tuple[str, ...], limit: int | None, output: Path, upstream: Path) -> Path:
    """Prepare selected tasks, recording source revisions and selection in a manifest."""
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError(f"Choose distinct tasks from {TASKS}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    output, upstream = output.resolve(), upstream.resolve()
    python = await asyncio.to_thread(ensure_runtime, upstream)
    output.mkdir(parents=True, exist_ok=True)
    downloaded = set()
    counts = {}
    rows = []
    for task in tasks:
        source_task = "qa" if task in {"vision_qa", "code_qa"} else task
        source = output / "source" / source_task
        options = ("--limit", str(limit)) if limit is not None else ()
        if source_task not in downloaded:
            await run_worker(
                python, upstream, "download", "--task", source_task, "--output", str(source), *options, timeout=86400
            )
            downloaded.add(source_task)
        await run_worker(
            python,
            upstream,
            "export",
            "--task",
            task,
            "--source",
            str(source),
            "--output",
            str(output),
            *options,
            timeout=86400,
        )
        task_rows = (output / f"{task}.jsonl").read_text().splitlines()
        counts[task] = len(task_rows)
        if not task_rows:
            raise ValueError(f"Upstream preparation produced no {task} records")
        rows.extend(task_rows)
    manifest = {
        "upstream_revision": UPSTREAM_REVISION,
        "dataset_revision": DATASET_REVISION,
        "protocol": "benchcad-opencode",
        "limit_per_task": limit,
        "counts": counts,
        "license": "CC-BY-4.0",
        "attribution": "BenchCAD authors, https://github.com/BenchCAD/BenchCAD-main",
        "source": "https://huggingface.co/datasets/BenchCAD/BenchCAD",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    destination = output / "benchcad_benchmark.jsonl"
    destination.write_text("\n".join(rows) + "\n")
    return destination


def prepare() -> Path:
    """Entry point for `gym eval prepare`: all four tasks, full pinned dataset."""
    return asyncio.run(prepare_async(tasks=TASKS, limit=None, output=ROOT / "data", upstream=ROOT / ".cache/upstream"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--limit", type=int, help="Maximum parts per task for a smoke run")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--upstream", type=Path, default=ROOT / ".cache/upstream")
    args = parser.parse_args()
    print(
        asyncio.run(
            prepare_async(tasks=tuple(args.tasks), limit=args.limit, output=args.output, upstream=args.upstream)
        )
    )
