# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_OUTPUT = DATA_DIR / "terminalbench.jsonl"
DEFAULT_TASKS_CACHE = Path.home() / ".cache" / "harbor" / "tasks"
DEFAULT_DATASET = "terminal-bench@2.0"


def _megabytes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().upper().removesuffix("B")
    factors = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    if text and text[-1] in factors:
        return int(float(text[:-1]) * factors[text[-1]])
    return int(float(text))


def _dataset_directory(dataset: str) -> str:
    return dataset.split("@", 1)[0].rstrip("/").split("/")[-1]


def _task_directories(cache: Path, dataset: str) -> list[Path]:
    root = cache / _dataset_directory(dataset)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if (path / "task.toml").is_file())


def _workdir(task_dir: Path) -> str | None:
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return None
    result = None
    for line in dockerfile.read_text().splitlines():
        fields = line.strip().split(None, 1)
        if fields and fields[0].upper() == "WORKDIR" and len(fields) == 2:
            result = fields[1]
    return result


def _task_row(task_dir: Path) -> dict[str, Any]:
    config = tomllib.loads((task_dir / "task.toml").read_text())
    environment = config.get("environment") or {}
    agent = config.get("agent") or {}
    verifier = config.get("verifier") or {}
    instruction = task_dir / "instruction.md"
    task_name = task_dir.name
    metadata = {
        "instance_id": f"terminalbench::{task_name}",
        "task_name": task_name,
        "docker_image": environment.get("docker_image", "ubuntu:22.04"),
        "task_dir": str(task_dir.resolve()),
        "agent_timeout_sec": str(agent["timeout_sec"]) if agent.get("timeout_sec") is not None else None,
        "verifier_timeout_sec": str(verifier["timeout_sec"]) if verifier.get("timeout_sec") is not None else None,
        "workdir": _workdir(task_dir),
        "cpus": str(environment["cpus"]) if environment.get("cpus") is not None else None,
        "memory_mb": str(
            environment.get("memory_mb")
            if environment.get("memory_mb") is not None
            else _megabytes(environment.get("memory"))
        )
        if environment.get("memory_mb") is not None or environment.get("memory") is not None
        else None,
        "storage_mb": str(
            environment.get("storage_mb")
            if environment.get("storage_mb") is not None
            else _megabytes(environment.get("storage"))
        )
        if environment.get("storage_mb") is not None or environment.get("storage") is not None
        else None,
        "gpus": str(environment.get("gpus", 0)),
    }
    return {
        "task_name": task_name,
        "responses_create_params": {
            "input": [{"role": "user", "content": instruction.read_text() if instruction.is_file() else ""}],
            "metadata": metadata,
        },
    }


def _download(cache: Path, dataset: str) -> None:
    root = cache / _dataset_directory(dataset)
    if root.is_dir() and any(root.iterdir()):
        return
    if shutil.which("harbor") is None:
        raise RuntimeError("harbor is required to download TerminalBench tasks")
    subprocess.run(
        ["harbor", "datasets", "download", dataset, "--output-dir", str(cache)],
        check=True,
    )


def prepare(
    *,
    output: Path = DEFAULT_OUTPUT,
    tasks_cache: Path = DEFAULT_TASKS_CACHE,
    dataset: str = DEFAULT_DATASET,
    task_names: Iterable[str] | None = None,
    limit: int | None = None,
) -> Path:
    _download(tasks_cache, dataset)
    task_dirs = _task_directories(tasks_cache, dataset)
    if task_names:
        requested = list(dict.fromkeys(task_names))
        by_name = {path.name: path for path in task_dirs}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(f"unknown TerminalBench task names: {', '.join(missing)}")
        task_dirs = [by_name[name] for name in requested]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        task_dirs = task_dirs[:limit]
    if not task_dirs:
        raise RuntimeError(f"no TerminalBench tasks found under {tasks_cache}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(_task_row(path)) + "\n" for path in task_dirs))
    print(f"Wrote {len(task_dirs)} TerminalBench tasks to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks-cache", type=Path, default=DEFAULT_TASKS_CACHE)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--task", dest="task_names", action="append")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    prepare(
        output=args.output,
        tasks_cache=args.tasks_cache,
        dataset=args.dataset,
        task_names=args.task_names,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
