# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare rows for the terminal_bench_4 benchmark from a LOCAL Terminal-Bench 4.0 task tree.

Unlike terminal_bench_2_1/prepare.py this never clones anything: the caller points
``TB4_TASKS_DIR`` (or ``tasks_dir``) at a directory of task folders (each with ``task.toml``,
``instruction.md``, ``tests/``), for example a vetted whitelist. Images are the prebuilt ones the
TB4 release pushes to Docker Hub, ``<image_repository>:<task>-environment-<release_tag>`` and
``<task>-verifier-<release_tag>``; an optional inventory JSON (``{"tasks": {<task>: {"environment_v4":
{"digest": ...}, "verifier_v4": {"digest": ...}}}}``) adds digests for provenance.

Multi-service compose tasks are skipped unless ``include_compose`` is set, because the
terminal_bench_4 server refuses them (a single sandbox cannot host sidecars). GPU tasks are skipped
unless ``include_gpu`` is set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark.jsonl"
DEFAULT_IMAGE_REPOSITORY = "harborframework/terminal-bench"
DEFAULT_RELEASE_TAG = "v4.0.0"
DEFAULT_INSTRUCTION_FILENAME = "instruction.md"


def _repo_root() -> Path:
    return BENCHMARK_DIR.parent.parent


def image_ref(
    task: str, role: str, *, image_repository: str = DEFAULT_IMAGE_REPOSITORY, release_tag: str = DEFAULT_RELEASE_TAG
) -> str:
    """TB4's release tag scheme (``scripts/release-build/release_prebuilt.py``: ``<task>-<role>-<tag>``)."""
    return f"{image_repository}:{task}-{role}-{release_tag}"


def build_row(
    task_dir: Path,
    *,
    image_repository: str = DEFAULT_IMAGE_REPOSITORY,
    release_tag: str = DEFAULT_RELEASE_TAG,
    inventory: Optional[dict] = None,
) -> dict:
    sys.path.insert(0, str(_repo_root()))
    from resources_servers.terminal_bench_4.task_manifest import load_task  # noqa: PLC0415

    task = load_task(task_dir)
    short = task_dir.name
    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": (task_dir / DEFAULT_INSTRUCTION_FILENAME).read_text()}]
        },
        "task_name": task.task_name,
        "docker_image": image_ref(short, "environment", image_repository=image_repository, release_tag=release_tag),
        "verifier_docker_image": image_ref(
            short, "verifier", image_repository=image_repository, release_tag=release_tag
        ),
        "task_folder": str(task_dir.resolve()),
    }
    entry = ((inventory or {}).get("tasks") or {}).get(short) or {}
    for key, field in (("environment_v4", "docker_image_digest"), ("verifier_v4", "verifier_docker_image_digest")):
        digest = (entry.get(key) or {}).get("digest")
        if digest:
            row[field] = digest
    return row


def iter_task_dirs(tasks_dir: Path, task_names: Optional[Iterable[str]] = None) -> list[Path]:
    wanted = set(task_names) if task_names else None
    dirs = []
    for child in sorted(tasks_dir.iterdir()):
        if not child.is_dir() or not (child / "task.toml").is_file():
            continue
        if wanted is not None and child.name not in wanted:
            continue
        dirs.append(child)
    if wanted is not None:
        missing = wanted - {d.name for d in dirs}
        if missing:
            raise FileNotFoundError(f"Requested tasks not found under {tasks_dir}: {sorted(missing)}")
    return dirs


def prepare(
    tasks_dir: Optional[str] = None,
    image_repository: str = DEFAULT_IMAGE_REPOSITORY,
    release_tag: str = DEFAULT_RELEASE_TAG,
    inventory_json: Optional[str] = None,
    task_names: Optional[Iterable[str]] = None,
    include_compose: bool = False,
    include_gpu: bool = False,
    output_path: Optional[str] = None,
) -> Path:
    sys.path.insert(0, str(_repo_root()))
    from resources_servers.terminal_bench_4.task_manifest import load_task  # noqa: PLC0415

    tasks_root = Path(tasks_dir or os.environ.get("TB4_TASKS_DIR", "")).expanduser()
    if not str(tasks_root) or not tasks_root.is_dir():
        raise FileNotFoundError(
            "Point TB4_TASKS_DIR (or prepare(tasks_dir=...)) at a directory of Terminal-Bench 4.0 task folders; "
            f"got {tasks_root!s}"
        )
    inventory = json.loads(Path(inventory_json).read_text()) if inventory_json else None
    output = Path(output_path) if output_path else OUTPUT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)

    skipped: dict[str, str] = {}
    num_samples = 0
    with output.open("w") as handle:
        for task_dir in iter_task_dirs(tasks_root, task_names):
            task = load_task(task_dir)
            if task.is_compose and not include_compose:
                skipped[task_dir.name] = f"compose task with services {list(task.compose_services)}"
                continue
            if task.requires_gpu and not include_gpu:
                skipped[task_dir.name] = "requires a GPU"
                continue
            handle.write(
                json.dumps(
                    build_row(
                        task_dir, image_repository=image_repository, release_tag=release_tag, inventory=inventory
                    )
                )
                + "\n"
            )
            num_samples += 1

    print(f"Wrote {num_samples} rows to {output}; skipped {len(skipped)}: {json.dumps(skipped, indent=1)}")
    if num_samples == 0:
        raise ValueError(f"No runnable tasks under {tasks_root}")
    return output


if __name__ == "__main__":
    prepare()
