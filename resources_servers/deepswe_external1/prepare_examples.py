# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate five self-contained public DeepSWE example rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from resources_servers.deepswe.prepare import DEEPSWE_REPOSITORY, ensure_source
from resources_servers.deepswe.task_schema import resolve_effective_verifier_env_config
from resources_servers.deepswe.task_store import DEEPSWE_SOURCE_REVISION, DeepSWETaskStore
from resources_servers.deepswe_external1.task_data import PhaseLimits, TaskData, TaskFiles


PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLE_TASK_IDS = (
    "abs-module-cache-flags",
    "abs-stepped-slices",
    "actionlint-action-pinning-lint",
    "adaptix-name-mapping-aliases",
    "aiomonitor-task-snapshots-diff",
)
FILE_PATHS = {
    "test_script": "tests/test.sh",
    "test_patch": "tests/test.patch",
    "grader": "tests/grader.py",
    "grader_config": "tests/config.json",
    "solve_script": "solution/solve.sh",
    "solution_patch": "solution/solution.patch",
}


def read_task_files(source_dir: Path) -> TaskFiles:
    """Read original UTF-8 bytes without normalizing line endings."""
    return TaskFiles(**{field: (source_dir / path).read_bytes().decode("utf-8") for field, path in FILE_PATHS.items()})


def task_row(data: TaskData, instruction: str) -> dict[str, object]:
    """Keep held-out file contents out of the model's input messages."""
    return data.model_dump(mode="json", exclude_none=True) | {
        "responses_create_params": {"input": [{"role": "user", "content": instruction}]},
    }


def prepare_examples(*, source_dir: Path, output_path: Path, allow_download: bool = True) -> list[dict[str, object]]:
    """Use the public benchmark's pinned source and original versioned images."""
    source = ensure_source(source_dir, allow_download=allow_download)
    original = DeepSWETaskStore(source / "tasks")
    rows = []
    for task_id in EXAMPLE_TASK_IDS:
        task = original.get(task_id)
        agent = task.config.environment
        verifier = resolve_effective_verifier_env_config(task.config, None)
        if verifier is None:
            raise ValueError("Public example requires a separate verifier")
        data = TaskData(
            task_id=task_id,
            image=agent.docker_image,
            verifier_image=agent.docker_image,
            base_commit=task.config.metadata["base_commit_hash"],
            agent=PhaseLimits(
                cpus=agent.cpus,
                memory_mb=agent.memory_mb,
                storage_mb=agent.storage_mb,
                timeout_sec=task.config.agent.timeout_sec,
                env=agent.env,
            ),
            verifier=PhaseLimits(
                cpus=verifier.cpus,
                memory_mb=verifier.memory_mb,
                storage_mb=verifier.storage_mb,
                timeout_sec=task.config.verifier.timeout_sec,
                env=verifier.env | task.config.verifier.env,
            ),
            files=read_task_files(task.task_dir),
            public_source={
                "repository": DEEPSWE_REPOSITORY,
                "revision": DEEPSWE_SOURCE_REVISION,
                "task_path": f"tasks/{task_id}",
                "license": "Apache-2.0",
                "upstream_project": task.config.metadata["repository_url"],
            },
        )
        rows.append(task_row(data, (task.task_dir / "instruction.md").read_bytes().decode("utf-8")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PACKAGE_DIR / "data/cache/source")
    parser.add_argument("--output", type=Path, default=PACKAGE_DIR / "data/example.jsonl")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    rows = prepare_examples(
        source_dir=args.source_dir,
        output_path=args.output,
        allow_download=not args.no_download,
    )
    print(f"Prepared {len(rows)} public examples; this does not execute or validate their solutions.")


if __name__ == "__main__":
    main()
