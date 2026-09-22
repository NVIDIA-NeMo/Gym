# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare five public DeepSWE examples and their gitignored verifier packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from resources_servers.deepswe.prepare import DEEPSWE_REPOSITORY, ensure_source
from resources_servers.deepswe.task_schema import resolve_effective_verifier_env_config
from resources_servers.deepswe.task_store import DEEPSWE_SOURCE_REVISION, DeepSWETaskStore
from resources_servers.deepswe_external1.task_store import (
    ASSET_PATHS,
    AssetDigest,
    PhaseLimits,
    PreparedTask,
    TaskDefinition,
)


PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLE_TASK_IDS = (
    "abs-module-cache-flags",
    "abs-stepped-slices",
    "actionlint-action-pinning-lint",
    "adaptix-name-mapping-aliases",
    "aiomonitor-task-snapshots-diff",
)


def describe_assets(source_dir: Path) -> dict[str, AssetDigest]:
    """Hash only the seven declared control-plane assets, never repository trees."""
    if source_dir.is_symlink():
        raise ValueError("Task source directory must not be a symlink")
    source_dir = source_dir.resolve(strict=True)
    result = {}
    for name in ASSET_PATHS:
        path = source_dir / name
        # Check only within the resolved source root, avoiding redundant remote-filesystem
        # stats for every ancestor on every asset in a large corpus.
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise ValueError(f"Task asset must be a regular, non-symlink file: {name}")
        result[name] = AssetDigest(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mode=path.stat().st_mode & 0o777
        )
    return result


def materialize_task(source_dir: Path, tasks_dir: Path, definition: TaskDefinition) -> PreparedTask:
    """Create a checksummed package, refusing to replace a different existing one."""
    tasks_dir.mkdir(parents=True, exist_ok=True)
    destination = tasks_dir / definition.task_id
    if destination.exists():
        prepared = PreparedTask(destination)
        if prepared.definition != definition:
            raise ValueError(f"Refusing to overwrite different prepared task: {definition.task_id}")
        return prepared
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=tasks_dir.parent) as temporary:
        staging = Path(temporary) / definition.task_id
        for name in ASSET_PATHS:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir / name, target)
        (staging / "task.json").write_text(definition.model_dump_json(indent=2) + "\n", encoding="utf-8")
        PreparedTask(staging)
        staging.rename(destination)
    return PreparedTask(destination)


def task_row(task: PreparedTask) -> dict[str, object]:
    """Build a row without exposing solution or verifier contents to the agent."""
    return {
        "task_id": task.definition.task_id,
        "image": task.definition.image,
        "task_fingerprint": task.definition.fingerprint(),
        "responses_create_params": {"input": [{"role": "user", "content": task.instruction}]},
        "verifier_metadata": {"task_id": task.definition.task_id},
    }


def prepare_examples(
    *, source_dir: Path, tasks_dir: Path, output_path: Path, allow_download: bool = True
) -> list[dict[str, object]]:
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
        definition = TaskDefinition(
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
            assets=describe_assets(task.task_dir),
        )
        prepared = materialize_task(task.task_dir, tasks_dir, definition)
        row = task_row(prepared)
        row["public_source"] = {
            "repository": DEEPSWE_REPOSITORY,
            "revision": DEEPSWE_SOURCE_REVISION,
            "task_path": f"tasks/{task_id}",
            "license": "Apache-2.0",
            "upstream_project": task.config.metadata["repository_url"],
        }
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output_path.parent / "example_metrics.json").write_text(
        json.dumps({"Number of examples": len(rows)}, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PACKAGE_DIR / "data/cache/source")
    parser.add_argument("--tasks-dir", type=Path, default=PACKAGE_DIR / "data/cache/tasks")
    parser.add_argument("--output", type=Path, default=PACKAGE_DIR / "data/example.jsonl")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    rows = prepare_examples(
        source_dir=args.source_dir,
        tasks_dir=args.tasks_dir,
        output_path=args.output,
        allow_download=not args.no_download,
    )
    print(f"Prepared {len(rows)} public examples; this does not execute or validate their solutions.")


if __name__ == "__main__":
    main()
