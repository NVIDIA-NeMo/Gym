# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize pinned LAB assets for Gym's Harbor Environment Server."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import string
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from nemo_gym.data.tasks.harbor import load_task, materialize_task
from nemo_gym.data.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.data.tasks.harbor.image_build import build_images
from nemo_gym.data.tasks.harbor.materialize import taskset_mapping
from resources_servers.legal_agent_bench.prepare import (
    CACHE_MARKER,
    LAB_SOURCE_REVISION,
    PACKAGE_DIR,
    _lock_cache,
    task_documents_dir,
)


TASKSET = "legal_agent_bench"
VERIFIER_VARIABLES = (
    "LAB_JUDGES",
    "LAB_JUDGE_BASE_URL",
    "LAB_JUDGE_API_KEY",
    "LAB_JUDGE_REASONING_EFFORT",
    "LAB_JUDGE_TEMPERATURE",
    "LAB_JUDGE_REQUEST_TIMEOUT_SECONDS",
    "LAB_JUDGE_MAX_RETRIES",
    "LAB_JUDGE_STRUCTURED_OUTPUT",
    "LAB_JUDGE_PARSE_REPAIR_ATTEMPTS",
    "LAB_JUDGE_REPAIR_MAX_TOKENS",
    "LAB_JUDGE_MAX_TOKENS",
    "LAB_JUDGE_PARALLELISM",
    "LEGAL_AGENT_BENCH_REWARD_MODE",
)


def _link_tree(source: Path, destination: Path) -> None:
    """Stage regular immutable cache files without duplicating a shared corpus."""
    if source.is_symlink():
        raise ValueError(f"LAB input tree must not be a symbolic link: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        target = destination / path.relative_to(source)
        if path.is_symlink():
            raise ValueError(f"LAB input must not be a symbolic link: {path}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError as error:
                raise OSError(
                    "LAB runtime inputs and asset caches must share a filesystem that supports hardlinks"
                ) from error
        else:
            raise ValueError(f"LAB input is not a regular file or directory: {path}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def prepare_environment(
    *,
    tasks_dir: Path,
    skills_dir: Path,
    rows: Sequence[Mapping[str, object]],
    output: Path,
    image: str | None = None,
    agent_timeout_seconds: float = 10800,
    verifier_timeout_seconds: float = 3600,
    build_tool: str | None = None,
    build_repository: str | None = None,
) -> Path:
    """Write canonical rows and their taskset mapping; credentials never enter either file."""
    tasks_dir, skills_dir, output = tasks_dir.resolve(), skills_dir.resolve(), output.resolve()
    if not rows:
        raise ValueError("LAB preparation requires at least one task")
    names = []
    for row in rows:
        pointer = row.get("instance_id")
        if isinstance(row.get("task_id"), Mapping):
            if row["task_id"].get("taskset") != TASKSET:
                raise ValueError("Expected the legal_agent_bench taskset")
            canonical = f"{TASKSET}::{row['task_id']['task_id']}"
            if pointer is not None and pointer != canonical:
                raise ValueError("Conflicting legacy and canonical LAB task identities")
            pointer = canonical
        if not isinstance(pointer, str) or not pointer.startswith(f"{TASKSET}::"):
            raise ValueError("Each LAB row requires a legal_agent_bench:: task identity")
        name = pointer.removeprefix(f"{TASKSET}::")
        if not name or any(c not in string.ascii_letters + string.digits + "_-" for c in name):
            raise ValueError(f"Invalid LAB task ID: {name!r}")
        if not (tasks_dir / name / "task.json").is_file():
            raise ValueError(f"LAB task is not prepared: {name}")
        names.append(name)
    if image is None:
        # Identical task contexts share one tools image. Only explicit task build files reach the builder.
        build_images(tasks_dir / names[0], builder=ImageBuilder(tool=build_tool, repository=build_repository))
        image = load_task(tasks_dir / names[0], root=tasks_dir).image
        if not image:
            raise ValueError("LAB tools image was not built; provide a prepared image")
    preamble = (PACKAGE_DIR / "harness/system-prompt.md").read_text().split("## Completion", 1)[0]
    manuals = "\n\n".join(
        f"## Skill: {path.parent.name}\n{path.read_text()}" for path in sorted(skills_dir.glob("*/SKILL.md"))
    )
    recipe = hashlib.sha256(
        Path(__file__).read_bytes()
        + (PACKAGE_DIR / "environment_verifier.py").read_bytes()
        + ((tasks_dir / CACHE_MARKER).read_bytes() if (tasks_dir / CACHE_MARKER).exists() else b"")
        + f"{LAB_SOURCE_REVISION}\n{image}\n{agent_timeout_seconds}\n{verifier_timeout_seconds}\n{preamble}\n{manuals}".encode()
    ).hexdigest()[:20]
    root = tasks_dir.parent / "environment" / recipe
    root.mkdir(parents=True, exist_ok=True)
    # The same flock requirement as the source cache: cluster-wide flock, not localflock/noflock.
    with (root / ".prepare.lock").open("a") as lock:
        _lock_cache(lock)
        tasks = {}
        for name in dict.fromkeys(names):
            cached = tasks_dir / name
            task = json.loads((cached / "task.json").read_text())
            if task.get("metadata", {}).get("lab_source_revision") != LAB_SOURCE_REVISION:
                raise ValueError(f"LAB task {name} is from a different source revision")
            destination = root / name
            if not destination.exists():
                with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as temporary:
                    staged = Path(temporary) / name
                    staged.mkdir()
                    _link_tree(task_documents_dir(cached, task, root=tasks_dir), staged / "environment/vdr")
                    _link_tree(skills_dir, staged / "environment/skills")
                    shutil.copytree(cached / "tests", staged / "tests")
                    (staged / "tests/test.sh").unlink(missing_ok=True)
                    shutil.copy2(PACKAGE_DIR / "environment_verifier.py", staged / "tests/verifier.py")
                    (staged / "instruction.md").write_text(
                        f"{preamble}\n\n{manuals}\n\n## Task\n\n{task['title']}\n\n{task['instructions']}\n"
                    )
                    _write_json(
                        staged / "task_data.json",
                        {
                            "instance_id": f"{TASKSET}::{name}",
                            "source_revision": LAB_SOURCE_REVISION,
                        },
                    )
                    env = {
                        "VDR_DIR": "/workspace/vdr",
                        "DOCUMENTS_DIR": "/workspace/vdr",
                        "OUTPUT_DIR": "/workspace/output",
                        "WORKSPACE_DIR": "/workspace",
                        "SKILLS_DIR": "/workspace/skills",
                    }
                    lines = [
                        'version = "1.0"',
                        'artifacts = ["/workspace/output"]',
                        "[agent]",
                        f"timeout_sec = {agent_timeout_seconds}",
                        "[verifier]",
                        'environment_mode = "separate"',
                        f"timeout_sec = {verifier_timeout_seconds}",
                        "[verifier.env]",
                    ]
                    lines.extend(f'{key} = "${{{key}}}"' for key in VERIFIER_VARIABLES)
                    lines.extend(
                        [
                            "[environment]",
                            f"docker_image = {json.dumps(image)}",
                            'workdir = "/workspace"',
                            "cpus = 1",
                            "memory_mb = 4096",
                            "storage_mb = 10240",
                            "[environment.env]",
                        ]
                    )
                    lines.extend(f"{key} = {json.dumps(value)}" for key, value in env.items())
                    (staged / "task.toml").write_text("\n".join(lines) + "\n")
                    staged.rename(destination)
            tasks[name] = load_task(destination, root=root)
        materialized = []
        for row, name in zip(rows, names, strict=True):
            converted = materialize_task(tasks[name], TASKSET).model_dump(mode="json", exclude_unset=True)
            supplied = (
                row.get("responses_create_params")
                or (row.get("task_input") or {}).get("responses_create_params")
                or {}
            )
            params = converted["task_input"]["responses_create_params"]
            instruction = params["input"]
            metadata = {**params.get("metadata", {}), **(supplied.get("metadata") or {})}
            params.update(supplied)
            params["metadata"] = metadata
            if not supplied.get("input"):
                params["input"] = instruction
            elif "task_input" not in row:
                extra = supplied["input"]
                params["input"] = instruction + (
                    extra if isinstance(extra, list) else [{"role": "user", "content": extra}]
                )
            converted["task_source"] = row.get("task_source", TASKSET)
            materialized.append(json.dumps(converted, ensure_ascii=False) + "\n")
        mapping = {TASKSET: taskset_mapping(list(tasks.values()), root)}
        _atomic_text(Path(str(output) + ".tasksets.json"), json.dumps(mapping, indent=2) + "\n")
        _atomic_text(output, "".join(materialized))
    return output
